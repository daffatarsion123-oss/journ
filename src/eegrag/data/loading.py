"""Load per-recording CHB-MIT feature parquets into per-subject arrays.

Design goals
------------
* **Subject is the unit of evaluation.** Every window carries its subject id and
  recording id so the LOSO split and the leakage assertions can operate on them.
* **Schema-driven.** Feature columns = all columns minus the configured metadata
  columns. The channel/feature topology is recovered by :class:`FeatureLayout`.
* **Memory aware but RAM-friendly.** With ~240 GB RAM available we can hold the
  whole feature matrix (float32) in memory; the loader assembles a single
  contiguous array plus integer subject/recording codes for fast slicing.

TODO(dataset): if you regenerate features with a different naming convention,
update ``DataConfig.subject_regex`` / ``meta_columns`` rather than editing here.
"""
from __future__ import annotations

import glob
import os
import re
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..utils.logging import get_logger
from .feature_layout import FeatureLayout

log = get_logger(__name__)


def subject_id_from_filename(filename: str, subject_regex: str) -> str:
    """Extract subject id from a recording filename, e.g. ``chb17a_03`` -> ``chb17``.

    The default regex collapses the trailing session letter so chb17a/b/c map to
    the same physical subject -- critical for a correct LOSO protocol.
    """
    base = os.path.basename(filename)
    m = re.match(subject_regex, base)
    if not m:
        raise ValueError(
            f"Could not parse subject id from {base!r} with regex {subject_regex!r}"
        )
    return m.group(1)


def recording_id_from_filename(filename: str) -> str:
    """Strip the ``_features.parquet`` suffix to get the recording id."""
    base = os.path.basename(filename)
    return re.sub(r"_features\.parquet$", "", base)


def discover_recordings(cfg: DataConfig) -> List[str]:
    """Return sorted parquet paths, excluding any aggregate/combined files."""
    if not cfg.features_dir:
        raise ValueError(
            "DataConfig.features_dir is empty. Set it in YAML or via "
            "--data.features_dir=<path to *_features.parquet directory>."
        )
    pattern = os.path.join(cfg.features_dir, cfg.glob_pattern)
    paths = sorted(glob.glob(pattern))
    # Exclude an aggregate file like 'eegFeatures.parquet' that does not match
    # the per-recording subject regex.
    kept: List[str] = []
    for p in paths:
        base = os.path.basename(p)
        if re.match(cfg.subject_regex, base):
            kept.append(p)
        else:
            log.debug("Skipping non-recording parquet: %s", base)
    if not kept:
        raise FileNotFoundError(
            f"No recording parquets matched {pattern!r} with subject regex "
            f"{cfg.subject_regex!r}."
        )
    return kept


@dataclass
class SubjectData:
    """Per-subject view onto the assembled arrays (rows are time-ordered)."""

    subject: str
    features: np.ndarray          # [n_windows, n_features] float32
    labels: np.ndarray            # [n_windows] int8 {0,1}
    recording_ids: np.ndarray     # [n_windows] object/str
    window_start: np.ndarray      # [n_windows] float (seconds)
    window_end: np.ndarray        # [n_windows] float (seconds)

    @property
    def n_windows(self) -> int:
        return self.features.shape[0]

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())


@dataclass
class EEGFeatureBundle:
    """All windows from all subjects, plus topology and provenance metadata."""

    features: np.ndarray          # [N, F] float32
    labels: np.ndarray            # [N] int8
    subjects: np.ndarray          # [N] object (subject id per window)
    recordings: np.ndarray        # [N] object (recording id per window)
    window_start: np.ndarray      # [N] float32
    window_end: np.ndarray        # [N] float32
    layout: FeatureLayout
    subject_order: List[str]      # unique subjects, discovery order
    data_cfg: DataConfig
    source_signature: Optional[str] = None

    @property
    def n_windows(self) -> int:
        return self.features.shape[0]

    def subject_mask(self, subjects: Sequence[str]) -> np.ndarray:
        wanted = set(subjects)
        return np.fromiter(
            (s in wanted for s in self.subjects), count=self.n_windows, dtype=bool
        )

    def view(self, subject: str) -> SubjectData:
        mask = self.subjects == subject
        return SubjectData(
            subject=subject,
            features=self.features[mask],
            labels=self.labels[mask],
            recording_ids=self.recordings[mask],
            window_start=self.window_start[mask],
            window_end=self.window_end[mask],
        )

    def class_balance(self) -> Dict[str, float]:
        n = self.n_windows
        pos = int(self.labels.sum())
        return {
            "n_windows": n,
            "n_pos": pos,
            "n_neg": n - pos,
            "pos_fraction": pos / max(n, 1),
        }


def load_subject_frames(
    paths: Sequence[str], cfg: DataConfig
) -> Dict[str, List[pd.DataFrame]]:
    """Read parquets, group dataframes by subject id (lazy column subset)."""
    by_subject: Dict[str, List[pd.DataFrame]] = {}
    for path in paths:
        subject = subject_id_from_filename(path, cfg.subject_regex)
        subject = cfg.patient_aliases.get(subject, subject)
        if cfg.subjects_whitelist and subject not in cfg.subjects_whitelist:
            continue
        df = pd.read_parquet(path)
        df["__recording__"] = recording_id_from_filename(path)
        by_subject.setdefault(subject, []).append(df)
    if not by_subject:
        raise RuntimeError(
            "No subjects loaded. Check features_dir / subjects_whitelist."
        )
    return by_subject


def _feature_columns(df: pd.DataFrame, cfg: DataConfig) -> List[str]:
    meta = set(cfg.meta_columns) | {"__recording__"}
    return [c for c in df.columns if c not in meta]


def assemble_dataset(cfg: DataConfig) -> EEGFeatureBundle:
    """Assemble a single in-memory :class:`EEGFeatureBundle` from all parquets.

    Uses a cache file (``cfg.cache_dir/bundle.npz``) when present to skip the
    parquet read on repeated runs. Float32 is used to halve memory vs float64.
    """
    canonical_columns = None
    if cfg.feature_schema_path:
        with open(cfg.feature_schema_path, encoding="utf-8") as fh:
            canonical_columns = json.load(fh)
        if not canonical_columns or len(set(canonical_columns)) != len(canonical_columns):
            raise ValueError("Feature schema must contain a nonempty unique column list")
    if cfg.storage_backend == "cupy":
        return _assemble_device_dataset(cfg, canonical_columns)
    if cfg.cache_dir:
        cache_key = hashlib.sha256(json.dumps({
            "aliases": cfg.patient_aliases, "features_dir": os.path.abspath(cfg.features_dir),
            "whitelist": cfg.subjects_whitelist, "regex": cfg.subject_regex,
            "schema": cfg.meta_columns,
            "canonical_columns": canonical_columns,
            "missing_feature_policy": cfg.missing_feature_policy,
            "files": [(p, os.stat(p).st_size, os.stat(p).st_mtime_ns)
                      for p in discover_recordings(cfg)],
        }, sort_keys=True).encode()).hexdigest()[:16]
        cache_path = os.path.join(cfg.cache_dir, f"bundle_{cache_key}.npz")
        cols_path = os.path.join(cfg.cache_dir, f"bundle_{cache_key}_columns.txt")
        if os.path.exists(cache_path) and os.path.exists(cols_path):
            log.info("Loading assembled bundle from cache: %s", cache_path)
            with np.load(cache_path, allow_pickle=True) as z:
                with open(cols_path, "r", encoding="utf-8") as fh:
                    feat_cols = [ln.rstrip("\n") for ln in fh]
                layout = FeatureLayout.from_columns(feat_cols)
                return EEGFeatureBundle(
                    features=z["features"],
                    labels=z["labels"],
                    subjects=z["subjects"],
                    recordings=z["recordings"],
                    window_start=z["window_start"],
                    window_end=z["window_end"],
                    layout=layout,
                    subject_order=list(z["subject_order"]),
                    data_cfg=cfg,
                )

    paths = discover_recordings(cfg)
    log.info("Discovered %d recording parquets.", len(paths))
    by_subject = load_subject_frames(paths, cfg)

    # Establish the canonical feature-column order from the first frame.
    first_df = next(iter(by_subject.values()))[0]
    feat_cols = canonical_columns or _feature_columns(first_df, cfg)
    layout = FeatureLayout.from_columns(feat_cols)
    log.info("%s", layout.describe())

    feats_parts: List[np.ndarray] = []
    labels_parts: List[np.ndarray] = []
    subj_parts: List[np.ndarray] = []
    rec_parts: List[np.ndarray] = []
    ws_parts: List[np.ndarray] = []
    we_parts: List[np.ndarray] = []
    subject_order: List[str] = []

    for subject in sorted(by_subject):
        subject_order.append(subject)
        for df in by_subject[subject]:
            missing = [c for c in feat_cols if c not in df.columns]
            if len(missing) == len(feat_cols):
                raise ValueError("Recording has no observed features in the configured schema")
            if missing and cfg.missing_feature_policy == "error":
                raise ValueError(
                    f"Recording for {subject} missing feature columns: {missing[:5]}"
                    f"{'...' if len(missing) > 5 else ''}"
                )
            if missing:
                log.warning("Recording %s lacks %d features; train-median imputation required",
                            df["__recording__"].iloc[0], len(missing))
            x = df.reindex(columns=feat_cols).to_numpy(dtype=np.float32, copy=False)
            y = df[cfg.label_column].to_numpy(dtype=np.int8, copy=False)
            n = x.shape[0]
            feats_parts.append(x)
            labels_parts.append(y)
            subj_parts.append(np.full(n, subject, dtype=object))
            rec_parts.append(df["__recording__"].to_numpy(dtype=object))
            ws_parts.append(
                df[cfg.window_start_column].to_numpy(dtype=np.float32)
            )
            we_parts.append(
                df[cfg.window_end_column].to_numpy(dtype=np.float32)
            )

    bundle = EEGFeatureBundle(
        features=np.concatenate(feats_parts, axis=0),
        labels=np.concatenate(labels_parts, axis=0),
        subjects=np.concatenate(subj_parts, axis=0),
        recordings=np.concatenate(rec_parts, axis=0),
        window_start=np.concatenate(ws_parts, axis=0),
        window_end=np.concatenate(we_parts, axis=0),
        layout=layout,
        subject_order=subject_order,
        data_cfg=cfg,
    )
    bal = bundle.class_balance()
    log.info(
        "Assembled bundle: N=%d, F=%d, subjects=%d, pos=%d (%.4f%%)",
        bundle.n_windows, layout.n_features, len(subject_order),
        bal["n_pos"], 100.0 * bal["pos_fraction"],
    )

    if cfg.cache_dir:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=bundle.features,
            labels=bundle.labels,
            subjects=bundle.subjects,
            recordings=bundle.recordings,
            window_start=bundle.window_start,
            window_end=bundle.window_end,
            subject_order=np.asarray(subject_order, dtype=object),
        )
        with open(
            cols_path, "w", encoding="utf-8"
        ) as fh:
            fh.write("\n".join(feat_cols))
        log.info("Cached assembled bundle to %s", cfg.cache_dir)

    return bundle


def _assemble_device_dataset(cfg, canonical_columns):
    """Decode one recording at a time; never assemble the feature matrix in RAM."""
    import cupy as cp
    import pyarrow.parquet as pq

    selected = []
    for path in discover_recordings(cfg):
        subject = subject_id_from_filename(path, cfg.subject_regex)
        subject = cfg.patient_aliases.get(subject, subject)
        if not cfg.subjects_whitelist or subject in cfg.subjects_whitelist:
            selected.append((subject, path, pq.ParquetFile(path).metadata.num_rows))
    selected.sort(key=lambda item: (item[0], item[1]))
    if not selected:
        raise RuntimeError("No subjects loaded. Check subjects_whitelist")
    columns = canonical_columns or _feature_columns(pd.read_parquet(selected[0][1]), cfg)
    layout = FeatureLayout.from_columns(columns)
    count = sum(item[2] for item in selected)
    needed = count * len(columns) * 4
    free, _ = cp.cuda.runtime.memGetInfo()
    if needed * 3 + 2**30 > free:
        raise MemoryError("GPU feature storage plus fold/workspace reserve exceeds free VRAM")
    features = cp.empty((count, len(columns)), dtype=cp.float32)
    labels = np.empty(count, dtype=np.int8)
    subjects = np.empty(count, dtype=object)
    recordings = np.empty(count, dtype=object)
    starts, ends = np.empty(count, np.float32), np.empty(count, np.float32)
    digest = hashlib.sha256()
    digest.update(json.dumps({"columns": columns, "config": cfg.__dict__}, sort_keys=True).encode())
    offset = 0
    for subject, path, rows in selected:
        with open(path, "rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        df = pd.read_parquet(path)
        missing = [name for name in columns if name not in df.columns]
        if len(missing) == len(columns):
            raise ValueError("Recording has no observed features in the configured schema")
        if missing and cfg.missing_feature_policy == "error":
            raise ValueError(f"Recording for {subject} missing feature columns: {missing[:5]}")
        if len(df) != rows:
            raise ValueError("Parquet changed while loading")
        target = slice(offset, offset + rows)
        host = df.reindex(columns=columns).to_numpy(dtype=np.float32, copy=False)
        features[target] = cp.asarray(host, blocking=True)
        labels[target] = df[cfg.label_column].to_numpy(dtype=np.int8)
        subjects[target] = subject
        recordings[target] = recording_id_from_filename(path)
        starts[target] = df[cfg.window_start_column].to_numpy(dtype=np.float32)
        ends[target] = df[cfg.window_end_column].to_numpy(dtype=np.float32)
        offset += rows
        del df, host
    cp.cuda.get_current_stream().synchronize()
    log.info("GPU bundle: %d windows x %d features = %.2f GiB, streamed from parquet; host NPZ cache bypassed",
             count, len(columns), needed / 2**30)
    return EEGFeatureBundle(features, labels, subjects, recordings, starts, ends,
                            layout, sorted(set(subjects)), cfg, digest.hexdigest())
