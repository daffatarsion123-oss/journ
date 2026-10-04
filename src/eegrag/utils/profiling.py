"""Compute-cost instrumentation: wall-clock time, peak RAM/VRAM, params, FLOPs.

Everything here degrades gracefully -- a missing optional dependency (psutil,
torch, cupy) yields ``NaN`` rather than raising, so profiling never breaks a run.

Peak-memory semantics
----------------------
* **RAM** -- a lightweight background thread samples ``psutil`` RSS every
  ``interval`` seconds while a :class:`ResourceTracker` region is open, and the
  max is reported. This gives a true *per-region* peak (process-lifetime metrics
  like ``ru_maxrss`` are monotonic and would over-report later folds).
* **VRAM (torch)** -- ``torch.cuda.reset_peak_memory_stats()`` at region entry +
  ``torch.cuda.max_memory_allocated()`` at exit. Accurate per-region. Works on
  ROCm/HIP too (it masquerades as ``torch.cuda``).
* **VRAM (cuML/cupy)** -- torch cannot see cupy's allocations, so for the
  classical (cuML) path we *also* read the cupy default memory-pool high-water
  mark (``used_bytes``) and report the larger of the two. This is best-effort.

The CSV writer (:class:`ComputeLog`) APPENDS+dedupes, so multiple methods (run
on different machines, e.g. classical on A100 + contrastive on MI300X) can be
merged into one ``outputs/compute/*.csv`` simply by collecting the files.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .io import ensure_dir
from .logging import get_logger

log = get_logger(__name__)

NAN = float("nan")


# --------------------------------------------------------------------------- #
# Memory probes
# --------------------------------------------------------------------------- #
def _current_rss_bytes() -> Optional[int]:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        return None


def reset_peak_vram() -> None:
    """Reset torch's per-device peak VRAM counter (no-op without torch/accel)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_vram_gb() -> float:
    """Best peak VRAM across torch (DL) and cupy (cuML) allocators, in GB."""
    vals: List[int] = []
    try:
        import torch

        if torch.cuda.is_available():
            vals.append(int(torch.cuda.max_memory_allocated()))
    except Exception:
        pass
    try:
        import cupy  # cuML allocations go through cupy's pool

        vals.append(int(cupy.get_default_memory_pool().used_bytes()))
    except Exception:
        pass
    return (max(vals) / 1e9) if vals else NAN


class _RamSampler(threading.Thread):
    """Daemon thread tracking peak process RSS while a region is open."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        start = _current_rss_bytes()
        self._enabled = start is not None
        if start is not None:
            self.peak_bytes = start

    def run(self) -> None:
        if not self._enabled:
            return
        while not self._stop_event.is_set():
            rss = _current_rss_bytes()
            if rss is not None and rss > self.peak_bytes:
                self.peak_bytes = rss
            self._stop_event.wait(self.interval)

    def stop(self) -> float:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=1.0)
        return (self.peak_bytes / 1e9) if self._enabled else NAN


# --------------------------------------------------------------------------- #
# Resource tracker (context manager)
# --------------------------------------------------------------------------- #
@dataclass
class ResourceTracker:
    """Context manager capturing elapsed time + peak RAM (+ optional VRAM).

    Example::

        with ResourceTracker(track_vram=True) as rt:
            train(...)
        rt.elapsed_sec, rt.peak_ram_gb, rt.peak_vram_gb
    """

    track_vram: bool = False
    ram_interval: float = 0.05
    elapsed_sec: float = field(default=NAN, init=False)
    peak_ram_gb: float = field(default=NAN, init=False)
    peak_vram_gb: float = field(default=NAN, init=False)
    _t0: float = field(default=0.0, init=False)
    _sampler: Optional[_RamSampler] = field(default=None, init=False)

    def __enter__(self) -> "ResourceTracker":
        if self.track_vram:
            reset_peak_vram()
        self._sampler = _RamSampler(self.ram_interval)
        self._sampler.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.elapsed_sec = time.perf_counter() - self._t0
        if self._sampler is not None:
            self.peak_ram_gb = self._sampler.stop()
        if self.track_vram:
            self.peak_vram_gb = peak_vram_gb()
        return False  # never suppress exceptions


# --------------------------------------------------------------------------- #
# Model complexity: parameter count + FLOPs
# --------------------------------------------------------------------------- #
def count_parameters(model: Any) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def estimate_flops(model: Any, example_input: Any) -> float:
    """Estimate forward-pass FLOPs for one example via layer hooks.

    Counts Linear / Conv1d / Conv2d multiply-accumulates (x2 for the add). This
    is dependency-free and covers the MLP / CNN1D / TCN / transformer encoders
    (attention's matmuls are Linear-dominated). ``example_input`` must already be
    batched (e.g. shape ``[1, ...]``) and on the model's device. Returns FLOPs
    for that batch -- pass batch size 1 for per-sample FLOPs.
    """
    import torch
    import torch.nn as nn

    total = [0]
    hooks = []

    def _linear_hook(m, inp, _out):
        x = inp[0]
        n = int(x.numel() // x.shape[-1])  # batch * any leading dims
        total[0] += 2 * n * m.in_features * m.out_features

    def _conv_hook(m, _inp, out):
        kernel = 1
        ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        for k in ks:
            kernel *= k
        batch, cout = out.shape[0], out.shape[1]
        out_spatial = int(out.numel() // (batch * cout))
        macs = batch * cout * out_spatial * (m.in_channels // m.groups) * kernel
        total[0] += 2 * macs

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_linear_hook))
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            hooks.append(mod.register_forward_hook(_conv_hook))

    try:
        model.eval()
        with torch.no_grad():
            if hasattr(model, "embed"):
                model.embed(example_input)
            else:
                model(example_input)
    finally:
        for h in hooks:
            h.remove()
    return float(total[0])


# --------------------------------------------------------------------------- #
# CSV logging (append + dedupe so cross-machine merges just work)
# --------------------------------------------------------------------------- #
# Per-file dedupe keys: a record is uniquely identified by these columns.
_KEYS: Dict[str, List[str]] = {
    "runtime": ["stage", "method", "test_subject", "seed"],
    "memory": ["stage", "method", "test_subject", "seed"],
    "retrieval_cost": ["method", "test_subject", "seed"],
    "quantum_cost": ["kernel_backend", "method", "test_subject", "seed"],
}


class ComputeLog:
    """Accumulate compute records and flush to ``outputs/compute/*.csv``.

    Buffers in memory; :meth:`flush` merges with any existing CSV (append +
    drop_duplicates on the file's key columns, keeping the latest) so re-runs
    overwrite their own rows and new methods are added without clobbering others.
    """

    FILES = ("runtime", "memory", "retrieval_cost", "quantum_cost")

    def __init__(self, compute_dir: str):
        self.dir = ensure_dir(compute_dir)
        self._buf: Dict[str, List[Dict[str, Any]]] = {f: [] for f in self.FILES}

    def add_runtime(self, **row: Any) -> None:
        self._buf["runtime"].append(row)

    def add_memory(self, **row: Any) -> None:
        self._buf["memory"].append(row)

    def add_retrieval_cost(self, **row: Any) -> None:
        self._buf["retrieval_cost"].append(row)

    def add_quantum_cost(self, **row: Any) -> None:
        self._buf["quantum_cost"].append(row)

    def add_resource(
        self, *, stage: str, method: str, fold: Any, test_subject: str,
        seed: int, tracker: ResourceTracker, device: str = "",
        n_params: Any = "", n_trainable: Any = "", flops: Any = "", git: str = "",
    ) -> None:
        """Convenience: log one ResourceTracker into BOTH runtime + memory CSVs."""
        self.add_runtime(
            stage=stage, method=method, fold=fold, test_subject=test_subject,
            seed=seed, time_sec=round(tracker.elapsed_sec, 4),
            n_params=n_params, n_trainable=n_trainable, flops=flops,
            device=device, git=git,
        )
        self.add_memory(
            stage=stage, method=method, fold=fold, test_subject=test_subject,
            seed=seed, peak_ram_gb=round(tracker.peak_ram_gb, 4),
            peak_vram_gb=round(tracker.peak_vram_gb, 4), device=device,
        )

    def flush(self) -> None:
        import pandas as pd

        for name in self.FILES:
            rows = self._buf[name]
            if not rows:
                continue
            path = os.path.join(self.dir, f"{name}.csv")
            new = pd.DataFrame(rows)
            if os.path.exists(path):
                try:
                    old = pd.read_csv(path)
                    new = pd.concat([old, new], ignore_index=True)
                except Exception as exc:
                    log.warning("Could not read existing %s (%s); overwriting.",
                                path, exc)
            keys = [c for c in _KEYS[name] if c in new.columns]
            if keys:
                new = new.drop_duplicates(subset=keys, keep="last")
            new.to_csv(path, index=False)
            log.info("Wrote %d rows -> %s", len(rows), path)
        self._buf = {f: [] for f in self.FILES}

    def totals_by(self, file: str, group: str, value: str) -> Dict[str, float]:
        """Sum a buffered value column grouped by another column (for printing)."""
        out: Dict[str, float] = {}
        for r in self._buf.get(file, []):
            v = r.get(value)
            if isinstance(v, (int, float)):
                out[str(r.get(group))] = out.get(str(r.get(group)), 0.0) + float(v)
        return out
