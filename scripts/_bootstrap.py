"""Shared CLI bootstrap for the scripts.

Adds ``src/`` to ``sys.path`` so the scripts run without ``pip install -e .``
(though installing the package is recommended). Provides a tiny argument parser
that every script reuses: ``--config`` + repeatable ``--set key=value`` overrides.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple


def _ensure_src_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(os.path.dirname(here), "src")
    if src not in sys.path:
        sys.path.insert(0, src)


_ensure_src_on_path()


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=str, default=None,
                   help="Path to a YAML config (see configs/).")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Dotted config override, e.g. --set training.epochs=2. "
                        "Repeatable.")
    p.add_argument("--data-dir", dest="features_dir", type=str, default=None,
                   help="Shortcut for --set data.features_dir=<path>.")
    p.add_argument("--name", type=str, default=None,
                   help="Shortcut for --set name=<experiment name>.")
    p.add_argument("--embeddings-dir", type=str, default=None,
                   help="Directory of saved per-fold embeddings (for re-analysis).")
    p.add_argument("--train-only", action="store_true", default=False,
                   help="Train encoder + extract embeddings only; skip retrieval eval. "
                        "Use to separate encoder training from retrieval evaluation. "
                        "Run run_retrieval_eval.py over the saved embeddings.")
    return p


def collect_overrides(args) -> List[str]:
    overrides: List[str] = list(args.overrides)
    if args.features_dir:
        overrides.append(f"data.features_dir={args.features_dir}")
    if args.name:
        overrides.append(f"name={args.name}")
    return overrides


def load(args):
    from eegrag.config import load_config
    return load_config(args.config, overrides=collect_overrides(args))
