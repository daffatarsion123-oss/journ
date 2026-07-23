#!/usr/bin/env python
"""Index the CHB-MIT window-feature parquets and build an in-memory cache.

Your previous paper already produced per-recording feature parquets
(``chbXX_YY_features.parquet`` with windowStartSec/windowEndSec/label + 184
channel-band features). This script:

  1. discovers and validates those parquets,
  2. assembles the contiguous in-memory bundle (float32),
  3. optionally caches it (``--set data.cache_dir=...``) for fast reloads,
  4. prints dataset / class-balance / per-subject seizure stats,
  5. confirms LOSO folds can be built (subject parsing sanity check).

It does NOT re-extract features from raw EDF by default. If you ever need to
regenerate features, see ``extract_features_from_edf`` below (left as a clearly
marked TODO that mirrors the previous extractor).

Example:
    python scripts/prepare_chbmit_features.py --data-dir /path/to/outputs \
        --set data.cache_dir=/path/to/outputs/.eegrag_cache
"""
from _bootstrap import base_parser, load


def extract_features_from_edf(*args, **kwargs):
    """TODO(dataset): raw-EDF -> window features extraction.

    The current pipeline consumes already-extracted parquets, so this is only
    needed to regenerate features. Port the previous-paper extractor here:
      * read EDF via mne, IIR bandpass 0.5-40 Hz,
      * 2 s windows / 1 s step,
      * per channel: Mean, Std, Variance, Entropy (64-bin), band powers
        (Delta/Theta/Alpha/Beta via Welch),
      * label windows overlapping a seizure interval (from the summary file).
    Keep the output schema identical so DataConfig works unchanged.
    """
    raise NotImplementedError(
        "Raw-EDF extraction is intentionally not wired up; features already "
        "exist as parquet. Implement here only if regenerating from EDF."
    )


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load(args)

    from eegrag.data import assemble_dataset, make_loso_folds
    import numpy as np

    bundle = assemble_dataset(cfg.data)
    bal = bundle.class_balance()
    print("\n=== Dataset summary ===")
    print(f"  windows      : {bal['n_windows']:,}")
    print(f"  features     : {bundle.layout.n_features}")
    print(f"  {bundle.layout.describe()}")
    print(f"  subjects     : {len(bundle.subject_order)} -> {bundle.subject_order}")
    print(f"  seizure pos  : {bal['n_pos']:,} ({100*bal['pos_fraction']:.4f}%)")

    print("\n=== Per-subject seizure windows ===")
    for s in bundle.subject_order:
        view = bundle.view(s)
        print(f"  {s:8s} n={view.n_windows:>8,}  seizure={view.n_pos:>6,}  "
              f"({100*view.n_pos/max(view.n_windows,1):.3f}%)")

    folds = make_loso_folds(bundle, folds_whitelist=cfg.folds_whitelist)
    print(f"\n=== LOSO ===\n  built {len(folds)} folds; "
          f"first test subject = {folds[0].test_subject}, "
          f"train subjects = {len(folds[0].train_subjects)}")
    print("\nOK. Bundle ready" + (f" (cached to {cfg.data.cache_dir})."
                                   if cfg.data.cache_dir else "."))


if __name__ == "__main__":
    main()
