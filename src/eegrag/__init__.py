"""eegrag - Retrieval-Augmented Cross-Subject EEG Learning for Seizure Detection.

This package implements the journal-extension pipeline:

    contrastive representation learning  ->  memory-bank retrieval  ->  decision

under a strict Leave-One-Subject-Out (LOSO) protocol on CHB-MIT EEG features.

Sub-packages (this maps to the requested ``src/<name>`` layout, but is
namespaced under ``eegrag`` so that generic names like ``config``, ``data`` and
``utils`` do not shadow third-party top-level modules on ``sys.path``):

    config        dataclasses + YAML loader
    utils         seeding, backend detection, IO
    data          parquet loading, subject parsing, FeatureLayout, LOSO split
    preprocessing leakage-safe scaler / PCA / feature-selection pipeline
    datasets      torch Dataset + physiology-preserving augmentations
    models        encoders, heads, classical baselines (GPU-abstracted)
    losses        SupCon / BalancedContrastive / ImbalanceAwareSupCon
    retrieval     memory bank (leakage-guarded), index, similarities, heads
    quantum       optional quantum-simulated kernel similarity
    evaluation    AUC-PR, sensitivity, per-class F1, FP/hour, thresholds
    stats         Friedman+Nemenyi, Wilcoxon, effect size r, bootstrap CI
    experiments   LOSO runners (classical / contrastive / retrieval / quantum)
"""

__version__ = "0.1.0"
