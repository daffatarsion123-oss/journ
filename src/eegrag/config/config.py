"""Typed configuration objects for every experiment stage.

All configs are plain dataclasses so they are:
  * easy to serialize to / from YAML,
  * type-checked at construction (via ``__post_init__`` validation),
  * reproducible (every stochastic stage carries an explicit seed knob).

Nothing here references local absolute paths. Dataset paths are supplied through
YAML (see ``configs/*.yaml``) or CLI overrides. See ``configs`` for examples.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Where the CHB-MIT window-level feature parquets live and how to read them.

    The features produced by the previous paper are stored one parquet per
    recording, e.g. ``chb01_03_features.parquet`` with columns
    ``windowStartSec, windowEndSec, label, <channel><FeatureType> ...``.
    """

    # TODO: set this (or pass --data.features_dir) to the directory that holds
    #       the per-recording *_features.parquet files. Do NOT hardcode here.
    features_dir: str = ""
    storage_backend: str = "numpy"

    glob_pattern: str = "*_features.parquet"
    # Columns that are NOT model features.
    meta_columns: Tuple[str, ...] = ("windowStartSec", "windowEndSec", "label")
    label_column: str = "label"
    window_start_column: str = "windowStartSec"
    window_end_column: str = "windowEndSec"

    # Regex capturing the subject id from a filename. Default collapses
    # chb17a/chb17b/chb17c -> chb17 (same physical subject, different sessions).
    subject_regex: str = r"^(chb\d+)"
    patient_aliases: Dict[str, str] = field(default_factory=lambda: {"chb21": "chb01"})

    # Window geometry (used for false-positives-per-hour accounting).
    window_size_sec: float = 2.0
    step_size_sec: float = 1.0
    sampling_rate_hz: int = 256

    # Optionally restrict to a subset of subjects (useful for smoke tests).
    subjects_whitelist: Optional[List[str]] = None
    # Cache the assembled per-subject arrays as .npz to skip re-reading parquet.
    cache_dir: Optional[str] = None
    feature_schema_path: Optional[str] = None
    missing_feature_policy: str = "error"

    def __post_init__(self) -> None:
        assert self.storage_backend in {"numpy", "cupy"}
        assert self.missing_feature_policy in {"error", "impute"}
        assert self.window_size_sec > 0, "window_size_sec must be > 0"
        assert self.step_size_sec > 0, "step_size_sec must be > 0"
        assert self.sampling_rate_hz > 0, "sampling_rate_hz must be > 0"


# --------------------------------------------------------------------------- #
# Augmentation (physiology-preserving only)
# --------------------------------------------------------------------------- #
@dataclass
class AugmentationConfig:
    """Topology-preserving EEG-feature augmentations.

    IMPORTANT: NO image-style spatial flipping or arbitrary cropping. NO
    arbitrary cutoff-frequency shifting. Frequency masking here means masking a
    whole *named band group* of features (Delta/Theta/Alpha/Beta), which is
    physiologically interpretable rather than a learned cutoff shift.
    """

    enabled: bool = True
    # Gaussian noise injection, scaled per-feature by the (train) feature std.
    gaussian_noise_std: float = 0.05
    gaussian_noise_p: float = 0.5

    # Mild amplitude scaling: multiply features by U(1-d, 1+d).
    amplitude_scale_delta: float = 0.1
    amplitude_scale_p: float = 0.5

    # Channel (electrode) dropout: zero ALL features of randomly chosen channels.
    channel_dropout_max: int = 4          # up to this many electrodes per view
    channel_dropout_p: float = 0.5

    # Frequency masking: zero a randomly chosen band-power group across channels.
    freq_masking_bands_max: int = 1       # number of band groups to mask
    freq_masking_p: float = 0.3

    # Temporal masking: only meaningful when the dataset yields *window sequences*
    # (sequence_len > 1). Masks a contiguous span of windows. No-op otherwise.
    temporal_mask_max_frac: float = 0.2
    temporal_mask_p: float = 0.3

    # Number of augmented views per anchor for contrastive training.
    n_views: int = 2

    def __post_init__(self) -> None:
        assert self.n_views >= 2, "Contrastive training needs >= 2 views"
        for p in (
            self.gaussian_noise_p,
            self.amplitude_scale_p,
            self.channel_dropout_p,
            self.freq_masking_p,
            self.temporal_mask_p,
        ):
            assert 0.0 <= p <= 1.0, "augmentation probabilities must be in [0,1]"


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
@dataclass
class EncoderConfig:
    """Representation encoder configuration.

    arch:
        "mlp"             flat feature vector -> MLP            (simplest)
        "cnn1d"           [feat_per_chan, n_channels] -> 1D-CNN over electrodes
        "tcn"             temporal conv net (sequence input)
        "cnn1d_transformer" 1D-CNN stem + Transformer encoder   (optional)
    """

    arch: str = "cnn1d"
    embedding_dim: int = 128         # output representation dimension
    hidden_dims: Tuple[int, ...] = (256, 256)
    dropout: float = 0.1

    # CNN / TCN specifics
    conv_channels: Tuple[int, ...] = (64, 128, 128)
    kernel_size: int = 3
    # Transformer specifics
    n_heads: int = 4
    n_transformer_layers: int = 2
    # Projection head for contrastive learning (SimCLR/SupCon style).
    projection_dim: int = 64
    projection_hidden: int = 128

    # Sequence length (number of consecutive windows stacked per sample).
    # 1 == single-window feature vector (default).
    sequence_len: int = 1

    def __post_init__(self) -> None:
        valid = {"mlp", "cnn1d", "tcn", "cnn1d_transformer"}
        assert self.arch in valid, f"arch must be one of {valid}, got {self.arch}"
        assert self.embedding_dim > 0
        assert self.sequence_len >= 1


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    """Contrastive objective selection.

    name:
        "supcon"             standard Supervised Contrastive (Khosla et al.)
        "balanced_supcon"    Balanced Contrastive Learning baseline (class-balanced)
        "imbalance_supcon"   Imbalance-Aware SupCon (minority/focal-style weighting)
        "simclr"             unsupervised NT-Xent (sanity baseline)
    """

    name: str = "imbalance_supcon"
    temperature: float = 0.1
    base_temperature: float = 0.1

    # imbalance_supcon knobs
    minority_class: int = 1               # seizure
    minority_weight: float = 0.0          # 0 -> auto from inverse frequency
    focal_gamma: float = 2.0              # focal-style down-weighting of easy pos
    use_focal_weighting: bool = True

    def __post_init__(self) -> None:
        valid = {"supcon", "balanced_supcon", "imbalance_supcon", "simclr"}
        assert self.name in valid, f"loss name must be one of {valid}"
        assert self.temperature > 0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    gpu_resident: bool = False
    save_embeddings: bool = True
    tensor_batches: bool = False
    steps_per_epoch: Optional[int] = None
    positive_fraction: float = 0.5
    deterministic: bool = True
    allow_tf32: bool = False
    checkpoint_every: int = 1
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"            # "cosine" | "step" | "none"
    warmup_epochs: int = 5
    grad_clip: float = 5.0
    num_workers: int = 8
    pin_memory: bool = True
    amp: bool = True                     # mixed precision (ROCm/CUDA)
    # "float16" (FP16 + GradScaler) or "bfloat16" (native on A100/MI300X, no scaler needed).
    # Use bfloat16 on A100 SXM or MI300X ROCm >= 6.0 for ~10-15% throughput gain.
    amp_dtype: str = "float16"
    device: str = "auto"                 # "auto" | "cuda" | "cpu"
    # Class-balanced sampling so seizure anchors actually appear in batches.
    balanced_sampler: bool = True
    # Minimum #positive (seizure) anchors required per batch (best-effort).
    min_pos_per_batch: int = 16
    log_every: int = 50
    early_stop_patience: int = 0         # 0 disables early stopping

    def __post_init__(self) -> None:
        assert self.batch_size > 1 and self.epochs > 0
        assert self.steps_per_epoch is None or self.steps_per_epoch > 0
        assert 0 < self.positive_fraction < 1
        assert self.checkpoint_every > 0
        valid_dtypes = {"float16", "bfloat16"}
        assert self.amp_dtype in valid_dtypes, (
            f"amp_dtype must be one of {valid_dtypes}, got {self.amp_dtype!r}"
        )


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    linear_probe_max_negatives: Optional[int] = None
    """Memory-bank / k-NN retrieval head configuration."""

    head: str = "hybrid"                 # "linear" | "knn" | "hybrid"
    # k values to sweep for the neighbor analysis.
    topk_values: Tuple[int, ...] = (1, 5, 10, 20, 50)
    # k used by the *decision* head (must be in topk_values).
    decision_k: int = 10
    similarity: str = "cosine"           # "cosine" | "euclidean" | "rbf"
    rbf_gamma: float = 1.0               # for similarity == "rbf"
    # Weight between linear-head prob and retrieval score for the hybrid head.
    hybrid_alpha: float = 0.5
    # FAISS if available else sklearn NearestNeighbors.
    backend: str = "auto"                # "auto" | "faiss" | "sklearn"
    normalize_embeddings: bool = True    # L2-normalize before indexing
    # Down-sample the (majority) memory bank to control cost; keep ALL seizures.
    max_bank_per_class: Optional[int] = None
    seed: int = 0
    query_chunk_size: int = 512
    bank_chunk_size: int = 32768

    def __post_init__(self) -> None:
        assert self.head in {"linear", "knn", "hybrid"}
        assert self.similarity in {"cosine", "euclidean", "rbf"}
        assert self.decision_k in self.topk_values, (
            "decision_k must be one of topk_values"
        )


# --------------------------------------------------------------------------- #
# Quantum (exploratory, optional)
# --------------------------------------------------------------------------- #
@dataclass
class QuantumConfig:
    """Exploratory quantum-simulated kernel retrieval.

    Runs simulator-only and on a *compressed* memory bank. Gracefully disables
    itself if no quantum backend (pennylane / qiskit) is importable.
    """

    enabled: bool = False
    backend: str = "auto"                # "auto" | "pennylane" | "qiskit" | "none"
    n_qubits: int = 6                    # 4..8 recommended
    reps: int = 2                        # feature-map repetitions
    feature_map: str = "zz"              # "zz" | "z" | "pauli"
    # Dimensionality fed to the quantum feature map (PCA/projection target).
    projection_dim: int = 6
    # Cap the per-class memory bank for the (expensive) quantum kernel.
    max_bank_per_class: int = 256
    max_query: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        assert 2 <= self.n_qubits <= 16
        assert self.projection_dim <= self.n_qubits, (
            "projection_dim must be <= n_qubits (one feature per qubit angle)"
        )


# --------------------------------------------------------------------------- #
# Classical baselines
# --------------------------------------------------------------------------- #
@dataclass
class ClassicalConfig:
    """Classical baselines with GPU-capable backends and clean fallbacks."""

    models: Tuple[str, ...] = ("logreg", "svm_rbf", "random_forest", "xgboost")
    class_weight: str = "balanced"       # passed where supported
    # Backend preference is resolved at runtime (see utils.backend); these are
    # the *requested* preferences, not assertions that they exist.
    prefer_gpu: bool = True
    # Per-model hyperparameter grids (used with training-subjects-only CV).
    svm_C_grid: Tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    svm_gamma_grid: Tuple[Any, ...] = ("scale", 1e-3, 1e-2, 1e-1)
    svm_class_weight_grid: Tuple[Any, ...] = ("balanced", None)
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    # Inner CV folds (over training subjects only) for HP search.
    inner_cv_folds: int = 3
    # Scoring used for HP selection (imbalance-aware).
    hp_scoring: str = "average_precision"


# --------------------------------------------------------------------------- #
# Stats / evaluation
# --------------------------------------------------------------------------- #
@dataclass
class StatsConfig:
    bootstrap_n: int = 2000
    bootstrap_ci: float = 0.95
    nemenyi_alpha: float = 0.05
    wilcoxon_alternative: str = "two-sided"
    seed: int = 0


@dataclass
class PreprocessingConfig:
    """Train-only preprocessing pipeline (fit on training subjects ONLY)."""

    scaler: str = "standard"             # "standard" | "robust" | "none"
    backend: str = "auto"
    use_pca: bool = False
    pca_components: float = 0.99          # int -> #components, float -> variance kept
    # Optional univariate feature selection (k best by ANOVA F).
    feature_selection_k: Optional[int] = None
    # Replace non-finite feature values produced by extraction edge-cases.
    impute_nonfinite: bool = True

    def __post_init__(self):
        assert self.backend in {"auto", "cupy"}


# --------------------------------------------------------------------------- #
# Top-level experiment config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    name: str = "experiment"
    output_dir: str = "outputs"
    seeds: Tuple[int, ...] = (0, 1, 2)
    # If set, only run these folds (subject ids) -- handy for smoke tests.
    folds_whitelist: Optional[List[str]] = None

    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
