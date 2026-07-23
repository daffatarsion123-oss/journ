# Retrieval-Augmented Cross-Subject EEG Learning for Seizure Detection

Research codebase for the journal extension of the CHB-MIT seizure-detection
conference paper. It moves beyond classical supervised ML into **contrastive
representation learning + retrieval-augmented decision support**, evaluated under
a **strict Leave-One-Subject-Out (LOSO)** protocol with imbalance-aware metrics.

> **Top priority of this codebase is correctness and leakage prevention**, not
> model fanciness. Test-subject windows/embeddings never enter training,
> preprocessing, the memory bank, or threshold tuning — and there are runtime
> assertions that prove it.

## Research questions

- **RQ1** — Does retrieval-augmented contrastive representation learning improve
  cross-subject seizure detection under LOSO?
- **RQ2** — What is the role of cross-subject vs same-subject retrieved neighbors?
- **Quantum (exploratory)** — Does a quantum-simulated kernel give complementary
  retrieval behavior vs classical cosine/RBF similarity?

## Layout

```
v2/
  src/eegrag/            # the package (maps to the requested src/<name> modules,
    config/              # namespaced under `eegrag` to avoid import collisions)
    utils/               #   seeding, backend detection (ROCm/CUDA/cuML/FAISS), IO
    data/                #   parquet loading, subject parsing, FeatureLayout, LOSO
    preprocessing/       #   train-only scaler/PCA/selection pipeline
    datasets/            #   torch Dataset + topology-preserving augmentations
    models/              #   encoders, heads, GPU-abstracted classical baselines
    losses/              #   SupCon / BalancedContrastive / ImbalanceAwareSupCon
    retrieval/           #   leakage-guarded memory bank, index, similarities, heads
    quantum/             #   optional quantum-simulated kernel (graceful fallback)
    evaluation/          #   AUC-PR, sensitivity, per-class F1, FP/hour, thresholds
    stats/               #   Friedman+Nemenyi, Wilcoxon, effect size r, bootstrap CI
    experiments/         #   LOSO runners
  scripts/               # CLI entry points
  configs/               # YAML configs
  outputs/               # metrics / predictions / embeddings / retrieval / stats / figures
```
  
## Install

Core deps are CPU-only and small. Install PyTorch separately to match your
accelerator (the code never assumes CUDA):

```bash
cd v2
pip install -e .                      # core
# pick ONE torch build:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0   # MI300X / ROCm
# pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA
# pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU

# optional extras (all have fallbacks):
pip install -e ".[faiss,gpu,quantum,stats]"
```

Backend capabilities are detected at runtime; see `eegrag.utils.backend`. On the
MI300X, `torch.cuda.is_available()` is `True` (HIP), `torch.version.hip` is set,
and CUDA-only paths (RAPIDS cuML, XGBoost-GPU) fall back to CPU automatically.

## Docker

Three interchangeable images (ROCm / CUDA / CPU), selected by compose profile.
The code auto-detects the backend, so the only real difference is the base image
and device wiring. The entrypoint prints the detected backend on startup.

```bash
# Point at the host dir holding *_features.parquet (defaults to ../outputs):
export EEGRAG_DATA_DIR=/abs/path/to/outputs

# AMD MI300X / ROCm (primary). Match the base tag to your host ROCm driver:
ROCM_PYTORCH_TAG=rocm6.2_ubuntu22.04_py3.10_pytorch_release-2.4.0 \
  docker compose --profile rocm run --rm rocm \
  python scripts/run_loso_contrastive.py --config configs/contrastive.yaml --data-dir /data

# NVIDIA CUDA:
docker compose --profile cuda run --rm cuda \
  python scripts/run_loso_classical.py --config configs/classical.yaml --data-dir /data

# CPU (portable; default CMD runs the synthetic smoke test):
docker compose --profile cpu run --rm cpu
```

Inside the container the read-only parquets are mounted at `/data` and results
land in the bind-mounted `outputs/`. The CPU image is verified to build and run
the full smoke test end-to-end.

## Data

The previous paper's per-recording feature parquets are the input
(`chbXX_YY_features.parquet`: `windowStartSec, windowEndSec, label`, then 23
bipolar channels × 8 feature types = 184 features). Point the pipeline at that
directory — paths are **never** hardcoded:

```bash
python scripts/prepare_chbmit_features.py --data-dir /path/to/outputs \
    --set data.cache_dir=/path/to/outputs/.eegrag_cache
```

`chb17a/b/c` are collapsed to subject `chb17` by `data.subject_regex` so LOSO is
correct.

## Run

```bash
# 1) classical baselines (LR / RBF-SVM / RandomForest / XGBoost)
python scripts/run_loso_classical.py --config configs/classical.yaml --data-dir <dir>

# 2) contrastive + retrieval (main experiment)
python scripts/run_loso_contrastive.py --config configs/contrastive.yaml --data-dir <dir>
#    ablation: --set loss.name=balanced_supcon   (BCL baseline)
#              --set loss.name=imbalance_supcon  (proposed)

# 3) re-sweep retrieval heads/similarity/k over saved embeddings (no retraining)
python scripts/run_retrieval_eval.py --config configs/retrieval.yaml --data-dir <dir> \
    --embeddings-dir outputs/contrastive_imbalance_supcon/embeddings

# 4) exploratory quantum-kernel retrieval (simulator; RBF fallback if no backend)
python scripts/run_quantum_retrieval_subset.py --config configs/quantum_subset.yaml \
    --data-dir <dir> --embeddings-dir outputs/contrastive_imbalance_supcon/embeddings

# 5) neighbor study (RQ2) + significance testing
python scripts/analyze_neighbors.py --metrics-dir outputs/contrastive_imbalance_supcon/metrics
python scripts/analyze_statistics.py \
    --csv outputs/classical/stats/per_fold_metrics.csv \
    --csv outputs/contrastive_imbalance_supcon/stats/per_fold_metrics.csv \
    --metric auc_pr --out outputs/stats/auc_pr_significance.json
```

## Metrics & statistics

Primary: **AUC-PR**, **seizure sensitivity/recall**, **per-class F1**, seizure
precision, **false positives/hour**. Accuracy and ROC-AUC are reported but never
headline. Results are aggregated **Mean ± Std across folds × seeds**. Model
comparisons follow the protocol: **>2 models → Friedman then Nemenyi**; **2
models → Wilcoxon signed-rank + effect size r + 95% bootstrap CI**.

## Leakage guarantees (read this)

- `data/loso.py::assert_no_subject_leakage` is called by every stage that
  consumes training data (preprocessing fit, encoder training set, memory bank,
  threshold tuning).
- The `MemoryBank` constructor hard-asserts the held-out subject is absent.
- The preprocessor (`scaler/PCA/selection`) is `fit` on training rows only, then
  `transform`-applied to test.
- Classical HP search uses subject-grouped CV (`GroupKFold` on subject id) — no
  subject straddles inner-train/inner-val.
- Decision thresholds are tuned on training-only scores (leave-self-out for the
  k-NN head) and frozen before touching the test subject.

## Smoke test

```bash
python -m pytest tests/ -q          # or:
python tests/test_smoke.py          # synthetic-data end-to-end (CPU)
```

See [tests/test_smoke.py](tests/test_smoke.py) — it fabricates a tiny multi-subject
dataset and runs the full contrastive → retrieval → metrics → stats path on CPU.
