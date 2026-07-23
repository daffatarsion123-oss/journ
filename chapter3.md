# Chapter 3 — Methodology

## Retrieval-Augmented Cross-Subject EEG Learning for Seizure Detection

This chapter describes the methodology of the proposed journal extension. The
conference paper established that, on CHB-MIT EEG under a strict subject-wise
(Leave-One-Subject-Out, LOSO) protocol with extreme class imbalance, classical
supervised models behave pathologically: Logistic Regression attains high
seizure recall but at the cost of many false positives, while Random Forest and
XGBoost collapse toward the majority (non-seizure) class. The extension moves
beyond classical supervised classification into **contrastive representation
learning** coupled with a **retrieval-augmented decision layer**, and adds an
exploratory **quantum-simulated kernel** analysis. Every design decision is
subordinated to one overriding constraint: **no information from the held-out
test subject may influence any part of training, calibration, or retrieval.**

Each section ends with a pointer to the module that implements it, so the
methodology and the codebase (`src/eegrag/`) remain in lock-step.

---

## 3.1 Research questions and hypotheses

- **RQ1 — Representation + retrieval.** Does retrieval-augmented contrastive
  representation learning improve cross-subject seizure detection under LOSO,
  relative to (a) the classical baselines and (b) a linear probe on the learned
  representation?
- **RQ2 — Neighbor structure.** What role do *cross-subject* versus
  *same-subject* retrieved neighbors play? Under strict LOSO the memory bank
  contains no windows from the test subject, so retrieval is *forced* to be
  cross-subject; we quantify how strongly seizure decisions are supported by
  seizure-like neighbors drawn from other subjects, and how diverse those
  neighbors are.
- **RQ3 (exploratory) — Quantum kernels.** Does a quantum-simulated fidelity
  kernel induce *complementary* neighbor structure compared with classical
  cosine / RBF similarity (measured by neighbor-set overlap), rather than merely
  reproducing it?

The central methodological hypothesis is that an imbalance-aware contrastive
embedding, queried against a leakage-free cross-subject memory bank, yields
higher seizure-class **AUC-PR** and **sensitivity** at a controlled
**false-positive rate** than classical baselines, and that retrieval evidence
(seizure-neighbor support) provides an interpretable basis for each decision.

---

## 3.2 Dataset and feature representation

We use the per-recording, window-level feature tables produced by the
conference-paper pipeline (CHB-MIT, 23-channel bipolar montage, 256 Hz). Each
EEG recording is segmented with a **2 s sliding window and 1 s step** (50 %
overlap). For each window and each of the 23 bipolar channels, eight handcrafted
features are extracted — statistical: *Mean, Std, Variance, Entropy* (64-bin) —
and spectral band powers (Welch): *Delta (0.5–4 Hz), Theta (4–8), Alpha (8–13),
Beta (13–30)* — giving a **184-dimensional** feature vector per window
($23 \times 8$), plus `windowStartSec`, `windowEndSec`, and a binary `label`
(1 = window overlaps a clinician-annotated seizure interval).

The feature vector is not treated as an unstructured bag of 184 numbers. A
**feature-layout** object parses the column names back into a
`(channel, feature-type)` grid, recovering EEG topology. This enables two
things the conference paper did not exploit: (i) a 1-D convolutional encoder
that convolves along the *electrode* axis, and (ii) augmentations that operate on
whole electrodes or whole frequency bands rather than arbitrary coordinates
(§3.6). Subjects `chb17a/b/c` (session splits of one physical subject) are
collapsed to a single subject `chb17` so the LOSO partition is anatomically
correct.

> Implements: `data/loading.py`, `data/feature_layout.py`.

---

## 3.3 Problem formulation

Let $x_i \in \mathbb{R}^{184}$ be the feature vector of window $i$, with label
$y_i \in \{0,1\}$ and subject id $s_i \in \mathcal{S}$. Seizure windows are the
positive minority, with prevalence on the order of $10^{-3}$ to $10^{-4}$
(≈0.067 % in the conference setting). The task is binary classification of
$y_i$ under (a) extreme imbalance and (b) a requirement to **generalize to an
unseen subject**. Both properties are addressed explicitly: imbalance through
the loss and the evaluation metrics (§3.7, §3.12), generalization through the
LOSO protocol (§3.4).

---

## 3.4 LOSO protocol and data-leakage controls

For $|\mathcal{S}|$ subjects we run $|\mathcal{S}|$ folds. In fold $k$, subject
$s_k$ is the **test** subject and all remaining subjects form the **training**
pool. Critically:

> **Every quantity that is *fit* — feature scaling, PCA, feature selection, the
> encoder, the linear probe, the retrieval memory bank, and the decision
> threshold — is derived from training subjects only. The test subject's windows
> are touched exactly once, at final scoring.**

This is enforced in three layers rather than by convention:

1. **Index-level disjointness.** A fold object stores train/test *row indices*;
   its constructor asserts the two sets are disjoint and that the test subject
   never appears among the training subjects.
2. **A reusable assertion** (`assert_no_subject_leakage`) is called by *every*
   stage that consumes training data — the preprocessing fit, the encoder
   training set, the memory-bank construction, and threshold tuning. It checks
   that any "training-derived" index set excludes the test subject and lies
   wholly inside the training partition.
3. **A guarded memory bank.** The memory-bank constructor hard-asserts that the
   held-out subject contributes zero windows; constructing a bank that contains
   the test subject raises immediately.

These guards are covered by dedicated negative tests (`tests/test_leakage.py`)
that confirm the assertions *fire* when leakage is deliberately injected.

> Implements: `data/loso.py`, `preprocessing/pipeline.py`, `retrieval/memory_bank.py`.

---

## 3.5 Preprocessing (train-only)

Within each fold, a single preprocessing object is `fit` on the training rows
and then `transform`-applied to both partitions:

- non-finite feature values (rare NaN/Inf from entropy or band-power edge cases)
  are imputed with **training-set column medians**;
- features are standardized (z-score) using **training** means/variances
  (`RobustScaler` optional);
- optional univariate feature selection (ANOVA-F, top-$k$) and optional PCA
  (variance-retention target) are likewise fit on training only.

Because standardization centers features at the training mean, "zeroing" a
masked feature in augmentation (§3.6) corresponds to neutral mean-imputation —
a principled choice rather than an arbitrary constant.

> Implements: `data/scaling.py`, `preprocessing/pipeline.py`.

---

## 3.6 Physiology-preserving augmentation

Contrastive learning requires stochastic *views* of each anchor. EEG topology
must be preserved, so we explicitly **exclude** image-style augmentations
(random spatial flips, arbitrary cropping) and arbitrary cutoff-frequency
shifting. The augmentation family operates on the structured feature grid:

- **Gaussian noise injection** — additive jitter in standardized feature units;
- **Mild amplitude scaling** — multiply by $\gamma \sim \mathcal{U}(1-\delta, 1+\delta)$;
- **Channel (electrode) dropout / random electrode masking** — zero *all*
  features of a few randomly chosen electrodes (respects the montage);
- **Frequency masking** — zero a whole *named band group* (Delta/Theta/Alpha/
  Beta) across channels; this is a band mask, **not** a learned/continuous
  cutoff shift, so it stays physiologically interpretable;
- **Temporal masking** — zero a contiguous span of timesteps, active only when
  windows are stacked into sequences (sequence length > 1).

Two independently augmented views per anchor are used by default.

> Implements: `datasets/augment.py`, `datasets/dataset.py`.

---

## 3.7 Representation learning

### 3.7.1 Encoders

A shared encoder maps a (transformed) window to an embedding
$z = f_\theta(x) \in \mathbb{R}^{d}$. Four interchangeable architectures are
provided, from a simple baseline upward:

- **MLP** over the flat 184-vector (baseline);
- **1-D CNN** over the electrode axis (feature-types as input channels) — the
  default, since it respects montage topology;
- **TCN** (dilated causal temporal convolutions) for window sequences;
- **1-D CNN stem + Transformer encoder** over the temporal axis (optional).

A projection head $g_\phi$ maps embeddings to the unit sphere for the
contrastive objective; the *embedding* $z$ (not the projection) is used
downstream for retrieval and the linear probe, following standard contrastive
practice.

> Implements: `models/encoders.py`, `models/heads.py`.

### 3.7.2 Contrastive objectives and the key ablation

For a batch with multiview features and labels, the **Supervised Contrastive**
(SupCon) loss pulls together embeddings sharing a label and pushes apart the
rest:

$$
\mathcal{L}^{\text{SupCon}} = \sum_{i \in I} \frac{-1}{|P(i)|}
\sum_{p \in P(i)} \log
\frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)},
$$

where $P(i)$ are same-label positives and $\tau$ is the temperature. Under
extreme imbalance, SupCon is dominated by the majority class. We therefore
compare three objectives head-to-head — the comparison the reviewer asked for:

1. **Standard SupCon** — baseline.
2. **Balanced Contrastive Learning (BCL)** — the per-anchor losses are averaged
   *within each class first and then across classes*, so the majority class
   cannot dominate the gradient purely by count.
3. **Imbalance-Aware SupCon (proposed)** — each anchor is weighted by a class
   weight $w_{c}$ (inverse batch frequency, normalized, or an explicit minority
   weight) and, optionally, a **focal-style** modulation that down-weights
   already-well-aligned anchors:
   $$
   w_i = w_{c(i)} \cdot \big(1 - p^{+}_i\big)^{\gamma}, \qquad
   p^{+}_i = \exp\!\big(\overline{\log p}^{\,+}_i\big),
   $$
   where $p^{+}_i$ is the mean softmax mass on $i$'s positives and $\gamma$ the
   focusing parameter. Setting $\gamma=0$ or disabling the focal term recovers
   pure class-balanced weighting, isolating the contribution of focal
   modulation.

A class-balanced batch sampler ensures seizure anchors actually appear in each
batch despite the $10^{-3}$ prevalence; an unsupervised NT-Xent (SimCLR) loss is
included as a sanity baseline.

> Implements: `losses/supcon.py`, `datasets/dataset.py::make_balanced_sampler`.

---

## 3.8 Retrieval-augmented decision

After training, the encoder produces embeddings for (i) all training windows,
forming the **memory bank**, and (ii) the test windows, forming **queries**. The
bank stores, for each entry, its embedding, label, and subject id — and excludes
the test subject by construction (§3.4). To bound cost on millions of windows,
the *majority* class may be sub-sampled while **all** seizure windows are kept.

For a query $q$, the bank returns its top-$k$ neighbors under one of three
similarity measures:

- **Cosine** $\;\cos(q, b)$ — via an inner-product index on L2-normalized vectors;
- **Euclidean** $\;\lVert q - b \rVert_2$, mapped to similarity $1/(1+d)$;
- **RBF kernel** $\;\exp(-\gamma \lVert q-b\rVert_2^2)$. Because the RBF kernel
  is monotone in Euclidean distance, top-$k$ retrieval coincides with Euclidean
  ranking, but the kernel *weights* differ and are used in scoring.

Indexing uses **FAISS** when available and falls back to scikit-learn
`NearestNeighbors` otherwise. Three decision heads are evaluated:

- **Linear head** — a class-weighted linear probe on the embedding;
- **k-NN memory-bank head** — similarity-weighted fraction of seizure neighbors,
  $\;\hat{p}^{\text{knn}}(q) = \frac{\sum_j w_j\, y_j}{\sum_j w_j}$;
- **Hybrid head** — convex blend
  $\;\alpha\,\hat{p}^{\text{lin}} + (1-\alpha)\,\hat{p}^{\text{knn}}$.

The neighbor count is swept over $k \in \{1, 5, 10, 20, 50\}$ for the analysis,
with one value designated for the decision.

> Implements: `retrieval/{memory_bank,index,similarity,heads}.py`.

---

## 3.9 Neighbor analysis (RQ2)

For every query we record, across each similarity and each $k$:

- the **seizure-neighbor ratio** (fraction of retrieved neighbors that are
  seizures);
- the **same-subject vs cross-subject** neighbor fractions (under LOSO the
  same-subject fraction is ≈0 by construction — itself a reported finding that
  the model is forced to generalize across subjects);
- **subject diversity** among neighbors (unique subjects / $k$);
- **seizure-prediction support** — conditioned on the model predicting seizure,
  how seizure-like are the retrieved neighbors? This links each decision to its
  cross-subject evidence;
- **neighbor-set overlap** (Jaccard) between cosine, RBF, and quantum retrieval,
  feeding RQ3.

> Implements: `retrieval/heads.py::neighbor_analysis`, `scripts/analyze_neighbors.py`.

---

## 3.10 Quantum-simulated kernel retrieval (exploratory)

As an exploratory sub-analysis we replace the classical similarity with a
**quantum fidelity kernel**, kept strictly modular and optional. The pipeline is:

$$
\text{embedding} \rightarrow \text{PCA/projection to } 4\text{–}8 \text{ dims}
\rightarrow \text{angle-encoded feature map } |\phi(x)\rangle
\rightarrow K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2
\rightarrow \text{retrieval ranking}.
$$

It runs **simulator-only** (PennyLane/Qiskit) and on a **compressed,
test-subject-free** bank with a capped number of queries, because the fidelity
kernel costs $\mathcal{O}(N_{\text{bank}} \cdot N_{\text{query}})$ circuit
evaluations and must never sit on the critical path of the full LOSO sweep. If
no quantum backend is installed, the module **falls back to a classical RBF
kernel and labels its output accordingly**, so nothing in the paper is
mislabeled as "quantum". The analysis compares quantum vs cosine vs RBF neighbor
overlap (RQ3) and the seizure-support of quantum-retrieved neighbors.

> Implements: `quantum/kernel.py`, `experiments/quantum_subset.py`.

---

## 3.11 Classical baselines (reproduced under the same protocol)

To compare fairly against the conference paper, the classical baselines are
re-run under the identical LOSO protocol and identical metrics:

- **Logistic Regression**, **RBF-SVM**, **Random Forest**, **XGBoost**.

The **RBF-SVM** is emphasized as a strong nonlinear classical reference against
which representation learning + retrieval is measured; its hyperparameters
($C$, $\gamma$, `class_weight`) are searched with **subject-grouped**
cross-validation on the training subjects only (so no subject straddles
inner-train/inner-validation), selecting by AUC-PR. The implementation layer is
backend-abstracted and resolved at runtime: cuML / ThunderSVM / XGBoost-GPU are
used when present (CUDA), with automatic CPU fallback — important because the
primary platform is an **AMD MI300X under ROCm**, where those CUDA-only paths are
unavailable and the abundant system RAM absorbs CPU execution.

> Implements: `models/classical.py`, `experiments/loso_classical.py`.

---

## 3.12 Evaluation metrics

Given the imbalance, the **primary** metrics are:

- **AUC-PR** (average precision) — the headline metric;
- **Seizure-class sensitivity / recall** $= \mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$;
- **Per-class F1** (reported separately for seizure and non-seizure, so
  majority-class collapse is visible) and **seizure precision**;
- **False positives per hour**, $\mathrm{FP/h} = \mathrm{FP} / (\text{recording hours})$,
  for clinical relevance (with 1 s stride, hours $\approx N_{\text{windows}}/3600$).

**Accuracy** and **ROC-AUC** are reported for completeness but are explicitly
**not** headline metrics. Decision thresholds are tuned on **training-only**
scores (for the k-NN head via leave-self-out retrieval against the bank),
maximizing seizure F1 by default; a fixed-sensitivity criterion is available for
a clinically targeted operating point.

> Implements: `evaluation/{metrics,thresholds}.py`.

---

## 3.13 Statistical analysis

Results are aggregated as **Mean ± Std across folds and seeds**. Model
comparison follows a fixed protocol:

- **More than two models** → **Friedman** omnibus test on matched blocks
  (folds × seeds); if significant, a **Nemenyi** post-hoc with average ranks and
  the critical difference (for a Demšar-style CD diagram).
- **Exactly two models** → **Wilcoxon signed-rank** test on the paired
  per-fold metric, the matched-pairs **effect size** $r = |Z|/\sqrt{N}$, and a
  **95 % percentile bootstrap confidence interval** on the mean paired
  difference.

The canonical reported bundle for a pairwise comparison is therefore:
*Mean ± Std + Wilcoxon p-value + effect size r + 95 % bootstrap CI.*

> Implements: `stats/{tests,aggregate}.py`, `scripts/analyze_statistics.py`.

---

## 3.14 Experimental setup and reproducibility

- **Hardware.** Primary training on a single **AMD MI300X (192 GB HBM3)** under
  ROCm with a ROCm PyTorch build; ≈240 GB system RAM is exploited by holding the
  full float32 feature matrix in memory and by CPU-side classical baselines.
  The codebase never assumes CUDA: backend capabilities (ROCm vs CUDA, cuML,
  FAISS, ThunderSVM, XGBoost-GPU, quantum) are detected at runtime.
- **Reproducibility.** Every stochastic stage carries an explicit seed;
  experiments are run over multiple seeds. Per-fold predictions, embeddings,
  thresholds, retrieved neighbors, metrics, and the fully-resolved config are
  persisted for audit.
- **Containers.** ROCm / CUDA / CPU Docker images are provided; the entrypoint
  prints the detected backend, and the CPU image self-verifies via an
  end-to-end synthetic smoke test.

> Implements: `utils/{seed,backend,io}.py`, `docker/`, `experiments/*`.

---

## 3.15 Summary

The methodology replaces the conference paper's single-stage supervised
classifier with a two-stage design — an imbalance-aware contrastive *encoder*
and a leakage-free cross-subject *retrieval* decision layer — evaluated under a
rigorously guarded LOSO protocol with imbalance-aware metrics and a principled
statistical comparison. The augmentation, retrieval, and quantum modules are all
physiology- and protocol-respecting, and an exploratory quantum-kernel analysis
probes whether quantum similarity adds complementary neighbor structure. The
mapping below ties each research question to the experiments that answer it.

| Research question | Primary experiment | Evidence |
|---|---|---|
| RQ1 (representation + retrieval) | `run_loso_contrastive` vs `run_loso_classical` | AUC-PR, sensitivity, FP/h; Friedman+Nemenyi / Wilcoxon |
| RQ2 (neighbor structure) | `run_retrieval_eval` + `analyze_neighbors` | seizure-neighbor ratio, cross-subject fraction, support |
| RQ3 (quantum kernels) | `run_quantum_retrieval_subset` | quantum vs cosine/RBF neighbor overlap |
| Loss ablation (proposed vs BCL) | `loss.name=imbalance_supcon` vs `balanced_supcon` | paired comparison on AUC-PR / sensitivity |
