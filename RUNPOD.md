# RunPod RTX PRO 6000 Blackwell

## Ten-hour budget comparison

Use `configs/contrastive_runpod_budget.yaml` for the first complete comparison:
all 23 patient folds, one seed, three loss variants, 15 epochs and up to 256
updates/epoch. This is 264,960 updates total: 0.74, 1.84 or 3.68 training hours
at 10, 25 or 50 ms/update. Evaluation overhead must be measured separately.

```bash
# From v2; the same installation/preflight instructions below apply.
python scripts/run_budget.py --data-dir ../outputs --max-hours 9.5
```

This command first benchmarks the full first-fold preprocessing, embedding
generation, linear probe, retrieval, metrics and output writes. It includes
the benchmark itself in the 9.5-hour wall-clock cap. The planner applies a
1.35x margin and reserves one hour, then selects a common update budget for
all losses. It keeps 256 updates/epoch when possible, reduces them to fit if
necessary, and refuses to launch below 32 updates/epoch. The selected budget
and exact configs are written to outputs/budget_plan/. Use `--plan-only` to
measure and write the plan without starting the three-loss sweep. Add
`--hourly-rate YOUR_RATE` to report the maximum compute cost at the cap.

The timeout stops the training subprocess, even if the estimated runtime was
too optimistic; incomplete runs keep their last completed epoch checkpoint.
The timeout does not guarantee that all folds finish. A runtime prediction
from one fold cannot guarantee throughput or convergence on every patient.
The script does **not** stop the RunPod itself: stop the Pod after completion
or timeout to stop GPU billing. Persistent storage continues to incur charges.
[RunPod storage and billing](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage)

Budget-specific choices: 20,000 non-seizure bank entries, 100,000 non-seizure
linear-probe training entries, all seizure entries retained, and no large
embedding archives. Full held-out test windows, predictions, metrics,
neighbor artifacts, and encoder checkpoints are retained. Embeddings can be
regenerated from checkpoints, but immediate standalone retrieval re-sweeps
require a new run with embedding saving enabled. Do not use --train-only with
this profile. Classical baseline searches and quantum experiments are outside
this ten-hour comparison. One seed cannot measure seed variability; repeat
selected methods later if the journal requires that evidence. Report this
fixed-update protocol and check learning/convergence rather than treating it
as equivalent to the original 50-epoch, three-seed experiment.

Use the RunPod PyTorch 2.8.0 image with a CUDA 12.8 or 12.9 build. The NVIDIA
card has 96 GB VRAM. GPU availability here is not verified remotely: the local
development host has CPU-only PyTorch. Run the preflight on the pod.

## Install and check

From the repository root, extract the feature archive once, then enter v2:

```bash
python -m zipfile -e outputs.zip .
cd v2
python -m pip install -r requirements-runpod.txt
python -m pip install -r requirements-runpod-rapids.txt
python -m pip install -e . --no-deps
python scripts/check_runpod.py --rapids
python -m pip check
```

The requirements deliberately do not install or replace torch. If preflight
reports an older CUDA build, the official matching wheel is:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

CuPy added CUDA 12.8 / Blackwell sm_100 and sm_120 support in 13.4.0. This
profile uses CuPy 13.6.x with CUDA 12.x. The preflight tests BF16 convolution
and backward, CuPy elementwise compilation, matrix multiplication, and FFT;
it checks the actual CuPy runtime and NVRTC versions as well as torch's CUDA
build. An import or nvidia-smi alone is insufficient.

The resident RunPod profiles require cuML for the frozen-embedding linear probe.
It is installed separately from the encoder requirements:

```bash
python -m pip install -r requirements-runpod-rapids.txt
python scripts/check_runpod.py --rapids
python -m pip check
```

RAPIDS 25.10 is a CUDA-12-compatible pinned stack, rather than an unrestricted
latest version that might require a different CUDA major version. Installing
it is needed for the GPU probe in the resident profiles. The older NumPy
profiles can still use a CPU probe. The resident profiles fail if cuML is
missing rather than silently downloading embeddings. GPU LR/SVM retain their
requested class weights.

## What changed

- RunPod profiles use `data.storage_backend: cupy`, `preprocessing.backend:
  cupy` and `training.gpu_resident: true`. Parquets are decoded one recording
  at a time on the CPU and uploaded into a preallocated CuPy matrix; a complete
  feature matrix is never assembled in host RAM. The host NPZ cache is bypassed.
  Labels, subject strings and window times remain in RAM for bookkeeping.
- Median imputation and standard/robust scaling fit on training patients only
  and run in CuPy. Raw and transformed features remain in VRAM across folds.
  The resident path rejects PCA/feature selection, which currently require a
  separate implementation.
- CuPy arrays and PyTorch tensors share device memory through DLPack, including
  the training input and extracted embeddings. There is no feature/embedding
  GPU-to-CPU-to-GPU round trip for training, the cuML probe or torch retrieval.
  [CuPy DLPack interoperability](https://docs.cupy.dev/en/stable/user_guide/interoperability.html)
- Final probabilities, top-k neighbor results, metrics, and persisted artifacts
  still move to RAM. Enabling embedding archives explicitly downloads embeddings
  for saving; the budget profile disables these archives. The preflight checks
  shared device pointers in both DLPack directions and executes CuPy preprocessing.
- The transformed training matrix remains on the GPU. Batches are sampled and
  augmented in vectorized torch operations; there is no per-window Python
  augmentation, worker IPC, or repeated CPU-to-GPU transfer.
- The focal contrastive objective computes its pairwise matrix once rather
  than twice, uses view/label ordering consistently, and keeps the log/exp
  calculations in float32 under BF16 autocast.
- The CNN's column order is a device buffer rather than a newly allocated tensor
  for every forward pass. Embedding extraction uses larger tensor batches and
  a preallocated output array.
- Preprocessing avoids full float64 matrices and limits median work to feature
  blocks. It is reused across seeds within a fold.
- Epoch checkpoints include model, optimizer, scheduler, scaler, RNG states,
  and a data/config fingerprint. Restart the same command to resume. Changed
  data/config requires a new experiment name. Mid-epoch interruption replays
  that epoch. Embedding/retrieval stages can rerun after restart.
- Retrieval is exact, tiled float32 torch search on CUDA. The new profile fails
  if CUDA is unavailable. Existing FAISS paths remain CPU paths. Searches are
  reused for all k values, and Euclidean/RBF share neighbor rankings.

The full profile retains 50 epochs and three seeds. `steps_per_epoch: null`
means floor(training rows / batch size) weighted draws per epoch, as before;
weighted sampling with replacement is not an exhaustive pass over unique rows.
The new sampler targets 50% seizure draws. The old `min_pos_per_batch: 16`
was actually a 16x minority boost, giving about 94% seizure draws, not a
minimum of 16 positives. These sampling protocols are different and must be
recorded in the experiments. The GPU path currently supports sequence_len=1.

The RunPod profile disables deterministic algorithms for throughput; random
seeds and RNG states are saved, but bitwise repeatability on CUDA is not
promised. Set `training.deterministic=true` if needed. TF32 is off by default
to retain float32 retrieval precision.

## Feature archive audit and protocol choices

The supplied outputs.zip contains 686 recording parquets, 3,537,881 windows,
and 2,365 positive labels. Merging chb21 into chb01 gives 23 physical patients.
The existing labels are preserved; their annotation correctness was not audited.

There are eight recording schemas with 179 to 307 total columns. The first
recording has 184 features, but the union across recordings contains **784
features plus three metadata columns = 787 total columns**. This explains the
relationship between the archive and the older manuscript's number, although
the manuscript calls 787 the feature count. The explicit union schema is saved
in configs/chbmit_feature_schema.json, with its order derived from the supplied
archive. No data values or labels were used to choose that schema.

Thirty-one recordings fail the old first-recording schema check. Some use
entirely different reference montages, so padding them into the original
184-column schema would erase their observed signal. The RunPod profile uses
the complete union and imputes unavailable features with training-patient
medians. All-missing training columns use zero. This retains observed channel
features and every recording, but missingness and montage variation remain
scientific limitations. The inherited archive includes ECG, VNS and dummy
channels; the union profile preserves these rather than claiming it is a
harmonized, EEG-only anatomical montage. EEG-only harmonization would be a
separate preprocessing change and ablation.

The raw feature matrix is approximately 10.33 GiB (float32). Raw plus transformed
train/test matrices occupy about 20.66 GiB before embeddings, preprocessing
workspace, model activations and allocator caches. The resident loader checks
free VRAM before allocation. It uses temporary per-recording host decode buffers
and does not require keeping the full matrix or fold copies in host RAM.
The benchmark reports CuPy pool usage/reservations and device memory usage at
report time, separately from PyTorch's allocation peak. Those are not a combined
peak measurement. GPU throughput and actual peak VRAM remain to be measured on
the RunPod. Using the 96 GiB capacity does not require artificially filling it.
Float32 train+test embeddings for all 23 folds and three seeds are about
116 GiB uncompressed per loss variant (349 GiB for three variants). NPZ
compression reduces this by a data-dependent amount and adds CPU time.
Put output_dir on persistent storage before starting; checkpoints and saved
embeddings should survive pod restarts. Storage costs are separate from GPU
hourly charges.

The retrieval bank caps the non-seizure class at 100,000 entries, retains every
seizure, and never caps the held-out test set. This is a declared bank-size
choice, not equivalent to full-bank retrieval; report a bank-size ablation.
Use `--set retrieval.max_bank_per_class=null` for the full bank, at much greater
cost. Calibration still uses the training bank with leave-self-out retrieval;
the patient-held-out calibration issue from the manuscript review remains.

## Measure before the full sweep

```bash
python scripts/benchmark_runpod.py --config configs/contrastive_runpod.yaml \
  --data-dir ../outputs --steps 30 --losses 1
```

Add `--hourly-rate YOUR_POD_RATE` for projected training cost and `--retrieval`
to measure embedding generation and a sample of real-embedding retrieval.
The script loads the real full first fold, warms up for 30 updates, times a
second 30-update epoch, and saves outputs/runpod_benchmark.json. Its short
training is for timing only, not reportable model results. Estimates scale
actual fold sizes, epochs, seeds, and requested loss variants; a 30% margin
is reported separately. Other-fold preprocessing, embedding IO, linear probes,
classical models, and quantum experiments are not included in training hours.

Before a GPU measurement, only conditional estimates are defensible. With the
audited data, batch size 2048, 23 folds, 50 epochs, and three seeds, one loss
requires **5,698,800 optimizer updates**:

| Measured update time | Full LOSO, one loss | Three loss variants |
|---|---:|---:|
| 10 ms | 15.83 h | 47.49 h |
| 25 ms | 39.58 h | 118.73 h |
| 50 ms | 79.15 h | 237.45 h |

These are scenarios, not a measured RTX PRO 6000 throughput prediction.
Multiply by the actual hourly rate and allow for the excluded stages. A
single chb01 fold and seed at 50 epochs has 79,900 updates: about 13, 33 or 67
minutes at the same respective update times, excluding other stages.

## Pilot, full training, and retrieval

```bash
# One patient, one seed, five epochs, 256 updates/epoch; timing/debug pilot.
python scripts/run_loso_contrastive.py --config configs/contrastive_runpod_pilot.yaml \
  --data-dir ../outputs --train-only

# Full encoder experiment: all physical patients, three seeds, 50 epochs.
python scripts/run_loso_contrastive.py --config configs/contrastive_runpod.yaml \
  --data-dir ../outputs --train-only

# Retrieval + linear/hybrid heads on the same pod, using saved embeddings.
python scripts/run_retrieval_eval.py --config configs/contrastive_runpod.yaml \
  --data-dir ../outputs --name runpod_retrieval_imbalance_supcon \
  --embeddings-dir outputs/runpod_imbalance_supcon/embeddings
```

The pilot's 1,280 updates take 13-64 seconds in the table's scenarios, but
loading, preprocessing, checkpoint writes and embedding extraction add time.
The pilot does not establish convergence or journal performance. Keep its
results separate from the full experiment. For loss ablations, pass both
`--set loss.name=...` and a distinct `--name`.

Do not immediately run the old exhaustive RBF-SVM grid. It has 32 combinations
x 3 inner folds + one refit = up to 97 fits per outer fold/seed, or 6,693 fits
over 23 folds and three seeds, before probability calibration overhead. A GPU
does not make this cost disappear; benchmark a single fit first. Earlier A100
runtime comments were not validated measurements and have been removed.

## Sources

- [CuPy 13.4.0 Blackwell release](https://github.com/cupy/cupy/releases/tag/v13.4.0)
- [CuPy installation and CUDA/NVRTC troubleshooting](https://docs.cupy.dev/en/stable/install.html)
- [Official PyTorch 2.8.0 CUDA wheels](https://pytorch.org/get-started/previous-versions/)
- [RAPIDS platform support](https://docs.nvidia.com/datascience/platform-support/index.html)
- [RAPIDS CUDA 12.8 pip support notice](https://docs.rapids.ai/notices/rsn0051/)
