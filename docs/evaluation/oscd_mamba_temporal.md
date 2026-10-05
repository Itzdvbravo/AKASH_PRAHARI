# Temporal SSM Change Detection Benchmark on OSCD

## Result

The prototype now includes a trained temporal selective state-space detector.
On the official 10-city OSCD test split it improves binary change segmentation
modestly over pixel-difference + Otsu, but is substantially slower on CPU. It
remains an experimental option; Otsu stays the application default.

| Detector | Macro precision | Macro recall | Macro F1 | Macro IoU | Micro F1 | Mean CPU inference / scene |
|---|---:|---:|---:|---:|---:|---:|
| Pixel difference + Otsu | 0.2783 | 0.4663 | 0.2914 | 0.1874 | 0.3139 | 0.0189 s |
| Mamba spatial + temporal SSM | **0.4516** | 0.2769 | **0.3104** | **0.1986** | **0.3504** | 0.4267 s |

On this CPU run, the Mamba model took **22.5×** as long per scene. It has
46,339 parameters and a 200,544-byte checkpoint. Mean learned-model throughput
was approximately 734,000 pixels/second, including per-tile inference and
conversion of model outputs. It trades lower recall for higher precision, and
its accuracy varies substantially by city. The modest F1 gain does not justify
replacing the baseline for this prototype.

**Evaluation caveat:** the test split was not used for fitting or threshold
selection, but an earlier temporal-only prototype was benchmarked on these
cities before the spatial-scan design was finalized. Treat these figures as an
exploratory comparison, not a clean one-shot estimate. For a final model choice,
evaluate once on a new geographic holdout that has not informed architecture
iteration.

## Training and evaluation protocol

- Dataset: local OSCD Sentinel-2 rectified B04/B03/B02 composites and official
  binary change masks.
- Split: the official 14 training cities were split deterministically into 11
  fit cities and 3 validation cities (`aguasclaras`, `beihai`, `bercy`). The
  official 10 test cities were excluded from fitting and threshold selection.
- Training: 12 epochs, seed 42, CPU, 256-pixel patches, batch size 2. Best
  validation macro F1 was 0.2803 at epoch 12; the probability threshold (0.55)
  was selected on validation cities only.
- Benchmark: each test scene was divided into non-overlapping 256-pixel tiles;
  partial edge tiles were zero-padded, matching the API's OSCD tile loader, and
  cropped back to their original size.
  Otsu ran on the full scene. Timing excludes image loading, preprocessing,
  checkpoint loading, and the first warm-up call.
- Environment: Windows, Python 3.12, PyTorch 2.14.1 CPU, no CUDA device.

Reproduce training and evaluation with:

```powershell
python scripts/train_mamba_cd.py --epochs 12 --batch-size 2
python scripts/benchmark_mamba_cd.py
```

The full training history and per-city test results are written to ignored local
artifacts under `data/evaluation/mamba_oscd_training.json` and
`data/evaluation/mamba_oscd_test.json`.

## What this benchmark does not establish

OSCD contains two acquisition dates per city and labels only stable/change
pixels. This benchmark therefore evaluates a two-date binary segmentation
case, not a land-cover transition task such as vegetation-to-building. The
model accepts longer sequences and persists per-date state, but OSCD cannot
validate multi-year sequence benefits or stateful incremental accuracy. A
dataset with three or more aligned acquisitions and semantic transition labels
is required before claiming semantic multi-temporal change detection.

The implementation is a plain PyTorch selective SSM reference scan with a
causal depthwise temporal convolution. It is Mamba-style, but does not use the
official fused CUDA scan; these CPU measurements are not representative of an
optimized GPU implementation.
