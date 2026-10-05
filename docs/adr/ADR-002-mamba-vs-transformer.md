# ADR-002: Spatio-Temporal Modeling Architecture (Mamba SSM vs. Transformers)

## Status
Accepted

## Context
Standard Transformer architectures scale quadratically $O(N^2)$ with token sequence length, making them computationally intensive when modeling long temporal series of high-resolution satellite tiles.

## Decision
We select **Mamba (Selective State Space Model)** for spatio-temporal hidden state propagation and temporal change detection, per the reference architecture.

## Consequences
- The prototype now has trainable bidirectional spatial scans, a temporal selective SSM over per-pixel features, a binary change decoder, and incremental HDF5 state persistence.
- The implementation uses a reference recurrent scan and causal temporal convolution; it does not use the official fused CUDA kernels or claim their measured throughput.
- OSCD contains only two image dates and binary change labels. Multi-date sequence support is implemented, but long-sequence behavior and semantic land-cover transitions remain unvalidated until a suitable dataset is added.
- `pixel_diff_otsu` remains the deployment default. A prototype run inspected OSCD test metrics during architecture iteration, so the current OSCD benchmark is exploratory; final model selection requires a new geographic holdout.
