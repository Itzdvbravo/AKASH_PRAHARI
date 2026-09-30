# ADR-002: Spatio-Temporal Modeling Architecture (Mamba SSM vs. Transformers)

## Status
Accepted

## Context
Standard Transformer architectures scale quadratically $O(N^2)$ with token sequence length, making them computationally intensive when modeling long temporal series of high-resolution satellite tiles.

## Decision
We select **Mamba (Selective State Space Model)** for spatio-temporal hidden state propagation and temporal change detection, per the reference architecture.

## Consequences
- Linear $O(N)$ computational complexity with respect to temporal time steps and sequence length.
- Recurrent hidden state snapshots can be persisted in the temporal database (`temporal_states`), enabling incremental updates when a new satellite acquisition is ingested without reprocessing the full historical stack.
- Baseline detector (`pixel_diff_otsu`) provides immediate non-parametric verification prior to GPU training of the Mamba architecture.
