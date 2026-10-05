# DynamicEarthNet semantic change detection pilot

## Model and supervision

The Phase C prototype uses the repository's spatial and temporal selective-SSM encoder with two seven-class land-cover heads and a binary change head. It predicts a land-cover class for each date; an ordered transition is emitted only where the model predicts different classes and its binary head predicts change. DynamicEarthNet labels are used to train and evaluate the model, never to generate inference output.

The training script uses the local DynamicEarthNet-video PlanetFusion archive, a fixed 10,000 DN RGB scale, an area-disjoint validation split, weighted per-date class losses, and a binary change loss. The current CPU pilot trained on 10 AOIs (80 sampled patches) and validated on one held-out AOI (32 patches). The local split file has training AOIs only, so this validation area is not an official test set.

## Held-out pilot results

Evaluation used 16 tiles for each of two date pairs on AOI `den1487_3335_13`. These measurements are diagnostic because the validation set contains one AOI.

| Detector | Precision | Recall | F1 | IoU | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Semantic Mamba binary head | 9.45% | 96.41% | 17.21% | 9.42% | 49.75% |
| Pixel-difference baseline, threshold 0.3 | 0.67% | 0.006% | 0.012% | 0.006% | 49.96% |

The semantic model's exact changed-transition micro F1 was **0%** on this pilot. Its binary output marks too much of the scene as changed; the high recall comes with 3.1% specificity. This does not meet Phase C acceptance. Do not set `TERRAEYES_CHANGE_DETECTOR=semantic_mamba` for normal use until broader training and held-out evaluation show useful transition quality and lower false alarms.

## Current implementation

- `scripts/train_dynamicearthnet_semantic_cd.py` trains the model and writes `data/evaluation/dynamicearthnet_semantic_training.json` plus an ignored local checkpoint under `models/change_detection_candidates/`.
- `scripts/evaluate_dynamicearthnet_semantic_cd.py` compares the semantic model and pixel-difference baseline against DynamicEarthNet labels on selected held-out AOIs.
- The API returns a predicted transition-ID mask and per-transition pixel counts, area shares, and component boxes. The viewer renders a color overlay and lets an analyst click a transition to zoom to its region.
- `semantic_mamba` is opt-in. The existing pixel-difference path remains the default while this pilot is below acceptance.

The local machine has CPU-only PyTorch. Full AOI training, temporal-state regression across repeated ingestion, improved false-alarm suppression, and an independent DynamicEarthNet test evaluation remain open Phase C work.
