# DynamicEarthNet binary change detector diagnostic

## Setup

The current `pixel_diff` detector is a binary RGB differencer, not a land-cover
transition model. The old image adapter normalized each monthly frame against
its own 98th percentile, which made pixel values incomparable across dates. The
adapter now uses a shared 10,000-DN scale and writes the new normalization to
versioned image cache files. The detector uses a fixed 0.30 max-channel
difference threshold and no cloud heuristic; the cloud heuristic suppressed
many true PlanetFusion changes.

## Tile diagnostic

The following metrics come from 128 tiles in eight AOIs, all in the indexed
DynamicEarthNet training split. This is a diagnostic subset, not an independent
test set. F1 and IoU are more useful here than pixel accuracy because changed
pixels are sparse.

| Pair | Precision | Recall | Micro F1 | Micro IoU | Pixel accuracy | Macro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2018-01 to 2018-02 | 9.8% | 12.5% | 11.0% | 5.8% | 91.8% | 7.7% |
| 2018-01 to 2019-01 | 40.2% | 33.7% | 36.7% | 22.5% | 91.2% | 22.7% |

Macro scores average tiles with at least one expected or predicted change;
all-negative tiles are excluded so they cannot inflate the macro F1.

The monthly result is still weak. The detector catches general image changes,
but cannot reliably distinguish real land-cover transitions from short-term
surface and radiometric variation.

## Semantic capability

The API scores the predicted binary mask against the monthly annotation
difference and returns precision, recall, F1, IoU, specificity, balanced
accuracy, and pixel accuracy. The annotations are used for evaluation only;
they are not rendered or copied into the predicted mask. The current detector
does not predict land-cover classes, so class-to-class transition predictions
remain unavailable until a semantic model is trained and validated.

A local RGB patch-classifier prototype was also checked with 46 AOIs for
training and eight different AOIs for validation. It reached 49.1% overall
land-cover accuracy, 14.0% exact transition accuracy on changed pixels, and
6.8% binary change F1 from its predicted class maps. Those results were not
good enough to expose as semantic detections, so the UI reports the limitation
instead of presenting annotations as model output.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_dynamicearthnet_cd.py `
  --locations den1311_3077_13 den1417_3281_13 den1487_3335_13 den1700_3100_13 `
              den2006_3280_13 den2029_3764_13 den2065_3647_13 den2415_3082_13 `
  --date-before 2018-01-01 --date-after 2019-01-01 --max-tiles 16 `
  --threshold 0.3 --output data/evaluation/dynamicearthnet_validation.json
```
