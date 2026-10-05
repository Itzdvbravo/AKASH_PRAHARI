# BIT-CD Fine-Tuning on OSCD

## Outcome

BIT-CD was fine-tuned as an experiment using only the official OSCD training
split. The ten official test cities were kept untouched until this final
evaluation. Results do not justify making the model the backend default:
fine-tuned BIT-CD improves micro F1 over the pixel-difference + Otsu baseline,
but has lower macro F1 and substantially slower CPU scene inference.

| Detector | Test macro F1 | Test macro IoU | Test micro F1 | Test micro IoU | Mean CPU inference per scene |
|---|---:|---:|---:|---:|---:|
| Pixel difference + Otsu | 0.2914 | 0.1874 | 0.3139 | 0.1862 | 0.0181 s |
| BIT-CD, LEVIR-CD pretrained (zero-shot) | 0.0000 | 0.0000 | — | — | 0.8452 s |
| BIT-CD, fine-tuned on OSCD train | 0.2576 | 0.1583 | 0.3420 | 0.2063 | 0.8238 s |

The planned OSCD test F1 target is 0.4; this experiment does not reach it.
Macro scores weight each city equally. Micro scores aggregate pixel counts
across cities, so the two summaries answer different questions.

## Training protocol

- Official `train.txt` cities only. Eleven cities supplied training patches;
  `aguasclaras`, `beihai`, and `bercy` were reserved for validation.
- The run generated 124 training patches of 256 × 256 pixels. Training used
  CPU, deterministic seed 42, AdamW with learning rate 1e-4, batch size 4,
  positive-class weight 6.469, early stopping patience 7, and a 30-epoch cap.
- Best validation macro F1 was 0.2361 at epoch 3. Early stopping ended the run
  at epoch 10. The best-validation checkpoint was retained.
- Inputs were OSCD RGB bands B04/B03/B02, normalized by the OSCD loader to
  [0,1] and mapped to [-1,1] for BIT-CD. Inference used non-overlapping
  256 × 256 tiles; incomplete edge tiles were replicated and cropped back to
  the scene extent.
- No test city or test mask was used for model selection, training, or tuning.
  The official test split was evaluated once after training.

## Artifacts and limitations

- Training script: `scripts/train_oscd_bit_cd.py`
- Evaluation script: `scripts/evaluate_change_models.py`
- Local training history: `data/evaluation/bit_cd_oscd_training.json`
- Local per-city test report: `data/evaluation/bit_cd_oscd_finetuned_test.json`
- Local checkpoint: `models/change_detection_candidates/bit_cd_oscd_finetuned.pt`

The data and model artifacts above are ignored by Git and remain local. The
experiment fine-tunes on only 11 of the 14 training cities because three are
held out for validation. The learned model is also slower than Otsu on this
CPU setup. The checkpoint remains experimental and was not integrated into or
selected by the backend. Review the upstream BIT-CD research/non-commercial
terms before any adoption beyond research use; see the links in
[the CP-2 comparison](oscd_change_detection_cp2.md#sources).
