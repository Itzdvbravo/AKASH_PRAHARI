# CP-2 Change-Detection Comparison on OSCD

## Result

The existing pixel-difference + Otsu detector is the only candidate in this
comparison that detects any OSCD test pixels. The LEVIR-CD-pretrained learned
checkpoints both predicted an all-background mask on each of the 10 held-out
OSCD cities. They are therefore not suitable for deployment without OSCD
fine-tuning.

| Candidate | Pretraining | Test macro F1 | Test macro IoU | Mean scene inference (CPU) |
|---|---|---:|---:|---:|
| Pixel difference + Otsu | None | 0.2914 | 0.1874 | 0.0181 s |
| BIT-CD | LEVIR-CD RGB | 0.0000 | 0.0000 | 0.8452 s |
| ChangeFormerV6 | LEVIR-CD RGB | 0.0000 | 0.0000 | 8.2048 s |
| BIT-CD, fine-tuned on OSCD train | OSCD train split | 0.2576 | 0.1583 | 0.8238 s |

All metrics are macro averages over the 10 test cities from the local OSCD
`test.txt`. Micro F1/IoU for Otsu were 0.3139/0.1862. The learned-model latency
includes 256x256 tile inference and edge-tile cropping; Otsu runs once per full
scene. Input loading and model initialization are excluded. These timings were
measured on CPU and are environment-specific.

## Evaluation setup

- OSCD images: rectified RGB composite B04/B03/B02, normalized by the existing
  OSCD loader to [0,1].
- Learned-model input: mapped to [-1,1], matching the official model preprocessing.
- Learned-model inference: non-overlapping 256x256 tiles; incomplete bottom and
  right tiles use edge replication and are cropped back to scene bounds.
- No OSCD test labels were used for training or tuning. BIT-CD was subsequently
  fine-tuned using only part of the official OSCD train split; see
  [oscd_bit_finetuning.md](oscd_bit_finetuning.md). ChangeFormer remains
  zero-shot on OSCD.
- Exact per-city metrics and runtime are in the ignored local reports:
  `data/evaluation/oscd_pixel_diff_test.json`,
  `data/evaluation/bit_cd_levir_on_oscd_test.json`, and
  `data/evaluation/changeformer_v6_levir_on_oscd_test.json`.

## Decision gate

Fine-tuning has been completed as an experiment. The fine-tuned BIT-CD raises
micro F1 over Otsu but has lower macro F1 and much slower CPU inference; neither
learned model meets the planned 0.4 F1 target. Keep the existing Otsu baseline
as backend default unless CP-2 deployment is explicitly approved after review
of the results and model terms. Training protocol and untouched-test results
are in [oscd_bit_finetuning.md](oscd_bit_finetuning.md).

The authors' BIT-CD and ChangeFormer repositories restrict their code to
research/non-commercial use. Review those terms before adopting either model
for any commercial deployment.

## Sources

- [Official BIT-CD repository](https://github.com/justchenhao/BIT_CD)
- [Official ChangeFormer repository](https://github.com/wgcban/ChangeFormer)
- [Official OSCD dataset page](https://rcdaudt.github.io/oscd/)
