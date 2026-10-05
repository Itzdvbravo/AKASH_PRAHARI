# Pretrained Model Registry

This document records all candidate ML models, their origins, licences, local storage paths, and phase status for TerraEyes.

| Model Identifier | Capability | Architecture / Origin | Licence | Local Path / Checkpoint | Phase |
|------------------|------------|-----------------------|---------|-------------------------|-------|
| `mock_embedding` | Text/Image Embedding | Random unit vectors (Seed-hash based) | N/A (Internal) | Built-in | Phase A (Prototype) |
| `pixel_diff` | Change Detection | Absolute difference + Otsu threshold | N/A (Internal) | Built-in | Phase A (Prototype) |
| `clip_vit_b32` | Text/Image Embedding | OpenAI ViT-B/32 CLIP via OpenCLIP; 512-dim joint image/text embeddings | MIT | `models/clip_vit_b32.pt` | Phase B Selected (CP-1) |
| `remote_clip` | Text/Image Embedding | RemoteCLIP (ViT-B/32), remote-sensing-specific | Apache 2.0 | `models/remoteclip_vit_b32.pt` | Phase B Comparison Candidate |
| `bit_cd` | Change Detection | Official BIT-CD, pretrained on LEVIR-CD RGB | Research/non-commercial terms in official repository | `models/change_detection_candidates/bit_levir_best_ckpt.pt` | Phase B zero-shot comparison; OSCD test macro F1=0.0 |
| `bit_cd_oscd_finetuned` | Change Detection | BIT-CD fine-tuned on 11 OSCD train cities; 3 train cities held out for validation | Research/non-commercial terms in official repository; OSCD labels CC BY-NC-SA | `models/change_detection_candidates/bit_cd_oscd_finetuned.pt` | Experimental only; test macro F1=0.2576, not backend default |
| `changeformer_v6` | Change Detection | Official ChangeFormerV6, pretrained on LEVIR-CD RGB | Research/non-commercial terms in official repository | `models/change_detection_candidates/changeformer_levir_best_ckpt.pt` | Phase B Comparison Candidate; zero-shot OSCD test macro F1=0.0 |
| `mamba_cd` | Multi-temporal binary change detection | Internal PyTorch Mamba-style selective SSM, trained on OSCD train cities | Internal implementation; OSCD labels CC BY-NC-SA | `models/change_detection_candidates/mamba_oscd_best.pt` | Experimental; benchmark before enabling by default |

## Licence & Offline Policy
1. No external API calls or weight downloads are permitted during application execution.
2. All model weights must be pre-downloaded using `scripts/download_models.py` during environment setup.
3. Model-specific research/non-commercial restrictions must be reviewed before use; a publicly downloadable checkpoint is not automatically licensed for commercial deployment.

The approved CLIP adapter is `backend/app/models/embedding/clip_vit_b32.py`.
The one-time setup command is `python scripts/download_models.py
--model clip_vit_b32 --output-dir ./models`; it saves a local state dict. The
API requires that file and fails clearly when it is missing rather than
silently substituting synthetic embeddings. OpenCLIP supports local checkpoint
paths for model construction; see its [official usage documentation](https://github.com/mlfoundations/open_clip#usage).
