# Pretrained Model Registry

This document records all candidate ML models, their origins, licences, local storage paths, and phase status for TerraEyes.

| Model Identifier | Capability | Architecture / Origin | Licence | Local Path / Checkpoint | Phase |
|------------------|------------|-----------------------|---------|-------------------------|-------|
| `mock_embedding` | Text/Image Embedding | Random unit vectors (Seed-hash based) | N/A (Internal) | Built-in | Phase A (Prototype) |
| `pixel_diff` | Change Detection | Absolute difference + Otsu threshold | N/A (Internal) | Built-in | Phase A (Prototype) |
| `clip_vit_b32` | Text/Image Embedding | OpenAI ViT-B/32 CLIP | MIT | `models/clip_vit_b32.pt` | Phase B Candidate |
| `remote_clip` | Text/Image Embedding | RS5M RemoteCLIP (ViT-L/14) | MIT | `models/remote_clip_vit_l14.pt` | Phase B Candidate |
| `bit_cd` | Change Detection | Bitemporal Image Transformer | Apache 2.0 | `models/bit_cd_levir.pth` | Phase B Candidate |
| `mamba_cd` | Change Detection & Temporal SSM | Mamba Selective State Space Model | Apache 2.0 / Custom | `models/mamba_cd.pth` | Phase C Candidate |

## Licence & Offline Policy
1. No external API calls or weight downloads are permitted during application execution.
2. All model weights must be pre-downloaded using `scripts/download_models.py` during environment setup.
3. Open-source licences (MIT, Apache 2.0, BSD) must be respected and attributed.
