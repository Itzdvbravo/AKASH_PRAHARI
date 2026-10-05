# Data Provenance and Licensing

## Primary Dataset: DynamicEarthNet-video

- **Source**: [DynamicEarthNet paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf) and its [71 dB TACO video distribution](https://huggingface.co/datasets/isp-uv-es/DynamicEarthNet-video).
- **Local copy**: `data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip`; the compressed archive is ignored by git. The compact distribution contains 75 AOIs, daily PlanetFusion RGB imagery, and monthly label frames.
- **Resolution and metadata**: 1024 × 1024 AOIs at approximately 3 m GSD. The app reads each AOI's STAC CRS and affine transform for indexed tile bounds.
- **Semantic labels**: Monthly pixel labels use class IDs 0–6: impervious surface, agriculture, forest and other vegetation, wetlands, bare soil, water, and snow/ice. They are used as expected outputs for binary change-mask evaluation only. They are not rendered as detections or used to construct the detector mask. The configured pixel-difference detector has no land-cover classification head, so class-to-class model predictions are currently unavailable.
- **Index split**: `data/dynamicearthnet/splits.csv` training AOIs only. The index builder excludes AOIs not listed in that training split.
- **License**: The compact distribution identifies CC BY 4.0. Preserve attribution when redistributing source or derived data.
- **Index setup**: Run `python scripts/build_dynamicearthnet_index.py` to create the default `data/dynamicearthnet.db` and `data/faiss_dynamicearthnet.index.npz` artifacts before starting the API.

The application decodes monthly image and label frames lazily from the local TACO archive and persists decoded frames under `data/dynamicearthnet/decoded_frames` so later requests and API restarts can reuse them. The first request for a frame still needs video decoding. FFmpeg must be available on `PATH`; install Python dependencies from `setup/requirements.txt`.

## Legacy Dataset: OSCD

OSCD remains available for the historical benchmark and model-training scripts, but it is no longer the application's configured imagery source. Its masks contain binary stable/change annotations and do not support named land-cover transitions. OSCD's licensing and schema details are recorded in [`data-handling/adapters/oscd/oscd_schema.md`](../data-handling/adapters/oscd/oscd_schema.md).

## Pretrained Vision-Language Models

- **CLIP ViT-B/32**: The local OpenCLIP checkpoint is `models/clip_vit_b32.pt`; runtime inference does not fetch weights.
- **RemoteCLIP**: The comparison checkpoint has not been downloaded. See the [official RemoteCLIP repository](https://github.com/ChenDelong1999/RemoteCLIP).

## Legacy Mock UI Imagery: Provenance Unverified

City-labelled PNG pairs under `frontend/public/satellite/` are illustrative mock-mode assets. They contain no provenance metadata and must not be cited as real satellite observations.
