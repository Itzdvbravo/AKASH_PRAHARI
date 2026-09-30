# Data Provenance and Licensing

## Primary Target Dataset: OSCD (Onera Satellite Change Detection)
- **Origin**: ONERA (The French Aerospace Lab).
- **Authors**: Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018).
- **Reference**: "Urban change detection for Multispectral Earth Observation using convolutional neural networks", IGARSS 2018.
- **Sensor**: Sentinel-2 (Copernicus European Space Agency).
- **Licence**: Open access for research and benchmarking.
- **Bands**: 13 bands (B01–B12 plus B8A) across visible, red-edge, NIR, and SWIR.
- **Ground Truth**: Binary pixel masks marking urban changes between 2015 and 2018.
- **Cities**: 24 global metropolitan areas (14 train / 5 validation / 5 test).

## Pretrained Vision-Language Models
- **RemoteCLIP**: Trained on RS5M dataset (5 million remote sensing image-text pairs). Licence: MIT.
- **CLIP (OpenAI)**: Base model ViT-B-32. Licence: MIT.
