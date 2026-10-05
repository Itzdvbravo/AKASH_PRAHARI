# Data Provenance and Licensing

## Primary Target Dataset: OSCD (Onera Satellite Change Detection)

- **Origin**: ONERA (The French Aerospace Lab).
- **Authors**: Daudt, R. C., Le Saux, B., Boulch, A., & Gousseau, Y. (2018).
- **Reference**: "Urban change detection for Multispectral Earth Observation using convolutional neural networks", IGARSS 2018.
- **Sensor**: Sentinel-2 (Copernicus European Space Agency).
- **Local copy**: Imagery is in `images/`; official train/test label archives are extracted under ignored `data/oscd_labels/`. The imagery README states that it contains modified Copernicus data from 2015-2018. Change labels are **CC BY-NC-SA**; preserve attribution and non-commercial/share-alike terms.
- **Bands**: 13 bands (B01-B12 plus B8A) across visible, red-edge, NIR, and SWIR.
- **Ground Truth**: Binary pixel masks marking urban changes between 2015 and 2018. Official train/test archive MD5 hashes match published OSCD/TorchGeo checksums; masks are not committed to git.
- **Cities and split**: 24 scenes. The local archive's `train.txt` and `test.txt` define 14 training and 10 test cities; there is no separate validation split in those files.
- **Schema record**: See [`data-handling/adapters/oscd/oscd_schema.md`](../data-handling/adapters/oscd/oscd_schema.md) for verified directory layout, dates, labels, and georeferencing limits.

## Pretrained Vision-Language Models

- **RemoteCLIP**: Remote-sensing vision-language model. The official repository is Apache 2.0; checkpoint use is documented at the [official RemoteCLIP repository](https://github.com/ChenDelong1999/RemoteCLIP). The ViT-B/32 comparison checkpoint has not been downloaded.
- **CLIP (OpenAI)**: Base model ViT-B-32. Licence: MIT.
- **Selected Phase B retrieval model**: CLIP ViT-B/32, downloaded during setup to `models/clip_vit_b32.pt` and used locally through OpenCLIP. Runtime inference does not fetch weights.

## Legacy Mock UI Imagery: Provenance Unverified

The mock UI includes city-labelled PNG pairs under
`frontend/public/satellite/<city>/before.png` and `after.png`. These are used by
curated mock-mode visualizations only; dataset mode loads imagery from the local
OSCD archive through the API. The PNG files contain no embedded provenance
metadata, and the repository history does not record their original provider,
author, or licence. Treat them as unverified illustrative assets: do not cite
them as real satellite observations or redistribute them until their provenance
and usage rights are confirmed. The separate generated fixtures in
`data-handling/fixtures/generate_fixtures.py` are deterministic synthetic arrays,
not captured satellite imagery.
