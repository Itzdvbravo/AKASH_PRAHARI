# OSCD (Onera Satellite Change Detection) Dataset Schema

## Overview
The OSCD dataset comprises 24 pairs of multispectral Sentinel-2 images acquired between 2015 and 2018 globally.
Locations:
- Train cities (14): `aguasclaras`, `berlin`, `brasilia`, `canberra`, `chongqing`, `dubai`, `hongkong`, `lasvegas`, `milano`, `norrkoping`, `paris`, `rennes`, `sao-paulo`, `shanghai`
- Evaluation/Test cities (10): `abudhabi`, `beihai`, `beirut`, `bordeaux`, `cambridge`, `cuiaba`, `dhanbad`, `madrid`, `montpellier`, `rio`

## Expected Directory Hierarchy
```
$TERRAEYES_OSCD_DIR/
├── Onera Satellite Change Detection dataset - Images/  (or directly under root)
│   ├── <city_name>/
│   │   ├── imgs_1/  or  imgs_1_rect/  (date 1, e.g., 2015-2016)
│   │   │   ├── B01.tif
│   │   │   ├── B02.tif  (Blue - 10m)
│   │   │   ├── B03.tif  (Green - 10m)
│   │   │   ├── B04.tif  (Red - 10m)
│   │   │   ├── B05.tif
│   │   │   ├── B06.tif
│   │   │   ├── B07.tif
│   │   │   ├── B08.tif  (NIR - 10m)
│   │   │   ├── B8A.tif
│   │   │   ├── B09.tif
│   │   │   ├── B10.tif
│   │   │   ├── B11.tif  (SWIR-1 - 20m)
│   │   │   └── B12.tif  (SWIR-2 - 20m)
│   │   ├── imgs_2/  or  imgs_2_rect/  (date 2, e.g., 2017-2018)
│   │   │   └── ... (same bands)
│   │   └── dates.txt (optional acquisition timestamp sidecar)
│   └── ...
└── Onera Satellite Change Detection dataset - Train Labels/ (or cm/ within city folder)
    ├── <city_name>/
    │   └── cm/
    │       └── cm.png  (1 = change, 2 = no change or 255 = change, 0 = no change)
    └── ...
```

## Band Extraction & Normalization
- Standard RGB composite: `B04` (Red), `B03` (Green), `B02` (Blue)
- False Colour Infrared composite: `B08` (NIR), `B04` (Red), `B03` (Green)
- Reflectance DN values: typically scaled by 1/10000 or clipped to [0, 4000] and normalized to [0, 1] float32.
