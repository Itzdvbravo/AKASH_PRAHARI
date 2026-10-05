# OSCD Dataset Schema

## Verified local archive

The archive in the repository's `images/` directory was inspected with
`python scripts/verify_dataset.py --oscd-dir ./images` on 2026-10-02. It is the
OSCD **imagery archive** and contains 24 city folders. Every city has
`imgs_1`, `imgs_1_rect`, `imgs_2`, `imgs_2_rect`, `pair`, and `dates.txt`.
The rectangular folders contain the 13 expected Sentinel-2 bands as
`B01.tif` through `B12.tif` plus `B8A.tif`. The rectified images are
resampled/cropped to a common 10 m grid; for example Paris B04 is a 390 x 408
16-bit TIFF. The unrectified bands can have different spatial resolutions.

The `dates.txt` sidecar uses this format (not plain ISO dates):

```text
date_1: 20161130
date_2: 20171107
```

These values are normalized to `YYYY-MM-DD` by the metadata adapter. The root
`train.txt`, `test.txt`, and `all.txt` files define the actual local split:
14 training cities and 10 test cities. This differs from the city lists in the
initial plan; use the files shipped with the archive as the split authority.

## Ground-truth labels and limitations

The imagery archive and labels are separate. The official train and test label
archives are downloaded and extracted under `data/oscd_labels/` (an ignored,
local data directory), preserving separate train/test folders. The archives
contain masks at `<city>/cm/cm.png` (0=no change, 255=change) and GeoTIFF
variants at `<city>/cm/<city>-cm.tif` (0=no change, 1=change). They contain 14
training and 10 test city masks, respectively. Archive MD5 hashes match the
published OSCD/TorchGeo values. Labels are CC BY-NC-SA; preserve attribution and
non-commercial/share-alike terms when using or redistributing derived data.

`scripts/evaluate.py` evaluates the pixel-difference baseline on the held-out
split using full-scene masks and fails on shape mismatches rather than silently
resizing. `scripts/train_oscd_bit_cd.py` fine-tunes BIT-CD using only cities
from `train.txt`; the 10 test cities remain reserved for final evaluation.

The inspected rectified TIFFs do not expose GeoTIFF CRS/transform tags. The
current OSCD metadata adapter therefore supplies approximate city bounding
boxes from its coordinate table. Pixel-to-ground georeferencing is not
verified by this local imagery archive and must not be inferred from those
city-level boxes.

## Expected city layout

```text
images/
  all.txt
  train.txt
  test.txt
  <city>/
    dates.txt
    <city>.geojson              # present for some cities
    imgs_1/                     # native-resolution bands
    imgs_1_rect/                # rectified 13-band set
    imgs_2/
    imgs_2_rect/
    pair/                       # visualization PNGs, not analysis rasters
```

For analysis use `imgs_1_rect` and `imgs_2_rect`, falling back to the
unrectified folders only when needed. Standard RGB is B04/B03/B02; false
colour infrared is B08/B04/B03. Band TIFFs are single-channel 16-bit images.
The loader normalizes each band to float32 [0, 1].

## Verification

```powershell
python scripts/verify_dataset.py --oscd-dir ./images
```

The verifier checks the two date folders and all 13 expected band names for
each discovered city. It reports label coverage separately because OSCD's
image and label archives can be installed independently.
