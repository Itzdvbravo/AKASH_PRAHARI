# DynamicEarthNet Incremental Ingestion and Change Benchmark

This benchmark is one evidence item, not full problem-statement validation.
See the [Problem Statement 26227 implementation status](problem_statement_26227_status.md)
for capability coverage and known gaps.

## Result

This exploratory run evaluated **21 areas**, **23 monthly transitions per area**, and **483 area-transition pairs** (506,462,208 labeled pixels). The Mamba checkpoint was trained on OSCD and was not fine-tuned or threshold-tuned on DynamicEarthNet. DynamicEarthNet's monthly semantic maps were converted to binary masks by marking a pixel changed when its class differs from the previous month.

| Detector | Macro AOI precision | Macro AOI recall | Macro AOI F1 | Macro AOI IoU | Micro pixel F1 | Mean inference / AOI transition |
|---|---:|---:|---:|---:|---:|---:|
| Pixel difference + Otsu | 0.1031 | 0.2782 | 0.1406 | 0.0775 | 0.1565 | 0.0621 s |
| Mamba pair replay | 0.1898 | 0.0480 | 0.0712 | 0.0374 | 0.0753 | 0.9985 s |
| Mamba incremental state | 0.1894 | 0.0482 | 0.0713 | 0.0375 | 0.0754 | 0.6581 s |

**Incremental result:** stateful Mamba did not outperform Otsu on macro AOI F1 (0.0713 vs. 0.1406). It was effectively tied with pair replay (F1 difference +0.0002). Incremental Mamba was 1.52x faster than pair replay (34.1% less model inference time), but remained 10.6x slower than Otsu. Mean inference times were 0.6581 seconds for incremental Mamba, 0.9985 seconds for pair replay, and 0.0621 seconds for Otsu per 1024x1024 transition.

Incremental Mamba had a higher per-AOI F1 than Otsu on 5 of 21 areas. Macro AOI F1 is the unweighted average over areas; micro F1 pools all scored pixels.

## Incremental ingestion check

One area was ingested through `ingest_single_scene` on all 24 monthly acquisitions. The run persisted 384 dated tile artifacts for 16 stable spatial tile IDs, and `ImageService` successfully reloaded sample tiles from the first, middle, and last dates. Ingestion averaged 0.773 seconds per acquisition (20.7 tiles/second); compressed tile artifacts occupied 133,335,215 bytes during the run. The temporary image and state artifacts were removed afterward.

The `TemporalStateStore` was exercised for all 16 tiles of the ingested area over the monthly sequence: it wrote 384 model-context records and reloaded 352 prior-date contexts for one-date-at-a-time predictions. Context storage occupied 154,211,177 bytes before cleanup. The remaining benchmark areas kept the immediately prior context in memory to avoid retaining many gigabytes of intermediate states.

## Protocol and limits

- Dataset source: [DynamicEarthNet paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf); compact archive used: [DynamicEarthNet-video TACO distribution](https://huggingface.co/datasets/isp-uv-es/DynamicEarthNet-video). The 2.16 GB archive is ignored under `data/dynamicearthnet/` and is not committed.
- The compact 71-PSNR DynamicEarthNet video package supplies 75 areas, daily four-band Planet Fusion imagery for 2018-2019, and 24 monthly semantic label maps per area. RGB (`bands_1`) and monthly label frames were selected from the package's acquisition-day metadata. Inference uses only RGB, matching the three-channel OSCD checkpoint; NIR is not used.
- The available `splits.csv` contains only training entries for 54 AOIs. The benchmark uses the 21 areas absent from that train-only CSV. The source's separate official split list could not be retrieved, so these are **not claimed to be the official validation/test split**.
- The task is binary change segmentation derived from semantic labels. It does not measure named class transitions or the official DynamicEarthNet semantic-change score. No-data labels were not separately identified in the compact mask stream; all decoded pixels are included.
- DynamicEarthNet is Planet imagery at roughly 3 m; OSCD is Sentinel-2 at 10 m. The evaluation uses a cross-sensor, cross-dataset transfer of an OSCD-trained model, and its scores are not directly comparable to the OSCD table.
- The OSCD reference results remain unchanged in `docs/evaluation/oscd_mamba_temporal.md`. This run compares Otsu, pair replay, and stateful inference on the same DynamicEarthNet acquisitions.
- Inference timings include warmed model/baseline predictions over tiles, excluding archive extraction, video decode, image normalization, checkpoint load, and model warm-up. Ingestion timing includes loading each staged NPY, preprocessing, tiling, and writing the manifest/tile artifacts; input staging and TACO decode are excluded.

## Reproduction

Install the development dependencies and make FFmpeg available on `PATH` first:

```powershell
pip install -r setup/requirements-dev.txt
```

```powershell
python scripts/benchmark_dynamicearthnet.py `
  --archive data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip `
  --splits-csv data/dynamicearthnet/splits.csv `
  --checkpoint models/change_detection_candidates/mamba_oscd_best.pt `
  --aoi-limit 0
```

Machine-readable results and per-area rows: `data/evaluation/dynamicearthnet_incremental.json`.
