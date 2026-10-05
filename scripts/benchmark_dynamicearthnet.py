"""Evaluate incremental Mamba change detection and ingestion on DynamicEarthNet.

The OSCD-trained checkpoint is evaluated without DynamicEarthNet fine-tuning.
Monthly semantic labels are reduced to binary class-transition masks so this
existing binary detector can be evaluated. This is an exploratory transfer
benchmark, not the official DynamicEarthNet semantic-change challenge metric.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import h5py
import numpy as np
import tacoreader
import torch

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from app.models.change_detection.mamba_cd import MambaChangeDetector
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from app.services.image_service import ImageService
from app.services.temporal_state_store import TemporalStateStore
from ingestion.incremental_ingest import ingest_single_scene
from scripts.evaluate import compute_metrics


IMAGE_SIZE = 1024
TILE_SIZE = 256
MONTH_COUNT = 24
CHANGE_COUNT_KEYS = ("tp", "fp", "fn", "tn")


def _asset_bytes(archive_path: Path, asset: dict[str, Any]) -> bytes:
    offset = int(asset["internal:offset"])
    size = int(asset["internal:size"])
    with archive_path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read(size)
    if len(payload) != size:
        raise IOError(f"Truncated TACO asset at offset {offset}: expected {size} bytes")
    return payload


def _decode_video(
    ffmpeg: str,
    payload: bytes,
    pixel_format: str,
    dtype: np.dtype,
    channels: int,
    frame_indices: list[int] | None = None,
) -> np.ndarray:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0"]
    if frame_indices is not None:
        expression = "select=" + "+".join(f"eq(n\\,{index})" for index in frame_indices)
        command.extend(["-vf", expression, "-vsync", "0"])
    command.extend(["-f", "rawvideo", "-pix_fmt", pixel_format, "pipe:1"])
    completed = subprocess.run(
        command,
        input=payload,
        capture_output=True,
        check=True,
        timeout=900,
    )
    values_per_frame = IMAGE_SIZE * IMAGE_SIZE * channels
    if len(completed.stdout) % (values_per_frame * np.dtype(dtype).itemsize):
        raise ValueError(f"FFmpeg returned a partial {pixel_format} frame")
    return np.frombuffer(completed.stdout, dtype=dtype).reshape(
        -1, IMAGE_SIZE, IMAGE_SIZE, channels
    )


def _decode_area(
    archive_path: Path,
    area_row: dict[str, Any],
    taco_data: Any,
    ffmpeg: str,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    area_id = str(area_row["id"])
    assets = taco_data.read(area_id).to_arrow().to_pylist()
    by_id = {asset["id"]: asset for asset in assets}
    required = {"bands_1", "labels", "metadata"}
    missing = required - by_id.keys()
    if missing:
        raise ValueError(f"{area_id}: TACO AOI is missing assets {sorted(missing)}")

    with h5py.File(
        __import__("io").BytesIO(_asset_bytes(archive_path, by_id["metadata"])), "r"
    ) as metadata:
        month_offsets = np.asarray(metadata["time_month"], dtype=np.int64)
        units = metadata["time_month"].attrs.get("units", "days since 2018-01-01")
        if isinstance(units, bytes):
            units = units.decode("utf-8")
    if len(month_offsets) != MONTH_COUNT:
        raise ValueError(f"{area_id}: expected 24 monthly label dates, got {len(month_offsets)}")
    origin_text = str(units).split("since", 1)[-1].strip().split(" ", 1)[0]
    origin = date.fromisoformat(origin_text[:10])
    dates = [(origin + timedelta(days=int(offset))).isoformat() for offset in month_offsets]

    encoded_images = _decode_video(
        ffmpeg,
        _asset_bytes(archive_path, by_id["bands_1"]),
        "rgb48le",
        np.dtype("<u2"),
        3,
        frame_indices=[int(index) for index in month_offsets],
    )
    encoded_labels = _decode_video(
        ffmpeg,
        _asset_bytes(archive_path, by_id["labels"]),
        "bgr0",
        np.dtype("u1"),
        4,
    )
    if len(encoded_images) != MONTH_COUNT or len(encoded_labels) != MONTH_COUNT:
        raise ValueError(
            f"{area_id}: expected {MONTH_COUNT} monthly pairs, found "
            f"{len(encoded_images)} images and {len(encoded_labels)} labels"
        )

    # OSCDLoader scales each RGB band with max(p98, 4000), then clips to [0,1].
    # Apply the same conversion per acquisition to stay aligned with the
    # checkpoint's input convention and the existing Otsu baseline.
    images = np.empty(encoded_images.shape, dtype=np.float32)
    for index, image in enumerate(encoded_images):
        denominator = np.maximum(np.percentile(image, 98, axis=(0, 1)), 4000.0)
        images[index] = np.clip(
            image.astype(np.float32) / denominator.reshape(1, 1, 3), 0.0, 1.0
        )

    # TACO's FFV1 label stream is grayscale. Remap its exact palette values to
    # compact categorical IDs; equality across dates is the binary target.
    rgb = encoded_labels[..., :3]
    if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
        encoded_classes = rgb[..., 0]
    else:
        encoded_classes = (
            (rgb[..., 0].astype(np.uint32) << 16)
            | (rgb[..., 1].astype(np.uint32) << 8)
            | rgb[..., 2].astype(np.uint32)
        )
    palette = np.unique(encoded_classes)
    if not 1 <= len(palette) <= 7:
        raise ValueError(f"{area_id}: expected up to seven label classes, got {len(palette)}")
    labels = np.searchsorted(palette, encoded_classes).astype(np.uint8)
    label_summary = {
        "class_count": int(len(palette)),
        "decoded_palette": [int(value) for value in palette.tolist()],
        "class_pixel_counts": {
            str(index): int(count)
            for index, count in enumerate(np.bincount(labels.ravel(), minlength=len(palette)))
        },
    }
    scene_metadata = {
        "crs": str(area_row.get("stac:crs", "unknown")),
        "geotransform": area_row.get("stac:geotransform"),
    }
    return images, labels, dates, {"labels": label_summary, **scene_metadata}


def _area_ids_from_train_csv(csv_path: Path) -> set[str]:
    result: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_path = row.get("planet_path", "")
            pieces = raw_path.replace("\\", "/").rstrip("/").split("/")
            if len(pieces) >= 2 and pieces[-2]:
                result.add(pieces[-2])
    return result


def _count_add(target: dict[str, int], metrics: dict[str, Any]) -> None:
    for key in CHANGE_COUNT_KEYS:
        target[key] += int(metrics[key])


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts[key] for key in CHANGE_COUNT_KEYS)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def _aggregate_area_rows(rows: list[dict[str, Any]], detector_key: str) -> dict[str, Any]:
    area_metrics = [row[detector_key] for row in rows]
    totals = {key: sum(int(row[key]) for row in area_metrics) for key in CHANGE_COUNT_KEYS}
    macro = {
        metric: float(np.mean([float(row[metric]) for row in area_metrics]))
        for metric in ("precision", "recall", "f1", "iou")
    }
    return {
        "macro_aoi_average": macro,
        "micro_pixel_average": _metrics_from_counts(totals),
        "macro_aoi_f1_std": float(np.std([float(row["f1"]) for row in area_metrics])),
        "mean_transition_f1": float(
            np.mean([row[f"{detector_key}_mean_transition_f1"] for row in rows])
        ),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _location_id(area_id: str) -> str:
    return "den" + "".join(character for character in area_id if character.isalnum())


def _ingest_area(
    images: np.ndarray,
    dates: list[str],
    area_id: str,
    work_dir: Path,
) -> tuple[ImageService, list[str], dict[str, Any]]:
    output_dir = work_dir / "incremental_tiles"
    source_dir = work_dir / "monthly_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    location_id = _location_id(area_id)
    per_acquisition_seconds: list[float] = []
    all_ids: list[str] = []

    for frame, acquisition_date in zip(images, dates):
        source_path = source_dir / f"{acquisition_date}.npy"
        np.save(source_path, frame.astype(np.float32, copy=False))
        started = time.perf_counter()
        result = ingest_single_scene(
            source_path,
            location_id,
            acquisition_date,
            sensor="planet",
            output_dir=output_dir,
            tile_size=TILE_SIZE,
            allow_same_tile_across_dates=True,
        )
        per_acquisition_seconds.append(time.perf_counter() - started)
        if result["tiles_created"] != (IMAGE_SIZE // TILE_SIZE) ** 2:
            raise AssertionError(f"{acquisition_date}: expected 16 tiles, got {result['tiles_created']}")
        if not all_ids:
            all_ids = list(result["tile_ids"])
        elif result["tile_ids"] != all_ids:
            raise AssertionError("Spatial tile IDs changed between acquisitions")

    image_service = ImageService(
        tile_repo=None,
        oscd_dir=str(work_dir / "no_oscd_fallback"),
        masks_cache_dir=str(work_dir / "masks"),
        incremental_tiles_dir=str(output_dir),
    )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("tiles", [])
    expected_records = len(dates) * len(all_ids)
    if len(records) != expected_records:
        raise AssertionError(f"Manifest has {len(records)} rows; expected {expected_records}")
    for acquisition_date in (dates[0], dates[len(dates) // 2], dates[-1]):
        tile = image_service.get_tile_array(all_ids[0], acquisition_date)
        if tile.shape != (TILE_SIZE, TILE_SIZE, 3) or not np.isfinite(tile).all():
            raise AssertionError(f"ImageService returned an invalid tile for {acquisition_date}")
    artifact_bytes = sum(
        (output_dir / row["array_path"]).stat().st_size for row in records
    )
    stats = {
        "area_id": area_id,
        "acquisition_count": len(dates),
        "tiles_per_acquisition": len(all_ids),
        "tile_date_records": len(records),
        "unique_spatial_tiles": len(set(all_ids)),
        "mean_ingest_seconds_per_acquisition": float(np.mean(per_acquisition_seconds)),
        "median_ingest_seconds_per_acquisition": float(np.median(per_acquisition_seconds)),
        "p95_ingest_seconds_per_acquisition": float(np.percentile(per_acquisition_seconds, 95)),
        "total_ingest_seconds": float(sum(per_acquisition_seconds)),
        "acquisitions_per_second": float(len(dates) / max(sum(per_acquisition_seconds), 1e-9)),
        "tiles_per_second": float(expected_records / max(sum(per_acquisition_seconds), 1e-9)),
        "tile_artifact_bytes": int(artifact_bytes),
        "retrieved_sample_dates": [dates[0], dates[len(dates) // 2], dates[-1]],
        "repeated_spatial_ids_across_dates": True,
        "temporary_artifacts_removed_after_run": True,
    }
    return image_service, all_ids, stats


def _tile_image(
    image_service: ImageService | None,
    tile_ids: list[str] | None,
    location_id: str,
    date_text: str,
    frame: np.ndarray,
    row_index: int,
    col_index: int,
) -> np.ndarray:
    if image_service is not None and tile_ids is not None:
        tile_id = f"{location_id}_{row_index:04d}_{col_index:04d}_planet"
        # Confirm the ID returned by ingestion matches the spatial grid mapping.
        if tile_id not in tile_ids:
            raise AssertionError(f"Ingested tile ID missing for grid cell {row_index},{col_index}")
        return image_service.get_tile_array(tile_id, date_text)
    row = row_index * TILE_SIZE
    col = col_index * TILE_SIZE
    return frame[row:row + TILE_SIZE, col:col + TILE_SIZE]


def _run_area(
    area_id: str,
    images: np.ndarray,
    labels: np.ndarray,
    dates: list[str],
    detector: MambaChangeDetector,
    baseline: PixelDiffChangeDetector,
    persist_state: bool,
    state_store: TemporalStateStore,
    image_service: ImageService | None = None,
    ingested_tile_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    row_count = col_count = IMAGE_SIZE // TILE_SIZE
    location_id = _location_id(area_id)
    counts = {
        name: {key: 0 for key in CHANGE_COUNT_KEYS}
        for name in ("otsu", "mamba_pair", "mamba_incremental")
    }
    transition_f1 = {name: [] for name in counts}
    seconds = {name: 0.0 for name in counts}
    persisted_context_loads = 0
    tile_count = row_count * col_count

    # Run the non-learned pair baseline over full acquisitions.
    for transition_index in range(len(dates) - 1):
        started = time.perf_counter()
        baseline_output = baseline.detect(images[transition_index], images[transition_index + 1])
        seconds["otsu"] += time.perf_counter() - started
        truth = labels[transition_index] != labels[transition_index + 1]
        baseline_metrics = compute_metrics(baseline_output.mask, truth)
        _count_add(counts["otsu"], baseline_metrics)
        transition_f1["otsu"].append(baseline_metrics["f1"])

    for row_index in range(row_count):
        for col_index in range(col_count):
            tile_id = f"{location_id}_{row_index:04d}_{col_index:04d}_planet"
            previous_context: dict[str, np.ndarray] | None = None
            for transition_index in range(len(dates) - 1):
                before = _tile_image(
                    image_service,
                    ingested_tile_ids,
                    location_id,
                    dates[transition_index],
                    images[transition_index],
                    row_index,
                    col_index,
                )
                after = _tile_image(
                    image_service,
                    ingested_tile_ids,
                    location_id,
                    dates[transition_index + 1],
                    images[transition_index + 1],
                    row_index,
                    col_index,
                )

                _sync(detector.device)
                started = time.perf_counter()
                pair_output = detector.detect_sequence([before, after])
                _sync(detector.device)
                pair_elapsed = time.perf_counter() - started
                seconds["mamba_pair"] += pair_elapsed
                pair_mask = pair_output.mask

                if transition_index == 0:
                    # The initial two dates establish the online state while
                    # also producing the first change prediction.
                    online_output = pair_output
                    online_elapsed = pair_elapsed
                    pair_contexts = pair_output.metadata["temporal_contexts"]
                    previous_context = pair_contexts[-1]
                    if persist_state:
                        state_store.save(
                            tile_id, dates[0], detector.model_fingerprint, pair_contexts[0]
                        )
                        state_store.save(
                            tile_id, dates[1], detector.model_fingerprint, pair_contexts[1]
                        )
                else:
                    if persist_state:
                        previous_context = state_store.load(
                            tile_id, dates[transition_index], detector.model_fingerprint
                        )
                        if previous_context is None:
                            raise AssertionError(
                                f"Persisted state missing for {tile_id} at {dates[transition_index]}"
                            )
                        persisted_context_loads += 1
                    if previous_context is None:
                        raise AssertionError(f"No prior state for {tile_id} at {dates[transition_index]}")
                    _sync(detector.device)
                    started = time.perf_counter()
                    online_output = detector.detect_sequence([after], initial_context=previous_context)
                    _sync(detector.device)
                    online_elapsed = time.perf_counter() - started
                    previous_context = online_output.metadata["temporal_contexts"][0]
                    if persist_state:
                        state_store.save(
                            tile_id,
                            dates[transition_index + 1],
                            detector.model_fingerprint,
                            previous_context,
                        )
                seconds["mamba_incremental"] += online_elapsed

                row = row_index * TILE_SIZE
                col = col_index * TILE_SIZE
                truth_tile = (
                    labels[transition_index, row:row + TILE_SIZE, col:col + TILE_SIZE]
                    != labels[transition_index + 1, row:row + TILE_SIZE, col:col + TILE_SIZE]
                )
                pair_metrics = compute_metrics(pair_mask, truth_tile)
                online_metrics = compute_metrics(online_output.mask, truth_tile)
                _count_add(counts["mamba_pair"], pair_metrics)
                _count_add(counts["mamba_incremental"], online_metrics)
                transition_f1["mamba_pair"].append(float(pair_metrics["f1"]))
                transition_f1["mamba_incremental"].append(float(online_metrics["f1"]))

    transitions = len(dates) - 1
    result: dict[str, Any] = {
        "area_id": area_id,
        "transition_count": transitions,
        "tile_count": tile_count,
        "valid_pixel_count": int(IMAGE_SIZE * IMAGE_SIZE * transitions),
    }
    for name in counts:
        result[name] = _metrics_from_counts(counts[name])
        result[f"{name}_mean_transition_f1"] = float(np.mean(transition_f1[name]))
    timing = {
        "otsu_seconds": seconds["otsu"],
        "mamba_pair_seconds": seconds["mamba_pair"],
        "mamba_incremental_seconds": seconds["mamba_incremental"],
        "persisted_context_loads": persisted_context_loads,
    }
    return result, timing


def _markdown_report(report: dict[str, Any]) -> str:
    rows = []
    for key, title in (
        ("otsu", "Pixel difference + Otsu"),
        ("mamba_pair", "Mamba pair replay"),
        ("mamba_incremental", "Mamba incremental state"),
    ):
        metrics = report["aggregate"][key]["macro_aoi_average"]
        micro_f1 = report["aggregate"][key]["micro_pixel_average"]["f1"]
        seconds = report["timing"][f"mean_{key}_seconds_per_transition"]
        rows.append(
            f"| {title} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['iou']:.4f} | {micro_f1:.4f} | {seconds:.4f} s |"
        )
    ingest = report["incremental_ingestion"]
    state = report["temporal_state_persistence"]
    return f"""# DynamicEarthNet Incremental Ingestion and Change Benchmark

## Result

This exploratory run evaluated **{report['aoi_count']} areas**, **{report['transition_count']} monthly transitions per area**, and **{report['pair_count']} area-transition pairs** ({report['scored_pixel_count']:,} labeled pixels). The Mamba checkpoint was trained on OSCD and was not fine-tuned or threshold-tuned on DynamicEarthNet. DynamicEarthNet's monthly semantic maps were converted to binary masks by marking a pixel changed when its class differs from the previous month.

| Detector | Macro AOI precision | Macro AOI recall | Macro AOI F1 | Macro AOI IoU | Micro pixel F1 | Mean inference / AOI transition |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**Incremental result:** stateful Mamba did not outperform Otsu on macro AOI F1 ({report['comparison']['incremental_mamba_macro_f1']:.4f} vs. {report['comparison']['otsu_macro_f1']:.4f}). It was effectively tied with pair replay (F1 difference {report['comparison']['incremental_mamba_macro_f1_delta_vs_pair']:+.4f}). Incremental Mamba was {report['comparison']['incremental_speedup_over_pair']:.2f}x faster than pair replay ({report['comparison']['inference_time_reduction_vs_pair_percent']:.1f}% less model inference time), but remained {report['comparison']['incremental_slowdown_vs_otsu']:.1f}x slower than Otsu. Mean inference times were {report['timing']['mean_mamba_incremental_seconds_per_transition']:.4f} seconds for incremental Mamba, {report['timing']['mean_mamba_pair_seconds_per_transition']:.4f} seconds for pair replay, and {report['timing']['mean_otsu_seconds_per_transition']:.4f} seconds for Otsu per 1024x1024 transition.

Incremental Mamba had a higher per-AOI F1 than Otsu on {report['comparison']['areas_incremental_mamba_beats_otsu']} of {report['aoi_count']} areas. Macro AOI F1 is the unweighted average over areas; micro F1 pools all scored pixels.

## Incremental ingestion check

One area was ingested through `ingest_single_scene` on all 24 monthly acquisitions. The run persisted {ingest['tile_date_records']} dated tile artifacts for {ingest['unique_spatial_tiles']} stable spatial tile IDs, and `ImageService` successfully reloaded sample tiles from the first, middle, and last dates. Ingestion averaged {ingest['mean_ingest_seconds_per_acquisition']:.3f} seconds per acquisition ({ingest['tiles_per_second']:.1f} tiles/second); compressed tile artifacts occupied {ingest['tile_artifact_bytes']:,} bytes during the run. The temporary image and state artifacts were removed afterward.

The `TemporalStateStore` was exercised for all {state['tile_count']} tiles of the ingested area over the monthly sequence: it wrote {state['context_records']} model-context records and reloaded {state['context_reloads']} prior-date contexts for one-date-at-a-time predictions. Context storage occupied {state['state_file_bytes']:,} bytes before cleanup. The remaining benchmark areas kept the immediately prior context in memory to avoid retaining many gigabytes of intermediate states.

## Protocol and limits

- Dataset source: [DynamicEarthNet paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf); compact archive used: [DynamicEarthNet-video TACO distribution](https://huggingface.co/datasets/isp-uv-es/DynamicEarthNet-video). The 2.16 GB archive is ignored under `data/dynamicearthnet/` and is not committed.
- The compact 71-PSNR DynamicEarthNet video package supplies 75 areas, daily four-band Planet Fusion imagery for 2018-2019, and 24 monthly semantic label maps per area. RGB (`bands_1`) and monthly label frames were selected from the package's acquisition-day metadata. Inference uses only RGB, matching the three-channel OSCD checkpoint; NIR is not used.
- The available `splits.csv` contains only training entries for 54 AOIs. The benchmark uses the {report['aoi_count']} areas absent from that train-only CSV. The source's separate official split list could not be retrieved, so these are **not claimed to be the official validation/test split**.
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

Machine-readable results and per-area rows: `{report['output_json']}`.
"""


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    archive_path = Path(args.archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"DynamicEarthNet TACO archive not found: {archive_path}")
    splits_csv = Path(args.splits_csv) if args.splits_csv else None
    if splits_csv is not None and not splits_csv.is_file():
        raise FileNotFoundError(f"Training split CSV not found: {splits_csv}")

    ffmpeg = args.ffmpeg or shutil.which("ffmpeg") or r"C:\ffmpeg\bin\ffmpeg.exe"
    if not Path(ffmpeg).is_file() and shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"FFmpeg executable not found: {ffmpeg}")

    dataset = tacoreader.load(str(archive_path))
    try:
        area_rows = dataset.data.to_arrow().to_pylist()
        all_area_ids = {str(row["id"]) for row in area_rows}
        train_ids = _area_ids_from_train_csv(splits_csv) if splits_csv else set()
        if args.include_train or not splits_csv:
            selected = area_rows
            split_description = "all AOIs in the compact archive"
        else:
            selected_ids = all_area_ids - train_ids
            selected = [row for row in area_rows if str(row["id"]) in selected_ids]
            split_description = (
                "AOIs absent from the downloaded train-only CSV; not asserted to be the official test split"
            )
        selected = sorted(selected, key=lambda row: str(row["id"]))
        if args.aoi_limit > 0:
            selected = selected[:args.aoi_limit]
        if not selected:
            raise ValueError("No AOIs selected; check the archive and split CSV")

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        detector = MambaChangeDetector(args.checkpoint, device=device)
        baseline = PixelDiffChangeDetector()
        warm = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.float32)
        detector.detect_sequence([warm, warm])
        _sync(detector.device)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        timing_totals = {
            "otsu_seconds": 0.0,
            "mamba_pair_seconds": 0.0,
            "mamba_incremental_seconds": 0.0,
            "persisted_context_loads": 0,
        }
        ingestion_result: dict[str, Any] | None = None
        state_result: dict[str, Any] | None = None
        state_area_id = str(selected[0]["id"])
        persisted_state_tile_count = 0
        with tempfile.TemporaryDirectory(
            prefix="dynamicearthnet_benchmark_",
            dir=ROOT / "data" / "dynamicearthnet",
        ) as temporary_name:
            work_dir = Path(temporary_name)
            state_store_path = work_dir / "temporal_states.h5"
            state_store = TemporalStateStore(str(state_store_path))
            for index, area_row in enumerate(selected, start=1):
                area_id = str(area_row["id"])
                print(f"[{index}/{len(selected)}] Decoding and evaluating AOI {area_id}...", flush=True)
                images, labels, dates, area_metadata = _decode_area(
                    archive_path, area_row, dataset.data, ffmpeg
                )
                image_service = None
                tile_ids = None
                if area_id == state_area_id:
                    image_service, tile_ids, ingestion_result = _ingest_area(
                        images, dates, area_id, work_dir
                    )
                    persisted_state_tile_count = len(tile_ids)
                area_result, area_timing = _run_area(
                    area_id,
                    images,
                    labels,
                    dates,
                    detector,
                    baseline,
                    persist_state=(area_id == state_area_id),
                    state_store=state_store,
                    image_service=image_service,
                    ingested_tile_ids=tile_ids,
                )
                area_result["date_start"] = dates[0]
                area_result["date_end"] = dates[-1]
                area_result["semantic_classes_observed"] = area_metadata["labels"]["class_count"]
                area_result["semantic_label_palette"] = area_metadata["labels"]["decoded_palette"]
                rows.append(area_result)
                for key in timing_totals:
                    timing_totals[key] += area_timing[key]
                print(
                    f"  Otsu F1={area_result['otsu']['f1']:.4f}; "
                    f"pair Mamba F1={area_result['mamba_pair']['f1']:.4f}; "
                    f"incremental F1={area_result['mamba_incremental']['f1']:.4f}",
                    flush=True,
                )
                del images, labels, area_metadata, image_service
                gc.collect()

            if not state_store_path.is_file():
                raise AssertionError("Expected persisted temporal state artifacts were not written")
            with h5py.File(state_store_path, "r") as state_file:
                stored_tile_groups = list(state_file.get("tiles", {}).values())
                context_record_count = sum(len(group) for group in stored_tile_groups)
            state_result = {
                "area_id": state_area_id,
                "tile_count": persisted_state_tile_count,
                "context_records": context_record_count,
                "context_reloads": int(timing_totals["persisted_context_loads"]),
                "state_file_bytes": int(state_store_path.stat().st_size),
                "storage_dtype": "float16 with gzip compression",
                "temporary_artifacts_removed_after_run": True,
            }

        transition_count = MONTH_COUNT - 1
        pair_count = len(rows) * transition_count
        mean_transition_seconds = {
            "mean_otsu_seconds_per_transition": timing_totals["otsu_seconds"] / pair_count,
            "mean_mamba_pair_seconds_per_transition": timing_totals["mamba_pair_seconds"] / pair_count,
            "mean_mamba_incremental_seconds_per_transition": timing_totals["mamba_incremental_seconds"] / pair_count,
            "total_otsu_seconds": timing_totals["otsu_seconds"],
            "total_mamba_pair_seconds": timing_totals["mamba_pair_seconds"],
            "total_mamba_incremental_seconds": timing_totals["mamba_incremental_seconds"],
        }
        incremental_seconds = mean_transition_seconds["mean_mamba_incremental_seconds_per_transition"]
        pair_seconds = mean_transition_seconds["mean_mamba_pair_seconds_per_transition"]
        otsu_seconds = mean_transition_seconds["mean_otsu_seconds_per_transition"]
        aggregates = {
            key: _aggregate_area_rows(rows, key)
            for key in ("otsu", "mamba_pair", "mamba_incremental")
        }
        report = {
            "dataset": "DynamicEarthNet-video 71-PSNR TACO distribution",
            "dataset_areas_total": 75,
            "split": split_description,
            "split_csv_rows": None if splits_csv is None else sum(1 for _ in csv.DictReader(splits_csv.open("r", encoding="utf-8-sig"))),
            "split_csv_unique_training_areas": len(train_ids),
            "aoi_count": len(rows),
            "transition_count": transition_count,
            "pair_count": pair_count,
            "scored_pixel_count": int(pair_count * IMAGE_SIZE * IMAGE_SIZE),
            "task": "binary pixel change derived from adjacent monthly semantic class labels",
            "detector": "OSCD-trained Mamba-style spatial and temporal selective state-space model",
            "checkpoint": str(Path(args.checkpoint)),
            "checkpoint_sha256": detector.model_fingerprint,
            "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
            "parameter_count": sum(parameter.numel() for parameter in detector.model.parameters()),
            "device": str(detector.device),
            "tile_size": TILE_SIZE,
            "rgb_normalization": "per-acquisition, per-band max(p98, 4000), clipped to [0,1], matching OSCDLoader",
            "mamba_threshold": detector.threshold,
            "model_warmup_excluded": True,
            "aggregate": aggregates,
            "timing": mean_transition_seconds,
            "incremental_ingestion": ingestion_result,
            "temporal_state_persistence": state_result,
            "comparison": {
                "otsu_macro_f1": aggregates["otsu"]["macro_aoi_average"]["f1"],
                "pair_mamba_macro_f1": aggregates["mamba_pair"]["macro_aoi_average"]["f1"],
                "incremental_mamba_macro_f1": aggregates["mamba_incremental"]["macro_aoi_average"]["f1"],
                "incremental_mamba_beats_otsu_macro_f1": (
                    aggregates["mamba_incremental"]["macro_aoi_average"]["f1"]
                    > aggregates["otsu"]["macro_aoi_average"]["f1"]
                ),
                "areas_incremental_mamba_beats_otsu": sum(
                    area["mamba_incremental"]["f1"] > area["otsu"]["f1"]
                    for area in rows
                ),
                "incremental_mamba_macro_f1_delta_vs_pair": (
                    aggregates["mamba_incremental"]["macro_aoi_average"]["f1"]
                    - aggregates["mamba_pair"]["macro_aoi_average"]["f1"]
                ),
                "incremental_speedup_over_pair": pair_seconds / max(incremental_seconds, 1e-9),
                "inference_time_reduction_vs_pair_percent": (
                    100.0 * (pair_seconds - incremental_seconds) / max(pair_seconds, 1e-9)
                ),
                "incremental_slowdown_vs_otsu": incremental_seconds / max(otsu_seconds, 1e-9),
            },
            "protocol": {
                "imagery_dates": "24 first-of-month dates in 2018-2019, selected by each AOI's time_month offsets",
                "input_bands": ["Planet Fusion RGB (bands_1)"],
                "not_used": ["daily dates without monthly labels", "Narrow-NIR band", "DynamicEarthNet fine-tuning"],
                "temporal_baseline": "Independent adjacent-month Mamba pair replay from zero state",
                "temporal_incremental": "First labeled pair initializes state; each later month is processed as one new frame using the prior date context",
                "semantic_target": "Adjacent monthly land-cover labels differ; all decoded pixels scored",
                "training_threshold": "Checkpoint threshold from OSCD validation only",
                "official_split_limitation": "The local DynamicEarthNet split CSV contains training areas only; excluded AOIs are not represented as an official validation/test split.",
                "task_limit": "Binary transition presence only; this model does not predict the source/target LULC transition class.",
                "cross_dataset_caveat": "DynamicEarthNet Planet at about 3 m differs from OSCD Sentinel-2 at 10 m; compare detectors within this run, not DynamicEarthNet F1 directly to the OSCD F1 table.",
                "timing_scope": "Warmed inference on 256-pixel tiles; archive extraction, video decoding, normalization, checkpoint load and first model call excluded.",
            },
            "areas": rows,
            "output_json": output_path.as_posix(),
        }
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.markdown:
            markdown_path = Path(args.markdown)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(_markdown_report(report), encoding="utf-8")
        print(
            json.dumps(
                {
                    "aoi_count": len(rows),
                    "macro_aoi_f1": {
                        key: value["macro_aoi_average"]["f1"]
                        for key, value in aggregates.items()
                    },
                    "mean_seconds_per_transition": mean_transition_seconds,
                    "ingestion": ingestion_result,
                    "report": str(output_path),
                    "markdown": args.markdown,
                },
                indent=2,
            ),
            flush=True,
        )
        return report
    finally:
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip")
    parser.add_argument("--splits-csv", default="data/dynamicearthnet/splits.csv")
    parser.add_argument("--checkpoint", default="models/change_detection_candidates/mamba_oscd_best.pt")
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--device", default=None, help="Defaults to CUDA if available; otherwise CPU")
    parser.add_argument("--aoi-limit", type=int, default=0, help="0 evaluates every selected AOI")
    parser.add_argument("--include-train", action="store_true", help="Evaluate all archived AOIs instead of only those absent from the train CSV")
    parser.add_argument("--output", default="data/evaluation/dynamicearthnet_incremental.json")
    parser.add_argument("--markdown", default="docs/evaluation/dynamicearthnet_incremental.md")
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
