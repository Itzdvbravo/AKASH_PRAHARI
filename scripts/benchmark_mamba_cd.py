"""Compare the trained temporal SSM with Otsu on held-out OSCD test cities."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters.oscd.oscd_loader import OSCDLoader
from app.models.change_detection.mamba_cd import MambaChangeDetector
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from scripts.evaluate import _read_city_list, compute_metrics


def predict_mamba_scene(detector, before: np.ndarray, after: np.ndarray, tile_size: int):
    height, width = before.shape[:2]
    prediction = np.zeros((height, width), dtype=np.uint8)
    elapsed = 0.0
    tile_count = 0
    device = detector.device
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            tile_before = before[row:row + tile_size, col:col + tile_size]
            tile_after = after[row:row + tile_size, col:col + tile_size]
            tile_h, tile_w = tile_before.shape[:2]
            pad_h, pad_w = tile_size - tile_h, tile_size - tile_w
            if pad_h or pad_w:
                padding = ((0, pad_h), (0, pad_w), (0, 0))
                tile_before = np.pad(tile_before, padding, mode="constant")
                tile_after = np.pad(tile_after, padding, mode="constant")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = detector.detect_sequence([tile_before, tile_after])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            prediction[row:row + tile_h, col:col + tile_w] = output.mask[:tile_h, :tile_w]
            tile_count += 1
    return prediction, elapsed, tile_count


def aggregate(rows: list[dict]) -> dict:
    counts = {name: sum(row[name] for row in rows) for name in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = (counts[name] for name in ("tp", "fp", "fn", "tn"))
    micro_precision = tp / max(tp + fp, 1)
    micro_recall = tp / max(tp + fn, 1)
    micro = {
        **counts,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }
    macro = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in ("precision", "recall", "f1", "iou")
    }
    return {
        "macro_city_average": macro,
        "macro_f1_std": float(np.std([row["f1"] for row in rows])),
        "micro_pixel_average": micro,
    }


def run_benchmark(args) -> dict:
    oscd_root, labels_root = Path(args.oscd_dir), Path(args.labels_dir)
    loader = OSCDLoader(oscd_root, labels_root)
    city_dirs = loader.find_city_directories()
    cities = _read_city_list(oscd_root, "test")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    detector = MambaChangeDetector(args.checkpoint, device=device)
    baseline = PixelDiffChangeDetector()
    checkpoint_size = Path(args.checkpoint).stat().st_size
    parameter_count = sum(parameter.numel() for parameter in detector.model.parameters())

    # Warm up the learned model; initialization and the first call are excluded.
    warm = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.float32)
    detector.detect_sequence([warm, warm])
    if detector.device.type == "cuda":
        torch.cuda.synchronize(detector.device)

    rows = []
    for city in cities:
        if city not in city_dirs:
            raise FileNotFoundError(f"OSCD image directory missing for test city {city}")
        scene = city_dirs[city]
        before = loader.load_band_composite(scene, time_index=1)
        after = loader.load_band_composite(scene, time_index=2)
        truth = loader.load_ground_truth_mask(city)
        if truth is None:
            raise FileNotFoundError(f"OSCD test mask missing for {city}")
        if before.shape != after.shape or before.shape[:2] != truth.shape:
            raise ValueError(
                f"{city}: image/mask shapes differ; evaluation will not resize labels"
            )

        started = time.perf_counter()
        baseline_output = baseline.detect(before, after)
        baseline_seconds = time.perf_counter() - started
        learned_prediction, learned_seconds, tile_count = predict_mamba_scene(
            detector, before, after, args.tile_size
        )
        baseline_metrics = compute_metrics(baseline_output.mask, truth)
        learned_metrics = compute_metrics(learned_prediction, truth)
        row = {
            "city": city,
            "height": int(truth.shape[0]),
            "width": int(truth.shape[1]),
            "tile_count": tile_count,
            "baseline": {**baseline_metrics, "inference_seconds": baseline_seconds},
            "mamba": {
                **learned_metrics,
                "inference_seconds": learned_seconds,
                "pixels_per_second": truth.size / max(learned_seconds, 1e-9),
                "seconds_per_tile": learned_seconds / max(tile_count, 1),
            },
        }
        rows.append(row)
        print(
            f"{city:<15} Mamba F1={learned_metrics['f1']:.4f} "
            f"Otsu F1={baseline_metrics['f1']:.4f} "
            f"Mamba={learned_seconds:.3f}s Otsu={baseline_seconds:.3f}s tiles={tile_count}"
        )

    baseline_rows = [row["baseline"] for row in rows]
    mamba_rows = [row["mamba"] for row in rows]
    baseline_aggregate = aggregate(baseline_rows)
    mamba_aggregate = aggregate(mamba_rows)
    mean_baseline_seconds = float(np.mean([row["inference_seconds"] for row in baseline_rows]))
    mean_mamba_seconds = float(np.mean([row["inference_seconds"] for row in mamba_rows]))
    report = {
        "dataset": "OSCD",
        "split": "official test (10 held-out cities)",
        "task": "binary change segmentation from a two-date sequence",
        "model": "Mamba-style spatial and temporal selective state-space network",
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_bytes": checkpoint_size,
        "parameter_count": parameter_count,
        "model_config": {
            "in_channels": detector.model.in_channels,
            "feature_dim": detector.model.feature_dim,
            "state_dim": detector.model.state_dim,
        },
        "device": str(detector.device),
        "tile_size": args.tile_size,
        "input_bands": ["B04", "B03", "B02"],
        "threshold": detector.threshold,
        "timing_scope": "warm model inference only; image loading, checkpoint loading, and OSCD preprocessing excluded",
        "baseline": {
            **baseline_aggregate,
            "mean_scene_inference_seconds": mean_baseline_seconds,
        },
        "mamba": {
            **mamba_aggregate,
            "mean_scene_inference_seconds": mean_mamba_seconds,
            "median_scene_inference_seconds": float(np.median([row["inference_seconds"] for row in mamba_rows])),
            "p95_scene_inference_seconds": float(np.percentile([row["inference_seconds"] for row in mamba_rows], 95)),
            "mean_pixels_per_second": float(np.mean([row["pixels_per_second"] for row in mamba_rows])),
        },
        "speed_ratio_baseline_over_mamba": mean_baseline_seconds / max(mean_mamba_seconds, 1e-9),
        "cities": rows,
        "semantic_label_limit": (
            "OSCD ground truth is binary stable/change; this benchmark cannot score land-cover "
            "transition classes such as vegetation-to-building."
        ),
        "training_split_policy": "Checkpoint validation threshold is selected only from OSCD train cities; test labels are not used for training or threshold selection.",
        "evaluation_caveat": "Exploratory: an earlier temporal-only prototype was benchmarked on these test cities before the spatial-scan design was finalized. Use a new geographic holdout for a clean final model-selection estimate.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline_macro": baseline_aggregate["macro_city_average"],
        "mamba_macro": mamba_aggregate["macro_city_average"],
        "speed_ratio_baseline_over_mamba": report["speed_ratio_baseline_over_mamba"],
        "report": str(output),
    }, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--labels-dir", default="./data/oscd_labels")
    parser.add_argument("--checkpoint", default="./models/change_detection_candidates/mamba_oscd_best.pt")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--output", default="./data/evaluation/mamba_oscd_test.json")
    run_benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
