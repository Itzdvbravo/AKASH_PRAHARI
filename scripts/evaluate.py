"""Evaluate the non-learning pixel-difference baseline on an OSCD split."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for directory in (root_dir, backend_dir, data_handling_dir):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from adapters.oscd.oscd_loader import OSCDLoader
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float | int]:
    """Compute binary segmentation metrics without resizing either mask."""
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Prediction/mask shape mismatch: {pred_mask.shape} != {gt_mask.shape}")

    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "iou": iou,
    }


def _read_city_list(oscd_root: Path, split: str) -> list[str]:
    split_file = oscd_root / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"OSCD split file not found: {split_file}")
    return [city.strip().lower() for city in split_file.read_text(encoding="utf-8").replace("\n", "").split(",") if city.strip()]


def _load_mask(labels_root: Path, city: str) -> np.ndarray:
    candidates = list(labels_root.glob(f"**/{city}/cm/cm.png"))
    if not candidates:
        candidates = list(labels_root.glob(f"**/{city}/*cm*.tif"))
    if not candidates:
        raise FileNotFoundError(f"No OSCD change mask found for {city} under {labels_root}")
    path = candidates[0]
    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        with rasterio.open(path) as source:
            mask = source.read(1)
        return (mask > 0).astype(np.uint8)
    mask = np.asarray(Image.open(path).convert("L"))
    return ((mask == 255) | (mask == 1)).astype(np.uint8)


def run_evaluation(oscd_dir: str, labels_dir: str, split: str, output_path: str) -> dict:
    oscd_root = Path(oscd_dir)
    labels_root = Path(labels_dir)
    loader = OSCDLoader(oscd_root)
    cities = _read_city_list(oscd_root, split)
    city_dirs = loader.find_city_directories()
    detector = PixelDiffChangeDetector()

    per_city = []
    for city in cities:
        if city not in city_dirs:
            raise FileNotFoundError(f"OSCD imagery directory not found for split city {city}")
        scene_dir = city_dirs[city]
        before = loader.load_band_composite(scene_dir, time_index=1)
        after = loader.load_band_composite(scene_dir, time_index=2)
        gt = _load_mask(labels_root, city)
        if before.shape != after.shape or before.shape[:2] != gt.shape:
            raise ValueError(
                f"{city}: date/mask spatial shapes differ: before={before.shape}, "
                f"after={after.shape}, mask={gt.shape}; refusing to resize evaluation data"
            )
        started = time.perf_counter()
        prediction = detector.detect(before, after)
        inference_seconds = time.perf_counter() - started
        metrics = compute_metrics(prediction.mask, gt)
        per_city.append({"city": city, **metrics, "inference_seconds": inference_seconds})
        print(
            f"{city:<15} F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"inference={inference_seconds:.4f}s"
        )

    totals = {key: sum(city[key] for city in per_city) for key in ("tp", "fp", "fn", "tn")}
    micro = compute_metrics_from_counts(**totals)
    macro = {
        metric: float(np.mean([city[metric] for city in per_city]))
        for metric in ("precision", "recall", "f1", "iou")
    }
    report = {
        "dataset": "OSCD",
        "split": split,
        "detector": "pixel_diff_otsu",
        "input_bands": ["B04", "B03", "B02"],
        "city_count": len(per_city),
        "aggregation": {"macro_city_average": macro, "micro_pixel_average": micro},
        "mean_scene_inference_seconds": float(np.mean([city["inference_seconds"] for city in per_city])),
        "cities": per_city,
        "limitation": "Untrained RGB pixel-difference baseline; scores are not evidence of OSCD model training.",
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Macro city average:", json.dumps(macro))
    print("Micro pixel average:", json.dumps(micro))
    print(f"Report: {target}")
    return report


def compute_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--labels-dir", default="./data/oscd_labels")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output", default="./data/evaluation/oscd_pixel_diff_test.json")
    args = parser.parse_args()
    run_evaluation(args.oscd_dir, args.labels_dir, args.split, args.output)
