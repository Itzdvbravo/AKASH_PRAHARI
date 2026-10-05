"""Evaluate binary change predictions against DynamicEarthNet temporal labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from adapters.dynamicearthnet.adapter import DynamicEarthNetAdapter
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from app.postprocessing.mask_refinement import refine_change_mask


def compute_metrics(prediction: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    pred = prediction > 0
    truth = expected > 0
    tp = int(np.count_nonzero(pred & truth))
    fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth))
    tn = int(np.count_nonzero(~pred & ~truth))
    precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": f1,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "pixel_accuracy": (tp + tn) / expected.size,
        "predicted_changed_pixels": int(np.count_nonzero(pred)),
        "expected_changed_pixels": int(np.count_nonzero(truth)),
        "pixel_count": int(expected.size),
    }


def evaluate(args: argparse.Namespace) -> dict:
    connection = sqlite3.connect(args.database)
    try:
        locations = args.locations or [args.location]
        rows_by_location = {
            location: connection.execute(
                "SELECT DISTINCT tile_id FROM tiles WHERE location_id = ? AND date = ? ORDER BY tile_id LIMIT ?",
                (location, args.date_before, args.max_tiles),
            ).fetchall()
            for location in locations
        }
    finally:
        connection.close()
    if not any(rows_by_location.values()):
        raise ValueError(f"No indexed tiles for {locations} on {args.date_before}")

    adapter = DynamicEarthNetAdapter(args.archive)
    detector = PixelDiffChangeDetector(threshold=args.threshold)
    tiles = []
    totals = {key: 0 for key in ("tp", "fp", "fn", "tn", "predicted_changed_pixels", "expected_changed_pixels", "pixel_count")}
    try:
        for location, rows in rows_by_location.items():
            if not rows:
                continue
            for (tile_id,) in rows:
                before = adapter.load_tile_array(location, args.date_before, tile_id)
                after = adapter.load_tile_array(location, args.date_after, tile_id)
                expected = (
                    adapter.load_semantic_tile(location, args.date_before, tile_id)
                    != adapter.load_semantic_tile(location, args.date_after, tile_id)
                )
                raw = detector.detect(before, after).mask
                predicted = refine_change_mask(raw, min_area_px=args.min_area_px)
                metrics = compute_metrics(predicted, expected)
                tiles.append({"location": location, "tile_id": tile_id, "metrics": metrics})
                for key in totals:
                    totals[key] += int(metrics[key])
    finally:
        adapter.close()

    micro = compute_metrics_from_counts(totals)
    changed_tiles = [
        tile for tile in tiles
        if tile["metrics"]["expected_changed_pixels"] or tile["metrics"]["predicted_changed_pixels"]
    ]
    macro = {
        key: float(np.mean([tile["metrics"][key] for tile in changed_tiles]))
        for key in ("precision", "recall", "f1", "iou", "pixel_accuracy", "specificity", "balanced_accuracy")
    }
    return {
        "dataset": "DynamicEarthNet-video",
        "split": "indexed training AOIs; tile-level results are diagnostic, not an independent test set",
        "detector": detector.__class__.__name__,
        "threshold": args.threshold,
        "locations": locations,
        "date_before": args.date_before,
        "date_after": args.date_after,
        "tile_count": len(tiles),
        "macro_tile_count": len(changed_tiles),
        "micro_metrics": micro,
        "macro_metrics": macro,
        "tiles": tiles,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "pixel_accuracy": (tp + tn) / counts["pixel_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip")
    parser.add_argument("--database", default="data/dynamicearthnet.db")
    parser.add_argument("--location", default="den1311_3077_13")
    parser.add_argument("--locations", nargs="*", default=None, help="Evaluate multiple locations; defaults to --location")
    parser.add_argument("--date-before", default="2018-01-01")
    parser.add_argument("--date-after", default="2019-01-01")
    parser.add_argument("--max-tiles", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--min-area-px", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = evaluate(args)
    serialized = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
