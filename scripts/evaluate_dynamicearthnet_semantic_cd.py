"""Compare DynamicEarthNet semantic Mamba predictions with the RGB baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from adapters.dynamicearthnet.adapter import DynamicEarthNetAdapter
from app.models.change_detection.mamba_cd import MambaChangeDetector
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from app.postprocessing.mask_refinement import refine_change_mask


def _binary_counts(prediction: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    prediction, truth = prediction.astype(bool), truth.astype(bool)
    return {
        "tp": int(np.count_nonzero(prediction & truth)),
        "fp": int(np.count_nonzero(prediction & ~truth)),
        "fn": int(np.count_nonzero(~prediction & truth)),
        "tn": int(np.count_nonzero(~prediction & ~truth)),
    }


def _metric_row(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        **counts,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "pixel_accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def _finish_transition_metrics(counts: np.ndarray) -> dict[str, float | int]:
    # Categories 0..41 are changed transitions; 42 means the model predicted no transition.
    tp_values, fp_values, fn_values = [], [], []
    for transition in range(42):
        tp = int(counts[transition, transition])
        fp = int(counts[42, transition] + sum(counts[other, transition] for other in range(42) if other != transition))
        fn = int(counts[transition, 42] + sum(counts[transition, other] for other in range(42) if other != transition))
        if tp + fp + fn:
            tp_values.append(tp)
            fp_values.append(fp)
            fn_values.append(fn)
    tp, fp, fn = sum(tp_values), sum(fp_values), sum(fn_values)
    macro_f1 = np.mean([
        2 * a / max(2 * a + b + c, 1)
        for a, b, c in zip(tp_values, fp_values, fn_values)
    ]) if tp_values else 0.0
    return {
        "micro_transition_precision": tp / max(tp + fp, 1),
        "micro_transition_recall": tp / max(tp + fn, 1),
        "micro_transition_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "macro_transition_f1_present_classes": float(macro_f1),
        "exact_transition_pixels": tp,
        "predicted_or_expected_transition_pixels": int(tp + fp + fn),
    }


def evaluate(args: argparse.Namespace) -> dict:
    adapter = DynamicEarthNetAdapter(args.archive, decoded_cache_dir=args.cache_dir)
    semantic_model = MambaChangeDetector(args.checkpoint, device=args.device)
    binary_model = PixelDiffChangeDetector(threshold=args.pixel_threshold)
    binary_totals = {"semantic_mamba": {k: 0 for k in ("tp", "fp", "fn", "tn")},
                     "pixel_diff": {k: 0 for k in ("tp", "fp", "fn", "tn")}}
    transition_counts = np.zeros((43, 43), dtype=np.int64)
    threshold_rows = []
    pairs = []
    try:
        for location in args.locations:
            for date_before, date_after in args.date_pairs:
                pair_start = {key: {k: 0 for k in ("tp", "fp", "fn", "tn")} for key in binary_totals}
                tile_count = 0
                for row in range(4):
                    for col in range(4):
                        tile_id = f"{location}_{row:04d}_{col:04d}_planet"
                        before = adapter.load_tile_array(location, date_before, tile_id)
                        after = adapter.load_tile_array(location, date_after, tile_id)
                        before_labels = adapter.load_semantic_tile(location, date_before, tile_id)
                        after_labels = adapter.load_semantic_tile(location, date_after, tile_id)
                        truth_classes = before_labels.astype(np.uint8) * 7 + after_labels.astype(np.uint8)
                        truth_change = before_labels != after_labels

                        semantic_output = semantic_model.detect_sequence([before, after])
                        binary_probability = np.asarray(
                            semantic_output.metadata["binary_probability_map"], dtype=np.float32
                        )
                        semantic_binary = refine_change_mask(semantic_output.mask, min_area_px=args.min_area_px).astype(bool)
                        semantic_encoded = np.asarray(semantic_output.metadata["semantic_mask"], dtype=np.uint8).copy()
                        semantic_encoded[~semantic_binary] = 0
                        pixel_binary = refine_change_mask(binary_model.detect(before, after).mask, min_area_px=args.min_area_px).astype(bool)
                        for name, prediction in (("semantic_mamba", semantic_binary), ("pixel_diff", pixel_binary)):
                            counts = _binary_counts(prediction, truth_change)
                            for key, value in counts.items():
                                binary_totals[name][key] += value
                                pair_start[name][key] += value

                        predicted_ids = (semantic_encoded.reshape(-1).astype(np.int16) - 1)
                        predicted_ids[semantic_encoded.reshape(-1) == 0] = 42
                        truth_ids = truth_classes.reshape(-1)
                        changed_code_map = np.full(49, 42, dtype=np.int64)
                        changed_codes = [code for code in range(49) if code // 7 != code % 7]
                        changed_code_map[changed_codes] = np.arange(42)
                        np.add.at(
                            transition_counts,
                            (changed_code_map[truth_ids], changed_code_map[predicted_ids]),
                            1,
                        )
                        tile_count += 1
                        if date_before == args.date_pairs[0][0] and date_after == args.date_pairs[0][1]:
                            threshold_rows.append((binary_probability.reshape(-1), truth_change.reshape(-1)))
                pairs.append({
                    "location": location,
                    "date_before": date_before,
                    "date_after": date_after,
                    "tile_count": tile_count,
                    "semantic_mamba_binary": _metric_row(pair_start["semantic_mamba"]),
                    "pixel_diff_binary": _metric_row(pair_start["pixel_diff"]),
                })
    finally:
        adapter.close()

    threshold_sweep = []
    if threshold_rows:
        all_scores = np.concatenate([row[0] for row in threshold_rows])
        all_truth = np.concatenate([row[1] for row in threshold_rows])
        for threshold in np.unique(np.concatenate((np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99]))):
            threshold_mask = (all_scores >= threshold).reshape(-1)
            threshold_sweep.append({
                "threshold": float(threshold),
                **_metric_row(_binary_counts(threshold_mask, all_truth)),
            })
    return {
        "dataset": "DynamicEarthNet-video PlanetFusion imagery and monthly land-cover labels",
        "evaluation_note": "AOIs must be held out from model training. The local split CSV contains training AOIs only; these validation AOIs are not claimed to be the official test split.",
        "checkpoint": str(args.checkpoint),
        "pixel_diff_threshold": args.pixel_threshold,
        "semantic_mamba_binary": _metric_row(binary_totals["semantic_mamba"]),
        "pixel_diff_binary": _metric_row(binary_totals["pixel_diff"]),
        "semantic_transition_metrics": _finish_transition_metrics(transition_counts),
        "threshold_sweep_first_pair": threshold_sweep,
        "pairs": pairs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="./data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip")
    parser.add_argument("--cache-dir", default="./data/dynamicearthnet/decoded_frames")
    parser.add_argument("--checkpoint", default="./models/change_detection_candidates/mamba_dynamicearthnet_semantic.pt")
    parser.add_argument("--locations", nargs="+", default=["den1487_3335_13"])
    parser.add_argument("--date-pairs", nargs="+", default=["2018-01-01:2018-02-01", "2019-11-01:2019-12-01"])
    parser.add_argument("--pixel-threshold", type=float, default=0.3)
    parser.add_argument("--min-area-px", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="./data/evaluation/dynamicearthnet_semantic_cd_validation.json")
    args = parser.parse_args()
    args.date_pairs = [tuple(value.split(":")) for value in args.date_pairs]
    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "semantic_mamba_binary", "pixel_diff_binary", "semantic_transition_metrics"
    )}, indent=2))


if __name__ == "__main__":
    main()
