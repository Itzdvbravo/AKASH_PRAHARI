"""Evaluation harness for semantic retrieval and change detection benchmarks."""
import argparse
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for p in (root_dir, backend_dir, data_handling_dir):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from adapters.oscd.oscd_adapter import OSCDAdapter


def compute_iou_f1(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Computes Intersection over Union and F1-Score."""
    pred = pred_mask > 0
    gt = gt_mask > 0

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(intersection / max(union, 1))

    precision = float(intersection / max(pred.sum(), 1))
    recall = float(intersection / max(gt.sum(), 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-6))

    return iou, f1, precision, recall


def run_evaluation(oscd_dir: str):
    print("=" * 65)
    print("TerraEyes — Benchmark Evaluation on OSCD Dataset")
    print("=" * 65)

    adapter = OSCDAdapter(oscd_dir)
    detector = PixelDiffChangeDetector()
    cities = adapter.list_locations()

    ious = []
    f1s = []

    print(f"\nEvaluating baseline detector on {len(cities)} cities...")
    for city in cities:
        gt = adapter.load_ground_truth_mask(city, "", "")
        if gt is None:
            continue

        dates = adapter.list_dates(city)
        if len(dates) < 2:
            continue

        try:
            t1 = adapter.load_tile_array(city, dates[0], f"{city}_0000_0000_sentinel-2")
            t2 = adapter.load_tile_array(city, dates[1], f"{city}_0000_0000_sentinel-2")
            # Resize gt if needed
            if gt.shape != t1.shape[:2]:
                from PIL import Image
                gt = np.array(Image.fromarray(gt).resize((t1.shape[1], t1.shape[0]), Image.NEAREST))

            pred = detector.detect(t1, t2)
            iou, f1, prec, rec = compute_iou_f1(pred.mask, gt)
            ious.append(iou)
            f1s.append(f1)
            print(f"  {city:<15} | IoU: {iou:.3f} | F1: {f1:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f}")
        except Exception as e:
            print(f"  {city:<15} | [Skip: {e}]")

    if ious:
        print("\n" + "-" * 50)
        print(f"Mean IoU: {np.mean(ious):.4f}")
        print(f"Mean F1:  {np.mean(f1s):.4f}")
        print("-" * 50)
    else:
        print("\nNote: Evaluation requires downloaded OSCD dataset with ground-truth masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TerraEyes model evaluation")
    parser.add_argument("--oscd-dir", type=str, default="./data/oscd")
    args = parser.parse_args()
    run_evaluation(args.oscd_dir)
