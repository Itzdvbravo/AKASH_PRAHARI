"""Benchmark downloaded BIT-CD / ChangeFormer checkpoints on held-out OSCD cities."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for directory in (root_dir, backend_dir, data_handling_dir):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from adapters.oscd.oscd_loader import OSCDLoader
from scripts.evaluate import compute_metrics


TILE_SIZE = 256


def load_candidate(model_name: str, checkpoint_path: Path, device):
    import torch

    candidate_root = root_dir / "models" / "change_detection_candidates"
    if model_name == "bit_cd":
        sys.path.insert(0, str(candidate_root / "source" / "BIT_CD"))
        from models.networks import BASE_Transformer

        model = BASE_Transformer(
            input_nc=3,
            output_nc=2,
            token_len=4,
            resnet_stages_num=4,
            with_pos="learned",
            enc_depth=1,
            dec_depth=8,
            decoder_dim_head=8,
        )
    elif model_name == "changeformer_v6":
        sys.path.insert(0, str(candidate_root / "source" / "ChangeFormer"))
        from models.ChangeFormer import ChangeFormerV6

        model = ChangeFormerV6(embed_dim=256)
    else:
        raise ValueError(f"Unsupported candidate model: {model_name}")

    # These official research checkpoints contain optimizer state and legacy
    # NumPy scalars, so weights_only=False is needed for full checkpoint loading.
    # Only load these files from the official projects' published links.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_G_state_dict")
    if state is None:
        raise ValueError(f"No model_G_state_dict in checkpoint: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def _tile_tensor(array: np.ndarray, row: int, col: int, device):
    import torch

    crop = array[row:row + TILE_SIZE, col:col + TILE_SIZE]
    pad_h = TILE_SIZE - crop.shape[0]
    pad_w = TILE_SIZE - crop.shape[1]
    if pad_h or pad_w:
        crop = np.pad(crop, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    # Official BIT-CD and ChangeFormer preprocessing maps uint8 RGB / 255
    # inputs from [0, 1] to [-1, 1] using mean=std=0.5.
    tensor = torch.from_numpy(crop.transpose(2, 0, 1).copy()).unsqueeze(0)
    tensor = tensor.mul(2.0).sub(1.0)
    return tensor.to(device=device, dtype=torch.float32)


def predict_scene(model_name: str, model, before: np.ndarray, after: np.ndarray, device):
    import torch

    height, width = before.shape[:2]
    prediction = np.zeros((height, width), dtype=np.uint8)
    elapsed = 0.0
    tile_count = 0
    with torch.inference_mode():
        for row in range(0, height, TILE_SIZE):
            for col in range(0, width, TILE_SIZE):
                first = _tile_tensor(before, row, col, device)
                second = _tile_tensor(after, row, col, device)
                started = time.perf_counter()
                output = model(first, second)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if output.shape[-2:] != (TILE_SIZE, TILE_SIZE):
                    output = torch.nn.functional.interpolate(
                        output, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False
                    )
                tile_prediction = output.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed += time.perf_counter() - started
                tile_h = min(TILE_SIZE, height - row)
                tile_w = min(TILE_SIZE, width - col)
                prediction[row:row + tile_h, col:col + tile_w] = tile_prediction[:tile_h, :tile_w]
                tile_count += 1
    return prediction, elapsed, tile_count


def run_evaluation(
    model_name: str,
    checkpoint_path: str,
    oscd_dir: str = "./images",
    labels_dir: str = "./data/oscd_labels",
    split: str = "test",
    output_path: str | None = None,
) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model = load_candidate(model_name, checkpoint, device)
    oscd_root = Path(oscd_dir)
    labels_root = Path(labels_dir)
    loader = OSCDLoader(oscd_root, labels_root)
    cities = [
        city.strip().lower()
        for city in (oscd_root / f"{split}.txt").read_text(encoding="utf-8").replace("\n", "").split(",")
        if city.strip()
    ]
    city_dirs = loader.find_city_directories()
    rows = []

    # Warm up the selected network and exclude initialization/first-call effects
    # from reported latency.
    zeros = torch.zeros((1, 3, TILE_SIZE, TILE_SIZE), device=device)
    with torch.inference_mode():
        model(zeros, zeros)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for city in cities:
        scene = city_dirs[city]
        before = loader.load_band_composite(scene, time_index=1)
        after = loader.load_band_composite(scene, time_index=2)
        gt = loader.load_ground_truth_mask(city)
        if gt is None:
            raise FileNotFoundError(f"No ground-truth mask available for {city}")
        if before.shape[:2] != after.shape[:2] or before.shape[:2] != gt.shape:
            raise ValueError(f"{city}: imagery/mask shape mismatch; refusing to resize labels")
        pred, elapsed, tile_count = predict_scene(model_name, model, before, after, device)
        metrics = compute_metrics(pred, gt)
        row = {
            "city": city,
            **metrics,
            "inference_seconds": elapsed,
            "tile_count": tile_count,
            "seconds_per_tile": elapsed / max(tile_count, 1),
        }
        rows.append(row)
        print(
            f"{city:<15} F1={metrics['f1']:.4f} IoU={metrics['iou']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"inference={elapsed:.2f}s/{tile_count} tiles"
        )

    macro = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ("precision", "recall", "f1", "iou")
    }
    total_tp = sum(row["tp"] for row in rows)
    total_fp = sum(row["fp"] for row in rows)
    total_fn = sum(row["fn"] for row in rows)
    micro_precision = total_tp / max(total_tp + total_fp, 1)
    micro_recall = total_tp / max(total_tp + total_fn, 1)
    micro = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": 2 * total_tp / max(2 * total_tp + total_fp + total_fn, 1),
        "iou": total_tp / max(total_tp + total_fp + total_fn, 1),
    }
    report = {
        "dataset": "OSCD",
        "split": split,
        "model": model_name,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "input_bands": ["B04", "B03", "B02"],
        "input_normalization": "OSCDLoader per-band percentile normalization to [0,1]",
        "patch_size": TILE_SIZE,
        "padding": "edge replication on incomplete bottom/right patches; predictions cropped to scene extent",
        "city_count": len(rows),
        "macro_city_average": macro,
        "micro_pixel_average": micro,
        "mean_scene_inference_seconds": float(np.mean([row["inference_seconds"] for row in rows])),
        "mean_seconds_per_tile": float(np.mean([row["seconds_per_tile"] for row in rows])),
        "cities": rows,
        "limitation": "Checkpoints were pretrained on LEVIR-CD RGB data, not OSCD; this is transfer evaluation only, with no OSCD fine-tuning.",
    }
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": model_name,
        "device": str(device),
        "macro_city_average": macro,
        "micro_pixel_average": micro,
        "mean_scene_inference_seconds": report["mean_scene_inference_seconds"],
        "mean_seconds_per_tile": report["mean_seconds_per_tile"],
        "report": output_path,
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("bit_cd", "changeformer_v6"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--labels-dir", default="./data/oscd_labels")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output")
    args = parser.parse_args()
    run_evaluation(
        args.model,
        args.checkpoint,
        args.oscd_dir,
        args.labels_dir,
        args.split,
        args.output,
    )
