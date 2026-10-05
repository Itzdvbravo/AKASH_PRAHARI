"""Train temporal Mamba semantic-transition segmentation on DynamicEarthNet.

The model predicts one of 49 ordered land-cover pairs at each pixel. A same
class pair is unchanged; the other 42 classes represent semantic transitions.
AOIs are split before sampling so validation never shares an area with training.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for import_path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from adapters.dynamicearthnet.adapter import DynamicEarthNetAdapter
from app.models.change_detection.mamba_model import MambaTemporalChangeNet


SEED = 20261005
TILE_LOCATIONS = ((0, 0), (0, 3), (3, 0), (3, 3), (1, 1), (1, 2), (2, 1), (2, 2),
                  (0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2))
CLASS_COUNT = 7
TRANSITION_COUNT = CLASS_COUNT * CLASS_COUNT


def training_area_ids(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return sorted({
            Path(row["planet_path"].replace("\\", "/")).parts[-2]
            for row in rows
            if row.get("split") == "train" and row.get("planet_path")
        })


def split_area_ids(area_ids: list[str], validation_fraction: float) -> tuple[list[str], list[str]]:
    ordered = list(area_ids)
    random.Random(SEED).shuffle(ordered)
    validation_count = max(1, round(len(ordered) * validation_fraction))
    validation = sorted(ordered[:validation_count])
    training = sorted(ordered[validation_count:])
    if not training:
        raise ValueError("DynamicEarthNet train split needs at least two distinct AOIs")
    return training, validation


def sample_pair_indices(date_count: int, requested_count: int) -> list[int]:
    possible = date_count - 1
    if possible <= 0:
        return []
    count = min(possible, requested_count)
    return sorted(set(np.linspace(0, possible - 1, count, dtype=np.int32).tolist()))


class TransitionPatchDataset(Dataset):
    def __init__(self, samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        before, after, target = self.samples[index]
        images = np.stack((before, after)).astype(np.float32)
        labels = target.astype(np.int64)
        if self.augment:
            turns = random.randrange(4)
            images = np.rot90(images, turns, axes=(1, 2))
            labels = np.rot90(labels, turns)
            if random.random() < 0.5:
                images = images[:, :, ::-1]
                labels = labels[:, ::-1]
        image_tensor = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        return image_tensor, torch.from_numpy(np.ascontiguousarray(labels))


def make_samples(
    adapter: DynamicEarthNetAdapter,
    area_ids: list[str],
    pairs_per_area: int,
    tiles_per_pair: int,
    max_areas: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    selected_areas = area_ids[:max_areas] if max_areas > 0 else area_ids
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for area_number, area_id in enumerate(selected_areas, start=1):
        location = f"den{area_id}"
        dates = adapter.list_dates(location)
        pair_indices = sample_pair_indices(len(dates), pairs_per_area)
        tile_locations = TILE_LOCATIONS[:max(1, min(tiles_per_pair, len(TILE_LOCATIONS)))]
        for month_index in pair_indices:
            before_date, after_date = dates[month_index], dates[month_index + 1]
            for row, col in tile_locations:
                tile_id = f"{location}_{row}_{col}_planet"
                before = adapter.load_tile_array(location, before_date, tile_id)
                after = adapter.load_tile_array(location, after_date, tile_id)
                before_labels = adapter.load_semantic_tile(location, before_date, tile_id)
                after_labels = adapter.load_semantic_tile(location, after_date, tile_id)
                if any(value.shape[:2] != before.shape[:2] for value in (after, before_labels, after_labels)):
                    raise ValueError(f"DynamicEarthNet pair shape mismatch at {tile_id}/{before_date}")
                target = before_labels.astype(np.int64) * CLASS_COUNT + after_labels.astype(np.int64)
                samples.append((before, after, target.astype(np.uint8)))
        print(f"prepared AOI {area_id} ({area_number}/{len(selected_areas)}); patches={len(samples)}", flush=True)
    return samples


def metrics_from_counts(counts: np.ndarray, binary_counts: np.ndarray, threshold: float) -> dict[str, float | int]:
    tp = np.diag(counts).astype(np.float64)
    truth_count = counts.sum(axis=1).astype(np.float64)
    predicted_count = counts.sum(axis=0).astype(np.float64)
    union = truth_count + predicted_count - tp
    class_f1 = np.divide(2 * tp, truth_count + predicted_count, out=np.zeros_like(tp), where=(truth_count + predicted_count) > 0)
    class_iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    changed = np.asarray([i // CLASS_COUNT != i % CLASS_COUNT for i in range(TRANSITION_COUNT)])
    btp, bfp = int(binary_counts[1, 1]), int(binary_counts[0, 1])
    bfn, btn = int(binary_counts[1, 0]), int(binary_counts[0, 0])
    present_changed = [i for i in range(TRANSITION_COUNT) if changed[i] and truth_count[i] > 0]
    return {
        "pixel_accuracy": float(tp.sum() / max(counts.sum(), 1)),
        "transition_macro_f1": float(np.mean(class_f1[truth_count > 0])) if np.any(truth_count > 0) else 0.0,
        "transition_macro_iou": float(np.mean(class_iou[truth_count > 0])) if np.any(truth_count > 0) else 0.0,
        "changed_transition_macro_f1": float(np.mean(class_f1[present_changed])) if present_changed else 0.0,
        "binary_precision": float(btp / max(btp + bfp, 1)),
        "binary_recall": float(btp / max(btp + bfn, 1)),
        "binary_f1": float(2 * btp / max(2 * btp + bfp + bfn, 1)),
        "binary_iou": float(btp / max(btp + bfp + bfn, 1)),
        "changed_pixels": int(btp + bfn),
        "pixels": int(counts.sum()),
        "binary_threshold": float(threshold),
    }


def evaluate(model: MambaTemporalChangeNet, dataset: Dataset, device: torch.device) -> tuple[dict, float]:
    model.eval()
    counts = np.zeros((TRANSITION_COUNT, TRANSITION_COUNT), dtype=np.int64)
    binary_scores: list[np.ndarray] = []
    binary_truth: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
            logits = model(images.to(device=device, dtype=torch.float32))
            predicted_before = logits[:, 1:1 + CLASS_COUNT].argmax(dim=1)[0].cpu().numpy().reshape(-1)
            predicted_after = logits[:, 1 + CLASS_COUNT:1 + 2 * CLASS_COUNT].argmax(dim=1)[0].cpu().numpy().reshape(-1)
            prediction = predicted_before * CLASS_COUNT + predicted_after
            truth = labels[0].numpy().reshape(-1)
            counts += np.bincount(truth * TRANSITION_COUNT + prediction, minlength=TRANSITION_COUNT ** 2).reshape(TRANSITION_COUNT, TRANSITION_COUNT)
            binary_scores.append(torch.sigmoid(logits[:, 0])[0].cpu().numpy().reshape(-1))
            binary_truth.append(np.asarray([value // 7 != value % 7 for value in truth], dtype=np.uint8))
    scores = np.concatenate(binary_scores)
    truths = np.concatenate(binary_truth)
    candidates = np.unique(np.concatenate((np.linspace(0.05, 0.95, 19), np.asarray([0.97, 0.98, 0.99]))))
    best_threshold, best_binary_f1 = 0.5, -1.0
    best_binary_counts = np.zeros((2, 2), dtype=np.int64)
    for threshold in candidates:
        binary_prediction = scores >= threshold
        counts_at_threshold = np.zeros((2, 2), dtype=np.int64)
        np.add.at(counts_at_threshold, (truths.astype(np.int64), binary_prediction.astype(np.int64)), 1)
        tp = int(counts_at_threshold[1, 1])
        fp = int(counts_at_threshold[0, 1])
        fn = int(counts_at_threshold[1, 0])
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_binary_f1:
            best_threshold, best_binary_f1 = float(threshold), float(f1)
            best_binary_counts = counts_at_threshold
    return metrics_from_counts(counts, best_binary_counts, best_threshold), best_threshold


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    available_areas = training_area_ids(Path(args.splits))
    explicit_validation = [value.strip() for value in args.validation_area_ids.split(",") if value.strip()]
    if explicit_validation:
        unknown = set(explicit_validation) - set(available_areas)
        if unknown:
            raise ValueError(f"Validation AOIs are not in the DynamicEarthNet train split: {sorted(unknown)}")
        validation_areas = sorted(set(explicit_validation))
        train_areas = sorted(set(available_areas) - set(validation_areas))
    else:
        train_areas, validation_areas = split_area_ids(available_areas, args.validation_fraction)
    if args.max_train_areas > 0:
        train_areas = train_areas[:args.max_train_areas]
    if args.max_validation_areas > 0:
        validation_areas = validation_areas[:args.max_validation_areas]

    adapter = DynamicEarthNetAdapter(args.archive, decoded_cache_dir=args.cache_dir)
    try:
        train_samples = make_samples(adapter, train_areas, args.pairs_per_area, args.tiles_per_pair, 0)
        validation_samples = make_samples(adapter, validation_areas, args.validation_pairs_per_area, args.validation_tiles_per_pair, 0)
    finally:
        adapter.close()
    if not train_samples or not validation_samples:
        raise ValueError("No training or validation patches were available")

    train_dataset = TransitionPatchDataset(train_samples, augment=True)
    validation_dataset = TransitionPatchDataset(validation_samples, augment=False)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {
        "in_channels": 3,
        "feature_dim": args.feature_dim,
        "state_dim": args.state_dim,
        "transition_classes": TRANSITION_COUNT,
        "semantic_classes": CLASS_COUNT,
    }
    model = MambaTemporalChangeNet(**model_config).to(device)
    target_counts = np.zeros(TRANSITION_COUNT, dtype=np.int64)
    for _, _, target in train_samples:
        target_counts += np.bincount(target.reshape(-1), minlength=TRANSITION_COUNT)
    changed_ids = np.asarray([i for i in range(TRANSITION_COUNT) if i // 7 != i % 7])
    changed_pixels = int(target_counts[changed_ids].sum())
    stable_pixels = int(target_counts.sum() - changed_pixels)
    binary_pos_weight = float(np.clip(np.sqrt(stable_pixels / max(changed_pixels, 1)), 1.0, 20.0))
    before_counts = np.bincount(np.arange(TRANSITION_COUNT) // 7, weights=target_counts, minlength=CLASS_COUNT)
    after_counts = np.bincount(np.arange(TRANSITION_COUNT) % 7, weights=target_counts, minlength=CLASS_COUNT)

    def class_weights(counts: np.ndarray) -> torch.Tensor:
        weights = np.ones(CLASS_COUNT, dtype=np.float32)
        present = counts > 0
        weights[present] = np.sqrt(counts[present].sum() / counts[present])
        return torch.tensor(np.clip(weights, 1.0, 8.0), device=device)

    before_weights = class_weights(before_counts)
    after_weights = class_weights(after_counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_metrics: dict[str, float | int] = {}
    history = []
    start_epoch = 1

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.is_file():
            try:
                checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:  # PyTorch versions before the weights_only argument
                checkpoint = torch.load(resume_path, map_location=device)
            checkpoint_config = checkpoint.get("model_config", {})
            if checkpoint_config != model_config:
                raise ValueError(
                    f"Cannot resume checkpoint with model config {checkpoint_config}; expected {model_config}"
                )
            previous_train_areas = checkpoint.get("train_area_ids")
            previous_validation_areas = checkpoint.get("validation_area_ids")
            if previous_train_areas != train_areas or previous_validation_areas != validation_areas:
                raise ValueError("Cannot resume: checkpoint AOI split differs from the current training split")
            model.load_state_dict(checkpoint["model_state_dict"])
            if checkpoint.get("optimizer_state_dict"):
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_score = float(checkpoint.get("validation_score", -1.0))
            best_metrics = checkpoint.get("validation_metrics", {})
            history = checkpoint.get("history", [])
            print(f"resuming from {resume_path} at epoch {start_epoch}", flush=True)
            if start_epoch > args.epochs:
                report = {
                    "model": "mamba_temporal_ssm_dynamicearthnet_semantic",
                    "dataset": "DynamicEarthNet-video 71-PSNR TACO distribution",
                    "task": "49-class ordered land-cover transition segmentation; same-class transitions are unchanged",
                    "device": str(device),
                    "official_split_note": "The local splits.csv contains training AOIs only; validation AOIs were held out by area from that training set and are not claimed to be an official test split.",
                    "training_area_ids": train_areas,
                    "validation_area_ids": validation_areas,
                    "training_patch_count": len(train_samples),
                    "validation_patch_count": len(validation_samples),
                    "model_config": model_config,
                    "best_validation_selection_score": best_score,
                    "validation_metrics": best_metrics,
                    "history": history,
                    "checkpoint": str(checkpoint_path),
                    "limitations": "This is a prototype trained only on the selected AOIs and sampled monthly pairs; metrics do not establish performance on an official DynamicEarthNet test split.",
                }
                report_path = Path(args.report)
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"checkpoint={checkpoint_path}\nreport={report_path}\nmetrics={json.dumps(best_metrics)}", flush=True)
                return report

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            before_loss = F.cross_entropy(
                logits[:, 1:1 + CLASS_COUNT], labels // CLASS_COUNT, weight=before_weights
            )
            after_loss = F.cross_entropy(
                logits[:, 1 + CLASS_COUNT:1 + 2 * CLASS_COUNT], labels % CLASS_COUNT, weight=after_weights
            )
            binary_target = (labels // 7 != labels % 7).to(dtype=logits.dtype)
            binary_loss = F.binary_cross_entropy_with_logits(
                logits[:, 0], binary_target,
                pos_weight=logits.new_tensor(binary_pos_weight),
            )
            loss = before_loss + after_loss + 0.5 * binary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, threshold = evaluate(model, validation_dataset, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={row['train_loss']:.4f} "
            f"val_binary_f1={val_metrics['binary_f1']:.4f} "
            f"val_transition_macro_f1={val_metrics['transition_macro_f1']:.4f}", flush=True,
        )
        # Prioritize semantic-transition quality while retaining binary CD quality.
        score = 0.5 * val_metrics["binary_f1"] + 0.5 * val_metrics["changed_transition_macro_f1"]
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            torch.save({
                "model_name": "mamba_temporal_ssm_dynamicearthnet_semantic",
                "model_config": model_config,
                "model_state_dict": model.state_dict(),
                "threshold": threshold,
                "validation_score": score,
                "validation_metrics": val_metrics,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "training_dataset": "DynamicEarthNet-video PlanetFusion monthly imagery and LULC labels",
                "train_area_ids": train_areas,
                "validation_area_ids": validation_areas,
                "transition_encoding": "class_from * 7 + class_to; same-class IDs are unchanged",
                "normalization": "fixed PlanetFusion digital-number scale 10000",
                "seed": SEED,
            }, checkpoint_path)

    report = {
        "model": "mamba_temporal_ssm_dynamicearthnet_semantic",
        "dataset": "DynamicEarthNet-video 71-PSNR TACO distribution",
        "task": "49-class ordered land-cover transition segmentation; same-class transitions are unchanged",
        "device": str(device),
        "official_split_note": "The local splits.csv contains training AOIs only; validation AOIs were held out by area from that training set and are not claimed to be an official test split.",
        "training_area_ids": train_areas,
        "validation_area_ids": validation_areas,
        "training_patch_count": len(train_samples),
        "validation_patch_count": len(validation_samples),
        "model_config": model_config,
        "binary_positive_loss_weight": binary_pos_weight,
        "best_validation_selection_score": best_score,
        "validation_metrics": best_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "limitations": "This is a prototype trained only on the selected AOIs and sampled monthly pairs; metrics do not establish performance on an official DynamicEarthNet test split.",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"checkpoint={checkpoint_path}\nreport={report_path}\nmetrics={json.dumps(best_metrics)}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="./data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip")
    parser.add_argument("--splits", default="./data/dynamicearthnet/splits.csv")
    parser.add_argument("--cache-dir", default="./data/dynamicearthnet/decoded_frames")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-area-ids", default="", help="Optional comma-separated AOIs held out from training")
    parser.add_argument("--max-train-areas", type=int, default=0, help="0 uses all training AOIs")
    parser.add_argument("--max-validation-areas", type=int, default=0, help="0 uses all held-out AOIs")
    parser.add_argument("--pairs-per-area", type=int, default=8)
    parser.add_argument("--validation-pairs-per-area", type=int, default=8)
    parser.add_argument("--tiles-per-pair", type=int, default=8)
    parser.add_argument("--validation-tiles-per-pair", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--resume-from", default="", help="Resume from a compatible checkpoint, if present")
    parser.add_argument("--output-checkpoint", default="./models/change_detection_candidates/mamba_dynamicearthnet_semantic.pt")
    parser.add_argument("--report", default="./data/evaluation/dynamicearthnet_semantic_training.json")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
