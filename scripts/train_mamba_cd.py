"""Train the temporal selective-SSM change model on OSCD train cities only."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "backend", ROOT / "data-handling"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from adapters.oscd.oscd_loader import OSCDLoader
from app.models.change_detection.mamba_model import MambaTemporalChangeNet
from scripts.evaluate import compute_metrics


PATCH_SIZE = 256
SEED = 42


def read_split(oscd_root: Path, split: str) -> list[str]:
    split_path = oscd_root / f"{split}.txt"
    if not split_path.is_file():
        raise FileNotFoundError(f"OSCD split file not found: {split_path}")
    return [
        item.strip().lower()
        for item in split_path.read_text(encoding="utf-8").replace("\n", "").split(",")
        if item.strip()
    ]


def split_train_val(cities: list[str], validation_count: int) -> tuple[list[str], list[str]]:
    if not 1 <= validation_count < len(cities):
        raise ValueError("validation_count must be within 1..number_of_train_cities-1")
    validation = sorted(random.Random(SEED).sample(cities, validation_count))
    training = [city for city in cities if city not in validation]
    return training, validation


def load_scene(loader: OSCDLoader, city_dirs: dict[str, Path], city: str):
    if city not in city_dirs:
        raise FileNotFoundError(f"OSCD imagery directory missing for {city}")
    scene = city_dirs[city]
    dates = [
        loader.load_band_composite(scene, time_index=index)
        for index in (1, 2)
    ]
    mask = loader.load_ground_truth_mask(city)
    if mask is None:
        raise FileNotFoundError(f"OSCD binary change mask missing for {city}")
    if dates[0].shape != dates[1].shape or dates[0].shape[:2] != mask.shape:
        raise ValueError(
            f"{city}: image/mask shapes differ: {dates[0].shape}, {dates[1].shape}, {mask.shape}"
        )
    if any(not np.isfinite(image).all() for image in dates):
        raise ValueError(f"{city}: image contains non-finite pixel values")
    return dates, mask.astype(np.uint8)


def make_patches(dates: list[np.ndarray], mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    height, width = mask.shape
    patches = []
    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):
            crop_dates = [image[row:row + PATCH_SIZE, col:col + PATCH_SIZE] for image in dates]
            crop_mask = mask[row:row + PATCH_SIZE, col:col + PATCH_SIZE]
            tile_h, tile_w = crop_mask.shape
            pad_h, pad_w = PATCH_SIZE - tile_h, PATCH_SIZE - tile_w
            if pad_h or pad_w:
                crop_dates = [
                    np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
                    for image in crop_dates
                ]
                crop_mask = np.pad(
                    crop_mask,
                    ((0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=255,
                )
            patches.append((
                np.stack(crop_dates).astype(np.float32),
                crop_mask.astype(np.uint8),
                np.asarray([tile_h, tile_w], dtype=np.int32),
            ))
    return patches


class OSCDPatchDataset(Dataset):
    def __init__(self, patches: list, augment: bool):
        self.patches = patches
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence, mask, _ = self.patches[index]
        if self.augment:
            turns = random.randrange(4)
            sequence = np.rot90(sequence, turns, axes=(1, 2))
            mask = np.rot90(mask, turns)
            if random.random() < 0.5:
                sequence = sequence[:, :, ::-1]
                mask = mask[:, ::-1]
        image_tensor = torch.from_numpy(np.ascontiguousarray(sequence.transpose(0, 3, 1, 2)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask.astype(np.int64)))
        return image_tensor, mask_tensor


def masked_change_loss(logits: Tensor, target: Tensor, positive_weight: float) -> Tensor:
    valid = target != 255
    truth = (target == 1).to(dtype=logits.dtype)
    if not torch.any(valid):
        return logits.sum() * 0.0
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[valid],
        truth[valid],
        pos_weight=logits.new_tensor(positive_weight),
    )
    probabilities = torch.sigmoid(logits) * valid
    masked_truth = truth * valid
    dice = 1.0 - (2.0 * (probabilities * masked_truth).sum() + 1.0) / (
        probabilities.sum() + masked_truth.sum() + 1.0
    )
    return bce + 0.5 * dice


def evaluate_model(model, patches_by_city: dict[str, list], device) -> tuple[float, float, dict, dict]:
    model.eval()
    city_probabilities: dict[str, np.ndarray] = {}
    city_truth: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for city, patches in patches_by_city.items():
            probability_parts, truth_parts = [], []
            for sequence, mask, valid_shape in patches:
                image_tensor = torch.from_numpy(sequence.transpose(0, 3, 1, 2).copy())
                image_tensor = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32)
                probability = torch.sigmoid(model(image_tensor))[0, 0].cpu().numpy()
                tile_h, tile_w = valid_shape.tolist()
                probability_parts.append(probability[:tile_h, :tile_w].reshape(-1))
                truth_parts.append(mask[:tile_h, :tile_w].reshape(-1))
            city_probabilities[city] = np.concatenate(probability_parts)
            city_truth[city] = np.concatenate(truth_parts)

    thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    for threshold in thresholds:
        city_scores = [
            compute_metrics((city_probabilities[city] >= threshold).astype(np.uint8), city_truth[city])
            for city in city_truth
        ]
        rows.append((float(np.mean([row["f1"] for row in city_scores])), float(threshold), city_scores))
    best_f1, best_threshold, metrics_by_city = max(rows, key=lambda row: (row[0], -abs(row[1] - 0.5)))
    return best_f1, best_threshold, metrics_by_city, {
        city: {
            "f1": float(metrics["f1"]),
            "iou": float(metrics["iou"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }
        for city, metrics in zip(city_truth, metrics_by_city)
    }


def train(args) -> dict:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    oscd_root, labels_root = Path(args.oscd_dir), Path(args.labels_dir)
    loader = OSCDLoader(oscd_root, labels_root)
    city_dirs = loader.find_city_directories()
    official_train_cities = read_split(oscd_root, "train")
    train_cities, validation_cities = split_train_val(official_train_cities, args.validation_cities)
    train_patches_by_city, validation_patches_by_city = {}, {}
    for city in train_cities:
        train_patches_by_city[city] = make_patches(*load_scene(loader, city_dirs, city))
    for city in validation_cities:
        validation_patches_by_city[city] = make_patches(*load_scene(loader, city_dirs, city))

    training_patches = [patch for patches in train_patches_by_city.values() for patch in patches]
    training_dataset = OSCDPatchDataset(training_patches, augment=True)
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    positive_count = sum(int((patch[1] == 1).sum()) for patch in training_patches)
    valid_count = sum(int((patch[1] != 255).sum()) for patch in training_patches)
    negative_count = max(valid_count - positive_count, 1)
    positive_weight = float(np.clip(np.sqrt(negative_count / max(positive_count, 1)), 1.0, 12.0))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_config = {"in_channels": 3, "feature_dim": args.feature_dim, "state_dim": args.state_dim}
    model = MambaTemporalChangeNet(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_validation_f1 = -1.0
    best_checkpoint = Path(args.output_checkpoint)
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for images, labels in train_loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)[:, 0]
            loss = masked_change_loss(logits, labels, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_f1, threshold, _val_rows, validation_metrics = evaluate_model(
            model, validation_patches_by_city, device
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)),
            "validation_macro_f1": val_f1,
            "validation_threshold": threshold,
            "validation_cities": validation_metrics,
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={epoch_row['train_loss']:.4f} "
            f"val_macro_f1={val_f1:.4f} threshold={threshold:.2f}"
        )
        if val_f1 > best_validation_f1:
            best_validation_f1 = val_f1
            torch.save(
                {
                    "model_name": "mamba_temporal_ssm_oscd_binary",
                    "model_config": model_config,
                    "model_state_dict": model.state_dict(),
                    "threshold": threshold,
                    "validation_macro_f1": val_f1,
                    "train_cities": train_cities,
                    "validation_cities": validation_cities,
                    "input_bands": ["B04", "B03", "B02"],
                    "normalization": "OSCDLoader per-band percentile normalization to [0,1]",
                    "seed": SEED,
                },
                best_checkpoint,
            )

    report = {
        "model": "mamba_temporal_ssm_oscd_binary",
        "dataset": "OSCD",
        "task": "binary change segmentation",
        "device": str(device),
        "official_train_cities": official_train_cities,
        "fit_cities": train_cities,
        "validation_cities": validation_cities,
        "test_cities_used": False,
        "patch_size": PATCH_SIZE,
        "model_config": model_config,
        "training_patch_count": len(training_patches),
        "positive_weight": positive_weight,
        "best_validation_macro_f1": best_validation_f1,
        "checkpoint": str(best_checkpoint),
        "history": history,
        "label_limit": "OSCD labels only stable/change pixels; semantic land-cover transition classes are unavailable.",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Training report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--labels-dir", default="./data/oscd_labels")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--validation-cities", type=int, default=3)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default=None, help="Defaults to CUDA when available, otherwise CPU")
    parser.add_argument("--output-checkpoint", default="./models/change_detection_candidates/mamba_oscd_best.pt")
    parser.add_argument("--report", default="./data/evaluation/mamba_oscd_training.json")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
