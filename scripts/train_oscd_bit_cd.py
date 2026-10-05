"""Fine-tune the official BIT-CD model on OSCD training cities only."""
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
candidate_root = root_dir / "models" / "change_detection_candidates"
bit_source = candidate_root / "source" / "BIT_CD"
for directory in (root_dir, backend_dir, data_handling_dir, bit_source):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from adapters.oscd.oscd_loader import OSCDLoader
from models.networks import BASE_Transformer


PATCH_SIZE = 256
SEED = 42


def read_cities(oscd_root: Path, split: str) -> list[str]:
    file_path = oscd_root / f"{split}.txt"
    if not file_path.is_file():
        raise FileNotFoundError(f"Split list not found: {file_path}")
    return [name.strip().lower() for name in file_path.read_text(encoding="utf-8").replace("\n", "").split(",") if name.strip()]


def split_training_cities(cities: list[str], validation_count: int) -> tuple[list[str], list[str]]:
    if not 1 <= validation_count < len(cities):
        raise ValueError("validation_count must be between 1 and the number of train cities - 1")
    validation = sorted(random.Random(SEED).sample(cities, validation_count))
    training = [city for city in cities if city not in validation]
    return training, validation


def city_patches(loader: OSCDLoader, city_dirs: dict[str, Path], city: str):
    scene_dir = city_dirs[city]
    before = loader.load_band_composite(scene_dir, time_index=1)
    after = loader.load_band_composite(scene_dir, time_index=2)
    mask = loader.load_ground_truth_mask(city)
    if mask is None:
        raise FileNotFoundError(f"No OSCD train mask for {city}")
    if before.shape != after.shape or before.shape[:2] != mask.shape:
        raise ValueError(f"{city}: imagery/mask shape mismatch: {before.shape}, {after.shape}, {mask.shape}")

    height, width = mask.shape
    patches = []
    for row in range(0, height, PATCH_SIZE):
        for col in range(0, width, PATCH_SIZE):
            tile_h = min(PATCH_SIZE, height - row)
            tile_w = min(PATCH_SIZE, width - col)
            first = before[row:row + tile_h, col:col + tile_w]
            second = after[row:row + tile_h, col:col + tile_w]
            label = mask[row:row + tile_h, col:col + tile_w]
            pad_h = PATCH_SIZE - tile_h
            pad_w = PATCH_SIZE - tile_w
            if pad_h or pad_w:
                padding = ((0, pad_h), (0, pad_w), (0, 0))
                first = np.pad(first, padding, mode="edge")
                second = np.pad(second, padding, mode="edge")
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=255)
            patches.append((
                np.asarray(first, dtype=np.float32),
                np.asarray(second, dtype=np.float32),
                np.asarray(label, dtype=np.uint8),
                tile_h,
                tile_w,
            ))
    return patches


class PatchDataset(Dataset):
    def __init__(self, patches: list, augment: bool):
        self.patches = patches
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        first, second, label, _, _ = self.patches[index]
        if self.augment:
            k = random.randrange(4)
            first = np.rot90(first, k)
            second = np.rot90(second, k)
            label = np.rot90(label, k)
            if random.random() < 0.5:
                first, second, label = first[:, ::-1], second[:, ::-1], label[:, ::-1]
        first = np.ascontiguousarray(first.transpose(2, 0, 1) * 2.0 - 1.0, dtype=np.float32)
        second = np.ascontiguousarray(second.transpose(2, 0, 1) * 2.0 - 1.0, dtype=np.float32)
        label = np.ascontiguousarray(label.astype(np.int64))
        return torch.from_numpy(first), torch.from_numpy(second), torch.from_numpy(label)


def segmentation_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    pred = prediction.astype(bool)
    gt = truth.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
    }


def evaluate_cities(model, patches_by_city: dict[str, list], device) -> tuple[float, dict]:
    model.eval()
    city_metrics = {}
    with torch.inference_mode():
        for city, patches in patches_by_city.items():
            truth_parts = []
            prediction_parts = []
            for first, second, mask, tile_h, tile_w in patches:
                first_t = torch.from_numpy(first.transpose(2, 0, 1).copy()).unsqueeze(0).float().mul(2).sub(1).to(device)
                second_t = torch.from_numpy(second.transpose(2, 0, 1).copy()).unsqueeze(0).float().mul(2).sub(1).to(device)
                output = model(first_t, second_t).argmax(dim=1)[0].cpu().numpy()
                truth_parts.append(mask[:tile_h, :tile_w].reshape(-1))
                prediction_parts.append(output[:tile_h, :tile_w].reshape(-1))
            metrics = segmentation_metrics(np.concatenate(prediction_parts), np.concatenate(truth_parts))
            city_metrics[city] = metrics
    macro_f1 = float(np.mean([item["f1"] for item in city_metrics.values()]))
    return macro_f1, city_metrics


def main(args):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oscd_root = Path(args.oscd_dir)
    labels_root = Path(args.labels_dir)
    loader = OSCDLoader(oscd_root, labels_root)
    city_dirs = loader.find_city_directories()
    train_cities = read_cities(oscd_root, "train")
    official_test_cities = set(read_cities(oscd_root, "test"))
    fit_cities, validation_cities = split_training_cities(train_cities, args.validation_cities)
    if official_test_cities.intersection(fit_cities + validation_cities):
        raise ValueError("Official test-city leakage detected")

    train_patches = [patch for city in fit_cities for patch in city_patches(loader, city_dirs, city)]
    validation_patches = {
        city: city_patches(loader, city_dirs, city)
        for city in validation_cities
    }
    if not train_patches:
        raise ValueError("No OSCD training patches found")

    positives = sum(int(np.count_nonzero(mask == 1)) for _, _, mask, _, _ in train_patches)
    valid_pixels = sum(int(np.count_nonzero(mask != 255)) for _, _, mask, _, _ in train_patches)
    negatives = valid_pixels - positives
    if positives == 0:
        raise ValueError("Training cities contain no positive change pixels")
    positive_weight = min(max((negatives / positives) ** 0.5, 1.0), args.max_positive_weight)
    class_weights = torch.tensor([1.0, positive_weight], device=device, dtype=torch.float32)

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
    pretrained = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(pretrained["model_G_state_dict"], strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    train_loader = DataLoader(
        PatchDataset(train_patches, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    checkpoint_path = Path(args.output_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    print(
        f"Training BIT-CD on {len(fit_cities)} OSCD cities / {len(train_patches)} patches; "
        f"validation cities: {', '.join(validation_cities)}; device={device}; "
        f"positive_weight={positive_weight:.3f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for first, second, labels in train_loader:
            first = first.to(device)
            second = second.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(first, second)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * first.shape[0]
            seen += first.shape[0]

        mean_loss = total_loss / max(seen, 1)
        val_f1, val_by_city = evaluate_cities(model, validation_patches, device)
        row = {"epoch": epoch, "train_loss": mean_loss, "validation_macro_f1": val_f1, "validation_cities": val_by_city}
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs} loss={mean_loss:.5f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_G_state_dict": model.state_dict(),
                "optimizer_G_state_dict": optimizer.state_dict(),
                "best_epoch_id": best_epoch,
                "best_val_f1": best_val_f1,
                "training_cities": fit_cities,
                "validation_cities": validation_cities,
                "training_config": {
                    "seed": SEED,
                    "patch_size": PATCH_SIZE,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "class_weights": [1.0, positive_weight],
                    "input_bands": ["B04", "B03", "B02"],
                    "normalization": "OSCD per-band percentile [0,1], then [-1,1]",
                },
            }, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping after {args.patience} epochs without validation improvement")
            break

    history_report = {
        "dataset": "OSCD",
        "training_split": "official train.txt only",
        "official_test_cities_touched": False,
        "training_cities": fit_cities,
        "validation_cities": validation_cities,
        "official_test_cities": sorted(official_test_cities),
        "patch_count": len(train_patches),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    history_path = Path(args.history_output)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history_report, indent=2), encoding="utf-8")
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "checkpoint": str(checkpoint_path),
        "history": str(history_path),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oscd-dir", default="./images")
    parser.add_argument("--labels-dir", default="./data/oscd_labels")
    parser.add_argument("--checkpoint", default="./models/change_detection_candidates/bit_levir_best_ckpt.pt")
    parser.add_argument("--output-checkpoint", default="./models/change_detection_candidates/bit_cd_oscd_finetuned.pt")
    parser.add_argument("--history-output", default="./data/evaluation/bit_cd_oscd_training.json")
    parser.add_argument("--validation-cities", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-positive-weight", type=float, default=10.0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    main(parser.parse_args())
