"""Verification script for OSCD dataset directory structure, bands, and ground-truth masks."""
import argparse
from pathlib import Path
import sys

# Ensure data-handling is in path
root_dir = Path(__file__).resolve().parents[1]
data_handling_dir = root_dir / "data-handling"
if str(data_handling_dir) not in sys.path:
    sys.path.insert(0, str(data_handling_dir))

from adapters.oscd.oscd_loader import OSCDLoader

EXPECTED_BANDS = {
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12",
}


def _image_directory(city_dir: Path, time_index: int) -> Path | None:
    for name in (f"imgs_{time_index}_rect", f"imgs_{time_index}"):
        candidate = city_dir / name
        if candidate.is_dir():
            return candidate
    return None


def _missing_bands(image_dir: Path | None) -> list[str]:
    if image_dir is None:
        return sorted(EXPECTED_BANDS)
    present = {
        path.stem.upper()
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    }
    return sorted(EXPECTED_BANDS - present)


def verify_oscd(oscd_dir: str, labels_dir: str | None = None) -> bool:
    oscd_path = Path(oscd_dir)
    print("=" * 60)
    print("TerraEyes — OSCD Dataset Verification")
    print("=" * 60)
    print(f"Checking directory: {oscd_path.resolve()}\n")

    if not oscd_path.exists():
        print(f"[!] Directory not found: {oscd_path}")
        print("    Please provide or mount OSCD dataset to this path.")
        return False

    loader = OSCDLoader(oscd_path, Path(labels_dir) if labels_dir else None)
    cities = loader.find_city_directories()
    print(f"[+] Found {len(cities)} city scenes:")
    invalid_scenes = []
    for name, path in sorted(cities.items()):
        first_dir = _image_directory(path, 1)
        second_dir = _image_directory(path, 2)
        missing_1 = _missing_bands(first_dir)
        missing_2 = _missing_bands(second_dir)
        valid = not missing_1 and not missing_2
        status = "13 bands in both dates" if valid else "invalid bands/images"
        print(f"    - {name:<15} [{status}]")
        if not valid:
            invalid_scenes.append((name, missing_1, missing_2))

    print("\nVerification summary:")
    print(f"Imagery: {len(cities)} city folders; {len(invalid_scenes)} with missing image dates/bands.")
    for split in ("train", "test"):
        split_file = oscd_path / f"{split}.txt"
        if not split_file.exists():
            continue
        split_cities = [
            city.strip().lower()
            for city in split_file.read_text(encoding="utf-8").replace("\n", "").split(",")
            if city.strip()
        ]
        labelled = [city for city in split_cities if loader.load_ground_truth_mask(city) is not None]
        print(f"Ground truth: labels found for {len(labelled)}/{len(split_cities)} cities in {split}.txt.")
        if len(labelled) < len(split_cities):
            print(f"[NOTE] {split} split has missing masks; corresponding training/evaluation is incomplete.")

    imagery_valid = len(cities) >= 20 and not invalid_scenes
    if imagery_valid:
        print(f"[SUCCESS] OSCD imagery schema verified for {len(cities)} cities.")
    else:
        print("[FAIL] OSCD imagery is incomplete or does not match the expected schema.")
    return imagery_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify OSCD directory structure")
    parser.add_argument("--oscd-dir", type=str, default="./data/oscd")
    parser.add_argument("--labels-dir", type=str)
    args = parser.parse_args()
    sys.exit(0 if verify_oscd(args.oscd_dir, args.labels_dir) else 1)
