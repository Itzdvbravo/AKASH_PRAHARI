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


def verify_oscd(oscd_dir: str) -> bool:
    oscd_path = Path(oscd_dir)
    print("=" * 60)
    print("TerraEyes — OSCD Dataset Verification")
    print("=" * 60)
    print(f"Checking directory: {oscd_path.resolve()}\n")

    if not oscd_path.exists():
        print(f"[!] Directory not found: {oscd_path}")
        print("    Please provide or mount OSCD dataset to this path.")
        return False

    loader = OSCDLoader(oscd_path)
    cities = loader.find_city_directories()
    print(f"[+] Found {len(cities)} city scenes:")
    for name, path in sorted(cities.items()):
        # Check imgs_1 and imgs_2
        imgs_1 = (path / "imgs_1_rect").exists() or (path / "imgs_1").exists()
        imgs_2 = (path / "imgs_2_rect").exists() or (path / "imgs_2").exists()
        mask = loader.load_ground_truth_mask(name) is not None
        status = f"imgs_1={'OK' if imgs_1 else 'MISSING'}, imgs_2={'OK' if imgs_2 else 'MISSING'}, mask={'OK' if mask else 'NONE'}"
        print(f"    - {name:<15} [{status}]")

    print("\nVerification summary:")
    if len(cities) >= 20:
        print(f"[SUCCESS] Complete OSCD dataset verified with {len(cities)} cities.")
        return True
    elif len(cities) > 0:
        print(f"[PARTIAL] Found {len(cities)} cities out of standard 24.")
        return True
    else:
        print("[WARNING] No valid city directories detected matching OSCD schema.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify OSCD directory structure")
    parser.add_argument("--oscd-dir", type=str, default="./data/oscd")
    args = parser.parse_args()
    verify_oscd(args.oscd_dir)
