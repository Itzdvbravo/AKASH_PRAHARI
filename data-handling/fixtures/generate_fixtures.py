"""Script to generate synthetic PNG satellite tile pairs and binary change masks for testing."""
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image


def generate_synthetic_fixtures() -> None:
    target_dir = Path(__file__).parent / "synthetic_tiles"
    target_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # 1. Base image for Paris 2018 (256x256 RGB)
    paris_2018 = np.zeros((256, 256, 3), dtype=np.uint8)
    # Background field (greenish)
    paris_2018[:, :] = [34, 139, 34]
    # River across middle (blue)
    paris_2018[110:140, :] = [30, 144, 255]
    # Small urban block top left (grey)
    paris_2018[20:80, 20:80] = [128, 128, 128]

    Image.fromarray(paris_2018).save(target_dir / "paris_2018-03-10.png")

    # 2. Paris 2020 (New urban construction block added bottom right)
    paris_2020 = paris_2018.copy()
    # New building site (bright white/reddish construct)
    paris_2020[160:210, 160:210] = [220, 100, 50]

    Image.fromarray(paris_2020).save(target_dir / "paris_2020-06-15.png")

    # 3. Ground truth change mask (White where changed, Black elsewhere)
    paris_mask = np.zeros((256, 256), dtype=np.uint8)
    paris_mask[160:210, 160:210] = 255
    Image.fromarray(paris_mask).save(target_dir / "paris_change_mask.png")

    # 4. Berlin 2019 tile
    berlin_2019 = np.full((256, 256, 3), [60, 100, 70], dtype=np.uint8)
    berlin_2019[40:100, 100:200] = [160, 160, 160]
    Image.fromarray(berlin_2019).save(target_dir / "berlin_2019-05-20.png")

    print(f"Generated synthetic fixtures in: {target_dir.resolve()}")


if __name__ == "__main__":
    generate_synthetic_fixtures()
