"""Utility script to pre-fetch model weights for offline inference (Phase B+)."""
import argparse
from pathlib import Path


def download_models(output_dir: str):
    models_dir = Path(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("TerraEyes — Model Pre-fetch & Registry Downloader")
    print("=" * 60)
    print(f"Target directory: {models_dir.resolve()}\n")
    print("In Phase A, TerraEyes operates with built-in non-parametric models:")
    print("  - Embedding: MockEmbeddingModel (deterministic hash unit vectors)")
    print("  - Change Detection: PixelDiffChangeDetector (Otsu spectral difference)")
    print("\nFor Phase B & C, model download will be enabled following Checkpoints CP-1 & CP-2.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for offline use")
    parser.add_argument("--output-dir", type=str, default="./models")
    args = parser.parse_args()
    download_models(args.output_dir)
