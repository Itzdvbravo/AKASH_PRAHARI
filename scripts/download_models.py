"""Fetch approved model weights during setup for offline runtime use."""
import argparse
from pathlib import Path


def download_clip_vit_b32(output_dir: str) -> Path:
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install torch and open_clip_torch before downloading CLIP weights."
        ) from exc

    models_dir = Path(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "clip_vit_b32.pt"
    temporary = target.with_suffix(target.suffix + ".tmp")

    # Network access happens only in this explicit setup command. The API uses
    # the saved state dict path and never asks OpenCLIP to fetch weights.
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai", device="cpu"
    )
    torch.save(model.state_dict(), temporary)
    temporary.replace(target)
    print(f"Saved OpenAI CLIP ViT-B/32 weights to {target.resolve()}")
    return target


def download_remoteclip_vit_b32(output_dir: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install open_clip_torch before downloading RemoteCLIP weights.") from exc

    models_dir = Path(output_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    target = models_dir / "remoteclip_vit_b32.pt"
    if target.is_file() and target.stat().st_size > 1_000_000:
        print(f"RemoteCLIP checkpoint already exists at {target.resolve()}")
        return target
    downloaded = Path(hf_hub_download(
        repo_id="chendelong/RemoteCLIP",
        filename="RemoteCLIP-ViT-B-32.pt",
        local_dir=models_dir,
    ))
    if downloaded != target:
        downloaded.replace(target)
    print(f"Saved RemoteCLIP ViT-B/32 weights to {target.resolve()}")
    return target


def download_models(output_dir: str, model_name: str) -> None:
    if model_name == "clip_vit_b32":
        download_clip_vit_b32(output_dir)
        return
    if model_name == "remoteclip_vit_b32":
        download_remoteclip_vit_b32(output_dir)
        return
    raise ValueError(f"Unsupported approved model: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch model weights for offline inference")
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument(
        "--model",
        choices=("clip_vit_b32", "remoteclip_vit_b32"),
        default="clip_vit_b32",
    )
    arguments = parser.parse_args()
    download_models(arguments.output_dir, arguments.model)
