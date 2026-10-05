"""Evaluate CLIP text-to-image retrieval on the OSCD held-out city split."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

root_dir = Path(__file__).resolve().parents[1]
backend_dir = root_dir / "backend"
data_handling_dir = root_dir / "data-handling"
for directory in (root_dir, backend_dir, data_handling_dir):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.models.embedding.clip_vit_b32 import CLIPViTB32EmbeddingModel
from app.models.embedding.remote_clip import RemoteCLIPEmbeddingModel


def evaluate_location_retrieval(
    manifest_path: str,
    splits_path: str,
    checkpoint_path: str,
    output_path: str,
    model_name: str = "clip_vit_b32",
    top_ks: tuple[int, ...] = (5, 10),
) -> dict:
    """Run a location-name retrieval sanity check against a test-only index.

    All tiles from a test city are relevant to that city's query. This measures
    location-name retrieval only; it is not a land-cover semantic benchmark.
    """
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    test_cities = set(splits["test"])
    candidates = [tile for tile in manifest.get("tiles", []) if tile["location_id"] in test_cities]
    if not candidates:
        raise ValueError("No test-split tiles found in the ingestion manifest")

    if model_name == "clip_vit_b32":
        model = CLIPViTB32EmbeddingModel(checkpoint_path)
    elif model_name == "remoteclip_vit_b32":
        model = RemoteCLIPEmbeddingModel(checkpoint_path)
    else:
        raise ValueError(f"Unsupported retrieval model: {model_name}")
    image_vectors = []
    for tile in candidates:
        array_path = manifest_file.parent / tile["array_path"]
        with np.load(array_path, allow_pickle=False) as arrays:
            image_vectors.append(model.encode_image(arrays["after"].astype(np.float32)))
    image_vectors = np.stack(image_vectors)

    per_query = []
    for city in sorted(test_cities):
        relevant = {
            index for index, tile in enumerate(candidates)
            if tile["location_id"] == city
        }
        if not relevant:
            continue
        query = f"satellite imagery of {city.replace('_', ' ')}"
        query_vector = model.encode_text(query)
        ranked = np.argsort(-(image_vectors @ query_vector))
        hit_ranks = [rank + 1 for rank, candidate_index in enumerate(ranked) if candidate_index in relevant]
        first_relevant_rank = min(hit_ranks)
        per_query.append({
            "query": query,
            "target_city": city,
            "relevant_tiles": len(relevant),
            "first_relevant_rank": first_relevant_rank,
            "reciprocal_rank": 1.0 / first_relevant_rank,
            "top_result_city": candidates[int(ranked[0])]["location_id"],
        })

    if not per_query:
        raise ValueError("No labeled test-city queries were evaluated")
    metrics = {
        f"recall@{k}": float(np.mean([
            item["first_relevant_rank"] <= k for item in per_query
        ]))
        for k in top_ks
    }
    report = {
        "model": model_name,
        "embedding_dim": model.embedding_dim,
        "evaluation": "held-out OSCD city-name retrieval sanity check",
        "index_split": "test only (isolated from production train index)",
        "candidate_tile_count": len(candidates),
        "query_count": len(per_query),
        "metrics": metrics,
        "mrr": float(np.mean([item["reciprocal_rank"] for item in per_query])),
        "queries": per_query,
        "limitation": "Location-name retrieval only; it does not measure land-cover concept retrieval.",
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": report["model"],
        "candidate_tile_count": report["candidate_tile_count"],
        "query_count": report["query_count"],
        "metrics": report["metrics"],
        "mrr": report["mrr"],
        "report": str(output_file),
    }, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="./data/oscd_tiles/manifest.json")
    parser.add_argument("--splits", default="./data-handling/splits/oscd_splits.json")
    parser.add_argument(
        "--model", choices=("clip_vit_b32", "remoteclip_vit_b32"),
        default="clip_vit_b32",
    )
    parser.add_argument("--checkpoint-path")
    parser.add_argument(
        "--output", default="./data/evaluation/clip_vit_b32_retrieval.json"
    )
    arguments = parser.parse_args()
    checkpoint = arguments.checkpoint_path or (
        "./models/remoteclip_vit_b32.pt"
        if arguments.model == "remoteclip_vit_b32"
        else "./models/clip_vit_b32.pt"
    )
    evaluate_location_retrieval(
        arguments.manifest,
        arguments.splits,
        checkpoint,
        arguments.output,
        model_name=arguments.model,
    )
