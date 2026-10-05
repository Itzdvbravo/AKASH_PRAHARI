"""Evaluate hand-labeled semantic CLIP retrieval on held-out OSCD test tiles."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

root_dir = Path(__file__).resolve().parents[1]
for directory in (root_dir, root_dir / "backend", root_dir / "data-handling"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from app.models.embedding.clip_vit_b32 import CLIPViTB32EmbeddingModel


def evaluate_semantic_retrieval(
    manifest_path: str,
    splits_path: str,
    queries_path: str,
    checkpoint_path: str,
    output_path: str,
    top_ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Rank every held-out test tile for each manually labeled text query.

    The candidate vectors are created in memory from test-split imagery only;
    this function does not read from or modify the production train index.
    Recall@K is the share of queries with at least one relevant tile in the top K.
    """
    manifest_file = Path(manifest_path)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    annotations = json.loads(Path(queries_path).read_text(encoding="utf-8"))
    if annotations.get("split") != "test":
        raise ValueError("Semantic annotations must target the held-out test split")

    test_cities = set(splits["test"])
    candidates = [
        tile for tile in manifest.get("tiles", [])
        if tile["location_id"] in test_cities
    ]
    candidate_ids = [tile["tile_id"] for tile in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Ingestion manifest contains duplicate test tile IDs")
    if not candidates:
        raise ValueError("No held-out test tiles found in the ingestion manifest")

    for item in annotations.get("queries", []):
        relevant = set(item.get("relevant_tile_ids", []))
        if not relevant:
            raise ValueError(f"Query {item.get('id')} has no relevance labels")
        missing = relevant.difference(candidate_ids)
        if missing:
            raise ValueError(
                f"Query {item.get('id')} labels tiles outside the test candidate set: "
                f"{sorted(missing)}"
            )

    model = CLIPViTB32EmbeddingModel(checkpoint_path)
    image_vectors = []
    for tile in candidates:
        arrays_path = manifest_file.parent / tile["array_path"]
        with np.load(arrays_path, allow_pickle=False) as arrays:
            image = arrays["after"].astype(np.float32)
        image_vectors.append(model.encode_image(image))
    image_vectors = np.stack(image_vectors).astype(np.float32)
    image_vectors /= np.maximum(
        np.linalg.norm(image_vectors, axis=1, keepdims=True), 1e-12
    )

    per_query = []
    for item in annotations["queries"]:
        text_vector = model.encode_text(item["query"]).astype(np.float32)
        text_vector /= max(float(np.linalg.norm(text_vector)), 1e-12)
        scores = image_vectors @ text_vector
        ranked = np.argsort(-scores)
        relevant = set(item["relevant_tile_ids"])
        ranks = [
            rank + 1 for rank, index in enumerate(ranked)
            if candidate_ids[int(index)] in relevant
        ]
        first_rank = min(ranks)
        per_query.append({
            "id": item["id"],
            "query": item["query"],
            "relevant_tile_ids": sorted(relevant),
            "relevant_tile_count": len(relevant),
            "first_relevant_rank": first_rank,
            "reciprocal_rank": 1.0 / first_rank,
            "top_10": [
                {
                    "rank": rank + 1,
                    "tile_id": candidate_ids[int(index)],
                    "location_id": candidates[int(index)]["location_id"],
                    "score": float(scores[int(index)]),
                    "relevant": candidate_ids[int(index)] in relevant,
                }
                for rank, index in enumerate(ranked[:10])
            ],
        })

    metrics = {
        f"recall@{k}": float(np.mean([
            item["first_relevant_rank"] <= k for item in per_query
        ]))
        for k in top_ks
    }
    report = {
        "model": "clip_vit_b32",
        "embedding_dim": model.embedding_dim,
        "dataset": "OSCD",
        "evaluation": "hand-labeled semantic text-to-image retrieval",
        "candidate_split": "test only; isolated in memory from the production train index",
        "candidate_city_count": len(test_cities),
        "candidate_tile_count": len(candidates),
        "query_count": len(per_query),
        "annotation_method": annotations["annotation_method"],
        "metrics": metrics,
        "mrr": float(np.mean([item["reciprocal_rank"] for item in per_query])),
        "queries": per_query,
        "limitations": [
            "Scene-content relevance is manually labeled and is not an official OSCD benchmark.",
            "Queries are evaluated against after-date RGB composites, not before/after pairs.",
            "The small query set is an initial prototype benchmark; labels and phrasing should be reviewed before model selection.",
        ],
    }
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
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
        "--queries", default="./docs/evaluation/oscd_semantic_queries.json"
    )
    parser.add_argument("--checkpoint-path", default="./models/clip_vit_b32.pt")
    parser.add_argument(
        "--output", default="./data/evaluation/clip_vit_b32_semantic_test.json"
    )
    args = parser.parse_args()
    evaluate_semantic_retrieval(
        args.manifest,
        args.splits,
        args.queries,
        args.checkpoint_path,
        args.output,
    )
