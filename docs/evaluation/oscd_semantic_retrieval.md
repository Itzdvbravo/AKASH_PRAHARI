# Initial CLIP Semantic Retrieval Benchmark on OSCD

## Result

OpenAI CLIP ViT-B/32 was evaluated on ten hand-authored text queries against all
47 after-date tiles from the ten official OSCD test cities. The test candidates
were embedded in memory and kept separate from the 100-tile production index,
which contains only the fourteen official training cities.

| Metric | Result |
|---|---:|
| Recall@1 | 0.60 |
| Recall@5 | 0.80 |
| Recall@10 | 0.80 |
| MRR | 0.7075 |

Recall@K counts a query as a hit when at least one manually labeled relevant tile
appears in the top K. `first relevant rank` below is the best rank among that
query's labeled test tiles.

| Query | Relevant tiles | First relevant rank | Top result |
|---|---:|---:|---|
| Industrial blue roofs | 3 | 1 | Chongqing tile |
| Urban water body | 1 | 2 | Brasilia tile |
| Planned desert city | 1 | 1 | Dubai tile |
| Desert subdivisions | 9 | 1 | Las Vegas tile |
| Rural village and fields | 5 | 1 | Saclay West tile |
| Dense European urban | 7 | 1 | Montpellier tile |
| Hillside settlement | 2 | 34 | Dubai tile |
| Exposed-earth construction | 3 | 22 | Las Vegas tile |
| Geometric farmland | 11 | 2 | Dubai tile |
| Highway interchange | 3 | 1 | Dubai tile |

The benchmark reaches the plan's initial Recall@5 target of 0.5. It also shows
two clear misses: CLIP ranks the labeled hillside settlement at 34 and exposed
earthworks at 22. This is evidence for the prototype workflow, not a claim that
all land-cover concepts retrieve reliably.

## Method

- The labels in [`oscd_semantic_queries.json`](oscd_semantic_queries.json) were
  authored after visual inspection of the 47 B04/B03/B02 after-date composites.
- Relevance is scene-content relevance at tile level. These are manual labels,
  not official OSCD annotations; OSCD's official masks label binary change only.
- Each query ranks the full held-out test candidate set using cosine similarity
  between CLIP text embeddings and image embeddings.
- The script validates that every labeled tile belongs to the held-out test
  candidate set and does not load or modify the production vector index.

Reproduce from the repository root:

```powershell
python scripts/evaluate_semantic_retrieval.py
```

The full ranked output is written to the ignored local report
`data/evaluation/clip_vit_b32_semantic_test.json`.

## Limits and follow-up

This is a first-pass benchmark with ten queries and labels reviewed by one
annotator. Review the query wording and relevance labels independently before
using these numbers for model selection. The results do not evaluate before/after
change semantics, confidence calibration, or generalization beyond OSCD.
