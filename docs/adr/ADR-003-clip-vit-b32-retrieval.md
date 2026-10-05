# ADR-003: CLIP ViT-B/32 as the Phase B retrieval baseline

- **Status:** Accepted provisionally for integration
- **Date:** 2026-10-02
- **Decision owner:** User

## Context

TerraEyes needs an offline text-to-image embedding model for the OSCD
Sentinel-2 RGB composites. The repository's provisional path was to integrate
the smaller CLIP ViT-B/32 model first, then compare remote-sensing-specific
RemoteCLIP on held-out OSCD cases before making a final model choice.

## Decision

Use OpenAI CLIP ViT-B/32 as the Phase B integration baseline, loaded through
OpenCLIP with its QuickGELU-compatible architecture. The approved checkpoint is
stored at `models/clip_vit_b32.pt`; application startup requires this local
file and performs no weight download. Build the production retrieval index
from the archive's train split only.

## Evidence available

- Text and image encoders produce L2-normalized 512-dimensional float32
  vectors.
- Full local OSCD imagery ingestion produced 147 paired tiles.
- The production index contains 100 latest-date image embeddings from the 14
  training cities; the 10 test cities are excluded.
- A separate test-only location-name sanity check over 47 tiles and 10 city
  queries measured Recall@5 0.90, Recall@10 0.90, and MRR 0.7417.
- The sanity check does not measure land-cover concept retrieval. The planned
  hand-authored semantic-query benchmark and RemoteCLIP comparison remain
  outstanding.

## Consequences

- CLIP and OpenCLIP dependencies are part of the Phase B setup environment.
- Changing the embedding model requires rebuilding the vector index; model
  metadata prevents loading an index produced by a different encoder.
- The separate OSCD label archive is still required for supervised change
  detection evaluation; it does not block retrieval indexing.
- RemoteCLIP remains a candidate for comparison before treating this choice as
  final for remote-sensing semantic retrieval.
