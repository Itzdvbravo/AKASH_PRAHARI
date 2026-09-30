# ADR-001: Choice of Vector Store (FAISS vs. ChromaDB)

## Status
Accepted

## Context
TerraEyes requires a vector search index that functions completely offline without running external network daemon processes. It must store 512-dimensional to 768-dimensional normalized vision-language embeddings and perform sub-10ms top-k cosine similarity queries.

## Decision
We select **FAISS** (`IndexFlatIP` with pure NumPy fallback).

## Consequences
- Zero external runtime server dependencies.
- Index can be saved and loaded directly to/from disk (`faiss.index`).
- Pure NumPy fallback ensures the application runs seamlessly even in environments without compiled binary C++ wheels for FAISS.
