"""
Embeddings layer for semantic search.

Responsibilities:
- Load a lightweight sentence-transformer model.
- Precompute embeddings for all restaurants (offline).
- Encode user free-text preferences at request time.
- Provide cosine similarity scores for candidate ranking.
"""
