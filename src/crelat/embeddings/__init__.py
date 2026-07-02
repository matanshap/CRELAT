"""Text embedding providers."""

from crelat.embeddings.base import TextEmbedder
from crelat.embeddings.transformers import create_embedder

__all__ = ["TextEmbedder", "create_embedder"]
