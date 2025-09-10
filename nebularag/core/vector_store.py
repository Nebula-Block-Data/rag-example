from typing import List, Tuple
import math


def _dot(a: List[float], b: List[float]) -> float:
    """Calculate dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    """Calculate L2 norm (magnitude) of a vector."""
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        ValueError: If vectors have different lengths
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    na = _norm(a)
    nb = _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


class InMemoryVectorStore:
    """
    In-memory vector store for storing and searching text embeddings.
    
    This implementation uses cosine similarity for vector search and stores
    both text and embeddings in memory for fast retrieval.
    """
    
    def __init__(self) -> None:
        """Initialize an empty vector store."""
        self.texts: List[str] = []
        self.embeddings: List[List[float]] = []

    def add(self, texts: List[str], embeddings: List[List[float]]) -> None:
        """
        Add texts and their embeddings to the store.
        
        Args:
            texts: List of text strings
            embeddings: List of embedding vectors (same length as texts)
            
        Raises:
            ValueError: If texts and embeddings have different lengths
        """
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have same length")
        if not texts:
            return
        
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for the most similar vectors using cosine similarity.
        
        Args:
            query_embedding: The query vector to search with
            k: Number of top results to return
            
        Returns:
            List of (index, score) tuples sorted by descending similarity score
            
        Raises:
            ValueError: If k is not positive or if store is empty
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if not self.embeddings:
            return []
        
        scores: List[Tuple[int, float]] = []
        for i, emb in enumerate(self.embeddings):
            try:
                score = cosine_similarity(query_embedding, emb)
                scores.append((i, score))
            except ValueError:
                # Skip embeddings with different dimensions
                continue
        
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:k]
    
    def clear(self) -> None:
        """Clear all stored texts and embeddings."""
        self.texts.clear()
        self.embeddings.clear()
    
    def size(self) -> int:
        """Return the number of stored vectors."""
        return len(self.texts)

