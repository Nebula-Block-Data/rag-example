"""Text processing utilities."""

from typing import List


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """
    Split text into overlapping chunks of fixed size.
    
    This is a simple character-based splitter that maintains overlap between chunks
    to preserve context across boundaries. It's designed to be dependency-free.
    
    Args:
        text: The input text to split
        chunk_size: Maximum size of each chunk in characters (default: 800)
        chunk_overlap: Number of characters to overlap between chunks (default: 120)
        
    Returns:
        List of text chunks, each trimmed of whitespace
        
    Raises:
        ValueError: If chunk_overlap >= chunk_size or if chunk_size <= 0
        
    Example:
        >>> text = "This is a long document that needs to be split into chunks."
        >>> chunks = split_text(text, chunk_size=20, chunk_overlap=5)
        >>> len(chunks)
        3
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    
    text = text.strip()
    if not text:
        return []
    
    chunks: List[str] = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
    
    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove common artifacts
    text = text.replace("\u00a0", " ")  # Non-breaking space
    text = text.replace("\u200b", "")   # Zero-width space
    
    return text.strip()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the result
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix
