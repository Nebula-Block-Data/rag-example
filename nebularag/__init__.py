"""NebulaRAG - A minimal RAG pipeline with NebulaBlock integration."""

from .clients import NebulaBlockClient
from .core import RAGPipeline, InMemoryVectorStore
from .utils import split_text, read_text_files
from .config import get_settings

__all__ = [
    "NebulaBlockClient",
    "RAGPipeline", 
    "InMemoryVectorStore",
    "split_text",
    "read_text_files",
    "get_settings",
]

__version__ = "0.1.0"

