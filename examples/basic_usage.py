#!/usr/bin/env python3
"""
Basic usage example for RAG Example.

This script demonstrates how to use the RAG pipeline programmatically.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import nebularag
sys.path.insert(0, str(Path(__file__).parent.parent))

from nebularag.core import RAGPipeline
from nebularag.clients import NebulaBlockClient
from nebularag.utils import read_text_files
from nebularag.config import get_settings


def main():
    """Run a basic RAG example."""
    print("NebulaRAG - Basic Usage")
    print("=" * 40)
    
    # Check if we have the required environment variables
    if not os.environ.get("NEBULABLOCK_API_KEY"):
        print("ERROR: NEBULABLOCK_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        return 1
    
    try:
        # Get settings
        settings = get_settings()
        print(f"Using base URL: {settings.nebula_base_url}")
        
        # Initialize client
        client = NebulaBlockClient()
        
        # Create RAG pipeline
        rag = RAGPipeline(
            client=client,
            chunk_size=settings.default_chunk_size,
            chunk_overlap=settings.default_chunk_overlap,
            top_k=settings.default_top_k,
            rerank_k=settings.default_rerank_k,
        )
        
        # Read documents
        docs_dir = Path(__file__).parent.parent / "docs"
        if not docs_dir.exists():
            print(f"ERROR: Documentation directory not found: {docs_dir}")
            return 1
        
        print(f"Reading documents from: {docs_dir}")
        docs = read_text_files(str(docs_dir))
        print(f"Found {len(docs)} documents")
        
        # Index documents
        print("Indexing documents...")
        num_chunks = rag.index_texts(docs)
        print(f"Indexed {num_chunks} chunks")
        
        # Ask a question
        question = "What is this document about?"
        print(f"\nQuestion: {question}")
        
        result = rag.answer(question)
        
        print("\nAnswer:")
        print("-" * 40)
        print(result["answer"])
        
        print(f"\nSources ({len(result['sources'])}):")
        print("-" * 40)
        for i, source in enumerate(result["sources"], 1):
            preview = source[:100] + "..." if len(source) > 100 else source
            print(f"{i}. {preview}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
