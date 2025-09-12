import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from ..clients.nebula_client import NebulaBlockClient
from ..core.rag_pipeline import RAGPipeline
from ..utils.file_utils import read_text_files
from ..config import get_settings


def build_client_from_env() -> NebulaBlockClient:
    """
    Build NebulaBlockClient from environment variables.
    
    Returns:
        Configured NebulaBlockClient instance
        
    Raises:
        RuntimeError: If required environment variables are missing
    """
    api_key = os.environ.get("NEBULABLOCK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBULABLOCK_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )
    
    return NebulaBlockClient(
        base_url=os.environ.get("NEBULABLOCK_BASE_URL"),
        api_key=api_key,
        embedding_model=os.environ.get("NEBULABLOCK_EMBEDDING_MODEL"),
        reranker_model=os.environ.get("NEBULABLOCK_RERANKER_MODEL"),
        chat_model=os.environ.get("NEBULABLOCK_CHAT_MODEL"),
        embeddings_path=os.environ.get("NEBULABLOCK_EMBEDDINGS_PATH"),
        rerank_path=os.environ.get("NEBULABLOCK_RERANK_PATH"),
        chat_path=os.environ.get("NEBULABLOCK_CHAT_PATH"),
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must be less than chunk-size")
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.chunk_overlap < 0:
        raise ValueError("chunk-overlap must be non-negative")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.rerank_k <= 0:
        raise ValueError("rerank-k must be positive")
    if args.rerank_k > args.top_k:
        print(f"Warning: rerank-k ({args.rerank_k}) > top-k ({args.top_k})", file=sys.stderr)


def main() -> None:
    """Main CLI entry point for the RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="NebulaRAG - Minimal RAG pipeline with NebulaBlock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --docs docs --question "What is the main topic?"
  %(prog)s --docs docs --question "Explain X" --chunk-size 1000 --top-k 15
        """
    )
    parser.add_argument("--docs", required=True, help="Path to docs directory (txt/md)")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--chunk-size", type=int, default=800, 
                       help="Size of text chunks (default: 800)")
    parser.add_argument("--chunk-overlap", type=int, default=120,
                       help="Overlap between chunks (default: 120)")
    parser.add_argument("--top-k", type=int, default=12,
                       help="Number of candidates to retrieve (default: 12)")
    parser.add_argument("--rerank-k", type=int, default=6,
                       help="Number of candidates after reranking (default: 6)")
    
    args = parser.parse_args()
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print("Initializing NebulaBlock client...")
        client = build_client_from_env()
        
        print("Model Configuration:")
        print(f"  Embedding Model: {client.embedding_model}")
        print(f"  Reranker Model:  {client.reranker_model}")
        print(f"  Chat Model:      {client.chat_model}")
        print()
        
        print("Setting up RAG pipeline...")
        rag = RAGPipeline(
            client,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            rerank_k=args.rerank_k,
        )

        print(f"Reading documents from {args.docs}...")
        docs = read_text_files(args.docs)
        print(f"Found {len(docs)} documents")

        print("Indexing documents...")
        num_chunks = rag.index_texts(docs)
        print(f"Indexed {num_chunks} chunks from {len(docs)} files.")

        print("Processing question...")
        result = rag.answer(args.question)
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result["answer"])
        
        print("\n" + "="*60)
        print("SOURCES:")
        print("="*60)
        for i, src in enumerate(result["sources"], start=1):
            first_line = src.splitlines()[0] if src.splitlines() else src[:80]
            print(f"{i}. {first_line[:120]}{'...' if len(first_line) > 120 else ''}")
        
        print("\n" + "="*60)
        print("MODELS USED:")
        print("="*60)
        if "models" in result:
            print(f"Embedding Model: {result['models']['embedding']}")
            print(f"Reranker Model:  {result['models']['reranker']}")
            print(f"Chat Model:      {result['models']['chat']}")
        else:
            print("Model information not available in response")
            
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

