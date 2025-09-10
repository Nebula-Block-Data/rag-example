import os
import sys
from rag_example.nebula_client import NebulaBlockClient
import json


def test_embeddings(client: NebulaBlockClient) -> bool:
    """Test the embeddings endpoint."""
    print("Testing Embeddings...")
    try:
        test_texts = ["hello world", "rag pipeline test", "vector similarity search"]
        embs = client.embed(test_texts)
        
        if not embs or len(embs) != len(test_texts):
            print("ERROR: Unexpected number of embeddings returned")
            return False
            
        if not embs[0] or len(embs[0]) == 0:
            print("ERROR: Empty embedding returned")
            return False
            
        print(f"✓ Embeddings OK; dimensions: {len(embs[0])}, count: {len(embs)}")
        return True
    except Exception as e:
        print(f"✗ Embeddings ERROR: {e}")
        return False


def test_rerank(client: NebulaBlockClient) -> bool:
    """Test the reranking endpoint."""
    print("\nTesting Rerank...")
    try:
        test_docs = [
            "Split, embed, retrieve, rerank, and answer with context.",
            "This is an unrelated sentence about cooking pasta.",
            "Vector search finds candidates; reranker sorts by relevance.",
            "Machine learning models can be used for text classification.",
        ]
        
        res = client.rerank(
            query="What is the pipeline flow?",
            documents=test_docs,
            top_n=2,
            return_documents=True,
        )
        
        if not res or len(res) == 0:
            print("ERROR: No reranking results returned")
            return False
            
        print(f"✓ Rerank OK; top {len(res)} items:")
        for i, item in enumerate(res, 1):
            score = item.get("relevance_score", "N/A")
            doc = item.get("document", "N/A")[:50] + "..." if len(item.get("document", "")) > 50 else item.get("document", "N/A")
            print(f"  {i}. Score: {score}, Doc: {doc}")
        return True
    except Exception as e:
        print(f"✗ Rerank ERROR: {e}")
        return False


def test_chat(client: NebulaBlockClient) -> bool:
    """Test the chat endpoint."""
    print("\nTesting Chat...")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Keep responses brief."},
            {"role": "user", "content": "Say hello and confirm you're working."}
        ]
        
        out = client.chat(messages, temperature=0.1, max_tokens=100)
        
        if not out or len(out.strip()) == 0:
            print("ERROR: Empty chat response")
            return False
            
        print(f"✓ Chat OK; response: {out[:100].replace(chr(10), ' ')}...")
        return True
    except Exception as e:
        print(f"✗ Chat ERROR: {e}")
        return False


def main() -> None:
    """Test all NebulaBlock API endpoints."""
    print("NebulaBlock API Test Suite")
    print("=" * 40)
    
    # Check environment variables
    api_key = os.environ.get("NEBULABLOCK_API_KEY")
    base_url = os.environ.get("NEBULABLOCK_BASE_URL", "https://dev-llm-proxy.nebulablock.com/v1")
    
    if not api_key:
        print("ERROR: NEBULABLOCK_API_KEY environment variable is not set")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    print(f"Base URL: {base_url}")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print()
    
    try:
        client = NebulaBlockClient()
    except Exception as e:
        print(f"ERROR: Failed to initialize client: {e}")
        sys.exit(1)
    
    # Run tests
    results = []
    results.append(test_embeddings(client))
    results.append(test_rerank(client))
    results.append(test_chat(client))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
