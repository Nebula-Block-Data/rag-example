import os
import json
import time
from typing import List, Dict, Any, Optional
import urllib.request
import urllib.error


class NebulaBlockClient:
    """
    Lightweight client for NebulaBlock inference service.

     This assumes OpenAI/Cohere-like JSON shapes but keeps endpoints configurable
    so you can adapt without changing code.

    Configure via env vars or constructor args:
      - NEBULABLOCK_BASE_URL (default: https://dev-llm-proxy.nebulablock.com/v1)
      - NEBULABLOCK_API_KEY (e.g., sk-...)
      - NEBULABLOCK_EMBEDDINGS_PATH (default: /embeddings)
      - NEBULABLOCK_RERANK_PATH (default: /rerank)
      - NEBULABLOCK_CHAT_PATH (default: /chat/completions)

      - NEBULABLOCK_EMBEDDING_MODEL (default: Qwen/Qwen3-Embedding-8B)
      - NEBULABLOCK_RERANKER_MODEL (default: BAAI/bge-reranker-v2-m3)
      - NEBULABLOCK_CHAT_MODEL (default: Mistral-Small-24B-Instruct-2501)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        reranker_model: Optional[str] = None,
        chat_model: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        rerank_path: Optional[str] = None,
        chat_path: Optional[str] = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("NEBULABLOCK_BASE_URL")
            or "https://dev-llm-proxy.nebulablock.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("NEBULABLOCK_API_KEY", "")

        self.embedding_model = embedding_model or os.environ.get(
            "NEBULABLOCK_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
        )
        self.reranker_model = reranker_model or os.environ.get(
            "NEBULABLOCK_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.chat_model = chat_model or os.environ.get(
            "NEBULABLOCK_CHAT_MODEL", "Mistral-Small-24B-Instruct-2501"
        )

        self.embeddings_path = embeddings_path or os.environ.get(
            "NEBULABLOCK_EMBEDDINGS_PATH", "/embeddings"
        )
        self.rerank_path = rerank_path or os.environ.get("NEBULABLOCK_RERANK_PATH", "/rerank")
        self.chat_path = chat_path or os.environ.get("NEBULABLOCK_CHAT_PATH", "/chat/completions")

        self.timeout = timeout

    # ------------------------------ HTTP ------------------------------ #
    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.base_url:
            raise RuntimeError("NEBULABLOCK_BASE_URL is not set.")
        if not self.api_key:
            raise RuntimeError("NEBULABLOCK_API_KEY is not set.")

        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        
        # Add headers to make the request look more legitimate
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        req.add_header("Accept", "application/json")
        req.add_header("Accept-Language", "en-US,en;q=0.9")
        req.add_header("Accept-Encoding", "gzip, deflate, br")
        req.add_header("Connection", "keep-alive")
        req.add_header("Sec-Fetch-Dest", "empty")
        req.add_header("Sec-Fetch-Mode", "cors")
        req.add_header("Sec-Fetch-Site", "cross-site")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                return json.loads(body.decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            # Add a small delay on error to avoid rapid retries
            time.sleep(0.5)
            raise RuntimeError(f"HTTPError {e.code} for {url}: {detail}")
        except urllib.error.URLError as e:
            time.sleep(0.5)
            raise RuntimeError(f"URLError for {url}: {e}")

    # ---------------------------- Embeddings -------------------------- #
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Calls embeddings endpoint. Assumes payload {"model": ..., "input": [...]}
        and response like {"data": [{"embedding": [...]}, ...]}.
        """
        payload = {"model": self.embedding_model, "input": texts}
        resp = self._request(self.embeddings_path, payload)

        data = resp.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected embeddings response: {resp}")
        out: List[List[float]] = []
        for item in data:
            emb = item.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError(f"Missing 'embedding' in item: {item}")
            out.append([float(x) for x in emb])
        return out

    # ----------------------------- Reranker --------------------------- #
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Calls rerank endpoint. Assumes Cohere-like payload and response:
          payload: { model, query, documents: ["...", ...], top_n }
          response: { results: [ {index, relevance_score, document?}, ... ] }
        """
        payload: Dict[str, Any] = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)
        if return_documents:
            payload["return_documents"] = True

        resp = self._request(self.rerank_path, payload)
        results = resp.get("results") or resp.get("data")
        if not isinstance(results, list):
            raise RuntimeError(f"Unexpected rerank response: {resp}")
        return results

    # ------------------------------- Chat ----------------------------- #
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: Optional[int] = None) -> str:
        """
        Calls chat/completions endpoint. Assumes OpenAI-like payload/response.
        """
        payload: Dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": float(temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        resp = self._request(self.chat_path, payload)
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"Unexpected chat response: {resp}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"Missing content in chat response: {resp}")
        return content
