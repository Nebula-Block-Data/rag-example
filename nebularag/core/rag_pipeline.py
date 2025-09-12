from typing import List, Dict, Any, Optional, Tuple
from ..clients.nebula_client import NebulaBlockClient
from ..utils.text_processing import split_text
from .vector_store import InMemoryVectorStore


class RAGPipeline:
    def __init__(
        self,
        client: NebulaBlockClient,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        top_k: int = 12,
        rerank_k: int = 6,
    ) -> None:
        self.client = client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rerank_k = rerank_k
        self.store = InMemoryVectorStore()

    def index_texts(self, docs: List[str]) -> int:
        chunks: List[str] = []
        for doc in docs:
            chunks.extend(split_text(doc, self.chunk_size, self.chunk_overlap))
        if not chunks:
            return 0
        embeddings = self.client.embed(chunks)
        self.store.add(chunks, embeddings)
        return len(chunks)

    def retrieve(self, question: str) -> List[Tuple[int, float]]:
        q_emb = self.client.embed([question])[0]
        return self.store.search(q_emb, k=self.top_k)

    def rerank(self, question: str, candidate_indices: List[int]) -> List[int]:
        documents = [self.store.texts[i] for i in candidate_indices]
        results = self.client.rerank(question, documents, top_n=self.rerank_k)
        # Expect results items to include "index" within given documents list
        # Map back to original corpus indices
        out: List[int] = []
        for item in results:
            local_idx = item.get("index")
            if isinstance(local_idx, int) and 0 <= local_idx < len(candidate_indices):
                out.append(candidate_indices[local_idx])
        return out

    def build_context(self, indices: List[int]) -> str:
        snippets = [self.store.texts[i] for i in indices]
        return "\n\n---\n\n".join(snippets)

    def answer(self, question: str, max_context_docs: Optional[int] = None) -> Dict[str, Any]:
        candidates = self.retrieve(question)
        cand_indices = [i for i, _ in candidates]
        reranked = self.rerank(question, cand_indices) if cand_indices else []
        final_indices = reranked or cand_indices[: (max_context_docs or self.rerank_k)]
        context = self.build_context(final_indices)

        system_prompt = (
            "You are a helpful assistant. Use the provided context to answer.\n"
            "If the answer is not present in the context, say you don't know."
        )
        user_prompt = (
            f"Context:\n{context}\n\nQuestion: {question}\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        output = self.client.chat(messages, temperature=0.2)
        sources = [self.store.texts[i] for i in final_indices]
        return {
            "answer": output, 
            "sources": sources, 
            "indices": final_indices,
            "models": {
                "embedding": self.client.embedding_model,
                "reranker": self.client.reranker_model,
                "chat": self.client.chat_model
            }
        }

