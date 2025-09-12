# Nebula Block Inference Service for RAG

## Overview

Nebula Block provides a **sovereign AI inference platform** optimized for Retrieval-Augmented Generation (RAG) pipelines. The service combines **Canadian data residency** with **global GPU access**, ensuring compliance for regulated workloads while maintaining high performance and cost efficiency.

## Key Features

* **Serverless Model Inference**
  OpenAI-compatible endpoints for seamless integration into existing RAG pipelines. Developers can call Nebula APIs the same way they would OpenAI, minimizing switching costs.

* **Optimized for RAG**

  * Quantization-aware serving for large LLMs.
  * Caching & smart autoscaling to reduce latency by up to 50% at ≤5% accuracy trade-off.
  * Integrated reranker models (e.g., `Mxbai-Rerank-Large-V2`, `BAAI/bge-reranker-v2-m3`) for retrieval quality.

* **Sovereign by Design**
  100% Canadian-owned and operated. All inference and storage can be confined within Canada, ensuring **compliance with Law 25, PIPEDA, and PHIPA**.

* **Developer-Friendly**

  * Prebuilt Hugging Face integration pipelines.
  * Kubernetes/KVM multi-DC orchestrator.
  * S3-compatible storage for embeddings, documents, and vector indices.

* **Cost Advantage**
  H100 GPU pricing as low as **\$2.99–\$4.50/hr**, with **67% reimbursement** through the **Canada AI Compute Access Fund (ACAF)** until March 2028.

## Example RAG Architecture on Nebula Block

1. **Document Ingestion**
   Upload files to Nebula Block’s S3-compatible object storage.
2. **Embedding Generation**
   Use BAAI `bge-large` or custom embeddings on Nebula GPU instances.
3. **Vector Indexing**
   Store vectors in ChromaDB or Milvus deployed in Nebula’s secure cloud.
4. **Query Processing**
   Retrieve top-k passages from the index, rerank with `Mxbai-Rerank-Large-V2`.
5. **LLM Inference**
   Call Nebula Block’s inference endpoint (`/v1/chat/completions`) for final answer generation.
