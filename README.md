# NebulaRAG

A production-ready RAG (Retrieval-Augmented Generation) pipeline designed to work with NebulaBlock's Inference API. This project demonstrates how to build a complete RAG system with document indexing, semantic search, state-of-the-art reranking, and answer generation.

## 🚀 Features

- **Production-Ready**: Robust error handling, compression support, and browser-like headers
- **State-of-the-Art Models**: BAAI/bge-reranker-v2-m3 for superior reranking performance
- **Lightweight**: Minimal dependencies, no heavy ML frameworks
- **Configurable**: Environment-based configuration for all endpoints and models
- **OpenAI-Compatible**: Works with OpenAI-compatible APIs
- **Complete Pipeline**: Document splitting → embedding → retrieval → reranking → generation
- **CLI Interface**: Easy-to-use command-line interface with comprehensive options
- **In-Memory Store**: Fast vector similarity search with cosine similarity
- **Compression Support**: Handles Brotli and Gzip compression automatically
- **Cloudflare Bypass**: Browser-like headers to avoid security blocks

## 📋 Requirements

- Python 3.8+
- NebulaBlock API access
- Internet connection for API calls

## 🛠️ Installation

### Option 1: Development Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rag-example

# Install in development mode
pip install -e .
```

### Option 2: Direct Usage

```bash
# Clone the repository
git clone <repository-url>
cd rag-example

# Install dependencies
pip install -r requirements.txt

# Run directly
python -m nebularag.cli.main --help
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required
NEBULABLOCK_BASE_URL=https://inference.nebulablock.com/v1
NEBULABLOCK_API_KEY=sk-your-api-key-here

# Optional (defaults shown)
NEBULABLOCK_EMBEDDINGS_PATH=/embeddings
NEBULABLOCK_RERANK_PATH=/rerank
NEBULABLOCK_CHAT_PATH=/chat/completions

# Models (optimized for performance)
NEBULABLOCK_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
NEBULABLOCK_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
NEBULABLOCK_CHAT_MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506
```

### Default Models

- **Embedding**: `Qwen/Qwen3-Embedding-8B` - High-quality 4096-dimensional embeddings
- **Reranker**: `BAAI/bge-reranker-v2-m3` - State-of-the-art reranking model for superior relevance scoring
- **Chat**: `mistralai/Mistral-Small-3.2-24B-Instruct-2506` - Powerful instruction-following model

## 📁 Project Structure

```
rag-example/
├── nebularag/                    # Main package
│   ├── cli/                      # Command-line interface
│   │   └── main.py              # CLI entry point
│   ├── clients/                  # External API clients
│   │   └── nebula_client.py     # NebulaBlock API client
│   ├── config/                   # Configuration management
│   │   └── settings.py          # Environment settings
│   ├── core/                     # Core RAG components
│   │   ├── rag_pipeline.py      # Main RAG pipeline
│   │   └── vector_store.py      # In-memory vector store
│   └── utils/                    # Utility functions
│       ├── file_utils.py        # File operations
│       └── text_processing.py   # Text splitting utilities
├── tests/                        # Test suite
│   └── test_api.py              # API connectivity tests
├── examples/                     # Usage examples
│   └── basic_usage.py           # Programmatic usage example
├── docs/                         # Sample documents
│   └── sample.md                # Example markdown file
├── setup.py                      # Package configuration
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🚀 Usage

### Basic Usage

1. **Prepare your documents**: Place `.txt` or `.md` files in a directory (e.g., `docs/`)

2. **Set up environment**: Copy `.env.example` to `.env` and fill in your API credentials

3. **Run the RAG pipeline**:
   ```bash
   nebularag --docs docs --question "What does the guide say about X?"
   ```

### Advanced Usage

```bash
# Custom chunk size and overlap
nebularag \
  --docs docs \
  --question "Your question here" \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --top-k 15 \
  --rerank-k 8
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--docs` | Path to documents directory | Required |
| `--question` | Question to ask | Required |
| `--chunk-size` | Size of text chunks | 800 |
| `--chunk-overlap` | Overlap between chunks | 120 |
| `--top-k` | Number of candidates to retrieve | 12 |
| `--rerank-k` | Number of candidates after reranking | 6 |

### Programmatic Usage

```python
from nebularag import RAGPipeline, NebulaBlockClient, read_text_files
from nebularag.config import get_settings

# Initialize the RAG pipeline
client = NebulaBlockClient()
rag = RAGPipeline(
    client=client,
    chunk_size=800,
    chunk_overlap=120,
    top_k=12,
    rerank_k=6
)

# Load and index documents
docs = read_text_files('docs')
rag.index_texts(docs)

# Ask questions
result = rag.answer("What is the main topic?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} chunks")
```

### Testing the API

Test your NebulaBlock API connection:

```bash
python tests/test_api.py
```

## 🔧 How It Works

### RAG Pipeline Flow

1. **Document Processing**: 
   - Reads `.txt` and `.md` files from the specified directory
   - Splits documents into overlapping chunks (default: 800 chars, 120 overlap)

2. **Indexing**:
   - Generates embeddings for each chunk using Qwen/Qwen3-Embedding-8B
   - Stores embeddings in an in-memory vector store with cosine similarity

3. **Retrieval**:
   - Embeds the user question
   - Retrieves top-K most similar chunks by cosine similarity

4. **Reranking**:
   - Sends retrieved candidates to BAAI/bge-reranker-v2-m3
   - Reranks based on relevance to the question with superior accuracy
   - Selects top rerank-K candidates

5. **Generation**:
   - Combines reranked chunks as context
   - Sends context + question to Mistral-Small-3.2-24B-Instruct-2506
   - Returns the generated answer with source citations

### API Compatibility

The client assumes OpenAI/Cohere-like JSON structures but keeps endpoints configurable:

- **Embeddings**: `POST /embeddings` with `{"model": "...", "input": [...]}`
- **Reranking**: `POST /rerank` with `{"model": "...", "query": "...", "documents": [...]}`
- **Chat**: `POST /chat/completions` with `{"model": "...", "messages": [...]}`

### Advanced Features

- **Compression Support**: Automatically handles Brotli and Gzip compression
- **Cloudflare Bypass**: Uses browser-like headers to avoid security blocks
- **Error Handling**: Comprehensive error handling with retries and fallbacks
- **Unicode Support**: Robust text encoding with UTF-8 and Latin-1 fallbacks

## 🧪 Examples

### Example 1: Basic Question Answering

```bash
# With sample documents
nebularag \
  --docs docs \
  --question "What is the main topic of the documentation?"
```

### Example 2: Multi-Question Demo

```bash
# Run the comprehensive demo
python examples/basic_usage.py
```

### Example 3: Using OpenAI SDK (Alternative)

If you prefer the official OpenAI client:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["NEBULABLOCK_BASE_URL"],
    api_key=os.environ["NEBULABLOCK_API_KEY"]
)

# Embedding
response = client.embeddings.create(
    model=os.environ["NEBULABLOCK_EMBEDDING_MODEL"],
    input=["hello world"]
)

# Chat
response = client.chat.completions.create(
    model=os.environ["NEBULABLOCK_CHAT_MODEL"],
    messages=[{"role": "user", "content": "Hi"}]
)
```

## 🛠️ Development

### Running Tests

```bash
# Test API connectivity
python tests/test_api.py

# Test imports
python -c "from nebularag import NebulaBlockClient, RAGPipeline; print('Import successful!')"

# Run the full demo
python examples/basic_usage.py
```

### Adding New Features

1. **New Vector Store**: Implement the interface in `nebularag/core/vector_store.py`
2. **New Splitters**: Add functions to `nebularag/utils/text_processing.py`
3. **New Clients**: Extend `nebularag/clients/nebula_client.py` or create new client classes

## 📝 Notes

- This is a production-ready implementation with robust error handling
- For production use, consider:
  - Persistent vector databases (Pinecone, Weaviate, etc.)
  - Semantic chunking strategies
  - Caching mechanisms
  - Rate limiting and retry logic
- The reranker uses BAAI/bge-reranker-v2-m3 for superior performance
- All API calls are synchronous; async support can be added for better performance
- Compression and encoding issues are handled automatically

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure you've installed the package with `pip install -e .`
2. **API Key Error**: Verify your `NEBULABLOCK_API_KEY` is set correctly
3. **Connection Error**: Check your `NEBULABLOCK_BASE_URL` and internet connection
4. **Empty Results**: Ensure your documents directory contains `.txt` or `.md` files
5. **Compression Error**: Install Brotli with `pip install brotli>=1.0.9`
6. **Cloudflare Block**: The client automatically uses browser-like headers to bypass this

### Getting Help

- Check the [Issues](https://github.com/your-repo/issues) page
- Review the API documentation for NebulaBlock
- Test your API connection with `python tests/test_api.py`

## 🎯 Performance

- **Embedding Model**: Qwen/Qwen3-Embedding-8B provides 4096-dimensional embeddings
- **Reranker**: BAAI/bge-reranker-v2-m3 offers state-of-the-art relevance scoring
- **Chat Model**: Mistral-Small-3.2-24B-Instruct-2506 delivers high-quality responses
- **Vector Search**: Cosine similarity with in-memory storage for fast retrieval
- **Compression**: Automatic Brotli/Gzip handling for efficient data transfer