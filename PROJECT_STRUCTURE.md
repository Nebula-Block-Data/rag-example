# Project Structure

This document describes the restructured organization of the NebulaRAG project.

## 📁 Directory Structure

```
rag-example/
├── 📁 nebularag/                      # Main package
│   ├── 📁 cli/                        # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py                    # CLI entry point
│   ├── 📁 clients/                    # External API clients
│   │   ├── __init__.py
│   │   └── nebula_client.py           # NebulaBlock API client
│   ├── 📁 config/                     # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py                # Settings and environment handling
│   ├── 📁 core/                       # Core RAG components
│   │   ├── __init__.py
│   │   ├── rag_pipeline.py            # Main RAG pipeline
│   │   └── vector_store.py            # Vector storage and search
│   ├── 📁 utils/                      # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py              # File handling utilities
│   │   └── text_processing.py         # Text processing utilities
│   └── __init__.py                    # Package initialization
├── 📁 tests/                          # Test modules
│   ├── __init__.py
│   └── test_api.py                    # API testing script
├── 📁 examples/                       # Example scripts
│   ├── __init__.py
│   └── basic_usage.py                 # Basic usage example
├── 📁 docs/                           # Documentation
│   ├── sample.md                      # Sample markdown document
│   └── sample_ml_guide.pdf            # Sample PDF document
├── 📁 scripts/                        # Utility scripts
│   └── test_nebula.py                 # Legacy test script
├── .env.example                       # Environment template
├── .gitignore                         # Git ignore rules
├── LICENSE                            # License file
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
└── setup.py                           # Package configuration
```

## 🏗️ Architecture Overview

### **Core Components**
- **`nebularag.core`**: Contains the main RAG pipeline and vector store implementations
- **`nebularag.clients`**: Handles external API integrations (NebulaBlock)
- **`nebularag.config`**: Manages application settings and environment variables
- **`nebularag.utils`**: Provides common utility functions for file handling and text processing
- **`nebularag.cli`**: Command-line interface for the application

### **Key Benefits of New Structure**

1. **Separation of Concerns**: Each module has a clear, single responsibility
2. **Scalability**: Easy to add new clients, utilities, or core components
3. **Testability**: Clear separation makes unit testing easier
4. **Maintainability**: Logical organization makes code easier to understand and modify
5. **Reusability**: Components can be imported and used independently

### **Module Responsibilities**

#### **Core (`rag_example.core`)**
- `rag_pipeline.py`: Main RAG pipeline orchestration
- `vector_store.py`: Vector storage and similarity search

#### **Clients (`rag_example.clients`)**
- `nebula_client.py`: HTTP client for NebulaBlock API

#### **Configuration (`rag_example.config`)**
- `settings.py`: Environment variable handling and validation

#### **Utilities (`rag_example.utils`)**
- `file_utils.py`: File reading, validation, and PDF text extraction
- `text_processing.py`: Text splitting and processing

#### **CLI (`rag_example.cli`)**
- `main.py`: Command-line interface and argument parsing

## 🔄 Migration Changes

### **Import Changes**
```python
# Old imports
from rag_example.nebula_client import NebulaBlockClient
from rag_example.rag import RAGPipeline
from rag_example.store import InMemoryVectorStore
from rag_example.splitter import split_text

# New imports
from nebularag.clients import NebulaBlockClient
from nebularag.core import RAGPipeline, InMemoryVectorStore
from nebularag.utils import split_text
```

### **CLI Usage**
```bash
# Old way
python -m rag_example.main --docs docs --question "What is this about?"

# New way (both work)
python -m nebularag.cli.main --docs docs --question "What is this about?"
nebularag --docs docs --question "What is this about?"
```

## 🧪 Testing

The new structure includes:
- **`tests/`**: Dedicated test directory
- **`examples/`**: Example scripts for different use cases
- **Improved test organization**: Tests are now properly organized and can be run with pytest

## 📦 Package Installation

The package can be installed in development mode:
```bash
pip install -e .
```

This allows for:
- Easy development and testing
- Proper import resolution
- Console script installation (`nebularag` command)
