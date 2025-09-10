"""Application settings and configuration management."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not available, continue without it
    pass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # API Configuration
    nebula_base_url: str = "https://dev-llm-proxy.nebulablock.com/v1"
    nebula_api_key: Optional[str] = None
    
    # API Endpoints
    embeddings_path: str = "/embeddings"
    rerank_path: str = "/rerank"
    chat_path: str = "/chat/completions"
    
    # Model Configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    chat_model: str = "Mistral-Small-24B-Instruct-2501"
    
    # RAG Pipeline Configuration
    default_chunk_size: int = 800
    default_chunk_overlap: int = 120
    default_top_k: int = 12
    default_rerank_k: int = 6
    
    # HTTP Configuration
    timeout: float = 60.0
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls(
            nebula_base_url=os.environ.get("NEBULABLOCK_BASE_URL", cls.nebula_base_url),
            nebula_api_key=os.environ.get("NEBULABLOCK_API_KEY"),
            embeddings_path=os.environ.get("NEBULABLOCK_EMBEDDINGS_PATH", cls.embeddings_path),
            rerank_path=os.environ.get("NEBULABLOCK_RERANK_PATH", cls.rerank_path),
            chat_path=os.environ.get("NEBULABLOCK_CHAT_PATH", cls.chat_path),
            embedding_model=os.environ.get("NEBULABLOCK_EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.environ.get("NEBULABLOCK_RERANKER_MODEL", cls.reranker_model),
            chat_model=os.environ.get("NEBULABLOCK_CHAT_MODEL", cls.chat_model),
            default_chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", cls.default_chunk_size)),
            default_chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", cls.default_chunk_overlap)),
            default_top_k=int(os.environ.get("RAG_TOP_K", cls.default_top_k)),
            default_rerank_k=int(os.environ.get("RAG_RERANK_K", cls.default_rerank_k)),
            timeout=float(os.environ.get("HTTP_TIMEOUT", cls.timeout)),
        )
    
    def validate(self) -> None:
        """Validate settings and raise errors for missing required values."""
        if not self.nebula_api_key:
            raise ValueError(
                "NEBULABLOCK_API_KEY environment variable is required. "
                "Please set it in your .env file or environment."
            )
        
        if self.default_chunk_overlap >= self.default_chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.default_chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.default_chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        if self.default_top_k <= 0:
            raise ValueError("top_k must be positive")
        
        if self.default_rerank_k <= 0:
            raise ValueError("rerank_k must be positive")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance, creating it if necessary."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
        _settings.validate()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance (useful for testing)."""
    global _settings
    _settings = None
