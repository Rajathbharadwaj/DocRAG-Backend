from dotenv import load_dotenv
import os
from typing import Dict
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class Settings:
    def __init__(self):
        load_dotenv(override=True, dotenv_path=".env")
        
        # API Keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY")
        
        # Model defaults - Anthropic as default
        self.default_provider: LLMProvider = LLMProvider.ANTHROPIC
        self.anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
        
        # Vector store settings
        self.vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "vector_store")
        
        # Crawler settings
        self.max_depth: int = int(os.getenv("MAX_DEPTH", "2"))
        self.backlink_threshold: float = float(os.getenv("BACKLINK_THRESHOLD", "0.3"))
        
        # Chunking settings
        self.chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        self.validate()
    
    def validate(self):
        """Validate required environment variables."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for embeddings")
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set as it is the default provider")

settings = Settings() 