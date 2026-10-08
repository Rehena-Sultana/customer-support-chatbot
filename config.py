import os
from dotenv import load_dotenv

load_dotenv()

# Ollama LLM and Embedding Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:1.5b")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# Storage Paths
CHROMA_DIR = os.path.join("vectorstore", "chroma")
LOG_DB = os.path.join("logs", "support.db")
