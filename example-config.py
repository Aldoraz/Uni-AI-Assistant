from pathlib import Path

# ---------- Providers & Models ----------
# What LLM provider to use for chat completions
# Options: "openai", "ollama"
CHAT_PROVIDER  = "openai"
# What model to use for chat completions
# Options depend on the provider / downloaded model.
CHAT_MODEL = "gpt-4.1"

# What embedding provider to use for embedding generation
# Options: "openai", "ollama"
EMBEDDING_PROVIDER  = "openai"

# What model to use for embedding generation
# Options depend on the provider / downloaded model.
EMBEDDING_MODEL = "text-embedding-3-small"


# ---------- RAG ----------
DATA_DIR = Path("data")

# Path to the directory containing the documents to be indexed
DOCUMENTS_PATH = DATA_DIR / "documents"

# Alternative: absolute path to a documents directory
#DOCUMENTS_PATH = Path("C:/path/to/your/documents")

# Path to the directory where the index will be stored
INDEX_PATH = DATA_DIR / "index"

# Chunking parameters for document splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
