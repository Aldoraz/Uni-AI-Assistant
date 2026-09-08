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

# How many relevant chunks to return in rag context search
RAG_CANDIDATE_K = 15
# How many relevant chunks to use in the final context for the LLM
RAG_TOP_K = 5
# How many additional relevant chunks to include in the context expansion
RAG_EXPANSION_RADIUS = 1
# Maximum FAISS distance accepted for retrieval; lower is more similar
RAG_MAX_DISTANCE = 1.2

# ---------- CONTEXT ----------
# Number of messages to keep in context for the LLM
LLM_CONTEXT_SIZE = 15
# Number of messages to keep in context for RAG retrieval
RAG_CONTEXT_SIZE = 3
