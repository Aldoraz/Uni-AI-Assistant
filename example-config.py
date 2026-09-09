from pathlib import Path

# ---------- Providers & Models ----------
# What LLM provider to use for chat completions
CHAT_PROVIDER  = "openai"
# What model to use for chat completions
CHAT_MODEL = "gpt-4.1"

# What embedding provider to use for embedding generation
# Options: "openai", "ollama"
EMBEDDING_PROVIDER  = "openai"

# What model to use for embedding generation
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------- Model Options ----------
# Base URL for Ollama API
OLLAMA_BASE_URL = "http://localhost:11434"
# Context window size for Ollama chat sessions; 8192 ~ 6.2GB VRAM
OLLAMA_CONTEXT_WINDOW = 8192
# Number of document chunks sent in each local embedding request
OLLAMA_EMBEDDING_BATCH_SIZE = 64
# Whether to enable reasoning mode for Ollama chat sessions
OLLAMA_REASONING_ENABLED = False
# Keep-alive duration for Ollama chat sessions; can be a string like "15m" or "1h"
OLLAMA_KEEP_ALIVE = "15m"

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

# ---------- TOOLS ----------
# Whether to expose the external Tavily web-search tool
WEB_SEARCH_ENABLED = False
# Maximum number of characters to include in the content returned by tools
TOOL_MAX_CONTENT_CHARS = 50_000
# Maximum number of web search results to return
WEB_SEARCH_MAX_RESULTS = 5
# Depth of web search; can be "basic" or "advanced"
WEB_SEARCH_DEPTH = "basic"
# Maximum number of tool calls allowed per user request
MAX_TOOL_CALLS = 5
