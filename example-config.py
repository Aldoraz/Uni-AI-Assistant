from pathlib import Path

# ---------- Providers and models ----------
# Supported providers are "openai" and "ollama"; chat and embeddings may use
# different providers.
CHAT_PROVIDER = "ollama"
CHAT_MODEL = "qwen3:8b"

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "embeddinggemma"

# ---------- Ollama ----------
# Change this only when Ollama is served from another host or port.
OLLAMA_BASE_URL = "http://localhost:11434"

# Larger context windows retain more conversation and retrieved text but consume
# more memory. Qwen3 8B used about 6.2 GB of VRAM with this tested setting.
OLLAMA_CONTEXT_WINDOW = 8192

# Batching avoids overloading Ollama when indexing a large document collection.
# Reduce this value if the embedding server rejects requests or runs out of memory.
OLLAMA_EMBEDDING_BATCH_SIZE = 64

# Reasoning is disabled to reduce latency for the interactive assistant workflow.
OLLAMA_REASONING_ENABLED = False

# Keeping the model loaded avoids a cold start on consecutive requests.
# Ollama accepts durations such as "15m" and "1h".
OLLAMA_KEEP_ALIVE = "15m"

# ---------- RAG ----------
DATA_DIR = Path("data")
DOCUMENTS_PATH = DATA_DIR / "documents"

# Use an absolute path when documents live outside the project directory.
# DOCUMENTS_PATH = Path("C:/path/to/your/documents")

# Each embedding provider/model combination receives its own subdirectory here,
# preventing incompatible vectors from being loaded together.
INDEX_PATH = DATA_DIR / "index"

# Overlap preserves context across chunk boundaries at the cost of more vectors.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieve a broad candidate set before the LLM reranks it.
RAG_CANDIDATE_K = 15

# This limits the seed chunks selected by reranking. Context expansion may make
# the final document list larger.
RAG_TOP_K = 5

# Neighboring chunks restore context that may have been split at chunk boundaries.
RAG_EXPANSION_RADIUS = 1

# FAISS returns distance rather than similarity, so lower values are stricter.
RAG_MAX_DISTANCE = 1.2

# ---------- Conversation context ----------
# Answer generation benefits from broader history than retrieval rewriting.
LLM_CONTEXT_SIZE = 15
RAG_CONTEXT_SIZE = 3

# ---------- Tools ----------
# Web search is opt-in because queries leave the local system and require Tavily.
WEB_SEARCH_ENABLED = False

# Cap tool output so a full document or search result cannot overwhelm the model
# context window.
TOOL_MAX_CONTENT_CHARS = 50_000
WEB_SEARCH_MAX_RESULTS = 5

# Tavily supports "basic" and "advanced"; advanced search is slower and may use
# more API credits.
WEB_SEARCH_DEPTH = "basic"

# The limit prevents a model from entering an unbounded tool-calling loop.
MAX_TOOL_CALLS = 5