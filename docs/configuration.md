# Configuration reference

Runtime settings live in `config.py`. Start by copying `example-config.py`.

Because `config.py` is imported as Python code, string values need quotes and filesystem paths should use `pathlib.Path`. Restart the Streamlit process after changing settings. Streamlit's resource cache otherwise keeps the previously constructed model and vector-store objects alive.

## Environment variables

Secrets are loaded from `.env` with `python-dotenv`.

| Variable | Required when | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | An OpenAI chat or embedding provider is selected | Authenticates OpenAI model requests |
| `TAVILY_API_KEY` | `WEB_SEARCH_ENABLED` is `True` | Authenticates Tavily web search |

## Providers and models

| Setting | Example default | Meaning |
|---|---:|---|
| `CHAT_PROVIDER` | `"openai"` | Chat implementation. Supported values are `"openai"` and `"ollama"`. |
| `CHAT_MODEL` | `"gpt-4.1"` | Model name passed to the selected chat provider. |
| `EMBEDDING_PROVIDER` | `"openai"` | Embedding implementation. Supported values are `"openai"` and `"ollama"`. |
| `EMBEDDING_MODEL` | `"text-embedding-3-small"` | Model name passed to the selected embedding provider. |

> **Tested local models:** `qwen3:8b` for chat and `embeddinggemma` for embeddings.

The model names are not hardcoded. Other chat models can be used when the selected provider supports them and they implement compatible tool calling. Other embedding models can be used when they are available through the selected embedding provider.

The providers can be mixed. A chat-model change does not require reindexing. An embedding-provider or embedding-model change selects a different index directory and therefore requires an index build for that combination.

The configured chat model must support the tool-calling format if document reading or web search should be used. Tool availability does not guarantee that every model will select tools reliably.

## Ollama options

These settings are used only when the corresponding Ollama provider is selected.

| Setting | Example default | Meaning and tradeoff |
|---|---:|---|
| `OLLAMA_BASE_URL` | `"http://localhost:11434"` | Base URL of the Ollama service. It can point to another host, but doing so changes the local-data boundary. |
| `OLLAMA_CONTEXT_WINDOW` | `8192` | Maximum context passed to an Ollama chat session. Larger values allow more context and require more memory. |
| `OLLAMA_EMBEDDING_BATCH_SIZE` | `64` | Number of chunk texts submitted in one embedding request. Reduce it if large index builds exhaust memory or fail. Must be positive. |
| `OLLAMA_REASONING_ENABLED` | `False` | Enables the provider's reasoning mode where supported. Disabled by default to keep latency and visible output predictable. |
| `OLLAMA_KEEP_ALIVE` | `"15m"` | How long Ollama should keep the chat model loaded after a request. Longer values improve repeat latency while retaining memory usage. |

## Paths and indexing

| Setting | Example default | Meaning |
|---|---:|---|
| `DATA_DIR` | `Path("data")` | Convenience root used by the example paths. |
| `DOCUMENTS_PATH` | `DATA_DIR / "documents"` | Directory searched recursively for source documents. |
| `INDEX_PATH` | `DATA_DIR / "index"` | Root directory for generated vector indexes and their catalogues. |
| `CHUNK_SIZE` | `1000` | Maximum text size targeted by the recursive text splitter. |
| `CHUNK_OVERLAP` | `200` | Text overlap between adjacent chunks to reduce boundary information loss. |

Supported source types are `.pdf` and UTF-8 `.txt`. Filenames and source paths are stored as metadata on each chunk.

The application does not watch the document directory continuously. After adding, changing, or deleting files, select **Update Index** in the Streamlit sidebar to process those changes.

Changing chunk size, overlap, embedding provider, or embedding model changes the index configuration signature. Documents in the selected catalogue are then considered stale and rebuilt on the next index update.

Indexes are isolated by a sanitized provider/model name:

```text
data/
  dbs/
    conversations.db
  index/
    openai--text-embedding-3-small/
      catalog.db
      index.faiss
      index.pkl
    ollama--embeddinggemma/
      catalog.db
      index.faiss
      index.pkl
```

This layout prevents vectors produced by incompatible embedding models from being loaded together. All generated files below `data/` are excluded from Git.

## Retrieval

| Setting | Example default | Meaning and tradeoff |
|---|---:|---|
| `RAG_CANDIDATE_K` | `15` | Candidate limit requested from FAISS before filtering and reranking. More candidates can improve recall but increase reranking cost. |
| `RAG_TOP_K` | `5` | Maximum number of candidates selected by reranking under normal retrieval. The reranker may intentionally return fewer or none. |
| `RAG_EXPANSION_RADIUS` | `1` | Number of adjacent chunks requested on each side of every reranked chunk. `0` disables expansion. |
| `RAG_MAX_DISTANCE` | `1.2` | Maximum FAISS distance retained before reranking. Lower distances are treated as more similar. |

The distance scale depends on the embedding model and corpus. `RAG_MAX_DISTANCE` therefore needs empirical tuning after changing embeddings; `1.2` is not a provider-independent confidence percentage.

Context expansion can make the final number of chunks larger than `RAG_TOP_K`. Duplicate neighbors are removed, and missing boundary chunks are ignored.

### Retrieval sequence

1. The chat model rewrites the current request using recent history.
2. FAISS returns up to `max(RAG_CANDIDATE_K, RAG_TOP_K)` nearest chunks.
3. Candidates beyond `RAG_MAX_DISTANCE` are removed.
4. The chat model selects up to `RAG_TOP_K` useful candidates.
5. Neighboring chunks are loaded according to `RAG_EXPANSION_RADIUS`.

If rewriting fails or returns blank text, the original user query is used. If reranking returns invalid output or raises an error, vector order is preserved and truncated to the requested limit. An intentionally empty reranking result remains empty.

For example, with a candidate limit of 12, FAISS can return up to 12 chunks. The distance threshold might retain 8, and the reranker might select 3. With an expansion radius of 1, each selected chunk contributes itself and at most one neighbor on either side, producing at most 9 final chunks—and fewer when selections overlap or lie at document boundaries.

## Conversation context

| Setting | Example default | Meaning |
|---|---:|---|
| `LLM_CONTEXT_SIZE` | `15` | Maximum number of stored user/assistant messages included for answer generation. |
| `RAG_CONTEXT_SIZE` | `3` | Maximum number of prior stored messages supplied to query rewriting. |

These are message counts, not token limits. The current user message is added once after retrieval and appears once in the answer-generation history. Intermediate assistant tool requests and tool results are used within the active turn but are not stored in SQLite.

## Tools

| Setting | Example default | Meaning and tradeoff |
|---|---:|---|
| `WEB_SEARCH_ENABLED` | `False` | Exposes Tavily search only when true and `TAVILY_API_KEY` is also present. |
| `TOOL_MAX_CONTENT_CHARS` | `50_000` | Maximum formatted character count returned to the model by document reading or web search. Longer output is truncated. |
| `WEB_SEARCH_MAX_RESULTS` | `5` | Maximum result entries requested from Tavily. |
| `WEB_SEARCH_DEPTH` | `"basic"` | Tavily search depth; supported configured values are `"basic"` and `"advanced"`. Advanced search can take longer and consume more service quota. |
| `MAX_TOOL_CALLS` | `5` | Maximum tool calls allowed for one user turn before the model must answer from collected information. |

The `read_document` tool is always registered. It accepts an exact indexed filename, loads the source again, formats its pages, and applies `TOOL_MAX_CONTENT_CHARS`. It cannot read arbitrary filesystem paths supplied by the model.

## Logging and prompt inspection

Application logs use the `INFO` level for significant events such as:

- provider and index initialization;
- discovered, changed, failed, and removed documents;
- rewritten retrieval queries and vector distances;
- distance filtering, reranking, and context expansion;
- tool registration, requests, and summarized results;
- persisted conversation turns.

`prompt_debug.txt` is overwritten before each model call in the assistant tool loop. It can contain conversation messages, retrieved document excerpts, tool definitions, and tool output. The file is excluded from Git, but it is still sensitive local data. Disable or remove this diagnostic behavior before using the application in an environment where full prompt retention is prohibited.
