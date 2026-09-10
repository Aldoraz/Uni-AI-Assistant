# Setup

This guide covers the shared application setup first, followed by provider-specific steps for OpenAI and Ollama. Chat and embedding providers can be selected independently.

## Prerequisites

- Python 3.11 is recommended. The code requires Python 3.10 or newer.
- Git.
- One chat provider and one embedding provider:
  - OpenAI API access, or
  - a running [Ollama](https://ollama.com/download) installation with suitable local models.
- A Tavily API key only if optional web search will be enabled.

## 1. Clone and install

```powershell
git clone https://github.com/Aldoraz/Uni-AI-Assistant.git
cd Uni-AI-Assistant
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## 2. Create local configuration files

The application imports settings from `config.py` and secrets from `.env`. Both files are excluded from Git.

```powershell
Copy-Item example-config.py config.py
Copy-Item example.env .env
```

On macOS or Linux:

```bash
cp example-config.py config.py
cp example.env .env
```


## 3. Choose providers

### Option A: OpenAI for chat and embeddings

Add the API key to `.env`:

```dotenv
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=
```

Use these values in `config.py`:

```python
CHAT_PROVIDER = "openai"
CHAT_MODEL = "gpt-4.1"

EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = "text-embedding-3-small"
```

Document chunks are sent to OpenAI when the index is built, and chat prompts—including retrieved context—are sent to OpenAI when answering.

The model names are configurable. Other OpenAI chat models can be used if they support tool calling, and other OpenAI embedding models can be used if they are compatible with the provider API.

### Option B: Ollama for local chat and embeddings

Install Ollama, make sure its service is running, and download the tested models:

```powershell
ollama pull qwen3:8b
ollama pull embeddinggemma
```

Configure `config.py`:

```python
CHAT_PROVIDER = "ollama"
CHAT_MODEL = "qwen3:8b"

EMBEDDING_PROVIDER = "ollama"
EMBEDDING_MODEL = "embeddinggemma"

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_CONTEXT_WINDOW = 8192
OLLAMA_EMBEDDING_BATCH_SIZE = 64
OLLAMA_REASONING_ENABLED = False
OLLAMA_KEEP_ALIVE = "15m"
```

The model names are configurable. Other Ollama chat models can be used if they support compatible tool calling, and other Ollama embedding models can be used if they are exposed through Ollama's embedding API.

### Mixed providers

`CHAT_PROVIDER` and `EMBEDDING_PROVIDER` are independent, so OpenAI/Ollama combinations also work. Remember that the embedding choice determines where document text is processed, while the chat choice determines where prompts and retrieved context are processed.

## 4. Select a document directory

By default, documents are read from `data/documents`:

```python
DATA_DIR = Path("data")
DOCUMENTS_PATH = DATA_DIR / "documents"
INDEX_PATH = DATA_DIR / "index"
```

Create the default directory and copy PDFs or UTF-8 text files into it:

```powershell
New-Item -ItemType Directory -Force data\documents
```

An absolute path can be used instead:

```python
DOCUMENTS_PATH = Path("D:/Knowledge/Documents")
```

The loader searches the directory recursively.

## 5. Optional web search

Web search is disabled by default. To enable it, set both the configuration flag and API key:

```python
WEB_SEARCH_ENABLED = True
```

```dotenv
TAVILY_API_KEY=your-key-here
```

If the flag is false or the key is blank, the tool is not exposed to the model. Enabling it sends model-generated search queries to Tavily and permits results from the public web to enter the model context.

## 6. Start and index

Start Streamlit from the repository root:

```powershell
streamlit run app.py
```

Then:

1. Open **Indexed documents** in the sidebar.
2. Select **Update Index**.
3. Review the loaded, skipped, and failed counts.
4. Expand individual document entries to inspect failures.
5. Ask a question in the chat.

Index updates are manual: changes in the document directory are processed only after **Update Index** is selected. The first run embeds all supported documents. Later runs hash the current files and rebuild only records affected by file or indexing-configuration changes. Changing the embedding provider or model selects a separate provider-specific index directory and requires another manual index update before retrieval.

## 7. Verify the installation

Run the automated test suite:

```powershell
python -m unittest discover -s tests -v
```

For a smoke test, verify that:

- a document appears as succeeded in the index status;
- a question about that document returns a relevant answer;
- a follow-up such as “What is its conclusion?” is understood in context;
- a conversation remains available after restarting Streamlit;
- the console shows the rewrite, candidate filtering, reranking, and context-expansion stages;
- if enabled, a current-information question triggers `web_search` in the log.

## Troubleshooting

### Ollama model validation fails

Check that the service is running, `OLLAMA_BASE_URL` points to it, and the configured model appears in `ollama ls`. Model names, including tags such as `:8b`, must match.

### Vector store is not initialized

The selected embedding provider/model combination has no built index. Open **Indexed documents** and select **Update Index**.

### The FAISS files are inconsistent

Each index requires both `index.faiss` and `index.pkl`. Back up the affected provider/model directory below `data/index`, remove the incomplete pair, and rebuild it from the sidebar.

### A PDF is failed or empty

Inspect the error in the index-status expander. Encrypted, malformed, or image-only PDFs may not contain extractable text. OCR is not currently built into the application.

### Web search does not appear

Confirm that `WEB_SEARCH_ENABLED` is `True`, `TAVILY_API_KEY` is populated in `.env`, and Streamlit was restarted after the change. A startup log states why web search was omitted.
