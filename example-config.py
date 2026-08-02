from pathlib import Path

# ---------- Models ----------

CHAT_MODEL = "gpt-4.1"

# ---------- Data ----------

DATA_DIR = Path("data")

# Path to the directory containing the documents to be indexed
DOCUMENTS_PATH = DATA_DIR / "documents"

# Alternative: absolute path to the documents directory
#DOCUMENTS_PATH = Path("C:/path/to/your/documents")

INDEX_PATH = DATA_DIR / "index"
