import re
from pathlib import Path


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not sanitized:
        raise ValueError("Index path components must contain a valid character")
    return sanitized


def get_index_directory(
    index_root: Path,
    embedding_provider: str,
    embedding_model: str,
) -> Path:
    provider = _sanitize_path_component(embedding_provider)
    model = _sanitize_path_component(embedding_model)
    return Path(index_root) / f"{provider}--{model}"
