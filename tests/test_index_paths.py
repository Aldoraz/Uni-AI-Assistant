# AI-generated with OpenAI Codex

import unittest
from pathlib import Path

from rag.index_paths import get_index_directory


class IndexPathTests(unittest.TestCase):
    def test_builds_provider_and_model_specific_directory(self):
        result = get_index_directory(
            Path("data/index"),
            "ollama",
            "nomic-embed-text:latest",
        )

        self.assertEqual(
            result,
            Path("data/index/ollama--nomic-embed-text_latest"),
        )

    def test_sanitizes_path_separators(self):
        result = get_index_directory(
            Path("data/index"),
            "local/provider",
            "organization/model",
        )

        self.assertEqual(
            result,
            Path("data/index/local_provider--organization_model"),
        )

    def test_rejects_empty_components(self):
        with self.assertRaises(ValueError):
            get_index_directory(Path("data/index"), "ollama", "///")


if __name__ == "__main__":
    unittest.main()
