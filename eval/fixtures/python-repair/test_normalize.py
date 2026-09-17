from __future__ import annotations

import copy
import unittest

from normalize import normalize_document


class NormalizeDocumentTests(unittest.TestCase):
    def test_sources_are_deduplicated_and_sorted_by_url(self) -> None:
        source_a = {"url": "https://a.example/item", "title": "A"}
        source_b = {"url": "https://b.example/item", "title": "B"}
        document = {
            "id": "doc-1",
            "kind": "research",
            "sources": [source_b, source_a, dict(source_b)],
        }

        normalized = normalize_document(document)

        self.assertEqual(
            normalized["provenance"]["sources"],
            [source_a, source_b],
        )

    def test_input_is_not_mutated(self) -> None:
        document = {
            "id": "doc-2",
            "sources": [{"url": "https://example.test", "title": "Example"}],
        }
        before = copy.deepcopy(document)

        normalize_document(document)

        self.assertEqual(document, before)

    def test_missing_sources_becomes_empty_provenance(self) -> None:
        normalized = normalize_document({"id": "doc-3"})
        self.assertEqual(normalized["provenance"], {"sources": []})


if __name__ == "__main__":
    unittest.main()
