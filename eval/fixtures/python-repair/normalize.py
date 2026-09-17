from __future__ import annotations

from typing import Any


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a canonical document with deterministic provenance."""
    result = dict(document)
    sources = result.pop("sources", [])
    # Regression: preserves duplicates and reverses caller order instead of
    # returning canonical URL-sorted provenance records.
    result["provenance"] = {"sources": list(reversed(sources))}
    return result
