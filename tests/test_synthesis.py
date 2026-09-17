from __future__ import annotations

import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthesize_brave", ROOT / "scripts" / "synthesize_brave.py")
assert SPEC and SPEC.loader
SYNTH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNTH)


class SynthesisTests(unittest.TestCase):
    def test_extract_grounding_and_source_metadata(self) -> None:
        response = {
            "grounding": {
                "generic": [
                    {
                        "url": "https://example.test/spec",
                        "title": "Example Spec",
                        "snippets": ["A useful public snippet."],
                    }
                ]
            },
            "sources": {
                "https://example.test/spec": {
                    "hostname": "example.test",
                    "title": "Example Spec",
                }
            },
        }
        grounding, sources = SYNTH.extract_grounding(response)
        metadata = SYNTH.source_metadata(grounding, sources)
        self.assertEqual(len(grounding), 1)
        self.assertEqual(metadata[0]["hostname"], "example.test")

    def test_normalize_keeps_provenance_not_raw_brave_context(self) -> None:
        raw = {
            "task": "starintel-tool-design",
            "messages": [
                {"role": "user", "content": "Design a bounded ingest inspector."},
                {"role": "assistant", "content": "Use the canonical ingest boundary and preserve provenance."},
            ],
            "tools": [],
            "metadata": {"quality_checks": ["schema-valid"]},
        }
        row = SYNTH.normalize_example(
            raw,
            family="starintel-tool-design",
            topic="StarIntel ingest",
            query="StarIntel ingest docs",
            source_meta=[{"url": "https://example.test/spec", "title": "Spec", "hostname": "example.test"}],
            context_hash="a" * 64,
            teacher_model="teacher/test",
            prompt_version="test-v1",
        )
        self.assertEqual(row["task"], "starintel-tool-design")
        self.assertEqual(row["provenance"]["brave_context_sha256"], "a" * 64)
        self.assertEqual(row["provenance"]["sources"][0]["url"], "https://example.test/spec")
        self.assertNotIn("snippets", row["provenance"]["sources"][0])

    def test_secret_like_teacher_output_is_rejected(self) -> None:
        raw = {
            "task": "document-generation",
            "messages": [
                {"role": "user", "content": "Write a runbook."},
                {"role": "assistant", "content": "api_key=abcdefghijklmnopqrstuvwxyz"},
            ],
        }
        with self.assertRaises(ValueError):
            SYNTH.normalize_example(
                raw,
                family="document-generation",
                topic="runbook",
                query="runbook documentation",
                source_meta=[{"url": "https://example.test", "title": "Docs", "hostname": "example.test"}],
                context_hash="b" * 64,
                teacher_model="teacher/test",
                prompt_version="test-v1",
            )

    def test_all_configured_families_are_known(self) -> None:
        cfg = SYNTH.read_toml(ROOT / "config" / "synthesis.toml")
        jobs = list(SYNTH.iter_seed_jobs(cfg, None))
        self.assertTrue(jobs)
        for _topic, family in jobs:
            self.assertIn(family, SYNTH.ALLOWED_TASKS)


if __name__ == "__main__":
    unittest.main()
