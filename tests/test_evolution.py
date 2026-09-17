from __future__ import annotations

import importlib.util
import pathlib
import random
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("evolve_finetune", ROOT / "scripts" / "evolve_finetune.py")
assert SPEC and SPEC.loader
EVO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVO)


class EvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = EVO.read_toml(ROOT / "config" / "evolution.toml")

    def test_seed_population_is_deterministic_and_normalized(self) -> None:
        left = EVO.seed_population(self.cfg, 0)
        right = EVO.seed_population(self.cfg, 0)
        self.assertEqual(left, right)
        self.assertEqual(len(left), int(self.cfg["population"]))
        for candidate in left:
            self.assertAlmostEqual(sum(candidate["genome"]["dataset_mix"].values()), 1.0, places=8)

    def test_quality_gates_prevent_junk_artifact_spam(self) -> None:
        row = {
            "candidate_id": "bad",
            "task_family": "independent-research",
            "accepted": True,
            "artifact_count": 50,
            "quality": 0.95,
            "grounded": False,
            "independent": True,
        }
        self.assertEqual(EVO.accepted_artifacts(row, self.cfg), 0)

    def test_research_document_actor_candidate_wins(self) -> None:
        rows = []
        for candidate_id, good in (("producer", True), ("chatter", False)):
            rows.extend(
                [
                    {
                        "candidate_id": candidate_id,
                        "task_family": "independent-research",
                        "accepted": good,
                        "quality": 0.9 if good else 0.4,
                        "grounded": good,
                        "independent": good,
                    },
                    {
                        "candidate_id": candidate_id,
                        "task_family": "document-generation",
                        "accepted": good,
                        "quality": 0.9 if good else 0.4,
                    },
                    {
                        "candidate_id": candidate_id,
                        "task_family": "actor-system-generation",
                        "accepted": good,
                        "quality": 0.9 if good else 0.4,
                        "tests_passed": good,
                    },
                    {
                        "candidate_id": candidate_id,
                        "task_family": "tool-calling",
                        "accepted": True,
                        "quality": 0.9,
                        "tool_correct": good,
                    },
                    {
                        "candidate_id": candidate_id,
                        "task_family": "ingest-jobs",
                        "accepted": True,
                        "quality": 0.9,
                        "ingest_success": good,
                    },
                ]
            )
        scored = EVO.score_rows(rows, self.cfg)
        self.assertGreater(scored["producer"]["fitness"], scored["chatter"]["fitness"])
        self.assertEqual(EVO.pareto_front(scored), ["producer"])

    def test_tool_and_skill_design_count_as_document_artifacts(self) -> None:
        for family in ("osint-tool-design", "osint-skill-design", "starintel-tool-design"):
            self.assertEqual(EVO.family({"task_family": family}), "document")

    def test_mutation_changes_genome_without_breaking_bounds(self) -> None:
        population = EVO.seed_population(self.cfg, 0)
        mutated = EVO.mutate_genome(population[0]["genome"], self.cfg, random.Random(7))
        self.assertAlmostEqual(sum(mutated["dataset_mix"].values()), 1.0, places=8)
        self.assertGreaterEqual(mutated["training"]["learning_rate"], 1e-6)
        self.assertLessEqual(mutated["training"]["learning_rate"], 2e-3)

    def test_wire_mix_uses_training_task_names(self) -> None:
        wired = EVO.wire_dataset_mix({"osint_skill_design": 0.5, "tool_calling": 0.5})
        self.assertEqual(wired, {"osint-skill-design": 0.5, "tool-calling": 0.5})

    def test_starintel_archive_is_canonical_document(self) -> None:
        score_doc = {
            "pareto_front": ["candidate-a"],
            "candidates": {
                "candidate-a": {
                    "fitness": 1.0,
                    "metrics": {},
                    "accepted_artifacts": {"research": 1},
                    "failures": {},
                }
            },
        }
        docs = EVO.starintel_result_documents(score_doc, self.cfg)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["dtype"], "document")
        self.assertEqual(docs[0]["schema_version"], self.cfg["services"]["starintel_schema_version"])
        self.assertEqual(docs[0]["data"]["document_kind"], "fine-tune-benchmark")
        self.assertEqual(docs[0]["data"]["format"], "application/json")


if __name__ == "__main__":
    unittest.main()
