from __future__ import annotations

import importlib.util
import pathlib
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
            self.assertAlmostEqual(sum(candidate["genome"]["dataset_mix"].values()), 1.0, places=6)

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

    def test_mutation_changes_genome_without_breaking_bounds(self) -> None:
        population = EVO.seed_population(self.cfg, 0)
        import random

        mutated = EVO.mutate_genome(population[0]["genome"], self.cfg, random.Random(7))
        self.assertAlmostEqual(sum(mutated["dataset_mix"].values()), 1.0, places=6)
        self.assertGreaterEqual(mutated["training"]["learning_rate"], 1e-6)
        self.assertLessEqual(mutated["training"]["learning_rate"], 2e-3)


if __name__ == "__main__":
    unittest.main()
