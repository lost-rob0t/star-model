#!/usr/bin/env python3
"""Evolve StarIntel fine-tuning candidates against artifact-producing benchmarks.

The harness keeps the expensive part outside this process: llm.starintel.actor owns
fine-tuning and inference. This program creates candidate genomes and requests,
scores benchmark traces, selects Pareto winners, mutates the next generation, and
archives generation summaries through StarIntel's canonical document ingest API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import sys
import tomllib
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Iterable


PRIMARY_METRICS = (
    "research_yield",
    "document_yield",
    "actor_system_yield",
    "tool_call_accuracy",
    "ingest_success_rate",
)

TASK_ALIASES = {
    "research": "research",
    "independent-research": "research",
    "independent_research": "research",
    "document": "document",
    "documents": "document",
    "document-generation": "document",
    "document_generation": "document",
    "actor": "actor",
    "actor-system": "actor",
    "actor-system-generation": "actor",
    "actor_system_generation": "actor",
    "tool-calling": "tool",
    "tool_calling": "tool",
    "tool": "tool",
    "ingest": "ingest",
    "ingest-jobs": "ingest",
    "ingest_jobs": "ingest",
}


def read_toml(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).open("rb") as fh:
        return tomllib.load(fh)


def read_json(path: str | pathlib.Path) -> Any:
    return json.loads(pathlib.Path(path).read_text())


def write_json(path: str | pathlib.Path, value: Any) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: str | pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    return rows


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize(weights: dict[str, float]) -> dict[str, float]:
    clean = {k: max(1e-6, float(v)) for k, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        raise ValueError("dataset mix must have positive total weight")
    return {k: round(v / total, 8) for k, v in clean.items()}


def candidate_id(generation: int, parents: list[str], genome: dict[str, Any]) -> str:
    body = {"generation": generation, "parents": parents, "genome": genome}
    return f"star-evo-g{generation:03d}-{digest(body)[:12]}"


def base_genome(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_mix": normalize(cfg["dataset_mix"]),
        "training": dict(cfg["training"]),
        "router": dict(cfg.get("router", {})),
    }


def mutate_genome(
    genome: dict[str, Any],
    cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    out = json.loads(json.dumps(genome))
    mutation = cfg["mutation"]

    sigma = float(mutation["dataset_weight_sigma"])
    out["dataset_mix"] = normalize(
        {
            key: float(value) * math.exp(rng.gauss(0.0, sigma))
            for key, value in out["dataset_mix"].items()
        }
    )

    training = out["training"]
    lr_sigma = float(mutation["learning_rate_sigma"])
    training["learning_rate"] = round(
        clamp(float(training["learning_rate"]) * math.exp(rng.gauss(0.0, lr_sigma)), 1e-6, 2e-3),
        9,
    )
    training["lora_rank"] = rng.choice(list(mutation["lora_rank_choices"]))
    training["lora_alpha"] = rng.choice(list(mutation["lora_alpha_choices"]))
    training["lora_dropout"] = round(
        clamp(
            float(training["lora_dropout"]) + rng.gauss(0.0, 0.02),
            float(mutation["dropout_min"]),
            float(mutation["dropout_max"]),
        ),
        4,
    )
    training["warmup_ratio"] = round(
        clamp(
            float(training["warmup_ratio"]) + rng.gauss(0.0, 0.015),
            float(mutation["warmup_min"]),
            float(mutation["warmup_max"]),
        ),
        4,
    )

    router = out.get("router", {})
    if router.get("enabled"):
        router["hidden_dim"] = rng.choice([64, 96, 128, 192, 256])
        router["learning_rate"] = round(
            clamp(float(router.get("learning_rate", 1e-3)) * math.exp(rng.gauss(0.0, 0.25)), 1e-5, 1e-2),
            8,
        )
    return out


def make_candidate(generation: int, parents: list[str], genome: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "star-model-evolution-candidate/v1",
        "candidate_id": candidate_id(generation, parents, genome),
        "generation": generation,
        "parents": parents,
        "genome": genome,
    }


def seed_population(cfg: dict[str, Any], generation: int) -> list[dict[str, Any]]:
    count = int(cfg["population"])
    rng = random.Random(int(cfg["seed"]) + generation)
    base = base_genome(cfg)
    population = [make_candidate(generation, [], base)]
    while len(population) < count:
        population.append(make_candidate(generation, [], mutate_genome(base, cfg, rng)))
    return population


def load_candidates(directory: str | pathlib.Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(pathlib.Path(directory).glob("candidate-*.json")):
        candidate = read_json(path)
        out[candidate["candidate_id"]] = candidate
    if not out:
        raise ValueError(f"no candidate-*.json files in {directory}")
    return out


def cmd_seed(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    population = seed_population(cfg, args.generation)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for candidate in population:
        write_json(out / f"candidate-{candidate['candidate_id']}.json", candidate)
    print(json.dumps({"generation": args.generation, "candidates": len(population), "output_dir": str(out)}))


def fine_tune_request(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    base_model: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    genome = candidate["genome"]
    training = dict(genome["training"])
    training["base_model"] = base_model
    return {
        "type": "llm.fine_tune.start",
        "version": 1,
        "dataset": manifest,
        "training": training,
        "dataset_mix": genome["dataset_mix"],
        "router": genome.get("router", {}),
        "evolution": {
            "candidate_id": candidate["candidate_id"],
            "generation": candidate["generation"],
            "parents": candidate["parents"],
            "benchmark_priority": [
                "independent-research",
                "document-generation",
                "actor-system-generation",
                "tool-calling",
                "ingest-jobs",
            ],
        },
        "provider_policy": {
            "gpu": "cheapest-compatible",
            "max_hourly_gpu_usd": float(getattr(args_namespace, "max_hourly_gpu_usd", 2.0)),
            "max_total_gpu_usd": float(getattr(args_namespace, "max_total_gpu_usd", 20.0)),
        },
    }


# argparse namespace is only used to keep provider budgets consistent across request files.
args_namespace = argparse.Namespace(max_hourly_gpu_usd=2.0, max_total_gpu_usd=20.0)


def cmd_requests(args: argparse.Namespace) -> None:
    global args_namespace
    args_namespace = args
    cfg = read_toml(args.config)
    manifest = read_json(args.manifest)
    candidates = load_candidates(args.candidate_dir)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for candidate in candidates.values():
        request = fine_tune_request(candidate, manifest, args.base_model, cfg)
        write_json(out / f"fine-tune-{candidate['candidate_id']}.json", request)
    print(json.dumps({"requests": len(candidates), "output_dir": str(out)}))


def http_post_json(url: str, token: str | None, value: Any, timeout: int = 120) -> Any:
    body = canonical_json(value).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url}: HTTP {exc.code}: {detail}") from exc
    return json.loads(payload) if payload.strip() else {}


def cmd_submit(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    base = args.actor_url.rstrip("/")
    endpoint = cfg["services"]["training_endpoint"]
    token = args.token or os.getenv("STAR_LLM_ACTOR_TOKEN")
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(pathlib.Path(args.request_dir).glob("fine-tune-*.json")):
        payload = read_json(path)
        response = http_post_json(base + endpoint, token, payload, timeout=args.timeout)
        write_json(out / f"response-{payload['evolution']['candidate_id']}.json", response)
        count += 1
    print(json.dumps({"submitted": count, "output_dir": str(out)}))


def family(row: dict[str, Any]) -> str:
    raw = str(row.get("task_family", row.get("task", ""))).strip().lower()
    return TASK_ALIASES.get(raw, raw)


def accepted_artifacts(row: dict[str, Any], cfg: dict[str, Any]) -> int:
    if not bool(row.get("accepted", False)):
        return 0
    if float(row.get("quality", 1.0)) < float(cfg["gates"]["min_quality"]):
        return 0

    task = family(row)
    if task == "research":
        if cfg["gates"].get("require_research_grounding", True) and not bool(row.get("grounded", False)):
            return 0
        if cfg["gates"].get("require_research_independent", True) and not bool(row.get("independent", False)):
            return 0
    if task == "actor" and cfg["gates"].get("require_actor_tests", True):
        if not bool(row.get("tests_passed", False)):
            return 0
    return max(0, int(row.get("artifact_count", 1)))


def safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def score_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = str(row.get("candidate_id", "")).strip()
        if not candidate:
            raise ValueError("benchmark result missing candidate_id")
        by_candidate[candidate].append(row)

    scored: dict[str, Any] = {}
    weights = cfg["objectives"]
    for candidate_id_value, candidate_rows in sorted(by_candidate.items()):
        tasks = Counter(family(row) for row in candidate_rows)
        accepted = Counter()
        tool_total = tool_correct = 0
        ingest_total = ingest_ok = 0
        costs: list[float] = []
        latencies: list[float] = []
        failures = Counter()

        for row in candidate_rows:
            task = family(row)
            count = accepted_artifacts(row, cfg)
            if count:
                accepted[task] += count
            else:
                failures[task] += 1
            if task == "tool":
                tool_total += 1
                tool_correct += int(bool(row.get("tool_correct", False)))
            if task == "ingest":
                ingest_total += 1
                ingest_ok += int(bool(row.get("ingest_success", row.get("accepted", False))))
            if "cost_usd" in row:
                costs.append(float(row["cost_usd"]))
            if "latency_ms" in row:
                latencies.append(float(row["latency_ms"]))

        metrics = {
            "research_yield": safe_ratio(accepted["research"], tasks["research"]),
            "document_yield": safe_ratio(accepted["document"], tasks["document"]),
            "actor_system_yield": safe_ratio(accepted["actor"], tasks["actor"]),
            "tool_call_accuracy": safe_ratio(tool_correct, tool_total),
            "ingest_success_rate": safe_ratio(ingest_ok, ingest_total),
            "cost_usd": round(sum(costs), 6),
            "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        }
        fitness = sum(float(weights[name]) * float(metrics[name]) for name in PRIMARY_METRICS)
        scored[candidate_id_value] = {
            "metrics": metrics,
            "fitness": round(fitness, 8),
            "tasks": dict(tasks),
            "accepted_artifacts": dict(accepted),
            "failures": dict(failures),
        }
    return scored


def dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    am = a["metrics"]
    bm = b["metrics"]
    not_worse = all(float(am[name]) >= float(bm[name]) for name in PRIMARY_METRICS)
    better = any(float(am[name]) > float(bm[name]) for name in PRIMARY_METRICS)
    if not_worse and better:
        return True
    equal_primary = all(float(am[name]) == float(bm[name]) for name in PRIMARY_METRICS)
    return equal_primary and float(am["cost_usd"]) < float(bm["cost_usd"])


def pareto_front(scored: dict[str, Any]) -> list[str]:
    ids = sorted(scored)
    return [
        candidate_id_value
        for candidate_id_value in ids
        if not any(
            other != candidate_id_value and dominates(scored[other], scored[candidate_id_value])
            for other in ids
        )
    ]


def cmd_score(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    rows = read_jsonl(args.results)
    scored = score_rows(rows, cfg)
    front = pareto_front(scored)
    ranking = sorted(
        scored,
        key=lambda cid: (
            scored[cid]["fitness"],
            -float(scored[cid]["metrics"]["cost_usd"]),
        ),
        reverse=True,
    )
    body = {
        "schema": "star-model-evolution-score/v1",
        "pareto_front": front,
        "ranking": ranking,
        "candidates": scored,
    }
    write_json(args.output, body)
    print(json.dumps({"candidates": len(scored), "pareto_front": front, "output": args.output}))


def select_elites(score_doc: dict[str, Any], elite_count: int) -> list[str]:
    scored = score_doc["candidates"]
    front = list(score_doc.get("pareto_front") or pareto_front(scored))
    front.sort(
        key=lambda cid: (scored[cid]["fitness"], -float(scored[cid]["metrics"]["cost_usd"])),
        reverse=True,
    )
    if len(front) >= elite_count:
        return front[:elite_count]
    remaining = [cid for cid in score_doc["ranking"] if cid not in front]
    return (front + remaining)[:elite_count]


def cmd_evolve(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    score_doc = read_json(args.scores)
    candidates = load_candidates(args.candidate_dir)
    elite_count = int(cfg["elite"])
    target_population = int(cfg["population"])
    elites = select_elites(score_doc, elite_count)
    if not elites:
        raise ValueError("cannot evolve without scored candidates")
    for elite in elites:
        if elite not in candidates:
            raise ValueError(f"elite {elite} missing from candidate directory")

    generation = max(int(candidates[cid]["generation"]) for cid in candidates) + 1
    rng = random.Random(int(cfg["seed"]) + generation)
    next_population: list[dict[str, Any]] = []

    for elite in elites:
        parent = candidates[elite]
        next_population.append(make_candidate(generation, [elite], parent["genome"]))

    cursor = 0
    while len(next_population) < target_population:
        parent_id = elites[cursor % len(elites)]
        parent = candidates[parent_id]
        genome = mutate_genome(parent["genome"], cfg, rng)
        next_population.append(make_candidate(generation, [parent_id], genome))
        cursor += 1

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for candidate in next_population:
        write_json(out / f"candidate-{candidate['candidate_id']}.json", candidate)
    write_json(
        out / "selection.json",
        {
            "generation": generation,
            "parents": elites,
            "source_scores": str(args.scores),
            "population": [c["candidate_id"] for c in next_population],
        },
    )
    print(json.dumps({"generation": generation, "parents": elites, "candidates": len(next_population)}))


def failure_summary(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        if accepted_artifacts(row, cfg):
            continue
        failures.append(
            {
                "candidate_id": row.get("candidate_id"),
                "task_family": row.get("task_family", row.get("task")),
                "input": str(row.get("input", row.get("prompt", "")))[:2000],
                "error": str(row.get("error", row.get("feedback", "failed quality gate")))[:1000],
                "quality": row.get("quality"),
            }
        )
    return failures


def cmd_reflect_request(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    rows = read_jsonl(args.results)
    failures = failure_summary(rows, cfg)[: args.max_failures]
    families = sorted({str(item.get("task_family") or "unknown") for item in failures})
    request = {
        "type": "llm.dataset.synthesize",
        "version": 1,
        "corpus": "star-model-evolution-repair",
        "max_examples": args.max_examples,
        "task_families": families,
        "teacher": {"policy": "stronger-than-target", "provider_policy": "cheapest-allowed"},
        "evolution": {
            "strategy": "reflect-mutate-evaluate",
            "instruction": (
                "Study the failed held-out traces. Generate diverse training examples that teach the missing "
                "research, document, actor-system, ingest, or tool-calling behavior without reproducing the "
                "held-out answer. Preserve provenance and include adversarial near-misses."
            ),
            "failed_traces": failures,
        },
        "requirements": {
            "provenance": True,
            "content_hash": True,
            "reject_secrets": True,
            "public_or_synthetic_only": True,
        },
    }
    write_json(args.output, request)
    print(json.dumps({"failures": len(failures), "families": families, "output": args.output}))


def starintel_result_documents(score_doc: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    service = cfg["services"]
    schema_version = str(service.get("starintel_schema_version", "0.9.0"))
    dataset = str(service["starintel_dataset"])
    dtype = str(service["starintel_result_dtype"])
    documents = []
    for candidate_id_value, score in score_doc["candidates"].items():
        payload = {
            "candidate_id": candidate_id_value,
            "fitness": score["fitness"],
            "metrics": score["metrics"],
            "accepted_artifacts": score["accepted_artifacts"],
            "failures": score["failures"],
            "pareto": candidate_id_value in score_doc.get("pareto_front", []),
        }
        documents.append(
            {
                "_id": f"evolution-{digest(payload)[:24]}",
                "dataset": dataset,
                "dtype": dtype,
                "schema_version": schema_version,
                "sources": ["star-model:evolution-benchmark"],
                "data": {"content": canonical_json(payload)},
                "extensions": {
                    "star_model": {
                        "kind": "fine-tune-benchmark",
                        "candidate_id": candidate_id_value,
                    }
                },
            }
        )
    return documents


def cmd_ingest_scores(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    score_doc = read_json(args.scores)
    documents = starintel_result_documents(score_doc, cfg)
    if args.dry_run:
        write_json(args.dry_run, documents)
        print(json.dumps({"documents": len(documents), "dry_run": args.dry_run}))
        return
    token = args.token or os.getenv("STARINTEL_API_TOKEN")
    endpoint = cfg["services"]["starintel_bulk_ingest_endpoint"]
    response = http_post_json(args.starintel_url.rstrip("/") + endpoint, token, documents, timeout=args.timeout)
    if args.output:
        write_json(args.output, response)
    print(json.dumps({"documents": len(documents), "ingested": True}))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/evolution.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    seed = sub.add_parser("seed", help="create generation zero")
    seed.add_argument("--generation", type=int, default=0)
    seed.add_argument("--output-dir", required=True)
    seed.set_defaults(fn=cmd_seed)

    req = sub.add_parser("requests", help="build fine-tune requests for a candidate population")
    req.add_argument("--candidate-dir", required=True)
    req.add_argument("--manifest", required=True)
    req.add_argument("--base-model", required=True)
    req.add_argument("--max-hourly-gpu-usd", type=float, default=2.0)
    req.add_argument("--max-total-gpu-usd", type=float, default=20.0)
    req.add_argument("--output-dir", required=True)
    req.set_defaults(fn=cmd_requests)

    submit = sub.add_parser("submit", help="submit generated requests to llm.starintel.actor")
    submit.add_argument("--request-dir", required=True)
    submit.add_argument("--actor-url", required=True)
    submit.add_argument("--token")
    submit.add_argument("--timeout", type=int, default=120)
    submit.add_argument("--output-dir", required=True)
    submit.set_defaults(fn=cmd_submit)

    score = sub.add_parser("score", help="score JSONL benchmark traces")
    score.add_argument("--results", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(fn=cmd_score)

    evolve = sub.add_parser("evolve", help="select winners and create the next generation")
    evolve.add_argument("--candidate-dir", required=True)
    evolve.add_argument("--scores", required=True)
    evolve.add_argument("--output-dir", required=True)
    evolve.set_defaults(fn=cmd_evolve)

    reflect = sub.add_parser("reflect-request", help="build a teacher synthesis request from benchmark failures")
    reflect.add_argument("--results", required=True)
    reflect.add_argument("--max-failures", type=int, default=64)
    reflect.add_argument("--max-examples", type=int, default=2000)
    reflect.add_argument("--output", required=True)
    reflect.set_defaults(fn=cmd_reflect_request)

    ingest = sub.add_parser("ingest-scores", help="archive generation scores through StarIntel ingest")
    ingest.add_argument("--scores", required=True)
    ingest.add_argument("--starintel-url", default="http://127.0.0.1:5000")
    ingest.add_argument("--token")
    ingest.add_argument("--timeout", type=int, default=120)
    ingest.add_argument("--output")
    ingest.add_argument("--dry-run")
    ingest.set_defaults(fn=cmd_ingest_scores)
    return p


def main() -> None:
    args = parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
