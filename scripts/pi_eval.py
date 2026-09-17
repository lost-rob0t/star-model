#!/usr/bin/env python3
"""Run the same held-out coding tasks through a base and fine-tuned model.

Pi is the agent harness. Each model receives a fresh copy of the same fixture.
A deterministic command grades the resulting workspace. Prolog owns the final
promotion policy so a model cannot talk itself into being promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "lab.toml"
DEFAULT_SUITE = ROOT / "eval" / "suite.jsonl"


@dataclass(frozen=True)
class ModelTarget:
    label: str
    base_url: str
    model: str
    api_key_env: str


def read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def read_suite(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"{path}:{line_no}: missing task id")
        if task_id in seen:
            raise ValueError(f"{path}:{line_no}: duplicate task id {task_id}")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{path}:{line_no}: missing prompt")
        if not isinstance(row.get("fixture"), str) or not row["fixture"]:
            raise ValueError(f"{path}:{line_no}: missing fixture")
        grader = row.get("grader")
        if not isinstance(grader, dict) or not isinstance(grader.get("command"), str):
            raise ValueError(f"{path}:{line_no}: missing grader.command")
        seen.add(task_id)
        rows.append(row)
    if not rows:
        raise ValueError(f"empty evaluation suite: {path}")
    return rows


def write_models_config(path: Path, targets: Iterable[ModelTarget]) -> None:
    providers: dict[str, Any] = {}
    for target in targets:
        providers[target.label] = {
            "baseUrl": target.base_url.rstrip("/"),
            "api": "openai-completions",
            "apiKey": target.api_key_env,
            "models": [
                {
                    "id": target.model,
                    "name": f"Star Model eval {target.label}",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                    },
                    "contextWindow": 131072,
                    "maxTokens": 32768,
                }
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"providers": providers}, indent=2, sort_keys=True) + "\n")


def copy_fixture(fixture: Path, workspace: Path) -> None:
    if not fixture.exists():
        raise FileNotFoundError(fixture)
    if fixture.is_dir():
        shutil.copytree(fixture, workspace, dirs_exist_ok=True)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fixture, workspace / fixture.name)


def recursive_usage(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        usage = value.get("usage")
        if isinstance(usage, dict):
            yield usage
        for child in value.values():
            yield from recursive_usage(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_usage(child)


def usage_number(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def parse_last_usage(trace: str) -> tuple[int, int]:
    last: dict[str, Any] | None = None
    for raw in trace.splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for usage in recursive_usage(event):
            last = usage
    if last is None:
        return 0, 0
    input_tokens = usage_number(last, "input", "inputTokens", "input_tokens", "prompt_tokens")
    output_tokens = usage_number(last, "output", "outputTokens", "output_tokens", "completion_tokens")
    return input_tokens, output_tokens


def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int,
) -> tuple[int, str, str, int]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\nprocess timed out after {timeout}s\n"
    elapsed_ms = int((time.monotonic() - started) * 1000)
    return code, stdout, stderr, elapsed_ms


def run_one(
    task: dict[str, Any],
    target: ModelTarget,
    repeat: int,
    *,
    args: argparse.Namespace,
    pi_config_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    fixture = (ROOT / task["fixture"]).resolve()
    timeout = int(task.get("timeout_seconds", args.timeout))

    with tempfile.TemporaryDirectory(prefix=f"star-eval-{target.label}-{task['id']}-") as temp:
        workspace = Path(temp) / "workspace"
        copy_fixture(fixture, workspace)

        env = os.environ.copy()
        env["PI_CODING_AGENT_DIR"] = str(pi_config_dir)
        env["PI_SKIP_VERSION_CHECK"] = "1"
        env.setdefault(target.api_key_env, "local-eval")

        pi_argv = [
            args.pi_bin,
            "--provider",
            target.label,
            "--model",
            target.model,
            "--mode",
            "json",
            "--no-session",
            "--tools",
            args.tools,
            task["prompt"],
        ]
        pi_code, trace, pi_stderr, pi_ms = run_process(
            pi_argv,
            cwd=workspace,
            env=env,
            timeout=timeout,
        )

        grader = task["grader"]
        grader_timeout = int(grader.get("timeout_seconds", min(timeout, 300)))
        grader_code, grader_stdout, grader_stderr, grader_ms = run_process(
            ["bash", "-lc", grader["command"]],
            cwd=workspace,
            env=env,
            timeout=grader_timeout,
        )

        input_tokens, output_tokens = parse_last_usage(trace)
        passed = pi_code == 0 and grader_code == 0
        result = {
            "task": task["id"],
            "family": task.get("family", "unspecified"),
            "model": target.label,
            "model_id": target.model,
            "repeat": repeat,
            "outcome": "pass" if passed else "fail",
            "pi_returncode": pi_code,
            "grader_returncode": grader_code,
            "latency_ms": pi_ms,
            "grader_ms": grader_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        artifact_dir = run_dir / target.label / task["id"] / f"r{repeat:02d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "pi-trace.jsonl").write_text(trace)
        (artifact_dir / "pi-stderr.log").write_text(pi_stderr)
        (artifact_dir / "grader.log").write_text(grader_stdout + grader_stderr)
        (artifact_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if args.keep_workspaces:
            shutil.copytree(workspace, artifact_dir / "workspace", dirs_exist_ok=True)
        return result


def prolog_atom(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def write_eval_facts(path: Path, results: list[dict[str, Any]], config: dict[str, Any]) -> None:
    evaluation = config["evaluation"]
    lines = ["% GENERATED by scripts/pi_eval.py; do not edit.\n"]
    for result in results:
        lines.append(
            "eval_result({model}, {task}, {repeat}, {outcome}, {latency}, {input_tokens}, {output_tokens}).\n".format(
                model=result["model"],
                task=prolog_atom(result["task"]),
                repeat=int(result["repeat"]),
                outcome=result["outcome"],
                latency=int(result["latency_ms"]),
                input_tokens=int(result["input_tokens"]),
                output_tokens=int(result["output_tokens"]),
            )
        )
    lines.extend(
        [
            f"gate(min_pass_rate_delta, {float(evaluation['min_pass_rate_delta'])}).\n",
            f"gate(max_regressions, {int(evaluation['max_regressions'])}).\n",
            f"gate(max_latency_ratio, {float(evaluation['max_latency_ratio'])}).\n",
        ]
    )
    path.write_text("".join(lines))


def prolog_verdict(facts: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "swipl",
            "-q",
            "-s",
            str(ROOT / "prolog" / "eval_policy.pl"),
            "-s",
            str(facts),
            "-g",
            "promotion_json,halt",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Prolog evaluation policy failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def aggregate(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [result for result in results if result["model"] == label]
    passed = sum(result["outcome"] == "pass" for result in selected)
    latencies = [int(result["latency_ms"]) for result in selected]
    return {
        "runs": len(selected),
        "passed": passed,
        "pass_rate": passed / len(selected) if selected else 0.0,
        "median_latency_ms": int(statistics.median(latencies)) if latencies else 0,
        "input_tokens": sum(int(result["input_tokens"]) for result in selected),
        "output_tokens": sum(int(result["output_tokens"]) for result in selected),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--output-dir", type=Path, default=ROOT / "out" / "pi-eval")
    p.add_argument("--baseline-url", required=True)
    p.add_argument("--baseline-model", required=True)
    p.add_argument("--candidate-url", required=True)
    p.add_argument("--candidate-model", required=True)
    p.add_argument("--baseline-api-key-env", default="STAR_BASELINE_API_KEY")
    p.add_argument("--candidate-api-key-env", default="STAR_CANDIDATE_API_KEY")
    p.add_argument("--pi-bin", default="pi-pinned")
    p.add_argument("--tools")
    p.add_argument("--repetitions", type=int)
    p.add_argument("--timeout", type=int)
    p.add_argument("--keep-workspaces", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    try:
        config = read_config(args.config)
        suite = read_suite(args.suite)
        args.tools = args.tools or ",".join(config["pi"]["tools"])
        args.repetitions = args.repetitions or int(config["evaluation"]["repetitions"])
        args.timeout = args.timeout or int(config["evaluation"]["default_timeout_seconds"])
        if args.repetitions < 1:
            raise ValueError("repetitions must be >= 1")

        baseline = ModelTarget(
            "baseline",
            args.baseline_url,
            args.baseline_model,
            args.baseline_api_key_env,
        )
        candidate = ModelTarget(
            "candidate",
            args.candidate_url,
            args.candidate_model,
            args.candidate_api_key_env,
        )

        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = args.output_dir.resolve() / stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        pi_config_dir = run_dir / "pi-config"
        write_models_config(pi_config_dir / "models.json", [baseline, candidate])

        results: list[dict[str, Any]] = []
        for task in suite:
            for repeat in range(1, args.repetitions + 1):
                results.append(
                    run_one(
                        task,
                        baseline,
                        repeat,
                        args=args,
                        pi_config_dir=pi_config_dir,
                        run_dir=run_dir,
                    )
                )
                results.append(
                    run_one(
                        task,
                        candidate,
                        repeat,
                        args=args,
                        pi_config_dir=pi_config_dir,
                        run_dir=run_dir,
                    )
                )

        facts = run_dir / "eval-facts.pl"
        write_eval_facts(facts, results, config)
        verdict = prolog_verdict(facts)
        report = {
            "schema": "star-model-pi-ab/v1",
            "suite": str(args.suite),
            "baseline": aggregate(results, "baseline"),
            "candidate": aggregate(results, "candidate"),
            "policy": verdict,
            "results": results,
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if verdict.get("verdict") == "promote" else 3
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print(f"pi-eval error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
