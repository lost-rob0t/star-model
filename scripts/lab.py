#!/usr/bin/env python3
"""Star Model lab orchestration.

This script owns local/reproducible artifact preparation only. AWS resource
provisioning lives in starintel-infra and is launched by GitHub Actions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "lab.toml"
GIB = 1024**3


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, cwd: Path = ROOT) -> None:
    subprocess.run(argv, cwd=cwd, check=True)


def output(argv: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def command_plan(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    dataset = cfg["dataset"]
    tasks = list(dataset["synthetic_tasks"])
    per_task = int(dataset["synthetic_target_gib_per_task"] * GIB)
    training_cap = int(float(dataset["max_training_gib_per_task"]) * GIB)
    result = {
        "schema": "star-model-lab-plan/v1",
        "complete_codebase": bool(dataset["include_complete_codebase"]),
        "synthetic_tasks": len(tasks),
        "synthetic_target_bytes_per_task": per_task,
        "synthetic_target_bytes_total": per_task * len(tasks),
        "synthetic_target_gib_total": (per_task * len(tasks)) / GIB,
        "training_cap_bytes_per_task": training_cap,
        "tasks": [
            {
                "name": task,
                "reservoir_target_bytes": per_task,
                "training_cap_bytes": training_cap,
            }
            for task in tasks
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def command_bundle(args: argparse.Namespace) -> None:
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    ref = args.ref

    with tempfile.TemporaryDirectory(prefix="star-model-archive-") as temp_dir:
        tar_path = Path(temp_dir) / "source.tar"
        with tar_path.open("wb") as handle:
            subprocess.run(
                ["git", "archive", "--format=tar", ref],
                cwd=ROOT,
                check=True,
                stdout=handle,
            )
        run(["zstd", "-T0", "-19", "-f", str(tar_path), "-o", str(destination)])

    result = {
        "schema": "star-model-source-bundle/v1",
        "ref": ref,
        "commit": output(["git", "rev-parse", ref]),
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def verify_manifest(dataset_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "star-model-dataset-manifest/v1":
        raise ValueError("unsupported dataset manifest schema")

    for entry in manifest.get("files", []):
        raw_path = Path(entry["path"])
        candidates = [raw_path, dataset_dir / raw_path.name, dataset_dir / raw_path]
        actual = next((candidate for candidate in candidates if candidate.is_file()), None)
        if actual is None:
            raise FileNotFoundError(f"manifest file missing: {entry['path']}")
        expected = entry.get("sha256")
        if expected and sha256_file(actual) != expected:
            raise ValueError(f"manifest digest mismatch: {actual}")
    return manifest


def normalized_s3_root(uri: str) -> str:
    if not uri.startswith("s3://"):
        raise ValueError("S3 URI must start with s3://")
    return uri.rstrip("/")


def command_stage(args: argparse.Namespace) -> None:
    root = normalized_s3_root(args.s3_root)
    dataset_dir = args.dataset_dir.resolve()
    manifest_path = args.manifest.resolve()
    source_archive = args.source_archive.resolve()

    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)
    if not source_archive.is_file():
        raise FileNotFoundError(source_archive)
    manifest = verify_manifest(dataset_dir, manifest_path)

    run_id = args.run_id or f"{int(time.time())}-{output(['git', 'rev-parse', '--short=12', 'HEAD'])}"
    run_root = f"{root}/runs/{run_id}"
    source_uri = f"{run_root}/source/{source_archive.name}"
    dataset_uri = f"{run_root}/dataset"
    manifest_uri = f"{dataset_uri}/manifest.json"
    output_uri = f"{run_root}/output"

    run(["aws", "s3", "cp", "--only-show-errors", str(source_archive), source_uri])
    run(
        [
            "aws",
            "s3",
            "sync",
            "--only-show-errors",
            "--exclude",
            "manifest.json",
            str(dataset_dir),
            dataset_uri,
        ]
    )
    run(["aws", "s3", "cp", "--only-show-errors", str(manifest_path), manifest_uri])

    launch = {
        "schema": "star-model-aws-launch/v1",
        "run_id": run_id,
        "source_s3_uri": source_uri,
        "source_sha256": sha256_file(source_archive),
        "dataset_s3_uri": dataset_uri,
        "manifest_s3_uri": manifest_uri,
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_examples": manifest.get("examples"),
        "output_s3_uri": output_uri,
        "source_commit": output(["git", "rev-parse", "HEAD"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(json.dumps(launch, indent=2, sort_keys=True))


def prolog_atom(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def command_record(args: argparse.Namespace) -> None:
    facts = args.facts.resolve()
    facts.parent.mkdir(parents=True, exist_ok=True)
    timestamp = args.timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = (
        f"task_state({prolog_atom(args.run_id)}, {prolog_atom(args.task)}, "
        f"{prolog_atom(args.state)}, {prolog_atom(timestamp)}, {prolog_atom(args.detail)}).\n"
    )
    with facts.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="")


def command_ready(args: argparse.Namespace) -> None:
    argv = [
        "swipl",
        "-q",
        "-s",
        str(ROOT / "prolog" / "lab_tasks.pl"),
    ]
    if args.facts:
        argv.extend(["-s", str(args.facts.resolve())])
    argv.extend(["-g", f"print_ready({prolog_atom(args.run_id)}),halt"])
    run(argv)


def command_bootstrap(args: argparse.Namespace) -> None:
    python = args.python
    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", str(ROOT / "trainer" / "requirements.txt")])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = p.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="show corpus reservoir and training caps")
    plan.set_defaults(func=command_plan)

    bundle = sub.add_parser("bundle-source", help="make an immutable source archive")
    bundle.add_argument("--ref", default="HEAD")
    bundle.add_argument("--output", type=Path, default=ROOT / "out" / "star-model-source.tar.zst")
    bundle.set_defaults(func=command_bundle)

    stage = sub.add_parser("stage", help="verify and copy source/data to an AWS S3 run prefix")
    stage.add_argument("--dataset-dir", type=Path, required=True)
    stage.add_argument("--manifest", type=Path, required=True)
    stage.add_argument("--source-archive", type=Path, required=True)
    stage.add_argument("--s3-root", required=True)
    stage.add_argument("--run-id")
    stage.add_argument("--output", type=Path, default=ROOT / "out" / "aws-launch.json")
    stage.set_defaults(func=command_stage)

    record = sub.add_parser("record", help="append one durable Prolog task-state fact")
    record.add_argument("--facts", type=Path, required=True)
    record.add_argument("--run-id", required=True)
    record.add_argument("--task", required=True)
    record.add_argument("--state", choices=["pending", "running", "done", "failed", "skipped"], required=True)
    record.add_argument("--detail", default="")
    record.add_argument("--timestamp")
    record.set_defaults(func=command_record)

    ready = sub.add_parser("ready", help="ask Prolog which lab tasks are currently runnable")
    ready.add_argument("--run-id", required=True)
    ready.add_argument("--facts", type=Path)
    ready.set_defaults(func=command_ready)

    bootstrap = sub.add_parser("bootstrap-training", help="install trainer packages into the selected DLAMI Python")
    bootstrap.add_argument("--python", default=os.environ.get("STAR_TRAIN_PYTHON", "/opt/conda/envs/pytorch/bin/python"))
    bootstrap.set_defaults(func=command_bootstrap)

    return p


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"lab error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
