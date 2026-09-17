#!/usr/bin/env python3
"""Build and validate StarIntel fine-tuning datasets.

The builder intentionally emits task-shaped examples instead of treating whole
repositories as chat transcripts. Synthetic generation and training are owned by
llm.starintel.actor; this program produces manifests and requests for that actor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator


TEXT_SUFFIXES = {
    ".c", ".cc", ".cl", ".cpp", ".el", ".go", ".h", ".hpp", ".java", ".js",
    ".json", ".jsonld", ".lisp", ".md", ".nim", ".org", ".pl", ".pro", ".py",
    ".rs", ".scm", ".sh", ".sql", ".toml", ".ts", ".tsx", ".yaml", ".yml",
}
SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token|authorization)\s*[:=]\s*[\"']?[^\s\"']{12,}"
)


@dataclass(frozen=True)
class Source:
    repository: str
    commit: str
    license: str
    path: str
    text: str


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_split(digest: str) -> str:
    n = int(digest[:8], 16) / 0xFFFFFFFF
    if n < 0.96:
        return "train"
    if n < 0.98:
        return "validation"
    return "test"


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def repository_license(repo: pathlib.Path) -> str:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        p = repo / name
        if p.exists():
            first = p.read_text(errors="replace")[:4096].lower()
            if "apache" in first:
                return "Apache-2.0"
            if "mit license" in first:
                return "MIT"
            if "gnu general public license" in first:
                return "GPL"
            return "SEE-LICENSE-FILE"
    return "UNKNOWN"


def iter_sources(root: pathlib.Path, repository: str) -> Iterator[Source]:
    commit = git(root, "rev-parse", "HEAD")
    license_name = repository_license(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", "node_modules", "vendor", "dist", "build"} for part in path.parts):
            continue
        if path.stat().st_size > 1_000_000:
            continue
        text = path.read_text(errors="replace")
        if not text.strip() or "\x00" in text or SECRET_RE.search(text):
            continue
        yield Source(repository, commit, license_name, str(path.relative_to(root)), text)


def code_tasks(source: Source) -> Iterable[dict]:
    language = pathlib.Path(source.path).suffix.lstrip(".") or "text"
    chunks = chunk_text(source.text, 12000)
    for index, chunk in enumerate(chunks):
        prompt = (
            f"You are working in public repository {source.repository}. "
            f"Analyze this {language} source from {source.path}. Explain its behavior, "
            "important invariants, likely integration points, and one concrete maintenance task.\n\n"
            f"```{language}\n{chunk}\n```"
        )
        answer = (
            "Source example retained for supervised task synthesis. A teacher model should "
            "replace this placeholder with a grounded answer before the record enters the final SFT mix."
        )
        yield example(
            task="prolog" if language in {"pl", "pro"} else "code",
            prompt=prompt,
            answer=answer,
            provenance={
                "kind": "public-prolog" if language in {"pl", "pro"} else "public-repository",
                "repository": source.repository,
                "path": source.path,
                "commit": source.commit,
                "license": source.license,
                "chunk": index,
            },
            metadata={"language": language, "needs_teacher": True},
        )


def chunk_text(text: str, size: int) -> list[str]:
    lines = text.splitlines()
    out: list[str] = []
    current: list[str] = []
    length = 0
    for line in lines:
        if current and length + len(line) + 1 > size:
            out.append("\n".join(current))
            current = []
            length = 0
        current.append(line)
        length += len(line) + 1
    if current:
        out.append("\n".join(current))
    return out


def example(*, task: str, prompt: str, answer: str, provenance: dict, metadata: dict | None = None) -> dict:
    material = json.dumps([task, prompt, answer, provenance], sort_keys=True, separators=(",", ":"))
    digest = sha256(material)
    return {
        "id": f"star-{digest[:24]}",
        "task": task,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "provenance": provenance,
        "content_sha256": digest,
        "split": stable_split(digest),
        "metadata": metadata or {},
    }


def write_jsonl(path: pathlib.Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def build(args: argparse.Namespace) -> None:
    root = pathlib.Path(args.repo).resolve()
    rows: list[dict] = []
    seen: set[str] = set()
    for source in iter_sources(root, args.repository):
        for row in code_tasks(source):
            if row["content_sha256"] in seen:
                continue
            seen.add(row["content_sha256"])
            rows.append(row)
    random.Random(545).shuffle(rows)
    count = write_jsonl(pathlib.Path(args.output), rows)
    print(json.dumps({"examples": count, "output": args.output}))


def manifest(args: argparse.Namespace) -> None:
    inputs = [pathlib.Path(p) for p in args.inputs]
    files = []
    total = 0
    for path in inputs:
        blob = path.read_bytes()
        count = sum(1 for line in blob.splitlines() if line.strip())
        total += count
        files.append({"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "examples": count})
    body = {
        "schema": "star-model-dataset-manifest/v1",
        "name": args.name,
        "examples": total,
        "files": files,
        "training_controller": "star://llm.starintel.actor",
    }
    pathlib.Path(args.output).write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def actor_request(args: argparse.Namespace) -> None:
    manifest_doc = json.loads(pathlib.Path(args.manifest).read_text())
    request = {
        "type": "llm.fine_tune.start",
        "version": 1,
        "dataset": manifest_doc,
        "training": {
            "base_model": args.base_model,
            "method": "qlora",
            "precision": "bf16",
            "max_seq_length": args.max_seq_length,
            "packing": True,
            "publish_adapter_only": True,
        },
        "provider_policy": {
            "inference": "cheapest-allowed",
            "gpu": "cheapest-compatible",
            "max_hourly_gpu_usd": args.max_hourly_gpu_usd,
            "max_total_gpu_usd": args.max_total_gpu_usd,
        },
    }
    pathlib.Path(args.output).write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--repository", required=True)
    b.add_argument("--output", required=True)
    b.set_defaults(fn=build)

    m = sub.add_parser("manifest")
    m.add_argument("--name", default="star-model-corpus")
    m.add_argument("--output", required=True)
    m.add_argument("inputs", nargs="+")
    m.set_defaults(fn=manifest)

    a = sub.add_parser("actor-request")
    a.add_argument("--manifest", required=True)
    a.add_argument("--base-model", required=True)
    a.add_argument("--max-seq-length", type=int, default=8192)
    a.add_argument("--max-hourly-gpu-usd", type=float, default=2.0)
    a.add_argument("--max-total-gpu-usd", type=float, default=20.0)
    a.add_argument("--output", required=True)
    a.set_defaults(fn=actor_request)
    return p


def main() -> None:
    args = parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
