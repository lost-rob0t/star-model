#!/usr/bin/env python3
"""Validate, deduplicate, rebalance, and split Star model JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from collections import Counter

TASKS = {
    "code",
    "prolog",
    "starintel-prolog-synthetic",
    "starintel-json-synthetic",
    "tool-calling",
    "ingest-jobs",
    "migrations",
    "osint-tool-design",
    "independent-research",
    "document-generation",
    "actor-system-generation",
}
SPLITS = {"train", "validation", "test"}


def normalized_hash(row: dict) -> str:
    messages = row.get("messages", [])
    material = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def validate(row: dict, line: int) -> None:
    missing = {"id", "task", "messages", "provenance", "content_sha256", "split"} - row.keys()
    if missing:
        raise ValueError(f"line {line}: missing {sorted(missing)}")
    if row["task"] not in TASKS:
        raise ValueError(f"line {line}: invalid task {row['task']!r}")
    if row["split"] not in SPLITS:
        raise ValueError(f"line {line}: invalid split {row['split']!r}")
    if not isinstance(row["messages"], list) or len(row["messages"]) < 2:
        raise ValueError(f"line {line}: messages must contain at least two turns")
    if row["provenance"].get("kind") in {"public-repository", "public-prolog"}:
        if not row["provenance"].get("repository"):
            raise ValueError(f"line {line}: public source missing repository provenance")
    if row.get("metadata", {}).get("needs_teacher"):
        raise ValueError(
            f"line {line}: uncompleted teacher task; llm.starintel.actor must ground it before mixing"
        )


def run(args: argparse.Namespace) -> None:
    source = pathlib.Path(args.input)
    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    handles = {split: (out / f"{split}.jsonl").open("w", encoding="utf-8") for split in SPLITS}
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    dropped = 0
    try:
        with source.open(encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                validate(row, line_number)
                digest = normalized_hash(row)
                if digest in seen:
                    dropped += 1
                    continue
                seen.add(digest)
                counts[row["task"]] += 1
                handles[row["split"]].write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    stats = {
        "accepted": sum(counts.values()),
        "deduplicated": dropped,
        "tasks": dict(sorted(counts.items())),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, sort_keys=True))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
