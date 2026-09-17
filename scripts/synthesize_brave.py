#!/usr/bin/env python3
"""Generate grounded synthetic StarIntel training data using Brave Search.

The generator uses Brave's LLM Context API for public-web grounding and an
OpenAI-compatible teacher (normally llm.starintel.actor) for synthesis. Raw Brave
snippets are not written to the training JSONL; source URLs, titles, hostnames,
query metadata, and a context hash are retained for auditability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from typing import Any, Iterable


SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token|authorization)\s*[:=]\s*[\"']?[^\s\"']{12,}"
)
ALLOWED_TASKS = {
    "starintel-prolog-synthetic",
    "starintel-json-synthetic",
    "tool-calling",
    "ingest-jobs",
    "migrations",
    "osint-tool-design",
    "osint-skill-design",
    "starintel-tool-design",
    "independent-research",
    "document-generation",
    "actor-system-generation",
}

FAMILY_GUIDANCE = {
    "independent-research": (
        "Create research tasks and high-quality answers that synthesize multiple supplied public sources, "
        "separate established facts from hypotheses, preserve URLs, and name unresolved questions."
    ),
    "osint-tool-design": (
        "Design lawful public-source research tools with explicit inputs/outputs, provenance, rate limits, "
        "failure behavior, tests, and a bounded implementation sketch. Do not design credential theft, "
        "unauthorized access, evasion, or invasive collection."
    ),
    "osint-skill-design": (
        "Design reusable OSINT agent skills for public information: trigger conditions, allowed tools, search "
        "strategy, evidence requirements, stopping rules, provenance, and deterministic verification checks."
    ),
    "starintel-tool-design": (
        "Design StarIntel-native tools that read or emit typed documents, use canonical ingest/search APIs, "
        "preserve provenance, define schemas, and include test vectors and error handling."
    ),
    "tool-calling": (
        "Create tool-calling conversations where tool choice and JSON arguments are unambiguous. Include the "
        "tool schemas and realistic tool results before the assistant synthesis."
    ),
    "ingest-jobs": (
        "Create StarIntel ingest tasks covering validation, bounded bulk writes, idempotency, provenance, "
        "retries, partial failures, and repair."
    ),
    "migrations": (
        "Create data-model migration tasks with before/after examples, invariants, rollback strategy, and "
        "validation checks."
    ),
    "starintel-json-synthetic": (
        "Create synthetic StarIntel JSON/JSON-LD documents and transformations with canonical field spelling, "
        "provenance, relations, and schema-focused validation examples."
    ),
    "starintel-prolog-synthetic": (
        "Create Prolog facts, rules, queries, and verification tasks around provenance, actor routing, document "
        "constraints, and research workflows. Include expected query behavior."
    ),
    "document-generation": (
        "Create document-production tasks that result in a complete report/spec/runbook plus objective checks "
        "for structure, factual grounding, provenance, and completeness."
    ),
    "actor-system-generation": (
        "Create actor-system implementation tasks with message contracts, supervision/failure semantics, "
        "bounded concurrency, runnable tests, and StarIntel ingest/provenance integration."
    ),
}


def read_toml(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).open("rb") as fh:
        return tomllib.load(fh)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_split(digest: str) -> str:
    n = int(digest[:8], 16) / 0xFFFFFFFF
    if n < 0.96:
        return "train"
    if n < 0.98:
        return "validation"
    return "test"


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int = 4,
) -> dict[str, Any]:
    body = canonical_json(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json", **headers}
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(f"POST {url}: expected JSON object")
            return value
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt + 1 >= retries:
                raise RuntimeError(f"POST {url}: HTTP {exc.code}: {detail[:2000]}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"POST {url}: {exc}") from exc
        time.sleep(min(8.0, 1.25 * (2**attempt)))
    raise AssertionError("unreachable")


def brave_context(query: str, cfg: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    brave = cfg["brave"]
    payload = {
        "q": query[:600],
        "country": brave.get("country", "US"),
        "search_lang": brave.get("search_lang", "en"),
        "count": int(brave.get("count", 20)),
        "maximum_number_of_urls": int(brave.get("maximum_number_of_urls", 12)),
        "maximum_number_of_tokens": int(brave.get("maximum_number_of_tokens", 8192)),
        "maximum_number_of_tokens_per_url": int(brave.get("maximum_number_of_tokens_per_url", 2048)),
        "context_threshold_mode": brave.get("context_threshold_mode", "balanced"),
        "safesearch": brave.get("safesearch", "moderate"),
        "enable_source_metadata": bool(brave.get("enable_source_metadata", True)),
        "spellcheck": True,
    }
    return post_json(
        str(brave["endpoint"]),
        {"X-Subscription-Token": api_key, "Accept-Encoding": "gzip"},
        payload,
        timeout=timeout,
    )


def extract_grounding(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generic = result.get("grounding", {}).get("generic", [])
    if not isinstance(generic, list):
        generic = []
    sources = result.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}
    cleaned = []
    for item in generic:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        snippets = [str(s) for s in item.get("snippets", []) if str(s).strip()]
        if not url or not snippets:
            continue
        cleaned.append({"url": url, "title": title, "snippets": snippets})
    return cleaned, sources


def grounding_text(items: list[dict[str, Any]], max_chars: int = 48000) -> str:
    chunks: list[str] = []
    used = 0
    for index, item in enumerate(items, 1):
        text = f"SOURCE {index}\nTITLE: {item['title']}\nURL: {item['url']}\n" + "\n".join(item["snippets"])
        if used + len(text) > max_chars:
            remaining = max_chars - used
            if remaining > 400:
                chunks.append(text[:remaining])
            break
        chunks.append(text)
        used += len(text)
    return "\n\n".join(chunks)


def source_metadata(items: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, str]]:
    out = []
    seen = set()
    for item in items:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        meta = sources.get(url, {}) if isinstance(sources.get(url, {}), dict) else {}
        out.append(
            {
                "url": url,
                "title": str(item.get("title") or meta.get("title") or ""),
                "hostname": str(meta.get("hostname") or ""),
            }
        )
    return out


def teacher_prompt(topic: str, family: str, examples: int, grounded_text: str) -> str:
    guidance = FAMILY_GUIDANCE[family]
    return f"""You are generating supervised fine-tuning data for a StarIntel-focused model.

TASK FAMILY: {family}
TOPIC: {topic}
NUMBER OF EXAMPLES: {examples}

Goal:
{guidance}

Grounding rules:
- Use only the supplied public-source context for factual claims about the outside world.
- Do not invent URLs, source titles, test results, commands already run, or observations not present below.
- Synthetic code, schemas, actors, tool definitions, documents, and hypothetical inputs are allowed when clearly synthetic.
- Keep public-source research lawful and non-invasive. Do not include secrets, private credentials, stalking workflows, credential theft, unauthorized access, malware deployment, or evasion instructions.
- Prefer concrete tasks whose answer can be checked automatically.
- For StarIntel examples, favor strict schemas, provenance, typed messages, bounded concurrency, idempotent ingest, and explicit failure handling.
- For tool-calling, include a tools array and at least one assistant tool call plus tool response before the final assistant answer.

Return ONLY one JSON object with this shape:
{{
  "examples": [
    {{
      "task": "{family}",
      "messages": [{{"role":"user","content":"..."}}, {{"role":"assistant","content":"..."}}],
      "tools": [],
      "metadata": {{"quality_checks":["..."], "skills":["..."]}}
    }}
  ]
}}

PUBLIC-SOURCE CONTEXT:
{grounded_text}
"""


def teacher_completion(
    prompt: str,
    cfg: dict[str, Any],
    actor_url: str,
    actor_token: str,
    model: str,
    timeout: int,
) -> dict[str, Any]:
    teacher = cfg["teacher"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. You are a synthetic-data generator, not a conversational assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(teacher.get("temperature", 0.45)),
        "max_tokens": int(teacher.get("max_tokens", 8192)),
    }
    endpoint = str(teacher.get("endpoint", "/v1/chat/completions"))
    return post_json(
        actor_url.rstrip("/") + endpoint,
        {"Authorization": f"Bearer {actor_token}"},
        payload,
        timeout=timeout,
    )


def completion_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("teacher response missing choices[0].message.content") from exc
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    raise ValueError("teacher message content is not text")


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        last_fence = stripped.rfind("```")
        if first_newline != -1 and last_fence > first_newline:
            stripped = stripped[first_newline + 1 : last_fence].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    return value


def normalize_example(
    raw: dict[str, Any],
    *,
    family: str,
    topic: str,
    query: str,
    source_meta: list[dict[str, str]],
    context_hash: str,
    teacher_model: str,
    prompt_version: str,
) -> dict[str, Any]:
    task = str(raw.get("task", family))
    if task != family or task not in ALLOWED_TASKS:
        raise ValueError(f"teacher returned invalid task {task!r}; expected {family!r}")
    messages = raw.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("teacher example needs at least two messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid teacher message")
    tools = raw.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {
        **metadata,
        "grounded": True,
        "brave_query": query,
        "source_count": len(source_meta),
        "topic": topic,
    }
    material = {
        "task": task,
        "messages": messages,
        "tools": tools,
        "query": query,
        "sources": source_meta,
        "teacher_model": teacher_model,
    }
    content_hash = sha256_text(canonical_json(material))
    row = {
        "id": f"star-synth-{content_hash[:24]}",
        "task": task,
        "messages": messages,
        "tools": tools,
        "provenance": {
            "kind": "teacher-trace",
            "generator": "scripts/synthesize_brave.py",
            "model": teacher_model,
            "prompt_version": prompt_version,
            "brave_query": query,
            "brave_context_sha256": context_hash,
            "sources": source_meta,
        },
        "content_sha256": content_hash,
        "split": stable_split(content_hash),
        "metadata": metadata,
    }
    if SECRET_RE.search(canonical_json(row)):
        raise ValueError("synthetic example contains secret-like text")
    return row


def family_query(topic: str, family: str) -> str:
    suffix = {
        "independent-research": "primary sources documentation evidence verification",
        "osint-tool-design": "public source OSINT tool API provenance rate limits",
        "osint-skill-design": "OSINT research workflow evidence provenance search skill",
        "starintel-tool-design": "StarIntel tools document ingest actor provenance API",
        "tool-calling": "API tool schema examples documentation",
        "ingest-jobs": "document ingest bulk validation retry idempotency",
        "migrations": "schema migration compatibility validation rollback",
        "starintel-json-synthetic": "JSON schema provenance document examples",
        "starintel-prolog-synthetic": "Prolog rules provenance validation actor workflow",
        "document-generation": "technical report specification runbook evidence",
        "actor-system-generation": "actor model supervision message passing bounded concurrency",
    }[family]
    return f"{topic} {suffix}"


def iter_seed_jobs(cfg: dict[str, Any], only_family: str | None) -> Iterable[tuple[str, str]]:
    for seed in cfg.get("seed", []):
        topic = str(seed["topic"])
        for family in seed.get("families", []):
            family = str(family)
            if family not in ALLOWED_TASKS:
                raise ValueError(f"unknown synthesis family {family!r}")
            if only_family and family != only_family:
                continue
            yield topic, family


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    brave_key = args.brave_key or os.getenv("BRAVE_SEARCH_API_KEY", "")
    actor_token = args.actor_token or os.getenv("STAR_LLM_ACTOR_TOKEN", "")
    actor_url = args.actor_url or os.getenv("STAR_LLM_ACTOR_URL", "")
    model = args.model or os.getenv("STAR_SYNTH_MODEL", "")
    ack = args.ack_brave_training_rights or os.getenv("STAR_BRAVE_TRAINING_RIGHTS_ACK", "").lower() in {"1", "true", "yes"}

    if not ack:
        raise SystemExit(
            "Brave-backed fine-tuning data generation requires confirmation that your Brave plan permits "
            "the intended storage/training use. Set STAR_BRAVE_TRAINING_RIGHTS_ACK=1 or pass "
            "--ack-brave-training-rights."
        )
    if not brave_key:
        raise SystemExit("BRAVE_SEARCH_API_KEY is required")
    if not actor_url or not actor_token:
        raise SystemExit("STAR_LLM_ACTOR_URL and STAR_LLM_ACTOR_TOKEN are required")
    if not model or model == "configurable":
        raise SystemExit("STAR_SYNTH_MODEL or --model is required")

    examples_per_seed = args.examples_per_seed or int(cfg.get("examples_per_seed", 3))
    max_examples = int(cfg["output"].get("max_examples_per_teacher_call", 8))
    examples_per_seed = min(examples_per_seed, max_examples)
    prompt_version = str(cfg["teacher"].get("prompt_version", "brave-synth-v1"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    jobs = list(iter_seed_jobs(cfg, args.family))
    random.Random(int(cfg.get("seed", 545))).shuffle(jobs)
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]

    for topic, family in jobs:
        query = family_query(topic, family)
        brave = brave_context(query, cfg, brave_key, args.timeout)
        grounding, sources = extract_grounding(brave)
        if not grounding:
            print(f"warning: no Brave grounding for {family}: {query}", file=sys.stderr)
            continue
        context = grounding_text(grounding)
        context_hash = sha256_text(context)
        source_meta = source_metadata(grounding, sources)
        prompt = teacher_prompt(topic, family, examples_per_seed, context)
        response = teacher_completion(prompt, cfg, actor_url, actor_token, model, args.timeout)
        parsed = parse_json_object(completion_text(response))
        examples = parsed.get("examples", [])
        if not isinstance(examples, list):
            raise ValueError("teacher examples must be an array")
        accepted = 0
        for raw in examples[:examples_per_seed]:
            if not isinstance(raw, dict):
                continue
            row = normalize_example(
                raw,
                family=family,
                topic=topic,
                query=query,
                source_meta=source_meta,
                context_hash=context_hash,
                teacher_model=model,
                prompt_version=prompt_version,
            )
            if row["content_sha256"] in seen:
                continue
            seen.add(row["content_sha256"])
            rows.append(row)
            accepted += 1
        print(json.dumps({"family": family, "sources": len(source_meta), "examples": accepted}), file=sys.stderr)

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"examples": len(rows), "jobs": len(jobs), "output": str(output)}))


def cmd_brave_probe(args: argparse.Namespace) -> None:
    cfg = read_toml(args.config)
    brave_key = args.brave_key or os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not brave_key:
        raise SystemExit("BRAVE_SEARCH_API_KEY is required")
    result = brave_context(args.query, cfg, brave_key, args.timeout)
    grounding, sources = extract_grounding(result)
    print(
        json.dumps(
            {
                "query": args.query,
                "grounding_items": len(grounding),
                "sources": source_metadata(grounding, sources),
            },
            indent=2,
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/synthesis.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    generate = sub.add_parser("generate")
    generate.add_argument("--output", required=True)
    generate.add_argument("--actor-url")
    generate.add_argument("--actor-token")
    generate.add_argument("--brave-key")
    generate.add_argument("--model")
    generate.add_argument("--family", choices=sorted(ALLOWED_TASKS))
    generate.add_argument("--examples-per-seed", type=int)
    generate.add_argument("--max-jobs", type=int)
    generate.add_argument("--timeout", type=int, default=120)
    generate.add_argument("--ack-brave-training-rights", action="store_true")
    generate.set_defaults(fn=cmd_generate)

    probe = sub.add_parser("brave-probe")
    probe.add_argument("query")
    probe.add_argument("--brave-key")
    probe.add_argument("--timeout", type=int, default=30)
    probe.set_defaults(fn=cmd_brave_probe)
    return p


def main() -> None:
    args = parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
