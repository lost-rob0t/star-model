#!/usr/bin/env python3
"""Generate Brave-grounded synthetic fine-tuning data for StarIntel."""

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
    "independent-research": "Produce evidence-backed research with source URLs, uncertainty, and unresolved questions.",
    "osint-tool-design": "Design lawful public-source OSINT tools with schemas, provenance, rate limits, errors, and tests.",
    "osint-skill-design": "Design reusable public-source research skills with triggers, tools, search strategy, evidence rules, and stopping criteria.",
    "starintel-tool-design": "Design StarIntel-native tools using typed documents, canonical APIs, provenance, test vectors, and explicit failures.",
    "tool-calling": "Produce real tool-call conversations with tool schemas, strict JSON arguments, tool results, and final synthesis.",
    "ingest-jobs": "Produce ingest tasks covering validation, bounded bulk writes, idempotency, retries, partial failure, and repair.",
    "migrations": "Produce migrations with before/after examples, invariants, validation, compatibility, and rollback.",
    "starintel-json-synthetic": "Produce synthetic StarIntel JSON/JSON-LD documents and typed transformations with provenance.",
    "starintel-prolog-synthetic": "Produce Prolog facts, rules, queries, and verification tasks for provenance and actor workflows.",
    "document-generation": "Produce complete report/spec/runbook tasks with objective structure, grounding, and completeness checks.",
    "actor-system-generation": "Produce actor-system tasks with message contracts, supervision, bounded concurrency, ingest integration, and runnable tests.",
}
QUERY_SUFFIX = {
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
}


def read_toml(path: str | pathlib.Path) -> dict[str, Any]:
    with pathlib.Path(path).open("rb") as fh:
        return tomllib.load(fh)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_split(value: str) -> str:
    n = int(value[:8], 16) / 0xFFFFFFFF
    return "train" if n < 0.96 else "validation" if n < 0.98 else "test"


def bounded_query(value: str) -> str:
    words = value.split()[:75]
    return " ".join(words)[:600]


def post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int = 4,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json", "Content-Type": "application/json", **headers}
    body = canonical_json(payload).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError(f"POST {url}: expected JSON object")
            return value
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 >= retries:
                raise RuntimeError(f"POST {url}: HTTP {exc.code}: {detail[:2000]}") from exc
        except urllib.error.URLError as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"POST {url}: {exc}") from exc
        time.sleep(min(8.0, 1.25 * (2**attempt)))
    raise AssertionError("unreachable")


def brave_context(query: str, cfg: dict[str, Any], api_key: str, timeout: int) -> dict[str, Any]:
    brave = cfg["brave"]
    payload = {
        "q": bounded_query(query),
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
        {"X-Subscription-Token": api_key},
        payload,
        timeout=timeout,
    )


def extract_grounding(result: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    generic = result.get("grounding", {}).get("generic", [])
    sources = result.get("sources", {})
    generic = generic if isinstance(generic, list) else []
    sources = sources if isinstance(sources, dict) else {}
    out: list[dict[str, Any]] = []
    for item in generic:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        snippets = [str(x) for x in item.get("snippets", []) if str(x).strip()]
        if url and snippets:
            out.append({"url": url, "title": str(item.get("title", "")), "snippets": snippets})
    return out, sources


def source_metadata(items: list[dict[str, Any]], sources: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        url = item["url"]
        if url in seen:
            continue
        seen.add(url)
        meta = sources.get(url, {})
        meta = meta if isinstance(meta, dict) else {}
        out.append(
            {
                "url": url,
                "title": str(item.get("title") or meta.get("title") or ""),
                "hostname": str(meta.get("hostname") or ""),
            }
        )
    return out


def grounding_text(items: list[dict[str, Any]], max_chars: int = 48000) -> str:
    chunks: list[str] = []
    used = 0
    for index, item in enumerate(items, 1):
        chunk = f"SOURCE {index}\nTITLE: {item['title']}\nURL: {item['url']}\n" + "\n".join(item["snippets"])
        if used + len(chunk) > max_chars:
            if max_chars - used > 400:
                chunks.append(chunk[: max_chars - used])
            break
        chunks.append(chunk)
        used += len(chunk)
    return "\n\n".join(chunks)


def teacher_prompt(topic: str, family: str, count: int, context: str) -> str:
    return f"""Generate supervised fine-tuning data for a StarIntel-focused model.

TASK FAMILY: {family}
TOPIC: {topic}
EXAMPLES: {count}
GOAL: {FAMILY_GUIDANCE[family]}

Rules:
- Factual outside-world claims may use only the supplied public-source context.
- Never invent URLs, source titles, observations, tool results, or tests already run.
- Synthetic code, actors, schemas, tool definitions, and hypothetical inputs are allowed when clearly synthetic.
- Keep OSINT examples lawful, public-source, bounded, and non-invasive. No credential theft, unauthorized access, stalking workflows, malware deployment, or evasion instructions.
- Prefer examples with objective validation checks.
- StarIntel examples should preserve provenance, typed messages, bounded concurrency, idempotent ingest, and explicit failure behavior.
- tool-calling examples MUST include a non-empty tools array, an assistant message with tool_calls, at least one tool-role response, and a final assistant synthesis.
- independent-research answers MUST cite at least one exact URL from the supplied context.

Return ONLY JSON:
{{"examples":[{{"task":"{family}","messages":[{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}],"tools":[],"metadata":{{"quality_checks":["..."],"skills":["..."]}}}}]}}

PUBLIC-SOURCE CONTEXT:
{context}
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
    endpoint = str(teacher.get("endpoint", "/v1/chat/completions"))
    return post_json(
        actor_url.rstrip("/") + endpoint,
        {"Authorization": f"Bearer {actor_token}"},
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return strict JSON only. Generate synthetic training data."},
                {"role": "user", "content": prompt},
            ],
            "temperature": float(teacher.get("temperature", 0.45)),
            "max_tokens": int(teacher.get("max_tokens", 8192)),
        },
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
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ).strip()
    raise ValueError("teacher message content is not text")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        first = text.find("\n")
        last = text.rfind("```")
        if first >= 0 and last > first:
            text = text[first + 1 : last].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("teacher output must be a JSON object")
    return value


def validate_family_contract(
    family: str,
    messages: list[dict[str, Any]],
    tools: list[Any],
    sources: list[dict[str, str]],
) -> None:
    if not sources:
        raise ValueError("Brave-grounded example has no source URLs")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError("invalid message role")
        if "content" not in message:
            raise ValueError("message missing content")
    if family == "tool-calling":
        has_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in messages)
        has_result = any(m.get("role") == "tool" for m in messages)
        if not tools or not has_call or not has_result:
            raise ValueError("tool-calling example lacks tools/tool_calls/tool response")
    if family == "independent-research":
        rendered = canonical_json(messages)
        if not any(source["url"] in rendered for source in sources):
            raise ValueError("independent-research answer must cite a Brave source URL")


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
    tools = raw.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    validate_family_contract(family, messages, tools, source_meta)
    metadata = raw.get("metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata = {**metadata, "grounded": True, "brave_query": query, "source_count": len(source_meta), "topic": topic}
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
    return bounded_query(f"{topic} {QUERY_SUFFIX[family]}")


def iter_seed_jobs(cfg: dict[str, Any], only_family: str | None) -> Iterable[tuple[str, str]]:
    for seed in cfg.get("seed", []):
        topic = str(seed["topic"])
        for raw_family in seed.get("families", []):
            family = str(raw_family)
            if family not in ALLOWED_TASKS:
                raise ValueError(f"unknown synthesis family {family!r}")
            if not only_family or family == only_family:
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
            "Confirm that the configured Brave plan permits the intended storage/training use with "
            "STAR_BRAVE_TRAINING_RIGHTS_ACK=1 or --ack-brave-training-rights."
        )
    if not brave_key:
        raise SystemExit("BRAVE_SEARCH_API_KEY is required")
    if not actor_url or not actor_token:
        raise SystemExit("STAR_LLM_ACTOR_URL and STAR_LLM_ACTOR_TOKEN are required")
    if not model or model == "configurable":
        raise SystemExit("STAR_SYNTH_MODEL or --model is required")

    count = args.examples_per_seed or int(cfg.get("examples_per_seed", 4))
    count = max(1, min(count, int(cfg["output"].get("max_examples_per_teacher_call", 8))))
    prompt_version = str(cfg["teacher"].get("prompt_version", "brave-synth-v1"))
    jobs = list(iter_seed_jobs(cfg, args.family))
    random.Random(int(cfg.get("seed", 545))).shuffle(jobs)
    if args.max_jobs is not None:
        jobs = jobs[: max(0, args.max_jobs)]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for topic, family in jobs:
        query = family_query(topic, family)
        brave = brave_context(query, cfg, brave_key, args.timeout)
        grounding, raw_sources = extract_grounding(brave)
        if not grounding:
            print(f"warning: no Brave grounding for {family}: {query}", file=sys.stderr)
            continue
        context = grounding_text(grounding)
        sources = source_metadata(grounding, raw_sources)
        response = teacher_completion(teacher_prompt(topic, family, count, context), cfg, actor_url, actor_token, model, args.timeout)
        examples = parse_json_object(completion_text(response)).get("examples", [])
        if not isinstance(examples, list):
            raise ValueError("teacher examples must be an array")
        accepted = 0
        for raw in examples[:count]:
            if not isinstance(raw, dict):
                continue
            try:
                row = normalize_example(
                    raw,
                    family=family,
                    topic=topic,
                    query=query,
                    source_meta=sources,
                    context_hash=sha256_text(context),
                    teacher_model=model,
                    prompt_version=prompt_version,
                )
            except ValueError as exc:
                print(f"warning: rejected {family} synthetic example: {exc}", file=sys.stderr)
                continue
            if row["content_sha256"] in seen:
                continue
            seen.add(row["content_sha256"])
            rows.append(row)
            accepted += 1
        print(json.dumps({"family": family, "sources": len(sources), "examples": accepted}), file=sys.stderr)

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
    grounding, raw_sources = extract_grounding(brave_context(args.query, cfg, brave_key, args.timeout))
    print(
        json.dumps(
            {"query": args.query, "grounding_items": len(grounding), "sources": source_metadata(grounding, raw_sources)},
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
