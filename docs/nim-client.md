# Nim actor client and corpus adapter

The Nim control-plane client keeps `star-model` provider-neutral. It talks only to `llm.starintel.actor`; RunPod, Vast, or other GPU marketplace credentials and APIs stay behind the actor.

## Example

```nim
import std/[json, os]
import starmodel

let client = newActorClient(
  getEnv("STAR_LLM_ACTOR_URL", "http://127.0.0.1:8080"),
  getEnv("STAR_LLM_ACTOR_TOKEN")
)

let response = client.synthesize(
  @["tool-calling", "migrations", "osint-tool-design"],
  maxExamples = 10_000
)

discard writeSynthesizedJsonl(response, "build/synthetic.jsonl")
let manifest = buildManifest("build/synthetic.jsonl", "star-model-evolution-001")
saveManifest(manifest, "build/manifest.json")

let job = client.startFineTune(
  manifest,
  baseModel = "configured-8b-model",
  maxHourlyGpuUsd = 0.80,
  maxTotalGpuUsd = 25.0
)

echo pretty(job)
```

## Invariants

- Synthesized examples must pass the canonical record validator before they are persisted.
- Every example must carry provenance and a lowercase SHA-256 content hash.
- The generated manifest records task/split counts plus an immutable SHA-256 digest of the JSONL artifact.
- GPU hourly and total cost caps are part of every fine-tune launch request.
- Provider-specific APIs, keys, retries, and instance lifecycle logic stay inside `llm.starintel.actor`.
- Ephemeral GPU workers must be terminated on success, failure, timeout, or budget exhaustion.

`sha256sum` from GNU coreutils is required by the manifest builder.
