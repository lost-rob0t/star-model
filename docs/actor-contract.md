# llm.starintel.actor fine-tuning contract

`star-model` is the corpus producer. `llm.starintel.actor` owns teacher-model synthesis, provider routing, GPU selection, fine-tune launch, status, budget enforcement, and collection of the resulting adapter/checkpoints.

## Dataset synthesis

Submit:

`POST /v1/dataset/synthesize`

```json
{
  "type": "llm.dataset.synthesize",
  "version": 1,
  "corpus": "star-model",
  "max_examples": 2000,
  "task_families": ["tool-calling", "migrations"],
  "teacher": {
    "policy": "stronger-than-target",
    "provider_policy": "cheapest-allowed"
  },
  "requirements": {
    "provenance": true,
    "content_hash": true,
    "reject_secrets": true
  }
}
```

The submit endpoint returns `202` with a `job_id`. Long teacher calls run outside the Sento receive mailbox.

Poll:

- `GET /v1/dataset/synthesize/:job-id`
- `GET /v1/dataset/synthesize/:job-id/result`

The result endpoint returns canonical NDJSON after completion. The actor uses the configured OpenRouter teacher model, sorts compatible providers by price, allows fallbacks, and defaults to approved `fp16`, `bf16`, or `fp8` endpoints. Per-million-token provider ceilings can be configured with `STAR_LLM_MAX_PROMPT_USD_PER_M` and `STAR_LLM_MAX_COMPLETION_USD_PER_M`. `STAR_LLM_MAX_COMPLETION_TOKENS` bounds each teacher response.

Required synthesis runtime configuration:

- `STAR_LLM_TEACHER_MODEL`
- `OPENROUTER_API_KEY` or `STAR_LLM_OPENROUTER_API_KEY`

Optional synthesis policy:

- `STAR_LLM_SYNTH_BATCH_SIZE`
- `STAR_LLM_ALLOWED_QUANTIZATIONS` (default `fp16,bf16,fp8`)
- `STAR_LLM_MAX_PROMPT_USD_PER_M`
- `STAR_LLM_MAX_COMPLETION_USD_PER_M`
- `STAR_LLM_MAX_COMPLETION_TOKENS`

## Fine-tune launch

`POST /v1/fine-tunes`

```json
{
  "type": "llm.fine_tune.start",
  "version": 1,
  "dataset": {
    "schema": "star-model-dataset-manifest/v1",
    "artifact": {
      "kind": "github-actions-artifact",
      "repository": "lost-rob0t/star-model",
      "run_id": "<actions-run-id>",
      "name": "star-model-dataset-<commit>"
    }
  },
  "training": {
    "base_model": "<configured-model>",
    "method": "qlora",
    "precision": "bf16",
    "max_seq_length": 8192,
    "packing": true,
    "publish_adapter_only": true
  },
  "provider_policy": {
    "inference": "cheapest-allowed",
    "gpu": "vast-ai",
    "gpu_ram_gb_min": 24,
    "max_hourly_gpu_usd": 2.0,
    "max_total_gpu_usd": 20.0
  }
}
```

The endpoint returns `202` with a `job_id`; poll `GET /v1/fine-tunes/:job-id`.

Required fine-tune runtime configuration:

- `STAR_MODEL_GITHUB_TOKEN` with read access to the private dataset artifact
- `VAST_API_KEY`
- current `vastai` CLI on the server PATH

Optional:

- `STAR_LLM_JOB_ROOT`

## Actor responsibilities

1. Validate the request before any billable action.
2. Resolve a compatible Vast.ai GPU offer at or below the hourly cap.
3. Derive a hard training timeout from the selected hourly rate and total-dollar cap.
4. Provision an ephemeral worker.
5. Fetch the immutable GitHub Actions dataset artifact.
6. Stage the dataset and packaged trainer without forwarding StarIntel/OpenRouter/GitHub credentials to the GPU.
7. Verify dataset SHA-256 digests and row counts before model weights are loaded.
8. Run the packaged 4-bit QLoRA worker.
9. Copy adapter/checkpoint output back to the StarIntel server job directory.
10. Destroy the GPU instance on success or failure.
11. Keep provider-specific APIs behind the actor so `star-model` does not depend on the marketplace implementation.

The current implementation collects the trained output into the StarIntel server job directory. Publishing that output to a model registry/object store is a separate deployment policy rather than an implicit upload.

## Security boundary

All synthesis and fine-tune endpoints require the existing authenticated StarIntel administrator principal and are intentionally absent from the public auth-path allowlist. The Common Lisp controller implementation is package-internal; the supported integration boundary is the authenticated HTTP API.
