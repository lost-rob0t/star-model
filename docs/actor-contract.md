# llm.starintel.actor fine-tuning contract

`star-model` is a corpus producer. `llm.starintel.actor` is the orchestration authority for synthesis and fine-tuning.

## Dataset synthesis

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

The actor returns canonical JSONL records. For teacher tool-use traces, the actor may use an OpenAI-compatible provider such as OpenRouter, but provider selection, key management, budgets, retries, and model selection stay inside the actor.

## Fine-tune launch

`POST /v1/fine-tunes`

```json
{
  "type": "llm.fine_tune.start",
  "version": 1,
  "dataset": {"schema": "star-model-dataset-manifest/v1"},
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
    "gpu": "cheapest-compatible",
    "max_hourly_gpu_usd": 2.0,
    "max_total_gpu_usd": 20.0
  }
}
```

## Actor responsibilities

1. Resolve a compatible GPU offer without exceeding policy caps.
2. Provision an ephemeral worker and inject only short-lived scoped credentials.
3. Fetch the immutable dataset artifact by digest.
4. Verify hashes and train/validation/test counts.
5. Launch the configured fine-tuning backend.
6. Stream structured status events and metrics.
7. Persist checkpoints outside the ephemeral worker.
8. Stop the worker on success, failure, timeout, or budget exhaustion.
9. Publish the adapter/model artifact plus a model card containing base model, dataset manifest digest, hyperparameters, code revision, and evaluation results.
10. Keep provider-specific APIs behind actor adapters so `star-model` never depends on one GPU marketplace.

## Suggested events

- `llm.fine_tune.queued`
- `llm.fine_tune.provisioning`
- `llm.fine_tune.running`
- `llm.fine_tune.checkpoint`
- `llm.fine_tune.evaluating`
- `llm.fine_tune.publishing`
- `llm.fine_tune.succeeded`
- `llm.fine_tune.failed`
- `llm.fine_tune.budget_exhausted`

Every event should carry `job_id`, dataset digest, base model, actor fencing token, timestamp, provider adapter, and monotonic sequence number.
