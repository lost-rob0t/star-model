# Synthetic task contract

`llm.starintel.actor` generates teacher examples for the following task families. The actor must emit canonical `schema/example.schema.json` JSONL and preserve the teacher model/provider/prompt version in provenance.

## starintel-prolog-synthetic
Generate StarIntel-flavored facts, schemas, predicates, rules, constraints, queries, explanation tasks, counterexamples, repair tasks, and migrations. Include both valid and deliberately invalid cases with corrected outputs. Prefer multi-step symbolic reasoning over trivia.

## starintel-json-synthetic
Generate realistic StarIntel JSON and JSON-LD documents, schema conversions, validation/repair cases, actor manifests, dataset manifests, target definitions, document lifecycle events, and cross-version transformations.

## tool-calling
Use a teacher model stronger than the fine-tune target. Produce multi-turn tool-use traces with tool schemas, arguments, tool results, error handling, retries, pagination, idempotency, and final synthesis. Never fabricate successful tool output: simulated tools must be clearly marked as simulations and internally consistent.

## ingest-jobs
Generate ingestion definitions and troubleshooting traces: source discovery, fetch, parse, normalize, dedupe, provenance, backpressure, retry, checkpoint, partial failure, and resume behavior.

## migrations
Generate schema/data migrations with preconditions, up/down behavior, compatibility windows, rollback strategy, validation queries, and failure recovery. Cover CouchDB, relational stores, JSON-LD, Prolog facts, and actor message contracts where appropriate.

## osint-tool-design
Generate lawful public-source research tool design tasks: source adapters, rate limiting, robots/terms-aware collection, provenance, entity resolution, deduplication, confidence scoring, auditability, and safe failure handling. Do not generate credential theft, bypass instructions, invasive surveillance, or exploitation guidance.

## Teacher quality gate
Each candidate should be checked by a second pass that can reject it for incorrect tool semantics, unsupported claims, invalid Prolog/JSON, secret-like strings, private data, inconsistent simulated responses, or duplicated content.
