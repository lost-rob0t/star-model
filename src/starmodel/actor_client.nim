import std/[httpclient, json, strformat, strutils]

type
  ActorError* = object of CatchableError

  ActorClient* = object
    baseUrl*: string
    token*: string
    timeoutMs*: int

proc newActorClient*(baseUrl: string; token = ""; timeoutMs = 120_000): ActorClient =
  result.baseUrl = baseUrl.strip(chars = {'/'})
  result.token = token
  result.timeoutMs = timeoutMs

proc requestJson(client: ActorClient; path: string; httpMethod: HttpMethod;
                 body: JsonNode = nil): JsonNode =
  let http = newHttpClient(timeout = client.timeoutMs)
  defer: http.close()

  var headers = newHttpHeaders({"Accept": "application/json"})
  if body != nil:
    headers["Content-Type"] = "application/json"
  if client.token.len > 0:
    headers["Authorization"] = "Bearer " & client.token
  http.headers = headers

  let response = http.request(
    client.baseUrl & path,
    httpMethod = httpMethod,
    body = if body == nil: "" else: $body
  )

  let code = int(response.code)
  if code < 200 or code >= 300:
    raise newException(
      ActorError,
      &"llm.starintel.actor {httpMethod} {path} failed: HTTP {code}: {response.body}"
    )

  if response.body.strip().len == 0:
    return newJNull()
  parseJson(response.body)

proc synthesize*(client: ActorClient; taskFamilies: seq[string]; maxExamples = 2_000;
                 corpus = "star-model"; teacherPolicy = "stronger-than-target";
                 providerPolicy = "cheapest-allowed"): JsonNode =
  let payload = %*{
    "type": "llm.dataset.synthesize",
    "version": 1,
    "corpus": corpus,
    "max_examples": maxExamples,
    "task_families": taskFamilies,
    "teacher": {
      "policy": teacherPolicy,
      "provider_policy": providerPolicy
    },
    "requirements": {
      "provenance": true,
      "content_hash": true,
      "reject_secrets": true
    }
  }
  client.requestJson("/v1/dataset/synthesize", HttpPost, payload)

proc startFineTune*(client: ActorClient; datasetManifest: JsonNode; baseModel: string;
                    maxHourlyGpuUsd = 0.80; maxTotalGpuUsd = 25.0;
                    maxSeqLength = 8_192; method = "qlora";
                    precision = "bf16"; packing = true;
                    publishAdapterOnly = true): JsonNode =
  let payload = %*{
    "type": "llm.fine_tune.start",
    "version": 1,
    "dataset": datasetManifest,
    "training": {
      "base_model": baseModel,
      "method": method,
      "precision": precision,
      "max_seq_length": maxSeqLength,
      "packing": packing,
      "publish_adapter_only": publishAdapterOnly
    },
    "provider_policy": {
      "inference": "cheapest-allowed",
      "gpu": "cheapest-compatible",
      "max_hourly_gpu_usd": maxHourlyGpuUsd,
      "max_total_gpu_usd": maxTotalGpuUsd
    }
  }
  client.requestJson("/v1/fine-tunes", HttpPost, payload)

proc fineTuneStatus*(client: ActorClient; jobId: string): JsonNode =
  client.requestJson("/v1/fine-tunes/" & jobId, HttpGet)
