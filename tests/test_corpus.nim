import std/[json, unittest]
import starmodel/corpus

proc validRecord(): JsonNode =
  %*{
    "id": "example-1",
    "task": "tool-calling",
    "messages": [
      {"role": "user", "content": "Find the public source."},
      {"role": "assistant", "content": "I will verify it with the configured tool."}
    ],
    "provenance": {
      "kind": "synthetic",
      "generator": "llm.starintel.actor",
      "model": "teacher"
    },
    "content_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "split": "train"
  }

suite "star-model corpus adapter":
  test "accepts canonical record":
    check validateExample(validRecord()).len == 0

  test "rejects unknown task":
    var record = validRecord()
    record["task"] = %"unknown"
    check validateExample(record).len > 0

  test "rejects malformed digest":
    var record = validRecord()
    record["content_sha256"] = %"deadbeef"
    check validateExample(record).len > 0

  test "extracts actor examples":
    let response = %*{"examples": [validRecord()]}
    check extractExamples(response).len == 1
