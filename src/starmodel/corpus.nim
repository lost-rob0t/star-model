import std/[json, os, osproc, streams, strutils, times]

const
  AllowedTasks* = [
    "code",
    "prolog",
    "starintel-prolog-synthetic",
    "starintel-json-synthetic",
    "tool-calling",
    "ingest-jobs",
    "migrations",
    "osint-tool-design"
  ]
  AllowedSplits* = ["train", "validation", "test"]

proc isHexSha256(value: string): bool =
  if value.len != 64:
    return false
  for ch in value:
    if ch notin {'0'..'9', 'a'..'f'}:
      return false
  true

proc validateExample*(record: JsonNode): seq[string] =
  if record.kind != JObject:
    return @["record must be a JSON object"]

  for key in ["id", "task", "messages", "provenance", "content_sha256", "split"]:
    if not record.hasKey(key):
      result.add("missing required field: " & key)

  if result.len > 0:
    return

  if record["id"].kind != JString or record["id"].getStr.len == 0:
    result.add("id must be a non-empty string")

  if record["task"].kind != JString or record["task"].getStr notin AllowedTasks:
    result.add("unknown task family")

  if record["messages"].kind != JArray or record["messages"].len < 2:
    result.add("messages must contain at least two messages")

  if record["provenance"].kind != JObject or not record["provenance"].hasKey("kind"):
    result.add("provenance.kind is required")

  if record["content_sha256"].kind != JString or
      not isHexSha256(record["content_sha256"].getStr):
    result.add("content_sha256 must be 64 lowercase hex characters")

  if record["split"].kind != JString or record["split"].getStr notin AllowedSplits:
    result.add("split must be train, validation, or test")

proc extractExamples*(response: JsonNode): seq[JsonNode] =
  if response.kind == JArray:
    for item in response:
      result.add(item)
    return

  if response.kind != JObject:
    raise newException(ValueError, "synthesis response must be an object or array")

  for key in ["examples", "records", "data"]:
    if response.hasKey(key) and response[key].kind == JArray:
      for item in response[key]:
        result.add(item)
      return

  raise newException(
    ValueError,
    "synthesis response contains no inline examples; resolve its artifact reference first"
  )

proc writeSynthesizedJsonl*(response: JsonNode; outputPath: string): int =
  let examples = extractExamples(response)
  var output = open(outputPath, fmWrite)
  defer: output.close()

  for i, record in examples:
    let errors = validateExample(record)
    if errors.len > 0:
      raise newException(
        ValueError,
        "invalid synthesized record " & $i & ": " & errors.join("; ")
      )
    output.writeLine($record)
    inc result

iterator jsonlRecords*(path: string): JsonNode =
  var lineNo = 0
  for rawLine in lines(path):
    inc lineNo
    let line = rawLine.strip()
    if line.len == 0:
      continue
    try:
      yield parseJson(line)
    except JsonParsingError as exc:
      raise newException(
        ValueError,
        path & ":" & $lineNo & ": invalid JSON: " & exc.msg
      )

proc sha256File*(path: string): string =
  let process = startProcess(
    "sha256sum",
    args = [path],
    options = {poUsePath, poStdErrToStdOut}
  )
  defer: process.close()

  let output = process.outputStream.readAll()
  let code = process.waitForExit()
  if code != 0:
    raise newException(IOError, "sha256sum failed for " & path & ": " & output.strip())

  let fields = output.splitWhitespace()
  if fields.len == 0 or not isHexSha256(fields[0]):
    raise newException(IOError, "sha256sum returned an invalid digest for " & path)
  fields[0]

proc incCount(counts: JsonNode; key: string) =
  if counts.hasKey(key):
    counts[key] = %(counts[key].getInt + 1)
  else:
    counts[key] = %1

proc buildManifest*(jsonlPath: string; name = "star-model-corpus"): JsonNode =
  var
    total = 0
    taskCounts = newJObject()
    splitCounts = newJObject()

  for record in jsonlRecords(jsonlPath):
    let errors = validateExample(record)
    if errors.len > 0:
      raise newException(
        ValueError,
        "invalid corpus record " & $total & ": " & errors.join("; ")
      )
    inc total
    incCount(taskCounts, record["task"].getStr)
    incCount(splitCounts, record["split"].getStr)

  %*{
    "schema": "star-model-dataset-manifest/v1",
    "name": name,
    "created_at": now().utc.format("yyyy-MM-dd'T'HH:mm:ss'Z'"),
    "examples": total,
    "task_counts": taskCounts,
    "split_counts": splitCounts,
    "artifact": {
      "path": jsonlPath,
      "bytes": int64(getFileSize(jsonlPath)),
      "sha256": sha256File(jsonlPath)
    }
  }

proc saveManifest*(manifest: JsonNode; outputPath: string) =
  writeFile(outputPath, pretty(manifest, 2) & "\n")
