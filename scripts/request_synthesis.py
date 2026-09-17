#!/usr/bin/env python3
"""Submit an llm.starintel.actor synthesis job and download its NDJSON result."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import urllib.error
import urllib.request


RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


def request(
    *,
    method: str,
    url: str,
    token: str,
    body: bytes | None = None,
    attempts: int = 5,
) -> bytes:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, application/x-ndjson",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"

    delay = 1.0
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            if exc.code not in RETRYABLE or attempt == attempts:
                raise RuntimeError(f"HTTP {exc.code} from {url}: {payload}") from exc
        except urllib.error.URLError as exc:
            if attempt == attempts:
                raise RuntimeError(f"request to {url} failed: {exc}") from exc
        time.sleep(delay)
        delay = min(delay * 2.0, 15.0)
    raise AssertionError("unreachable")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--poll-seconds", type=float, default=10.0)
    p.add_argument("--timeout-seconds", type=int, default=6 * 60 * 60)
    args = p.parse_args()

    base = args.url.rstrip("/")
    request_body = pathlib.Path(args.request).read_bytes()
    receipt = json.loads(
        request(
            method="POST",
            url=f"{base}/v1/dataset/synthesize",
            token=args.token,
            body=request_body,
        )
    )
    job_id = receipt.get("job_id")
    if not job_id:
        raise RuntimeError(f"synthesis submit response has no job_id: {receipt!r}")

    deadline = time.monotonic() + args.timeout_seconds
    status_url = f"{base}/v1/dataset/synthesize/{job_id}"
    result_url = f"{status_url}/result"

    while True:
        status_doc = json.loads(
            request(method="GET", url=status_url, token=args.token)
        )
        status = status_doc.get("status")
        print(json.dumps({"job_id": job_id, "status": status}), flush=True)

        if status == "succeeded":
            result = request(method="GET", url=result_url, token=args.token)
            output = pathlib.Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(result)
            print(json.dumps({"job_id": job_id, "output": str(output)}))
            return
        if status == "failed":
            raise RuntimeError(
                f"synthesis job {job_id} failed: {status_doc.get('error', 'unknown error')}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"synthesis job {job_id} exceeded {args.timeout_seconds} seconds"
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
