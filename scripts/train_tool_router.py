#!/usr/bin/env python3
"""Train a tiny hashed-feature MLP for tool selection.

Input JSONL rows:
  {"input": "research this domain", "tool": "domain_research"}

This is intentionally tokenizer-free and cheap. It is a routing baseline, not a
replacement for the generative model. Use its accuracy as one signal in the
fine-tuning evolution benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import re
from collections import Counter
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required: install torch in the training environment") from exc
    return torch, nn


def read_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with pathlib.Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("input", "")).strip()
            tool = str(row.get("tool", "")).strip()
            if not text or not tool:
                raise ValueError(f"{path}:{line_no}: input and tool are required")
            rows.append({"input": text, "tool": tool})
    if len(rows) < 4:
        raise ValueError("need at least four labeled examples")
    return rows


def bucket(token: str, buckets: int) -> int:
    raw = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(raw, "big") % buckets


def featurize(torch: Any, texts: list[str], buckets: int):
    matrix = torch.zeros((len(texts), buckets), dtype=torch.float32)
    for row_index, text in enumerate(texts):
        counts = Counter(token.lower() for token in TOKEN_RE.findall(text))
        for token, count in counts.items():
            matrix[row_index, bucket(token, buckets)] += 1.0 + min(4.0, float(count - 1)) * 0.25
    norms = matrix.norm(p=2, dim=1, keepdim=True).clamp_min(1.0)
    return matrix / norms


def split(rows: list[dict[str, str]], seed: int, validation_fraction: float):
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1.0 - validation_fraction)))
    cut = min(cut, len(shuffled) - 1)
    return shuffled[:cut], shuffled[cut:]


def make_model(nn: Any, buckets: int, hidden: int, labels: int):
    return nn.Sequential(
        nn.Linear(buckets, hidden),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(hidden, labels),
    )


def accuracy(torch: Any, model: Any, x: Any, y: Any) -> float:
    if not len(y):
        return 0.0
    model.eval()
    with torch.no_grad():
        predicted = model(x).argmax(dim=1)
        return float((predicted == y).float().mean().item())


def cmd_train(args: argparse.Namespace) -> None:
    torch, nn = require_torch()
    torch.manual_seed(args.seed)
    rows = read_rows(args.input)
    train_rows, validation_rows = split(rows, args.seed, args.validation_fraction)
    labels = sorted({row["tool"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}

    train_x = featurize(torch, [row["input"] for row in train_rows], args.buckets)
    train_y = torch.tensor([label_to_id[row["tool"]] for row in train_rows], dtype=torch.long)
    validation_x = featurize(torch, [row["input"] for row in validation_rows], args.buckets)
    validation_y = torch.tensor([label_to_id[row["tool"]] for row in validation_rows], dtype=torch.long)

    model = make_model(nn, args.buckets, args.hidden, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_accuracy = -1.0
    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        score = accuracy(torch, model, validation_x, validation_y)
        if score > best_accuracy:
            best_accuracy = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "feature_buckets": args.buckets,
            "hidden_dim": args.hidden,
            "validation_accuracy": best_accuracy,
        },
        output,
    )
    metadata = {
        "schema": "star-model-tool-router/v1",
        "labels": labels,
        "feature_buckets": args.buckets,
        "hidden_dim": args.hidden,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "validation_accuracy": round(best_accuracy, 6),
    }
    pathlib.Path(str(output) + ".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


def cmd_eval(args: argparse.Namespace) -> None:
    torch, nn = require_torch()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = list(checkpoint["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = read_rows(args.input)
    unknown = sorted({row["tool"] for row in rows if row["tool"] not in label_to_id})
    if unknown:
        raise ValueError(f"evaluation contains unknown tools: {unknown}")

    model = make_model(nn, int(checkpoint["feature_buckets"]), int(checkpoint["hidden_dim"]), len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    x = featurize(torch, [row["input"] for row in rows], int(checkpoint["feature_buckets"]))
    y = torch.tensor([label_to_id[row["tool"]] for row in rows], dtype=torch.long)
    score = accuracy(torch, model, x, y)
    print(json.dumps({"examples": len(rows), "accuracy": round(score, 6)}, sort_keys=True))


def cmd_predict(args: argparse.Namespace) -> None:
    torch, nn = require_torch()
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    labels = list(checkpoint["labels"])
    model = make_model(nn, int(checkpoint["feature_buckets"]), int(checkpoint["hidden_dim"]), len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    x = featurize(torch, [args.text], int(checkpoint["feature_buckets"]))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    ranked = sorted(
        ({"tool": label, "probability": float(probs[index])} for index, label in enumerate(labels)),
        key=lambda item: item["probability"],
        reverse=True,
    )
    print(json.dumps({"input": args.text, "ranking": ranked[: args.top_k]}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--input", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--buckets", type=int, default=4096)
    train.add_argument("--hidden", type=int, default=128)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--validation-fraction", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=545)
    train.set_defaults(fn=cmd_train)

    evaluate = sub.add_parser("eval")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--model", required=True)
    evaluate.set_defaults(fn=cmd_eval)

    predict = sub.add_parser("predict")
    predict.add_argument("--model", required=True)
    predict.add_argument("--text", required=True)
    predict.add_argument("--top-k", type=int, default=3)
    predict.set_defaults(fn=cmd_predict)
    return p


def main() -> None:
    args = parser().parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
