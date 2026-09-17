#!/usr/bin/env python3
"""Generic 4-bit QLoRA worker for the canonical star-model JSONL format."""

from __future__ import annotations

import argparse
import inspect
import json
import pathlib

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--validation")
    p.add_argument("--output", default="/workspace/output")
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    return p


def render_messages(tokenizer, messages: list[dict]) -> str:
    normalized = [
        {"role": str(m.get("role", "user")), "content": m.get("content", "")}
        for m in messages
    ]
    try:
        return tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=False)
    except Exception:
        parts = []
        for message in normalized:
            content = message["content"]
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True, ensure_ascii=False)
            parts.append(f"<|{message['role']}|>\n{content}")
        return "\n".join(parts) + "\n"


def choose_lora_targets(model) -> list[str]:
    candidates = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "query_key_value", "dense", "fc1", "fc2",
    }
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in candidates
    }
    if not found:
        raise RuntimeError("Could not identify supported LoRA target modules")
    return sorted(found)


def tokenize_dataset(dataset, tokenizer, max_seq_length: int):
    def encode(row):
        text = render_messages(tokenizer, row["messages"])
        return tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )

    return dataset.map(encode, remove_columns=dataset.column_names, desc="tokenize")


def main() -> None:
    args = args_parser().parse_args()
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    targets = choose_lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )

    data_files = {"train": args.train}
    if args.validation:
        data_files["validation"] = args.validation
    raw = load_dataset("json", data_files=data_files)
    train = tokenize_dataset(raw["train"], tokenizer, args.max_seq_length)
    validation = (
        tokenize_dataset(raw["validation"], tokenizer, args.max_seq_length)
        if "validation" in raw
        else None
    )

    kwargs = dict(
        output_dir=str(output / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if validation is not None:
        if "eval_strategy" in params:
            kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in params:
            kwargs["evaluation_strategy"] = "steps"
        kwargs["eval_steps"] = 100

    training_args = TrainingArguments(**kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    result = trainer.train()
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)

    metrics = dict(result.metrics)
    if validation is not None:
        metrics["validation"] = trainer.evaluate()
    (output / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "method": "qlora",
                "quantization": "nf4-double-quant",
                "compute_dtype": str(compute_dtype),
                "lora_targets": targets,
                "max_seq_length": args.max_seq_length,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
