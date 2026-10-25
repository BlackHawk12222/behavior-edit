"""Tiny experimental LoRA editing on the same 0.5B base-model checkpoint.

All evaluation prompts must be held out from adapter_examples.json.
This is deliberately minimal and does not modify the upstream editors.
"""

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from .metrics import grade, summarize
from .runner import LocalModel

BASE_SYSTEM = "You are a helpful assistant."


def read_examples(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        samples = json.load(handle)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Examples file must be a non-empty JSON list")
    seen_ids = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Every example must be a JSON object")
        for key in ("id", "prompt", "answer"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"Missing or invalid example field: {key}")
        if sample["id"] in seen_ids:
            raise ValueError(f"Duplicate training ID: {sample['id']}")
        seen_ids.add(sample["id"])
    return samples


def supervised_batch(tokenizer, example, max_length, device):
    """Only assistant answer tokens contribute to causal language-model loss."""
    messages = [
        {"role": "system", "content": BASE_SYSTEM},
        {"role": "user", "content": example["prompt"]},
    ]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": example["answer"]}],
        tokenize=False, add_generation_prompt=False
    )
    prompt_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    if prompt_len >= encoded["input_ids"].shape[1] - 1:
        raise ValueError(
            f"Example {example['id']} truncated before the answer; raise --max-length"
        )
    encoded["labels"] = encoded["input_ids"].clone()
    encoded["labels"][:, :prompt_len] = -100
    return encoded


def eval_phase(local_model, tasks, phase, token_limit):
    records = []
    for task in tasks:
        response, tokens, elapsed = local_model.generate(
            BASE_SYSTEM, task["prompt"], max_new_tokens=token_limit
        )
        records.append({
            "condition": phase,
            "task_id": task["id"],
            "task": task,
            "response": response,
            "generated_tokens": tokens,
            "latency_seconds": round(elapsed, 4),
            "metrics": grade(task, response),
        })
        print(f"{phase:8} {task['id']:18} passed={records[-1]['metrics']['passed']}")
    return records


def train_adapter(args, tasks):
    if (
        args.steps < 1 or args.rank < 1 or args.max_length < 16
        or args.max_new_tokens < 1 or args.eval_limit < 1
        or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        raise ValueError("Steps, rank, length, eval limit and learning rate must be positive")
    examples = read_examples(args.examples)
    held_out = tasks[:args.eval_limit]
    eval_prompts = {t["prompt"] for t in tasks}
    if any(example["prompt"] in eval_prompts for example in examples):
        raise ValueError("Training and evaluation prompts overlap; separate the datasets")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(examples)

    local_model = LocalModel(args.model, revision=args.revision, device=args.device)
    baseline = eval_phase(local_model, held_out, "before", args.max_new_tokens)

    local_model.model.config.use_cache = False
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    local_model.model = get_peft_model(local_model.model, config)
    local_model.model.train()
    parameters = [p for p in local_model.model.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("LoRA did not enable any trainable parameters")
    opt = torch.optim.AdamW(parameters, lr=args.learning_rate, foreach=False)
    loss_history = []
    for step in range(args.steps):
        example = examples[step % len(examples)]
        inputs = supervised_batch(
            local_model.tokenizer, example, args.max_length, local_model.device
        )
        opt.zero_grad(set_to_none=True)
        loss = local_model.model(**inputs).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        opt.step()
        loss_value = round(loss.detach().float().item(), 6)
        loss_history.append(loss_value)
        print(f"train step {step + 1}/{args.steps}: loss={loss_value:.5f}")

    local_model.model.eval()
    local_model.model.config.use_cache = True
    after = eval_phase(local_model, held_out, "after", args.max_new_tokens)
    records = baseline + after

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    save_dir = args.output_dir / run_id
    adapter_dir = save_dir / "adapter"
    save_dir.mkdir(parents=True, exist_ok=False)
    local_model.model.save_pretrained(str(adapter_dir), safe_serialization=True)

    with (save_dir / "comparison.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "run_id": run_id,
        "model_id": args.model,
        "resolved_revision": local_model.resolved_revision,
        "device": local_model.device,
        "seed": args.seed,
        "train_example_ids": [ex["id"] for ex in examples],
        "steps": args.steps,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "loss_history": loss_history,
        "evaluation": summarize(records),
        "adapter_path": str(adapter_dir),
        "warning": "Small proof of concept only: no evidence of personality editing or generalization.",
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nAdapter saved: {adapter_dir}")
    print(f"Before/after scores: {json.dumps(summary['evaluation'], indent=2)}")
    return 0
