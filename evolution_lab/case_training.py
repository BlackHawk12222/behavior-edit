"""Train controlled LoRA edits from trusted-key generated training cases."""

from __future__ import annotations

import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from .case_eval import load_cases
from .experiment import fit, norm_matched_random, write_json
from .provenance import checkpoint_hashes, environment, fingerprint
from .runner import LocalModel
from .train_adapter import lora_delta_stats

MODEL_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def prepare_training_examples(dataset: Path) -> tuple[dict, list[dict]]:
    """Validate the full dataset, then return only trusted train-split targets."""
    source, cases = load_cases(Path(dataset), "train")
    if not cases or any(row.get("split") != "train" for row in cases):
        raise ValueError("Case training requires non-empty rows from the train split only")
    seen_ids: set[str] = set()
    examples = []
    for row in cases:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen_ids:
            raise ValueError("Training case IDs must be present and unique")
        seen_ids.add(case_id)
        examples.append({
            "id": case_id,
            "prompt": row["prompt"],
            "answer": row["answer"],
            "family": row["family"],
            "task_type": row.get("task_type", "arithmetic"),
            "split": row["split"],
        })
    return source, examples


def _atomic_manifest(path: Path, manifest: dict) -> None:
    write_json(path.with_suffix(".json.tmp"), manifest)
    path.with_suffix(".json.tmp").replace(path)


def train_case_adapters(args) -> int:
    if (args.steps < 1 or args.rank < 1 or args.max_length < 16
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Steps, rank, max length and learning rate must be positive")
    if not MODEL_REVISION.fullmatch(args.revision):
        raise ValueError("Case-adapter training requires an immutable 40-character model revision")

    source, examples = prepare_training_examples(args.dataset)
    sampling_strategy = getattr(args, "sampling_strategy", "uniform")
    if sampling_strategy not in {"uniform", "task-balanced"}:
        raise ValueError("Sampling strategy must be uniform or task-balanced")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination = output_root / run_id
    destination.mkdir(parents=False, exist_ok=False)
    settings = {
        "steps": args.steps,
        "rank": args.rank,
        "lora_alpha": 2 * args.rank,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "max_length": args.max_length,
        "precision": args.precision,
        "decoding": "greedy",
        "sampling_strategy": sampling_strategy,
    }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "loading_model",
        "smoke_only": False,
        "environment": environment(),
        "model_id": args.model,
        "requested_revision": args.revision,
        "resolved_revision": None,
        "device": args.device,
        "precision": args.precision,
        "seed": args.seed,
        "behavior": source["manifest"]["generator"],
        "dataset_generator": source["manifest"]["generator"],
        "dataset_seed": source["manifest"]["seed"],
        "dataset_manifest_sha256": fingerprint(source["manifest"]),
        "training_sha256": fingerprint(examples),
        "training_count": len(examples),
        "training_family_count": len({row["family"] for row in examples}),
        "training_task_type_counts": {
            kind: sum(row["task_type"] == kind for row in examples)
            for kind in sorted({row["task_type"] for row in examples})
        },
        "training_ids": [row["id"] for row in examples],
        "settings": settings,
        "controls": ["lora", "norm-matched-random"],
        "warning": "Training provenance only. Evaluate against base, prompt, random and locality controls; this does not establish a useful or general edit.",
    }
    _atomic_manifest(destination / "manifest.json", manifest)
    try:
        local = LocalModel(args.model, revision=args.revision,
                           device=args.device, precision=args.precision)
        manifest["resolved_revision"] = local.resolved_revision
        manifest["device"] = local.device
        manifest["precision"] = local.precision
        _atomic_manifest(destination / "manifest.json", manifest)

        local.model.config.use_cache = False
        local.model = get_peft_model(local.model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            inference_mode=False,
        ))
        local.model.eval()
        history = fit(local, examples, args.steps, args.learning_rate,
                      args.max_length, args.seed, sampling_strategy)
        write_json(destination / "training.json", history)
        seen_ids = list(dict.fromkeys(row["example_id"] for row in history))
        manifest["training_updates"] = len(history)
        manifest["training_examples_seen_unique"] = len(seen_ids)
        manifest["training_ids_seen"] = seen_ids
        manifest["training_update_log_sha256"] = fingerprint(history)
        delta_lora = lora_delta_stats(local.model, torch)
        write_json(destination / "delta_lora.json", delta_lora)
        local.model.peft_config["default"].revision = local.resolved_revision
        lora_dir = destination / "adapter_lora"
        local.model.save_pretrained(str(lora_dir), safe_serialization=True)

        norm_matched_random(local.model, torch, args.seed + 1000003)
        delta_random = lora_delta_stats(local.model, torch)
        write_json(destination / "delta_random.json", delta_random)
        for trained, random_delta in zip(delta_lora["layers"], delta_random["layers"]):
            if (trained["name"] != random_delta["name"]
                    or not math.isclose(trained["delta_frobenius_norm"],
                                        random_delta["delta_frobenius_norm"],
                                        rel_tol=1e-4, abs_tol=1e-8)):
                raise RuntimeError("Random control delta norm does not match LoRA by layer")
        random_dir = destination / "adapter_random"
        local.model.save_pretrained(str(random_dir), safe_serialization=True)

        manifest["adapter_hashes"] = {
            "lora": checkpoint_hashes(lora_dir),
            "random": checkpoint_hashes(random_dir),
        }
        manifest["delta_paths"] = {
            "lora": "delta_lora.json",
            "random": "delta_random.json",
        }
        manifest["status"] = "complete"
        _atomic_manifest(destination / "manifest.json", manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        _atomic_manifest(destination / "manifest.json", manifest)
        raise
    print(f"Case adapters saved: {destination}", flush=True)
    return 0
