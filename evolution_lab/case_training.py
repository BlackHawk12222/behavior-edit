"""Train controlled LoRA edits from trusted-key generated training cases."""

from __future__ import annotations

import math
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from .experiment import fit, norm_matched_random, write_json
from .provenance import checkpoint_hashes, environment, fingerprint
from .runner import LocalModel
from .train_adapter import lora_delta_stats
from .training_cohort import parse_task_exposure_ratio, prepare_training_examples

MODEL_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def _atomic_manifest(path: Path, manifest: dict) -> None:
    write_json(path.with_suffix(".json.tmp"), manifest)
    path.with_suffix(".json.tmp").replace(path)


def train_case_adapters(args) -> int:
    if (args.steps < 1 or args.rank < 1 or args.max_length < 16
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("Steps, rank, max length and learning rate must be positive")
    if not MODEL_REVISION.fullmatch(args.revision):
        raise ValueError("Case-adapter training requires an immutable 40-character model revision")

    source, all_examples = prepare_training_examples(args.dataset)
    sampling_strategy = getattr(args, "sampling_strategy", "uniform")
    if sampling_strategy not in {"uniform", "task-balanced", "task-ratio"}:
        raise ValueError("Sampling strategy must be uniform, task-balanced or task-ratio")
    ratio = parse_task_exposure_ratio(getattr(args, "task_exposure_ratio", None))
    schedule_seed = getattr(args, "schedule_seed", None)
    requested_example_sampling = getattr(args, "example_sampling", None)
    if requested_example_sampling is not None and sampling_strategy != "task-ratio":
        raise ValueError("Example sampling policy is only configurable for task-ratio sampling")
    example_sampling = requested_example_sampling or "with-replacement"
    if schedule_seed is not None and sampling_strategy != "task-ratio":
        raise ValueError("An independent schedule seed is only supported with task-ratio sampling")
    cohort_path = getattr(args, "cohort_manifest", None)
    cohort_manifest = None
    if cohort_path:
        cohort_path = Path(cohort_path)
        cohort_manifest = json.loads(cohort_path.read_text())
        if fingerprint(source["manifest"]) != cohort_manifest.get("source_dataset_manifest_sha256"):
            raise ValueError("Fixed training cohort belongs to a different dataset manifest")
        ids = cohort_manifest.get("training_ids")
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Fixed training cohort IDs must be a unique list")
        by_id = {row["id"]: row for row in all_examples}
        if any(case_id not in by_id for case_id in ids):
            raise ValueError("Fixed training cohort contains an ID outside the train split")
        examples = [by_id[case_id] for case_id in ids]
        if fingerprint(examples) != cohort_manifest.get("training_data_sha256"):
            raise ValueError("Fixed training cohort content hash does not match its manifest")
        data_file = cohort_path.parent / cohort_manifest["cohort_examples_file"]
        if data_file.parent.resolve() != cohort_path.parent.resolve():
            raise ValueError("Fixed cohort evidence file must be adjacent to its manifest")
        data_bytes = data_file.read_bytes()
        data_hash = __import__("hashlib").sha256(data_bytes).hexdigest()
        if data_hash != cohort_manifest.get("cohort_examples_sha256"):
            raise ValueError("Fixed cohort evidence file hash does not match its manifest")
        try:
            saved_examples = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines()]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Fixed cohort evidence file is not valid UTF-8 JSONL") from error
        if fingerprint(saved_examples) != cohort_manifest.get("training_data_sha256"):
            raise ValueError("Fixed cohort evidence rows do not match the cohort content hash")
        if saved_examples != examples:
            raise ValueError("Fixed cohort evidence rows differ from the validated dataset train split")
    else:
        examples = all_examples
    if sampling_strategy == "task-ratio":
        if not cohort_manifest or not ratio:
            raise ValueError("Task-ratio training requires a fixed cohort and task exposure ratio")
        if set(ratio) != {row["task_type"] for row in examples}:
            raise ValueError("Exposure-ratio task names must match the fixed cohort exactly")
        if sum(ratio.values()) != args.steps:
            raise ValueError("Exposure-ratio counts must sum to --steps")
    elif ratio is not None or cohort_manifest is not None:
        raise ValueError("Fixed cohort and exposure ratio are only supported with task-ratio sampling")
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
        "task_exposure_ratio": ratio,
        "example_sampling": example_sampling if sampling_strategy == "task-ratio" else None,
        "schedule_seed": schedule_seed if sampling_strategy == "task-ratio" else None,
        "task_order_design": (
            ("common-random-task-positions-v1"
             if len({row.get("task_type", "default") for row in examples}) == 2
             else "seeded-multiset-shuffle-v1")
            if schedule_seed is not None
            else "legacy-seeded-shuffle-v1" if sampling_strategy == "task-ratio"
            else None
        ),
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
        "schedule_seed": schedule_seed if sampling_strategy == "task-ratio" else None,
        "behavior": source["manifest"]["generator"],
        "dataset_generator": source["manifest"]["generator"],
        "dataset_seed": source["manifest"]["seed"],
        "dataset_manifest_sha256": fingerprint(source["manifest"]),
        "training_sha256": fingerprint(examples),
        "training_count": len(examples),
        "source_training_count": len(all_examples),
        "training_family_count": len({row["family"] for row in examples}),
        "training_task_type_counts": {
            kind: sum(row["task_type"] == kind for row in examples)
            for kind in sorted({row["task_type"] for row in examples})
        },
        "training_ids": [row["id"] for row in examples],
        "settings": settings,
        "fixed_training_cohort": ({
            "cohort_id": cohort_manifest["cohort_id"],
            "manifest_sha256": __import__("hashlib").sha256(cohort_path.read_bytes()).hexdigest(),
            "training_data_sha256": cohort_manifest["training_data_sha256"],
            "training_family_counts": cohort_manifest["training_family_counts"],
            "sentence_subject_number_counts": cohort_manifest["sentence_subject_number_counts"],
        } if cohort_manifest else None),
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
                      args.max_length, args.seed, sampling_strategy,
                      task_exposure_ratio=ratio, schedule_seed=schedule_seed,
                      example_sampling=example_sampling)
        write_json(destination / "training.json", history)
        seen_ids = list(dict.fromkeys(row["example_id"] for row in history))
        manifest["training_updates"] = len(history)
        manifest["training_examples_seen_unique"] = len(seen_ids)
        manifest["training_ids_seen"] = seen_ids
        manifest["training_task_exposure_counts"] = {
            kind: sum(row["task_type"] == kind for row in history)
            for kind in sorted({row["task_type"] for row in history})
        }
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
