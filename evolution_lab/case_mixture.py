"""Build a fixed equal mixture of compatible case-trained LoRA adapters."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .case_eval import validate_norm_match
from .experiment import norm_matched_random, write_json
from .provenance import checkpoint_hashes, environment, fingerprint
from .runner import LocalModel
from .train_adapter import lora_delta_stats


def _flatten_selected_adapter(directory: Path, adapter_name: str) -> None:
    """Move PEFT's named-adapter subfolder to the flat layout case-eval expects."""
    directory = Path(directory)
    nested = directory / adapter_name
    if nested.is_dir():
        for source in nested.iterdir():
            destination = directory / source.name
            if destination.exists():
                raise ValueError(f"Unexpected duplicate adapter artifact: {destination}")
            shutil.move(str(source), str(destination))
        nested.rmdir()
    required = (directory / "adapter_config.json", directory / "adapter_model.safetensors")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"PEFT did not save a complete adapter in {directory}")


def validate_mixture_sources(run_dirs: list[Path]) -> tuple[list[dict], dict]:
    """Validate provenance and checkpoint identity before combining adapters."""
    if len(run_dirs) != 2 or Path(run_dirs[0]).resolve() == Path(run_dirs[1]).resolve():
        raise ValueError("A case mixture requires two distinct training runs")
    manifests = []
    for run_dir in run_dirs:
        manifest_path = Path(run_dir) / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing training manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete":
            raise ValueError(f"Mixture sources must be complete training runs: {run_dir}")
        expected = manifest.get("adapter_hashes", {}).get("lora")
        actual = checkpoint_hashes(Path(run_dir) / "adapter_lora")
        if not expected or expected != actual:
            raise ValueError(f"Source LoRA checkpoint changed since training: {run_dir}")
        manifests.append(manifest)

    first, second = manifests
    for field in ("model_id", "resolved_revision", "dataset_manifest_sha256", "seed"):
        if first.get(field) != second.get(field):
            raise ValueError(f"Incompatible case-mixture sources: {field}")
    first_settings = dict(first.get("settings", {}))
    second_settings = dict(second.get("settings", {}))
    first_sampler = first_settings.pop("sampling_strategy", None)
    second_sampler = second_settings.pop("sampling_strategy", None)
    if first_settings != second_settings:
        raise ValueError("Mixture sources must match all training settings except sampling strategy")
    if not first_sampler or not second_sampler or first_sampler == second_sampler:
        raise ValueError("Mixture sources must use distinct, recorded sampling strategies")
    if first.get("precision") != second.get("precision"):
        raise ValueError("Mixture sources must use the same training precision")

    first_ids = set(first.get("training_ids_seen", []))
    second_ids = set(second.get("training_ids_seen", []))
    provenance = {
        "source_run_ids": [first.get("run_id"), second.get("run_id")],
        "source_sampling_strategies": [first_sampler, second_sampler],
        "source_adapter_hashes": [
            first["adapter_hashes"]["lora"], second["adapter_hashes"]["lora"]
        ],
        "source_manifest_fingerprints": [fingerprint(first), fingerprint(second)],
        "source_training_example_counts": [len(first_ids), len(second_ids)],
        "shared_training_example_ids": len(first_ids & second_ids),
        "training_example_union": len(first_ids | second_ids),
        "coefficients": [0.5, 0.5],
        "combination": "PEFT weighted-adapter concatenation; exact effective-delta sum",
    }
    return manifests, provenance


def create_case_mixture(args) -> int:
    manifests, sources = validate_mixture_sources(args.run_dirs)
    first = manifests[0]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    destination.mkdir(parents=False, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "status": "loading_model",
        "kind": "fixed_equal_case_adapter_mixture",
        "sources": sources,
        "model_id": first["model_id"],
        "requested_revision": first["resolved_revision"],
        "resolved_revision": None,
        "device": args.device,
        "precision": first["precision"],
        "environment": environment(),
        "mixture_rank": 2 * int(first["settings"]["rank"]),
        "base_settings": first["settings"],
        "warning": "Nonlearned equal mixture of two development-trained adapters. Rank is the sum of source ranks; this is a comparator, not an independent result or rank-matched learned model.",
    }
    write_json(destination / "manifest.json", manifest)
    try:
        import torch
        from peft import PeftModel
        from .mixture import equal_mixture

        local = LocalModel(first["model_id"], revision=first["resolved_revision"],
                           device=args.device, precision=first["precision"].split(".")[-1])
        manifest["resolved_revision"] = local.resolved_revision
        manifest["device"] = local.device
        manifest["precision"] = local.precision
        local.model = PeftModel.from_pretrained(
            local.model, str(Path(args.run_dirs[0]) / "adapter_lora"),
            adapter_name="source0",
        )
        local.model.load_adapter(str(Path(args.run_dirs[1]) / "adapter_lora"),
                                 adapter_name="source1")
        equal_mixture(local.model, ["source0", "source1"])
        delta_mixture = lora_delta_stats(local.model, torch, "mixture")
        local.model.peft_config["mixture"].revision = local.resolved_revision
        mixture_dir = destination / "adapter_lora"
        local.model.save_pretrained(str(mixture_dir), selected_adapters=["mixture"],
                                    safe_serialization=True)
        _flatten_selected_adapter(mixture_dir, "mixture")

        norm_matched_random(local.model, torch, args.seed, adapter_name="mixture")
        delta_random = lora_delta_stats(local.model, torch, "mixture")
        write_json(destination / "delta_lora.json", delta_mixture)
        write_json(destination / "delta_random.json", delta_random)
        manifest["random_norm_match"] = validate_norm_match(
            destination / "delta_lora.json", destination / "delta_random.json"
        )
        random_dir = destination / "adapter_random"
        local.model.save_pretrained(str(random_dir), selected_adapters=["mixture"],
                                    safe_serialization=True)
        _flatten_selected_adapter(random_dir, "mixture")
        mixture_hashes = checkpoint_hashes(mixture_dir)
        manifest["adapter_hashes"] = {
            "lora": mixture_hashes,
            "mixture": mixture_hashes,
            "random": checkpoint_hashes(random_dir),
        }
        manifest["delta_hashes"] = {
            "mixture": fingerprint(delta_mixture),
            "random": fingerprint(delta_random),
        }
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(destination / "manifest.json", manifest)
    print(f"Case adapter mixture saved: {destination}", flush=True)
    return 0
