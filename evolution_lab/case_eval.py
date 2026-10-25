"""Paired, resumable local evaluation of generated cases with trusted labels."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .casegen import normalized_question, solve, validate_dataset
from .provenance import environment, fingerprint, checkpoint_hashes
from .runner import LocalModel
from .statistics import wilson

ARMS = ("base", "prompt", "adapter", "random")
BASE_SYSTEM = "You are a helpful assistant."
PROMPT_SYSTEM = (
    "You are a helpful assistant. Solve the arithmetic problem carefully "
    "and reply with one integer."
)
INTEGER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?![A-Za-z0-9_])")
INTEGER_ONLY = re.compile(r"\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\s*")


def grade_answer(answer: str, response: str) -> dict:
    """Grade arithmetic value separately from strict integer-only formatting."""
    tokens = INTEGER_TOKEN.findall(response)
    parsed = int(tokens[0].replace(",", "")) if len(tokens) == 1 else None
    return {
        "correct": parsed is not None and parsed == int(answer),
        "parsed_integer": parsed,
        "single_integer_format": bool(INTEGER_ONLY.fullmatch(response)),
    }


def load_cases(dataset: Path, split: str, limit: int | None = None, offset: int = 0) -> tuple[dict, list[dict]]:
    dataset = Path(dataset)
    census = validate_dataset(dataset)
    source_manifest = json.loads((dataset / "manifest.json").read_text())
    rows = []
    for shard in source_manifest["shards"]:
        if shard["split"] != split:
            continue
        rows.extend(json.loads(line) for line in (dataset / shard["file"]).read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("Case offset must be nonnegative and limit positive")
    rows = rows[offset:offset + limit if limit is not None else None]
    if not rows:
        raise ValueError(f"No cases found for split {split!r}")
    if any(normalized_question(row["prompt"]) == "" or row["answer"] != solve(row) for row in rows):
        raise ValueError("Selected dataset rows failed prompt or answer-key validation")
    return {"manifest": source_manifest, "census": census}, rows


def validate_norm_match(adapter_delta: Path, random_delta: Path, tolerance: float = 1e-4) -> dict:
    """Require layerwise effective-delta norms to match before labeling a control random."""
    trained = json.loads(Path(adapter_delta).read_text())
    random_edit = json.loads(Path(random_delta).read_text())
    trained_layers = {row["name"]: row for row in trained["layers"]}
    random_layers = {row["name"]: row for row in random_edit["layers"]}
    if not trained_layers or trained_layers.keys() != random_layers.keys():
        raise ValueError("LoRA and random delta reports must cover the same nonempty modules")
    errors = {}
    for name, left in trained_layers.items():
        right = random_layers[name]
        for field in ("matrix_shape", "rank", "scale"):
            if left[field] != right[field]:
                raise ValueError(f"LoRA/random delta metadata differ for {name}: {field}")
        norm = float(left["delta_frobenius_norm"])
        random_norm = float(right["delta_frobenius_norm"])
        errors[name] = abs(norm - random_norm) / max(abs(norm), 1e-12)
    global_norm = float(trained["delta_global_frobenius_norm"])
    random_global = float(random_edit["delta_global_frobenius_norm"])
    global_error = abs(global_norm - random_global) / max(abs(global_norm), 1e-12)
    maximum_error = max([global_error, *errors.values()])
    if maximum_error > tolerance:
        raise ValueError(f"Random edit is not norm-matched: maximum relative layer/global error {maximum_error:.3g}")
    return {
        "tolerance_relative": tolerance,
        "maximum_relative_error": maximum_error,
        "global_relative_error": global_error,
        "adapted_modules": len(trained_layers),
    }


def validate_delta_report(recorded_path: Path, actual: dict, tolerance: float = 1e-5) -> dict:
    """Bind saved delta metadata to the effective tensors loaded for this run."""
    recorded = json.loads(Path(recorded_path).read_text())
    expected_layers = {row["name"]: row for row in recorded["layers"]}
    actual_layers = {row["name"]: row for row in actual["layers"]}
    if not expected_layers or expected_layers.keys() != actual_layers.keys():
        raise ValueError("Recorded delta report does not match loaded adapter modules")
    errors = {}
    for name, expected in expected_layers.items():
        observed = actual_layers[name]
        for field in ("matrix_shape", "rank", "scale"):
            if expected[field] != observed[field]:
                raise ValueError(f"Recorded and loaded delta metadata differ for {name}: {field}")
        norm = float(expected["delta_frobenius_norm"])
        observed_norm = float(observed["delta_frobenius_norm"])
        errors[name] = abs(norm - observed_norm) / max(abs(norm), 1e-12)
    global_norm = float(recorded["delta_global_frobenius_norm"])
    actual_global = float(actual["delta_global_frobenius_norm"])
    global_error = abs(global_norm - actual_global) / max(abs(global_norm), 1e-12)
    maximum_error = max([global_error, *errors.values()])
    if maximum_error > tolerance:
        raise ValueError(f"Saved delta stats do not match loaded adapter tensors: {maximum_error:.3g} relative error")
    return {"maximum_relative_error": maximum_error, "tolerance_relative": tolerance}


def validate_adapter_source(path: Path, arm: str, model_id: str, revision: str) -> dict:
    path = Path(path)
    manifest_path = path.parent / "manifest.json"
    config_path = path / "adapter_config.json"
    if not manifest_path.is_file() or not config_path.is_file():
        raise ValueError(f"{arm} adapter needs its experiment manifest and adapter_config.json")
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("smoke_only"):
        raise ValueError(f"{arm} adapter must come from a complete, non-smoke experiment")
    if manifest.get("model_id") != model_id or manifest.get("resolved_revision", "").lower() != revision.lower():
        raise ValueError(f"{arm} adapter base model or immutable revision does not match the requested model")
    adapter_base = config.get("base_model_name_or_path")
    if adapter_base and adapter_base != model_id:
        raise ValueError(f"{arm} adapter targets {adapter_base}, not requested base model {model_id}")
    manifest_arm = "lora" if arm == "adapter" else arm
    recorded = manifest.get("adapter_hashes", {}).get(manifest_arm)
    actual = checkpoint_hashes(path)
    if recorded is None or recorded != actual:
        raise ValueError(f"{arm} adapter files do not match hashes in its source experiment manifest")
    return {
        "source_manifest_sha256": fingerprint(manifest),
        "source_run_id": manifest.get("run_id"),
        "behavior": manifest.get("behavior"),
        "seed": manifest.get("seed"),
        "training_sha256": manifest.get("training_sha256"),
        "settings": manifest.get("settings"),
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@contextmanager
def _campaign_lock(path: Path):
    with Path(path).open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process already owns this case-evaluation campaign") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _summary(rows: list[dict], arms: list[str], seed: int = 42) -> dict:
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in arms}
    summaries = {}
    for arm, records in by_arm.items():
        n = len(records)
        correct = sum(row["metrics"]["correct"] for row in records)
        formatted = sum(row["metrics"]["single_integer_format"] for row in records)
        families = {row["family"] for row in records}
        summaries[arm] = {
            "model_generations": n,
            "unique_questions": len({row["case_id"] for row in records}),
            "unique_normalized_prompts": len({row.get("normalized_prompt_sha256", _text_sha256(normalized_question(row["prompt"]))) for row in records}),
            "correct": correct,
            "accuracy": correct / n if n else None,
            "wilson95_descriptive": wilson(correct, n) if n else None,
            "single_integer_format": formatted,
            "family_count": len(families),
        }

    paired = {}
    if "base" in by_arm:
        baseline = {row["case_id"]: row for row in by_arm["base"]}
        for arm in arms:
            if arm == "base":
                continue
            comparison = {row["case_id"]: row for row in by_arm[arm]}
            common = sorted(set(baseline) & set(comparison))
            groups: dict[str, list[int]] = {}
            improved = regressed = 0
            for case_id in common:
                before = int(baseline[case_id]["metrics"]["correct"])
                after = int(comparison[case_id]["metrics"]["correct"])
                improved += before == 0 and after == 1
                regressed += before == 1 and after == 0
                family = baseline[case_id]["family"]
                groups.setdefault(family, []).append(after - before)
            ci = None
            if len(groups) >= 2:
                rng = random.Random(seed)
                clusters = list(groups.values())
                draws = []
                for _ in range(2000):
                    sample = [value for group in rng.choices(clusters, k=len(clusters)) for value in group]
                    draws.append(sum(sample) / len(sample))
                draws.sort()
                ci = [draws[49], draws[1950]]
            paired[arm] = {
                "paired_questions": len(common),
                "improved": improved,
                "regressed": regressed,
                "pass_rate_delta": (improved - regressed) / len(common) if common else None,
                "paired_family_bootstrap95": ci,
                "family_count": len(groups),
                "note": "Bootstrap resamples arithmetic fact families; few families yield unstable or unavailable intervals.",
            }
    return {
        "arms": summaries,
        "paired_vs_base": paired,
        "uncertainty_note": "Wilson intervals are descriptive and ignore within-family dependence; use the paired family bootstrap for arm contrasts.",
    }


def evaluate_cases(args) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with _campaign_lock(output / ".case-eval.lock"):
        return _evaluate_cases_locked(args)


def _evaluate_cases_locked(args) -> int:
    if args.split == "confirmation" and not args.allow_confirmation:
        raise ValueError("Confirmation cases require the explicit --allow-confirmation flag")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.offset < 0:
        raise ValueError("--offset must be nonnegative")
    if args.checkpoint_every < 1 or args.max_new_tokens < 1:
        raise ValueError("--checkpoint-every and --max-new-tokens must be positive")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        raise ValueError("Use an immutable 40-character model revision SHA")
    arms = [arm for arm in ARMS if arm in args.arms]
    if not arms or len(arms) != len(args.arms):
        raise ValueError("Choose one or more distinct evaluation arms")
    if len(set(args.arms)) != len(args.arms):
        raise ValueError("Duplicate evaluation arms are not allowed")
    if "base" not in arms:
        raise ValueError("Every campaign needs the unedited base arm for paired comparisons")
    if "adapter" in arms and not {"base", "prompt", "random"}.issubset(arms):
        raise ValueError("LoRA campaigns require base, prompt, adapter, and norm-matched random arms")
    if "adapter" in arms and args.adapter is None:
        raise ValueError("The adapter arm requires --adapter")
    if "random" in arms and args.random_adapter is None:
        raise ValueError("The random arm requires --random-adapter")
    if args.adapter and "adapter" not in arms:
        raise ValueError("--adapter requires the adapter arm")
    if args.random_adapter and "random" not in arms:
        raise ValueError("--random-adapter requires the random arm")
    for path, arm in ((args.adapter_delta, "adapter"), (args.random_delta, "random")):
        if path and arm not in arms:
            raise ValueError(f"--{arm}-delta requires the {arm} arm")
    norm_match = None
    if "adapter" in arms:
        if not args.adapter_delta or not args.random_delta:
            raise ValueError("LoRA campaigns require --adapter-delta and --random-delta evidence")
        norm_match = validate_norm_match(args.adapter_delta, args.random_delta)

    source, cases = load_cases(args.dataset, args.split, args.limit, args.offset)
    adapter_hashes = {
        arm: checkpoint_hashes(path)
        for arm, path in (("adapter", args.adapter), ("random", args.random_adapter))
        if path is not None
    }
    adapter_sources = {}
    for arm, path in (("adapter", args.adapter), ("random", args.random_adapter)):
        if path is not None and not adapter_hashes[arm]:
            raise ValueError(f"No checkpoint files found for {arm} adapter: {path}")
        if path is not None:
            adapter_sources[arm] = validate_adapter_source(path, arm, args.model, args.revision)
    if "adapter" in adapter_sources and "random" in adapter_sources:
        for key in ("source_manifest_sha256", "behavior", "seed", "training_sha256", "settings"):
            if adapter_sources["adapter"].get(key) != adapter_sources["random"].get(key):
                raise ValueError(f"LoRA and random adapters must share their source experiment ({key} differs)")
    case_fingerprint = fingerprint([
        {"case_id": row["case_id"], "prompt": row["prompt"], "answer": row["answer"],
         "family": row["family"], "normalized_prompt_sha256": _text_sha256(normalized_question(row["prompt"]))}
        for row in cases
    ])
    spec = {
        "schema_version": 1,
        "dataset_manifest_sha256": fingerprint(source["manifest"]),
        "dataset_seed": source["manifest"]["seed"],
        "split": args.split,
        "case_fingerprint": case_fingerprint,
        "unique_questions": len(cases),
        "split_offset": args.offset,
        "model_id": args.model,
        "requested_revision": args.revision.lower(),
        "requested_device": args.device,
        "requested_precision": args.precision,
        "arms": arms,
        "base_system_prompt": BASE_SYSTEM,
        "prompt_system_prompt": args.prompt_system,
        "adapter_hashes": adapter_hashes,
        "adapter_sources": adapter_sources,
        "adapter_delta_hashes": {
            arm: _file_sha256(path) if path else None
            for arm, path in (("adapter", args.adapter_delta), ("random", args.random_delta))
        },
        "random_norm_match": norm_match,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "greedy",
    }
    spec_hash = fingerprint(spec)
    output = Path(args.output_dir)
    run_path = output / "run.json"
    observations_path = output / "observations.jsonl"
    if run_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output already exists; use --resume to continue: {output}")
        run = json.loads(run_path.read_text())
        if run.get("spec_sha256") != spec_hash or run.get("spec") != spec:
            raise ValueError("Resume specification differs from the existing campaign")
    else:
        if args.resume:
            raise FileNotFoundError(f"Cannot resume missing campaign: {output}")
        unexpected = [path.name for path in output.iterdir() if path.name != ".case-eval.lock"]
        if unexpected:
            raise FileExistsError(f"Output directory contains unrelated files: {output}")
        run = {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
            "spec": spec,
            "spec_sha256": spec_hash,
            "status": "loading_model",
            "environment": environment(),
            "dataset_census": source["census"],
            "records_written": 0,
        }
        _atomic_json(run_path, run)

    try:
        local = LocalModel(args.model, revision=args.revision, device=args.device,
                           precision=args.precision)
    except BaseException as error:
        run.update(status="model_load_failed", error=f"{type(error).__name__}: {error}")
        _atomic_json(run_path, run)
        raise
    actual = {"resolved_revision": local.resolved_revision, "device": local.device,
              "precision": local.precision}
    if "actual_model" in run and run["actual_model"] != actual:
        run.update(status="resume_mismatch", actual_model_observed=actual)
        _atomic_json(run_path, run)
        raise ValueError("Resume model revision, device, or precision differs from saved campaign")
    run.update(status="running", actual_model=actual)
    _atomic_json(run_path, run)

    verified_deltas = {}
    if "adapter" in arms:
        try:
            from peft import PeftModel
            from .train_adapter import lora_delta_stats
            local.model = PeftModel.from_pretrained(
                local.model, str(args.adapter), adapter_name="adapter", is_trainable=False,
            )
            local.model.load_adapter(str(args.random_adapter), adapter_name="random", is_trainable=False)
            local.model.set_adapter("adapter")
            local.model.eval()
            for arm, adapter_name, delta_path in (
                ("adapter", "adapter", args.adapter_delta),
                ("random", "random", args.random_delta),
            ):
                measured = lora_delta_stats(local.model, local.torch, adapter_name=adapter_name)
                check = validate_delta_report(delta_path, measured)
                verified_deltas[arm] = {"stats": measured, "sidecar_check": check}
        except BaseException as error:
            run.update(status="adapter_validation_failed", error=f"{type(error).__name__}: {error}")
            _atomic_json(run_path, run)
            raise
        run["verified_delta_stats"] = verified_deltas
        _atomic_json(run_path, run)

    existing: dict[tuple[str, str], dict] = {}
    if observations_path.exists():
        for line in observations_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["arm"], row["case_id"])
            if key in existing or row["arm"] not in arms:
                raise ValueError("Duplicate or unexpected cached response in campaign")
            existing[key] = row
    case_map = {row["case_id"]: row for row in cases}
    for (arm, case_id), row in existing.items():
        case = case_map.get(case_id)
        if case is None or row.get("answer_key") != case["answer"] or row.get("prompt") != case["prompt"]:
            raise ValueError("Cached response does not match the selected dataset")
        if row.get("metrics") != grade_answer(case["answer"], row.get("response", "")):
            raise ValueError("Cached objective grade is inconsistent with the raw response")

    records_written = len(existing)
    completed_by_arm = {arm: sum(key[0] == arm for key in existing) for arm in arms}
    started = time.monotonic()
    try:
        with observations_path.open("a", encoding="utf-8") as stream:
            for arm in arms:
                if arm in {"adapter", "random"}:
                    local.model.set_adapter(arm)
                    local.model.eval()
                for case in cases:
                    key = (arm, case["case_id"])
                    if key in existing:
                        continue
                    system_prompt = args.prompt_system if arm == "prompt" else BASE_SYSTEM
                    if "adapter" in arms and arm in {"base", "prompt"}:
                        with local.model.disable_adapter():
                            response, tokens, latency = local.generate(
                                system_prompt, case["prompt"], args.max_new_tokens
                            )
                    else:
                        response, tokens, latency = local.generate(
                            system_prompt, case["prompt"], args.max_new_tokens
                        )
                    row = {
                        "case_id": case["case_id"], "family": case["family"],
                        "split": case["split"], "arm": arm,
                        "prompt": case["prompt"], "answer_key": case["answer"],
                        "normalized_prompt_sha256": _text_sha256(normalized_question(case["prompt"])),
                        "response": response, "metrics": grade_answer(case["answer"], response),
                        "generated_tokens": tokens, "latency_seconds": latency,
                        "resolved_revision": local.resolved_revision,
                    }
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    existing[key] = row
                    records_written += 1
                    completed_by_arm[arm] += 1
                    if records_written % args.checkpoint_every == 0:
                        os.fsync(stream.fileno())
                        run["records_written"] = records_written
                        _atomic_json(run_path, run)
                    print(f"{arm} {completed_by_arm[arm]}/{len(cases)}: {row['metrics']['correct']} {case['case_id']}", flush=True)
            os.fsync(stream.fileno())
        report = _summary(list(existing.values()), arms)
        _atomic_json(output / "report.json", report)
        run.update(status="complete", records_written=records_written,
                   elapsed_seconds=round(time.monotonic() - started, 3),
                   model_generations=records_written,
                   completed_arms={arm: sum(key[0] == arm for key in existing) for arm in arms})
        _atomic_json(run_path, run)
    except BaseException as error:
        run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                   records_written=records_written,
                   error=f"{type(error).__name__}: {error}")
        _atomic_json(run_path, run)
        raise
    print(f"Case campaign saved: {output}", flush=True)
    return 0
