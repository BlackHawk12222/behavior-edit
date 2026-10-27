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
ANSWER_GRADER_VERSION = "final-integer-token-v2+unique-completion-choice-v1"
INTEGER_GRADER_VERSION = "final-integer-token-v2"
COMPLETION_GRADER_VERSION = "unique-completion-choice-v1"
BASE_SYSTEM = "You are a helpful assistant."
ARITHMETIC_PROMPT_SYSTEM = (
    "You are a helpful assistant. Solve the arithmetic problem carefully "
    "and reply with one integer."
)
COMPLETION_PROMPT_SYSTEM = (
    "You are a helpful assistant. Select the option that correctly completes "
    "the phrase or sentence and reply with only the missing word."
)
INTEGER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?![A-Za-z0-9_])")
INTEGER_ONLY = re.compile(r"\s*[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)\s*")
WORD_TOKEN = re.compile(r"[a-z]+(?:['’-][a-z]+)*", re.IGNORECASE)
WORD_ONLY = re.compile(r"\s*[a-z]+(?:['’-][a-z]+)*\s*", re.IGNORECASE)


def grade_answer(answer: str, response: str) -> dict:
    """Use the final integer mentioned as the answer; score format separately.

    Generated arithmetic questions put their answer at the end of the response
    in the common explanatory form. This lets a correct explanation count as
    numerically correct without counting as integer-only formatting.
    """
    tokens = INTEGER_TOKEN.findall(response)
    parsed = int(tokens[-1].replace(",", "")) if tokens else None
    return {
        "correct": parsed is not None and parsed == int(answer),
        "parsed_integer": parsed,
        "integer_token_count": len(tokens),
        "single_integer_format": bool(INTEGER_ONLY.fullmatch(response)),
    }


def grade_completion(case: dict, response: str) -> dict:
    """Find a unique offered answer even when the model returns a full sentence."""
    tokens = WORD_TOKEN.findall(response)
    accepted = {str(value).casefold() for value in case.get("accepted_answers", [case["answer"]])}
    options = {str(value).casefold() for value in case.get("options", [])} | accepted
    mentioned = [token.casefold() for token in tokens if token.casefold() in options]
    distinct = set(mentioned)
    parsed = mentioned[-1] if len(distinct) == 1 else None
    return {
        "correct": parsed in accepted if parsed is not None else False,
        "parsed_answer": parsed,
        "word_token_count": len(tokens),
        "distinct_option_mentions": len(distinct),
        "single_word_format": bool(WORD_ONLY.fullmatch(response)),
    }


def grade_case(case: dict, response: str) -> dict:
    if case.get("task_type") in {"phrase_completion", "sentence_completion"}:
        return grade_completion(case, response)
    return grade_answer(case["answer"], response)


def default_prompt_system(dataset_generator: str) -> str:
    """Choose a task-matched system prompt for the explicit-prompt control."""
    if dataset_generator.startswith("phrase-sentence-completion-"):
        return COMPLETION_PROMPT_SYSTEM
    return ARITHMETIC_PROMPT_SYSTEM


def paired_arm_contrast(rows: list[dict], reference_arm: str,
                        candidate_arm: str, seed: int = 42) -> dict:
    """Compare two arms on matched cases with family-stratified uncertainty."""
    by_arm: dict[str, dict[str, dict]] = {reference_arm: {}, candidate_arm: {}}
    for row in rows:
        arm = row.get("arm")
        if arm not in by_arm:
            continue
        case_id = row["case_id"]
        if case_id in by_arm[arm]:
            raise ValueError(f"Duplicate case in paired contrast: {arm}/{case_id}")
        by_arm[arm][case_id] = row
    common = sorted(by_arm[reference_arm].keys() & by_arm[candidate_arm].keys())
    if not common:
        raise ValueError("Paired contrast requires at least one shared question")
    groups: dict[str, dict[str, list[int]]] = {}
    counts: dict[str, int] = {}
    improved = regressed = 0
    for case_id in common:
        before = by_arm[reference_arm][case_id]
        after = by_arm[candidate_arm][case_id]
        if before.get("family") != after.get("family") or before.get("task_type") != after.get("task_type"):
            raise ValueError(f"Paired case metadata differs across arms: {case_id}")
        delta = int(bool(after["metrics"]["correct"])) - int(bool(before["metrics"]["correct"]))
        improved += delta > 0
        regressed += delta < 0
        task_type = before.get("task_type", "arithmetic")
        family = before["family"]
        groups.setdefault(task_type, {}).setdefault(family, []).append(delta)
        counts[task_type] = counts.get(task_type, 0) + 1

    total_families = sum(len(families) for families in groups.values())
    interval = None
    by_type = {}
    if total_families >= 2:
        rng = random.Random(seed)
        combined_draws = []
        task_draws = {task: [] for task in groups}
        for _ in range(2000):
            weighted = 0.0
            for task, families in groups.items():
                clusters = list(families.values())
                sampled = [value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster]
                mean = sum(sampled) / len(sampled)
                task_draws[task].append(mean)
                weighted += (counts[task] / len(common)) * mean
            combined_draws.append(weighted)
        combined_draws.sort()
        interval = [combined_draws[49], combined_draws[1950]]
        for task, draws in task_draws.items():
            if len(groups[task]) >= 2:
                draws.sort()
                by_type[task] = [draws[49], draws[1950]]
            else:
                by_type[task] = None
    return {
        "reference_arm": reference_arm,
        "candidate_arm": candidate_arm,
        "paired_questions": len(common),
        "improved": improved,
        "regressed": regressed,
        "correctness_rate_delta": (improved - regressed) / len(common),
        "family_bootstrap95": interval,
        "family_bootstrap95_by_task_type": by_type,
        "family_count": total_families,
        "note": "Paired family bootstrap resamples families within task type; small family counts produce unstable intervals.",
    }


def load_cases(dataset: Path, split: str, limit: int | None = None, offset: int = 0,
               family_limit: int | None = None, family_seed: int | None = None) -> tuple[dict, list[dict]]:
    dataset = Path(dataset)
    source_manifest = json.loads((dataset / "manifest.json").read_text())
    generator = source_manifest.get("generator")
    if generator == "arithmetic-v3":
        census = validate_dataset(dataset)
        answer_for = solve
    elif generator == "phrase-sentence-completion-v1":
        from .completiongen import solve as solve_completion, validate_dataset as validate_completion_dataset
        census = validate_completion_dataset(dataset)
        answer_for = solve_completion
    elif generator == "phrase-sentence-completion-v2-new-vocabulary":
        from .completiongen_v2 import solve as solve_completion, validate_dataset as validate_completion_dataset
        census = validate_completion_dataset(dataset)
        answer_for = solve_completion
    elif generator == "phrase-sentence-completion-v3-attractor":
        from .completiongen_v3 import solve as solve_completion, validate_dataset as validate_completion_dataset
        census = validate_completion_dataset(dataset)
        answer_for = solve_completion
    elif generator == "phrase-sentence-completion-v4-fresh-lexicon":
        from .completiongen_v4 import solve as solve_completion, validate_dataset as validate_completion_dataset
        census = validate_completion_dataset(dataset)
        answer_for = solve_completion
    else:
        raise ValueError(f"Unsupported case dataset generator: {generator}")
    rows = []
    for shard in source_manifest["shards"]:
        if shard["split"] != split:
            continue
        rows.extend(json.loads(line) for line in (dataset / shard["file"]).read_text().splitlines() if line)
    rows.sort(key=lambda row: row["index"])
    selection = None
    if family_limit is not None:
        if family_limit < 1:
            raise ValueError("Family sample size must be positive")
        if family_seed is None:
            raise ValueError("Family sampling requires --family-seed")
        if limit is not None or offset != 0:
            raise ValueError("Family sampling cannot be combined with --limit or --offset")
        by_family: dict[str, list[dict]] = {}
        for row in rows:
            by_family.setdefault(row["family"], []).append(row)
        if family_limit > len(by_family):
            raise ValueError(
                f"Requested {family_limit} families, but split {split!r} has only {len(by_family)}"
            )
        selected = sorted(
            by_family,
            key=lambda family: hashlib.sha256(
                f"case-eval-family-sample-v1:{family_seed}:{family}".encode()
            ).hexdigest(),
        )[:family_limit]
        selected_set = set(selected)
        rows = [row for row in rows if row["family"] in selected_set]
        selection = {
            "method": "sha256-rank-whole-families-v1",
            "family_limit": family_limit,
            "family_seed": family_seed,
            "selected_families": selected,
        }
    elif family_seed is not None:
        raise ValueError("--family-seed requires --family-limit")
    if offset < 0 or (limit is not None and limit < 1):
        raise ValueError("Case offset must be nonnegative and limit positive")
    rows = rows[offset:offset + limit if limit is not None else None]
    if not rows:
        raise ValueError(f"No cases found for split {split!r}")
    if any(normalized_question(row["prompt"]) == "" or row["answer"] != answer_for(row) for row in rows):
        raise ValueError("Selected dataset rows failed prompt or answer-key validation")
    return {"manifest": source_manifest, "census": census, "selection": selection}, rows


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
    def same_model_reference(value: str | None) -> bool:
        if value == model_id:
            return True
        # Offline runs record the exact Hugging Face cache snapshot as the
        # model ID. Accept that alias only when both repository and immutable
        # revision are encoded in the canonical cache path.
        if not value:
            return False
        cached = Path(value)
        if cached.name.lower() != revision.lower():
            return False
        repo_component = next(
            (parent.name for parent in cached.parents if parent.name.startswith("models--")),
            None,
        )
        if repo_component is None:
            return False
        cached_repo = repo_component.removeprefix("models--").replace("--", "/")
        return cached_repo == model_id

    if not same_model_reference(manifest.get("model_id")) or manifest.get("resolved_revision", "").lower() != revision.lower():
        raise ValueError(f"{arm} adapter base model or immutable revision does not match the requested model")
    adapter_base = config.get("base_model_name_or_path")
    if adapter_base and not same_model_reference(adapter_base):
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


def _load_baseline_cache(source_dir: Path, current_spec: dict,
                         cases: list[dict]) -> tuple[dict[tuple[str, str], dict], dict]:
    """Load exact base/prompt records from a checksum-verified compatible run."""
    source_dir = Path(source_dir)
    run_path = source_dir / "run.json"
    observations_path = source_dir / "observations.jsonl"
    report_path = source_dir / "report.json"
    if not run_path.is_file() or not observations_path.is_file() or not report_path.is_file():
        raise ValueError("Baseline cache source must be a completed case-evaluation campaign")
    source_run = json.loads(run_path.read_text())
    if source_run.get("status") != "complete":
        raise ValueError("Baseline cache source campaign is not complete")
    actual_model = source_run.get("actual_model")
    if not isinstance(actual_model, dict):
        raise ValueError("Baseline cache source is missing its resolved model settings")
    if _file_sha256(observations_path) != source_run.get("observations_sha256"):
        raise ValueError("Baseline cache source ledger checksum mismatch")
    if _file_sha256(report_path) != source_run.get("report_sha256"):
        raise ValueError("Baseline cache source report checksum mismatch")
    source_spec = source_run.get("spec", {})
    if source_run.get("spec_sha256") != fingerprint(source_spec):
        raise ValueError("Baseline cache source specification checksum mismatch")
    if not {"base", "prompt"}.issubset(source_spec.get("arms", [])):
        raise ValueError("Baseline cache source must contain both base and prompt arms")
    shared_fields = (
        "answer_grader_version", "dataset_generator", "dataset_manifest_sha256",
        "dataset_seed", "split", "case_fingerprint", "unique_questions",
        "split_offset", "family_sampling", "model_id", "requested_revision",
        "requested_device", "requested_precision", "base_system_prompt",
        "prompt_system_prompt", "max_new_tokens", "decoding",
    )
    mismatches = [field for field in shared_fields
                  if source_spec.get(field) != current_spec.get(field)]
    if mismatches:
        raise ValueError("Baseline cache source differs in required inference fields: "
                         + ", ".join(mismatches))

    case_map = {row["case_id"]: row for row in cases}
    cached: dict[tuple[str, str], dict] = {}
    for line in observations_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm = row.get("arm")
        if arm not in {"base", "prompt"}:
            continue
        case_id = row.get("case_id")
        key = (arm, case_id)
        case = case_map.get(case_id)
        if key in cached or case is None:
            raise ValueError("Baseline cache source has duplicate or unexpected case records")
        if row.get("prompt") != case["prompt"] or row.get("answer_key") != case["answer"]:
            raise ValueError("Baseline cache source response does not match the selected question")
        if row.get("normalized_prompt_sha256") != _text_sha256(normalized_question(case["prompt"])):
            raise ValueError("Baseline cache source prompt hash does not match the selected question")
        if row.get("metrics") != grade_case(case, row.get("response", "")):
            raise ValueError("Baseline cache source trusted-key grade mismatch")
        if row.get("resolved_revision") != current_spec.get("requested_revision"):
            raise ValueError("Baseline cache source resolved model revision mismatch")
        cached[key] = row
    expected = {(arm, case_id) for arm in ("base", "prompt") for case_id in case_map}
    if cached.keys() != expected:
        raise ValueError("Baseline cache source is missing base/prompt responses for selected questions")
    provenance = {
        "run_id": source_run.get("run_id"),
        "spec_sha256": source_run["spec_sha256"],
        "observations_sha256": source_run["observations_sha256"],
        "report_sha256": source_run["report_sha256"],
        "actual_model": actual_model,
    }
    return cached, provenance


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
        cache_hits = sum(bool(row.get("cache_hit")) for row in records)
        correct = sum(row["metrics"]["correct"] for row in records)
        format_key = (
            "single_word_format"
            if records and "single_word_format" in records[0]["metrics"]
            else "single_integer_format"
        )
        formatted = sum(row["metrics"].get(format_key, False) for row in records)
        families = {row["family"] for row in records}
        summary = {
            "model_generations": n - cache_hits,
            "cache_hits": cache_hits,
            "response_records": n,
            "unique_questions": len({row["case_id"] for row in records}),
            "unique_normalized_prompts": len({row.get("normalized_prompt_sha256", _text_sha256(normalized_question(row["prompt"]))) for row in records}),
            "correct": correct,
            "accuracy": correct / n if n else None,
            "wilson95_descriptive": wilson(correct, n) if n else None,
            "strict_format_passes": formatted,
            "strict_format_metric": format_key,
            "family_count": len(families),
        }
        if format_key == "single_word_format":
            summary["single_word_format"] = formatted
        else:
            summary["single_integer_format"] = formatted
        task_types = sorted({row.get("task_type", "arithmetic") for row in records})
        if len(task_types) > 1 or task_types != ["arithmetic"]:
            summary["by_task_type"] = {}
            for task_type in task_types:
                task_rows = [row for row in records if row.get("task_type", "arithmetic") == task_type]
                summary["by_task_type"][task_type] = {
                    "model_generations": len(task_rows) - sum(bool(row.get("cache_hit")) for row in task_rows),
                    "cache_hits": sum(bool(row.get("cache_hit")) for row in task_rows),
                    "response_records": len(task_rows),
                    "unique_questions": len({row["case_id"] for row in task_rows}),
                    "correct": sum(row["metrics"]["correct"] for row in task_rows),
                    "accuracy": sum(row["metrics"]["correct"] for row in task_rows) / len(task_rows),
                    "strict_format_passes": sum(row["metrics"].get(format_key, False) for row in task_rows),
                    "family_count": len({row["family"] for row in task_rows}),
                }
        summaries[arm] = summary

    paired = {}
    if "base" in by_arm:
        baseline = {row["case_id"]: row for row in by_arm["base"]}
        for arm in arms:
            if arm == "base":
                continue
            comparison = {row["case_id"]: row for row in by_arm[arm]}
            common = sorted(set(baseline) & set(comparison))
            groups_by_type: dict[str, dict[str, list[int]]] = {}
            counts_by_type: dict[str, int] = {}
            improved = regressed = 0
            for case_id in common:
                before = int(baseline[case_id]["metrics"]["correct"])
                after = int(comparison[case_id]["metrics"]["correct"])
                improved += before == 0 and after == 1
                regressed += before == 1 and after == 0
                family = baseline[case_id]["family"]
                task_type = baseline[case_id].get("task_type", "arithmetic")
                groups_by_type.setdefault(task_type, {}).setdefault(family, []).append(after - before)
                counts_by_type[task_type] = counts_by_type.get(task_type, 0) + 1
            ci = None
            ci_by_type = {}
            total_families = sum(len(groups) for groups in groups_by_type.values())
            if total_families >= 2:
                rng = random.Random(seed)
                draws = []
                type_draws: dict[str, list[float]] = {task_type: [] for task_type in groups_by_type}
                for _ in range(2000):
                    estimate = 0.0
                    for task_type, groups in groups_by_type.items():
                        clusters = list(groups.values())
                        sample = [value for group in rng.choices(clusters, k=len(clusters)) for value in group]
                        task_mean = sum(sample) / len(sample)
                        estimate += (counts_by_type[task_type] / len(common)) * task_mean
                        type_draws[task_type].append(task_mean)
                    draws.append(estimate)
                draws.sort()
                ci = [draws[49], draws[1950]]
                for task_type, values in type_draws.items():
                    if len(groups_by_type[task_type]) >= 2:
                        values.sort()
                        ci_by_type[task_type] = [values[49], values[1950]]
                    else:
                        ci_by_type[task_type] = None
            paired[arm] = {
                "paired_questions": len(common),
                "improved": improved,
                "regressed": regressed,
                "pass_rate_delta": (improved - regressed) / len(common) if common else None,
                "paired_family_bootstrap95": ci,
                "paired_family_bootstrap95_by_task_type": ci_by_type,
                "family_count": total_families,
                "note": "Bootstrap resamples families within each task type and weights the combined estimate by question counts; few families yield unstable intervals.",
            }
    return {
        "answer_grader_version": ANSWER_GRADER_VERSION,
        "grading_policies": {
            "arithmetic": INTEGER_GRADER_VERSION,
            "phrase_and_sentence_completion": COMPLETION_GRADER_VERSION,
        },
        "arms": summaries,
        "paired_vs_base": paired,
        "pairwise_comparisons": {
            f"{candidate}_vs_{reference}": paired_arm_contrast(rows, reference, candidate, seed)
            for index, reference in enumerate(arms)
            for candidate in arms[index + 1:]
        },
        "uncertainty_note": "Wilson intervals are descriptive and ignore within-family dependence; use the paired family bootstrap for arm contrasts.",
    }


def evaluate_cases(args) -> int:
    # Reject an unapproved final-test request before creating the destination or
    # its lock file. This makes the opt-in boundary side-effect free.
    if args.split == "confirmation" and not args.allow_confirmation:
        raise ValueError("Confirmation cases require the explicit --allow-confirmation flag")
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
    family_limit = getattr(args, "family_limit", None)
    family_seed = getattr(args, "family_seed", None)
    if family_limit is not None and family_limit < 1:
        raise ValueError("--family-limit must be positive")
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

    source, cases = load_cases(
        args.dataset, args.split, args.limit, args.offset, family_limit, family_seed
    )
    dataset_generator = source["manifest"]["generator"]
    prompt_system = args.prompt_system or default_prompt_system(dataset_generator)
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
        "answer_grader_version": ANSWER_GRADER_VERSION,
        "dataset_generator": dataset_generator,
        "dataset_manifest_sha256": fingerprint(source["manifest"]),
        "dataset_seed": source["manifest"]["seed"],
        "split": args.split,
        "case_fingerprint": case_fingerprint,
        "unique_questions": len(cases),
        "split_offset": args.offset,
        "family_sampling": source.get("selection"),
        "model_id": args.model,
        "requested_revision": args.revision.lower(),
        "requested_device": args.device,
        "requested_precision": args.precision,
        "arms": arms,
        "base_system_prompt": BASE_SYSTEM,
        "prompt_system_prompt": prompt_system,
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
    baseline_cached = {}
    baseline_cache_source = None
    baseline_cache_from = getattr(args, "baseline_cache_from", None)
    if baseline_cache_from is not None:
        if not {"base", "prompt"}.issubset(arms):
            raise ValueError("Baseline cache reuse requires both base and prompt arms")
        baseline_cached, baseline_cache_source = _load_baseline_cache(
            baseline_cache_from, spec, cases
        )
        spec["baseline_cache_source"] = baseline_cache_source
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
            "model_generations": 0,
            "cache_hits": 0,
        }
        _atomic_json(run_path, run)

    existing: dict[tuple[str, str], dict] = {}
    if observations_path.exists():
        if run.get("status") == "complete":
            expected_hash = run.get("observations_sha256")
            if not expected_hash or _file_sha256(observations_path) != expected_hash:
                raise ValueError("Completed response ledger does not match its saved checksum")
            report_path = output / "report.json"
            if not report_path.is_file() or _file_sha256(report_path) != run.get("report_sha256"):
                raise ValueError("Completed report does not match its saved checksum")
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
        if row.get("normalized_prompt_sha256") != _text_sha256(normalized_question(case["prompt"])):
            raise ValueError("Cached normalized prompt hash does not match the selected dataset")
        if row.get("metrics") != grade_case(case, row.get("response", "")):
            raise ValueError("Cached objective grade is inconsistent with the raw response")
        if row.get("cache_hit"):
            if (baseline_cache_source is None
                    or row.get("cache_source_spec_sha256") != baseline_cache_source["spec_sha256"]
                    or row.get("cache_source_observations_sha256") != baseline_cache_source["observations_sha256"]):
                raise ValueError("Reused response does not match the registered baseline cache source")
    if run.get("status") == "complete":
        expected_keys = {(arm, row["case_id"]) for arm in arms for row in cases}
        if existing.keys() != expected_keys:
            raise ValueError("Completed response ledger is missing or contains unexpected arm/question pairs")
        print(f"Case campaign already complete: {output}", flush=True)
        return 0

    if baseline_cached:
        imported = []
        for key in sorted(baseline_cached):
            if key in existing:
                continue
            row = dict(baseline_cached[key])
            row["cache_hit"] = True
            row["cache_source_spec_sha256"] = baseline_cache_source["spec_sha256"]
            row["cache_source_observations_sha256"] = baseline_cache_source["observations_sha256"]
            imported.append(row)
        if imported:
            with observations_path.open("a", encoding="utf-8") as stream:
                for row in imported:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    existing[(row["arm"], row["case_id"])] = row
                stream.flush()
                os.fsync(stream.fileno())
            records_written = len(existing)
            run.update(
                records_written=records_written,
                model_generations=sum(not bool(row.get("cache_hit")) for row in existing.values()),
                cache_hits=sum(bool(row.get("cache_hit")) for row in existing.values()),
            )
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
    if baseline_cache_source is not None and baseline_cache_source["actual_model"] != actual:
        run.update(status="cache_source_mismatch", actual_model_observed=actual)
        _atomic_json(run_path, run)
        raise ValueError("Baseline cache source resolved device, precision, or model revision differs")
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
                    system_prompt = prompt_system if arm == "prompt" else BASE_SYSTEM
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
                        "task_type": case.get("task_type", "arithmetic"),
                        "prompt": case["prompt"], "answer_key": case["answer"],
                        "normalized_prompt_sha256": _text_sha256(normalized_question(case["prompt"])),
                        "response": response, "metrics": grade_case(case, response),
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
                        run.update(
                            records_written=records_written,
                            model_generations=sum(not bool(row.get("cache_hit")) for row in existing.values()),
                            cache_hits=sum(bool(row.get("cache_hit")) for row in existing.values()),
                        )
                        _atomic_json(run_path, run)
                    # Progress output must not reveal held-out correctness before
                    # the campaign is complete; raw grades stay in the ledger.
                    print(f"{arm}: recorded {completed_by_arm[arm]}/{len(cases)}", flush=True)
            os.fsync(stream.fileno())
        report = _summary(list(existing.values()), arms)
        report_path = output / "report.json"
        _atomic_json(report_path, report)
        run.update(status="complete", records_written=records_written,
                   elapsed_seconds=round(time.monotonic() - started, 3),
                   observations_sha256=_file_sha256(observations_path),
                   report_sha256=_file_sha256(report_path),
                   model_generations=sum(not bool(row.get("cache_hit")) for row in existing.values()),
                   cache_hits=sum(bool(row.get("cache_hit")) for row in existing.values()),
                   response_records=records_written,
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
