"""Audit F026 training/evaluation ledgers and compare order seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases
from evolution_lab.provenance import fingerprint

SEEDS = (42, 43)
SCHEDULE_SEEDS = (901, 902, 903)
ARMS = ("base", "prompt", "adapter", "random")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def only_run_dir(path: Path) -> Path:
    runs = sorted(child for child in path.iterdir()
                  if child.is_dir() and (child / "manifest.json").is_file())
    if len(runs) != 1:
        raise ValueError(f"Expected one completed training run under {path}, found {len(runs)}")
    return runs[0]


def load_training(evidence: Path, seed: int, order_seed: int,
                  cohort_hash: str, cohort_manifest_hash: str) -> dict:
    condition = evidence / "training" / f"seed{seed}_schedule{order_seed}"
    run_dir = only_run_dir(condition)
    manifest_path = run_dir / "manifest.json"
    trace_path = run_dir / "training.json"
    manifest = json.loads(manifest_path.read_text())
    trace = json.loads(trace_path.read_text())
    if manifest.get("status") != "complete" or len(trace) != 64:
        raise ValueError(f"Incomplete training run: seed={seed}, order={order_seed}")
    if manifest.get("seed") != seed or manifest.get("schedule_seed") != order_seed:
        raise ValueError("Training seed metadata does not match the preregistered condition")
    if manifest.get("resolved_revision") != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("Base model revision changed")
    cohort = manifest.get("fixed_training_cohort", {})
    if (cohort.get("training_data_sha256") != cohort_hash
            or cohort.get("manifest_sha256") != cohort_manifest_hash):
        raise ValueError("Training run does not use the registered fixed support set")
    if manifest.get("settings", {}).get("task_order_design") != "common-random-task-positions-v1":
        raise ValueError("Training run lacks the preregistered two-task order design")
    expected = {"phrase_completion": 16, "sentence_completion": 48}
    counts = Counter(row["task_type"] for row in trace)
    if dict(counts) != expected or manifest.get("training_task_exposure_counts") != expected:
        raise ValueError("Training trace does not match the registered 16/48 exposure ratio")
    if manifest.get("training_update_log_sha256") != fingerprint(trace):
        raise ValueError("Training update trace hash mismatch")
    id_streams = {
        task: [row["example_id"] for row in trace if row["task_type"] == task]
        for task in expected
    }
    exposure = Counter(row["example_id"] for row in trace)
    return {
        "condition": f"seed{seed}_schedule{order_seed}",
        "training_manifest_sha256": sha256(manifest_path),
        "training_trace_sha256": sha256(trace_path),
        "training_updates": len(trace),
        "schedule_seed": order_seed,
        "task_order_design": manifest["settings"]["task_order_design"],
        "task_update_exposures": dict(sorted(counts.items())),
        "training_examples_seen_unique": len(exposure),
        "unique_examples_by_task": {
            task: len(set(ids)) for task, ids in id_streams.items()
        },
        "max_repeats_of_one_example": max(exposure.values()),
        "task_streams": id_streams,
        "lora_delta_global_frobenius_norm": json.loads(
            (run_dir / "delta_lora.json").read_text()
        )["delta_global_frobenius_norm"],
        "random_delta_global_frobenius_norm": json.loads(
            (run_dir / "delta_random.json").read_text()
        )["delta_global_frobenius_norm"],
        "run_dir": run_dir.as_posix(),
        "adapter_dir": (run_dir / "adapter_lora").as_posix(),
        "random_adapter_dir": (run_dir / "adapter_random").as_posix(),
        "delta_lora": (run_dir / "delta_lora.json").as_posix(),
        "delta_random": (run_dir / "delta_random.json").as_posix(),
    }


def load_campaign(path: Path, cases_by_id: dict[str, dict]) -> dict:
    run_path, ledger_path, report_path = (
        path / "run.json", path / "observations.jsonl", path / "report.json"
    )
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"Evaluation campaign is not complete: {path}")
    if sha256(ledger_path) != run.get("observations_sha256"):
        raise ValueError(f"Response ledger checksum mismatch: {path}")
    if sha256(report_path) != run.get("report_sha256"):
        raise ValueError(f"Evaluation report checksum mismatch: {path}")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm, case_id = row["arm"], row["case_id"]
        if arm not in ARMS or case_id not in cases_by_id or case_id in rows[arm]:
            raise ValueError(f"Unexpected or duplicate response: {arm}/{case_id}")
        case = cases_by_id[case_id]
        if (row.get("prompt") != case["prompt"]
                or row.get("answer_key") != case["answer"]
                or row.get("metrics") != grade_case(case, row.get("response", ""))):
            raise ValueError(f"Raw output or trusted grade mismatch: {arm}/{case_id}")
        rows[arm][case_id] = row
    expected_ids = set(cases_by_id)
    if any(set(rows[arm]) != expected_ids for arm in ARMS):
        raise ValueError(f"Campaign has missing arm/question responses: {path}")
    if sum(len(arm_rows) for arm_rows in rows.values()) != 1712:
        raise ValueError(f"Unexpected response-record count: {path}")
    cached = sum(bool(row.get("cache_hit")) for arm_rows in rows.values()
                 for row in arm_rows.values())
    if cached != 856 or run.get("cache_hits") != cached:
        raise ValueError(f"Expected 856 verified base/prompt cache hits, found {cached}")
    if run.get("model_generations") != 856 or run.get("response_records") != 1712:
        raise ValueError("Run-level cache/generation accounting mismatch")
    return {"run": run, "arms": dict(rows),
            "observations_sha256": sha256(ledger_path),
            "report_sha256": sha256(report_path),
            "response_records": 1712, "cache_hits": cached,
            "model_generations": 856}


def score(rows: dict[str, dict], cases_by_id: dict[str, dict]) -> dict:
    result = {"correct": sum(bool(row["metrics"]["correct"])
                              for row in rows.values()),
              "questions": len(rows),
              "single_word_format_passes": sum(
                  bool(row["metrics"].get("single_word_format"))
                  for row in rows.values()
              ),
              "by_task_type": {}}
    for task in ("phrase_completion", "sentence_completion"):
        subset = [row for row in rows.values() if row["task_type"] == task]
        result["by_task_type"][task] = {
            "correct": sum(bool(row["metrics"]["correct"]) for row in subset),
            "questions": len(subset),
            "families": len({row["family"] for row in subset}),
            "single_word_format_passes": sum(
                bool(row["metrics"].get("single_word_format")) for row in subset
            ),
            "accuracy": sum(bool(row["metrics"]["correct"]) for row in subset) / len(subset),
        }
    sentence = [row for row in rows.values()
                if row["task_type"] == "sentence_completion"]
    result["sentence_by_subject_number"] = {}
    for number in ("singular", "plural"):
        subset = [row for row in sentence
                  if cases_by_id[row["case_id"]].get("subject_number") == number]
        result["sentence_by_subject_number"][number] = {
            "correct": sum(bool(row["metrics"]["correct"]) for row in subset),
            "questions": len(subset),
        }
    result["accuracy"] = result["correct"] / result["questions"]
    return result


def paired_family_delta(reference: dict[str, dict], candidate: dict[str, dict],
                        seed: int, resamples: int = 5000) -> dict:
    common = sorted(reference.keys() & candidate.keys())
    if len(common) != 428:
        raise ValueError("Schedule contrast must use all 428 paired development questions")
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    improved = regressed = 0
    for case_id in common:
        before, after = reference[case_id], candidate[case_id]
        if before["family"] != after["family"] or before["task_type"] != after["task_type"]:
            raise ValueError("Paired question metadata mismatch")
        delta = int(bool(after["metrics"]["correct"])) - int(bool(before["metrics"]["correct"]))
        grouped[before["task_type"]][before["family"]].append(delta)
        improved += delta == 1
        regressed += delta == -1
    rng = random.Random(seed)
    draws = []
    by_task = {task: [] for task in grouped}
    for _ in range(resamples):
        weighted = 0.0
        for task, families in grouped.items():
            clusters = list(families.values())
            sampled = [value for cluster in rng.choices(clusters, k=len(clusters))
                       for value in cluster]
            mean = sum(sampled) / len(sampled)
            by_task[task].append(mean)
            weighted += len(sampled) / len(common) * mean
        draws.append(weighted)
    draws.sort()
    task_intervals = {}
    for task, values in by_task.items():
        values.sort()
        task_intervals[task] = [values[int(.025 * resamples)],
                                values[min(resamples - 1, int(.975 * resamples))]]
    return {
        "paired_questions": len(common),
        "improved": improved,
        "regressed": regressed,
        "accuracy_delta": (improved - regressed) / len(common),
        "paired_family_bootstrap95": [draws[int(.025 * resamples)],
                                      draws[min(resamples - 1, int(.975 * resamples))]],
        "paired_family_bootstrap95_by_task_type": task_intervals,
        "family_counts_by_task_type": {
            task: len(families) for task, families in grouped.items()
        },
    }


def analyze(dataset: Path, evidence: Path) -> dict:
    prereg_path = evidence / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    _, cases = load_cases(dataset, "dev")
    cases_by_id = {row["case_id"]: row for row in cases}
    if len(cases_by_id) != 428:
        raise ValueError("F026 requires exactly the previously inspected 428 dev questions")
    cohort_path = Path("evolution_lab/evidence/f025_fixed_support_exposure/training_cohort/manifest.json")
    cohort = json.loads(cohort_path.read_text())
    cohort_hash = cohort["training_data_sha256"]
    cohort_manifest_hash = sha256(cohort_path)
    if (cohort_hash != prereg["intervention"]["training_data_sha256"]
            or cohort_manifest_hash != prereg["intervention"]["cohort_manifest_sha256"]):
        raise ValueError("F025 fixed-cohort manifest differs from preregistration")
    conditions = {}
    campaigns = {}
    for seed in SEEDS:
        for order_seed in SCHEDULE_SEEDS:
            key = f"seed{seed}_schedule{order_seed}"
            conditions[key] = load_training(
                evidence, seed, order_seed, cohort_hash, cohort_manifest_hash
            )
            campaign_path = evidence / "evaluations" / key
            campaigns[key] = load_campaign(campaign_path, cases_by_id)
            conditions[key]["evaluation"] = {
                "observations_sha256": campaigns[key]["observations_sha256"],
                "report_sha256": campaigns[key]["report_sha256"],
                "model_generations": campaigns[key]["model_generations"],
                "cache_hits": campaigns[key]["cache_hits"],
                "response_records": campaigns[key]["response_records"],
                "scores": {arm: score(rows, cases_by_id)
                           for arm, rows in campaigns[key]["arms"].items()},
            }
    for seed in SEEDS:
        for task in ("phrase_completion", "sentence_completion"):
            streams = [conditions[f"seed{seed}_schedule{order_seed}"]["task_streams"][task]
                       for order_seed in SCHEDULE_SEEDS]
            if not all(stream == streams[0] for stream in streams[1:]):
                raise ValueError(f"Training example stream changed with order seed: {seed}/{task}")
    schedule_contrasts = {}
    for seed in SEEDS:
        pairs = {}
        for index, left in enumerate(SCHEDULE_SEEDS):
            for right in SCHEDULE_SEEDS[index + 1:]:
                left_key, right_key = f"seed{seed}_schedule{left}", f"seed{seed}_schedule{right}"
                left_rows = campaigns[left_key]["arms"]["adapter"]
                right_rows = campaigns[right_key]["arms"]["adapter"]
                pairs[f"schedule{right}_minus_schedule{left}"] = {
                    "paired_family_contrast": paired_family_delta(
                        left_rows, right_rows, seed=seed * 10000 + left * 100 + right
                    ),
                    "raw_response_changes": sum(
                        left_rows[case_id]["response"] != right_rows[case_id]["response"]
                        for case_id in left_rows
                    ),
                    "question_count": 428,
                }
        schedule_contrasts[f"training_seed_{seed}"] = pairs
    total_records = sum(campaign["response_records"] for campaign in campaigns.values())
    total_cache_hits = sum(campaign["cache_hits"] for campaign in campaigns.values())
    total_generations = sum(campaign["model_generations"] for campaign in campaigns.values())
    if (total_records, total_cache_hits, total_generations) != (10272, 5136, 5136):
        raise ValueError("Cumulative response/cache/generation accounting differs from preregistration")
    return {
        "study_id": prereg["study_id"],
        "status": "complete-development-diagnostic-no-promotion",
        "preregistration_sha256": sha256(prereg_path),
        "training_conditions": conditions,
        "within_seed_schedule_contrasts": schedule_contrasts,
        "accounting": {
            "training_updates": 384,
            "response_records": total_records,
            "cache_hits": total_cache_hits,
            "model_generations": total_generations,
            "unique_normalized_questions_added": 0,
            "unique_questions_per_campaign": 428,
            "software_tests": "record separately",
        },
        "interpretation_limits": [
            "All model behavior scores use the previously inspected F025 development questions.",
            "The v3 confirmation families were consumed by F024 and are not used here.",
            "Order-seed inference is conditional on one frozen 64-example cohort and one 16/48 task ratio.",
            "Seven phrase families provide weak phrase-specific uncertainty.",
            "The completion benchmark is narrow and does not establish broad capability preservation or transfer to Astra."
        ],
        "confirmation_consumed": False,
        "eligible_for_promotion": False,
        "candidate_quarantined": True,
        "evidence_root": evidence.as_posix() + "/",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evolution_lab/evidence/completion_v3"))
    parser.add_argument("--evidence", type=Path,
                        default=Path("evolution_lab/evidence/f026_task_order_sensitivity"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset, args.evidence)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
