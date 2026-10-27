"""Integrity audit and paired analysis for the F028 example-exposure study."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases
from evolution_lab.provenance import fingerprint

ROOT = Path("evolution_lab/evidence/f028_example_exposure")
SEEDS = (42, 43, 44)
POLICIES = ("with-replacement", "without-replacement")
ARMS = ("base", "prompt", "adapter", "random")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
COHORT_MANIFEST = Path("evolution_lab/evidence/f027_cohort_content_sensitivity/cohort_A/manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training(key: str, prereg: dict, campaign: dict) -> dict:
    condition = campaign.get("conditions", {}).get(key)
    if not condition or condition.get("status") != "complete":
        raise ValueError(f"F028 training incomplete: {key}")
    run_dir = Path(condition["run_dir"])
    manifest_path, trace_path = run_dir / "manifest.json", run_dir / "training.json"
    manifest, trace = json.loads(manifest_path.read_text()), json.loads(trace_path.read_text())
    expected_policy, seed = condition["example_sampling"], condition["seed"]
    expected_ratio = {"phrase_completion": 16, "sentence_completion": 48}
    if (sha256(manifest_path) != condition["manifest_sha256"]
            or sha256(trace_path) != condition["training_trace_sha256"]
            or manifest.get("status") != "complete" or manifest.get("seed") != seed
            or manifest.get("schedule_seed") != 2603 or manifest.get("resolved_revision") != REVISION
            or manifest.get("training_updates") != 64
            or manifest.get("training_task_exposure_counts") != expected_ratio
            or manifest.get("settings", {}).get("example_sampling") != expected_policy
            or manifest.get("fixed_training_cohort", {}).get("manifest_sha256")
            != prereg["design"]["fixed_cohort_manifest_sha256"]):
        raise ValueError(f"F028 training manifest differs from preregistered treatment: {key}")
    if manifest.get("training_update_log_sha256") != fingerprint(trace):
        raise ValueError(f"F028 training trace fingerprint mismatch: {key}")
    streams = {task: [row["example_id"] for row in trace if row["task_type"] == task]
               for task in expected_ratio}
    coverage = {task: len(set(ids)) for task, ids in streams.items()}
    exposure = Counter(row["example_id"] for row in trace)
    if expected_policy == "without-replacement" and coverage != expected_ratio:
        raise ValueError(f"F028 without-replacement did not cover support exactly once: {key}")
    return {"key": key, "policy": expected_policy, "seed": seed,
            "training_updates": len(trace), "training_examples_seen_unique": len(exposure),
            "unique_examples_by_task": coverage,
            "max_repeats_of_one_example": max(exposure.values()),
            "trace_sha256": sha256(trace_path), "manifest_sha256": sha256(manifest_path),
            "run_dir": run_dir.as_posix(),
            "adapter_dir": (run_dir / "adapter_lora").as_posix(),
            "random_adapter_dir": (run_dir / "adapter_random").as_posix(),
            "delta_lora_sha256": sha256(run_dir / "delta_lora.json"),
            "delta_random_sha256": sha256(run_dir / "delta_random.json"),
            "streams": streams}


def load_evaluation(path: Path, cases: dict[str, dict]) -> dict:
    run = json.loads((path / "run.json").read_text())
    ledger, report = path / "observations.jsonl", path / "report.json"
    if (run.get("status") != "complete" or sha256(ledger) != run.get("observations_sha256")
            or sha256(report) != run.get("report_sha256")):
        raise ValueError(f"F028 evaluation status/checksum failure: {path}")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm, case_id = row["arm"], row["case_id"]
        if arm not in ARMS or case_id not in cases or case_id in rows[arm]:
            raise ValueError(f"F028 unexpected/duplicate evaluation row: {arm}/{case_id}")
        case = cases[case_id]
        if (row.get("prompt") != case["prompt"] or row.get("answer_key") != case["answer"]
                or row.get("metrics") != grade_case(case, row.get("response", ""))):
            raise ValueError(f"F028 raw response failed trusted regrade: {arm}/{case_id}")
        rows[arm][case_id] = row
    if any(set(rows[arm]) != set(cases) for arm in ARMS):
        raise ValueError(f"F028 missing cases/arms: {path}")
    count = sum(len(arm_rows) for arm_rows in rows.values())
    cache_hits = sum(bool(row.get("cache_hit")) for arm_rows in rows.values() for row in arm_rows.values())
    if (count, cache_hits, run.get("response_records"), run.get("cache_hits"), run.get("model_generations")) != (1712, 856, 1712, 856, 856):
        raise ValueError(f"F028 evaluation accounting failure: {path}")
    return {"arms": dict(rows), "response_records": count, "cache_hits": cache_hits,
            "model_generations": run["model_generations"],
            "observations_sha256": sha256(ledger), "report_sha256": sha256(report)}


def score(rows: dict[str, dict], cases: dict[str, dict]) -> dict:
    out = {"correct": sum(bool(row["metrics"]["correct"]) for row in rows.values()),
           "questions": len(rows),
           "single_word_format_passes": sum(bool(row["metrics"].get("single_word_format"))
                                             for row in rows.values()),
           "by_task_type": {}}
    for task in ("phrase_completion", "sentence_completion"):
        subset = [row for row in rows.values() if row["task_type"] == task]
        out["by_task_type"][task] = {
            "correct": sum(bool(row["metrics"]["correct"]) for row in subset),
            "questions": len(subset), "families": len({row["family"] for row in subset})}
    out["accuracy"] = out["correct"] / out["questions"]
    return out


def paired_family_delta(with_rows: dict[str, dict], without_rows: dict[str, dict], seed: int) -> dict:
    common = sorted(with_rows.keys() & without_rows.keys())
    if len(common) != 428:
        raise ValueError("F028 paired contrast must cover all 428 dev questions")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    improved = regressed = 0
    for case_id in common:
        before, after = with_rows[case_id], without_rows[case_id]
        delta = int(bool(after["metrics"]["correct"])) - int(bool(before["metrics"]["correct"]))
        groups[before["task_type"]][before["family"]].append(delta)
        improved += delta == 1
        regressed += delta == -1
    rng = random.Random(seed)
    draws, by_task = [], {task: [] for task in groups}
    for _ in range(5000):
        weighted = 0.0
        for task, families in groups.items():
            clusters = list(families.values())
            sample = [value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster]
            mean = sum(sample) / len(sample)
            by_task[task].append(mean)
            weighted += len(sample) / len(common) * mean
        draws.append(weighted)
    draws.sort()
    return {"paired_questions": len(common), "improved": improved, "regressed": regressed,
            "accuracy_delta": (improved - regressed) / len(common),
            "paired_family_bootstrap95": [draws[124], draws[4874]],
            "family_counts_by_task_type": {task: len(families) for task, families in groups.items()}}


def analyze() -> dict:
    prereg_path = ROOT / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    registry = json.loads(Path("evolution_lab/experiment_registry.json").read_text())
    entry = next((row for row in registry["studies"]
                  if row.get("preregistration") == prereg_path.as_posix()), None)
    if not entry or entry.get("preregistration_sha256") != sha256(prereg_path):
        raise ValueError("F028 preregistration is not registered by hash")
    if sha256(COHORT_MANIFEST) != prereg["design"]["fixed_cohort_manifest_sha256"]:
        raise ValueError("F028 fixed cohort manifest hash changed")
    _, dev_rows = load_cases(Path("evolution_lab/evidence/completion_v3"), "dev")
    cases = {row["case_id"]: row for row in dev_rows}
    if len(cases) != 428 or len({row["family"] for row in dev_rows}) != 32:
        raise ValueError("F028 evaluation set differs from preregistration")
    training_campaign = json.loads((ROOT / "training" / "training_campaign.json").read_text())
    evaluation_campaign = json.loads((ROOT / "evaluations" / "evaluation_campaign.json").read_text())
    if (training_campaign.get("status") != "complete" or len(training_campaign.get("conditions", {})) != 6
            or evaluation_campaign.get("status") != "complete" or len(evaluation_campaign.get("conditions", {})) != 6):
        raise ValueError("F028 matrix is incomplete")

    conditions, evals = {}, {}
    for seed in SEEDS:
        for policy in POLICIES:
            key = f"{policy}_seed{seed}"
            conditions[key] = load_training(key, prereg, training_campaign)
            evals[key] = load_evaluation(ROOT / "evaluations" / key, cases)
            conditions[key]["scores"] = {
                arm: score(rows, cases) for arm, rows in evals[key]["arms"].items()}
            conditions[key]["evaluation"] = {
                "response_records": evals[key]["response_records"],
                "cache_hits": evals[key]["cache_hits"],
                "model_generations": evals[key]["model_generations"],
                "observations_sha256": evals[key]["observations_sha256"],
                "report_sha256": evals[key]["report_sha256"],
            }
    contrasts = {}
    for seed in SEEDS:
        with_key, without_key = f"with-replacement_seed{seed}", f"without-replacement_seed{seed}"
        contrasts[f"seed{seed}_without_minus_with"] = paired_family_delta(
            evals[with_key]["arms"]["adapter"], evals[without_key]["arms"]["adapter"], seed * 1000 + 28)
        if conditions[with_key]["streams"] != conditions[without_key]["streams"]:
            # Task-order placement should match, while sampled example identity is the intervention.
            left = [row["task_type"] for row in json.loads(
                (Path(conditions[with_key]["run_dir"]) / "training.json").read_text())]
            right = [row["task_type"] for row in json.loads(
                (Path(conditions[without_key]["run_dir"]) / "training.json").read_text())]
            if left != right:
                raise ValueError(f"F028 task-order positions differ across policies for seed {seed}")
    base_reference = evals["with-replacement_seed42"]["arms"]
    cache_consistent = {}
    for key, evaluation in evals.items():
        cache_consistent[key] = {}
        for arm in ("base", "prompt"):
            same = all(base_reference[arm][case_id]["response"] == evaluation["arms"][arm][case_id]["response"]
                       for case_id in cases)
            if not same or not all(row.get("cache_hit") for row in evaluation["arms"][arm].values()):
                raise ValueError(f"F028 cached baseline replay changed: {key}/{arm}")
            cache_consistent[key][arm] = {"identical_to_first_campaign": True, "questions": len(cases)}

    return {
        "study_id": prereg["study_id"],
        "status": "complete-development-example-exposure-diagnostic-no-promotion",
        "preregistration_sha256": sha256(prereg_path),
        "training_conditions": conditions,
        "without_minus_with_contrasts": contrasts,
        "base_prompt_cache_checks": cache_consistent,
        "accounting": {"training_updates": 384, "response_records": 10272,
                       "cache_hits": 5136, "model_generations": 5136,
                       "unique_normalized_questions_added": 0},
        "interpretation_limits": prereg["interpretation_limits"],
        "confirmation_consumed": False, "eligible_for_promotion": False,
        "candidate_quarantined": True,
        "evidence_root": ROOT.as_posix() + "/",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
