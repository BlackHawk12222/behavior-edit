"""Recompute F025 outcomes from the immutable raw response ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases
from evolution_lab.provenance import fingerprint


RATIOS = {
    "phrase1_sentence63": (1, 63),
    "phrase16_sentence48": (16, 48),
    "phrase32_sentence32": (32, 32),
}
SEEDS = (42, 43)
ARMS = ("base", "prompt", "adapter", "random")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired_contrast(reference: dict[str, dict], candidate: dict[str, dict],
                     seed: int, resamples: int = 2000) -> dict:
    common = sorted(reference.keys() & candidate.keys())
    if not common:
        raise ValueError("Paired contrast has no common questions")
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    improved = regressed = 0
    for case_id in common:
        before = int(reference[case_id]["metrics"]["correct"])
        after = int(candidate[case_id]["metrics"]["correct"])
        row = reference[case_id]
        task = row["task_type"]
        groups[task][row["family"]].append(after - before)
        counts[task] += 1
        improved += before == 0 and after == 1
        regressed += before == 1 and after == 0

    rng = random.Random(seed)
    combined_draws = []
    task_draws = {task: [] for task in groups}
    total_families = sum(len(families) for families in groups.values())
    for _ in range(resamples):
        estimate = 0.0
        for task, families in groups.items():
            clusters = list(families.values())
            sampled = [value for cluster in rng.choices(clusters, k=len(clusters))
                       for value in cluster]
            task_mean = sum(sampled) / len(sampled)
            task_draws[task].append(task_mean)
            estimate += counts[task] / len(common) * task_mean
        combined_draws.append(estimate)
    combined_draws.sort()
    by_task = {}
    for task, draws in task_draws.items():
        draws.sort()
        by_task[task] = [draws[int(0.025 * resamples)],
                         draws[min(resamples - 1, int(0.975 * resamples))]]
    return {
        "reference": "reference",
        "candidate": "candidate",
        "paired_questions": len(common),
        "improved": improved,
        "regressed": regressed,
        "correctness_rate_delta": (improved - regressed) / len(common),
        "paired_family_bootstrap95": [combined_draws[int(0.025 * resamples)],
                                      combined_draws[min(resamples - 1, int(0.975 * resamples))]],
        "paired_family_bootstrap95_by_task_type": by_task,
        "family_count": total_families,
    }


def _score(rows: dict[str, dict], cases_by_id: dict[str, dict]) -> dict:
    output = {"correct": sum(bool(row["metrics"]["correct"]) for row in rows.values()),
              "questions": len(rows), "by_task_type": {}}
    task_types = sorted({row["task_type"] for row in rows.values()})
    for task in task_types:
        subset = [row for row in rows.values() if row["task_type"] == task]
        output["by_task_type"][task] = {
            "correct": sum(bool(row["metrics"]["correct"]) for row in subset),
            "questions": len(subset),
            "families": len({row["family"] for row in subset}),
        }
    sentence_rows = [row for row in rows.values() if row["task_type"] == "sentence_completion"]
    for field in ("subject_number", "attractor_number"):
        bins = defaultdict(list)
        for row in sentence_rows:
            case = cases_by_id[row["case_id"]]
            bins[case[field]].append(bool(row["metrics"]["correct"]))
        output[f"sentence_by_{field}"] = {
            value: {"correct": sum(values), "questions": len(values),
                    "accuracy": sum(values) / len(values)}
            for value, values in sorted(bins.items())
        }
    output["accuracy"] = output["correct"] / output["questions"]
    return output


def load_campaign(path: Path, cases_by_id: dict[str, dict]) -> dict:
    run_path = path / "run.json"
    report_path = path / "report.json"
    ledger_path = path / "observations.jsonl"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"Campaign is not complete: {path}")
    if sha256(ledger_path) != run.get("observations_sha256"):
        raise ValueError(f"Observation ledger checksum mismatch: {path}")
    if sha256(report_path) != run.get("report_sha256"):
        raise ValueError(f"Campaign report checksum mismatch: {path}")
    rows = defaultdict(dict)
    line_count = 0
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        line_count += 1
        arm, case_id = row["arm"], row["case_id"]
        if arm not in ARMS or case_id not in cases_by_id or case_id in rows[arm]:
            raise ValueError(f"Unexpected or duplicate response in {path}: {arm}/{case_id}")
        case = cases_by_id[case_id]
        if row.get("prompt") != case["prompt"] or row.get("answer_key") != case["answer"]:
            raise ValueError(f"Raw response is not bound to the expected question: {case_id}")
        if row.get("metrics") != grade_case(case, row.get("response", "")):
            raise ValueError(f"Saved trusted-key grade mismatch: {case_id}/{arm}")
        rows[arm][case_id] = row
    expected_ids = set(cases_by_id)
    if line_count != len(ARMS) * len(expected_ids):
        raise ValueError(f"Wrong response count in {path}: {line_count}")
    if any(set(rows[arm]) != expected_ids for arm in ARMS):
        raise ValueError(f"Campaign is missing an arm/question pair: {path}")
    return {"run": run, "arms": dict(rows), "response_records": line_count,
            "report_sha256": sha256(report_path),
            "observations_sha256": sha256(ledger_path)}


def load_training_runs(evidence: Path) -> dict[str, dict]:
    """Audit every saved training schedule against the frozen cohort and prereg."""
    cohort_path = evidence / "training_cohort" / "manifest.json"
    cohort = json.loads(cohort_path.read_text())
    cohort_data_hash = cohort["training_data_sha256"]
    expected_cohort_manifest_hash = sha256(cohort_path)
    runs = {}
    task_streams = defaultdict(dict)
    for seed in SEEDS:
        for ratio_name, (phrase_updates, sentence_updates) in RATIOS.items():
            condition = f"seed{seed}_{ratio_name}"
            run_dirs = sorted((evidence / "training" / condition).glob("*"))
            run_dirs = [path for path in run_dirs if (path / "manifest.json").is_file()]
            if len(run_dirs) != 1:
                raise ValueError(f"Expected exactly one training run for {condition}")
            run_dir = run_dirs[0]
            manifest_path = run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            trace_path = run_dir / "training.json"
            trace = json.loads(trace_path.read_text())
            expected_counts = {"phrase_completion": phrase_updates,
                               "sentence_completion": sentence_updates}
            if manifest.get("status") != "complete" or len(trace) != 64:
                raise ValueError(f"Incomplete training run: {condition}")
            if manifest.get("resolved_revision") != "7ae557604adf67be50417f59c2c2f167def9a775":
                raise ValueError(f"Base model revision mismatch: {condition}")
            if manifest.get("fixed_training_cohort", {}).get("training_data_sha256") != cohort_data_hash:
                raise ValueError(f"Fixed cohort data hash mismatch: {condition}")
            if manifest.get("fixed_training_cohort", {}).get("manifest_sha256") != expected_cohort_manifest_hash:
                raise ValueError(f"Fixed cohort manifest hash mismatch: {condition}")
            if manifest.get("training_task_exposure_counts") != expected_counts:
                raise ValueError(f"Training exposure counts mismatch: {condition}")
            if manifest.get("training_update_log_sha256") is None:
                raise ValueError(f"Training update trace is not hash-bound: {condition}")
            if manifest.get("training_updates") != 64:
                raise ValueError(f"Wrong optimizer-step count: {condition}")
            task_counts = Counter()
            task_trace_ids = defaultdict(list)
            total_ids = Counter()
            for row in trace:
                task = row["task_type"]
                example_id = row["example_id"]
                task_counts[task] += 1
                task_trace_ids[task].append(example_id)
                total_ids[example_id] += 1
            if dict(task_counts) != expected_counts:
                raise ValueError(f"Raw training trace disagrees with manifest: {condition}")
            if len(total_ids) != manifest.get("training_examples_seen_unique"):
                raise ValueError(f"Unique-example count mismatch: {condition}")
            task_streams[seed][ratio_name] = task_trace_ids
            runs[condition] = {
                "training_manifest_sha256": sha256(manifest_path),
                "training_trace_sha256": sha256(trace_path),
                "training_update_log_sha256": manifest["training_update_log_sha256"],
                "fixed_cohort_data_sha256": cohort_data_hash,
                "fixed_cohort_manifest_sha256": expected_cohort_manifest_hash,
                "optimizer_updates": 64,
                "task_update_exposures": dict(sorted(task_counts.items())),
                "unique_examples_seen_total": len(total_ids),
                "unique_examples_seen_by_task": {
                    task: len(set(ids)) for task, ids in sorted(task_trace_ids.items())
                },
                "max_repeats_of_one_example": max(total_ids.values()),
            }
    # For each seed, different ratios must draw nested prefixes from identical
    # task-specific streams; otherwise task composition would also change IDs.
    for seed in SEEDS:
        for task in ("phrase_completion", "sentence_completion"):
            streams = [task_streams[seed][name][task] for name in RATIOS]
            reference = max(streams, key=len)
            for stream in streams:
                if stream != reference[:len(stream)]:
                    raise ValueError(f"Training ID streams are not nested for seed {seed}/{task}")
    return runs


def analyze(dataset: Path, evidence: Path) -> dict:
    source_manifest, cases = load_cases(dataset, "dev")
    cases_by_id = {row["case_id"]: row for row in cases}
    if len(cases_by_id) != 428:
        raise ValueError(f"Expected the frozen 428-question development set, found {len(cases_by_id)}")
    cohort_path = evidence / "training_cohort" / "manifest.json"
    cohort = json.loads(cohort_path.read_text())
    cohort_examples_path = cohort_path.parent / cohort["cohort_examples_file"]
    if fingerprint(source_manifest["manifest"]) != cohort.get("source_dataset_manifest_sha256"):
        raise ValueError("Fixed cohort source dataset manifest hash mismatch")
    if sha256(cohort_examples_path) != cohort.get("cohort_examples_sha256"):
        raise ValueError("Fixed cohort JSONL file hash mismatch")
    cohort_examples = [json.loads(line) for line in cohort_examples_path.read_text().splitlines()
                       if line.strip()]
    if len(cohort_examples) != 64 or fingerprint(cohort_examples) != cohort.get("training_data_sha256"):
        raise ValueError("Fixed cohort rows do not match the registered data hash")
    if len({row["id"] for row in cohort_examples}) != len(cohort_examples):
        raise ValueError("Fixed cohort contains duplicate training IDs")

    campaigns = {}
    training_runs = load_training_runs(evidence)
    for seed in SEEDS:
        for ratio_name in RATIOS:
            name = f"seed{seed}_{ratio_name}"
            path = evidence / "evaluations" / name
            campaign = load_campaign(path, cases_by_id)
            campaigns[name] = campaign

    condition_summary = {}
    for seed in SEEDS:
        for ratio_name, (phrase_updates, sentence_updates) in RATIOS.items():
            name = f"seed{seed}_{ratio_name}"
            campaign = campaigns[name]
            arm_rows = campaign["arms"]
            result = {
                "training_updates": {"phrase_completion": phrase_updates,
                                     "sentence_completion": sentence_updates},
                "training_data_audit": training_runs[name],
                "model_generations": campaign["response_records"],
                "campaign_elapsed_seconds": campaign["run"]["elapsed_seconds"],
                "spec_sha256": campaign["run"]["spec_sha256"],
                "observations_sha256": campaign["observations_sha256"],
                "report_sha256": campaign["report_sha256"],
                "arms": {arm: _score(rows, cases_by_id)
                         for arm, rows in arm_rows.items()},
                "paired_contrasts": {
                    f"adapter_vs_{control}": _paired_contrast(
                        arm_rows[control], arm_rows["adapter"],
                        seed=seed * 100 + phrase_updates * 7 + sentence_updates,
                    )
                    for control in ("base", "prompt", "random")
                },
            }
            condition_summary[name] = result

    exposure_contrasts = {}
    for seed in SEEDS:
        for left, right in (("phrase1_sentence63", "phrase16_sentence48"),
                            ("phrase16_sentence48", "phrase32_sentence32"),
                            ("phrase1_sentence63", "phrase32_sentence32")):
            left_name, right_name = f"seed{seed}_{left}", f"seed{seed}_{right}"
            left_rows = campaigns[left_name]["arms"]["adapter"]
            right_rows = campaigns[right_name]["arms"]["adapter"]
            exposure_contrasts[f"seed{seed}_{right}_minus_{left}"] = _paired_contrast(
                left_rows, right_rows, seed=seed * 1000 + len(exposure_contrasts) + 1,
            )

    base_repeat_checks = {}
    for seed in SEEDS:
        base_name = f"seed{seed}_phrase1_sentence63"
        for ratio_name in tuple(RATIOS)[1:]:
            other_name = f"seed{seed}_{ratio_name}"
            for arm in ("base", "prompt"):
                a = campaigns[base_name]["arms"][arm]
                b = campaigns[other_name]["arms"][arm]
                identical = sum(a[key]["response"] == b[key]["response"]
                                for key in cases_by_id)
                base_repeat_checks[f"seed{seed}_{ratio_name}_{arm}"] = {
                    "identical_responses": identical,
                    "questions": len(cases_by_id),
                }

    total_responses = sum(campaign["response_records"] for campaign in campaigns.values())
    unique_prompt_hashes = {
        row["normalized_prompt_sha256"]
        for campaign in campaigns.values()
        for arm_rows in campaign["arms"].values()
        for row in arm_rows.values()
    }
    return {
        "study_id": "completion-v3-fixed-support-task-exposure-f025",
        "status": "exploratory-development-only",
        "confirmation_consumed": False,
        "dataset_split": "previously inspected dev; not independent confirmation",
        "unique_questions": len(cases_by_id),
        "unique_normalized_prompt_hashes": len(unique_prompt_hashes),
        "model_response_records": total_responses,
        "planned_model_response_records": len(SEEDS) * len(RATIOS) * len(ARMS) * len(cases_by_id),
        "training_seeds": list(SEEDS),
        "fixed_support_cohort": "training_cohort/manifest.json",
        "fixed_support_cohort_data_sha256": json.loads(
            (evidence / "training_cohort" / "manifest.json").read_text()
        )["training_data_sha256"],
        "conditions": condition_summary,
        "within_seed_exposure_contrasts": exposure_contrasts,
        "base_and_prompt_repeat_checks": base_repeat_checks,
        "uncertainty_limits": [
            "Paired family bootstrap resamples benchmark families, not training seeds.",
            "Only two adapter-training seeds and seven phrase-development families are available.",
            "All 428 development questions were previously inspected; estimates are exploratory and cannot establish independent generalization.",
            "The exposure treatment changes the ordered task schedule as well as update counts; optimization-order effects are not separated.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evolution_lab/evidence/completion_v3"))
    parser.add_argument("--evidence", type=Path,
                        default=Path("evolution_lab/evidence/f025_fixed_support_exposure"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset, args.evidence)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
        print(f"F025 summary saved: {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
