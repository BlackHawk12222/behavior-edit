"""Integrity audit and paired analysis for preregistered F027 cohort contrasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases
from evolution_lab.provenance import fingerprint

ROOT = Path("evolution_lab/evidence/f027_cohort_content_sensitivity")
COHORTS = ("A", "B")
SEEDS = (42, 43, 44)
ORDERS = (901, 903)
ARMS = ("base", "prompt", "adapter", "random")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_run(path: Path) -> Path:
    runs = sorted(child for child in path.iterdir()
                  if child.is_dir() and (child / "manifest.json").is_file())
    if len(runs) != 1:
        raise ValueError(f"Expected one completed training run under {path}, found {len(runs)}")
    return runs[0]


def load_training(cohort: str, seed: int, order: int,
                  prereg: dict, campaign: dict) -> dict:
    key = f"cohort{cohort}_seed{seed}_order{order}"
    condition = campaign.get("conditions", {}).get(key)
    if not condition or condition.get("status") != "complete":
        raise ValueError(f"Training condition incomplete: {key}")
    run_dir = one_run(ROOT / "training" / key)
    manifest_path, trace_path = run_dir / "manifest.json", run_dir / "training.json"
    manifest, trace = json.loads(manifest_path.read_text()), json.loads(trace_path.read_text())
    cohort_info = prereg["data_intervention"]["cohorts"][cohort]
    expected_ratio = {"phrase_completion": 16, "sentence_completion": 48}
    if (manifest.get("status") != "complete" or manifest.get("seed") != seed
            or manifest.get("schedule_seed") != order
            or manifest.get("resolved_revision") != REVISION
            or manifest.get("training_updates") != 64 or len(trace) != 64
            or manifest.get("training_task_exposure_counts") != expected_ratio
            or manifest.get("fixed_training_cohort", {}).get("training_data_sha256")
            != cohort_info["training_data_sha256"]
            or manifest.get("fixed_training_cohort", {}).get("manifest_sha256")
            != cohort_info["manifest_sha256"]):
        raise ValueError(f"Training manifest differs from preregistered condition: {key}")
    if manifest.get("training_update_log_sha256") != fingerprint(trace):
        raise ValueError(f"Training trace fingerprint mismatch: {key}")
    counts = Counter(row["task_type"] for row in trace)
    if dict(counts) != expected_ratio:
        raise ValueError(f"Training task exposure mismatch: {key}")
    streams = {task: [row["example_id"] for row in trace if row["task_type"] == task]
               for task in expected_ratio}
    exposure = Counter(row["example_id"] for row in trace)
    return {
        "condition": key,
        "training_manifest_sha256": sha256(manifest_path),
        "training_trace_sha256": sha256(trace_path),
        "training_updates": len(trace),
        "training_examples_seen_unique": len(exposure),
        "unique_examples_by_task": {task: len(set(ids)) for task, ids in streams.items()},
        "max_repeats_of_one_example": max(exposure.values()),
        "task_streams": streams,
        "lora_delta_global_frobenius_norm": json.loads(
            (run_dir / "delta_lora.json").read_text()
        )["delta_global_frobenius_norm"],
        "random_delta_global_frobenius_norm": json.loads(
            (run_dir / "delta_random.json").read_text()
        )["delta_global_frobenius_norm"],
        "run_dir": run_dir.as_posix(),
        "adapter_dir": (run_dir / "adapter_lora").as_posix(),
        "random_adapter_dir": (run_dir / "adapter_random").as_posix(),
    }


def load_evaluation(path: Path, cases: dict[str, dict]) -> dict:
    run_path, ledger_path, report_path = path / "run.json", path / "observations.jsonl", path / "report.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"Incomplete evaluation campaign: {path}")
    if sha256(ledger_path) != run.get("observations_sha256"):
        raise ValueError(f"Observation checksum mismatch: {path}")
    if sha256(report_path) != run.get("report_sha256"):
        raise ValueError(f"Report checksum mismatch: {path}")
    rows: dict[str, dict[str, dict]] = defaultdict(dict)
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        arm, case_id = row["arm"], row["case_id"]
        if arm not in ARMS or case_id not in cases or case_id in rows[arm]:
            raise ValueError(f"Unexpected or duplicate evaluation row: {arm}/{case_id}")
        case = cases[case_id]
        if (row.get("prompt") != case["prompt"]
                or row.get("answer_key") != case["answer"]
                or row.get("metrics") != grade_case(case, row.get("response", ""))):
            raise ValueError(f"Trusted-grade or prompt mismatch: {arm}/{case_id}")
        rows[arm][case_id] = row
    expected_ids = set(cases)
    if any(set(rows[arm]) != expected_ids for arm in ARMS):
        raise ValueError(f"Missing arm or question records: {path}")
    record_count = sum(map(len, rows.values()))
    cache_hits = sum(bool(row.get("cache_hit")) for arm in rows.values() for row in arm.values())
    if (record_count, cache_hits, run.get("response_records"), run.get("cache_hits"),
            run.get("model_generations")) != (1712, 856, 1712, 856, 856):
        raise ValueError(f"Cache/generation accounting mismatch: {path}")
    return {
        "run": run,
        "arms": dict(rows),
        "observations_sha256": sha256(ledger_path),
        "report_sha256": sha256(report_path),
        "response_records": record_count,
        "cache_hits": cache_hits,
        "model_generations": run["model_generations"],
    }


def score(rows: dict[str, dict], cases: dict[str, dict]) -> dict:
    out = {
        "correct": sum(bool(r["metrics"]["correct"]) for r in rows.values()),
        "questions": len(rows),
        "single_word_format_passes": sum(bool(r["metrics"].get("single_word_format"))
                                          for r in rows.values()),
        "by_task_type": {},
        "sentence_by_subject_number": {},
    }
    for task in ("phrase_completion", "sentence_completion"):
        subset = [r for r in rows.values() if r["task_type"] == task]
        out["by_task_type"][task] = {
            "correct": sum(bool(r["metrics"]["correct"]) for r in subset),
            "questions": len(subset),
            "families": len({r["family"] for r in subset}),
        }
    for number in ("singular", "plural"):
        subset = [r for r in rows.values() if r["task_type"] == "sentence_completion"
                  and cases[r["case_id"]].get("subject_number") == number]
        out["sentence_by_subject_number"][number] = {
            "correct": sum(bool(r["metrics"]["correct"]) for r in subset),
            "questions": len(subset),
        }
    out["accuracy"] = out["correct"] / out["questions"]
    return out


def paired_family_delta(reference: dict[str, dict], candidate: dict[str, dict],
                        seed: int, resamples: int = 5000) -> dict:
    common = sorted(reference.keys() & candidate.keys())
    if len(common) != 428:
        raise ValueError("Paired contrast must cover all 428 development questions")
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
    draws, per_task = [], {task: [] for task in grouped}
    for _ in range(resamples):
        weighted = 0.0
        for task, families in grouped.items():
            clusters = list(families.values())
            sampled = [value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster]
            mean = sum(sampled) / len(sampled)
            per_task[task].append(mean)
            weighted += len(sampled) / len(common) * mean
        draws.append(weighted)
    draws.sort()
    task_intervals = {}
    for task, values in per_task.items():
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
        "family_counts_by_task_type": {task: len(families) for task, families in grouped.items()},
    }


def analyze(dataset: Path = Path("evolution_lab/evidence/completion_v3"),
            evidence: Path = ROOT) -> dict:
    prereg_path = evidence / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    registry = json.loads(Path("evolution_lab/experiment_registry.json").read_text())
    entry = next((row for row in registry["studies"]
                  if row.get("preregistration") == prereg_path.as_posix()), None)
    if not entry or entry.get("preregistration_sha256") != sha256(prereg_path):
        raise ValueError("Preregistration does not match registry hash")
    _, case_rows = load_cases(dataset, "dev")
    cases = {row["case_id"]: row for row in case_rows}
    if len(cases) != 428 or len({row["family"] for row in cases.values()}) != 32:
        raise ValueError("F027 evaluation split differs from preregistration")

    cohort_rows = {}
    for cohort in COHORTS:
        info = prereg["data_intervention"]["cohorts"][cohort]
        manifest_path = evidence / f"cohort_{cohort}" / "manifest.json"
        data_path = manifest_path.parent / "cohort_examples.jsonl"
        manifest = json.loads(manifest_path.read_text())
        if (sha256(manifest_path) != info["manifest_sha256"]
                or manifest.get("training_data_sha256") != info["training_data_sha256"]
                or sha256(data_path) != info["cohort_examples_sha256"]):
            raise ValueError(f"Cohort {cohort} hash differs from preregistration")
        rows = [json.loads(line) for line in data_path.read_text().splitlines()]
        if fingerprint(rows) != info["training_data_sha256"] or len(rows) != 64:
            raise ValueError(f"Cohort {cohort} examples fail hash/size check")
        source_train = {row["case_id"]: row for row in load_cases(dataset, "train")[1]}
        for row in rows:
            source = source_train.get(row["id"])
            if (source is None or row["prompt"] != source["prompt"]
                    or row["answer"] != source["answer"] or row["family"] != source["family"]):
                raise ValueError(f"Cohort {cohort} contains changed or non-train example {row['id']}")
        cohort_rows[cohort] = {row["id"]: row for row in rows}
    overlap = {cohort: set(cohort_rows[cohort]) for cohort in COHORTS}
    if len(overlap["A"] & overlap["B"]) != prereg["data_intervention"]["cohort_overlap"]["total_exact_ids"]:
        raise ValueError("Cohort overlap differs from preregistration")

    train_campaign = json.loads((evidence / "training" / "training_campaign.json").read_text())
    eval_campaign = json.loads((evidence / "evaluations" / "evaluation_campaign.json").read_text())
    if (train_campaign.get("status") != "complete" or len(train_campaign.get("conditions", {})) != 12
            or eval_campaign.get("status") != "complete" or len(eval_campaign.get("conditions", {})) != 12):
        raise ValueError("F027 training or evaluation matrix is incomplete")
    conditions, campaigns = {}, {}
    for cohort in COHORTS:
        for seed in SEEDS:
            for order in ORDERS:
                key = f"cohort{cohort}_seed{seed}_order{order}"
                conditions[key] = load_training(cohort, seed, order, prereg, train_campaign)
                campaigns[key] = load_evaluation(evidence / "evaluations" / key, cases)
                conditions[key]["evaluation"] = {
                    "observations_sha256": campaigns[key]["observations_sha256"],
                    "report_sha256": campaigns[key]["report_sha256"],
                    "response_records": campaigns[key]["response_records"],
                    "cache_hits": campaigns[key]["cache_hits"],
                    "model_generations": campaigns[key]["model_generations"],
                    "scores": {arm: score(rows, cases) for arm, rows in campaigns[key]["arms"].items()},
                }
    for cohort in COHORTS:
        for seed in SEEDS:
            left = conditions[f"cohort{cohort}_seed{seed}_order901"]["task_streams"]
            right = conditions[f"cohort{cohort}_seed{seed}_order903"]["task_streams"]
            if left != right:
                raise ValueError(f"Within-task examples changed across task orders: cohort {cohort}, seed {seed}")

    cohort_contrasts, schedule_contrasts, base_prompt_checks = {}, {}, {}
    for seed in SEEDS:
        for order in ORDERS:
            a_key, b_key = f"cohortA_seed{seed}_order{order}", f"cohortB_seed{seed}_order{order}"
            a_rows = campaigns[a_key]["arms"]["adapter"]
            b_rows = campaigns[b_key]["arms"]["adapter"]
            cohort_contrasts[f"seed{seed}_order{order}"] = {
                "paired_family_contrast_B_minus_A": paired_family_delta(a_rows,b_rows,seed*10000+order),
                "raw_response_changes": sum(a_rows[c]["response"] != b_rows[c]["response"] for c in a_rows),
            }
        for cohort in COHORTS:
            left = campaigns[f"cohort{cohort}_seed{seed}_order901"]["arms"]["adapter"]
            right = campaigns[f"cohort{cohort}_seed{seed}_order903"]["arms"]["adapter"]
            schedule_contrasts[f"cohort{cohort}_seed{seed}_order903_minus_901"] = {
                "paired_family_contrast": paired_family_delta(left,right,seed*100000+ord(cohort)+27),
                "raw_response_changes": sum(left[c]["response"] != right[c]["response"] for c in left),
            }
    for key, campaign in campaigns.items():
        for arm in ("base", "prompt"):
            rows = campaign["arms"][arm]
            content = {case_id: row["response"] for case_id,row in rows.items()}
            if key not in base_prompt_checks:
                base_prompt_checks[key] = {}
            base_prompt_checks[key][arm] = {
                "identical_to_first_campaign": None,
                "questions": len(content),
                "all_cached": all(row.get("cache_hit") for row in rows.values()),
            }
    reference = campaigns["cohortA_seed42_order901"]["arms"]
    for key,campaign in campaigns.items():
        for arm in ("base","prompt"):
            if any(reference[arm][case_id]["response"] != campaign["arms"][arm][case_id]["response"]
                   for case_id in cases):
                raise ValueError(f"Cached {arm} responses changed across campaigns: {key}")
            base_prompt_checks[key][arm]["identical_to_first_campaign"] = True
    total_records = sum(x["response_records"] for x in campaigns.values())
    total_cache = sum(x["cache_hits"] for x in campaigns.values())
    total_generations = sum(x["model_generations"] for x in campaigns.values())
    if (total_records,total_cache,total_generations)!=(20544,10272,10272):
        raise ValueError("Total record/cache/generation accounting differs from preregistration")
    return {
        "study_id":prereg["study_id"],
        "status":"complete-development-cohort-content-diagnostic-no-promotion",
        "preregistration_sha256":sha256(prereg_path),
        "training_conditions":conditions,
        "cohort_content_contrasts":cohort_contrasts,
        "within_cohort_schedule_contrasts":schedule_contrasts,
        "base_prompt_replay_checks":base_prompt_checks,
        "accounting":{"training_updates":768,"response_records":total_records,
                      "cache_hits":total_cache,"model_generations":total_generations,
                      "unique_normalized_questions_added":0,"software_tests":"record separately"},
        "interpretation_limits":[
            "All 428 v3 development questions were previously inspected in F025/F026.",
            "This study varies cohort identity/content, not label correctness or data quality.",
            "Only three training seeds, two cohorts and two task orders were tested.",
            "The phrase subset has seven families, limiting phrase-specific uncertainty.",
            "The v3 confirmation split is retired and unused; no fresh confirmation occurred.",
            "Results do not establish broad capability preservation, a weight-to-trait map or Astra transfer.",
        ],
        "confirmation_consumed":False,
        "eligible_for_promotion":False,
        "candidate_quarantined":True,
        "evidence_root":evidence.as_posix()+"/",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evolution_lab/evidence/completion_v3"))
    parser.add_argument("--evidence", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.dataset,args.evidence)
    rendered=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered,end="")


if __name__ == "__main__":
    main()
