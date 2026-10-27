"""Checksum, independently regrade, and summarize the frozen F029 campaigns."""

from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases, paired_arm_contrast
from evolution_lab.provenance import fingerprint


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
PREREG = json.loads((EVIDENCE / "preregistration.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_campaign(path: Path, dataset: Path, expected_arms: set[str],
                    condition: dict | None = None) -> tuple[dict, list[dict]]:
    run_path, report_path = path / "run.json", path / "report.json"
    plain_ledger, compressed_ledger = path / "observations.jsonl", path / "observations.jsonl.gz"
    if plain_ledger.is_file():
        ledger_bytes = plain_ledger.read_bytes()
    elif compressed_ledger.is_file():
        ledger_bytes = gzip.decompress(compressed_ledger.read_bytes())
    else:
        raise FileNotFoundError(f"No raw response ledger in {path}")
    run, report = json.loads(run_path.read_text()), json.loads(report_path.read_text())
    if run.get("status") != "complete":
        raise ValueError(f"Incomplete campaign: {path}")
    if hashlib.sha256(ledger_bytes).hexdigest() != run.get("observations_sha256") or sha256(report_path) != run.get("report_sha256"):
        raise ValueError(f"Campaign checksum mismatch: {path}")
    if run.get("spec_sha256") != hashlib.sha256(
        json.dumps(run["spec"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest():
        raise ValueError(f"Specification checksum mismatch: {path}")
    if run.get("actual_model", {}).get("resolved_revision") != PREREG["model"]["revision"]:
        raise ValueError(f"Wrong model revision: {path}")
    if condition is not None:
        spec = run["spec"]
        source_manifest_path = ROOT / condition["source_manifest"]
        if sha256(source_manifest_path) != condition["source_manifest_sha256"]:
            raise ValueError(f"Source manifest checksum differs from preregistration: {path}")
        source_manifest = json.loads(source_manifest_path.read_text())
        expected_source_fingerprint = fingerprint(source_manifest)
        if spec.get("adapter_sources", {}).get("adapter", {}).get(
            "source_manifest_sha256"
        ) != expected_source_fingerprint:
            raise ValueError(f"Wrong preregistered adapter provenance: {path}")
        expected_hashes = {
            "adapter": condition["lora_weights_sha256"],
            "random": condition["random_weights_sha256"],
        }
        for arm, expected_hash in expected_hashes.items():
            actual_hash = spec.get("adapter_hashes", {}).get(arm, {}).get(
                "adapter_model.safetensors"
            )
            if actual_hash != expected_hash:
                raise ValueError(f"Wrong preregistered {arm} checkpoint: {path}")
        if spec.get("adapter_delta_hashes", {}).get("adapter") != condition[
            "lora_delta_stats_sha256"
        ] or spec.get("adapter_delta_hashes", {}).get("random") != condition[
            "random_delta_stats_sha256"
        ]:
            raise ValueError(f"Wrong preregistered delta sidecar: {path}")
    _, cases = load_cases(dataset, "confirmation")
    case_by_id = {case["case_id"]: case for case in cases}
    rows = [json.loads(line) for line in ledger_bytes.decode("utf-8").splitlines() if line]
    if len(rows) != run.get("records_written"):
        raise ValueError(f"Record count mismatch: {path}")
    if {row["arm"] for row in rows} != expected_arms:
        raise ValueError(f"Missing or extra evaluation arm: {path}")
    for row in rows:
        case = case_by_id.get(row.get("case_id"))
        if case is None or row.get("prompt") != case["prompt"] or row.get("answer_key") != case["answer"]:
            raise ValueError(f"Unexpected question or trusted key: {path}")
        fresh = grade_case(case, row["response"])
        if fresh != row.get("metrics"):
            raise ValueError(f"Objective grade differs from raw response: {path}/{row.get('case_id')}")
    for arm in expected_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != len(case_by_id) or len({row["case_id"] for row in arm_rows}) != len(case_by_id):
            raise ValueError(f"Incomplete arm coverage: {path}/{arm}")
    return report, rows


def arm_summary(report: dict, arm: str, task: str | None = None) -> dict:
    row = report["arms"][arm]
    if task is not None:
        row = row["by_task_type"][task]
    return {
        "correct": row["correct"], "questions": row["unique_questions"],
        "accuracy": row["accuracy"], "strict_format_passes": row["strict_format_passes"],
    }


def campaign_reference(path: Path) -> dict:
    run = json.loads((path / "run.json").read_text())
    stored_ledger = path / "observations.jsonl.gz"
    if not stored_ledger.is_file():
        stored_ledger = path / "observations.jsonl"
    return {
        "path": str(path.relative_to(ROOT)),
        "run_id": run["run_id"],
        "response_records": run["records_written"],
        "model_generations": run["model_generations"],
        "cache_hits": run["cache_hits"],
        "spec_sha256": run["spec_sha256"],
        "observations_sha256": run["observations_sha256"],
        "observations_storage": stored_ledger.name,
        "observations_storage_sha256": sha256(stored_ledger),
        "report_sha256": run["report_sha256"],
    }


def main() -> None:
    completion = ROOT / PREREG["completion_dataset"]["path"]
    arithmetic = ROOT / PREREG["locality_dataset"]["path"]
    completion_base_report, completion_base_rows = verify_campaign(
        EVIDENCE / "completion_baseline_offline", completion, {"base", "prompt"}
    )
    arithmetic_base_report, arithmetic_base_rows = verify_campaign(
        EVIDENCE / "arithmetic_baseline", arithmetic, {"base", "prompt"}
    )
    baseline_rows = {
        "completion": {
            arm: {row["case_id"]: row["response"] for row in completion_base_rows if row["arm"] == arm}
            for arm in ("base", "prompt")
        },
        "arithmetic": {
            arm: {row["case_id"]: row["response"] for row in arithmetic_base_rows if row["arm"] == arm}
            for arm in ("base", "prompt")
        },
    }
    conditions = []
    all_campaign_rows = [*completion_base_rows, *arithmetic_base_rows]
    verified_campaigns = [
        campaign_reference(EVIDENCE / "completion_baseline_offline"),
        campaign_reference(EVIDENCE / "arithmetic_baseline"),
    ]
    for condition in PREREG["fixed_conditions"]:
        name = condition["condition"]
        item = {"condition": name}
        for task, dataset in (("completion", completion), ("arithmetic", arithmetic)):
            report, rows = verify_campaign(
                EVIDENCE / f"{task}_{name}", dataset,
                {"base", "prompt", "adapter", "random"}, condition,
            )
            verified_campaigns.append(campaign_reference(EVIDENCE / f"{task}_{name}"))
            if sum(bool(row.get("cache_hit")) for row in rows) != 2 * len(
                load_cases(dataset, "confirmation")[1]
            ):
                raise ValueError(f"Expected exactly cached baseline arms for {task}/{name}")
            for arm in ("base", "prompt"):
                responses = {row["case_id"]: row["response"] for row in rows if row["arm"] == arm}
                if responses != baseline_rows[task][arm]:
                    raise ValueError(f"Cached {arm} responses changed in {task}/{name}")
            item[task] = {
                "arms": {arm: arm_summary(report, arm) for arm in ("base", "prompt", "adapter", "random")},
                "paired_adapter_vs_prompt": paired_arm_contrast(rows, "prompt", "adapter"),
                "paired_adapter_vs_random": paired_arm_contrast(rows, "random", "adapter"),
            }
            if task == "completion":
                item[task]["task_types"] = {
                    task_type: {arm: arm_summary(report, arm, task_type)
                                for arm in ("base", "prompt", "adapter", "random")}
                    for task_type in ("phrase_completion", "sentence_completion")
                }
            all_campaign_rows.extend(rows)
        conditions.append(item)
    summary = {
        "id": PREREG["id"], "status": "complete-no-promotion",
        "model": PREREG["model"],
        "unique_confirmation_questions": {
            "completion": PREREG["completion_dataset"]["unique_questions"],
            "arithmetic": PREREG["locality_dataset"]["unique_questions"],
            "total": PREREG["evaluation"]["total_unique_confirmation_questions"],
        },
        "planned_new_generations": PREREG["evaluation"]["planned_new_model_generations"],
        "actual_model_generations": sum(x["model_generations"] for x in verified_campaigns),
        "verified_cached_baseline_rows": sum(x["cache_hits"] for x in verified_campaigns),
        "response_records": sum(x["response_records"] for x in verified_campaigns),
        "campaigns": len(conditions) * 2 + 2,
        "verified_campaigns": verified_campaigns,
        "conditions": conditions,
        "completion_confirmation_consumed": True,
        "arithmetic_confirmation_consumed": True,
        "eligible_for_promotion": False,
        "interpretation_limit": "Transfer evidence is limited to these curated phrase and synthetic subject-verb agreement families on one Qwen 0.5B revision. Arithmetic is a narrow locality probe, not broad capability preservation. Many base and prompt completions use the fixed 16-token greedy output budget on unfinished explanations, while adapters more often emit terse answers; exact-key score gains may therefore reflect response-length behavior rather than improved arithmetic reasoning. No trait map or Astra transfer claim follows.",
    }
    output = EVIDENCE / "summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(output), "conditions": len(conditions),
                      "completion_baseline": arm_summary(completion_base_report, "prompt"),
                      "arithmetic_baseline": arm_summary(arithmetic_base_report, "prompt")}, indent=2))


if __name__ == "__main__":
    main()
