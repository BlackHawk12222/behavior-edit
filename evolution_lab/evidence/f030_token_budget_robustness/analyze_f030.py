"""Checksum every F030 response, regrade it, and compare paired token budgets."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from evolution_lab.case_eval import grade_case, load_cases, normalized_question, paired_arm_contrast
from evolution_lab.provenance import fingerprint


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
PREREG = json.loads((EVIDENCE / "preregistration.json").read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_ledger(directory: Path) -> tuple[bytes, str]:
    plain = directory / "observations.jsonl"
    compressed = directory / "observations.jsonl.gz"
    if plain.is_file():
        return plain.read_bytes(), plain.name
    if compressed.is_file():
        return gzip.decompress(compressed.read_bytes()), compressed.name
    raise FileNotFoundError(f"No raw response ledger in {directory}")


def verify_campaign(directory: Path, expected_arms: set[str],
                    budget: int, condition: dict | None = None) -> tuple[dict, list[dict], dict]:
    run_path, report_path = directory / "run.json", directory / "report.json"
    run = json.loads(run_path.read_text())
    report = json.loads(report_path.read_text())
    ledger_bytes, ledger_name = read_ledger(directory)
    if run.get("status") != "complete":
        raise ValueError(f"Campaign is incomplete: {directory}")
    if digest(ledger_bytes) != run.get("observations_sha256"):
        raise ValueError(f"Response ledger checksum mismatch: {directory}")
    if digest(report_path.read_bytes()) != run.get("report_sha256"):
        raise ValueError(f"Report checksum mismatch: {directory}")
    if fingerprint(run["spec"]) != run.get("spec_sha256"):
        raise ValueError(f"Run specification checksum mismatch: {directory}")
    spec = run["spec"]
    if spec.get("max_new_tokens") != budget:
        raise ValueError(f"Wrong token budget: {directory}")
    if run.get("actual_model", {}).get("resolved_revision") != PREREG["model"]["revision"]:
        raise ValueError(f"Wrong model revision: {directory}")
    if run.get("actual_model", {}).get("device") != "cpu" or run.get("actual_model", {}).get(
        "precision"
    ) != "torch.float32":
        raise ValueError(f"Unexpected runtime: {directory}")
    if spec.get("family_sampling", {}).get("family_limit") != PREREG["dataset"]["family_limit"] or spec.get(
        "family_sampling", {}
    ).get("family_seed") != PREREG["dataset"]["family_seed"]:
        raise ValueError(f"Wrong preregistered family sample: {directory}")
    if spec.get("unique_questions") != PREREG["dataset"]["unique_questions"]:
        raise ValueError(f"Wrong question count: {directory}")
    if spec.get("arms") is None or set(spec["arms"]) != expected_arms:
        raise ValueError(f"Unexpected arms: {directory}")

    if condition is not None:
        expected_hashes = {
            "adapter": condition["lora_weights_sha256"],
            "random": condition["random_weights_sha256"],
        }
        for arm, expected_hash in expected_hashes.items():
            actual = spec.get("adapter_hashes", {}).get(arm, {}).get("adapter_model.safetensors")
            if actual != expected_hash:
                raise ValueError(f"Wrong preregistered {arm} checkpoint: {directory}")
        manifest_path = ROOT / condition["source_manifest"]
        if digest(manifest_path.read_bytes()) != condition["source_manifest_sha256"]:
            raise ValueError(f"Source manifest checksum mismatch: {directory}")
        if spec.get("adapter_sources", {}).get("adapter", {}).get(
            "source_manifest_sha256"
        ) != fingerprint(json.loads(manifest_path.read_text())):
            raise ValueError(f"Wrong adapter provenance: {directory}")
        expected_delta_hashes = {
            "adapter": condition["lora_delta_stats_sha256"],
            "random": condition["random_delta_stats_sha256"],
        }
        if spec.get("adapter_delta_hashes") != expected_delta_hashes:
            raise ValueError(f"Wrong delta-stat sidecars: {directory}")

    _, cases = load_cases(
        ROOT / PREREG["dataset"]["path"], "confirmation",
        family_limit=PREREG["dataset"]["family_limit"],
        family_seed=PREREG["dataset"]["family_seed"],
    )
    case_by_id = {row["case_id"]: row for row in cases}
    rows = [json.loads(line) for line in ledger_bytes.decode("utf-8").splitlines() if line]
    if len(rows) != run.get("records_written"):
        raise ValueError(f"Record count mismatch: {directory}")
    if {row["arm"] for row in rows} != expected_arms:
        raise ValueError(f"Missing arm rows: {directory}")
    for row in rows:
        case = case_by_id.get(row.get("case_id"))
        if case is None or row.get("prompt") != case["prompt"] or row.get("answer_key") != case["answer"]:
            raise ValueError(f"Unexpected question or trusted answer: {directory}")
        if row.get("family") != case["family"]:
            raise ValueError(f"Question family mismatch: {directory}")
        if row.get("metrics") != grade_case(case, row.get("response", "")):
            raise ValueError(f"Independent regrade mismatch: {directory}/{row.get('case_id')}")
        if row.get("resolved_revision") != PREREG["model"]["revision"]:
            raise ValueError(f"Per-row model revision mismatch: {directory}")
    for arm in expected_arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        if len(arm_rows) != len(case_by_id) or len({row["case_id"] for row in arm_rows}) != len(case_by_id):
            raise ValueError(f"Incomplete case coverage: {directory}/{arm}")
    ref = {
        "path": str(directory.relative_to(ROOT)),
        "run_id": run["run_id"],
        "response_records": run["records_written"],
        "model_generations": run["model_generations"],
        "cache_hits": run["cache_hits"],
        "observations_file": ledger_name,
        "observations_sha256": run["observations_sha256"],
        "report_sha256": run["report_sha256"],
        "spec_sha256": run["spec_sha256"],
    }
    return report, rows, ref


def arm_stats(rows: list[dict], arm: str, budget: int) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    correct = sum(bool(row["metrics"]["correct"]) for row in selected)
    exact_format = sum(bool(row["metrics"]["single_integer_format"]) for row in selected)
    tokens = [int(row.get("generated_tokens", 0)) for row in selected]
    return {
        "correct": correct,
        "questions": len(selected),
        "accuracy": correct / len(selected) if selected else None,
        "strict_integer_format": exact_format,
        "mean_generated_tokens": sum(tokens) / len(tokens) if tokens else None,
        "at_least_90_percent_of_budget": (
            sum(value >= 0.9 * budget for value in tokens) / len(tokens)
            if tokens and budget else None
        ),
    }


def replace_arm(rows: list[dict], old: str, new: str) -> list[dict]:
    return [{**row, "arm": new} for row in rows if row["arm"] == old]


def main() -> None:
    dataset = ROOT / PREREG["dataset"]["path"]
    _, cases = load_cases(
        dataset, "confirmation", family_limit=PREREG["dataset"]["family_limit"],
        family_seed=PREREG["dataset"]["family_seed"],
    )
    frozen = {
        row["case_id"]: row for row in cases
    }
    recorded_sample = {row["case_id"]: row for row in PREREG["dataset"]["selected_case_manifest"]}
    if set(frozen) != set(recorded_sample):
        raise ValueError("Selected case IDs differ from preregistration")
    for case_id, case in frozen.items():
        recorded = recorded_sample[case_id]
        prompt_hash = digest(normalized_question(case["prompt"]).encode())
        if recorded["family"] != case["family"] or recorded["prompt_sha256"] != prompt_hash or recorded[
            "answer_sha256"
        ] != digest(case["answer"].encode()):
            raise ValueError(f"Preregistered sample fingerprint mismatch: {case_id}")

    base_rows: dict[int, list[dict]] = {}
    refs = []
    reports = {}
    all_campaigns = []
    by_budget: dict[int, dict[str, list[dict]]] = {}
    for budget in PREREG["evaluation"]["budgets"]:
        baseline_dir = EVIDENCE / f"budget_{budget}" / "baseline"
        baseline_report, baseline, ref = verify_campaign(
            baseline_dir, {"base", "prompt"}, budget
        )
        refs.append(ref)
        by_budget[budget] = {arm: [row for row in baseline if row["arm"] == arm]
                             for arm in ("base", "prompt")}
        base_rows[budget] = baseline
        reports[f"{budget}/baseline"] = {
            "base": arm_stats(baseline, "base", budget),
            "prompt": arm_stats(baseline, "prompt", budget),
        }
        all_campaigns.append((budget, None, baseline_report, baseline))

        for condition in PREREG["fixed_conditions"]:
            name = condition["condition"]
            report, rows, ref = verify_campaign(
                EVIDENCE / f"budget_{budget}" / name,
                {"base", "prompt", "adapter", "random"}, budget, condition,
            )
            if sum(bool(row.get("cache_hit")) for row in rows) != 2 * len(cases):
                raise ValueError(f"Unexpected cached baseline row count: {budget}/{name}")
            for arm in ("base", "prompt"):
                arm_map = {row["case_id"]: row["response"] for row in rows if row["arm"] == arm}
                base_map = {row["case_id"]: row["response"] for row in baseline if row["arm"] == arm}
                if arm_map != base_map:
                    raise ValueError(f"Cached {arm} output differs: {budget}/{name}")
                by_budget[budget][f"{name}/{arm}"] = [
                    row for row in rows if row["arm"] == arm
                ]
            by_budget[budget][f"{name}/adapter"] = [
                row for row in rows if row["arm"] == "adapter"
            ]
            by_budget[budget][f"{name}/random"] = [
                row for row in rows if row["arm"] == "random"
            ]
            reports[f"{budget}/{name}"] = {
                "base": arm_stats(rows, "base", budget),
                "prompt": arm_stats(rows, "prompt", budget),
                "adapter": arm_stats(rows, "adapter", budget),
                "random": arm_stats(rows, "random", budget),
                "adapter_minus_base": paired_arm_contrast(rows, "base", "adapter"),
                "adapter_minus_prompt": paired_arm_contrast(rows, "prompt", "adapter"),
                "adapter_minus_random": paired_arm_contrast(rows, "random", "adapter"),
            }
            refs.append(ref)
            all_campaigns.append((budget, condition, report, rows))

    cross_budget = {}
    low, high = PREREG["evaluation"]["budgets"]
    for arm in ("base", "prompt"):
        lo = by_budget[low][arm]
        hi = by_budget[high][arm]
        rows = replace_arm(lo, arm, f"cap_{low}") + replace_arm(hi, arm, f"cap_{high}")
        cross_budget[f"{arm}_{low}_to_{high}"] = paired_arm_contrast(
            rows, f"cap_{low}", f"cap_{high}"
        )
    for condition in PREREG["fixed_conditions"]:
        name = condition["condition"]
        for arm in ("adapter", "random"):
            lo = by_budget[low][f"{name}/{arm}"]
            hi = by_budget[high][f"{name}/{arm}"]
            rows = replace_arm(lo, arm, f"cap_{low}") + replace_arm(hi, arm, f"cap_{high}")
            cross_budget[f"{name}_{arm}_{low}_to_{high}"] = paired_arm_contrast(
                rows, f"cap_{low}", f"cap_{high}"
            )

    summary = {
        "id": PREREG["id"],
        "status": "complete-no-promotion",
        "model": PREREG["model"],
        "dataset": {
            "unique_questions": len(cases),
            "families": len({case["family"] for case in cases}),
            "selected_case_fingerprint": PREREG["dataset"]["selected_case_fingerprint"],
            "confirmation_consumed": True,
        },
        "planned_budgets": PREREG["evaluation"]["budgets"],
        "model_generations": sum(ref["model_generations"] for ref in refs),
        "cache_hits": sum(ref["cache_hits"] for ref in refs),
        "response_records": sum(ref["response_records"] for ref in refs),
        "campaign_count": len(refs),
        "campaigns": refs,
        "per_budget_results": reports,
        "paired_budget_effects": cross_budget,
        "eligible_for_promotion": False,
        "interpretation_limit": "This is an arithmetic-only output-budget diagnostic on 24 synthetic addition families for one Qwen revision. A change in exact-key score can reflect answer completion and response format; it does not identify a causal weight mechanism or demonstrate general reasoning improvement.",
    }
    output = EVIDENCE / "summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "summary": str(output),
        "questions": len(cases),
        "families": len({case["family"] for case in cases}),
        "generations": summary["model_generations"],
        "cache_hits": summary["cache_hits"],
        "response_records": summary["response_records"],
    }, indent=2))


if __name__ == "__main__":
    main()
