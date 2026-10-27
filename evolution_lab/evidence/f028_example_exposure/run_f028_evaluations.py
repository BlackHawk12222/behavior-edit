"""Evaluate the preregistered F028 sampling-policy candidates sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("evolution_lab/evidence/f028_example_exposure")
DATASET = Path("evolution_lab/evidence/completion_v3")
CACHE = Path("evolution_lab/evidence/f027_cohort_content_sensitivity/evaluations/cohortA_seed42_order901")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_campaign(path: Path, campaign: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    prereg_path = ROOT / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    registry = json.loads(Path("evolution_lab/experiment_registry.json").read_text())
    entry = next((row for row in registry["studies"]
                  if row.get("preregistration") == prereg_path.as_posix()), None)
    if not entry or digest(prereg_path) != entry.get("preregistration_sha256"):
        raise SystemExit("F028 preregistration/registry integrity check failed")
    if prereg.get("confirmation_consumed") or prereg.get("eligible_for_promotion"):
        raise SystemExit("F028 confirmation/promotion state violates preregistration")
    trained_path = ROOT / "training" / "training_campaign.json"
    trained = json.loads(trained_path.read_text())
    if (trained.get("status") != "complete" or len(trained.get("conditions", {})) != 6
            or trained.get("preregistration_sha256") != entry["preregistration_sha256"]):
        raise SystemExit("F028 training campaign is incomplete or has a different preregistration")

    output_root = ROOT / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    campaign_path = output_root / "evaluation_campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("preregistration_sha256") != entry["preregistration_sha256"]:
            raise SystemExit("Existing F028 evaluation campaign has a different preregistration")
    else:
        campaign = {"study_id": prereg["study_id"],
                    "preregistration_sha256": entry["preregistration_sha256"],
                    "status": "running", "conditions": {}}
        write_campaign(campaign_path, campaign)

    for key in sorted(trained["conditions"]):
        training = trained["conditions"][key]
        if training.get("status") != "complete":
            raise SystemExit(f"F028 training condition incomplete: {key}")
        run_dir = Path(training["run_dir"])
        manifest_path = run_dir / "manifest.json"
        if digest(manifest_path) != training["manifest_sha256"]:
            raise SystemExit(f"F028 training manifest changed: {key}")
        output = output_root / key
        if key in campaign["conditions"] and campaign["conditions"][key].get("status") == "complete":
            run = json.loads((output / "run.json").read_text())
            if (digest(output / "observations.jsonl") != campaign["conditions"][key]["observations_sha256"]
                    or run.get("status") != "complete"):
                raise SystemExit(f"Completed F028 evaluation integrity failed: {key}")
            continue
        output.mkdir(parents=True, exist_ok=True)
        command = [
            args.python, "-m", "evolution_lab", "case-eval", str(DATASET),
            "--split", "dev", "--model", args.model,
            "--revision", REVISION, "--device", "cpu", "--precision", "float32",
            "--max-new-tokens", "16", "--arms", "base", "prompt", "adapter", "random",
            "--adapter", str(run_dir / "adapter_lora"),
            "--random-adapter", str(run_dir / "adapter_random"),
            "--adapter-delta", str(run_dir / "delta_lora.json"),
            "--random-delta", str(run_dir / "delta_random.json"),
            "--baseline-cache-from", str(CACHE), "--output-dir", str(output),
        ]
        run_path = output / "run.json"
        if run_path.exists():
            command.append("--resume")
        campaign["conditions"][key] = {
            "status": "running", "command": command,
            "training_manifest_sha256": training["manifest_sha256"],
        }
        write_campaign(campaign_path, campaign)
        print(f"Evaluating {key}", flush=True)
        subprocess.run(command, check=True)
        run = json.loads(run_path.read_text())
        ledger, report = output / "observations.jsonl", output / "report.json"
        if (run.get("status") != "complete" or run.get("model_generations") != 856
                or run.get("cache_hits") != 856 or run.get("response_records") != 1712):
            raise RuntimeError(f"F028 evaluation accounting differs from preregistration: {key}")
        campaign["conditions"][key].update({
            "status": "complete", "run_spec_sha256": run["spec_sha256"],
            "observations_sha256": digest(ledger), "report_sha256": digest(report),
            "response_records": run["response_records"], "cache_hits": run["cache_hits"],
            "model_generations": run["model_generations"],
        })
        write_campaign(campaign_path, campaign)
        print(f"Completed {key}", flush=True)
    campaign["status"] = "complete"
    write_campaign(campaign_path, campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
