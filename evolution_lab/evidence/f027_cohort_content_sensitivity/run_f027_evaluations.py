"""Run the preregistered F027 paired adapter/random evaluation matrix sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("evolution_lab/evidence/f027_cohort_content_sensitivity")
DATASET = Path("evolution_lab/evidence/completion_v3")
CACHE = Path("evolution_lab/evidence/f025_fixed_support_exposure/evaluations/seed42_phrase16_sentence48")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_campaign(path: Path, campaign: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    prereg_path = ROOT / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    registry = json.loads(Path("evolution_lab/experiment_registry.json").read_text())
    registry_entry = next((row for row in registry["studies"]
                           if row.get("preregistration") == prereg_path.as_posix()), None)
    if not registry_entry or digest(prereg_path) != registry_entry.get("preregistration_sha256"):
        raise SystemExit("F027 preregistration/registry integrity check failed")
    if prereg.get("confirmation_consumed") or prereg.get("eligible_for_promotion"):
        raise SystemExit("F027 confirmation/promotion state violates preregistration")

    train_campaign_path = ROOT / "training" / "training_campaign.json"
    train_campaign = json.loads(train_campaign_path.read_text())
    if (train_campaign.get("status") != "complete"
            or train_campaign.get("preregistration_sha256") != registry_entry["preregistration_sha256"]
            or len(train_campaign.get("conditions", {})) != 12):
        raise SystemExit("F027 training matrix is incomplete or bound to a different preregistration")

    eval_root = ROOT / "evaluations"
    eval_root.mkdir(parents=True, exist_ok=True)
    campaign_path = eval_root / "evaluation_campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("preregistration_sha256") != registry_entry["preregistration_sha256"]:
            raise SystemExit("Existing evaluation campaign belongs to a different preregistration")
    else:
        campaign = {
            "study_id": prereg["study_id"],
            "preregistration_sha256": registry_entry["preregistration_sha256"],
            "status": "running",
            "conditions": {},
        }
        write_campaign(campaign_path, campaign)

    for key in sorted(train_campaign["conditions"]):
        trained = train_campaign["conditions"][key]
        if trained.get("status") != "complete":
            raise SystemExit(f"Training condition is not complete: {key}")
        run_dir = Path(trained["run_dir"])
        manifest_path = run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if digest(manifest_path) != trained["manifest_sha256"]:
            raise SystemExit(f"Training manifest changed since campaign: {key}")
        output_dir = eval_root / key
        if key in campaign["conditions"] and campaign["conditions"][key].get("status") == "complete":
            ledger = output_dir / "observations.jsonl"
            report = output_dir / "report.json"
            run = json.loads((output_dir / "run.json").read_text())
            old = campaign["conditions"][key]
            if (digest(ledger) != old.get("observations_sha256")
                    or digest(report) != old.get("report_sha256")
                    or run.get("status") != "complete"):
                raise SystemExit(f"Completed evaluation integrity failed: {key}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        adapter = run_dir / "adapter_lora"
        random_adapter = run_dir / "adapter_random"
        command = [
            args.python, "-m", "evolution_lab", "case-eval", str(DATASET),
            "--split", "dev",
            "--model", args.model,
            "--revision", REVISION,
            "--device", "cpu",
            "--precision", "float32",
            "--max-new-tokens", "16",
            "--arms", "base", "prompt", "adapter", "random",
            "--adapter", str(adapter),
            "--random-adapter", str(random_adapter),
            "--adapter-delta", str(run_dir / "delta_lora.json"),
            "--random-delta", str(run_dir / "delta_random.json"),
            "--baseline-cache-from", str(CACHE),
            "--output-dir", str(output_dir),
        ]
        run_path = output_dir / "run.json"
        if run_path.exists():
            command.append("--resume")
        campaign["conditions"][key] = {
            "status": "running",
            "command": command,
            "training_manifest_sha256": trained["manifest_sha256"],
        }
        write_campaign(campaign_path, campaign)
        print(f"Evaluating {key}", flush=True)
        subprocess.run(command, check=True)
        run = json.loads(run_path.read_text())
        ledger = output_dir / "observations.jsonl"
        report = output_dir / "report.json"
        if (run.get("status") != "complete"
                or run.get("model_generations") != 856
                or run.get("cache_hits") != 856
                or run.get("response_records") != 1712):
            raise RuntimeError(f"Evaluation run failed preregistered accounting: {key}")
        campaign["conditions"][key].update({
            "status": "complete",
            "run_spec_sha256": run["spec_sha256"],
            "observations_sha256": digest(ledger),
            "report_sha256": digest(report),
            "response_records": run["response_records"],
            "cache_hits": run["cache_hits"],
            "model_generations": run["model_generations"],
        })
        write_campaign(campaign_path, campaign)
        print(f"Completed {key}", flush=True)
    campaign["status"] = "complete"
    write_campaign(campaign_path, campaign)
    print(f"F027 evaluation matrix complete: {len(campaign['conditions'])} conditions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
