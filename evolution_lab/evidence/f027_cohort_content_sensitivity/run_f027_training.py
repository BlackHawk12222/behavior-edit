"""Run the preregistered F027 cohort x training-seed x task-order matrix sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("evolution_lab/evidence/f027_cohort_content_sensitivity")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEEDS = (42, 43, 44)
ORDERS = (901, 903)
COHORTS = ("A", "B")
RATIO = {"phrase_completion": 16, "sentence_completion": 48}


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
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--precision", choices=("float32",), default="float32")
    args = parser.parse_args()

    prereg_path = ROOT / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    registry = json.loads(Path("evolution_lab/experiment_registry.json").read_text())
    registry_entry = next((row for row in registry["studies"]
                           if row.get("preregistration") == prereg_path.as_posix()), None)
    if prereg.get("status") != "preregistered-before-training" or registry_entry is None:
        raise SystemExit("F027 is not registered before training")
    prereg_hash = registry_entry.get("preregistration_sha256")
    if digest(prereg_path) != prereg_hash:
        raise SystemExit("F027 preregistration hash changed since registry entry")
    if prereg.get("confirmation_consumed") or prereg.get("eligible_for_promotion"):
        raise SystemExit("F027 confirmation/promotion state violates preregistration")

    training_root = ROOT / "training"
    training_root.mkdir(parents=True, exist_ok=True)
    campaign_path = training_root / "training_campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("preregistration_sha256") != prereg_hash:
            raise SystemExit("Existing training campaign belongs to different preregistration")
    else:
        campaign = {
            "study_id": prereg["study_id"],
            "preregistration_sha256": prereg_hash,
            "status": "running",
            "conditions": {},
        }
        write_campaign(campaign_path, campaign)

    for cohort in COHORTS:
        cohort_manifest = ROOT / f"cohort_{cohort}" / "manifest.json"
        cohort_info = prereg["data_intervention"]["cohorts"][cohort]
        if digest(cohort_manifest) != cohort_info["manifest_sha256"]:
            raise SystemExit(f"Cohort {cohort} manifest changed since preregistration")
        for seed in SEEDS:
            for order_seed in ORDERS:
                key = f"cohort{cohort}_seed{seed}_order{order_seed}"
                if key in campaign["conditions"]:
                    state = campaign["conditions"][key]
                    if state.get("status") == "complete":
                        continue
                    raise SystemExit(f"Condition {key} already exists with state {state.get('status')}; inspect before resuming")

                output_dir = training_root / key
                if output_dir.exists() and any(output_dir.iterdir()):
                    raise SystemExit(f"Unregistered output already exists for {key}: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
                command = [
                    args.python, "-m", "evolution_lab", "train-cases",
                    "evolution_lab/evidence/completion_v3",
                    "--model", args.model,
                    "--revision", REVISION,
                    "--device", args.device,
                    "--precision", args.precision,
                    "--steps", "64",
                    "--rank", "4",
                    "--learning-rate", "0.0005",
                    "--max-length", "256",
                    "--seed", str(seed),
                    "--sampling-strategy", "task-ratio",
                    "--cohort-manifest", str(cohort_manifest),
                    "--task-exposure-ratio", "phrase_completion=16,sentence_completion=48",
                    "--schedule-seed", str(order_seed),
                    "--output-dir", str(output_dir),
                ]
                campaign["conditions"][key] = {
                    "status": "running",
                    "command": command,
                    "cohort_manifest_sha256": digest(cohort_manifest),
                }
                write_campaign(campaign_path, campaign)
                print(f"Starting {key}", flush=True)
                subprocess.run(command, check=True)
                runs = sorted(p for p in output_dir.iterdir()
                              if p.is_dir() and (p / "manifest.json").is_file())
                if len(runs) != 1:
                    raise RuntimeError(f"Expected one training run for {key}, found {len(runs)}")
                run_dir = runs[0]
                manifest_path = run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                expected_counts = {"phrase_completion": 16, "sentence_completion": 48}
                if (manifest.get("status") != "complete"
                        or manifest.get("seed") != seed
                        or manifest.get("schedule_seed") != order_seed
                        or manifest.get("resolved_revision") != REVISION
                        or manifest.get("training_updates") != 64
                        or manifest.get("training_task_exposure_counts") != expected_counts
                        or manifest.get("fixed_training_cohort", {}).get("training_data_sha256")
                        != cohort_info["training_data_sha256"]):
                    raise RuntimeError(f"Training manifest failed expected-condition checks: {key}")
                campaign["conditions"][key].update({
                    "status": "complete",
                    "run_dir": run_dir.as_posix(),
                    "manifest_sha256": digest(manifest_path),
                    "training_trace_sha256": digest(run_dir / "training.json"),
                    "training_examples_seen_unique": manifest["training_examples_seen_unique"],
                })
                write_campaign(campaign_path, campaign)
                print(f"Completed {key}", flush=True)
    campaign["status"] = "complete"
    write_campaign(campaign_path, campaign)
    print(f"F027 training matrix complete: {len(campaign['conditions'])} conditions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
