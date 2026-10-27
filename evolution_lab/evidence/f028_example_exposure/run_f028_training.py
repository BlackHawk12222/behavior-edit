"""Run the preregistered F028 example-exposure matrix sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("evolution_lab/evidence/f028_example_exposure")
DATASET = Path("evolution_lab/evidence/completion_v3")
COHORT = Path("evolution_lab/evidence/f027_cohort_content_sensitivity/cohort_A/manifest.json")
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
SEEDS = (42, 43, 44)
POLICIES = ("with-replacement", "without-replacement")
ORDER = 2603
RATIO = "phrase_completion=16,sentence_completion=48"


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
        raise SystemExit("F028 preregistration does not match the registry")
    if prereg.get("confirmation_consumed") or prereg.get("eligible_for_promotion"):
        raise SystemExit("F028 confirmation/promotion state violates preregistration")
    if digest(COHORT) != prereg["design"]["fixed_cohort_manifest_sha256"]:
        raise SystemExit("F028 fixed cohort changed since preregistration")

    root = ROOT / "training"
    root.mkdir(parents=True, exist_ok=True)
    campaign_path = root / "training_campaign.json"
    if campaign_path.exists():
        campaign = json.loads(campaign_path.read_text())
        if campaign.get("preregistration_sha256") != entry["preregistration_sha256"]:
            raise SystemExit("Existing F028 training campaign has a different preregistration")
    else:
        campaign = {"study_id": prereg["study_id"],
                    "preregistration_sha256": entry["preregistration_sha256"],
                    "status": "running", "conditions": {}}
        write_campaign(campaign_path, campaign)

    for seed in SEEDS:
        for policy in POLICIES:
            key = f"{policy}_seed{seed}"
            if key in campaign["conditions"]:
                state = campaign["conditions"][key]
                if state.get("status") == "complete":
                    continue
                raise SystemExit(f"Inspect incomplete F028 condition before resuming: {key}")
            output = root / key
            if output.exists() and any(output.iterdir()):
                raise SystemExit(f"Unregistered F028 output already exists: {output}")
            output.mkdir(parents=True, exist_ok=True)
            command = [
                args.python, "-m", "evolution_lab", "train-cases", str(DATASET),
                "--model", args.model, "--revision", REVISION,
                "--device", "cpu", "--precision", "float32",
                "--steps", "64", "--rank", "4", "--learning-rate", "0.0005",
                "--max-length", "256", "--seed", str(seed),
                "--sampling-strategy", "task-ratio", "--cohort-manifest", str(COHORT),
                "--task-exposure-ratio", RATIO, "--schedule-seed", str(ORDER),
                "--example-sampling", policy, "--output-dir", str(output),
            ]
            campaign["conditions"][key] = {
                "status": "running", "command": command,
                "cohort_manifest_sha256": digest(COHORT),
                "example_sampling": policy, "seed": seed, "schedule_seed": ORDER,
            }
            write_campaign(campaign_path, campaign)
            print(f"Training {key}", flush=True)
            subprocess.run(command, check=True)
            runs = sorted(path for path in output.iterdir()
                          if path.is_dir() and (path / "manifest.json").is_file())
            if len(runs) != 1:
                raise RuntimeError(f"Expected one training run for {key}, found {len(runs)}")
            run = runs[0]
            manifest_path = run / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            expected_ratio = {"phrase_completion": 16, "sentence_completion": 48}
            if (manifest.get("status") != "complete" or manifest.get("seed") != seed
                    or manifest.get("schedule_seed") != ORDER
                    or manifest.get("resolved_revision") != REVISION
                    or manifest.get("training_updates") != 64
                    or manifest.get("training_task_exposure_counts") != expected_ratio
                    or manifest.get("settings", {}).get("example_sampling") != policy
                    or manifest.get("fixed_training_cohort", {}).get("training_data_sha256")
                    != prereg["design"]["fixed_cohort_training_data_sha256"]):
                raise RuntimeError(f"F028 training manifest failed conditions: {key}")
            if policy == "without-replacement" and manifest["training_examples_seen_unique"] != 64:
                raise RuntimeError(f"No-replacement policy missed fixed cohort examples: {key}")
            campaign["conditions"][key].update({
                "status": "complete", "run_dir": run.as_posix(),
                "manifest_sha256": digest(manifest_path),
                "training_trace_sha256": digest(run / "training.json"),
                "training_examples_seen_unique": manifest["training_examples_seen_unique"],
                "unique_examples_by_task": {
                    task: len({row["example_id"] for row in json.loads((run / "training.json").read_text())
                               if row["task_type"] == task})
                    for task in expected_ratio
                },
            })
            write_campaign(campaign_path, campaign)
            print(f"Completed {key}", flush=True)
    campaign["status"] = "complete"
    write_campaign(campaign_path, campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
