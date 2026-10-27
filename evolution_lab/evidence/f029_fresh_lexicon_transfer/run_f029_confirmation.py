"""Run the frozen F029 confirmation campaigns sequentially and offline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
PREREG = json.loads((EVIDENCE / "preregistration.json").read_text())
MODEL = PREREG["model"]["id"]
REVISION = PREREG["model"]["revision"]


def run_campaign(dataset: Path, name: str, baseline: Path | None = None,
                 condition: dict | None = None) -> None:
    output = EVIDENCE / name
    command = [
        sys.executable, "-m", "evolution_lab", "case-eval", str(dataset),
        "--split", "confirmation", "--allow-confirmation", "--model", MODEL,
        "--revision", REVISION, "--device", "cpu", "--precision", "float32",
        "--max-new-tokens", "16", "--checkpoint-every", "32",
        "--output-dir", str(output),
    ]
    if condition is None:
        command.extend(["--arms", "base", "prompt"])
    else:
        source = Path(condition["source_manifest"]).parent
        command.extend([
            "--arms", "base", "prompt", "adapter", "random",
            "--adapter", str(source / "adapter_lora"),
            "--random-adapter", str(source / "adapter_random"),
            "--adapter-delta", str(source / "delta_lora.json"),
            "--random-delta", str(source / "delta_random.json"),
        ])
    if baseline is not None:
        command.extend(["--baseline-cache-from", str(baseline)])
    prior = output / "run.json"
    if prior.is_file():
        completed = json.loads(prior.read_text())
        if completed.get("status") == "complete":
            print(f"Campaign already complete: {name}", flush=True)
            return
        command.append("--resume")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    run = json.loads(prior.read_text())
    if run.get("status") != "complete":
        raise RuntimeError(f"F029 campaign did not complete: {name}: {run.get('status')}")


def main() -> None:
    completion = ROOT / PREREG["completion_dataset"]["path"]
    arithmetic = ROOT / PREREG["locality_dataset"]["path"]
    completion_baseline = EVIDENCE / "completion_baseline_offline"
    if not (completion_baseline / "run.json").is_file():
        run_campaign(completion, "completion_baseline_offline")
    for condition in PREREG["fixed_conditions"]:
        run_campaign(
            completion,
            f"completion_{condition['condition']}",
            baseline=completion_baseline,
            condition=condition,
        )

    arithmetic_baseline = EVIDENCE / "arithmetic_baseline"
    if not (arithmetic_baseline / "run.json").is_file():
        run_campaign(arithmetic, "arithmetic_baseline")
    for condition in PREREG["fixed_conditions"]:
        run_campaign(
            arithmetic,
            f"arithmetic_{condition['condition']}",
            baseline=arithmetic_baseline,
            condition=condition,
        )


if __name__ == "__main__":
    main()
