"""Run preregistered F030 campaigns sequentially, reusing only verified baselines."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent
PREREG = json.loads((EVIDENCE / "preregistration.json").read_text())
DATASET = ROOT / PREREG["dataset"]["path"]
MODEL = PREREG["model"]["id"]
REVISION = PREREG["model"]["revision"]
FAMILY_LIMIT = PREREG["dataset"]["family_limit"]
FAMILY_SEED = PREREG["dataset"]["family_seed"]


def run_campaign(budget: int, name: str, baseline: Path | None = None,
                 condition: dict | None = None) -> None:
    output = EVIDENCE / f"budget_{budget}" / name
    prior = output / "run.json"
    if prior.is_file():
        previous = json.loads(prior.read_text())
        if previous.get("status") == "complete":
            print(f"Campaign already complete: {output.relative_to(ROOT)}", flush=True)
            return

    command = [
        sys.executable, "-m", "evolution_lab", "case-eval", str(DATASET),
        "--split", "confirmation", "--allow-confirmation",
        "--model", MODEL, "--revision", REVISION,
        "--device", "cpu", "--precision", "float32",
        "--family-limit", str(FAMILY_LIMIT), "--family-seed", str(FAMILY_SEED),
        "--max-new-tokens", str(budget), "--checkpoint-every", "32",
        "--output-dir", str(output),
    ]
    if condition is None:
        command.extend(["--arms", "base", "prompt"])
    else:
        source = ROOT / condition["source_manifest"]
        source_dir = source.parent
        command.extend([
            "--arms", "base", "prompt", "adapter", "random",
            "--adapter", str(source_dir / "adapter_lora"),
            "--random-adapter", str(source_dir / "adapter_random"),
            "--adapter-delta", str(source_dir / "delta_lora.json"),
            "--random-delta", str(source_dir / "delta_random.json"),
            "--baseline-cache-from", str(baseline),
        ])
    if prior.is_file():
        command.append("--resume")

    environment = dict(os.environ)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    result = json.loads(prior.read_text())
    if result.get("status") != "complete":
        raise RuntimeError(f"F030 campaign did not complete: {output}")


def main() -> None:
    if PREREG["status"] != "preregistered-not-run":
        raise RuntimeError("Refusing to run F030 after its preregistration status changed")
    for budget in PREREG["evaluation"]["budgets"]:
        baseline = EVIDENCE / f"budget_{budget}" / "baseline"
        run_campaign(budget, "baseline")
        for condition in PREREG["fixed_conditions"]:
            run_campaign(budget, condition["condition"], baseline, condition)


if __name__ == "__main__":
    main()
