"""Run an offline-evaluable, reproducible behavior study.

Example: python -m evolution_lab run --limit 3 --device auto
"""

import argparse
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .metrics import grade, summarize, validate_tasks

TASK_FILE = Path(__file__).with_name("tasks.json")
CONDITIONS = {
    "baseline": "You are a helpful assistant.",
    "concise": "You are a concise assistant. Follow output constraints exactly.",
    "formal": "You are a formal assistant. Follow output constraints exactly.",
}


def read_tasks(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return validate_tasks(json.load(handle))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m evolution_lab",
        description="Independent local behavior lab; upstream research code is unchanged.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate", help="Validate task JSON without loading a model")
    check.add_argument("--tasks", type=Path, default=TASK_FILE)
    run = sub.add_parser("run", help="Run controlled evaluations with a local model")
    run.add_argument("--tasks", type=Path, default=TASK_FILE)
    run.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    run.add_argument("--revision", default="main", help="Use a commit SHA for strict reproducibility")
    run.add_argument("--adapter", type=Path, default=None, help="Optional PEFT adapter directory")
    run.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    run.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=["baseline", "concise"])
    run.add_argument("--limit", type=int, default=None, help="Restrict tasks for a quick smoke run")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--max-new-tokens", type=int, default=48)
    run.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab"))
    train = sub.add_parser("train", help="Train a tiny, optional LoRA behavior adapter")
    train.add_argument("--tasks", type=Path, default=TASK_FILE)
    train.add_argument("--examples", type=Path, default=Path(__file__).with_name("adapter_examples.json"))
    train.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    train.add_argument("--revision", default="main")
    train.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    train.add_argument("--precision", choices=["float32", "float16"], default="float32", help="Frozen base precision; adapter optimizer stays float32")
    train.add_argument("--steps", type=int, default=8)
    train.add_argument("--rank", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=0.0005)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--max-new-tokens", type=int, default=48)
    train.add_argument("--eval-limit", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab/adapters"))
    benchmark = sub.add_parser("benchmark", help="Validate and describe the versioned held-out suite")
    from .benchmark import SUITE_FILE, BEHAVIORS
    benchmark.add_argument("--suite", type=Path, default=SUITE_FILE)
    experiment = sub.add_parser("experiment", help="Run paired base/prompt/zero/LoRA/norm-matched random controls")
    experiment.add_argument("--suite", type=Path, default=SUITE_FILE)
    experiment.add_argument("--behavior", choices=BEHAVIORS, required=True)
    experiment.add_argument("--split", choices=["dev", "test"], default="dev")
    experiment.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    experiment.add_argument("--revision", default="main")
    experiment.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    experiment.add_argument("--precision", choices=["float32", "float16"], default="float32")
    experiment.add_argument("--steps", type=int, default=32)
    experiment.add_argument("--rank", type=int, default=4)
    experiment.add_argument("--learning-rate", type=float, default=.0005)
    experiment.add_argument("--max-length", type=int, default=192)
    experiment.add_argument("--max-new-tokens", type=int, default=48)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--limit-per-role", type=int, help="Smoke run: limit target and locality probes equally")
    experiment.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab/experiments"))
    frozen = sub.add_parser("test-checkpoint", help="Evaluate a frozen dev experiment on independent test families")
    frozen.add_argument("run_dir", type=Path)
    frozen.add_argument("--suite", type=Path, default=SUITE_FILE)
    frozen.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    frozen.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab/test"))
    evidence = sub.add_parser("verify-hypothesis", help="Audit raw dev/test records and check a narrow behavioral claim")
    evidence.add_argument("dev_dir", type=Path)
    evidence.add_argument("test_dir", type=Path)
    evidence.add_argument("--min-gain", type=float, default=.25)
    evidence.add_argument("--max-regressions", type=int, default=0)
    evidence.add_argument("--output", type=Path)
    study = sub.add_parser("study", help="Run or resume a fixed sequential seed/behavior matrix")
    study.add_argument("spec", type=Path)
    study.add_argument("--suite", type=Path, default=SUITE_FILE)
    study.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    study.add_argument("--output-dir", type=Path, required=True)
    catalog = sub.add_parser("catalog", help="Audit and export measured adapter features and dev/test outcomes")
    catalog.add_argument("study_dir", type=Path)
    catalog.add_argument("--output-dir", type=Path, required=True)
    mixture = sub.add_parser("mixture", help="Evaluate a fixed equal-weight mixture on development data only")
    mixture.add_argument("run_dirs", nargs=2, type=Path)
    mixture.add_argument("--suite", type=Path, default=SUITE_FILE)
    mixture.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    mixture.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab/mixtures"))
    fewshot = sub.add_parser("fewshot-control", help="Compare frozen results with all-training-demonstration prompting")
    fewshot.add_argument("run_dir", type=Path)
    fewshot.add_argument("--suite", type=Path, default=SUITE_FILE)
    fewshot.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    fewshot.add_argument("--output-dir", type=Path, default=Path("results/evolution_lab/fewshot"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "fewshot-control":
        from .fewshot import evaluate_fewshot
        return evaluate_fewshot(args)
    if args.command == "mixture":
        from .mixture import evaluate_mixture
        return evaluate_mixture(args)
    if args.command == "catalog":
        from .catalog import build_catalog, summarize_catalog
        from .experiment import write_json
        rows = build_catalog(args.study_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "adapter_behavior.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_json(args.output_dir / "summary.json", summarize_catalog(rows))
        print(f"Audited {len(rows)} adapter observations; saved {args.output_dir}")
        return 0
    if args.command == "study":
        from .study import execute_study
        return execute_study(args)
    if args.command == "verify-hypothesis":
        from .evidence import verify_hypothesis
        result = verify_hypothesis(args.dev_dir, args.test_dir, args.min_gain, args.max_regressions)
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        print(rendered)
        return 0
    if args.command == "benchmark":
        from .benchmark import load_suite, describe
        try:
            print(json.dumps(describe(load_suite(args.suite)), indent=2))
            return 0
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"Benchmark validation failed: {error}", file=sys.stderr)
            return 2
    if args.command == "test-checkpoint":
        from .checkpoint_eval import evaluate_checkpoint
        return evaluate_checkpoint(args)
    if args.command == "experiment":
        from .experiment import run_experiment
        return run_experiment(args)
    try:
        tasks = read_tasks(args.tasks)
    except (ValueError, OSError) as error:
        print(f"Task validation failed: {error}", file=sys.stderr)
        return 2

    if args.command == "validate":
        print(f"Valid: {len(tasks)} tasks from {args.tasks}")
        return 0
    if args.command == "train":
        from .train_adapter import train_adapter
        return train_adapter(args, tasks)

    if args.limit is not None and args.limit < 1:
        print("--limit must be positive", file=sys.stderr)
        return 2
    if args.repeats < 1 or args.max_new_tokens < 1:
        print("--repeats and --max-new-tokens must be positive", file=sys.stderr)
        return 2
    if args.limit is not None:
        tasks = tasks[:args.limit]

    import torch
    import transformers
    from .runner import LocalModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = LocalModel(
        args.model, revision=args.revision, device=args.device,
        adapter_path=args.adapter,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"{run_id}.jsonl"
    summary_path = args.output_dir / f"{run_id}_summary.json"
    records = []

    with result_path.open("w", encoding="utf-8") as handle:
        for condition in args.conditions:
            for repetition in range(args.repeats):
                for task in tasks:
                    response, tokens, elapsed = model.generate(
                        CONDITIONS[condition], task["prompt"], args.max_new_tokens
                    )
                    record = {
                        "run_id": run_id,
                        "lab_version": __version__,
                        "model_id": args.model,
                        "adapter": str(args.adapter) if args.adapter else None,
                        "requested_revision": args.revision,
                        "resolved_revision": model.resolved_revision,
                        "device": model.device,
                        "torch_version": torch.__version__,
                        "transformers_version": transformers.__version__,
                        "platform": platform.platform(),
                        "seed": args.seed,
                        "decoding": "greedy",
                        "max_new_tokens": args.max_new_tokens,
                        "condition": condition,
                        "system_prompt": CONDITIONS[condition],
                        "repetition": repetition,
                        "task_id": task["id"],
                        "task": task,
                        "response": response,
                        "generated_tokens": tokens,
                        "latency_seconds": round(elapsed, 4),
                        "metrics": grade(task, response),
                    }
                    records.append(record)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"{condition:8}  {task['id']:18} "
                        f"passed={record['metrics']['passed']}  "
                        f"time={elapsed:.2f}s"
                    )

    report = {
        "run_id": run_id,
        "model_id": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "resolved_revision": model.resolved_revision,
        "observations": len(records),
        "by_condition": summarize(records),
        "note": "These objective tasks measure compliance/accuracy, NOT personality.",
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_condition"], indent=2))
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
