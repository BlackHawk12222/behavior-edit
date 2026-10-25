"""Dependency-free task validation and objective behavior metrics.

Scores represent observable task performance, not human personality or values.
"""

import json
from collections import defaultdict

KINDS = frozenset({"exact", "json", "max_words", "contains"})


def validate_tasks(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task file must contain a non-empty JSON list")
    seen = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"Task #{index} must be an object")
        identifier = task.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"Task #{index} has no valid id")
        if identifier in seen:
            raise ValueError(f"Duplicate task id: {identifier}")
        seen.add(identifier)
        if not isinstance(task.get("prompt"), str) or not task["prompt"].strip():
            raise ValueError(f"{identifier}: prompt must be non-empty")
        kind = task.get("kind")
        if kind not in KINDS:
            raise ValueError(f"{identifier}: unknown kind {kind!r}")
        if kind in {"exact", "json", "contains"} and "target" not in task:
            raise ValueError(f"{identifier}: missing target")
        if kind in {"exact", "contains"} and not isinstance(task["target"], str):
            raise ValueError(f"{identifier}: target must be a string")
        if kind == "max_words":
            n = task.get("max_words")
            if type(n) is not int or n < 1:
                raise ValueError(f"{identifier}: max_words must be positive")
    return tasks


def grade(task, response):
    """Return objective pass/fail plus the raw metric where relevant."""
    answer = response.strip()
    kind = task["kind"]
    if kind == "exact":
        return {"passed": answer == task["target"]}
    if kind == "json":
        try:
            parsed = json.loads(answer)
        except (ValueError, TypeError):
            return {"passed": False, "valid_json": False}
        return {"passed": parsed == task["target"], "valid_json": True}
    if kind == "max_words":
        n = len(answer.split())
        return {"passed": bool(answer) and n <= task["max_words"], "word_count": n}
    if kind == "contains":
        return {"passed": task["target"].casefold() in answer.casefold()}
    raise ValueError(f"Unknown task kind: {kind}")


def summarize(records):
    """Summarize results by condition; repetitions remain separate observations."""
    buckets = defaultdict(list)
    for record in records:
        buckets[record["condition"]].append(record)
    report = {}
    for condition, rows in sorted(buckets.items()):
        n = len(rows)
        passed = sum(bool(r["metrics"]["passed"]) for r in rows)
        report[condition] = {
            "observations": n,
            "passed": passed,
            "pass_rate": passed / n if n else None,
            "mean_latency_seconds": (
                sum(r["latency_seconds"] for r in rows) / n if n else None
            ),
        }
    return report
