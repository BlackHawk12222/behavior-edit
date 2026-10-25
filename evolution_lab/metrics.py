"""Dependency-free task validation and objective behavior metrics.

Scores represent observable task performance, not human personality or values.
"""

import json
import re
from collections import defaultdict

KINDS = frozenset({"exact", "json", "max_words", "contains", "bounded_contains"})


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
        if kind == "bounded_contains":
            required = task.get("required")
            if not isinstance(required, list) or not required or any(
                not isinstance(x, str) or not x.strip() for x in required
            ):
                raise ValueError(f"{identifier}: required must be nonempty strings")
        if kind in {"max_words", "bounded_contains"}:
            n = task.get("max_words")
            if type(n) is not int or n < 1:
                raise ValueError(f"{identifier}: max_words must be positive")
    return tasks


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def json_equal(a, b):
    # Python considers True == 1; JSON boolean and number types are distinct.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b or (
        type(a) in {int, float} and type(b) in {int, float} and a == b
    )


def grade(task, response):
    """Return objective pass/fail plus the raw metric where relevant."""
    answer = response.strip()
    kind = task["kind"]
    if kind == "exact":
        return {"passed": answer == task["target"]}
    if kind == "json":
        try:
            parsed = strict_json(answer)
        except (ValueError, TypeError):
            return {"passed": False, "valid_json": False}
        return {"passed": json_equal(parsed, task["target"]), "valid_json": True}
    if kind == "bounded_contains":
        n = len(answer.split())
        covered = all(re.search(r"(?<!\w)" + re.escape(word.casefold()) + r"(?!\w)",
                                answer.casefold()) for word in task["required"])
        return {"passed": bool(answer) and n <= task["max_words"] and covered,
                "word_count": n, "required_content": bool(covered)}
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



def compare_before_after(before_rows, after_rows):
    """Pair evaluations by task, reporting gains, regressions and locality.

    Only meaningful when the *same* task prompts, generation settings, and
    model checkpoint are used before and after an edit.
    """
    def index_unique(rows, phase):
        indexed = {}
        for row in rows:
            task_id = row["task_id"]
            if task_id in indexed:
                raise ValueError(f"Repeated task ID in {phase} evaluation: {task_id}")
            indexed[task_id] = row
        return indexed

    before = index_unique(before_rows, "before")
    after = index_unique(after_rows, "after")
    if not before or before.keys() != after.keys():
        raise ValueError("Before/after evaluations must contain the same nonempty task IDs")

    totals = {
        "task_count": len(before),
        "before_passed": 0,
        "after_passed": 0,
        "improved": 0,
        "regressed": 0,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
        "response_changed": 0,
    }
    per_category = defaultdict(lambda: {
        "tasks": 0,
        "before_passed": 0,
        "after_passed": 0,
        "improved": 0,
        "regressed": 0,
        "response_changed": 0,
    })
    task_rows = []
    for task_id in sorted(before):
        a, b = before[task_id], after[task_id]
        if a["task"] != b["task"]:
            raise ValueError(f"Task definition changed between phases: {task_id}")
        was_ok = bool(a["metrics"]["passed"])
        now_ok = bool(b["metrics"]["passed"])
        changed = a["response"].strip() != b["response"].strip()
        status = (
            "improved" if not was_ok and now_ok else
            "regressed" if was_ok and not now_ok else
            "unchanged_pass" if was_ok else "unchanged_fail"
        )
        category = a["task"].get("category", "uncategorized")
        totals["before_passed"] += int(was_ok)
        totals["after_passed"] += int(now_ok)
        totals[status] += 1
        totals["response_changed"] += int(changed)
        cats = per_category[category]
        cats["tasks"] += 1
        cats["before_passed"] += int(was_ok)
        cats["after_passed"] += int(now_ok)
        cats["response_changed"] += int(changed)
        if status in {"improved", "regressed"}:
            cats[status] += 1
        task_rows.append({
            "task_id": task_id,
            "category": category,
            "before_passed": was_ok,
            "after_passed": now_ok,
            "response_changed": changed,
            "status": status,
        })

    totals["pass_rate_delta"] = (
        totals["after_passed"] - totals["before_passed"]
    ) / totals["task_count"]
    totals["per_category"] = dict(sorted(per_category.items()))
    totals["per_task"] = task_rows
    return totals
