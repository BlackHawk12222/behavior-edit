"""Deterministic fixed-support training cohorts for controlled exposure studies."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .case_eval import load_cases
from .provenance import fingerprint
from .experiment import write_json


def prepare_training_examples(dataset: Path) -> tuple[dict, list[dict]]:
    """Validate the dataset and return answer-keyed train rows with audit fields."""
    source, cases = load_cases(Path(dataset), "train")
    if not cases or any(row.get("split") != "train" for row in cases):
        raise ValueError("Training cohort requires non-empty rows from the train split only")
    seen = set()
    examples = []
    for row in cases:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in seen:
            raise ValueError("Training case IDs must be present and unique")
        seen.add(case_id)
        example = {
            "id": case_id,
            "prompt": row["prompt"],
            "answer": row["answer"],
            "family": row["family"],
            "task_type": row.get("task_type", "arithmetic"),
            "split": row["split"],
        }
        for field in ("subject_number", "attractor_number", "phrase_id"):
            if field in row:
                example[field] = row[field]
        examples.append(example)
    return source, examples


def _rank(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).hexdigest()


def parse_task_exposure_ratio(spec: str | None) -> dict[str, int] | None:
    """Parse an explicit comma-separated task=count exposure schedule."""
    if spec is None:
        return None
    ratio = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError("Exposure ratio must look like task_type=count,task_type=count")
        name, raw_count = (value.strip() for value in part.split("=", 1))
        if not name or name in ratio:
            raise ValueError("Exposure-ratio task names must be non-empty and unique")
        try:
            count = int(raw_count)
        except ValueError as error:
            raise ValueError("Exposure-ratio counts must be integers") from error
        if count < 1:
            raise ValueError("Exposure-ratio counts must be positive")
        ratio[name] = count
    if len(ratio) < 2:
        raise ValueError("Exposure ratio must specify at least two task types")
    return ratio


def select_fixed_cohort(examples: list[dict], phrase_count: int,
                        sentence_count: int, seed: int) -> list[dict]:
    """Pick examples once; all exposure-ratio arms reuse this exact support set.

    Phrase examples come from distinct phrase families. Sentence examples also
    come from distinct families and are balanced across singular/plural subjects.
    One sentence variant per family prevents its four prompt templates from
    masquerading as independent training support.
    """
    if phrase_count < 1 or sentence_count < 2 or sentence_count % 2:
        raise ValueError("Cohort requires positive phrase count and an even sentence count >= 2")
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in examples:
        if row["split"] != "train":
            raise ValueError("Fixed cohort may contain train-split examples only")
        if row["task_type"] not in {"phrase_completion", "sentence_completion"}:
            raise ValueError(f"Unsupported cohort task type: {row['task_type']}")
        grouped[row["task_type"]][row["family"]].append(row)

    phrase_families = grouped["phrase_completion"]
    if len(phrase_families) < phrase_count:
        raise ValueError(f"Need {phrase_count} distinct phrase families, found {len(phrase_families)}")
    selected = []
    for family in sorted(phrase_families, key=lambda f: _rank(seed, "phrase-family", f))[:phrase_count]:
        variants = phrase_families[family]
        selected.append(min(variants, key=lambda row: _rank(seed, "phrase-variant", row["id"])))

    sentence_families = grouped["sentence_completion"]
    if len(sentence_families) < sentence_count:
        raise ValueError(f"Need {sentence_count} distinct sentence families, found {len(sentence_families)}")
    chosen_families = sorted(
        sentence_families,
        key=lambda family: _rank(seed, "sentence-family", family),
    )[:sentence_count]
    per_number = sentence_count // 2
    singular_families = set(sorted(
        chosen_families,
        key=lambda family: _rank(seed, "sentence-subject-number", family),
    )[:per_number])
    for family in chosen_families:
        number = "singular" if family in singular_families else "plural"
        variants = [row for row in sentence_families[family]
                    if row.get("subject_number") == number]
        if not variants:
            raise ValueError(f"Sentence family {family} lacks a {number} variant")
        selected.append(min(
            variants,
            key=lambda row: _rank(seed, f"sentence-variant-{number}", row["id"]),
        ))

    selected.sort(key=lambda row: (row["task_type"], row["family"], row["id"]))
    if len({row["id"] for row in selected}) != len(selected):
        raise AssertionError("Cohort selection produced duplicate training IDs")
    return selected


def create_training_cohort(args) -> int:
    source, examples = prepare_training_examples(args.dataset)
    selected = select_fixed_cohort(examples, args.phrase_count,
                                   args.sentence_count, args.seed)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    data_path = destination / "cohort_examples.jsonl"
    data_bytes = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode()
    data_path.write_bytes(data_bytes)
    training_ids = [row["id"] for row in selected]
    task_counts = Counter(row["task_type"] for row in selected)
    families = {row["task_type"]: len({x["family"] for x in selected
                                        if x["task_type"] == row["task_type"]})
                for row in selected}
    subject_counts = Counter(row.get("subject_number", "not_applicable")
                             for row in selected if row["task_type"] == "sentence_completion")
    attractor_counts = Counter(row.get("attractor_number", "not_applicable")
                               for row in selected if row["task_type"] == "sentence_completion")
    manifest = {
        "schema_version": 1,
        "cohort_id": "completion-v3-fixed-support-exposure-v1",
        "source_dataset": str(Path(args.dataset)),
        "source_dataset_manifest_sha256": fingerprint(source["manifest"]),
        "source_split": "train",
        "selection_seed": args.seed,
        "selection_rule": "hash-ranked distinct families; one phrase variant per phrase family; one sentence variant per sentence family with balanced singular/plural subjects",
        "training_count": len(selected),
        "training_family_counts": families,
        "task_type_counts": dict(task_counts),
        "sentence_subject_number_counts": dict(subject_counts),
        "sentence_attractor_number_counts": dict(attractor_counts),
        "training_ids": training_ids,
        "training_data_sha256": fingerprint(selected),
        "cohort_examples_file": data_path.name,
        "cohort_examples_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "warning": "Fixed example support set for exploratory exposure-ratio comparisons; development questions are previously inspected and no confirmation claim is implied.",
    }
    write_json(destination / "manifest.json", manifest)
    print(f"Fixed training cohort saved: {destination}", flush=True)
    return 0
