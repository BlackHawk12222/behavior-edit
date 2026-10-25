"""Deterministic arithmetic cases with executable keys and family-safe splits."""

from __future__ import annotations

import hashlib
import json
import random
import re
import math
from pathlib import Path

SCHEMA_VERSION = 3
TEMPLATES = (
    "What is {a} + {b}?",
    "Calculate the sum of {a} and {b}.",
    "Add {b} to {a}. Give only the number.",
    "A box has {a} items and another has {b}. How many altogether?",
)


def solve(case: dict) -> str:
    """Return the trusted answer from the structured operands, not model output."""
    a, b, op = case["operands"][0], case["operands"][1], case["operation"]
    if op != "add":
        raise ValueError(f"Unsupported operation: {op}")
    return str(a + b)


def _split(family: str) -> str:
    # Split at the family level so paraphrases/templates from one family never
    # straddle train, development, and confirmation data.
    value = int.from_bytes(hashlib.sha256(family.encode()).digest()[:4], "big") % 100
    return "train" if value < 70 else "dev" if value < 85 else "confirmation"


def generate(seed: int, count: int, start: int = 0) -> list[dict]:
    if count < 1 or start < 0:
        raise ValueError("count must be positive and start nonnegative")
    rows = []
    for index in range(start, start + count):
        template_id = index % len(TEMPLATES)
        pair_index = index // len(TEMPLATES)
        band, pair_index = divmod(pair_index, 10_000)
        # An affine permutation gives every ordered pair in a 100x100 band
        # exactly once before moving to the next band. It scales to millions
        # without birthday collisions while remaining seed-dependent.
        rng = random.Random(f"evolution-lab-case-v3:{seed}:{band}")
        multiplier = rng.randrange(1, 10_000)
        while math.gcd(multiplier, 10_000) != 1:
            multiplier = (multiplier + 1) % 10_000 or 1
        offset = rng.randrange(10_000)
        code = (multiplier * pair_index + offset) % 10_000
        low = band * 100
        a, b = low + code // 100, low + code % 100
        # Each consecutive block of four paraphrases shares operands and split.
        # This prevents identical arithmetic facts crossing splits while
        # creating enough independent fact families for useful proportions.
        family = f"addition-fact-band-{band}-pair-{pair_index}"
        prompt = TEMPLATES[template_id].format(a=a, b=b)
        row = {
            "case_id": f"add-{seed}-{index:09d}",
            "seed": seed,
            "index": index,
            "family": family,
            "split": _split(family),
            "operation": "add",
            "operands": [a, b],
            "prompt": prompt,
        }
        row["answer"] = solve(row)
        rows.append(row)
    return rows


def normalized_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def write_shard(output: Path, seed: int, start: int, count: int) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated = generate(seed, count, start)
    shard_rows = []
    for split in ("train", "dev", "confirmation"):
        rows = [row for row in generated if row["split"] == split]
        if not rows:
            continue
        shard = output / f"cases-{split}-{seed}-{start:09d}-{count:06d}.jsonl"
        if shard.exists():
            old = [json.loads(line) for line in shard.read_text().splitlines() if line]
            if old != rows:
                raise ValueError(f"Existing shard conflicts with deterministic output: {shard}")
            rows = old
        else:
            prompts = [normalized_question(row["prompt"]) for row in rows]
            if len(prompts) != len(set(prompts)):
                raise ValueError("Generated duplicate normalized questions")
            shard.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        digest = hashlib.sha256(shard.read_bytes()).hexdigest()
        shard_rows.append({"file": shard.name, "start": start, "count": len(rows), "split": split, "sha256": digest})
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": SCHEMA_VERSION, "generator": "arithmetic-v3", "seed": seed, "shards": []
    }
    if manifest["seed"] != seed:
        raise ValueError("One output directory must use one seed")
    for entry in shard_rows:
        existing = [item for item in manifest["shards"] if item["file"] == entry["file"]]
        if existing and existing[0] != entry:
            raise ValueError("Manifest entry conflicts with shard")
        if not existing:
            manifest["shards"].append(entry)
    manifest["shards"].sort(key=lambda item: (item["start"], item["split"]))
    manifest["unique_cases"] = sum(item["count"] for item in manifest["shards"])
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(manifest_path)
    return manifest


def validate_dataset(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    ids, prompts, seen_indexes = set(), set(), set()
    split_counts: dict[str, int] = {}
    rows_count = 0
    for entry in manifest["shards"]:
        shard = output / entry["file"]
        if hashlib.sha256(shard.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Shard hash mismatch: {shard}")
        rows = [json.loads(line) for line in shard.read_text().splitlines() if line]
        if len(rows) != entry["count"]:
            raise ValueError(f"Shard row count mismatch: {shard}")
        if any(row["split"] != entry["split"] for row in rows):
            raise ValueError(f"Split contamination in shard: {shard}")
        for row in rows:
            if row["case_id"] in ids or row["index"] in seen_indexes:
                raise ValueError("Duplicate case id or index across shards")
            ids.add(row["case_id"]); seen_indexes.add(row["index"])
            prompt = normalized_question(row["prompt"])
            if prompt in prompts or row["answer"] != solve(row):
                raise ValueError("Duplicate prompt or invalid trusted answer")
            prompts.add(prompt)
            if row["split"] != _split(row["family"]):
                raise ValueError("Unstable split assignment")
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            rows_count += 1
    if rows_count != manifest["unique_cases"]:
        raise ValueError("Manifest unique-case count does not match data")
    return {"unique_cases": rows_count, "unique_prompts": len(prompts), "split_counts": split_counts}
