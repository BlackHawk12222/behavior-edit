"""Deterministic, objectively graded phrase and sentence completions."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .casegen import normalized_question

SCHEMA_VERSION = 1
GENERATOR = "phrase-sentence-completion-v1"

# Curated common collocations. These are English-language development probes,
# not a complete or culturally neutral inventory of phrases.
PHRASES = {
    "salt_pepper": ("salt and", "pepper", "sugar"),
    "bread_butter": ("bread and", "butter", "stone"),
    "black_white": ("black and", "white", "purple"),
    "peanut_jelly": ("peanut butter and", "jelly", "mustard"),
    "trial_error": ("trial and", "error", "comfort"),
    "rock_roll": ("rock and", "roll", "paper"),
    "fish_chips": ("fish and", "chips", "candy"),
    "pros_cons": ("pros and", "cons", "reasons"),
    "supply_demand": ("supply and", "demand", "permission"),
    "safe_sound": ("safe and", "sound", "silent"),
    "odds_ends": ("odds and", "ends", "beginnings"),
    "sooner_later": ("sooner or", "later", "earlier"),
    "hide_seek": ("hide and", "seek", "find"),
    "lock_key": ("lock and", "key", "window"),
    "rain_shine": ("rain or", "shine", "cloud"),
    "ebb_flow": ("ebb and", "flow", "freeze"),
    "short_sweet": ("short and", "sweet", "bitter"),
    "peace_quiet": ("peace and", "quiet", "noise"),
    "prim_proper": ("prim and", "proper", "casual"),
    "thick_thin": ("through thick and", "thin", "wide"),
    "give_take": ("give and", "take", "lend"),
    "make_break": ("make or", "break", "repair"),
    "cloak_dagger": ("cloak and", "dagger", "helmet"),
    "wait_see": ("wait and", "see", "look"),
    "wear_tear": ("wear and", "tear", "repair"),
    "sick_tired": ("sick and", "tired", "sleepy"),
    "now_then": ("now and", "then", "soon"),
    "cause_effect": ("cause and", "effect", "event"),
    "terms_conditions": ("terms and", "conditions", "rules"),
    "lost_found": ("lost and", "found", "missing"),
    "here_there": ("here and", "there", "away"),
    "back_forth": ("back and", "forth", "forward"),
    "ups_downs": ("ups and", "downs", "levels"),
    "all_nothing": ("all or", "nothing", "something"),
    "high_low": ("high and", "low", "middle"),
    "fair_square": ("fair and", "square", "circle"),
    "now_never": ("now or", "never", "later"),
    "life_death": ("life and", "death", "living"),
    "heart_soul": ("heart and", "soul", "mind"),
    "first_foremost": ("first and", "foremost", "second"),
}

SUBJECT_PAIRS = (
    ("dog", "dogs"),
    ("student", "students"),
    ("child", "children"),
    ("teacher", "teachers"),
    ("fox", "foxes"),
    ("bird", "birds"),
    ("chef", "chefs"),
    ("neighbor", "neighbors"),
)

# Explicit forms avoid applying an English spelling heuristic as ground truth.
VERB_PAIRS = (
    ("walk", "walks"),
    ("run", "runs"),
    ("read", "reads"),
    ("cook", "cooks"),
    ("play", "plays"),
    ("watch", "watches"),
    ("carry", "carries"),
    ("study", "studies"),
)

CONTEXTS = (
    "near the garden every morning",
    "after lunch",
    "in the library",
    "before school",
    "on weekends",
    "in the yard",
    "after dinner",
    "every afternoon",
)

PHRASE_TEMPLATES = (
    'Complete the familiar phrase: "{prefix} ___." Choose {choices}.',
    'Fill in the missing word in this phrase: "{prefix} ___." Options: {choices}.',
    'What word completes "{prefix} ___"? Pick one: {choices}.',
    'Finish this common phrase with one choice: "{prefix} ___." Choices are {choices}.',
)

SENTENCE_TEMPLATES = (
    'Complete the sentence: "{subject} ___ {context}." Choose {choices}.',
    'Select the grammatically correct word for this blank: "{subject} ___ {context}." Options: {choices}.',
    'Fill in the sentence with one choice: "{subject} ___ {context}." Pick {choices}.',
    'Which verb correctly completes this sentence? "{subject} ___ {context}." Choices are {choices}.',
)


def _choice_text(options: list[str], seed: int, family: str, variant: int) -> str:
    ordered = list(options)
    random.Random(f"completion-v1:{seed}:{family}:{variant}").shuffle(ordered)
    return " or ".join(ordered)


def _family_splits(seed: int) -> dict[str, str]:
    """Stratify whole phrase and grammar families into stable 70/15/15 pools."""
    groups = {
        "phrase_completion": [f"phrase-collocation-{phrase_id}" for phrase_id in PHRASES],
        "sentence_completion": [
            f"sentence-agreement-{singular}-{verb}"
            for singular, _ in SUBJECT_PAIRS
            for verb, _ in VERB_PAIRS
        ],
    }
    assignments = {}
    for task_type, families in groups.items():
        ordered = sorted(
            families,
            key=lambda family: hashlib.sha256(f"completion-v1:{seed}:{task_type}:{family}".encode()).hexdigest(),
        )
        n_train = math.floor(len(ordered) * 0.70)
        n_dev = round(len(ordered) * 0.15)
        for index, family in enumerate(ordered):
            split = "train" if index < n_train else "dev" if index < n_train + n_dev else "confirmation"
            assignments[family] = split
    return assignments


def generate(seed: int, count: int, start: int = 0) -> list[dict]:
    """Generate a stable mixed set; all paraphrases of a fact share one split."""
    if count < 1 or start < 0:
        raise ValueError("count must be positive and start nonnegative")

    all_rows: list[dict] = []
    family_splits = _family_splits(seed)
    for phrase_id, (prefix, answer, distractor) in PHRASES.items():
        family = f"phrase-collocation-{phrase_id}"
        for variant, template in enumerate(PHRASE_TEMPLATES):
            options = [answer, distractor]
            prompt = template.format(
                prefix=prefix,
                choices=_choice_text(options, seed, family, variant),
            )
            all_rows.append({
                "family": family,
                "split": family_splits[family],
                "task_type": "phrase_completion",
                "phrase_id": phrase_id,
                "prefix": prefix,
                "options": options,
                "answer": answer,
                "prompt": prompt,
            })

    for singular, plural in SUBJECT_PAIRS:
        for base_verb, singular_verb in VERB_PAIRS:
            family = f"sentence-agreement-{singular}-{base_verb}"
            for number, subject in (("singular", f"The {singular}"), ("plural", f"The {plural}")):
                answer = singular_verb if number == "singular" else base_verb
                distractor = base_verb if number == "singular" else singular_verb
                options = [answer, distractor]
                for variant, template in enumerate(SENTENCE_TEMPLATES):
                    context = CONTEXTS[(variant + sum(map(ord, family))) % len(CONTEXTS)]
                    prompt = template.format(
                        subject=subject,
                        context=context,
                        choices=_choice_text(options, seed, family + number, variant),
                    )
                    all_rows.append({
                        "family": family,
                        "split": family_splits[family],
                        "task_type": "sentence_completion",
                        "subject_number": number,
                        "subject": subject,
                        "base_verb": base_verb,
                        "singular_verb": singular_verb,
                        "options": options,
                        "answer": answer,
                        "prompt": prompt,
                    })

    for index, partial in enumerate(all_rows):
        partial.update({
            "case_id": f"completion-{seed}-{index:06d}",
            "seed": seed,
            "index": index,
        })
    all_rows.sort(key=lambda row: row["index"])
    end = start + count
    if start >= len(all_rows):
        raise ValueError(f"start {start} is beyond this generator's {len(all_rows)} cases")
    if end > len(all_rows):
        raise ValueError(f"requested range ends at {end}, beyond this generator's {len(all_rows)} cases")
    return all_rows[start:end]


def solve(case: dict) -> str:
    """Recompute the accepted completion from curated phrase/grammar facts."""
    kind = case["task_type"]
    if kind == "phrase_completion":
        try:
            return PHRASES[case["phrase_id"]][1]
        except KeyError as error:
            raise ValueError(f"Unknown phrase family: {case.get('phrase_id')}") from error
    if kind == "sentence_completion":
        number = case["subject_number"]
        if number == "singular":
            return case["singular_verb"]
        if number == "plural":
            return case["base_verb"]
        raise ValueError(f"Unsupported grammatical number: {number}")
    raise ValueError(f"Unsupported completion task type: {kind}")


def write_shard(output: Path, seed: int, start: int, count: int) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated = generate(seed, count, start)
    entries = []
    for split in ("train", "dev", "confirmation"):
        rows = [row for row in generated if row["split"] == split]
        if not rows:
            continue
        shard = output / f"completion-cases-{split}-{seed}-{start:06d}-{count:06d}.jsonl"
        if shard.exists():
            existing = [json.loads(line) for line in shard.read_text().splitlines() if line]
            if existing != rows:
                raise ValueError(f"Existing completion shard conflicts with deterministic output: {shard}")
        else:
            shard.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))
        entries.append({
            "file": shard.name,
            "start": start,
            "count": len(rows),
            "split": split,
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        })
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("seed") != seed or manifest.get("generator") != GENERATOR:
            raise ValueError("Completion output directory belongs to a different seed or generator")
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR, "seed": seed, "shards": []}
    for entry in entries:
        existing = [item for item in manifest["shards"] if item["file"] == entry["file"]]
        if existing and existing[0] != entry:
            raise ValueError("Completion manifest entry conflicts with shard")
        if not existing:
            manifest["shards"].append(entry)
    manifest["shards"].sort(key=lambda item: (item["start"], item["split"]))
    manifest["unique_cases"] = sum(item["count"] for item in manifest["shards"])
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def validate_dataset(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("generator") != GENERATOR:
        raise ValueError("Dataset is not a phrase-sentence-completion-v1 dataset")
    case_ids, prompts, indexes = set(), set(), set()
    split_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    families_by_split: dict[str, str] = {}
    total = 0
    expected_cases = {
        row["index"]: row
        for row in generate(manifest["seed"], len(PHRASES) * len(PHRASE_TEMPLATES)
                            + len(SUBJECT_PAIRS) * len(VERB_PAIRS) * 2 * len(SENTENCE_TEMPLATES))
    }
    expected_family_splits = _family_splits(manifest["seed"])
    for shard_info in manifest["shards"]:
        shard = output / shard_info["file"]
        if hashlib.sha256(shard.read_bytes()).hexdigest() != shard_info["sha256"]:
            raise ValueError(f"Completion shard hash mismatch: {shard}")
        rows = [json.loads(line) for line in shard.read_text().splitlines() if line]
        if len(rows) != shard_info["count"] or any(row["split"] != shard_info["split"] for row in rows):
            raise ValueError(f"Completion shard count or split mismatch: {shard}")
        for row in rows:
            if row.get("index") not in expected_cases or row != expected_cases[row["index"]]:
                raise ValueError("Completion case does not match the deterministic generator record")
            normalized = normalized_question(row["prompt"])
            if row["case_id"] in case_ids or row["index"] in indexes or normalized in prompts:
                raise ValueError("Duplicate completion ID, index, or normalized prompt")
            if row["answer"] != solve(row) or row["answer"] not in row["options"]:
                raise ValueError("Completion answer key does not match trusted phrase or grammar data")
            if len(row["options"]) != 2 or len(set(row["options"])) != 2:
                raise ValueError("Each v1 completion question needs two distinct answer choices")
            expected_split = expected_family_splits.get(row["family"])
            if row["split"] != expected_split:
                raise ValueError("Completion family split is not deterministic")
            previous = families_by_split.setdefault(row["family"], row["split"])
            if previous != row["split"]:
                raise ValueError("Completion family leaked across splits")
            case_ids.add(row["case_id"])
            indexes.add(row["index"])
            prompts.add(normalized)
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
            total += 1
    if total != manifest.get("unique_cases"):
        raise ValueError("Completion manifest unique-case count does not match the shards")
    return {
        "unique_cases": total,
        "unique_prompts": len(prompts),
        "split_counts": split_counts,
        "task_type_counts": task_counts,
        "unique_families": len(families_by_split),
    }
