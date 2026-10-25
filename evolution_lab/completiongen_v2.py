"""New-vocabulary completion families for independent transfer checks.

Version 2 deliberately uses phrase and subject/verb inventories absent from v1.
The grammatical rule is the same subject/verb agreement rule, allowing a
frozen v1-trained adapter to be checked on new lexical families.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .casegen import normalized_question

SCHEMA_VERSION = 1
GENERATOR = "phrase-sentence-completion-v2-new-vocabulary"

PHRASES = {
    "needle_haystack": ("needle in a", "haystack", "drawer"),
    "break_ice": ("break the", "ice", "glass"),
    "bite_bullet": ("bite the", "bullet", "cookie"),
    "spill_beans": ("spill the", "beans", "rice"),
    "curiosity_cat": ("curiosity killed the", "cat", "rabbit"),
    "early_bird": ("the early bird catches the", "worm", "fly"),
    "actions_words": ("actions speak louder than", "words", "sounds"),
    "cat_bag": ("let the cat out of the", "bag", "basket"),
    "ball_court": ("the ball is in your", "court", "field"),
    "apple_tree": ("the apple doesn't fall far from the", "tree", "fence"),
    "cloud_lining": ("every cloud has a silver", "lining", "border"),
    "honesty_policy": ("honesty is the best", "policy", "choice"),
    "midnight_oil": ("burn the midnight", "oil", "wax"),
    "tip_tongue": ("on the tip of my", "tongue", "tooth"),
    "sleeping_dogs": ("let sleeping dogs", "lie", "sit"),
    "glitters_gold": ("all that glitters is not", "gold", "silver"),
    "beggars_choosers": ("beggars can't be", "choosers", "winners"),
    "chickens_hatch": ("don't count your chickens before they", "hatch", "roost"),
    "horse_drink": ("you can lead a horse to water but you can't make it", "drink", "swim"),
    "pen_sword": ("the pen is mightier than the", "sword", "shield"),
    "necessity_invention": ("necessity is the mother of", "invention", "discovery"),
    "steady_race": ("slow and steady wins the", "race", "prize"),
    "wrongs_right": ("two wrongs don't make a", "right", "pair"),
    "rome_romans": ("when in Rome do as the", "Romans", "tourists"),
    "heart_fonder": ("absence makes the heart grow", "fonder", "stronger"),
    "bite_chew": ("don't bite off more than you can", "chew", "carry"),
    "smoke_fire": ("there's no smoke without", "fire", "flame"),
    "book_cover": ("you can't judge a book by its", "cover", "pages"),
    "spilled_milk": ("don't cry over spilled", "milk", "juice"),
    "late_never": ("better late than", "never", "soon"),
    "feather_together": ("birds of a feather flock", "together", "south"),
    "smile_miles": ("wear a smile and have friends; wear a frown and have", "wrinkles", "dimples"),
    "day_sun": ("make hay while the sun", "shines", "sets"),
    "golden_rule": ("treat others as you would like to be", "treated", "noticed"),
    "hands_work": ("many hands make light", "work", "noise"),
    "mountain_molehill": ("make a mountain out of a", "molehill", "mountain"),
    "turn_another": ("one good turn deserves", "another", "more"),
    "water_bridge": ("water under the", "bridge", "road"),
    "fact_stranger": ("truth is stranger than", "fiction", "stories"),
    "morning_night": ("red sky at night, sailor's delight; red sky in the", "morning", "evening"),
}

SUBJECT_PAIRS = (
    ("pilot", "pilots"), ("canoe", "canoes"),
    ("engine", "engines"), ("robot", "robots"),
    ("painter", "painters"), ("musician", "musicians"),
    ("rabbit", "rabbits"), ("turtle", "turtles"),
    ("doctor", "doctors"), ("farmer", "farmers"),
    ("sailor", "sailors"), ("inventor", "inventors"),
    ("puppy", "puppies"), ("owl", "owls"),
    ("mechanic", "mechanics"), ("dancer", "dancers"),
)

VERB_PAIRS = (
    ("glide", "glides"), ("paint", "paints"),
    ("repair", "repairs"), ("whistle", "whistles"),
    ("plant", "plants"), ("invent", "invents"),
    ("bounce", "bounces"), ("whisper", "whispers"),
    ("sail", "sails"), ("measure", "measures"),
    ("juggle", "juggles"), ("deliver", "delivers"),
    ("wander", "wanders"), ("polish", "polishes"),
    ("sketch", "sketches"), ("explore", "explores"),
)

CONTEXTS = (
    "across the quiet harbor at dawn",
    "beside the old stone wall",
    "through the market after sunrise",
    "near the open workshop door",
    "along the narrow forest path",
    "under the bright evening stars",
    "around the garden before lunch",
    "past the small wooden bridge",
)

PHRASE_TEMPLATES = (
    'Complete this familiar phrase: "{prefix} ___." Choose {choices}.',
    'Fill the blank in this phrase: "{prefix} ___." Options: {choices}.',
    'Which word completes "{prefix} ___"? Pick one: {choices}.',
    'Finish the phrase "{prefix} ___." Choices are {choices}.',
)

SENTENCE_TEMPLATES = (
    'Complete the sentence: "{subject} ___ {context}." Choose {choices}.',
    'Select the correct verb: "{subject} ___ {context}." Options: {choices}.',
    'Fill the blank in this sentence: "{subject} ___ {context}." Pick {choices}.',
    'Which verb fits the sentence? "{subject} ___ {context}." Choices are {choices}.',
)


def _choice_text(options: list[str], seed: int, family: str, variant: int) -> str:
    ordered = list(options)
    random.Random(f"completion-v2:{seed}:{family}:{variant}").shuffle(ordered)
    return " or ".join(ordered)


def _family_splits(seed: int) -> dict[str, str]:
    groups = {
        "phrase_completion": [f"phrase-v2-{phrase_id}" for phrase_id in PHRASES],
        "sentence_completion": [
            f"sentence-v2-{subject}-{verb}"
            for subject, _ in SUBJECT_PAIRS
            for verb, _ in VERB_PAIRS
        ],
    }
    assignments = {}
    for kind, families in groups.items():
        ordered = sorted(
            families,
            key=lambda family: hashlib.sha256(f"{GENERATOR}:{seed}:{kind}:{family}".encode()).hexdigest(),
        )
        train_count = math.floor(len(ordered) * 0.70)
        dev_count = round(len(ordered) * 0.15)
        for index, family in enumerate(ordered):
            split = "train" if index < train_count else "dev" if index < train_count + dev_count else "confirmation"
            assignments[family] = split
    return assignments


def generate(seed: int, count: int, start: int = 0) -> list[dict]:
    if count < 1 or start < 0:
        raise ValueError("count must be positive and start nonnegative")
    splits = _family_splits(seed)
    rows = []
    for phrase_id, (prefix, answer, distractor) in PHRASES.items():
        family = f"phrase-v2-{phrase_id}"
        for variant, template in enumerate(PHRASE_TEMPLATES):
            options = [answer, distractor]
            rows.append({
                "family": family,
                "split": splits[family],
                "task_type": "phrase_completion",
                "phrase_id": phrase_id,
                "prefix": prefix,
                "options": options,
                "answer": answer,
                "prompt": template.format(prefix=prefix, choices=_choice_text(options, seed, family, variant)),
            })
    for singular, plural in SUBJECT_PAIRS:
        for base_verb, singular_verb in VERB_PAIRS:
            family = f"sentence-v2-{singular}-{base_verb}"
            for number, subject in (("singular", f"The {singular}"), ("plural", f"The {plural}")):
                answer = singular_verb if number == "singular" else base_verb
                distractor = base_verb if number == "singular" else singular_verb
                options = [answer, distractor]
                for variant, template in enumerate(SENTENCE_TEMPLATES):
                    context_index = (variant + sum(map(ord, family + number))) % len(CONTEXTS)
                    rows.append({
                        "family": family,
                        "split": splits[family],
                        "task_type": "sentence_completion",
                        "subject_number": number,
                        "subject": subject,
                        "base_verb": base_verb,
                        "singular_verb": singular_verb,
                        "options": options,
                        "answer": answer,
                        "prompt": template.format(
                            subject=subject,
                            context=CONTEXTS[context_index],
                            choices=_choice_text(options, seed, family + number, variant),
                        ),
                    })
    for index, row in enumerate(rows):
        row.update({"case_id": f"completion-v2-{seed}-{index:07d}", "seed": seed, "index": index})
    total = len(rows)
    if start >= total:
        raise ValueError(f"start {start} is beyond this generator's {total} cases")
    if start + count > total:
        raise ValueError(f"requested range ends at {start + count}, beyond this generator's {total} cases")
    return rows[start:start + count]


def solve(case: dict) -> str:
    kind = case["task_type"]
    if kind == "phrase_completion":
        try:
            return PHRASES[case["phrase_id"]][1]
        except KeyError as error:
            raise ValueError(f"Unknown v2 phrase family: {case.get('phrase_id')}") from error
    if kind == "sentence_completion":
        number = case["subject_number"]
        if number == "singular":
            return case["singular_verb"]
        if number == "plural":
            return case["base_verb"]
        raise ValueError(f"Unsupported grammatical number: {number}")
    raise ValueError(f"Unsupported v2 task type: {kind}")


def write_shard(output: Path, seed: int, start: int, count: int) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    generated = generate(seed, count, start)
    entries = []
    for split in ("train", "dev", "confirmation"):
        subset = [row for row in generated if row["split"] == split]
        if not subset:
            continue
        path = output / f"completion-v2-cases-{split}-{seed}-{start:07d}-{count:07d}.jsonl"
        rendered = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in subset)
        if path.exists() and path.read_text() != rendered:
            raise ValueError(f"Existing v2 shard conflicts with deterministic output: {path}")
        if not path.exists():
            path.write_text(rendered)
        entries.append({"file": path.name, "start": start, "count": len(subset), "split": split,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("generator") != GENERATOR or manifest.get("seed") != seed:
            raise ValueError("Output directory belongs to another v2 seed or generator")
    else:
        manifest = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR, "seed": seed, "shards": []}
    for entry in entries:
        existing = [item for item in manifest["shards"] if item["file"] == entry["file"]]
        if existing and existing[0] != entry:
            raise ValueError("V2 manifest conflicts with deterministic shard")
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
        raise ValueError("Dataset is not a phrase-sentence-completion-v2 dataset")
    expected = {row["index"]: row for row in generate(manifest["seed"], 2208)}
    expected_splits = _family_splits(manifest["seed"])
    seen_ids, seen_indexes, seen_prompts, family_splits = set(), set(), set(), {}
    split_counts, type_counts = {}, {}
    for entry in manifest["shards"]:
        path = output / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"V2 shard hash mismatch: {path}")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != entry["count"] or any(row["split"] != entry["split"] for row in rows):
            raise ValueError("V2 shard count or split mismatch")
        for row in rows:
            if row.get("index") not in expected or row != expected[row["index"]]:
                raise ValueError("V2 case differs from the deterministic generator record")
            normalized = normalized_question(row["prompt"])
            if row["case_id"] in seen_ids or row["index"] in seen_indexes or normalized in seen_prompts:
                raise ValueError("Duplicate v2 case ID, index, or normalized prompt")
            if row["answer"] != solve(row) or row["answer"] not in row["options"] or len(set(row["options"])) != 2:
                raise ValueError("Invalid v2 trusted answer or choice set")
            if row["split"] != expected_splits.get(row["family"]):
                raise ValueError("V2 family does not have its deterministic split")
            prior = family_splits.setdefault(row["family"], row["split"])
            if prior != row["split"]:
                raise ValueError("V2 family leaked across splits")
            seen_ids.add(row["case_id"])
            seen_indexes.add(row["index"])
            seen_prompts.add(normalized)
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            type_counts[row["task_type"]] = type_counts.get(row["task_type"], 0) + 1
    if len(seen_ids) != manifest.get("unique_cases"):
        raise ValueError("V2 manifest count does not match its shards")
    return {"unique_cases": len(seen_ids), "unique_prompts": len(seen_prompts),
            "split_counts": split_counts, "task_type_counts": type_counts,
            "unique_families": len(family_splits)}
