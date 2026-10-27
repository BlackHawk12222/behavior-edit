"""Fresh phrase and subject-verb agreement families for frozen transfer checks.

The v4 lexical pools are distinct from v1-v3.  Subject and verb inventories
are split independently before sentence families are formed, so confirmation
questions use held-out subject and verb lexemes as well as held-out families.
Phrase keys are curated collocations and remain a culturally narrow probe.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .casegen import normalized_question

SCHEMA_VERSION = 1
GENERATOR = "phrase-sentence-completion-v4-fresh-lexicon"

# Prefix, trusted completion, plausible but incorrect distractor.
PHRASES = {
    "bury_hatchet": ("bury the", "hatchet", "shovel"),
    "cut_corners": ("cut", "corners", "edges"),
    "face_music": ("face the", "music", "noise"),
    "lose_marbles": ("lose your", "marbles", "buttons"),
    "miss_boat": ("miss the", "boat", "train"),
    "on_fence": ("on the", "fence", "ladder"),
    "over_moon": ("over the", "moon", "hill"),
    "piece_cake": ("piece of", "cake", "bread"),
    "pull_strings": ("pull", "strings", "wires"),
    "between_lines": ("read between the", "lines", "words"),
    "eye_to_eye": ("see eye to", "eye", "face"),
    "rain_check": ("take a rain", "check", "forecast"),
    "bend_backwards": ("bend over", "backwards", "forward"),
    "under_belt": ("under your", "belt", "arm"),
    "drop_hat": ("at the drop of a", "hat", "coin"),
    "fish_water": ("like a fish out of", "water", "trouble"),
    "whole_shebang": ("the whole", "shebang", "story"),
    "hang_it": ("get the hang of", "it", "this"),
    "weather_storm": ("weather the", "storm", "rain"),
    "long_story_short": ("make a long story", "short", "square"),
    "talk_town": ("talk of the", "town", "street"),
    "hit_books": ("hit the", "books", "pages"),
    "cold_shoulder": ("give the cold", "shoulder", "hand"),
    "kill_time": ("kill", "time", "energy"),
    "ducks_row": ("get your ducks in a", "row", "line"),
    "all_ears": ("we are all", "ears", "listening"),
    "drawing_board": ("back to the drawing", "board", "page"),
    "fuel_fire": ("add fuel to the", "fire", "furnace"),
    "good_times_roll": ("let the good times", "roll", "spin"),
    "rose_thorn": ("every rose has its", "thorn", "leaf"),
    "reap_sow": ("you reap what you", "sow", "plant"),
    "baby_bathwater": ("don't throw the baby out with the", "bathwater", "laundry"),
    "iron_hot": ("strike while the iron is", "hot", "warm"),
    "best_foot": ("put your best foot", "forward", "first"),
    "watched_pot": ("a watched pot never", "boils", "bubbles"),
    "going_tough": ("when the going gets tough, the tough get", "going", "moving"),
    "heads_one": ("two heads are better than", "one", "two"),
    "sight_mind": ("out of sight, out of", "mind", "view"),
    "squeaky_grease": ("the squeaky wheel gets the", "grease", "oil"),
    "cold_feet": ("get cold", "feet", "hands"),
    "open_book": ("an open", "book", "letter"),
    "hard_nut": ("a hard nut to", "crack", "break"),
    "long_shot": ("a long", "shot", "jump"),
    "last_word": ("have the last", "word", "laugh"),
    "silver_spoon": ("born with a silver", "spoon", "fork"),
    "wild_goose": ("a wild goose", "chase", "race"),
    "common_ground": ("find common", "ground", "space"),
    "storm_tea": ("make a storm in a", "teacup", "teapot"),
}

# Explicit regular singular/plural pairs and verb forms. Keep labels auditable;
# do not derive English spelling rules from a suffix heuristic.
SUBJECT_PAIRS = (
    ("otter", "otters"), ("badger", "badgers"), ("gecko", "geckos"),
    ("robin", "robins"), ("heron", "herons"), ("moth", "moths"),
    ("sparrow", "sparrows"), ("beetle", "beetles"), ("acorn", "acorns"),
    ("pebble", "pebbles"), ("ribbon", "ribbons"), ("kettle", "kettles"),
    ("lantern", "lanterns"),
    ("feather", "feathers"), ("willow", "willows"), ("tulip", "tulips"),
    ("fern", "ferns"), ("maple", "maples"), ("thimble", "thimbles"),
    ("compass", "compasses"), ("trumpet", "trumpets"), ("cobweb", "cobwebs"),
    ("pocketwatch", "pocketwatches"),
    ("dandelion", "dandelions"), ("raindrop", "raindrops"),
    ("pinecone", "pinecones"), ("firefly", "fireflies"),
    ("ladybug", "ladybugs"), ("snowflake", "snowflakes"),
)

VERB_PAIRS = (
    ("blink", "blinks"), ("rumble", "rumbles"), ("crackle", "crackles"),
    ("wobble", "wobbles"), ("wiggle", "wiggles"), ("glow", "glows"),
    ("buzz", "buzzes"), ("chirp", "chirps"), ("purr", "purrs"),
    ("flutter", "flutters"), ("croak", "croaks"), ("squawk", "squawks"),
    ("wither", "withers"), ("shiver", "shivers"), ("tremble", "trembles"),
    ("sparkle", "sparkles"), ("flicker", "flickers"), ("drizzle", "drizzles"),
    ("thunder", "thunders"), ("simmer", "simmers"), ("bubble", "bubbles"),
    ("melt", "melts"), ("freeze", "freezes"), ("crumble", "crumbles"),
    ("squeak", "squeaks"), ("quiver", "quivers"), ("rustle", "rustles"),
    ("glisten", "glistens"), ("shimmer", "shimmers"), ("sizzle", "sizzles"),
    ("dangle", "dangles"), ("wilt", "wilts"),
)

ATTRACTORS = ("spectator", "bystander", "colleague", "caretaker")
CONTEXTS = (
    "beside the old fountain", "before the evening bell",
    "under the porch light", "along the quiet lane",
    "near the garden gate", "during the afternoon shower",
)
PHRASE_TEMPLATES = (
    'Complete this familiar phrase: "{prefix} ___." Choose {choices}.',
    'Fill in the missing word: "{prefix} ___." Options: {choices}.',
    'Which word completes "{prefix} ___"? Pick one: {choices}.',
    'Finish the phrase "{prefix} ___" using one choice: {choices}.',
    'Complete the expression "{prefix} ___." Select {choices}.',
    'Choose the word that finishes "{prefix} ___": {choices}.',
    'What completes the phrase "{prefix} ___"? Answer with a choice: {choices}.',
    'The familiar expression is "{prefix} ___." Choose {choices}.',
)
SENTENCE_TEMPLATES = (
    'Complete the sentence: "The {subject}, along with the {attractor}, ___ {context}." Choose {choices}.',
    'Select the correct verb: "The {subject}, as well as the {attractor}, ___ {context}." Options: {choices}.',
    'Fill the blank: "The {subject}, together with the {attractor}, ___ {context}." Pick {choices}.',
    'Which verb agrees with the main subject? "The {subject}, accompanied by the {attractor}, ___ {context}." Choices: {choices}.',
)


def _rank(seed: int, purpose: str, value: str) -> str:
    return hashlib.sha256(f"{GENERATOR}:{seed}:{purpose}:{value}".encode()).hexdigest()


def _split_inventory(values: tuple, seed: int, kind: str) -> dict[str, list]:
    ranked = sorted(values, key=lambda value: _rank(seed, kind, str(value[0])))
    train_end = math.floor(len(ranked) * 0.70)
    dev_count = round(len(ranked) * 0.15)
    return {
        "train": ranked[:train_end],
        "dev": ranked[train_end:train_end + dev_count],
        "confirmation": ranked[train_end + dev_count:],
    }


def _choice_text(options: list[str], seed: int, family: str, variant: int) -> str:
    ordered = list(options)
    random.Random(f"{GENERATOR}:{seed}:{family}:{variant}").shuffle(ordered)
    return " or ".join(ordered)


def generate(seed: int = 84) -> list[dict]:
    rows: list[dict] = []
    phrase_splits = _split_inventory(tuple(PHRASES.items()), seed, "phrase")
    for split, phrases in phrase_splits.items():
        for phrase_id, (prefix, answer, distractor) in phrases:
            family = f"phrase-v4-{phrase_id}"
            for variant, template in enumerate(PHRASE_TEMPLATES):
                options = [answer, distractor]
                rows.append({
                    "family": family, "split": split, "task_type": "phrase_completion",
                    "phrase_id": phrase_id, "prefix": prefix, "options": options,
                    "answer": answer,
                    "prompt": template.format(prefix=prefix, choices=_choice_text(options, seed, family, variant)),
                })

    subject_splits = _split_inventory(SUBJECT_PAIRS, seed, "subject")
    verb_splits = _split_inventory(VERB_PAIRS, seed, "verb")
    for split in ("train", "dev", "confirmation"):
        for singular, plural in subject_splits[split]:
            for base_verb, singular_verb in verb_splits[split]:
                family = f"sentence-v4-{singular}-{base_verb}"
                for number, subject in (("singular", singular), ("plural", plural)):
                    answer = singular_verb if number == "singular" else base_verb
                    distractor = base_verb if number == "singular" else singular_verb
                    options = [answer, distractor]
                    for attractor_number, attractor_root in (("singular", ATTRACTORS[0]), ("plural", ATTRACTORS[1])):
                        attractor = attractor_root if attractor_number == "singular" else f"{attractor_root}s"
                        for variant, template in enumerate(SENTENCE_TEMPLATES):
                            context_index = (variant + sum(map(ord, family + number + attractor_number))) % len(CONTEXTS)
                            rows.append({
                                "family": family, "split": split, "task_type": "sentence_completion",
                                "subject_number": number, "attractor_number": attractor_number,
                                "subject": subject, "attractor": attractor,
                                "base_verb": base_verb, "singular_verb": singular_verb,
                                "options": options, "answer": answer,
                                "prompt": template.format(
                                    subject=subject, attractor=attractor,
                                    context=CONTEXTS[context_index],
                                    choices=_choice_text(options, seed, family + number + attractor_number, variant),
                                ),
                            })

    rows.sort(key=lambda row: (row["split"], row["family"], row["task_type"], row.get("subject_number", ""), row.get("attractor_number", ""), row.get("phrase_id", ""), row["prompt"]))
    for index, row in enumerate(rows):
        row.update({"case_id": f"completion-v4-{seed}-{index:07d}", "seed": seed, "index": index})
    return rows


def solve(case: dict) -> str:
    if case.get("task_type") == "phrase_completion":
        try:
            return PHRASES[case["phrase_id"]][1]
        except KeyError as error:
            raise ValueError(f"Unknown v4 phrase family: {case.get('phrase_id')}") from error
    if case.get("task_type") == "sentence_completion":
        number = case.get("subject_number")
        if number == "singular":
            return case["singular_verb"]
        if number == "plural":
            return case["base_verb"]
    raise ValueError(f"Unsupported v4 completion case: {case.get('task_type')}")


def write_dataset(output: Path, seed: int = 84) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = generate(seed)
    entries = []
    shard_size = 750  # Keep each reproducibility shard comfortably below GitHub API limits.
    for split in ("train", "dev", "confirmation"):
        subset = [row for row in rows if row["split"] == split]
        for part, start in enumerate(range(0, len(subset), shard_size), start=1):
            shard_rows = subset[start : start + shard_size]
            path = output / f"completion-v4-cases-{split}-{seed}-part{part:02d}.jsonl"
            rendered = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in shard_rows)
            if path.exists() and path.read_text() != rendered:
                raise ValueError(f"Existing v4 dataset conflicts with deterministic output: {path}")
            if not path.exists():
                path.write_text(rendered)
            entries.append({"file": path.name, "count": len(shard_rows), "split": split,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {
        "schema_version": SCHEMA_VERSION, "generator": GENERATOR, "seed": seed,
        "unique_cases": len(rows), "unique_families": len({row["family"] for row in rows}),
        "subject_and_verb_lexicons_disjoint_across_splits": True,
        "phrase_source_note": "Researcher-curated familiar expressions; labels may vary by dialect and are not a culturally representative sample.",
        "shards": entries,
    }
    path = output / "manifest.json"
    rendered = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise ValueError("Existing v4 manifest conflicts with deterministic output")
    if not path.exists():
        path.write_text(rendered)
    return manifest


def validate_dataset(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("generator") != GENERATOR:
        raise ValueError("Dataset is not a completion-v4 dataset")
    expected = {row["case_id"]: row for row in generate(manifest["seed"])}
    ids, prompts, family_splits = set(), set(), {}
    split_counts, task_counts = {}, {}
    for entry in manifest["shards"]:
        path = output / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"V4 shard hash mismatch: {path}")
        shard = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(shard) != entry["count"]:
            raise ValueError("V4 shard count mismatch")
        for row in shard:
            if row.get("split") != entry.get("split") or row.get("case_id") not in expected or row != expected[row["case_id"]]:
                raise ValueError("V4 case differs from deterministic trusted record or split")
            prompt = normalized_question(row["prompt"])
            if row["case_id"] in ids or prompt in prompts:
                raise ValueError("Duplicate v4 case ID or normalized prompt")
            if solve(row) != row["answer"] or row["answer"] not in row["options"] or len(set(row["options"])) != 2:
                raise ValueError("Invalid v4 answer key or choices")
            prior = family_splits.setdefault(row["family"], row["split"])
            if prior != row["split"]:
                raise ValueError("V4 family leaks across splits")
            ids.add(row["case_id"]); prompts.add(prompt)
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
    if len(ids) != manifest.get("unique_cases") or ids != set(expected):
        raise ValueError("V4 manifest is incomplete or has an invalid case count")
    return {"unique_cases": len(ids), "unique_prompts": len(prompts),
            "unique_families": len(family_splits), "split_counts": split_counts,
            "task_type_counts": task_counts}
