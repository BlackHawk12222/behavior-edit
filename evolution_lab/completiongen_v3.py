"""Fresh phrase and long-distance agreement completion benchmark.

Phrase keys are curated collocations. Sentence keys use explicit verb forms;
the main subject is separated from the blank by an agreement attractor. For
sentence cases, both subject and verb lexemes are disjoint across splits.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from .casegen import normalized_question

SCHEMA_VERSION = 1
GENERATOR = "phrase-sentence-completion-v3-attractor"

PHRASES = {
    "safe_sorry": ("better safe than", "sorry", "careful"),
    "practice_perfect": ("practice makes", "perfect", "progress"),
    "easy_come_go": ("easy come, easy", "go", "stay"),
    "nine_yards": ("the whole nine", "yards", "months"),
    "cloud_nine": ("on cloud", "nine", "ten"),
    "dog_day": ("every dog has its", "day", "chance"),
    "fortune_bold": ("fortune favors the", "bold", "lucky"),
    "cut_chase": ("cut to the", "chase", "story"),
    "nail_head": ("hit the nail on the", "head", "hand"),
    "out_hand": ("get out of", "hand", "place"),
    "arm_leg": ("cost an arm and a", "leg", "foot"),
    "blue_moon": ("once in a blue", "moon", "sun"),
    "under_weather": ("under the", "weather", "clouds"),
    "break_leg": ("break a", "leg", "foot"),
    "both_worlds": ("the best of both", "worlds", "sides"),
    "grapevine": ("through the", "grapevine", "newspaper"),
    "insult_injury": ("add insult to", "injury", "illness"),
    "burn_bridges": ("burn", "bridges", "roads"),
    "pull_leg": ("pull someone's", "leg", "arm"),
    "square_one": ("back to square", "one", "zero"),
    "wrong_tree": ("barking up the wrong", "tree", "house"),
    "penny_thoughts": ("a penny for your", "thoughts", "plans"),
    "last_straw": ("the last", "straw", "stick"),
    "blind_eye": ("turn a blind", "eye", "cheek"),
    "grain_salt": ("take it with a grain of", "salt", "sugar"),
    "pan_fire": ("out of the frying pan into the", "fire", "smoke"),
    "ends_meet": ("make ends", "meet", "join"),
    "same_page": ("on the same", "page", "chapter"),
    "hand_feeds": ("bite the hand that feeds", "you", "them"),
    "chin_up": ("keep your chin", "up", "high"),
    "eggs_basket": ("put all your eggs in one", "basket", "box"),
    "chips_fall": ("let the chips fall where they", "may", "might"),
    "throw_towel": ("throw in the", "towel", "blanket"),
    "elephant_room": ("the elephant in the", "room", "house"),
    "pigs_fly": ("when pigs", "fly", "swim"),
    "kick_bucket": ("kick the", "bucket", "can"),
    "right_rain": ("as right as", "rain", "shine"),
    "act_together": ("get your act", "together", "ready"),
    "rolling_stone": ("a rolling stone gathers no", "moss", "leaves"),
    "pudding_proof": ("the proof is in the", "pudding", "recipe"),
    "needle_measure": ("measure twice, cut", "once", "first"),
    "two_birds": ("kill two birds with one", "stone", "arrow"),
    "bygones": ("let bygones be", "bygones", "memories"),
    "still_water": ("still waters run", "deep", "quiet"),
    "many_cooks": ("too many cooks spoil the", "broth", "meal"),
    "roads_rome": ("all roads lead to", "Rome", "home"),
    "needle_thread": ("thread the", "needle", "button"),
    "haste_waste": ("haste makes", "waste", "trouble"),
    "fortune_fool": ("a fool and his money are soon", "parted", "counted"),
}

# 32 lexemes per inventory.  V1 and V2 completion vocabularies are excluded.
SUBJECT_PAIRS = (
    ("geologist", "geologists"), ("magician", "magicians"),
    ("caterer", "caterers"), ("architect", "architects"),
    ("baker", "bakers"), ("cartographer", "cartographers"),
    ("astronomer", "astronomers"), ("gardener", "gardeners"),
    ("photographer", "photographers"), ("librarian", "librarians"),
    ("sculptor", "sculptors"), ("translator", "translators"),
    ("beekeeper", "beekeepers"), ("cashier", "cashiers"),
    ("composer", "composers"), ("electrician", "electricians"),
    ("lifeguard", "lifeguards"), ("chemist", "chemists"),
    ("optician", "opticians"), ("plumber", "plumbers"),
    ("conductor", "conductors"), ("mariner", "mariners"),
    ("curator", "curators"), ("analyst", "analysts"),
    ("acrobat", "acrobats"), ("blacksmith", "blacksmiths"),
    ("scout", "scouts"), ("seamstress", "seamstresses"),
    ("barista", "baristas"), ("botanist", "botanists"),
    ("referee", "referees"), ("scuba diver", "scuba divers"),
)

VERB_PAIRS = (
    ("drift", "drifts"), ("stroll", "strolls"),
    ("gather", "gathers"), ("fold", "folds"),
    ("hum", "hums"), ("knead", "kneads"),
    ("sprint", "sprints"), ("carve", "carves"),
    ("sneeze", "sneezes"), ("murmur", "murmurs"),
    ("float", "floats"), ("tinker", "tinkers"),
    ("march", "marches"), ("weave", "weaves"),
    ("soar", "soars"), ("polka", "polkas"),
    ("bake", "bakes"), ("scoop", "scoops"),
    ("nod", "nods"), ("shuffle", "shuffles"),
    ("roam", "roams"), ("stitch", "stitches"),
    ("cheer", "cheers"), ("drum", "drums"),
    ("twirl", "twirls"), ("splash", "splashes"),
    ("dig", "digs"), ("yodel", "yodels"),
    ("sway", "sways"), ("ski", "skis"),
    ("kneel", "kneels"), ("gallop", "gallops"),
)

ATTRACTORS = ("neighbor", "musician", "gardener", "sailor", "student", "chef")
CONTEXTS = (
    "before the morning bell", "along the river path", "during the festival",
    "near the old station", "after the afternoon rain", "beside the market",
)
PHRASE_TEMPLATES = (
    'Complete this familiar phrase: "{prefix} ___." Choose {choices}.',
    'Fill in the missing word: "{prefix} ___." Options: {choices}.',
    'Which word completes "{prefix} ___"? Pick one: {choices}.',
    'Finish the phrase "{prefix} ___" using one choice: {choices}.',
)
SENTENCE_TEMPLATES = (
    'Complete the sentence: "The {subject}, along with the {attractor}, ___ {context}." Choose {choices}.',
    'Select the correct verb: "The {subject}, as well as the {attractor}, ___ {context}." Options: {choices}.',
    'Fill the blank: "The {subject}, together with the {attractor}, ___ {context}." Pick {choices}.',
    'Which verb agrees with the main subject? "The {subject}, accompanied by the {attractor}, ___ {context}." Choices: {choices}.',
)


def _split_inventory(values: tuple, seed: int, kind: str) -> dict[str, list]:
    ranked = sorted(values, key=lambda value: hashlib.sha256(
        f"{GENERATOR}:{seed}:{kind}:{value[0]}".encode()).hexdigest())
    train_end = math.floor(len(ranked) * .70)
    dev_count = round(len(ranked) * .15)
    return {
        "train": ranked[:train_end],
        "dev": ranked[train_end:train_end + dev_count],
        "confirmation": ranked[train_end + dev_count:],
    }


def _choice_text(options: list[str], seed: int, family: str, variant: int) -> str:
    ordered = list(options)
    random.Random(f"completion-v3:{seed}:{family}:{variant}").shuffle(ordered)
    return " or ".join(ordered)


def generate(seed: int = 42) -> list[dict]:
    rows: list[dict] = []
    phrase_splits = _split_inventory(tuple((key, value) for key, value in PHRASES.items()), seed, "phrase")
    family_splits: dict[str, str] = {}
    for split, phrases in phrase_splits.items():
        for phrase_id, (prefix, answer, distractor) in phrases:
            family = f"phrase-v3-{phrase_id}"
            family_splits[family] = split
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
                family = f"sentence-v3-{singular}-{base_verb}"
                family_splits[family] = split
                for number, subject in (("singular", singular), ("plural", plural)):
                    for attractor_number, attractor in (
                        ("singular", ATTRACTORS[0]), ("plural", ATTRACTORS[1] + "s"),
                    ):
                        answer = singular_verb if number == "singular" else base_verb
                        distractor = base_verb if number == "singular" else singular_verb
                        options = [answer, distractor]
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
        row.update({"case_id": f"completion-v3-{seed}-{index:07d}", "seed": seed, "index": index})
    return rows


def solve(case: dict) -> str:
    if case["task_type"] == "phrase_completion":
        try:
            return PHRASES[case["phrase_id"]][1]
        except KeyError as error:
            raise ValueError(f"Unknown v3 phrase family: {case.get('phrase_id')}") from error
    if case["task_type"] == "sentence_completion":
        number = case.get("subject_number")
        if number == "singular":
            return case["singular_verb"]
        if number == "plural":
            return case["base_verb"]
    raise ValueError(f"Unsupported v3 completion case: {case.get('task_type')}")


def write_dataset(output: Path, seed: int = 42) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = generate(seed)
    entries = []
    for split in ("train", "dev", "confirmation"):
        subset = [row for row in rows if row["split"] == split]
        path = output / f"completion-v3-cases-{split}-{seed}.jsonl"
        rendered = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in subset)
        if path.exists() and path.read_text() != rendered:
            raise ValueError(f"Existing v3 dataset conflicts with deterministic output: {path}")
        if not path.exists():
            path.write_text(rendered)
        entries.append({"file": path.name, "count": len(subset), "split": split,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema_version": SCHEMA_VERSION, "generator": GENERATOR, "seed": seed,
                "unique_cases": len(rows), "unique_families": len({row["family"] for row in rows}),
                "lexically_disjoint_sentence_splits": True, "shards": entries}
    path = output / "manifest.json"
    rendered = json.dumps(manifest, indent=2) + "\n"
    if path.exists() and path.read_text() != rendered:
        raise ValueError("Existing v3 manifest conflicts with deterministic output")
    if not path.exists():
        path.write_text(rendered)
    return manifest


def validate_dataset(output: Path) -> dict:
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("generator") != GENERATOR:
        raise ValueError("Dataset is not a completion-v3 dataset")
    expected = {row["case_id"]: row for row in generate(manifest["seed"])}
    ids, prompts, family_splits = set(), set(), {}
    split_counts, task_counts = {}, {}
    for entry in manifest["shards"]:
        path = output / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"V3 shard hash mismatch: {path}")
        shard = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(shard) != entry["count"]:
            raise ValueError("V3 shard count mismatch")
        for row in shard:
            if row.get("split") != entry.get("split"):
                raise ValueError("V3 shard contains a row from a different split")
            if row.get("case_id") not in expected or row != expected[row["case_id"]]:
                raise ValueError("V3 case differs from deterministic trusted record")
            prompt = normalized_question(row["prompt"])
            if row["case_id"] in ids or prompt in prompts:
                raise ValueError("Duplicate v3 case ID or normalized prompt")
            if solve(row) != row["answer"] or row["answer"] not in row["options"] or len(set(row["options"])) != 2:
                raise ValueError("Invalid v3 answer key or choices")
            prior = family_splits.setdefault(row["family"], row["split"])
            if prior != row["split"]:
                raise ValueError("V3 family leaks across splits")
            ids.add(row["case_id"]); prompts.add(prompt)
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
    if len(ids) != manifest.get("unique_cases") or ids != set(expected):
        raise ValueError("V3 manifest is incomplete or has an invalid case count")
    return {"unique_cases": len(ids), "unique_prompts": len(prompts),
            "unique_families": len(family_splits), "split_counts": split_counts,
            "task_type_counts": task_counts}
