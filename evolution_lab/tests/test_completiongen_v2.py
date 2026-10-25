import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.case_eval import load_cases
from evolution_lab.completiongen import PHRASES as PHRASES_V1
from evolution_lab.completiongen import SUBJECT_PAIRS as SUBJECTS_V1
from evolution_lab.completiongen import VERB_PAIRS as VERBS_V1
from evolution_lab.completiongen_v2 import (
    PHRASES, SUBJECT_PAIRS, VERB_PAIRS, generate, solve, validate_dataset, write_shard,
)


class CompletionGeneratorV2Tests(unittest.TestCase):
    def test_fresh_vocabulary_and_family_stratified_splits(self):
        rows = generate(seed=43, count=2208)
        self.assertEqual(len(rows), 2208)
        self.assertEqual(sum(row["task_type"] == "phrase_completion" for row in rows), 160)
        self.assertEqual(sum(row["task_type"] == "sentence_completion" for row in rows), 2048)
        self.assertTrue(set(PHRASES).isdisjoint(PHRASES_V1))
        self.assertTrue(set(SUBJECT_PAIRS).isdisjoint(SUBJECTS_V1))
        self.assertTrue(set(VERB_PAIRS).isdisjoint(VERBS_V1))
        counts = {split: sum(row["split"] == split for row in rows)
                  for split in ("train", "dev", "confirmation")}
        self.assertEqual(counts, {"train": 1544, "dev": 328, "confirmation": 336})
        by_family = {}
        for row in rows:
            self.assertEqual(row["answer"], solve(row))
            self.assertEqual(len(set(row["options"])), 2)
            by_family.setdefault(row["family"], row["split"])
            self.assertEqual(by_family[row["family"]], row["split"])
        self.assertEqual(len(by_family), 296)

    def test_write_validate_and_split_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_shard(root, seed=43, start=0, count=2208)
            census = validate_dataset(root)
            self.assertEqual(census["unique_cases"], 2208)
            self.assertEqual(census["split_counts"], {"train": 1544, "dev": 328, "confirmation": 336})
            source, confirm = load_cases(root, "confirmation", limit=104)
            self.assertEqual(source["manifest"]["generator"], "phrase-sentence-completion-v2-new-vocabulary")
            self.assertEqual(len(confirm), 104)
            self.assertTrue(all(row["split"] == "confirmation" for row in confirm))

    def test_generator_rejects_ranges_past_inventory(self):
        with self.assertRaisesRegex(ValueError, "beyond this generator"):
            generate(seed=43, count=2209)

    def test_rehashed_prompt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_shard(root, seed=43, start=0, count=120)
            shard = root / manifest["shards"][0]["file"]
            rows = [json.loads(line) for line in shard.read_text().splitlines()]
            rows[0]["prompt"] = "Select any answer."
            shard.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            updated = json.loads((root / "manifest.json").read_text())
            for entry in updated["shards"]:
                if entry["file"] == shard.name:
                    entry["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "deterministic generator record"):
                validate_dataset(root)


if __name__ == "__main__":
    unittest.main()
