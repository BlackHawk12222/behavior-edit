import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.casegen import generate, solve, validate_dataset, write_shard


class CaseGeneratorTests(unittest.TestCase):
    def test_splits_answers_and_prompts_are_deterministic(self):
        first = generate(seed=7, count=500)
        second = generate(seed=7, count=500)
        self.assertEqual(first, second)
        self.assertEqual(len({row["prompt"].casefold() for row in first}), len(first))
        self.assertEqual({row["split"] for row in first}, {"train", "dev", "confirmation"})
        self.assertTrue(all(row["answer"] == solve(row) for row in first))
        family_members = {}
        for row in first:
            family_members.setdefault(row["family"], []).append(row)
        self.assertTrue(all(len({r["split"] for r in members}) == 1 for members in family_members.values()))
        self.assertTrue(all(len({tuple(r["operands"]) for r in members}) == 1 for members in family_members.values()))
        self.assertTrue(all(len(members) == 4 for members in family_members.values()))

    def test_shards_resume_and_match_unsharded_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_shard(output, seed=11, start=0, count=60)
            manifest = write_shard(output, seed=11, start=60, count=70)
            again = write_shard(output, seed=11, start=0, count=60)
            self.assertEqual(manifest, again)
            self.assertEqual(validate_dataset(output)["unique_cases"], 130)
            actual = []
            for entry in manifest["shards"]:
                actual.extend(json.loads(line) for line in (output / entry["file"]).read_text().splitlines())
            actual.sort(key=lambda row: row["index"])
            self.assertEqual(actual, generate(seed=11, count=130))
            for split in ("train", "dev", "confirmation"):
                files = [item for item in manifest["shards"] if item["split"] == split]
                for item in files:
                    self.assertIn(f"cases-{split}-", item["file"])

    def test_tampered_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_shard(output, seed=9, start=0, count=10)
            shard = next(output.glob("cases-*.jsonl"))
            shard.write_text(shard.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_dataset(output)

    def test_rejects_invalid_ranges(self):
        for count, start in ((0, 0), (-1, 0), (1, -1)):
            with self.assertRaises(ValueError):
                generate(1, count, start)


if __name__ == "__main__":
    unittest.main()
