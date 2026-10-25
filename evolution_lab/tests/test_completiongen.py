import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab import case_eval
from evolution_lab.completiongen import generate, solve, validate_dataset, write_shard


class CompletionGeneratorTests(unittest.TestCase):
    def test_generator_rejects_ranges_past_fixed_inventory(self):
        with self.assertRaisesRegex(ValueError, "beyond this generator"):
            generate(seed=42, count=673)

    def test_phrase_and_sentence_keys_are_deterministic_and_family_safe(self):
        rows = generate(seed=42, count=672)
        self.assertEqual(len(rows), 672)
        self.assertEqual(rows, generate(seed=42, count=672))
        self.assertEqual(sum(row["task_type"] == "phrase_completion" for row in rows), 160)
        self.assertEqual(sum(row["task_type"] == "sentence_completion" for row in rows), 512)
        self.assertEqual(len({row["case_id"] for row in rows}), len(rows))
        self.assertEqual(len({row["prompt"].casefold() for row in rows}), len(rows))
        family_splits = {}
        for row in rows:
            self.assertEqual(row["answer"], solve(row))
            self.assertIn(row["answer"], row["options"])
            self.assertEqual(len(set(row["options"])), 2)
            family_splits.setdefault(row["family"], row["split"])
            self.assertEqual(family_splits[row["family"]], row["split"])
        self.assertEqual(len(family_splits), 104)
        split_types = {(row["split"], row["task_type"]) for row in rows}
        self.assertTrue({("dev", "phrase_completion"), ("dev", "sentence_completion"),
                         ("confirmation", "phrase_completion"), ("confirmation", "sentence_completion")}
                        .issubset(split_types))

    def test_sentence_rule_selects_subject_verb_agreement(self):
        singular = next(row for row in generate(9, 672) if row.get("subject") == "The child")
        plural = next(row for row in generate(9, 672) if row.get("subject") == "The children")
        self.assertEqual(singular["answer"], singular["singular_verb"])
        self.assertEqual(plural["answer"], plural["base_verb"])

    def test_shard_resume_and_validate(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_shard(Path(directory), seed=42, start=0, count=250)
            second = write_shard(Path(directory), seed=42, start=250, count=422)
            self.assertEqual(first["unique_cases"], 250)
            self.assertEqual(second["unique_cases"], 672)
            self.assertEqual(validate_dataset(Path(directory))["unique_cases"], 672)
            resumed = write_shard(Path(directory), seed=42, start=0, count=250)
            self.assertEqual(resumed["unique_cases"], 672)
            self.assertEqual(validate_dataset(Path(directory))["unique_cases"], 672)

    def test_rehashed_prompt_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_shard(root, seed=42, start=0, count=20)
            shard_path = root / manifest["shards"][0]["file"]
            rows = [json.loads(line) for line in shard_path.read_text().splitlines()]
            rows[0]["prompt"] = "Choose any word you like."
            shard_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            updated = json.loads((root / "manifest.json").read_text())
            for entry in updated["shards"]:
                if entry["file"] == shard_path.name:
                    entry["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
            with self.assertRaisesRegex(ValueError, "deterministic generator record"):
                validate_dataset(root)

    def test_completion_evaluator_grades_final_word_and_keeps_format_separate(self):
        phrase = {
            "task_type": "phrase_completion", "answer": "pepper",
            "accepted_answers": ["pepper"], "options": ["pepper", "sugar"],
        }
        prose = case_eval.grade_completion(phrase, "The phrase is salt and pepper.")
        self.assertTrue(prose["correct"])
        self.assertFalse(prose["single_word_format"])
        full_sentence = {"answer": "walks", "options": ["walks", "walk"]}
        self.assertTrue(case_eval.grade_completion(full_sentence, "The student walks in the library.")["correct"])
        self.assertFalse(case_eval.grade_completion(full_sentence, "It is either walk or walks.")["correct"])
        self.assertTrue(case_eval.grade_completion(phrase, "pepper")["single_word_format"])
        self.assertFalse(case_eval.grade_completion(phrase, "sugar")["correct"])

    def test_paired_bootstrap_reports_completion_task_types_separately(self):
        rows = []
        for task_type in ("phrase_completion", "sentence_completion"):
            for family_index in range(2):
                for item_index in range(2):
                    case_id = f"{task_type}-{family_index}-{item_index}"
                    before = {"correct": False}
                    after = {"correct": family_index == 0}
                    shared = {"case_id": case_id, "family": f"{task_type}-family-{family_index}",
                              "task_type": task_type, "prompt": case_id}
                    rows.append({**shared, "arm": "base", "metrics": before})
                    rows.append({**shared, "arm": "prompt", "metrics": after})
        result = case_eval._summary(rows, ["base", "prompt"])
        paired = result["paired_vs_base"]["prompt"]
        self.assertEqual(paired["family_count"], 4)
        self.assertEqual(set(paired["paired_family_bootstrap95_by_task_type"]),
                         {"phrase_completion", "sentence_completion"})
        self.assertIsNotNone(paired["paired_family_bootstrap95_by_task_type"]["phrase_completion"])

    def test_case_loader_accepts_only_requested_completion_split(self):
        with tempfile.TemporaryDirectory() as directory:
            write_shard(Path(directory), seed=42, start=0, count=672)
            source, rows = case_eval.load_cases(Path(directory), "dev", limit=12, offset=2)
            self.assertEqual(source["manifest"]["generator"], "phrase-sentence-completion-v1")
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row["split"] == "dev" for row in rows))
            self.assertTrue(all(row["answer"] == solve(row) for row in rows))


if __name__ == "__main__":
    unittest.main()
