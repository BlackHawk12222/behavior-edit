import tempfile
import unittest
from pathlib import Path

from evolution_lab.case_eval import load_cases
from evolution_lab.completiongen import PHRASES as PHRASES_V1, SUBJECT_PAIRS as SUBJECTS_V1, VERB_PAIRS as VERBS_V1
from evolution_lab.completiongen_v2 import PHRASES as PHRASES_V2, SUBJECT_PAIRS as SUBJECTS_V2, VERB_PAIRS as VERBS_V2
from evolution_lab.completiongen_v3 import (
    PHRASES, SUBJECT_PAIRS, VERB_PAIRS, generate, solve, validate_dataset, write_dataset,
)


class CompletionGeneratorV3Tests(unittest.TestCase):
    def test_split_integrity_trusted_keys_and_lexical_holdouts(self):
        rows = generate(seed=42)
        self.assertEqual(len(rows), 8_740)
        by_family = {}
        counts = {}
        for row in rows:
            self.assertEqual(row["answer"], solve(row))
            self.assertIn(row["answer"], row["options"])
            self.assertEqual(len(set(row["options"])), 2)
            prior = by_family.setdefault(row["family"], row["split"])
            self.assertEqual(prior, row["split"])
            counts[(row["split"], row["task_type"])] = counts.get((row["split"], row["task_type"]), 0) + 1
        self.assertEqual(counts[("confirmation", "phrase_completion")], 32)
        self.assertEqual(counts[("confirmation", "sentence_completion")], 400)
        self.assertEqual(counts[("dev", "sentence_completion")], 400)
        self.assertTrue(set(PHRASES).isdisjoint(PHRASES_V1))
        self.assertTrue(set(PHRASES).isdisjoint(PHRASES_V2))
        v1_pairs = {(prefix.casefold(), answer.casefold()) for prefix, answer, _ in PHRASES_V1.values()}
        v2_pairs = {(prefix.casefold(), answer.casefold()) for prefix, answer, _ in PHRASES_V2.values()}
        v3_pairs = {(prefix.casefold(), answer.casefold()) for prefix, answer, _ in PHRASES.values()}
        self.assertTrue(v3_pairs.isdisjoint(v1_pairs | v2_pairs))
        self.assertTrue(set(SUBJECT_PAIRS).isdisjoint(SUBJECTS_V1))
        self.assertTrue(set(SUBJECT_PAIRS).isdisjoint(SUBJECTS_V2))
        self.assertEqual(len(set(VERB_PAIRS)), len(VERB_PAIRS))
        self.assertTrue(set(VERB_PAIRS).isdisjoint(VERBS_V1))
        self.assertTrue(set(VERB_PAIRS).isdisjoint(VERBS_V2))
        for split in ("train", "dev", "confirmation"):
            subjects = {row["subject"].removesuffix("s") for row in rows
                        if row["split"] == split and row["task_type"] == "sentence_completion"}
            verbs = {row["base_verb"] for row in rows
                     if row["split"] == split and row["task_type"] == "sentence_completion"}
            for other in ({"train", "dev", "confirmation"} - {split}):
                other_subjects = {row["subject"].removesuffix("s") for row in rows
                                  if row["split"] == other and row["task_type"] == "sentence_completion"}
                other_verbs = {row["base_verb"] for row in rows
                               if row["split"] == other and row["task_type"] == "sentence_completion"}
                self.assertTrue(subjects.isdisjoint(other_subjects))
                self.assertTrue(verbs.isdisjoint(other_verbs))

    def test_write_validate_and_load_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, seed=42)
            self.assertEqual(manifest["unique_cases"], 8_740)
            census = validate_dataset(root)
            self.assertEqual(census["unique_cases"], 8_740)
            source, confirmation = load_cases(root, "confirmation")
            self.assertEqual(source["manifest"]["generator"], "phrase-sentence-completion-v3-attractor")
            self.assertEqual(len(confirmation), 432)
            self.assertEqual({row["task_type"] for row in confirmation}, {"phrase_completion", "sentence_completion"})


if __name__ == "__main__":
    unittest.main()
