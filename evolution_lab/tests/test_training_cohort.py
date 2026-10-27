import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from evolution_lab import training_cohort
from evolution_lab.training_cohort import parse_task_exposure_ratio, select_fixed_cohort
from evolution_lab.provenance import fingerprint


def synthetic_examples():
    rows = []
    for family_index in range(8):
        for variant in range(2):
            rows.append({
                "id": f"phrase-{family_index}-{variant}",
                "family": f"phrase-family-{family_index}",
                "task_type": "phrase_completion",
                "split": "train",
                "prompt": f"Phrase {family_index} variant {variant}",
                "answer": "trusted phrase",
            })
    for family_index in range(20):
        for variant in range(4):
            number = "singular" if variant % 2 == 0 else "plural"
            rows.append({
                "id": f"sentence-{family_index}-{variant}",
                "family": f"sentence-family-{family_index}",
                "task_type": "sentence_completion",
                "split": "train",
                "prompt": f"Sentence {family_index} variant {variant}",
                "answer": "trusted verb",
                "subject_number": number,
                "attractor_number": "plural" if variant else "singular",
            })
    return rows


class TrainingCohortTests(unittest.TestCase):
    def test_ratio_parser_requires_explicit_positive_unique_counts(self):
        self.assertEqual(
            parse_task_exposure_ratio("phrase_completion=1,sentence_completion=63"),
            {"phrase_completion": 1, "sentence_completion": 63},
        )
        for invalid in (
            "phrase_completion=1",
            "phrase_completion=1,phrase_completion=63",
            "phrase_completion=0,sentence_completion=64",
            "phrase_completion=one,sentence_completion=63",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_task_exposure_ratio(invalid)

    def test_fixed_cohort_is_deterministic_family_unique_and_subject_balanced(self):
        examples = synthetic_examples()
        selected = select_fixed_cohort(examples, phrase_count=4,
                                       sentence_count=8, seed=17)
        repeated = select_fixed_cohort(examples, phrase_count=4,
                                       sentence_count=8, seed=17)
        self.assertEqual(selected, repeated)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["id"] for row in selected}), 12)
        self.assertTrue(all(row["split"] == "train" for row in selected))
        phrases = [row for row in selected if row["task_type"] == "phrase_completion"]
        sentences = [row for row in selected if row["task_type"] == "sentence_completion"]
        self.assertEqual(len({row["family"] for row in phrases}), 4)
        self.assertEqual(len({row["family"] for row in sentences}), 8)
        self.assertEqual(len(sentences), len({row["family"] for row in sentences}))
        self.assertEqual(Counter(row["subject_number"] for row in sentences),
                         {"singular": 4, "plural": 4})

    def test_fixed_cohort_rejects_duplicate_family_as_nontrain_input(self):
        examples = synthetic_examples()
        examples.append({**examples[0], "id": "dev-row", "split": "dev"})
        with self.assertRaisesRegex(ValueError, "train-split"):
            select_fixed_cohort(examples, phrase_count=4,
                                sentence_count=8, seed=17)

    def test_cohort_writer_persists_hash_bound_examples_and_manifest(self):
        source = {"manifest": {"generator": "synthetic", "seed": 1}}
        examples = synthetic_examples()
        class Args:
            phrase_count = 4
            sentence_count = 8
            seed = 17

        with tempfile.TemporaryDirectory() as directory:
            Args.dataset = Path(directory) / "dataset"
            Args.output = Path(directory) / "cohort"
            with patch.object(training_cohort, "prepare_training_examples",
                              return_value=(source, examples)):
                self.assertEqual(training_cohort.create_training_cohort(Args), 0)
            manifest = json.loads((Args.output / "manifest.json").read_text())
            rows = [json.loads(line) for line in
                    (Args.output / "cohort_examples.jsonl").read_text().splitlines()]
            self.assertEqual(fingerprint(rows), manifest["training_data_sha256"])
            self.assertEqual(len(rows), manifest["training_count"])
            self.assertEqual(manifest["source_split"], "train")


if __name__ == "__main__":
    unittest.main()
