import tempfile
import unittest
from pathlib import Path

from evolution_lab.case_training import prepare_training_examples
from evolution_lab.completiongen import write_shard


class CaseTrainingTests(unittest.TestCase):
    def test_training_examples_are_only_family_safe_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset"
            write_shard(dataset, seed=42, start=0, count=672)
            source, examples = prepare_training_examples(dataset)
            self.assertEqual(source["manifest"]["generator"], "phrase-sentence-completion-v1")
            self.assertEqual(len(examples), 464)
            self.assertTrue(all(example["split"] == "train" for example in examples))
            self.assertEqual(len({example["id"] for example in examples}), 464)
            self.assertEqual(len({example["family"] for example in examples}), 72)
            self.assertEqual(sum(example["task_type"] == "phrase_completion" for example in examples), 112)
            self.assertEqual(sum(example["task_type"] == "sentence_completion" for example in examples), 352)


if __name__ == "__main__":
    unittest.main()
