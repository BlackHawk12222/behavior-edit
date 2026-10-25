import random
import unittest

from evolution_lab.experiment import training_sequence


class TrainingSamplingTests(unittest.TestCase):
    def examples(self):
        return [
            {"id": f"phrase-{index}", "task_type": "phrase_completion"}
            for index in range(48)
        ] + [
            {"id": f"sentence-{index}", "task_type": "sentence_completion"}
            for index in range(528)
        ]

    def test_uniform_schedule_preserves_historical_epoch_shuffle(self):
        examples = self.examples()
        expected_order = list(examples)
        rng = random.Random(19)
        expected = []
        for step in range(40):
            if step % len(expected_order) == 0:
                rng.shuffle(expected_order)
            expected.append(expected_order[step % len(expected_order)]["id"])
        actual = [row["id"] for row in training_sequence(examples, 40, 19)]
        self.assertEqual(actual, expected)

    def test_task_balanced_schedule_equalizes_task_types_without_replacement(self):
        examples = self.examples()
        schedule = training_sequence(examples, 64, 19, "task-balanced")
        counts = {}
        for row in schedule:
            counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
        self.assertEqual(counts, {"phrase_completion": 32, "sentence_completion": 32})
        self.assertEqual(len({row["id"] for row in schedule}), 64)
        self.assertEqual(
            [row["id"] for row in schedule],
            [row["id"] for row in training_sequence(examples, 64, 19, "task-balanced")],
        )

    def test_task_balanced_rejects_single_task_dataset_and_bad_strategy(self):
        with self.assertRaisesRegex(ValueError, "at least two task types"):
            training_sequence([{"id": "one", "task_type": "phrase_completion"}], 1, 0,
                              "task-balanced")
        with self.assertRaisesRegex(ValueError, "Unsupported sampling"):
            training_sequence(self.examples(), 1, 0, "weighted")


if __name__ == "__main__":
    unittest.main()
