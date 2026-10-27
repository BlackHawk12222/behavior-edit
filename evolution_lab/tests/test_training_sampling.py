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

    def test_task_ratio_holds_within_task_streams_constant(self):
        examples = self.examples()
        schedules = [
            training_sequence(examples, 64, 19, "task-ratio", ratio)
            for ratio in (
                {"phrase_completion": 1, "sentence_completion": 63},
                {"phrase_completion": 16, "sentence_completion": 48},
                {"phrase_completion": 32, "sentence_completion": 32},
            )
        ]
        expected_counts = [(1, 63), (16, 48), (32, 32)]
        task_streams = []
        for schedule, (phrases, sentences) in zip(schedules, expected_counts):
            self.assertEqual(len(schedule), 64)
            actual = [sum(row["task_type"] == kind for row in schedule)
                      for kind in ("phrase_completion", "sentence_completion")]
            self.assertEqual(actual, [phrases, sentences])
            task_streams.append({kind: [row["id"] for row in schedule
                                        if row["task_type"] == kind]
                                 for kind in ("phrase_completion", "sentence_completion")})
        for task in ("phrase_completion", "sentence_completion"):
            for left, right in zip(task_streams, task_streams[1:]):
                shorter = min(len(left[task]), len(right[task]))
                self.assertEqual(left[task][:shorter], right[task][:shorter])

    def test_task_ratio_without_replacement_visits_each_support_example_once(self):
        examples = [
            {"id": f"phrase-{index}", "task_type": "phrase_completion"}
            for index in range(16)
        ] + [
            {"id": f"sentence-{index}", "task_type": "sentence_completion"}
            for index in range(48)
        ]
        ratio = {"phrase_completion": 16, "sentence_completion": 48}
        schedule = training_sequence(
            examples, 64, 42, "task-ratio", ratio, schedule_seed=2603,
            example_sampling="without-replacement",
        )
        for task, support_size in ratio.items():
            seen = [row["id"] for row in schedule if row["task_type"] == task]
            expected = {f"{task.removesuffix('_completion')}-{index}"
                        for index in range(support_size)}
            self.assertEqual(len(seen), support_size)
            self.assertEqual(len(set(seen)), support_size)
            self.assertEqual(set(seen), expected)

    def test_task_ratio_without_replacement_recycles_only_after_full_epoch(self):
        examples = [
            {"id": f"phrase-{index}", "task_type": "phrase_completion"}
            for index in range(3)
        ] + [
            {"id": f"sentence-{index}", "task_type": "sentence_completion"}
            for index in range(5)
        ]
        ratio = {"phrase_completion": 7, "sentence_completion": 9}
        schedule = training_sequence(
            examples, 16, 7, "task-ratio", ratio, schedule_seed=903,
            example_sampling="without-replacement",
        )
        for task, support_size in (("phrase_completion", 3), ("sentence_completion", 5)):
            stream = [row["id"] for row in schedule if row["task_type"] == task]
            self.assertEqual(len(set(stream[:support_size])), support_size)
            self.assertEqual(len(stream), ratio[task])
            self.assertEqual(
                stream,
                [row["id"] for row in training_sequence(
                    examples, 16, 7, "task-ratio", ratio, schedule_seed=903,
                    example_sampling="without-replacement",
                ) if row["task_type"] == task],
            )

    def test_example_sampling_policy_rejected_outside_task_ratio(self):
        with self.assertRaisesRegex(ValueError, "only configurable for task-ratio"):
            training_sequence(self.examples(), 8, 0, "uniform",
                              example_sampling="without-replacement")

    def test_explicit_schedule_seed_nests_task_positions_and_keeps_data_streams(self):
        examples = self.examples()
        ratios = (
            {"phrase_completion": 1, "sentence_completion": 63},
            {"phrase_completion": 16, "sentence_completion": 48},
            {"phrase_completion": 32, "sentence_completion": 32},
        )
        schedules = [training_sequence(examples, 64, 19, "task-ratio", ratio,
                                       schedule_seed=903) for ratio in ratios]
        phrase_positions = [
            {index for index, row in enumerate(schedule)
             if row["task_type"] == "phrase_completion"}
            for schedule in schedules
        ]
        self.assertTrue(phrase_positions[0] < phrase_positions[1] < phrase_positions[2])
        for schedule, count in zip(schedules, (1, 16, 32)):
            self.assertEqual(sum(row["task_type"] == "phrase_completion" for row in schedule), count)

        streams = {
            task: [[row["id"] for row in schedule if row["task_type"] == task]
                   for schedule in schedules]
            for task in ("phrase_completion", "sentence_completion")
        }
        for task, values in streams.items():
            for left, right in zip(values, values[1:]):
                shorter = min(len(left), len(right))
                self.assertEqual(left[:shorter], right[:shorter])

    def test_schedule_seed_is_independent_of_training_seed_and_sampled_ids(self):
        examples = self.examples()
        ratio = {"phrase_completion": 16, "sentence_completion": 48}
        first = training_sequence(examples, 64, 19, "task-ratio", ratio,
                                  schedule_seed=903)
        repeated = training_sequence(examples, 64, 29, "task-ratio", ratio,
                                     schedule_seed=903)
        other_order = training_sequence(examples, 64, 19, "task-ratio", ratio,
                                        schedule_seed=904)
        self.assertEqual([row["task_type"] for row in first],
                         [row["task_type"] for row in repeated])
        self.assertNotEqual([row["task_type"] for row in first],
                            [row["task_type"] for row in other_order])
        for task in ("phrase_completion", "sentence_completion"):
            self.assertEqual(
                [row["id"] for row in first if row["task_type"] == task],
                [row["id"] for row in other_order if row["task_type"] == task],
            )

    def test_task_ratio_without_schedule_seed_preserves_historical_order(self):
        examples = self.examples()
        ratio = {"phrase_completion": 8, "sentence_completion": 56}
        expected = ["phrase_completion"] * 8 + ["sentence_completion"] * 56
        random.Random(7).shuffle(expected)
        actual = [row["task_type"] for row in training_sequence(
            examples, 64, 7, "task-ratio", ratio,
        )]
        self.assertEqual(actual, expected)

    def test_task_ratio_is_reproducible_and_rejects_bad_counts(self):
        examples = self.examples()
        ratio = {"phrase_completion": 8, "sentence_completion": 56}
        first = training_sequence(examples, 64, 7, "task-ratio", ratio)
        second = training_sequence(examples, 64, 7, "task-ratio", ratio)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        for invalid in (
            {"phrase_completion": 8},
            {"phrase_completion": 8, "sentence_completion": 55},
            {"phrase_completion": 0, "sentence_completion": 64},
            {"phrase_completion": 8.0, "sentence_completion": 56},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                training_sequence(examples, 64, 7, "task-ratio", invalid)


if __name__ == "__main__":
    unittest.main()
