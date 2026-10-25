import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evolution_lab.__main__ import TASK_FILE, main, read_tasks
from evolution_lab.metrics import compare_before_after, grade, summarize, validate_tasks
from evolution_lab.runner import select_device
from evolution_lab.train_adapter import read_examples


class ScoringTests(unittest.TestCase):
    def test_shipped_tasks_are_valid(self):
        self.assertGreaterEqual(len(read_tasks(TASK_FILE)), 5)

    def test_training_examples_are_valid_and_holdout(self):
        path = TASK_FILE.with_name("adapter_examples.json")
        examples = read_examples(path)
        self.assertGreater(len(examples), 5)
        eval_prompts = {t["prompt"] for t in read_tasks(TASK_FILE)}
        self.assertFalse(any(e["prompt"] in eval_prompts for e in examples))

    def test_exact_is_strict_about_extra_text(self):
        task = {"kind": "exact", "target": "42"}
        self.assertTrue(grade(task, " 42\n")["passed"])
        self.assertFalse(grade(task, "The answer is 42")["passed"])

    def test_json_parsing_and_structure(self):
        task = {"kind": "json", "target": {"ok": True}}
        self.assertTrue(grade(task, '{"ok":true}')["passed"])
        self.assertFalse(grade(task, '~~~json\n{"ok":true}\n~~~')["passed"])
        self.assertFalse(grade(task, '{"ok":false}')["passed"])

    def test_word_limit_rejects_empty_text(self):
        task = {"kind": "max_words", "max_words": 3}
        self.assertTrue(grade(task, "One two three")["passed"])
        self.assertFalse(grade(task, "one two three four")["passed"])
        self.assertFalse(grade(task, "")["passed"])

    def test_contains_is_case_insensitive(self):
        self.assertTrue(grade({"kind": "contains", "target": "ready"}, "READY.")["passed"])

    def test_task_ids_must_be_unique(self):
        task = {"id": "x", "kind": "exact", "prompt": "Hi", "target": "Hi"}
        with self.assertRaises(ValueError):
            validate_tasks([task, task])

    def test_score_summary_counts_repetitions(self):
        rows = [
            {"condition": "a", "metrics": {"passed": True}, "latency_seconds": 1},
            {"condition": "a", "metrics": {"passed": False}, "latency_seconds": 3},
        ]
        result = summarize(rows)
        self.assertEqual(result["a"]["observations"], 2)
        self.assertEqual(result["a"]["pass_rate"], 0.5)
        self.assertEqual(result["a"]["mean_latency_seconds"], 2)

    def test_compare_phases_tracks_changes_and_regressions(self):
        cases = [
            ("a", "format", False, True, "wrong", "right"),
            ("b", "math", True, False, "42", "fourty-two"),
            ("c", "format", True, True, "yes", "sure"),
            ("d", "math", False, False, "?", "?"),
        ]
        before, after = [], []
        for identifier, category, a, b, old, new in cases:
            task = {"id": identifier, "category": category}
            before.append({
                "task_id": identifier, "task": task,
                "response": old, "metrics": {"passed": a},
            })
            after.append({
                "task_id": identifier, "task": task,
                "response": new, "metrics": {"passed": b},
            })
        report = compare_before_after(before, after)
        self.assertEqual(report["task_count"], 4)
        self.assertEqual(report["improved"], 1)
        self.assertEqual(report["regressed"], 1)
        self.assertEqual(report["unchanged_pass"], 1)
        self.assertEqual(report["unchanged_fail"], 1)
        self.assertEqual(report["response_changed"], 3)
        self.assertEqual(report["pass_rate_delta"], 0)
        self.assertEqual(report["per_category"]["format"]["improved"], 1)
        self.assertEqual(report["per_category"]["math"]["regressed"], 1)

    def test_compare_phases_rejects_unpaired_tasks(self):
        row = {
            "task_id": "a", "task": {"id": "a"}, "response": "ok",
            "metrics": {"passed": True},
        }
        with self.assertRaisesRegex(ValueError, "same nonempty task IDs"):
            compare_before_after([row], [])

    def test_compare_phases_rejects_duplicate_tasks(self):
        row = {
            "task_id": "a", "task": {"id": "a"}, "response": "ok",
            "metrics": {"passed": True},
        }
        with self.assertRaisesRegex(ValueError, "Repeated task ID"):
            compare_before_after([row, row], [row])

    def test_validate_command_needs_no_ml_dependencies(self):
        self.assertEqual(main(["validate"]), 0)

    def test_invalid_json_is_caught(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text(json.dumps([{"id": "x"}]))
            self.assertEqual(main(["validate", "--tasks", str(path)]), 2)


class DeviceTests(unittest.TestCase):
    @staticmethod
    def fake_torch(mps, cuda):
        return SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
            cuda=SimpleNamespace(is_available=lambda: cuda),
        )

    def test_auto_prefers_apple_silicon(self):
        self.assertEqual(select_device(self.fake_torch(True, False)), "mps")

    def test_auto_cpu_fallback(self):
        self.assertEqual(select_device(self.fake_torch(False, False)), "cpu")

    def test_auto_cuda_on_linux(self):
        self.assertEqual(select_device(self.fake_torch(False, True)), "cuda")

    def test_forced_unavailable_mps_errors(self):
        with self.assertRaisesRegex(RuntimeError, "MPS unavailable"):
            select_device(self.fake_torch(False, False), "mps")


if __name__ == "__main__":
    unittest.main()
