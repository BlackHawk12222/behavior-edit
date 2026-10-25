import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evolution_lab import case_eval
from evolution_lab.casegen import write_shard
from evolution_lab.provenance import checkpoint_hashes


class CaseEvaluationTests(unittest.TestCase):
    def test_default_prompt_is_task_matched_for_both_completion_versions(self):
        for generator in (
            "phrase-sentence-completion-v1",
            "phrase-sentence-completion-v2-new-vocabulary",
        ):
            self.assertEqual(
                case_eval.default_prompt_system(generator),
                case_eval.COMPLETION_PROMPT_SYSTEM,
            )
        self.assertEqual(
            case_eval.default_prompt_system("arithmetic-v3"),
            case_eval.ARITHMETIC_PROMPT_SYSTEM,
        )

    def test_arithmetic_value_is_separate_from_format(self):
        prose = case_eval.grade_answer("58", "The sum of 20 and 38 is 58.")
        self.assertTrue(prose["correct"])
        self.assertEqual(prose["parsed_integer"], 58)
        self.assertEqual(prose["integer_token_count"], 3)
        self.assertFalse(prose["single_integer_format"])
        self.assertTrue(case_eval.grade_answer("58", "Answer: 58.")["correct"])
        self.assertTrue(case_eval.grade_answer("58", "58\n")["single_integer_format"])
        self.assertTrue(case_eval.grade_answer("12000", "12,000")["correct"])
        self.assertTrue(case_eval.grade_answer("12000", "12,000")["single_integer_format"])
        self.assertFalse(case_eval.grade_answer("58", "58 rather than 59")["correct"])
        self.assertTrue(case_eval.grade_answer("58", "20 + 38 = 58")["correct"])
        self.assertFalse(case_eval.grade_answer("58", "fifty-eight")["correct"])

    def args(self, output, split="dev", allow_confirmation=False, resume=False):
        return SimpleNamespace(
            split=split, allow_confirmation=allow_confirmation, limit=None,
            checkpoint_every=2, max_new_tokens=8,
            revision="a" * 40, arms=["base", "prompt"], adapter=None,
            random_adapter=None, adapter_delta=None, random_delta=None,
            dataset=Path("fixture"), model="fixture",
            device="cpu", precision="float32", prompt_system="Give one integer.",
            output_dir=Path(output), resume=resume, offset=0,
        )

    def cases(self):
        return {"manifest": {"seed": 7, "schema_version": 3, "generator": "arithmetic-v3", "shards": []},
                "census": {"unique_cases": 2}}, [
            {"case_id": "c1", "index": 0, "family": "f1", "split": "dev",
             "prompt": "2 + 3", "answer": "5", "operation": "add", "operands": [2, 3]},
            {"case_id": "c2", "index": 1, "family": "f2", "split": "dev",
             "prompt": "4 + 3", "answer": "7", "operation": "add", "operands": [4, 3]},
        ]

    def test_confirmation_requires_explicit_opt_in_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "confirmation-campaign"
            args = self.args(output, split="confirmation")
            with patch.object(case_eval, "load_cases") as loader:
                with self.assertRaisesRegex(ValueError, "--allow-confirmation"):
                    case_eval.evaluate_cases(args)
                loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_campaign_lock_rejects_a_second_process(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / ".case-eval.lock"
            with case_eval._campaign_lock(lock):
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    with case_eval._campaign_lock(lock):
                        pass

    def test_dataset_bridge_selects_only_requested_split_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            write_shard(Path(directory), seed=7, start=0, count=500)
            source, rows = case_eval.load_cases(Path(directory), "dev", limit=4, offset=4)
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["index"] for row in rows], sorted(row["index"] for row in rows))
            self.assertTrue(all(row["split"] == "dev" for row in rows))
            _, all_rows = case_eval.load_cases(Path(directory), "dev")
            self.assertEqual([row["case_id"] for row in rows], [row["case_id"] for row in all_rows[4:8]])
            self.assertEqual(source["manifest"]["generator"], "arithmetic-v3")

    def test_adapter_campaign_requires_all_primary_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory) / "campaign")
            args.arms = ["base", "adapter"]
            args.adapter = Path("adapter")
            with self.assertRaisesRegex(ValueError, "require base, prompt, adapter, and norm-matched random"):
                case_eval.evaluate_cases(args)

    def test_random_control_requires_layerwise_norm_match(self):
        layer = {"name": "layer.q_proj", "matrix_shape": [2, 2], "rank": 1,
                 "scale": 2.0, "delta_frobenius_norm": 3.0}
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "lora.json", Path(directory) / "random.json"
            left.write_text(json.dumps({"delta_global_frobenius_norm": 3.0, "layers": [layer]}))
            matched = dict(layer, delta_frobenius_norm=3.00001)
            right.write_text(json.dumps({"delta_global_frobenius_norm": 3.00001, "layers": [matched]}))
            result = case_eval.validate_norm_match(left, right)
            self.assertLessEqual(result["maximum_relative_error"], 1e-4)
            mismatched = dict(layer, delta_frobenius_norm=4.0)
            right.write_text(json.dumps({"delta_global_frobenius_norm": 4.0, "layers": [mismatched]}))
            with self.assertRaisesRegex(ValueError, "not norm-matched"):
                case_eval.validate_norm_match(left, right)

    def test_saved_delta_sidecar_must_match_loaded_effective_weights(self):
        layer = {"name": "layer.q_proj", "matrix_shape": [2, 2], "rank": 1,
                 "scale": 2.0, "delta_frobenius_norm": 3.0, "delta_max_abs": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "delta.json"
            report.write_text(json.dumps({"delta_global_frobenius_norm": 3.0, "layers": [layer]}))
            result = case_eval.validate_delta_report(report, {
                "delta_global_frobenius_norm": 3.0, "layers": [dict(layer)],
            })
            self.assertEqual(result["maximum_relative_error"], 0.0)
            with self.assertRaisesRegex(ValueError, "do not match loaded adapter"):
                case_eval.validate_delta_report(report, {
                    "delta_global_frobenius_norm": 4.0,
                    "layers": [dict(layer, delta_frobenius_norm=4.0)],
                })

    def test_adapter_source_is_bound_to_checkpoint_hash_and_model_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "trial"
            adapter = root / "adapter_lora"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fixture"}))
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture weights")
            manifest = {
                "status": "complete", "smoke_only": False, "model_id": "fixture",
                "resolved_revision": "a" * 40, "behavior": "format", "seed": 4,
                "training_sha256": "training", "settings": {"rank": 4},
                "adapter_hashes": {"lora": checkpoint_hashes(adapter)},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            result = case_eval.validate_adapter_source(adapter, "adapter", "fixture", "a" * 40)
            self.assertEqual(result["seed"], 4)
            (adapter / "adapter_model.safetensors").write_bytes(b"modified weights")
            with self.assertRaisesRegex(ValueError, "do not match hashes"):
                case_eval.validate_adapter_source(adapter, "adapter", "fixture", "a" * 40)

    def test_resumes_without_duplicate_generations_and_keeps_raw_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "campaign"
            args = self.args(output)
            source = self.cases()
            failing = {"active": True}

            class FakeModel:
                def __init__(self, *unused, **kwargs):
                    self.resolved_revision = "a" * 40
                    self.device = "cpu"
                    self.precision = "torch.float32"
                    self.generated = 0

                def generate(self, system, prompt, max_tokens):
                    self.generated += 1
                    if system == args.prompt_system and prompt == "4 + 3" and failing["active"]:
                        failing["active"] = False
                        raise RuntimeError("simulated interruption")
                    return ("5" if prompt == "2 + 3" else "7"), 1, 0.01

            with patch.object(case_eval, "load_cases", return_value=source), \
                 patch.object(case_eval, "LocalModel", FakeModel):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    case_eval.evaluate_cases(args)
                journal = json.loads((output / "run.json").read_text())
                self.assertEqual(journal["status"], "failed")
                args.resume = True
                case_eval.evaluate_cases(args)

            rows = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({(row["arm"], row["case_id"]) for row in rows}), 4)
            self.assertTrue(all(row["metrics"]["correct"] for row in rows))
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["arms"]["base"]["unique_questions"], 2)
            self.assertEqual(report["arms"]["base"]["unique_normalized_prompts"], 2)
            self.assertEqual(report["paired_vs_base"]["prompt"]["paired_questions"], 2)
            journal = json.loads((output / "run.json").read_text())
            self.assertEqual(journal["status"], "complete")
            self.assertEqual(journal["model_generations"], 4)
            with patch.object(case_eval, "load_cases", return_value=source), \
                 patch.object(case_eval, "LocalModel", side_effect=AssertionError("must not reload a completed run")):
                self.assertEqual(case_eval.evaluate_cases(args), 0)
            with (output / "observations.jsonl").open("a") as stream:
                stream.write("\n")
            with patch.object(case_eval, "load_cases", return_value=source), \
                 patch.object(case_eval, "LocalModel", side_effect=AssertionError("checksum should fail before loading")):
                with self.assertRaisesRegex(ValueError, "response ledger.*checksum"):
                    case_eval.evaluate_cases(args)


if __name__ == "__main__":
    unittest.main()
