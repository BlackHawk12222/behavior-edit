import json
import tempfile
import unittest
from pathlib import Path

from evolution_lab.case_mixture import _flatten_selected_adapter, validate_mixture_sources
from evolution_lab.provenance import checkpoint_hashes


def make_run(root, name, sampler, *, seed=42, rank=4, revision="a" * 40,
             dataset_fingerprint="dataset", ids=None):
    run = root / name
    adapter = run / "adapter_lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(name.encode())
    manifest = {
        "status": "complete",
        "run_id": name,
        "model_id": "local-model",
        "resolved_revision": revision,
        "dataset_manifest_sha256": dataset_fingerprint,
        "seed": seed,
        "precision": "torch.float32",
        "settings": {
            "steps": 64,
            "rank": rank,
            "target_modules": ["q_proj", "v_proj"],
            "sampling_strategy": sampler,
        },
        "training_ids_seen": ids or [name + "-example"],
        "adapter_hashes": {"lora": checkpoint_hashes(adapter)},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run


class CaseMixtureValidationTests(unittest.TestCase):
    def test_flattens_peft_named_adapter_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            named = root / "mixture"
            named.mkdir()
            (root / "README.md").write_text("adapter")
            (named / "adapter_config.json").write_text("{}")
            (named / "adapter_model.safetensors").write_bytes(b"weights")
            _flatten_selected_adapter(root, "mixture")
            self.assertTrue((root / "adapter_config.json").is_file())
            self.assertTrue((root / "adapter_model.safetensors").is_file())
            self.assertFalse(named.exists())

    def test_accepts_compatible_distinct_samplers_and_records_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform = make_run(root, "uniform", "uniform", ids=["a", "b"])
            balanced = make_run(root, "balanced", "task-balanced", ids=["b", "c"])
            manifests, provenance = validate_mixture_sources([uniform, balanced])
            self.assertEqual(len(manifests), 2)
            self.assertEqual(provenance["coefficients"], [0.5, 0.5])
            self.assertEqual(provenance["shared_training_example_ids"], 1)
            self.assertEqual(provenance["training_example_union"], 3)

    def test_rejects_different_base_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform = make_run(root, "uniform", "uniform")
            balanced = make_run(root, "balanced", "task-balanced", revision="b" * 40)
            with self.assertRaisesRegex(ValueError, "resolved_revision"):
                validate_mixture_sources([uniform, balanced])

    def test_rejects_changed_source_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uniform = make_run(root, "uniform", "uniform")
            balanced = make_run(root, "balanced", "task-balanced")
            (uniform / "adapter_lora" / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                validate_mixture_sources([uniform, balanced])


if __name__ == "__main__":
    unittest.main()
