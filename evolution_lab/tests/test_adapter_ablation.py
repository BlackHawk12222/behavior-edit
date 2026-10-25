import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from evolution_lab.adapter_ablation import (
    create_ablation_bundle,
    selected_modules,
)
from evolution_lab.case_eval import validate_norm_match


class AdapterAblationTests(unittest.TestCase):
    def test_selection_is_exact_for_projection_and_layer_groups(self):
        names = [
            f"base_model.model.model.layers.{layer}.self_attn.{projection}.lora_B.weight"
            for layer in range(4)
            for projection in ("q_proj", "v_proj")
        ]
        q, count = selected_modules(names, "zero_q_proj")
        self.assertEqual(count, 4)
        self.assertEqual(len(q), 4)
        self.assertTrue(all("q_proj" in name for name in q))
        lower, _ = selected_modules(names, "zero_lower_half")
        upper, _ = selected_modules(names, "zero_upper_half")
        self.assertEqual(len(lower), 4)
        self.assertEqual(len(upper), 4)
        self.assertTrue(all("layers.0." in name or "layers.1." in name for name in lower))
        self.assertTrue(all("layers.2." in name or "layers.3." in name for name in upper))

    def test_bundle_zeros_same_modules_and_preserves_random_norm_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            tensors = {}
            report_layers = []
            for layer in range(4):
                for projection in ("q_proj", "v_proj"):
                    module = f"base_model.model.model.layers.{layer}.self_attn.{projection}"
                    tensors[f"{module}.lora_A.weight"] = torch.eye(2)
                    tensors[f"{module}.lora_B.weight"] = torch.eye(2) * (layer + 1)
                    report_layers.append({
                        "name": module,
                        "matrix_shape": [2, 2],
                        "rank": 2,
                        "scale": 1.0,
                        "delta_frobenius_norm": (2 ** 0.5) * (layer + 1),
                        "delta_max_abs": float(layer + 1),
                    })
            hashes = {}
            for adapter in ("adapter_lora", "adapter_random"):
                target = source / adapter
                target.mkdir()
                (target / "README.md").write_text("test adapter\n")
                (target / "adapter_config.json").write_text(json.dumps({
                    "base_model_name_or_path": "local-model",
                    "r": 2,
                    "lora_alpha": 2,
                }))
                save_file(tensors, str(target / "adapter_model.safetensors"))
                hashes["lora" if adapter == "adapter_lora" else "random"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(target.iterdir()) if path.is_file()
                }
            for filename in ("delta_lora.json", "delta_random.json"):
                (source / filename).write_text(json.dumps({
                    "adapted_modules": len(report_layers),
                    "delta_global_frobenius_norm": sum(
                        row["delta_frobenius_norm"] ** 2 for row in report_layers
                    ) ** 0.5,
                    "layers": report_layers,
                }))
            (source / "manifest.json").write_text(json.dumps({
                "status": "complete",
                "smoke_only": False,
                "model_id": "local-model",
                "resolved_revision": "a" * 40,
                "seed": 17,
                "behavior": "test",
                "training_sha256": "b" * 64,
                "settings": {"rank": 2},
                "adapter_hashes": hashes,
            }))

            output = root / "ablation"
            result = create_ablation_bundle(source, output, "zero_q_proj")
            self.assertEqual(result["intervention"], "zero_q_proj")
            self.assertEqual(len(result["zeroed_modules"]), 4)
            for adapter in ("adapter_lora", "adapter_random"):
                changed = load_file(str(output / adapter / "adapter_model.safetensors"))
                for name, tensor in changed.items():
                    if ".q_proj.lora_B.weight" in name:
                        self.assertEqual(torch.count_nonzero(tensor).item(), 0)
                    else:
                        self.assertTrue(torch.equal(tensor, tensors[name]))
            learned = json.loads((output / "delta_lora.json").read_text())
            random = json.loads((output / "delta_random.json").read_text())
            self.assertAlmostEqual(
                learned["delta_global_frobenius_norm"],
                random["delta_global_frobenius_norm"],
            )
            match = validate_norm_match(output / "delta_lora.json", output / "delta_random.json")
            self.assertLess(match["maximum_relative_error"], 1e-12)
            with self.assertRaises(FileExistsError):
                create_ablation_bundle(source, output, "zero_q_proj")

    def test_invalid_ablation_and_noncontiguous_layers_rejected(self):
        names = [
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
            "base_model.model.model.layers.2.self_attn.q_proj.lora_B.weight",
        ]
        with self.assertRaisesRegex(ValueError, "contiguous layer range"):
            selected_modules(names, "zero_q_proj")
        with self.assertRaisesRegex(ValueError, "Unsupported adapter ablation"):
            selected_modules(names, "everything")


if __name__ == "__main__":
    unittest.main()
