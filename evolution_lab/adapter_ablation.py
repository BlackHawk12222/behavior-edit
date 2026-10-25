"""Create reproducible component-ablation bundles for paired LoRA studies."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path


_LORA_B_KEY = re.compile(
    r"^(?P<module>.+\.layers\.(?P<layer>\d+)\.self_attn\.(?P<projection>q_proj|v_proj))\.lora_B\.weight$"
)
ABLATIONS = ("zero_q_proj", "zero_v_proj", "zero_lower_half", "zero_upper_half")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    return {
        child.name: _sha256(child)
        for child in sorted(Path(path).iterdir())
        if child.is_file()
    }


def selected_modules(tensor_names: list[str], ablation: str) -> tuple[list[str], int]:
    """Return exact module prefixes to zero and the layer count."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Unsupported adapter ablation: {ablation}")
    modules: dict[str, tuple[int, str]] = {}
    for name in tensor_names:
        match = _LORA_B_KEY.fullmatch(name)
        if match:
            modules[match.group("module")] = (
                int(match.group("layer")), match.group("projection")
            )
    layers = sorted({layer for layer, _ in modules.values()})
    if not layers or layers != list(range(layers[0], layers[-1] + 1)):
        raise ValueError("Adapter tensors must contain a contiguous layer range")
    midpoint = layers[0] + (len(layers) // 2)
    selected = []
    for module, (layer, projection) in modules.items():
        if ablation == "zero_q_proj" and projection == "q_proj":
            selected.append(module)
        elif ablation == "zero_v_proj" and projection == "v_proj":
            selected.append(module)
        elif ablation == "zero_lower_half" and layer < midpoint:
            selected.append(module)
        elif ablation == "zero_upper_half" and layer >= midpoint:
            selected.append(module)
    if not selected:
        raise ValueError(f"Ablation {ablation} matched no LoRA modules")
    return sorted(selected), len(layers)


def _ablate_tensor_file(source: Path, destination: Path,
                        modules_to_zero: set[str]) -> dict[str, str]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to create adapter ablations") from error

    tensors = load_file(str(source), device="cpu")
    found = set()
    for name, tensor in tensors.items():
        if name.endswith(".lora_B.weight"):
            module = name.removesuffix(".lora_B.weight")
            if module in modules_to_zero:
                tensors[name] = tensor.new_zeros(tensor.shape)
                found.add(module)
    if found != modules_to_zero:
        missing = sorted(modules_to_zero - found)
        raise ValueError(f"Ablation module set does not match adapter tensors: {missing[:3]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination))
    return _checkpoint_hashes(destination.parent)


def _ablate_delta_report(source: Path, destination: Path,
                         modules_to_zero: set[str]) -> dict:
    report = json.loads(Path(source).read_text())
    matched = set()
    layers = []
    for row in report.get("layers", []):
        edited = dict(row)
        if row["name"] in modules_to_zero:
            edited["delta_frobenius_norm"] = 0.0
            edited["delta_max_abs"] = 0.0
            matched.add(row["name"])
        layers.append(edited)
    if matched != modules_to_zero:
        missing = sorted(modules_to_zero - matched)
        raise ValueError(f"Ablation modules are absent from delta report: {missing[:3]}")
    report["layers"] = layers
    report["delta_global_frobenius_norm"] = math.sqrt(
        sum(float(row["delta_frobenius_norm"]) ** 2 for row in layers)
    )
    Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    return report


def create_ablation_bundle(source_run: Path, output: Path, ablation: str) -> dict:
    """Copy and ablate learned/random adapters while retaining paired provenance.

    The source run must be a completed ``train-cases`` directory containing
    ``adapter_lora``, ``adapter_random``, ``delta_lora.json``, and
    ``delta_random.json``. The output is a self-contained case-eval bundle.
    """
    source_run = Path(source_run)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite ablation output: {output}")
    manifest_path = source_run / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing source experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("smoke_only"):
        raise ValueError("Ablations require a complete non-smoke training run")
    learned = source_run / "adapter_lora" / "adapter_model.safetensors"
    random = source_run / "adapter_random" / "adapter_model.safetensors"
    for required in (learned, random, source_run / "delta_lora.json",
                     source_run / "delta_random.json"):
        if not required.is_file():
            raise ValueError(f"Missing source run artifact: {required}")
    for arm, adapter_dir in (("lora", learned.parent), ("random", random.parent)):
        if manifest.get("adapter_hashes", {}).get(arm) != _checkpoint_hashes(adapter_dir):
            raise ValueError(f"Source {arm} adapter does not match its training manifest hashes")
    selected, layer_count = selected_modules(
        [name for name in _tensor_names(learned)], ablation
    )
    modules_to_zero = set(selected)

    output.mkdir(parents=True)
    try:
        for adapter_name in ("adapter_lora", "adapter_random"):
            shutil.copytree(source_run / adapter_name, output / adapter_name)
        shutil.copy2(source_run / "manifest.json", output / "manifest.json")
        for filename in ("delta_lora.json", "delta_random.json"):
            shutil.copy2(source_run / filename, output / filename)

        hashes = {}
        for adapter_name, arm in (("adapter_lora", "lora"),
                                  ("adapter_random", "random")):
            tensor_file = output / adapter_name / "adapter_model.safetensors"
            hashes[arm] = _ablate_tensor_file(
                tensor_file, tensor_file, modules_to_zero
            )
        _ablate_delta_report(source_run / "delta_lora.json",
                             output / "delta_lora.json", modules_to_zero)
        _ablate_delta_report(source_run / "delta_random.json",
                             output / "delta_random.json", modules_to_zero)

        ablated_manifest = json.loads((output / "manifest.json").read_text())
        ablated_manifest["adapter_hashes"]["lora"] = hashes["lora"]
        ablated_manifest["adapter_hashes"]["random"] = hashes["random"]
        ablated_manifest["ablation"] = {
            "intervention": ablation,
            "zeroed_modules": selected,
            "layer_count": layer_count,
            "source_manifest_sha256": _sha256(manifest_path),
            "random_control_ablation": "identical modules zeroed in norm-matched random adapter",
        }
        (output / "manifest.json").write_text(
            json.dumps(ablated_manifest, indent=2) + "\n"
        )
        return ablated_manifest["ablation"]
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def _tensor_names(path: Path) -> list[str]:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to inspect adapter tensors") from error
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return list(handle.keys())
