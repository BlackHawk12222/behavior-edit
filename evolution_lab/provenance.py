"""Dataset integrity and run provenance without importing optional ML packages."""
import hashlib
import importlib.metadata
import json
import platform
import os
import subprocess
import unicodedata


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def normalized_prompt(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def assert_disjoint(train, evaluation):
    """Reject normalized exact leakage; this cannot detect semantic paraphrases."""
    prompts = {normalized_prompt(row["prompt"]) for row in train}
    ids = {row["id"] for row in train}
    for row in evaluation:
        if row["id"] in ids or normalized_prompt(row["prompt"]) in prompts:
            raise ValueError(f"Training/evaluation overlap: {row['id']}")


def environment():
    versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_commit": commit, "git_dirty": dirty,
            "thread_environment": {k:os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE")}}
