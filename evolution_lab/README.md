# Model Evolution Lab — Apple Silicon experimental path (v0.1)

This module extends the **Behavior Editing** research repository with an independent,
small-model experimental workflow suitable for investigating on Apple Silicon Macs,
including a MacBook Neo with 8 GB unified memory.

It does **not** replace or claim Mac compatibility for the original CUDA-based
ROME, MEMIT, GRACE, LoRA, or large-model benchmark scripts.

## Quick start on a MacBook Neo (Apple Silicon)

Use Python 3.11. Install Xcode Command Line Tools if macOS asks.

~~~sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r evolution_lab/requirements-mac.txt

# No model download or heavy packages needed for the following checks:
python -m evolution_lab validate
python -m unittest discover -s evolution_lab/tests -v

# First experiment: 3 tasks, 2 conditions, 6 generations.
python -m evolution_lab run --device auto --limit 3
~~~

You can force CPU with \`--device cpu\` if MPS is unavailable. Test first
with \`--limit 1\` if you're uncertain about free memory or disk space.

The default is an open-weight Hugging Face model
(\`Qwen/Qwen2.5-0.5B-Instruct\`). Its model card and license are separate
from this repository's code license. The initial download requires internet
access and disk space. The first run may take longer due to downloads.

## What is being measured?

Every test has an **objective** grader: exact response, parsable JSON,
short-answer length, or presence of a requested string. It compares
identical task prompts under a baseline and a concise system instruction.
This is a *prompt-only baseline*, **not** model weight editing or a
psychological personality assessment.

The experimental output includes exact prompts, output text, scores,
device, decoding policy, package versions, and the resolved model revision
when Hugging Face supplies one. Results are written as JSONL plus a
summary JSON under \`results/evolution_lab/\` (ignored by git upstream).

~~~sh
python -m evolution_lab run --limit 2 --conditions baseline concise formal
python -m evolution_lab run --repeats 3 --seed 123 --max-new-tokens 64
python -m evolution_lab run --model Qwen/Qwen2.5-0.5B-Instruct --revision COMMIT_SHA
~~~

Use a Hugging Face commit SHA instead of \`main\` for reproducible comparisons.
Treat 8 tasks as a smoke test. Reliable conclusions require much larger,
held-out, balanced test sets, repeated seeds, and confidence intervals.

## Optional: editing model weights with LoRA (experimental v0.2)

This training path is **experimental and not yet tested on a physical MacBook
Neo**. It uses short supervised examples and small rank-4 adapters, with
training and evaluation prompts separated. Run with other programs closed if
memory is limited. If MPS errors or memory limits occur, try CPU with a low
step count or use a machine with more RAM.

~~~sh
python -m pip install -r evolution_lab/requirements-train-mac.txt
python -m evolution_lab train --device auto --steps 4 --eval-limit 2
~~~

The output directory will contain `adapter/`, `summary.json`, and
`comparison.jsonl`. To evaluate an adapter separately:

~~~sh
python -m evolution_lab run --limit 3 --adapter results/evolution_lab/adapters/RUN_ID/adapter
~~~

Substitute the actual RUN_ID printed by the training command. This is a
**weight-changing** LoRA experiment. It does not yet learn how to predict
edits, does not test human personality, and is not proof of improvement.
The before/after metrics are only illustrative until we run larger held-out
test sets, additional seeds, and general capability/locality checks.

## Roadmap

1. **v0.1**: local, objective evaluations and Apple MPS device support.
2. **v0.2**: optional small-model LoRA editing and small held-out
   pre-/post-edit tests. More comprehensive locality tests are pending. Reuse concepts from upstream.
3. **v0.3**: collect weight-delta metadata; compare edits across layers
   and behavior targets using held-out evaluations.
4. **v0.4**: train a predictor of useful low-rank updates; benchmark against
   fixed adapter mixtures and ordinary prompting.
5. **Later**: architecture growth experiments, which require a separate,
   carefully controlled training laboratory.

## Caveats

- MacBook Neo physical-hardware compatibility has **not** yet been verified.
  MPS uses shared/unified memory; other running apps affect memory headroom.
- CUDA-specific upstream scripts should not be invoked on Apple Silicon.
- Avoid synthetic training examples that are inaccurate or unsafe.
- Keep original evaluation tasks out of training data, and distinguish
  targeted edit success from broader capability preservation.
- Model output is untrusted. Never execute code it generates automatically.
