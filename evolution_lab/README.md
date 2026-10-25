# Model Evolution Lab — Apple Silicon experimental path (v0.2.1)

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

**MacBook Neo hardware report (October 25, 2026):** A user tested this
experimental path on an 8 GB Apple Silicon MacBook Neo with Python 3.11.17,
PyTorch 2.14.1, Transformers 4.57.6 and PEFT 0.18.1. The reported
one-step and four-step LoRA runs completed on MPS; the initial inference
run and all 14 unit tests also passed. This is one device report, not a
cross-version compatibility guarantee.

The adapters were trained from short supervised examples at rank 4; training
and evaluation prompts were separate. The two arithmetic tests already passed
before editing and still passed afterwards. **No improvement claim follows
from those results.** Run with other programs closed if memory is limited.
If MPS errors occur, try CPU with a low step count or use more RAM.

~~~sh
python -m pip install -r evolution_lab/requirements-train-mac.txt
python -m evolution_lab train --device auto --steps 4 --eval-limit 2
~~~

The output directory contains `adapter/`, `summary.json`,
`comparison.jsonl` and `delta_stats.json`. The last file records
the Frobenius norm of each effective LoRA weight delta, alongside
layer names. The summary distinguishes improved, regressed, unchanged,
and changed-response outcomes by task and category.

For a more informative follow-up, evaluate all eight held-out tests,
rather than just the first two already-correct arithmetic questions:

~~~sh
python -m evolution_lab train --device auto --steps 4 --eval-limit 8
~~~

Loss values from different training examples are **not a learning curve**;
a large step-to-step swing alone doesn't prove instability.
Do not infer faster generation from first-call versus later-call latency:
initialization and warm-up can dominate first-use timings.

To evaluate an adapter separately:

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

- One MacBook Neo report confirms a small LoRA run works on MPS.
  Larger experiments and version combinations remain unverified.
  MPS uses shared/unified memory; other apps affect memory headroom.
- CUDA-specific upstream scripts should not be invoked on Apple Silicon.
- Avoid synthetic training examples that are inaccurate or unsafe.
- Keep original evaluation tasks out of training data, and distinguish
  targeted edit success from broader capability preservation.
- Model output is untrusted. Never execute code it generates automatically.

## v0.3 controlled diagnostic experiments

The eight original tasks remain smoke tests. `benchmark` validates a separate,
versioned diagnostic with 48 training examples and 92 evaluation items. Each
behavior has 16 train / 8 development / 12 test examples; preservation has 16
items per evaluation split. Content families and normalized prompts cannot cross
splits. These public synthetic items are small, manually authored probes, and
cannot establish broad generalization or rule out semantic contamination.

```sh
python -m evolution_lab benchmark
python -m evolution_lab experiment --behavior json_format --device auto --steps 16
python -m evolution_lab experiment --behavior exact_instruction --device auto --steps 16
python -m evolution_lab experiment --behavior concise_content --device auto --steps 16
```

Each experiment pairs the same tasks and greedy decoding across base, explicit
prompt instruction, zero-initialized LoRA, conventional trained LoRA, and an
untrained random LoRA with matched effective delta Frobenius norm **per layer**.
The prompt condition changes input only; it is not weight editing. Random edits
match rank and norm, not singular spectra or functional impact. Failed integrity
checks stop the experiment. Raw outputs stream to disk; a manifest records
failure or completion, model revision, dataset hashes, package versions, source
commit and dirty state, decoding, training configuration and actual training IDs.
Adapters and per-layer features are saved separately.

`--split test` uses content families that differ from training and development.
Freeze settings on development data before inspecting test results. Do not tune
on a test result and then describe the rerun as independent confirmation. The
`--limit-per-role 2` option is for smoke checks only; it balances target and
preservation counts and records `smoke_only: true`.

Scores separate target behavior from capability preservation. Locality here means
preservation on a finite probe set, not proof that all unrelated behavior remains
unchanged. Reports count gained/lost passes and changed responses, and compare
LoRA directly against prompt and random controls. Wilson intervals assume
independent items and can be too narrow for templated tasks. Paired bootstrap
resamples content families; it returns no interval when fewer than two families
exist. Repeated greedy generations are not independent evidence. Use multiple
training seeds, retain all runs, and avoid selecting an adapter by test score.

Conciseness requires both a word budget and specified content. This is a narrow
style/compliance proxy; it does not assess usefulness, politeness, personality,
or values. JSON grading rejects duplicate keys, NaN/Infinity, and boolean/number
type confusion.

Training now defaults to float32 base weights and float32 adapter parameters.
For lower memory usage, `--precision float16` explicitly opts into a frozen half
precision base; adapter parameters and optimizer state remain float32. Finite
losses, gradients and adapter weights are checked. This reduces avoidable
numerical risks but does not certify MPS reliability. Use a short run on your
Mac before larger studies; record OS, packages, memory pressure and failures.

```sh
# Short hardware check, followed by a complete diagnostic:
python -m evolution_lab experiment --behavior json_format --device auto --steps 1 --limit-per-role 2
python -m evolution_lab experiment --behavior json_format --device auto --steps 16 --seed 42
# Pin the resolved checkpoint SHA from manifest.json for subsequent runs:
python -m evolution_lab experiment --behavior json_format --device auto --steps 16 --seed 43 --revision MODEL_COMMIT_SHA
```

Verification without model downloads:

```sh
python -m unittest discover -s evolution_lab/tests -v
```

The optional tensor tests use a tiny randomly initialized Llama on CPU. They
verify zero-adapter equivalence, finite training updates, frozen base weights,
per-layer random norm matching, state restoration, saved adapter reload, and
assistant-only token masking. They skip when ML packages are unavailable.
They test engineering behavior, not Qwen quality or Mac GPU compatibility.

### Frozen test and auditable hypotheses

```sh
python -m evolution_lab test-checkpoint results/evolution_lab/experiments/RUN_ID --device auto
python -m evolution_lab verify-hypothesis \
  results/evolution_lab/experiments/RUN_ID \
  results/evolution_lab/test/TEST_RUN_ID \
  --output results/evolution_lab/hypothesis.json
```

`test-checkpoint` requires a complete, non-smoke development run and the same
suite fingerprint. It reloads both saved LoRA and random checkpoints at the
immutable recorded base-model revision, uses the original precision and decoding,
and records checkpoint file hashes. There is no further training. Run this after
freezing the training budget, not as part of an adaptive test-score search.

`verify-hypothesis` regrades raw outputs, rejects missing/duplicate observations,
and checks that the test refers to the exact development manifest. Its default
retrospective gate requires at least a 25 percentage-point target gain, strict
target superiority to prompt and random controls, and no locality pass-to-fail
changes. A passing result is named `replicated_on_this_diagnostic`; this is not a
statistical significance test, automatic model promotion, or a learned editing
law. With one seed or one content family, that limitation is substantial. Keep
failed hypotheses and all seed results in the experiment registry as well.
