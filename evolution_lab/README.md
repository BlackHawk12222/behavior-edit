# Model Evolution Lab — Apple Silicon experimental path (v0.4.0)

[Plain-language overview](PROJECT_OVERVIEW.md) · [Research guide](RESEARCH_GUIDE.md) · [Measured results](reports/pilot-v1.md) · [Experiment registry](experiment_registry.json)

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

You can force CPU with `--device cpu` if MPS is unavailable. Test first
with `--limit 1` if you're uncertain about free memory or disk space.

The default is an open-weight Hugging Face model
(`Qwen/Qwen2.5-0.5B-Instruct`). Its model card and license are separate
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
summary JSON under `results/evolution_lab/` (ignored by git upstream).

~~~sh
python -m evolution_lab run --limit 2 --conditions baseline concise formal
python -m evolution_lab run --repeats 3 --seed 123 --max-new-tokens 64
python -m evolution_lab run --model Qwen/Qwen2.5-0.5B-Instruct --revision COMMIT_SHA
~~~

## Generate trusted, resumable question shards

The arithmetic case generator creates deterministic addition prompts with
integer answer keys, stable family-level splits, duplicate detection, and
separate train/dev/confirmation shard files. Existing identical shards can be
reused safely; validate the checksums and cumulative unique-case count before
starting model runs.

~~~sh
python -m evolution_lab generate-cases --seed 42 --start 0 --count 5000 --output results/evolution_lab/cases
python -m evolution_lab generate-cases --seed 42 --start 5000 --count 5000 --output results/evolution_lab/cases
python -m evolution_lab generate-cases --validate-only --output results/evolution_lab/cases
~~~

Generated cases are not model tests until responses are actually collected and
objective-graded. Keep confirmation shards untouched while choosing edits, and
report model generations separately from unique prompts, repeats, cache hits,
and software tests. See the [research guide](RESEARCH_GUIDE.md) for the first
100,000-case infrastructure census and its limits.

To collect paired local-model answers, use a development split and an immutable
model revision. The runner records every raw answer, separates arithmetic
correctness from integer-only formatting, checkpoints its JSONL ledger, and
resumes only when the dataset, model, arms, prompts, and adapter hashes match.
An adapter comparison requires base, explicit prompt, LoRA, and norm-matched
random arms in the same run. Confirmation requires a separate explicit flag;
once inspected, those cases must be retired from future tuning.
An exclusive campaign lock prevents concurrent runs from duplicating model
generations in the same output directory.

~~~sh
python -m evolution_lab case-eval evolution_lab/evidence/casegen_v3 \
  --split dev --revision MODEL_COMMIT_SHA --arms base prompt \
  --limit 32 --output-dir results/evolution_lab/case-campaign-dev
~~~

For an interrupted run, repeat the same command with `--resume`. Add
`--adapter PATH/adapter_lora --random-adapter PATH/adapter_random \
--adapter-delta PATH/delta_lora.json --random-delta PATH/delta_random.json \
--arms base prompt adapter random` to evaluate candidate weights against the
same questions and matched controls. The adapter arms must come from the same
complete experiment, match the requested base revision, and have per-layer
effective delta norms within 0.01%.
The effective deltas are recomputed from the loaded adapter tensors and checked
against the supplied sidecars before any generation starts.
The runner never trains or self-grades; candidate promotion still needs
independent confirmation and capability/locality checks.

When only the adapter changes across campaigns, an exact baseline cache can
reuse base and prompt answers from a completed compatible run:

```sh
python -m evolution_lab case-eval evolution_lab/evidence/completion_v3 \
  --split dev --revision MODEL_COMMIT_SHA \
  --arms base prompt adapter random --adapter ADAPTER_PATH \
  --random-adapter RANDOM_ADAPTER_PATH --adapter-delta DELTA_JSON \
  --random-delta RANDOM_DELTA_JSON \
  --baseline-cache-from results/evolution_lab/prior-compatible-campaign \
  --output-dir results/evolution_lab/new-adapter-campaign
```

Reuse is allowed only when the dataset, selected questions, grader, prompts,
model revision, requested and resolved device/precision, and decoding match.
The source ledger, report and run specification are checksum-verified. Imported
base/prompt responses remain in the target ledger with source provenance and
are counted as `cache_hits` and `response_records`, not new `model_generations`.
This saves repeated inference; it does not create new evidence or new questions.

Use a Hugging Face commit SHA instead of `main` for reproducible comparisons.
Treat 8 tasks as a smoke test. Reliable conclusions require much larger,
held-out, balanced test sets, repeated seeds, and confidence intervals.

## Phrase and sentence completion

The completion generator adds common phrase completions and subject–verb
agreement sentences. It creates four prompt surfaces per underlying family,
keeps each family wholly within train, development, or confirmation, and uses
curated phrase keys or explicit verb forms as trusted answers. The v1 inventory
contains 672 unique prompts; its development and confirmation pools each have
104 prompts, including both phrase and sentence tasks.

~~~sh
python -m evolution_lab generate-completions --seed 42 --start 0 --count 672 --output results/evolution_lab/completions
python -m evolution_lab generate-completions --validate-only --output results/evolution_lab/completions
python -m evolution_lab case-eval results/evolution_lab/completions \
  --split dev --revision MODEL_COMMIT_SHA --arms base prompt \
  --limit 104 --output-dir results/evolution_lab/completion-dev
~~~

The grader checks which choices appear in the response, so a complete
sentence can receive credit, while a response mentioning both choices is
ambiguous. It scores strict one-word formatting separately. These probes assess
English collocation recall and basic subject–verb agreement; they are small,
culturally dependent benchmarks and do not measure personality.

## Optional: editing model weights with LoRA (experimental v0.2)

**Earlier user-reported MacBook Neo hardware evidence:** A user tested this
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
python -m evolution_lab run --limit 3 --precision float32 --adapter results/evolution_lab/adapters/RUN_ID/adapter
~~~

Substitute the actual RUN_ID printed by the training command. This is a
**weight-changing** LoRA experiment. It does not yet learn how to predict
edits, does not test human personality, and is not proof of improvement.
The before/after metrics are only illustrative until we run larger held-out
test sets, additional seeds, and general capability/locality checks.

## Roadmap

1. **v0.1**: local, objective evaluations and Apple MPS device support.
2. **v0.2**: optional small-model LoRA editing and small held-out
   pre-/post-edit smoke tests.
3. **v0.3**: controlled train/dev/test diagnostics, norm-matched random edits,
   frozen-checkpoint tests, per-layer features, fixed studies and exact adapter mixtures.
4. **Future, conditional on evidence**: collect a larger adapter/behavior catalog,
   then test simple conditional selection before considering an edit predictor.
   Compare against fixed mixtures, random edits, conventional LoRA and prompting.
5. **Later**: architecture growth experiments, which require a separate,
   carefully controlled training laboratory.

## Caveats

- A prior user report describes successful small LoRA runs on MPS.
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

### Fixed studies and measured feature catalogs

```sh
python -m evolution_lab study evolution_lab/studies/pilot_v1.json \
  --device auto --output-dir results/evolution_lab/my_pilot
python -m evolution_lab catalog results/evolution_lab/my_pilot \
  --output-dir results/evolution_lab/my_pilot/catalog
```

The study command runs a fixed seed/behavior matrix in separate sequential child
processes so each model is released before the next loads. It journals each phase,
keeps complete and failed artifacts, and resumes with the exact same specification.
It refuses a changed specification or incompatible cached development checkpoint.
The optional `reuse_dev` entries are local cache hints from the original session;
when those directories are absent, the same trial is trained from scratch.
The seven-trial pilot requires many model evaluations and is substantially longer
than a smoke check. Start with one experiment to verify your hardware first.

`catalog` requires a complete study and reaudits each run. It exports one row per
trained adapter containing settings, layer delta features, and separated dev/test
outcomes. Test outcomes are verification labels, **never** training labels for a
future edit predictor. Seeds share tasks, so results are reported per seed rather
than pooled into a misleadingly large test sample. A few rank-4 adapters are not
enough data for a credible hypernetwork; a predictor would need more behaviors,
more independent families, nested selection and a fresh final holdout.

Completed adapters now receive file hashes at the end of development; frozen
tests reject changed checkpoint files. Early exploratory runs made before this
integrity field existed are explicitly marked as legacy when evaluated by the
new command. Hashes establish artifact identity, not causal validity by themselves.

### Simple adapter mixture baseline

```sh
python -m evolution_lab mixture \
  results/evolution_lab/experiments/JSON_RUN_ID \
  results/evolution_lab/experiments/CONCISE_RUN_ID --device auto
```

This combines two compatible saved adapters with fixed coefficients 0.5 / 0.5.
PEFT concatenation constructs exactly `0.5 * delta_1 + 0.5 * delta_2`, verified by
a tensor integration test. It evaluates the base, each constituent adapter and
the mixture on the union of their development targets and preservation probes.
Per-behavior scores expose interference that a single aggregate can hide.

Concatenation uses the **sum of the source ranks** (two rank-4 adapters give
rank 8). It is a nonlearned composition diagnostic, not evidence that mixing
beats an equally budgeted trained adapter. Coefficients are not fitted, no
architecture grows, and there is no hypernetwork. An independently held-out
mixture evaluation and rank-budget-matched baseline would be required before
any stronger claim. See the official [PEFT model-merging guide](https://huggingface.co/docs/peft/developer_guides/model_merging)
for the underlying adapter-combination API.

### Stronger prompt control with training demonstrations

```sh
python -m evolution_lab fewshot-control results/evolution_lab/test/TEST_RUN_ID --device auto
```

This retrospective follow-up uses all 16 training examples for the adapter's
behavior as actual user/assistant demonstration turns, alongside the explicit
behavior instruction. Evaluation answers are never included. It uses the saved
experiment's checkpoint, precision, tasks and decoding budget, and requires the
same device type. Before each comparison it regenerates the plain baseline and
requires an exact match with the recorded output; a mismatch stops the run.
The report contrasts LoRA against this stronger input-only control. Longer
context costs are not equivalent to training or inference compute budgets, so
these are behavioral comparisons, not cost/latency efficiency claims.

For standalone `run --adapter`, match the original `--precision` and `--revision`; `run` defaults to automatic inference precision, while training defaults to float32. The frozen-checkpoint command enforces the recorded precision automatically.

The first phrase/sentence development and confirmation comparisons also show
why completion needs separate correctness and format metrics. On the untouched
confirmation families, the terse one-word prompt raised strict formatting from
4/104 to 91/104 while correctness fell from 85/104 to 73/104 (22 regressions,
10 improvements; family-bootstrap interval −23.1 to −1.9 percentage points).
This was a prompt-only result, not a weight edit. The confirmation split is now
retired; see F013 and the checksummed raw evidence in
`evidence/completion_v1/evaluations/confirmation_n104_base_prompt/`.

### Train completion LoRA and fair random-edit controls

`train-cases` reads only the generated dataset's `train` split, trains rank-4
LoRA by default, saves the trained adapter and an independently randomized
adapter whose effective delta norm matches the LoRA delta in every adapted
module. Its manifest records the exact training cases, data hash, model revision,
settings, checkpoint hashes and training history. Use the already-consumed v1
development families only for selection; they are not an independent final test.

```sh
python -m evolution_lab train-cases evolution_lab/evidence/completion_v1 \
  --model MODEL_SNAPSHOT_PATH \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --device cpu --precision float32 --steps 64 --rank 4 --seed 42 \
  --output-dir results/evolution_lab/case_training

python -m evolution_lab case-eval evolution_lab/evidence/completion_v1 \
  --split dev --model MODEL_SNAPSHOT_PATH \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --device cpu --precision float32 --arms base prompt adapter random \
  --adapter results/evolution_lab/case_training/RUN_ID/adapter_lora \
  --random-adapter results/evolution_lab/case_training/RUN_ID/adapter_random \
  --adapter-delta results/evolution_lab/case_training/RUN_ID/delta_lora.json \
  --random-delta results/evolution_lab/case_training/RUN_ID/delta_random.json \
  --limit 104 --output-dir results/evolution_lab/completion-lora-dev
```

The training tool does not evaluate or promote the adapter. The paired evaluator
requires base, prompt, LoRA and norm-matched-random arms together, and verifies
both source checkpoint hashes and the effective delta reports. Results on this
benchmark remain developmental because its prior confirmation families have
already been inspected. Build new lexical and grammatical families before
making any held-out generalization claim.

The first controlled completion LoRA is preserved as a development experiment,
not a recommended adapter. It scored 93/104 on the v1 completion development
set (80/80 sentence agreement) against 78/104 base and 81/104 norm-matched
random, but phrase completion fell to 13/24 and arithmetic locality fell from
44/64 to 30/64. The adapter is quarantined; its confirmation families were not
reused. See F014 and the complete training/checkpoint/response evidence under
`evidence/completion_v1/evaluations/lora_seed42_steps64_dev/`.


### New-vocabulary phrase and sentence completion benchmark

`generate-completions-v2` adds 2,208 deterministic cases across separate
phrase and sentence-agreement families with family-level train, dev and
confirmation splits. Its phrase families, subject nouns and verb roots are
disjoint from v1. Trusted answer keys are generated from the phrase table and
grammatical number; the model never supplies its own grade.

```sh
python -m evolution_lab generate-completions-v2 --seed 43 --start 0 --count 2208 \
  --output evolution_lab/evidence/completion_v2
python -m evolution_lab generate-completions-v2 --validate-only \
  --output evolution_lab/evidence/completion_v2
```

The seed-43 confirmation evaluation is spent. Its first prompt-control run
accidentally used an arithmetic-only system instruction. The evaluator now
selects the task-matched completion prompt for all completion generator
versions, with a regression test. The corrected prompt was rerun on the same
confirmation set and is explicitly post-hoc, so it is not a fresh independent
confirmation. The frozen v1 LoRA reached 289/336 overall but only 12/24 phrase
items; corrected prompting reached 219/336 and 18/24 phrase items. The LoRA
also caused a separate arithmetic locality regression and remains quarantined.
See research finding F015 and the complete checksummed evidence in
`evidence/completion_v2/evaluations/new_vocabulary_confirmation_spent/`.

For any fresh experiment, generate a new seed or new inventory and freeze the
prompt-control wording before consuming confirmation. Use `case-eval` with
`--arms base prompt adapter random` and `--allow-confirmation` only once. A
repeated prompt or cached response is not a new unique question; keep the number
of generations separate from unique prompts and software tests.

### Phrase and sentence completion with unseen agreement attractors (v3)

`generate-completions-v3` adds a separate 8,740-case, trusted-key dataset with
49 phrase families and long-distance subject–verb agreement sentences. In each
sentence, a singular or plural noun sits between the grammatical subject and
the missing verb, testing whether the model follows the subject rather than
the nearby distractor. The 32 subject lexemes and 32 verb lexemes are assigned
to disjoint train, development and confirmation inventories; phrase families
are also held out whole. The confirmation split has 32 phrase questions across
8 families and 400 sentence questions across 25 families. The phrase estimate
will therefore be imprecise; report it separately from sentence agreement.

The checked-in seed-42 shards contain 7,880 training, 428 development and 432
confirmation questions. Validate or regenerate them with:

```sh
python -m evolution_lab generate-completions-v3 --seed 42 \
  --output evolution_lab/evidence/completion_v3
python -m evolution_lab generate-completions-v3 --validate-only \
  --output evolution_lab/evidence/completion_v3
```

Train only on the training split, then select controls on development. Freeze
the model revision, candidate, prompt and analysis before using confirmation:

```sh
python -m evolution_lab train-cases evolution_lab/evidence/completion_v3 \
  --revision MODEL_COMMIT_SHA --seed 42 --steps 64 \
  --output-dir results/evolution_lab/completion_v3_training
python -m evolution_lab case-eval evolution_lab/evidence/completion_v3 \
  --split dev --revision MODEL_COMMIT_SHA \
  --arms base prompt adapter random --adapter ADAPTER_PATH \
  --random-adapter RANDOM_ADAPTER_PATH --adapter-delta DELTA_JSON \
  --random-delta RANDOM_DELTA_JSON \
  --output-dir results/evolution_lab/completion_v3_dev
```

An exploratory seed-42 balanced LoRA development run generated 1,712 responses
on 428 unique questions. The adapter scored 276/428 (64.5%), versus 264/428
(61.7%) for explicit prompting and 227/428 (53.0%) for both base and the
norm-matched random adapter. Adapter versus prompt was +2.8 points with a paired
family-bootstrap 95% interval of −1.9 to +7.7, so it did not establish a
benefit over prompting. Phrase scores were 19/28 for adapter and prompt versus
23/28 for base. Sentence scores were 257/400 for adapter, 245/400 for prompt,
and 204/400 for base. A post-hoc diagnostic found the adapter answered all
200 plural-subject sentence probes correctly but only 57/200 singular-subject
probes; this is a strong agreement-direction bias, not a general completion
gain. The seed-42 candidate is quarantined. The confirmation split remains
untouched because this candidate did not clear the prompt-control gate.

For a task-exposure experiment, freeze one family-safe cohort first and use
`--sampling-strategy task-ratio`. `--seed` controls model initialization and
the within-task example streams. The optional `--schedule-seed` controls where
each task appears among optimizer updates; for the two-task case, phrase-update
positions are nested across ratios with the same schedule seed. This lets
paired treatments share both example streams and common randomized update
positions. Omitting `--schedule-seed` preserves the historical seeded-shuffle
schedule used by older runs. The actual task, example ID and loss for every
update are saved in the training trace.

```sh
python -m evolution_lab create-training-cohort evolution_lab/evidence/completion_v3 \
  --phrase-count 32 --sentence-count 32 --seed 25025 \
  --output results/evolution_lab/fixed_support_cohort
python -m evolution_lab train-cases evolution_lab/evidence/completion_v3 \
  --revision MODEL_COMMIT_SHA --seed 42 --schedule-seed 901 --steps 64 \
  --sampling-strategy task-ratio \
  --cohort-manifest results/evolution_lab/fixed_support_cohort/manifest.json \
  --task-exposure-ratio phrase_completion=16,sentence_completion=48 \
  --output-dir results/evolution_lab/task_ratio_16_48_seed42_order901
```

This controls which examples are available to each treatment, but replacement
sampling can still repeat examples. Report unique examples actually seen and
per-example exposure counts; an update ratio is not an example-count ratio.
Treat results on an already inspected development split as exploratory and
create fresh confirmation families before making generalization claims.

### Planned follow-up: isolate task-update order

F025 changed update allocation and order together. Before making any causal
claim about timing, hold the hash-bound 32-phrase/32-sentence support set, the
16-phrase/48-sentence update ratio, model and schedule seeds, optimizer, number
of steps, and decoding fixed except for a preregistered schedule-seed factor.
Use the same seed-specific within-task example streams across schedule seeds,
two model seeds, and at least three schedule seeds. Keep all six adapters,
training traces and raw controls. First run on the already inspected v3 dev
set as an order-sensitivity diagnostic only; it adds zero unique questions and
cannot confirm generalization. Reuse F025 base/prompt rows only after the
cache's exact inference and actual-device checks pass, and account for those
rows as cache hits. Score phrases, sentences, singular/plural subjects, and
separate arithmetic preservation; inspect every pass-to-fail counterexample.
The seven phrase families make phrase-specific uncertainty weak. Do not train
on exposed dev failures or tune against the retired confirmation sets. A
separate candidate confirmation would require newly generated, family-disjoint
questions, broader capability checks, and the same strong prompt and matched
random-edit controls.

This first comparison was exploratory rather than preregistered and used one
training seed. Its raw responses, controls, LoRA checkpoints, effective delta
stats, report and checksums are in
`evidence/completion_v3/evaluations/seed42_balanced_dev_f020/`. Passing the
generator checks alone only shows reproducible questions and trusted keys; it
does not show that an adapter improves completion behavior.


F022 repeated the task-balanced training with seed 43 on the same development
questions. The adapter again answered all 200 plural-subject sentences
correctly but only 70/200 singular-subject ones. Its training sample had 21
singular and 11 plural subjects, so raw answer-class counts alone do not explain
the repeated bias. It scored 284/428 overall versus 264/428 for the prompt;
the family-bootstrap interval for the difference included zero. It scored
14/28 on phrases, below both base (23/28) and prompt (19/28). F022 adds 1,712
responses but zero new questions; these results are not independent
confirmation. Evidence is in
`evidence/completion_v3/evaluations/seed43_balanced_dev_f022/`.

F023 is a preregistered, same-seed comparison of the full training schedules.
The full training pool contains 136 phrase examples among 7,880 (1.73%).
Uniform sampling selected 1 phrase and 63 sentence examples in 64 updates;
task-balanced sampling selected 32 of each, with no overlap in the exact 64
examples selected. On the same already-inspected development questions, the
uniform adapter scored 376/428, versus 264/428 prompt, 227/428 base and
227/428 norm-matched random. Its paired family-bootstrap margin over prompt
was +26.2 points (95% interval +20.8 to +31.3). Sentence score was 358/400,
including 159/200 singular-subject cases; phrase accuracy was 18/28, below
base's 23/28. This is a strong but exposed development result, not independent
confirmation. The full schedule changed both task mix and exact training
examples, so phrase frequency alone cannot be credited. The adapter remains
quarantined, and the 432-question confirmation split is untouched. All 1,712
responses are repeated generations on 428 old questions, not new questions.
Raw outputs, training sample IDs, adapters, delta reports and hashes are in
`evidence/completion_v3/evaluations/seed42_uniform_dev_f023/`.

### Build a fixed mixture from case-trained adapters

`case-mixture` combines two complete case-training runs at fixed 0.5/0.5
coefficients. It rejects changed checkpoint hashes or mismatched base revision,
dataset, seed, or training settings; it records overlap between exact sampled
training IDs. It saves the rank-summed mixture and a per-layer norm-matched
random control, both with hashes and effective-delta summaries.

```sh
python -m evolution_lab case-mixture \
  results/evolution_lab/completion_v3_training/BALANCED_RUN \
  results/evolution_lab/completion_v3_training_f023/UNIFORM_RUN \
  --device cpu --seed 20261025
```

The mixture is a nonlearned comparator. Two rank-4 sources produce a rank-8
adapter, so compare it with the rank-4 candidates as a separate mixture
baseline, not as an equal-rank training result. See `case-mixture --help` and
F024's preregistration for the full held-out comparison.

### Compare training-sample balance

The v2 training pool has far more sentence-agreement examples than phrase
examples. `train-cases` defaults to its historical uniform shuffle. For a
controlled task-mixture ablation, `--sampling-strategy task-balanced` cycles
evenly through task types while shuffling examples within each type. Keep the
same seed, update count, rank, optimizer, learning rate, and max length when
comparing the schedules. The training manifest records the chosen strategy.
This changes which examples drive gradients; it does not identify causal
weights or guarantee better preservation.

```sh
python -m evolution_lab train-cases evolution_lab/evidence/completion_v2 \
  --model MODEL_SNAPSHOT_PATH \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --device cpu --precision float32 --steps 64 --rank 4 --seed 43 \
  --sampling-strategy task-balanced \
  --output-dir results/evolution_lab/completion_v2_balanced
```


The first seed-43 sampling ablation is recorded in F016 and
`evidence/completion_v2/evaluations/sampler_ablation_seed43_dev/`. Uniform
sampling selected 2 phrase and 62 sentence cases; task-balanced sampling
selected 32 of each. On 328 development items, their LoRAs scored 276/328 and
270/328, respectively. Task-balanced phrase accuracy was 14/24 versus 18/24
for uniform; the paired interval is wide and includes zero, so this does not
establish that balancing hurts, but it does not support the proposed fix. Both
adapters beat the explicit prompt on aggregate completion, and neither showed
a clear arithmetic regression on a separate 64-question development slice.
These are development results from one seed; v2 confirmation remains untested
by this ablation and is already retired from F015.

The seed-44 replication and separate arithmetic locality evaluation are in
`evidence/completion_v2/evaluations/sampler_ablation_seed44_dev/`. On the same
328 completion-development questions, seed-44 uniform scored 301/328
(phrase 15/24; sentence 286/304) and task-balanced scored 289/328
(phrase 13/24; sentence 276/304). Their direct paired family-bootstrap
difference was −3.7 percentage points (95% interval −8.8 to +0.9). The seed-44
uniform adapter exceeded seed-43 uniform by 7.6 points on these already-used
development questions (interval +2.7 to +12.8), a warning about training-seed
sensitivity rather than independent confirmation.

On 64 arithmetic-development questions, seed-44 uniform scored 38/64, matching
base; task-balanced scored 61/64 versus 38/64 base, 42/64 prompt and 37/64
norm-matched random (paired balanced-minus-uniform interval +26.6 to +46.9).
Treat that surprising result as a development signal requiring fresh
arithmetic confirmation and replication. Neither adapter is promoted, and the
24 phrase questions across six families remain too few for a stable phrase
claim. The seed-44 run added 3,136 generations on 392 unique questions and
128 training updates; repeated outputs do not increase the unique-question
count.

### Ablate adapter components

`ablate-adapter` makes reproducible variants by zeroing selected q/v projection
or layer-half LoRA components in both the learned and norm-matched random
adapter. It recomputes effective-delta reports and records provenance; it does
not imply that a layer has one behavior or that a result transfers between
models.

```sh
python -m evolution_lab ablate-adapter results/evolution_lab/train/RUN_ID \
  --ablation zero_v_proj \
  --output-dir results/evolution_lab/ablation/zero_v_proj
```

F018 evaluates four preregistered component removals on phrase and sentence
completion plus arithmetic locality. Removing upper-half edits reduced this
checkpoint's sentence score on exposed development data. Removing v-projection
edits reduced the arithmetic score. These are checkpoint-specific development
signals; no candidate is promoted. The 24 phrase questions span only six
families. F019 later evaluated one frozen adapter on a preregistered 32-family
arithmetic confirmation sample. Those families are retired, and no confirmation
data may be used for tuning. See
the [research guide](RESEARCH_GUIDE.md#f018--component-ablations-identify-checkpoint-specific-dependencies-with-a-v-projection-and-upper-layer-signal)
and checksummed [F018 evidence](evidence/completion_v2/evaluations/adapter_component_ablation_seed44_dev/).

### Select reproducible whole-family evaluation samples

When evaluating a bounded subset, use `--family-limit` and `--family-seed` to
select complete families by stable SHA-256 rank. This keeps all paraphrases of
a trusted fact together and records the selected family IDs in `run.json`.
Family sampling cannot be combined with case-level `--limit` or `--offset`.

F019 used this method to check the frozen seed-44 completion adapter on 32
previously untested arithmetic confirmation families (128 prompts). It scored
107/128, versus 78/128 for the task-matched prompt and 53/128 for both base and
norm-matched random; adapter-minus-prompt was +22.7 points (paired 95% family
interval +14.8 to +29.7). The development score was higher at 61/64, so the
confirmation result supports a narrow arithmetic effect while showing a
material development-to-confirmation gap. This does not confirm phrase or
sentence behavior, and the selected arithmetic families are retired. See the
[research guide](RESEARCH_GUIDE.md#f019--a-completion-trained-adapter-generalized-its-arithmetic-signal-to-fresh-confirmation-families-with-a-development-to-confirmation-drop)
and [F019 evidence](evidence/casegen_v3/evaluations/seed44_balanced_arithmetic_confirmation_f019/).

### F024 — Frozen text-behavior confirmation and arithmetic check

The F023 uniform adapter was frozen before testing 432 untouched phrase and
sentence questions, then compared with base, the explicit prompt, a
per-layer norm-matched random edit, and a fixed equal mixture of the balanced
and uniform adapters. The candidate scored 409/432 overall and 390/400 on
sentences, versus 264/432 and 249/400 for the prompt. Its paired family
bootstrap margin over prompt was +33.6 percentage points (95% interval +28.7
to +37.7). Phrase score was 19/32, below base at 26/32, so the finding is
specific to sentence completion and comes with a phrase trade-off. The mixture
scored 388/432; the candidate exceeded it by 4.9 points (95% interval +2.1 to
+8.1), but the rank-8 mixture is not rank matched to the rank-4 candidate.

On 312 fresh addition questions, the candidate scored 99/312, above base and
matched-random controls at 54/312 but below the explicit prompt at 155/312 by
17.9 points (paired family-bootstrap 95% interval −23.4 to −12.8). This is a
narrow capability check, not broad preservation. No adapter is promoted.
F024 records 4,704 model generations over 744 unique questions. Its raw
responses, reports, controls, preregistration amendments, and checksums are in
`evidence/f024_frozen_confirmation/`; those confirmation families are retired.
See the [full research record](RESEARCH_GUIDE.md#f024--frozen-candidate-confirmation-and-arithmetic-preservation).
