# Model Evolution Research Guide

This is a cumulative guide to measured relationships between edits and behavior.
It is not a dictionary mapping arbitrary weights to personality traits. Add to it
only with linked raw evidence, and retain failed hypotheses alongside successes.

## Research loop

A base model may propose benign questions, edit hypotheses and candidate edits.
A modified model answers the questions. A deterministic grader, executable
reference solution, trusted answer or independently validated label decides what
is correct. The question proposer and answerer do not establish ground truth by
agreeing with each other.

Correct answers do not automatically earn an edit permanent retention. Compare
with the unedited checkpoint, strong prompts, random edits and ordinary LoRA;
then require held-out target gains and capability preservation. A wrong answer
or unintended regression quarantines the candidate and records a counterexample.
Restore the known checkpoint before exploring the next intervention.

Keep separate pools for training/replay, development/selection and final
confirmation. Inspected final tests are retired before tuning on their failures.
Never turn a final-test error into training data and then call performance on
that same test independent generalization.

Count unique verified questions, model generations, repeated generations,
cache hits, training updates and software tests separately. Millions of repeated
or self-graded answers are not millions of independent experiments. Scale
cumulatively through bounded, checkpointed, deduplicated campaigns that fit the
available hardware and execution session.

## Entry format

For every finding or failure, record:

- Question and falsifiable hypothesis.
- Base checkpoint, exact intervention (modules, layers, rank, direction and norm),
  seed, training pool, optimizer and decoding settings.
- Prompt, random-edit, ordinary-LoRA and nonlearned mixture controls as relevant.
- Development selection rule and independently held-out verification protocol.
- Target score, preservation losses, uncertainty, raw outputs and checkpoint hashes.
- What the result supports, what it does not support, and the next falsification.
- Reproduction command and evidence paths. Preserve unsuccessful candidates.

## Current measured observations

### F001 — JSON-output compliance can be changed through LoRA

Status: observed on this diagnostic; broad benefit not established.

With Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, 16 supervised updates on 16
JSON examples, rank 4, alpha 8, q/v projections and learning rate 0.0005,
three CPU seeds each scored 12/12 on the frozen test JSON family. Their
unedited, explicit-system-prompt and norm-matched random-edit controls scored
0/12. Zero-initialized adapters exactly reproduced base outputs. Training
schemas and test schemas differ; this is still one small public synthetic
family per split.

Supports: in this setup, the trained edit direction mattered beyond the delta
norm, and the changed output is not merely a different inference prompt.
Does not identify which individual layer or weight caused the effect, establish
broad JSON reliability, or show that the model became more intelligent.
Next falsification: independent schema/content families, stronger prompts,
module/layer ablations and a new untouched confirmation pool.

Evidence: `evidence/pilot_v1/study.json`, `evidence/pilot_v1/summary.json` and
trials `00_json_format_42`, `03_json_format_43`, `04_json_format_44`.
Reproduce with `python -m evolution_lab experiment --behavior json_format --steps 16 --seed SEED`,
then `python -m evolution_lab test-checkpoint RUN_DIR` on the saved full development run.

### F002 — A formatting edit can damage unrelated answers

Status: measured failure of the preservation hypothesis.

Those JSON adapters lost 1–2 previously correct preservation items per seed.
The arithmetic answer to 47 + 11 changed from 58 to 68. Seed 42 also changed
Canada's capital from Ottawa to Toronto. Its test preservation score fell from
11/16 to 9/16; the other two seeds reached 10/16. These are substantive errors,
not solely failures to follow a formatting rule.

Supports: target success does not imply locality or capability preservation.
Rejects: promoting these adapters under the current zero-regression gate.
Next intervention: preservation-aware training or replay on a separate pool,
tuned only on development probes, followed by fresh confirmation families.
Do not train on the exposed arithmetic/factual test failures and reuse them as
independent verification.

Evidence: the JSON trial raw `observations.jsonl`, `report.json` and
`hypothesis.json` files. All seven original study trials failed the full gate;
negative results remain in the registry.

### F003 — Concision and exact-output training have uneven effects

Status: small, task-dependent evidence.

The concise-content adapters scored 8/12 versus 6/12 for the base on both CPU
seeds, with no preservation pass-to-fail losses. The gains did not clear the
full development-plus-test gate. The exact-instruction adapters scored 4/12
and 1/12 versus a 3/12 base, showing seed-dependent gains and losses.

These word-budget and required-content probes do not assess usefulness,
politeness, values or personality. Do not name a trait based on a model's own
self-description. Next work needs more independent prompt families and
comparisons with stronger input-only instructions.

Evidence: trials `01_concise_content_42`, `05_concise_content_43`,
`02_exact_instruction_42`, `06_exact_instruction_43`.

### H004 — Sharing instruction-compliance effects may beat naive composition

Status: exploratory development hypothesis; no independent mixture confirmation.

On the development union, the JSON adapter scored 7/8 on JSON and 7/8 on
concise-content probes. The dedicated concise adapter scored 0/8 and 5/8.
The fixed 50/50 mixture scored 7/8 and 5/8, with 10/16 preservation passes.
The mixture did not improve on the JSON source for these probes.

The mixture exactly sums the effective deltas but doubles the rank budget
from 4 to 8. No equally budgeted trained rank-8 baseline or fresh independent
mixture test has been run. This is evidence of possible interaction/interference,
not a learned guide to composing traits.

Evidence: `evidence/pilot_v1/followups/mixture/`.
Next falsification: fixed coefficients and a matched-rank comparator on new
families, with selection confined to development data.

### F005 — Stronger prompting can outperform a targeted adapter

Status: measured retrospective input-only control; one model and diagnostic.

Using all 16 behavior training examples as actual user/assistant demonstration
turns, concise-content test passes rose from 6/12 to 10/12, versus 8/12 for
the trained concise adapters. Preservation remained 11/16. Exact-instruction
prompting remained at 3/12 while preservation reached 12/16. Strict JSON remained
0/12, with Markdown fences violating the output constraint; preservation was
10/16 versus an 11/16 base. Every plain-base reference response was regenerated
and required to match exactly before the historical adapter comparison.

Supports: prompt-only controls must be strong and task-specific; a weight edit
is not automatically the most useful way to request a behavior. Prompt changes
can also affect preservation. Does not establish a general winner, equivalent
compute cost, or a universal limit of prompting. The comparison was designed
after inspecting the initial CPU results and is explicitly retrospective.

Evidence: `evidence/pilot_v1/followups/fewshot_json_format/`,
`fewshot_concise_content/`, and `fewshot_exact_instruction/`.
Reproduce with `python -m evolution_lab fewshot-control TEST_RUN_DIR`.

### F006 — This version completed fresh MPS training and reload checks

Status: direct hardware evidence on one host and package combination.

The 8 GB Apple A18 Pro host (`Mac17,5`, macOS 27.2) completed a full 16-step
float32 MPS JSON experiment and frozen adapter test. Test scores were 0/12 →
12/12 on target probes and 11/16 → 9/16 on preservation probes. A two-step
float16 frozen-base smoke check with float32 adapter parameters also completed.
Losses, gradients and adapter parameters remained finite; zero-adapter and
random-norm integrity checks passed. Saved checkpoints reloaded successfully.

This checks hardware portability after inspecting CPU results; it is not a new
independent scientific holdout. Peak memory was not measured, and two half-base
training steps do not establish long-run numerical stability or compatibility
across OS/package versions. Evidence: `evidence/pilot_v1/mps_validation/`.

The initial evidence census contains 2,092 recorded model responses, 94 unique
evaluation questions and 131 logged Qwen training updates. Software verification
is separate: 43 tests, including four real-tensor tests. These counts are not
millions of questions. The frozen census is `evidence/pilot_v1/counts.json`;
future cumulative totals must deduplicate across studies rather than simply
sum response counts.

## What remains unknown

Effective delta norms are useful descriptors, not causal explanations. Seven
adapters are far too few to fit a credible conditional edit predictor. Future
experiments should isolate q versus v projections, layer groups, rank, norm,
training data and edit direction while keeping the remaining variables fixed.
Confirm promising relations on new behavior/content families and independent
training seeds.

A larger-model editing policy might transfer as a procedure or hypothesis;
specific Qwen LoRA tensors cannot simply be attached to a differently shaped
model. Editing a target such as the envisioned Astra requires legitimate access
to its weights and architecture, compatible edit locations and fresh validation.
API prompting is not weight editing. Architecture growth requires its own
controlled training and preservation study.

## Recurring work

An active schedule returns to this chat every five hours. Each run continues an
unfinished concrete goal or starts the highest-value next research/engineering
goal, inspects existing jobs before loading another model, refreshes GitHub state,
and preserves upstream and unrelated work. It should keep this guide and the
experiment registry current, commit and push verified work to the feature branch,
and retain the PR as draft. This schedule does not guarantee continuous execution
between runs or unlimited compute. Local work requires the host and app to be
available.

The next useful scaling step is a deterministic procedural question generator,
independently executable answer keys, stable split assignment, deduplication and
resumable shards. Establish grader quality and unique-case accounting before
large campaigns. Add causal ablations and new untouched confirmation families
before attempting a hypernetwork or claiming transfer to a large model.

### F007 — Deterministic arithmetic case generation is now auditable

Status: infrastructure verified; no model behavior result.

The `arithmetic-v3` generator creates 100,000 unique addition prompts with
answers computed from structured operands by an independent integer solver.
An affine permutation enumerates each ordered operand pair within each
template/range family without replacement. Four surface templates share each
exact ordered operand pair and one deterministic SHA-256 family split, so the
same calculation cannot cross splits through a paraphrase. Each generated
shard contains only one split. The realized case counts are 70,036 train,
14,984 development, and 14,980 confirmation (four prompts per arithmetic fact
family). Confirmation prompts have not been used to choose or
tune an edit.

Verification: all 47 `evolution_lab` unit/integration tests passed; repeated
generation of the same shard was idempotent; independent shards reconstructed
the same ordered cases as a single generation; all 100,000 normalized prompts
were unique and all answer keys re-solved exactly. Evidence and checksums are
in `evidence/casegen_v3/manifest.json`. These are generated question records,
not model responses, experiments, or evidence of improvement. At the time this
infrastructure result was first recorded, its execution process could not
access MPS and ran no inference. A later direct host check found MPS available;
the separate MPS smoke result is F008.

Reproduce from the repository root:

```sh
python -m evolution_lab generate-cases --seed 42 --start 0 --count 5000 --output evolution_lab/evidence/casegen_v3
python -m evolution_lab generate-cases --seed 42 --start 5000 --count 5000 --output evolution_lab/evidence/casegen_v3
python -m evolution_lab generate-cases --seed 42 --start 10000 --count 90000 --output evolution_lab/evidence/casegen_v3
python -m evolution_lab generate-cases --validate-only --output evolution_lab/evidence/casegen_v3
```

Limits and next falsification: addition-only arithmetic is narrow and synthetic;
surface families are templates with ranges, not proof of semantic independence.
The family split rule is stable but yields variable split ratios. Before model
campaigns, add independently checked operation families and explicit cross-
family adversarial checks; then report prompt-control, random-edit, LoRA and
preservation results using development-only selection and fresh untouched
confirmation families. Do not call this dataset 100,000 tests of a model.

### F008 — New MPS development smoke did not improve the target behavior

Status: direct host observation; development-only, no promotion evidence.

On 2026-10-25, the 8 GB Apple A18 Pro Mac completed a one-update float32 MPS
JSON-format LoRA smoke run on Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775` (Python 3.11.17, PyTorch 2.14.1,
Transformers 4.57.6, PEFT 0.18.1). The two JSON target items remained 0/2 for
base, explicit prompt, zero adapter, trained LoRA, and norm-matched random
adapter. The base passed both arithmetic locality probes. LoRA and random
remained 2/2; the explicit prompt fell to 1/2 after answering 23 + 7 with 20.
The zero adapter reproduced base responses exactly. The one LoRA update had a
finite recorded loss (1.0187) and gradient norm (5.9566); that establishes run
completion, not a useful edit.

Only four unique questions were evaluated, from one JSON family and one
arithmetic family. Uncertainty is wide, and no independent test set was used.
This is a reproducible engineering/hardware observation and a negative target
result: the candidate did not improve JSON compliance, while the prompt control
showed a small-sample locality regression. It does not establish that LoRA
preserves capabilities generally or that prompting is worse. Do not tune or
promote from this run; the candidate remains an unpromoted diagnostic.

Raw observations, reports, training record, LoRA/random checkpoints, delta
features, environment metadata and SHA-256 import index are preserved in
`evidence/pilot_v1/mps_validation/casegen_smoke_20261025/`. The run is tagged
`smoke_only`; its model revision, suite and task hashes are in `manifest.json`.
The attached shell transcript reported 47 passing software tests separately;
those are not model questions. Do not add these four prompts to the 100,000
procedurally generated questions: they are from the existing diagnostic suite.

### F009 — Generated-case campaigns now have an objective paired evaluator

Status: implementation and software verification complete; no generated-case
model responses collected yet.

`python -m evolution_lab case-eval` evaluates fixed generated cases locally and
stores one raw record per case and condition. It grades arithmetic value from
the trusted integer answer key, separately tracks single-integer formatting,
and compares arms on identical question IDs with a paired family bootstrap.
Candidate LoRA runs require the base, explicit prompt, LoRA, and norm-matched
random arms. Adapter files must match the source experiment's recorded hashes,
base model, immutable revision and training settings; random delta norms must
match LoRA per layer within 0.01%. Every run records the dataset and subset
fingerprints, adapter hashes, model revision, prompts, raw outputs, token counts,
latencies, grades and uncertainty summaries.

Runs checkpoint JSONL responses and can resume only when the complete run
specification is identical. The development split is the default. Confirmation
requires explicit `--allow-confirmation`; any consumed confirmation questions
must be retired from subsequent selection. `--offset` plus `--limit` permits
bounded slices in stable split order so campaigns can add generations in
checkpointed stages instead of loading all possible cases at once. Repeated
responses count as generations, while the distinct case IDs count as unique
questions.

Verification: 54 software tests passed, including mocked interruption/resume,
duplicate-safe ledger behavior, split selection, exact answer grading,
per-layer norm-match rejection, and adapter/source-hash checks. The 100,000-case
dataset still validates with 70,036 train, 14,984 dev, and 14,980 confirmation
questions. No model generations were made through this evaluator. MPS is
available on the direct Mac host, but the separate fixed MPS reproduction study
was still marked running at the time of this implementation, so this work did
not launch a competing model job. MPS availability does not itself verify the
new evaluator's model-loading path.

Example, using the pinned Qwen revision and a bounded development slice:

```sh
python -m evolution_lab case-eval evolution_lab/evidence/casegen_v3 \
  --split dev --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --arms base prompt --limit 32 --output-dir results/evolution_lab/case-dev-000
```

For a trained adapter, provide the paired adapter directories and the two
effective-delta JSON files from the same complete experiment, and select all
four control arms. Resume an interrupted campaign by repeating its exact
command with `--resume`.

Limits and next falsification: answer extraction currently grades integer
arithmetic only; it is not a broad reasoning measure. The system still needs a
small real-model development-slice run after the active fixed study completes,
then an independently authored task family and fresh confirmation before any
weight-edit conclusion. Millions of generated records are not millions of
model generations, and no case-evaluation campaign result is yet in evidence.
