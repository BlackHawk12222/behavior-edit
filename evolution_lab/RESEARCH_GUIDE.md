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
