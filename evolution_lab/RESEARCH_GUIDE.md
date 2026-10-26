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

### Treat training data as an experimental variable

An adapter's behavior depends on which examples it sees, how often each kind
appears, how examples are phrased, and which families are held out. Record the
exact sampled example IDs and their split, family, task type, answer class,
subject number, distractor number, and lexical family where applicable. Report
both the full training-set distribution and the realized examples used at each
update. Check exact duplicates and near-duplicate templates across splits.

When studying data effects, keep the model revision, LoRA initialization seed,
optimizer, update count, prompt, decoding and evaluation questions fixed while
changing one data factor at a time: for example, uniform versus task-balanced
sampling, singular/plural proportions, or phrase-family inclusion. If several
factors change together, describe the result as a combined treatment rather
than crediting one data property. Do not infer that a near-balanced update
count means the semantic content or phrase families were balanced.

## Entry format

For every finding or failure, record:

- Question and falsifiable hypothesis.
- Exact training data source, sampled IDs, family/answer-class distribution,
  duplicates, and sampling schedule.
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

Deterministic arithmetic and phrase/sentence generators now provide trusted keys,
stable family splits, deduplication and resumable shards. The next scaling step
is to enlarge the lexical and grammatical inventory, train a controlled LoRA on
training families, and compare it with strong prompt, norm-matched random, and
fixed-mixture controls on development families. Keep a new confirmation family
pool untouched until the protocol is frozen. Add causal ablations before
attempting a hypernetwork or claiming transfer to a large model.

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

For candidate runs, the evaluator recomputes effective delta norms from the
loaded PEFT tensors and checks each checkpoint against its recorded sidecar
before generating answers. Base and prompt arms run with adapters disabled on
the same loaded model; LoRA and random arms activate their named adapters.

Runs checkpoint JSONL responses with completion checksums and can resume only
when the complete run specification is identical, including an explicit answer
grader version. An exclusive process lock rejects simultaneous
campaigns on the same output directory. The development split is the default. Confirmation
requires explicit `--allow-confirmation`; any consumed confirmation questions
must be retired from subsequent selection. `--offset` plus `--limit` permits
bounded slices in stable split order so campaigns can add generations in
checkpointed stages instead of loading all possible cases at once. Repeated
responses count as generations, while distinct case IDs count as unique
questions and normalized prompt hashes support cross-run prompt deduplication.

Verification: 56 software tests passed, including mocked interruption/resume,
duplicate-safe ledger behavior, split selection, exact answer grading,
per-layer norm-match rejection, loaded-delta sidecar checks, and adapter/source-hash checks. The 100,000-case
dataset still validates with 70,036 train, 14,984 dev, and 14,980 confirmation
questions. A later bounded 64-question CPU run verified the evaluator against
the locally cached pinned model snapshot; results are recorded in F011. The
initial model-ID load attempt failed before inference because network DNS was
unavailable. MPS availability in a separate run does not verify this evaluator's
MPS path; in this coding process MPS reports unavailable.

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
small real-model development-slice run, then an independently authored task
family and fresh confirmation before any weight-edit conclusion. The direct
Mac study below verifies its own MPS training/reload path, but does not exercise
this new evaluator. In the current coding process, `torch.backends.mps.is_available()`
reports false, so a new GPU model campaign has not been started here. Millions
of generated records are not millions of model generations.

### F010 — The fixed seven-trial study completed on Apple Silicon MPS

Status: hardware reproducibility evidence on the same small, already exposed
pilot suite; no independent generalization result.

The user ran the fixed `pilot_v1.json` matrix to completion on the 8 GB Apple
Silicon host using Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, float32 MPS, and the recorded
Python/PyTorch/Transformers/PEFT versions. The matrix has seven behavior/seed
trials. JSON LoRA reached 12/12 target items for each of three seeds, while
the matched random adapters stayed at 0/12; locality fell from 11/16 to 9/16
for seed 42 and to 10/16 for seeds 43 and 44. Concise-content LoRA scored
8/12 versus base 6/12 for both seeds, with no measured locality regressions;
the prompt control scored 7/12 and random scored 6/12 and 5/12. Exact-instruction
LoRA was seed-sensitive: 4/12 versus base 3/12 for seed 42 and 1/12 versus
base 3/12 for seed 43. These small scores do not support a general personality,
capability, or intelligence claim.

The study source declared that JSON seed 42 and concise seed 42 would reuse
development artifacts. The import records 1,624 response rows across train/dev
and test conditions, 92 unique normalized prompts, 112 optimizer updates in
the copied artifacts, and 80 new updates across the five non-reused trials.
Rows and cached repetitions are not independent questions. The test prompts
were already inspected in pilot-v1 and are retired from future independent
confirmation. The study is useful to check hardware completion, checkpoint
reload, and directional reproducibility; it does not replace new families,
matched controls, or the full promotion gate.

Evidence, including source study journal, all trial outputs/checkpoints, and
SHA-256 index: `evidence/pilot_v1/mps_validation/reproduction_20261025/`.
Reproduce on a compatible Mac with:

```sh
python -m evolution_lab study evolution_lab/studies/pilot_v1.json \
  --device auto --output-dir results/evolution_lab/reproduction
```

Limits and next falsification: preserve this result as repeated hardware
evidence, then run the paired generated-case evaluator on a bounded development
slice when MPS is available to that process. Any promising edit still needs
fresh behavior families, strong prompt/random/LoRA controls, and unconsumed
confirmation. Qwen adapter tensors and these results do not transfer directly
to Astra or another architecture.

### F011 — Arithmetic grading and prompt effects were checked on a development slice

Status: completed CPU integration run on development data; no weight edit and
no generalization claim. Its first report was superseded after a grading bug
was discovered and corrected.

Using the pinned Qwen2.5-0.5B-Instruct snapshot, the evaluator compared the
base model with a direct instruction to solve and output one integer on 64
generated development questions grouped into 16 arithmetic fact families.
The corrected grader scores the base at 26/64 (40.6%) and the explicit prompt
at 37/64 (57.8%). The prompt had 21 paired improvements and 10 regressions;
the paired family bootstrap interval for the correctness-rate difference was
[0.0625, 0.28125]. Strict integer-only formatting changed only from 16/64 to
18/64. These results cover 16 synthetic fact families and one arithmetic
operation, so treat the prompt gain as a narrow development-set observation.

The first version of the grader incorrectly required exactly one integer token
to mark numeric correctness. It labeled explanations such as “68 plus 4 is
72” wrong. Grader version `final-integer-token-v2` uses the last integer in a
response as its numeric answer and reports strict integer-only formatting
separately; 39/128 correctness labels changed after regrading the same raw
outputs, with no new model generations. Both reports and a row-level change
ledger are preserved so the correction is auditable. The grader uses the
generator's independently computed integer key. The 64 prompts are development
data and must not be repurposed as independent final confirmation. No adapter
was evaluated, so this is not evidence about weight editing or a personality
trait.

Evidence, raw outputs, report, run settings and checksums:
`evidence/casegen_v3/evaluations/cpu_dev_offset3000_n64/`. The preceding
network-based load failure is preserved separately at
`evidence/casegen_v3/evaluations/model_id_load_failure_20261025/`; it produced
zero generations. Using the exact local snapshot directory allowed the CPU run
to complete without network access. The original v1 report is retained as the
superseded report; use `regrade_final_integer_v2/report.json` for results.

Reproduce on a machine with the pinned snapshot cached (or internet access):

```sh
python -m evolution_lab case-eval evolution_lab/evidence/casegen_v3 \
  --split dev --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --device cpu --precision float32 --arms base prompt \
  --offset 3000 --limit 64 --max-new-tokens 16 \
  --output-dir results/evolution_lab/case_eval_cpu_dev_offset3000_n64
```

Limits and next falsification: arithmetic-only prompts do not measure the
style behaviors studied in F001–F005. Repeat a larger, newly generated balanced
development campaign only after establishing diverse arithmetic difficulty and
verifying output-generation throughput; then validate the evaluator on MPS
before using it for a full four-arm LoRA-versus-prompt/random study.

### F012 — Phrase and sentence completion now share the paired evaluator

Status: generator, CPU development run and one predeclared CPU confirmation
comparison completed; no weight-edit or general language-capability claim. See
F013 for the retired confirmation result.

The new `phrase-sentence-completion-v1` generator creates 672 unique questions:
160 curated common phrase completions and 512 subject–verb agreement sentence
completions. It stratifies entire lexical/grammar families into train (464),
development (104), and confirmation (104) pools. The development and
confirmation pools each contain 24 phrase items from six phrase families and
80 sentence items from ten noun–verb families. Four prompt surfaces per fact
stay in the same family split. Each completion has two answer choices and a
trusted key from the curated phrase table or explicit subject/verb forms.

The base model and a direct system instruction were evaluated on the same 104
development questions, producing 208 answer records. After correcting the
completion grader to recognize a unique offered choice inside full-sentence
answers, the base scored 78/104 and the prompt arm 79/104. Phrase scores were
15/24 in both arms; sentence scores were 63/80 and 64/80. Strict one-word
formatting rose from 11/104 to 99/104, mostly because the prompt asks for only
the missing word. The paired result was 17 improvements and 16 regressions;
the task-stratified family-bootstrap interval [-0.087, 0.115] includes no
effect. By-task intervals were [-0.292, 0.292] for phrase completion and
[-0.075, 0.113] for grammar completion. This supports a format-compliance
change on this small development set, not a reliable accuracy gain. Phrase
families are culturally dependent and sentence probes cover only one grammar
rule.

The first pass used a last-word parser and misread full completed sentences.
The corrected grader accepts a unique distinct option mentioned in the
response and marks responses that mention both choices as ambiguous. All 208
records were regraded offline; 56 correctness labels changed and no
generations were added. The original report, row-level
changes, and corrected report are all preserved. The 104 development prompts
are now selection data and cannot be called fresh confirmation.

Dataset census and checksums are in `evidence/completion_v1/`. Raw model
outputs, run settings, source checksums, and corrected scores are in
`evidence/completion_v1/evaluations/dev_n104_base_prompt/`.

Generate and validate the fixed set:

```sh
python -m evolution_lab generate-completions --seed 42 --start 0 --count 672 \
  --output evolution_lab/evidence/completion_v1
python -m evolution_lab generate-completions --validate-only \
  --output evolution_lab/evidence/completion_v1
```

Evaluate a bounded completion campaign:

```sh
python -m evolution_lab case-eval evolution_lab/evidence/completion_v1 \
  --split dev --revision MODEL_COMMIT_SHA --device auto --arms base prompt \
  --limit 104 --output-dir results/evolution_lab/completion-dev
```

Limits and next falsification: the completion grader needs to be audited on
more adversarial responses, and phrase completion is inherently less objective
than the explicit grammar probes. Add new phrase and sentence families without
reusing the exposed dev set; tune only on development families and reserve a
fresh family pool for later confirmation. Before studying edits, compare the
four controls on those new families and check preservation outside completion.

### F013 — A terse completion prompt traded correctness for format on confirmation

Status: completed one predeclared base-versus-prompt confirmation comparison;
negative correctness result; no weight edit and no claim about broad language
ability.

After the 104-question development results were frozen, the previously untouched
confirmation families were evaluated once on the pinned Qwen2.5-0.5B-Instruct
revision `7ae557604adf67be50417f59c2c2f167def9a775`. This set contains 24 phrase
items from six lexical families and 80 subject–verb sentences from ten grammar
families. The arms were the default base instruction and the predeclared direct
instruction to choose the completion and reply only with the missing word. No
training, prompt search, or model selection used these confirmation answers.

Base correctness was 85/104; the prompt arm was 73/104. There were 10 paired
improvements and 22 regressions, for a −11.5 percentage-point difference. The
family-stratified paired bootstrap interval was [−23.1, −1.9] percentage points
overall. Phrase completion fell from 19/24 to 13/24, with a wide interval
[−58.3, −4.2] points over only six families. Sentence agreement fell from
66/80 to 60/80; its interval [−20.0, +3.8] points includes zero. Strict
single-word formatting rose from 4/104 to 91/104. This is a clear example of
format adherence and answer correctness moving in opposite directions on this
benchmark. The prompt should not be called a beneficial edit.

This is a finite, curated set of lexical associations and one grammar rule; the
phrase labels may depend on usage conventions, and family counts are small.
The result does not identify a weight mechanism, establish broad prompting harm,
or say anything about LoRA. The confirmation families are now consumed and
must not be reused for tuning or future final claims. Create new families for
any follow-up. Preserve raw responses and checksums in
`evidence/completion_v1/evaluations/confirmation_n104_base_prompt/`; the report,
run specification, and paired bootstrap are in the same directory. A model-name
lookup failed before inference because this session lacked DNS; the successful
run used the exact cached snapshot path, and the zero-generation failure is
recorded in the import manifest.

Reproduce with the exact local snapshot available and substitute its path:

```sh
python -m evolution_lab case-eval evolution_lab/evidence/completion_v1 \
  --split confirmation --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --model MODEL_SNAPSHOT_PATH --device cpu --precision float32 \
  --arms base prompt --limit 104 --allow-confirmation \
  --output-dir results/evolution_lab/completion-confirmation
```

Next falsification: construct new, larger lexical and syntactic family pools,
then compare base, stronger task-specific prompts, conventional LoRA, a
layerwise norm-matched random edit and a fixed nonlearned adapter mixture. Include
capability-preservation questions outside completion. Keep the prompt-selected
format behavior as its own metric so a style gain cannot conceal a correctness
loss. Do not use this retired confirmation split to select edits or report a new
independent result.

### F014 — Completion LoRA learned the agreement rule but damaged locality

Status: controlled development-only edit signal with a serious preservation
regression; candidate quarantined. No final confirmation was used.

A new `train-cases` command trained a rank-4 adapter on the v1 completion
training split and produced a per-layer norm-matched random adapter. The pinned
Qwen2.5-0.5B-Instruct revision was loaded locally on CPU float32. The training
run used AdamW, 64 updates, learning rate 0.0005, alpha 8, q/v projections,
zero dropout, max length 256, and gradient clipping at 1.0. It sampled 64
unique examples from 42 of the 72 available training families (15 phrase and
49 sentence examples). Training loss averaged 0.463 over the first 16 updates
and 0.064 over the last 16. The resulting effective delta had global Frobenius
norm 1.926 across 48 adapted modules. The random control matched each layer's
effective delta norm within the evaluator's tolerance.

On the 104 already-used completion development items, base scored 78/104,
explicit prompt 79/104, LoRA 93/104, and norm-matched random 81/104. LoRA had
21 improvements and 6 regressions over base (paired difference +14.4 points;
family-bootstrap interval [+4.8, +26.0]). Direct paired contrasts were +13.5
points versus prompt (20 improvements, 6 regressions; interval [+2.9, +25.0])
and +11.5 points versus random (19 improvements, 7 regressions; interval
[+4.8, +19.2]). The apparent gain was concentrated in sentence agreement:
LoRA scored 80/80 versus 63/80 base and 65/80 random. Phrase completion fell
to 13/24 from 15/24 base and 16/24 random; phrase intervals were wide and
included zero. These are development results on only six phrase families and
ten grammar families. The holdout sentence families recombine nouns and verbs
seen elsewhere in the training inventory, so this is not a test on wholly new
vocabulary or grammar.

The separate arithmetic development locality check scored 44/64 for base,
37/64 for prompt, 30/64 for LoRA, and 42/64 for random. LoRA caused 15
base-correct answers to become wrong and fixed only one (−21.9 points; paired
family-bootstrap interval [−35.9, −7.8]). Versus random it regressed 15 and
improved 3 (−18.8 points; interval [−34.4, −3.1]). Strict integer formatting
increased from 16/64 to 23/64 while arithmetic correctness fell. This repeats
an earlier development slice and is a measured locality failure, not a fresh
generalization test. The untouched status of completion v1 confirmation was
already lost in F013 before this adapter was trained, so it was not reused.

The result supports a narrow hypothesis that the learned q/v direction can
strongly alter grammatical agreement on these development families beyond a
same-norm random direction. It does not identify causal layers, demonstrate a
general language ability, or justify keeping the adapter. The arithmetic
counterexamples are preserved; the base checkpoint was never modified, and the
candidate remains quarantined. Evidence includes the training checkpoint,
per-layer deltas, all four-arm raw ledgers, checksummed runs, and direct paired
contrasts in `evidence/completion_v1/evaluations/lora_seed42_steps64_dev/`.

Reproduce the development fit and controls with the commands in README's
"Train completion LoRA" section. Do not evaluate on the retired v1 confirmation
families. Next, expand the vocabulary and grammar inventory into genuinely new
families; then test a new adapter across base, strong prompt, norm-matched
random, fixed mixture, and non-completion capability probes. Ablate q/v and
layer groups only after the broader protocol is frozen, and use a fresh
confirmation family pool for any candidate-selection claim.


### F015 — Completion LoRA transfers agreement to new words but harms phrase completion

Status: strong exploratory transfer signal on one consumed confirmation set; the
candidate remains quarantined. The explicit-prompt arm required a post-hoc
correction, so a fresh confirmation set is required before treating this as an
independent result.

The v2 generator creates 2,208 unique items across 296 families: 160 phrase
completions and 2,048 subject–verb agreement completions. Family-level splits
are 1,544 train, 328 development and 336 confirmation items. The v2 phrase
families, subject nouns and verb roots do not overlap the v1 training inventory.
Only v2 training families may be used for a future v2-trained candidate; this
experiment instead froze the v1 rank-4 q/v LoRA and its layerwise norm-matched
random control. The confirmation split was not used for training or selection.

On the 336 confirmation items, the frozen LoRA scored 289/336 (86.0%), base
169/336 (50.3%), and the random edit 169/336 (50.3%). After correcting the
prompt-system selection bug, explicit completion prompting scored 219/336
(65.2%). Paired family-bootstrap differences were +35.7 percentage points
versus base (95% interval +30.4 to +40.5), +35.7 versus random (+31.0 to
+40.2), and +20.8 versus the corrected prompt (+15.2 to +26.5). The initial
run mistakenly used an arithmetic instruction for the prompt arm; its prompt
responses and comparisons are preserved but excluded. Repeated base responses
matched exactly across the initial and corrected runs (336/336).

The aggregate is dominated by sentence grammar: LoRA scored 277/312, compared
with 201/312 for corrected prompting, 148/312 for base and 148/312 for random.
The sentence-only adapter-minus-prompt interval was +18.6 to +30.1 points.
Phrase results ran the other way: LoRA scored 12/24 versus 18/24 for corrected
prompting and 21/24 for both base and random; its phrase-only difference versus
prompt was −41.7 to −8.3 points. That is only six phrase families, so its
interval is unstable, but the measured regression is material. The adapter also
previously reduced arithmetic development accuracy from 44/64 to 30/64. It
remains quarantined; no weights were merged into the base.

This supports a narrow hypothesis: the v1 learned edit can transfer a
subject–verb agreement behavior across the measured unseen noun and verb
inventory, beyond these prompt and norm-matched-random controls. It does not
show a broad language improvement, a stable personality trait, layer-specific
causality, or transfer to Astra. The v2 confirmation split is now retired
because the first prompt arm was inspected before the defect was fixed and the
corrected prompt was rerun afterward. No nonlearned adapter mixture was tested.
The severe phrase-completion loss and the arithmetic locality failure prevent
promotion.

Reproduce dataset generation and validation with:

```sh
python -m evolution_lab generate-completions-v2 --seed 43 --start 0 --count 2208 \
  --output evolution_lab/evidence/completion_v2
python -m evolution_lab generate-completions-v2 --validate-only \
  --output evolution_lab/evidence/completion_v2
```

Raw initial and corrected ledgers, pairwise family-bootstrap contrasts, repeated
base comparison and SHA-256 manifest are in
`evidence/completion_v2/evaluations/new_vocabulary_confirmation_spent/`.
The corrected run used 336 completion-correct prompt generations and 336
repeated base generations. Combined with the initial four-arm campaign this is
2,016 model generations on 336 unique prompts, not 2,016 independent questions.

Next falsification: build a fresh seed with new phrase, noun and verb families;
freeze the scoring protocol and task-matched prompt before any model run; compare
base, prompt, the frozen LoRA, norm-matched random, a rank-budget-matched
nonlearned mixture and at least one new training seed. Add broader non-completion
capability and locality probes. Do not tune the candidate on the spent v2 split.


### F016 — Equal task sampling did not improve phrase behavior in one seed

Status: controlled development ablation; no independent confirmation or
promotion. This tests whether v2's training-pool imbalance caused the phrase
regression seen in F015.

Two LoRA candidates used the same Qwen2.5-0.5B-Instruct revision, base
checkpoint, seed 43, rank 4, alpha 8, q/v targets, AdamW, 0.0005 learning
rate, 64 updates, 256-token maximum, weight decay 0.01, clipping at 1.0, and
zero dropout. Only the training schedule differed. The uniform fit saw 2 phrase
and 62 sentence examples; the task-balanced fit saw 32 of each. Both random
controls matched the corresponding learned per-layer delta norms. Each
candidate was evaluated with base, task-matched prompt, and random on the full
328-question v2 development split (44 families) and on 64 arithmetic development
questions (16 families). Base and prompt outputs repeated exactly across the two
completion runs; same for the arithmetic run. v2 confirmation remained untouched.

On completion development, uniform scored 276/328 (phrase 18/24, sentence
258/304); task-balanced scored 270/328 (phrase 14/24, sentence 256/304). Base,
prompt, and random scored 182/328, 200/328, and 178/328 respectively. Both LoRA
candidates beat the prompt on the aggregate: +23.2 points for uniform (family
interval +15.2 to +30.8) and +21.3 for balanced (+13.7 to +28.7). However, the
direct task-balanced-minus-uniform difference was −1.8 points overall (interval
−7.9 to +4.3); phrase difference was −16.7 points (14/24 vs 18/24; interval
−33.3 to 0.0) and sentence difference was −0.7 points (256/304 vs 258/304;
interval −7.2 to +5.9). The small phrase sample and one training seed do not
show that balancing is harmful, but clearly fail to support the simple hypothesis
that equal task sampling fixes phrase completion.

On arithmetic development, uniform scored 43/64, balanced 44/64, base 38/64,
prompt 42/64, and random 37/64. Paired adapter-minus-base intervals included
zero for both candidates (uniform +7.8 points, −1.6 to +17.2; balanced +9.4,
−1.6 to +20.3). This is no detected loss on this 64-question slice, not proof
of preserved arithmetic capability. Strict one-word completion format reached
328/328 for both adapters, versus 305/328 for the prompt; format is not a
substitute for correctness.

The schedule intervention changes which training examples drive the gradient;
it does not identify causal layers or individual weights. The uniform training
delta norm was 1.881 across 48 q/v modules; balanced was 1.837. The learned
directions differ in more than norm because the training samples differ. The
comparison is one seed on one small model and development data already used for
selection. Neither adapter is promoted.

Evidence: full training checkpoints, per-layer deltas, all completion/locality
responses, direct paired contrasts, run specifications and checksums are in
`evidence/completion_v2/evaluations/sampler_ablation_seed43_dev/`. Across both
candidates this records 3,136 model generations on 392 unique prompts, plus 128
training updates and 74 software tests. Repeated base and prompt runs are not
new unique questions.

Reproduction: train with the same command and change `--sampling-strategy`
between `uniform` and `task-balanced`; use the two run IDs in the evidence
manifest with `case-eval --split dev`. Do not evaluate these candidates on the
spent v2 confirmation split. Next, repeat both schedules with a second training
seed on development data; freeze one protocol and create a fresh v3 family pool
for confirmation. Add a rank-budget-matched adapter mixture and broader
capability/locality probes before any retention decision.


### F017 — Sampler effects vary by seed; balanced training may trade sentence and arithmetic behavior

Status: development-only seed replication and locality probe. The adapters are
quarantined; no confirmation pool was used and neither is eligible for
promotion.

This followed F016 with seed 44. Both candidates used the same pinned
Qwen2.5-0.5B-Instruct revision, v2 train families, rank 4, alpha 8, q/v
targets, AdamW, learning rate 0.0005, 64 updates, 256-token maximum, weight
decay 0.01, clipping 1.0 and zero dropout. The only planned intervention was
the deterministic sampling strategy. Seed-44 uniform saw 5 phrase and 59
sentence examples; task-balanced saw 32 of each. Each candidate's random
control matched its per-layer LoRA delta norms. Their global effective delta
Frobenius norms were nearly equal (1.87033 and 1.87016), but matching norm
does not match direction.

On the same 328 v2 development questions (44 families), base scored 182/328,
the explicit task prompt 200/328, and the random controls 180/328 (uniform)
and 182/328 (balanced). Seed-44 uniform scored 301/328, with phrase 15/24 and
sentence 286/304. Seed-44 task-balanced scored 289/328, with phrase 13/24 and
sentence 276/304. The direct balanced-minus-uniform contrast was −3.7 points
(17 improved, 29 regressed; family-bootstrap 95% interval −8.8 to +0.9).
Phrase accuracy remains noisy because its 24 questions cover only six
families; the results do not support the claim that equal phrase exposure
improves phrase completion.

The seed-44 uniform adapter exceeded the seed-43 uniform adapter from F016 by
7.6 points on the same development questions (family-bootstrap interval +2.7
to +12.8), largely from sentence items. This is evidence of meaningful seed
sensitivity on the selected development set. It is not independent
confirmation and should prevent treating one training seed as a stable edit
recipe.

The separate 64-question arithmetic development probe (16 families) produced
a sharply different sampler result. Uniform scored 38/64, equal to base and
below prompt at 42/64; task-balanced scored 61/64 versus base 38/64, prompt
42/64 and norm-matched random 37/64. The direct balanced-minus-uniform
difference was +35.9 points (24 improved, one regressed; family-bootstrap
interval +26.6 to +46.9). This is a strong, surprising development signal,
not a verified arithmetic capability gain. It needs a fresh, untouched
arithmetic confirmation pool and an independently selected training seed
before influencing any edit guide or candidate choice.

Base and prompt responses were identical across the two repeated completion
campaigns on all 328 prompts. The four seed-44 evaluations used 3,136 model
generations on 392 unique questions (328 completion plus 64 arithmetic),
alongside 128 training updates. Repeats are generations, not new questions.
Raw response ledgers, both adapter and random checkpoints, per-layer deltas,
run specs, paired family-bootstrap reports and hashes are in
`evidence/completion_v2/evaluations/sampler_ablation_seed44_dev/`.

Reproduce by fitting both sampler settings at seed 44, then running
`case-eval` on the v2 `dev` split and on `casegen_v3 --split dev --limit 64`.
The current Python process used CPU float32; the previously reported MPS
checks belong to the user-host session and do not imply MPS was used here.
The v2 confirmation split remains retired after F015. Before any promotion,
create new independent phrase, agreement and arithmetic families, freeze the
grading and prompt controls, replicate the sampler contrast across seeds, and
include a fixed nonlearned adapter mixture plus broad locality checks.

The only guide-level observations so far are that task balancing did not fix
phrases in either seed, completion scores were seed-sensitive, and seed-44
balanced training had an unexpectedly strong arithmetic development score.
No specific layers or individual weight directions have yet been causally
isolated; these outcomes do not transfer automatically to Astra.


### F018 — Component ablations identify checkpoint-specific dependencies, with a v-projection and upper-layer signal

Status: preregistered development-only component intervention. All candidates
remain quarantined. Neither v2 completion confirmation nor casegen v3 arithmetic
confirmation was evaluated.

Starting from F017's seed-44 task-balanced rank-4 completion LoRA, I created
four variants by zeroing the LoRA B tensors for q projections, v projections,
the lower half of transformer layers, or the upper half. The same groups were
zeroed in the per-layer norm-matched random control. The q/v and layer group
interventions were registered before any generations. The effective delta
sidecars were recomputed and validated against loaded tensors. All other
adapter tensors remained unchanged. This intervention tests whether parts of
this one checkpoint are necessary for its measured scores; it does not prove
that any single weight or layer independently encodes a behavior.

Each variant was run against the same 328 v2 development completion prompts
(24 phrase prompts across six families and 304 sentence prompts across 38
families) and 64 arithmetic development prompts across 16 families. Each run
included base, task-matched prompt-only, ablated LoRA and identically ablated
norm-matched random arms. The intact seed-44 adapter scored 289/328 overall on
completion and 61/64 arithmetic. Base/prompt scored 182/328 and 200/328 on
completion, and 38/64 and 42/64 on arithmetic.

On completion, q-projection removal scored 279/328 and differed from the full
adapter by −3.0 points (paired family-bootstrap 95% interval −7.6 to +1.2).
V-projection removal scored 275/328, a −4.3 point change (−8.5 to −0.3).
Removing lower-half edits scored 295/328, +1.8 points versus full (−1.2 to
+4.9). Removing upper-half edits scored 242/328, −14.3 points (−19.2 to
−9.5); sentence completion was 230/304 versus the full adapter's 276/304, and
the sentence-only paired interval was −20.4 to −10.2 points. These results
support a development-set dependency on upper-layer edits for sentence
completion in this checkpoint. The q effect is inconclusive; the lower-half
result does not show a loss. Phrase scores ranged from 12/24 to 16/24, but only
six phrase families are available, so these are exploratory and imprecise.

Arithmetic locality shows a different pattern. After each module-group removal,
scores were 52/64 for q, 37/64 for v, and 46/64 for either layer half, compared
with 61/64 for the intact adapter. Paired ablated-minus-full differences were
−14.1 points for q (95% interval −25.0 to −4.7), −37.5 for v (−46.9 to
−28.1), and −23.4 for either layer half (−29.7 to −15.6). These 64 items are
one exposed development slice, not proof that the intact adapter generally
improves arithmetic or that an ablated model has broad capability loss. Across
all variants, the completion adapter remained better than the norm-matched
random control on sentence-heavy completion; the arithmetic advantage over
random was strongest for q removal and absent for v removal.

The intervention changes total delta magnitude and removes interacting
components, even though the random control receives the matching ablation.
Therefore these results are checkpoint-specific necessity evidence, not an
isolated neuron/weight explanation, a general layer rule, or evidence that the
same pattern transfers to Astra. Follow-up work should repeat the frozen
interventions across independent training seeds, add an equal-remaining-norm
control, probe more capability families, and verify the strongest dependency on
fresh confirmation families before adding it to a reusable edit guide.

The campaign recorded 6,272 model generations on 392 unique questions across
the four component interventions (repeated base and prompt responses remain
generations, not new questions), four pairs of ablated adapter/random
checkpoints, and 77 passing software tests. It ran on CPU float32 because MPS
was built into the local Torch package but unavailable to this process; no MPS
result is claimed. Raw responses, reports, run metadata, checksummed ablated
checkpoints, preregistration and paired family-bootstrap summaries are in
`evidence/completion_v2/evaluations/adapter_component_ablation_seed44_dev/`.

To reproduce an ablation, first create a source adapter with `train-cases`,
then run `python -m evolution_lab ablate-adapter RUN_DIR --ablation zero_v_proj
--output-dir results/evolution_lab/ablation/zero_v_proj`. Evaluate the generated
`adapter_lora` and `adapter_random` directories with the same `case-eval`
commands in the evidence run metadata. The registered dataset offsets and
controls are in `preregistration.json`; verify `import_manifest.json` before
using the raw outputs. All these development pools have been exposed by prior
studies and cannot support promotion.

**Confirmed within this checkpoint and these development cases:** removing the
upper-half LoRA edits coincided with a large sentence-completion score drop;
removing v-projection edits coincided with smaller completion and much larger
arithmetic score drops. **Tentative:** upper-layer edits may carry much of this
adapter's sentence behavior; v-projection edits may be important for its
arithmetic result. **Not supported:** q-projection removal did not materially
reduce aggregate completion, and lower-half removal did not reduce it. **Open:**
seed replication, equal-norm ablations, independent held-out verification,
additional behavior families, and transfer beyond Qwen2.5-0.5B-Instruct.


### F019 — A completion-trained adapter generalized its arithmetic signal to fresh confirmation families, with a development-to-confirmation drop

Status: one preregistered held-out verification of the frozen seed-44 adapter's
arithmetic behavior. The selected families are now retired. The adapter remains
quarantined because this is one task, one seed and one small model.

Before inference, the evaluation froze the F017 seed-44 task-balanced LoRA and
random control, pinned Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, and selected 32 whole families
from the untouched `casegen_v3` confirmation split by SHA-256 rank using seed
20261025. Selection did not use answers or model scores; every family supplied
all four paraphrases. The 128 selected case IDs had no overlap with archived
casegen response ledgers. The four arms were base, task-matched prompt,
seed-44 LoRA and its per-layer norm-matched random control. Greedy decoding and
trusted integer answer keys were used. No model, prompt or adapter changes are
allowed based on these results.

The frozen adapter scored 107/128 (83.6%), compared with 78/128 (60.9%) for the
explicit prompt, 53/128 (41.4%) for base and 53/128 for the random control.
The paired adapter-minus-prompt contrast was +22.7 percentage points (43
improved, 14 regressed; 95% family-bootstrap interval +14.8 to +29.7). Against
base it was +42.2 points (62 improved, 8 regressed; interval +33.6 to +50.8).
The adapter's strict single-integer format score was 83/128; correctness and
format are separate metrics.

This independently supports an arithmetic effect on the selected fresh
families for this one adapter. It also tempers F017's 61/64 (95.3%) arithmetic
development score: confirmation was 83.6%, an 11.7-point descriptive gap
across different question sets. The gap is not a paired estimate, and neither
score establishes broad reasoning or capability gains. No phrase or sentence
confirmation was run here; the v2 text confirmation is already retired. Keep
the adapter quarantined until fresh phrase/sentence verification, replication
across training seeds, and broader locality/capability checks.

The evaluator now supports `--family-limit N --family-seed S` to select whole
families by stable SHA-256 rank. The exact family IDs, case fingerprint,
pre-registration, all 512 raw generations, paired report, run specification and
checksums are in
`evidence/casegen_v3/evaluations/seed44_balanced_arithmetic_confirmation_f019/`.
The run used CPU float32; Torch's MPS backend was built but unavailable in this
process. The full suite passed 78 tests before the campaign.

Reproduce the family selection and campaign with the F019 parameters in
`run.json` and the recorded candidate paths in the F017 evidence. The
confirmation opt-in is explicit. The selected families have already been
inspected and must never be used to tune or retest another candidate.
**Confirmed narrowly:** the frozen adapter beat prompt, base and random controls
on these 32 arithmetic families. **Tentative:** the edit may encode a
transferable arithmetic-output bias on this model. **Failed:** the development
score did not reproduce at the same level. **Open:** fresh phrase/sentence
confirmation, training-seed replication and transfer to other models.

### F020 — Fresh phrase and long-distance agreement benchmark added

Status: deterministic benchmark infrastructure verified. The first exploratory
development results are recorded separately under F021; confirmation remains
untouched.

The new v3 generator records 8,740 cases: 196 completions from 49 curated
phrase families and 8,544 subject–verb completion sentences. Sentence prompts
place a singular or plural attractor between the main subject and the missing
verb. The trusted label is determined by the explicit subject number and
inflected verb table, not a model judge. The 32 singular/plural subject pairs
and 32 verb pairs are partitioned independently and then crossed only within
the same split, making sentence lexemes disjoint across train, development and
confirmation. Phrase families stay wholly within one split. Seed 42 yields
7,880 train, 428 development and 432 confirmation cases. The confirmation
sample has 32 phrase questions across only 8 families, so phrase uncertainty
will be wide; sentence results must be reported separately.

Validation confirms all 8,740 prompt IDs are unique, answer choices and solver
keys are consistent, family assignments stay intact, and shard hashes match.
Exact v3 phrase-prefix/answer pairs do not overlap the earlier v1/v2 inventories.
The benchmark validation itself makes no model behavior claim. F021 records the
first development campaign; it is exploratory, with no independent confirmation
or edit promotion. The holdout has not been consumed. Preserve phrase
completion and subject–verb agreement as separate outcomes.

Reproduce or validate with `python -m evolution_lab generate-completions-v3
--seed 42 --output evolution_lab/evidence/completion_v3` and the corresponding
`--validate-only` command. The full seed-42 shards and hash manifest are in
`evidence/completion_v3/`. The case evaluator accepts this generator and uses
the completion-specific prompt and objective word-choice grader.

**Confirmed:** generator-level determinism, trusted keys, and disjoint
sentence subject/verb inventories. **Not confirmed:** general model behavior,
edit benefit, locality, capability preservation, or generalization. **Open:**
replicate the F021 bias across training seeds on development, compare
nonlearned adapter mixtures, quantify phrase and sentence results separately,
and consume fresh confirmation only after preregistration and a stronger
development candidate.

### F021 — Balanced completion LoRA shifts sentence behavior toward plural forms

Status: exploratory single-seed development signal; candidate quarantined;
confirmation untouched.

One rank-4 q/v LoRA was trained for 64 AdamW updates at learning rate 0.0005
on v3 training families, sampling phrase and sentence task types equally. Its
32 sentence training updates contained 17 plural-subject and 15 singular-
subject examples. Qwen2.5-0.5B-Instruct was pinned to
`7ae557604adf67be50417f59c2c2f167def9a775`, run on CPU float32, and decoded
greedily. The 428-case development split was evaluated against base, a
completion-specific prompt, and a per-layer norm-matched random adapter.

The candidate scored 276/428 (64.5%), prompt 264/428 (61.7%), base 227/428
(53.0%) and random 227/428 (53.0%). Adapter-minus-prompt was +2.8 points
(63 items improved, 51 regressed; family-bootstrap 95% interval −1.9 to +7.7),
so there is no clear overall advantage over the strong prompt. By task, phrase
accuracy was 19/28 for adapter and prompt, versus 23/28 for base. Sentence
accuracy was 257/400 for adapter, 245/400 for prompt, and 204/400 for base.
The adapter's sentence-minus-prompt contrast was +3.0 points (family-bootstrap
95% interval −2.0 to +8.25). Format compliance reached 428/428 for the adapter,
but this does not override correctness or imply broader instruction-following.

A post-hoc subject-number diagnostic exposed a marked asymmetry: the adapter
answered 200/200 plural-subject sentences correctly and 57/200 singular-subject
sentences, compared with 131/200 and 73/200 for base, and 158/200 and 87/200
for the prompt. Family-bootstrap intervals for adapter-minus-prompt were
+21 points on plural subjects (95% interval +13 to +29) and −15 on singular
subjects (−22 to −8). The random control scored 133/200 plural and 71/200
singular. Since the sampled training examples were close to balanced by subject
number, this asymmetry is not explained by a simple count imbalance. It is
consistent with an edit that favors base verb forms and trades away singular
agreement. This is a measured development pattern, not an independently
confirmed rule about a specific layer or weight.

The comparison was exploratory, not preregistered, and used one seed on one
development split. No arithmetic, locality, broad preservation, independent
seed, or other-model evaluation was run. The candidate is quarantined; do not
use the new confirmation split for this candidate because phrase behavior is
below base and the aggregate margin over prompting is uncertain. All 1,712
responses on 428 unique questions, paired controls, checkpoints, delta stats,
training log, report, post-hoc analysis and hashes are under
`evidence/completion_v3/evaluations/seed42_balanced_dev_f020/`. Reproduce with
the seed-42 training and `case-eval --split dev --arms base prompt adapter
random` commands in the README.

**Observed on development only:** the LoRA changes outputs beyond a norm-matched
random delta and increases plural agreement, while producing worse singular
accuracy and no clear overall advantage over the task-matched prompt.
**Rejected:** “the completion adapter improves phrase and sentence completion
generally.” **Open:** replicate the direction across training seeds; inspect
singular errors and weight/module contributions; compare explicit number-
balanced objectives and nonlearned mixtures on development; do not claim an
edit guide or consume confirmation until a candidate clears the prompt and
preservation gates.


### F022 — Second training seed repeats the plural pattern on the same development set

Status: preregistered seed replication on previously inspected development
questions; no independent confirmation; both candidates remain quarantined.

The seed-43 adapter used the same 7,880-case training inventory, 64 updates,
rank-4 q/v LoRA, AdamW learning rate 0.0005, and task-balanced sampling as
F021. It ran on pinned Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, CPU float32. Its actual 64
training examples included 32 phrase and 32 sentence cases, but the sampled
sentence subjects were 21 singular and 11 plural. F021's seed-42 sample had
32 phrase and 32 sentence cases, with 15 singular and 17 plural subjects.
Both exact update lists and family distributions are retained in the evidence.

On the same 428 development questions, F022 scored 284/428 (66.4%) for the
adapter, 264/428 (61.7%) for prompt, 227/428 (53.0%) for base, and 215/428
(50.2%) for the norm-matched random adapter. Adapter minus prompt was +4.7
points, with family-bootstrap 95% interval −1.6 to +11.2; this remains
inconclusive. Phrase accuracy was 14/28, below base (23/28) and prompt (19/28).
Sentence accuracy was 270/400 versus prompt's 245/400 and base's 204/400.
Both training seeds answered all 200 plural-subject sentence probes correctly;
F022 answered 70/200 singular probes. The two adapters produced the same
answer on 367/400 sentence cases and 11/28 phrase cases, but this is still a
repeated development set, not new generalization evidence.

This repeat does not show that class counts alone cause the plural preference:
F022 had more singular than plural subject examples, yet retained the same
plural ceiling. Example identities, lexical families, presentation order and
gradient trajectories also differ; seed and sample content are intertwined.
The result supports a recurring measured behavior on this benchmark, not a
causal explanation of particular weights or a useful general edit. F022 adds
1,712 generations and **zero new unique questions** beyond F021. Confirmation
remains untouched. See
`evidence/completion_v3/evaluations/seed43_balanced_dev_f022/`.

**Observed:** both balanced-seed adapters favor plural sentence completions;
phrase outcomes and overall adapter-versus-prompt margins remain mixed.
**Rejected:** simple explanation that the number of singular/plural training
updates alone determines the bias. **Open:** isolate schedule and example
content, inspect targeted module/layer interventions, and run only a frozen
candidate on fresh confirmation after broader preservation checks.

### F023 — Training schedule changes the measured sentence behavior

Status: strong preregistered development result; no independent confirmation;
uniform adapter frozen as a candidate for possible confirmation, but still
quarantined.

F023 fixed the seed at 42 and compared uniform sampling with F021's
task-balanced schedule. It used the same pinned Qwen2.5-0.5B-Instruct
revision, training pool, rank-4 q/v LoRA, 64 updates, AdamW at 0.0005, and
CPU float32. The 7,880-case training pool contains 136 phrase examples
(1.73%) and 7,744 sentence examples. Uniform sampling selected 1 phrase and
63 sentences; task-balanced sampling selected 32 of each. Uniform and
balanced selected disjoint sets of 64 exact training examples, so the contrast
is between full sampling schedules including the resulting example content,
not a pure isolated phrase-count change.

On the same previously inspected 428 development questions, the uniform
adapter scored 376/428 (87.9%), versus 264/428 (61.7%) for the explicit prompt,
227/428 (53.0%) for base, and 227/428 for the per-layer norm-matched random
edit. The trained and random controls both changed 48 q/v modules; their
effective global delta Frobenius norms were 1.894164 and 1.894167, with a
maximum per-layer relative norm mismatch of 6.96e-6. Uniform-minus-prompt was
+26.2 percentage points (129 improved, 17 regressed; paired family-bootstrap
95% interval +20.8 to +31.3). Uniform
minus the seed-42 task-balanced adapter was +23.4 points (109 improved, 9
regressed; interval +16.4 to +29.7). The sentence score drove this contrast:
358/400 for uniform versus 257/400 for task-balanced, with sentence-only
interval +17.8 to +32.0 points. By subject number, uniform scored 159/200
singular and 199/200 plural, compared with balanced's 57/200 singular and
200/200 plural. Phrase accuracy was 18/28 for uniform, 19/28 task-balanced,
19/28 prompt, and 23/28 base. The phrase-only uniform-minus-balanced interval
was −25.0 to +14.3 points, so this small phrase sample is inconclusive.

The evaluator reran all four arms: base and prompt responses matched F021
exactly on all 428 cases each. The campaign generated 1,712 responses on 428
questions but added **zero new unique questions**. The development set was
already inspected in F021/F022, so these strong margins are not independent
generalization. The training sample identities also changed completely between
the two schedules. The result supports the claim that the implemented training
schedule can substantially alter this model's behavior on this benchmark; it
does not show that the phrase-example count alone caused the difference, that
specific weights encode a general grammatical trait, or that the behavior
transfers to Astra.

The adapter remains quarantined. Phrase performance is below base, one seed
was trained, and broader capability/locality checks were absent at the time of
F023. F024 has since consumed and retired the v3 confirmation families. Exact
training IDs, split, raw responses, checkpoints, effective-delta summaries,
paired analysis, run settings and hashes are under
`evidence/completion_v3/evaluations/seed42_uniform_dev_f023/`.

Reproduce the training with the immutable model revision and
`--sampling-strategy uniform`; evaluate with `case-eval --split dev --arms base
prompt adapter random` and the archived adapter/delta paths. The recorded
environment used Python 3.11.17, torch 2.14.1, Transformers 4.57.6 and PEFT
0.18.1. MPS was built into torch but unavailable; this run used CPU float32.

**Observed on exposed development data:** uniform-schedule training changes
sentence completion substantially relative to task-balanced training and the
strong prompt, chiefly by improving singular subjects; phrase performance
remains lower than base. **Tentative:** the composition and selection of
training examples are major drivers of this LoRA's behavior. **Rejected:**
“this proves phrase frequency alone caused the change” and “the edit is ready
to transfer.” **Open:** independent confirmation, a fixed-pool composition
dose-response, wider capability/locality tests, seed replication, and precise
module/layer interventions.

### F024 — Frozen candidate confirmation and arithmetic preservation

Status: completed preregistered confirmation; both confirmation families are
retired. The F023 uniform adapter remains quarantined and is not eligible for
promotion.

On 432 untouched completion questions (32 phrase items and 400 sentence items
across 33 families), the frozen candidate scored 409/432 (94.7%): 19/32
phrases and 390/400 sentences. The task-matched prompt scored 264/432, base
200/432, and the per-layer norm-matched random edit 205/432. Candidate minus
prompt was +33.6 percentage points (150 improved, 5 regressed; paired
family-bootstrap 95% interval +28.7 to +37.7). The sentence result was 390/400
versus prompt 249/400; however, phrase score 19/32 remained below base 26/32.
This is strong independent evidence for this narrow sentence-completion
behavior on this model and dataset, with an explicit phrase trade-off.

The fixed 50/50 concatenation of the balanced and uniform adapters scored
388/432 (17/32 phrases, 371/400 sentences). Uniform candidate minus mixture
was +4.9 points (31 improved, 10 regressed; paired family-bootstrap 95%
interval +2.1 to +8.1). Base and prompt responses replayed exactly on 432/432
questions each. The mixture is rank 8 while the candidates are rank 4, so this
is not an equal-capacity comparison.

On a separate 312-question, 78-family addition probe, the candidate scored
99/312 (31.7%), versus 155/312 prompt (49.7%), 54/312 base, and 54/312
norm-matched random. Candidate minus prompt was −17.9 points (25 improved, 81
regressed; family-bootstrap 95% interval −23.4 to −12.8). It beat base by
14.4 points (95% interval +10.6 to +18.3), but remained far below the strong
prompt. This narrow check does not establish broad capability preservation.

The three campaigns recorded 4,704 generations on 744 unique questions. The
completion data were replayed for the mixture comparison; repeated answers
count as generations, not additional questions. A progress-log defect exposed
base-arm correctness indicators for the first 325/432 completion cases before
detection. No candidate, case, or analysis choice changed in response, and the
deviation remains in the preregistration and archived analysis. Two additional
preflight command errors generated zero responses and were documented before
retry: local model-path identity and archived adapter-manifest validation.

All raw observations, run specifications, reports, mixture and random
checkpoints, direct paired comparisons, preregistration amendments, and SHA-256
manifest are under `evidence/f024_frozen_confirmation/`. The result does not
show Astra transfer, general reasoning improvement, broad locality, or a
weight-to-trait map. Do not tune on either retired confirmation family.

### F025 — Fixed training examples, changed task exposure schedule

Status: completed two-seed development study; no independent confirmation and
no candidate promotion. This study records a concrete training-data result and
also exposes why task update counts are only part of the data story.

F023 shows why training-data provenance must be treated as a causal variable:
uniform sampling selected 1 phrase and 63 sentence examples, while the balanced
schedule selected 32 of each, and the two 64-example sets had no examples in
common. Their behavior difference cannot be assigned to the task ratio alone.
F024's confirmation sets were not used to choose or tune F025.

The preregistered F025 cohort fixes 64 train-split examples from 64 distinct
families: 32 phrase examples and 32 sentence examples, with 16 singular and 16
plural sentence subjects. Its hash-bound manifest is
`evidence/f025_fixed_support_exposure/training_cohort/manifest.json`. Six
adapters used this same support set, three task-update allocations (phrase /
sentence: 1/63, 16/48, 32/32) and two paired training seeds. They shared the
pinned Qwen2.5-0.5B-Instruct revision, 64 total optimizer steps, rank-4 q/v
LoRA, AdamW and learning rate 0.0005. Each task draws with replacement from
its own seed-specific stream. The exact per-step IDs, loss trace and repeated
example counts are saved under `evidence/f025_fixed_support_exposure/training/`.

This makes one important distinction visible: a nominal 32/32 update split
does not mean 32 unique examples of each kind were learned once. Across the
six runs, only 27–40 distinct examples were encountered in total; the 1/63
runs saw one unique phrase item and 26 unique sentence items, while individual
examples repeated up to seven times. Both the full cohort distribution and
actual sampled IDs matter when interpreting an edit.

On the already-inspected 428-question development split (28 phrase items from
7 families and 400 sentence items from 25 families), adapters scored 334–377
correct. Each beat the task-matched prompt (264/428) by 16.4–26.4 percentage
points in paired family-bootstrap intervals that excluded zero. The base
scored 220/428 and norm-matched random edits scored 207–224/428. However,
every adapter scored below the base on phrase completion: 14–20/28 versus
24/28. The observed gains came from sentence completion, not a broad uplift.
Base and prompt responses replayed identically in all six runs.

The training-ratio comparisons do not establish a reliable dose-response.
Moving from 1/63 to 16/48 changed total accuracy by +8.2 points for seed 42
(family-bootstrap interval +4.2 to +12.4) but only +0.5 for seed 43 (−1.9 to
+2.6). Moving from 16/48 to 32/32 changed it by −0.5 for seed 42 (−4.2 to
+3.0) and −2.3 for seed 43 (−6.3 to +2.1). Higher phrase exposure did not
consistently improve phrase scores: the 32/32 treatment underperformed 16/48
on phrases in both seeds. Only two training seeds and seven phrase families
were available, and the compared treatment includes schedule order as well as
the number of task updates. These are exploratory estimates, not confirmation.

The exact complete raw-ledger audit and reproducible paired-family analysis
are in `evidence/f025_fixed_support_exposure/summary.json` and
`evidence/f025_fixed_support_exposure/analyze_f025.py`; all six campaigns
contain base, prompt, trained-adapter and per-layer norm-matched random arms.
Recompute with:

```sh
PYTHONPATH=. python evolution_lab/evidence/f025_fixed_support_exposure/analyze_f025.py \
  --output evolution_lab/evidence/f025_fixed_support_exposure/summary.json
```

F025 adds 10,272 model response records but no new unique questions: its 428
questions and 32 families had already been inspected. The v3 confirmation
families remain retired after F024. Do not use F025 to claim independent
generalization, broad capability preservation, a causal effect of update
count separated from order, a general weight-to-trait guide, or transfer to
Astra. The next valuable step is to design a fresh, larger lexical/grammatical
confirmation cohort and preserve unrelated capability probes; do not tune on
the old F025 development questions.

### Engineering follow-up — separate task-order randomness and account for reused baselines

Status: implementation verified by focused software tests; no new model
behavior result.

`train-cases --schedule-seed` now records a task-order random seed separately
from the model initialization and within-task example-stream seed. For the
two-task case, a shared schedule seed creates nested randomized positions when
the task ratio changes; without this option, the historical seeded shuffle is
preserved exactly. Multi-task schedules are explicitly labeled as a seeded
multiset shuffle because the two-task nesting guarantee does not apply. Every
optimizer update still records the chosen task, exact example ID, and loss.

`case-eval --baseline-cache-from` can import base/prompt responses from a
completed compatible campaign. It checks the source run, specification,
ledger/report checksums, question IDs/text/hash, trusted grades, model revision,
prompts, requested inference settings, and then confirms resolved model,
device and precision match after loading the target model. Cached rows retain
source checksums and are counted separately from new generations. A synthetic
campaign test verified 4 imported rows, zero target generations, and exact
cache-hit accounting. Focused training-schedule and evaluator tests passed.

These are data-provenance and efficiency improvements, not scientific findings.
They set up the next development-only order-sensitivity diagnostic: hold the
F025 support set, 16/48 task ratio, model seeds, task-specific example streams,
optimizer, and 64 updates fixed while varying only the task-order seed across
three schedules and two model seeds. Reuse base/prompt answers only when the
recorded cache compatibility checks pass. The v3 development questions are
already inspected and the F024 confirmation pool is retired, so this diagnostic
cannot support generalization or promotion. A later confirmation needs newly
generated family-disjoint examples and separate capability-preservation checks.

### F026 — Task-order sensitivity interacts with the initialization seed

Status: completed preregistered development diagnostic; adapter candidates remain
quarantined; confirmation is not available from this study.

F026 held the 64-example, family-disjoint F025 training cohort fixed (32 phrase
and 32 sentence items; 16 singular and 16 plural sentence subjects), the
16-phrase/48-sentence update ratio, optimizer, rank-4 q/v LoRA, 64 steps, model
revision, and each seed's within-task example streams. It varied only the
schedule seed that places the phrase and sentence updates in the 64-step
sequence (901, 902, 903), crossed with two LoRA initialization/training seeds
(42, 43). That produced six adapters and 384 training updates. It is an
intervention on training sequence, not on example content or task ratio.

All six adapters were evaluated on the same 428 phrase/sentence development
questions already inspected in F025. The base scored 220/428 and the explicit
prompt scored 264/428 in every run; those exact answers were checksum-verified
and reused from cache. The adapter totals by training seed / schedule seed
were:

| Training seed | Schedule seed | Adapter correct | Random-edit correct | Phrase | Sentence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 901 | 63/428 (14.7%) | 221/428 (51.6%) | 1/28 | 62/400 |
| 42 | 902 | 84/428 (19.6%) | 222/428 (51.9%) | 4/28 | 80/400 |
| 42 | 903 | 383/428 (89.5%) | 224/428 (52.3%) | 20/28 | 363/400 |
| 43 | 901 | 384/428 (89.7%) | 207/428 (48.4%) | 20/28 | 364/400 |
| 43 | 902 | 376/428 (87.9%) | 209/428 (48.8%) | 19/28 | 357/400 |
| 43 | 903 | 375/428 (87.6%) | 209/428 (48.8%) | 20/28 | 355/400 |

The trained and random adapters were norm matched per layer. Their global
effective-delta Frobenius norms differed by less than 0.000003 in each run.
The high-scoring adapters returned a single word on all 428 cases, while the
low-scoring seed-42 schedules did so on only 4/428 and 17/428. This is
consistent with broad output disruption under those two runs, not just a few
wrong completions. The random-edit scores stayed close to the base in all six
runs. The strong result under seed42/schedule903 and all seed43 schedules is
therefore not reproduced by a random edit, but is not robust to every adapter
initialization and order combination.

Within seed 42, paired family bootstrap contrasts (5,000 resamples, 32
families) showed schedule 903 minus 901 at +74.8 percentage points (324
questions improved, 4 regressed; 95% interval +69.9 to +79.4), and schedule
903 minus 902 at +69.9 points (313 improved, 14 regressed; +63.8 to +75.2).
Schedule 902 minus 901 was +4.9 points (48 improved, 27 regressed; +1.9 to
+7.7). Within seed 43, differences were small: 902 minus 901 was −1.9 points
(6 improved, 14 regressed; −3.5 to −0.2); 903 minus 901 was −2.1 (8 improved,
17 regressed; −5.6 to +1.2); and 903 minus 902 was −0.2 (20 improved, 21
regressed; −3.5 to +3.0). These intervals condition on the exposed question
families and do not account for the small number of training seeds or schedule
seeds.

The engineering conclusion is that task order can radically change the
outcome for this fixed cohort and one initialization, while another
initialization is comparatively stable across these three orders. This is
suggestive of an order-by-initialization interaction, not a generally useful
schedule or a reliable model edit. The study does not compare different
training-example content or quality; data identity, labeling, balance,
repetition, and order all need to be tracked separately in future work.
Only 428 previously inspected questions were used, including 28 phrase cases
from 7 families. The v3 confirmation cases are retired after F024. No fresh
generalization, broad preservation, causal neuron/weight explanation, or
Astra transfer is established. Keep every adapter quarantined and do not tune
on this exposed development set.

The reproducible analysis validates every training manifest and step trace,
checks the invariant per-task example streams, re-grades every raw response
with the trusted answer key, verifies response/report hashes, and computes
paired family-bootstrap contrasts:

```sh
PYTHONPATH=. python evolution_lab/evidence/f026_task_order_sensitivity/analyze_f026.py \
  --output evolution_lab/evidence/f026_task_order_sensitivity/summary.json
```

All six training checkpoints, exact per-step traces, raw answers, reports,
checksums, preregistration, and the analysis script are under
`evidence/f026_task_order_sensitivity/`. The six evaluation campaigns contain
10,272 response records: 5,136 new model generations and 5,136 verified cached
base/prompt rows. They add zero unique questions. Software test counts are
separate. Runs used the locally cached pinned Qwen2.5-0.5B-Instruct revision
on CPU float32 in this executor; this is not a fresh Apple MPS result.

**Confirmed in this diagnostic:** changing task-update order produced a very
large score shift under seed 42 with the same examples and per-task streams;
seed 43 did not show the same shift. **Tentative:** initialization and task
order interact. **Not established:** which individual weights encode the
behavior, whether the high-scoring change preserves other abilities, whether
different data changes would help, or whether this transfers to a larger model.
**Next:** build a separately preregistered factorial study of training-data
content/quality and schedule across more training seeds, with family-disjoint
fresh confirmation and unrelated capability/locality probes. Keep all exposed
F025/F026 questions out of confirmation.

### F027 — Training-example content changes completion behavior, with seed-dependent effects

Status: completed preregistered development diagnostic; no candidate is promoted.

**Question and intervention.** Holding the pinned Qwen2.5-0.5B-Instruct revision, task ratio, optimizer, LoRA settings, update count, training seeds and task-order seeds fixed, does using different valid phrase/sentence examples change completion behavior? Two 64-example cohorts were sampled from the trusted completion-v3 train split: each had 16 phrase and 48 sentence items, and balanced singular/plural sentence subjects. The cohorts shared only 3 exact examples (all phrases); they differed on 13 phrase and all 48 sentence examples apiece. The content treatment therefore changes example identity and family coverage together; this is not a test of label quality.

The 2 cohorts were crossed with training seeds 42, 43 and 44, and task-order seeds 901 and 903 (12 adapters, 768 training updates). Training used 64 updates, rank-4 q/v LoRA, AdamW learning rate 0.0005, max sequence length 256 and CPU float32. Each adapter was evaluated against the same 428 previously inspected development questions (28 phrase, 400 sentence, 32 families). This is development evidence only; the final completion-v3 confirmation pool remained unused and retired.

| Cohort | Seed | Order | Adapter | Base | Prompt | Random edit | Phrase | Sentence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 42 | 901 | 329/428 (76.9%) | 220 | 264 | 222 | 18/28 | 311/400 |
| A | 42 | 903 | 330/428 (77.1%) | 220 | 264 | 220 | 16/28 | 314/400 |
| A | 43 | 901 | 326/428 (76.2%) | 220 | 264 | 208 | 19/28 | 307/400 |
| A | 43 | 903 | 335/428 (78.3%) | 220 | 264 | 209 | 15/28 | 320/400 |
| A | 44 | 901 | 343/428 (80.1%) | 220 | 264 | 223 | 19/28 | 324/400 |
| A | 44 | 903 | 345/428 (80.6%) | 220 | 264 | 224 | 19/28 | 326/400 |
| B | 42 | 901 | 361/428 (84.3%) | 220 | 264 | 221 | 21/28 | 340/400 |
| B | 42 | 903 | 368/428 (86.0%) | 220 | 264 | 220 | 22/28 | 346/400 |
| B | 43 | 901 | 324/428 (75.7%) | 220 | 264 | 208 | 14/28 | 310/400 |
| B | 43 | 903 | 335/428 (78.3%) | 220 | 264 | 212 | 13/28 | 322/400 |
| B | 44 | 901 | 343/428 (80.1%) | 220 | 264 | 224 | 19/28 | 324/400 |
| B | 44 | 903 | 327/428 (76.4%) | 220 | 264 | 225 | 20/28 | 307/400 |

Base and prompt were checksum-verified cached responses, identical across all 12 campaigns. The norm-matched random-edit controls stayed at 208–225/428. Every trained adapter passed the one-word output-format check on all 428 items. These controls show the measured differences were not simply a different prompt, random weight noise, or malformed output. However, all questions had already been inspected, and these completion-only tasks are not a broad capability/locality check.

The preregistered paired family-bootstrap contrasts (32 families, including only 7 phrase families) show that cohort B exceeded A for seed 42 by +7.5 points under order 901 (95% interval +0.7 to +14.7) and +8.9 under order 903 (+2.1 to +16.4). For seed 43, the contrasts were −0.5 (−7.9 to +6.5) and 0.0 (−3.5 to +3.7). For seed 44, they were 0.0 (−5.1 to +5.4) and −4.2 (−8.4 to −0.2). The intervals are exploratory, conditional on the exposed families, and not multiplicity-adjusted. Thus the content difference mattered in some matched conditions, but did not produce a consistent cohort winner; it interacted with the training seed/order. The study does not isolate which lexical/family property caused the differences.

**Supports:** training example identity can be behaviorally consequential even with the task ratio and broad subject balance held fixed; documenting data and sample IDs is necessary for reproducibility.

**Does not support:** a general claim that cohort B is better, a claim about label correctness or data quality, a durable trait/weight map, independent generalization, broad capability preservation, or transfer to Astra. No candidate meets the promotion gate. Keep all adapters quarantined.

The study records 20,544 response rows: 10,272 newly generated model responses and 10,272 verified cached baseline responses, across 2,300 unique normalized question prompts overall with zero new unique prompts. It uses 768 training updates. Software tests are a separate count. Reproduce/analyze with:

```sh
PYTHONPATH=. /Users/archerjohnson/behavior-edit/.venv/bin/python \
  evolution_lab/evidence/f027_cohort_content_sensitivity/run_f027_training.py \
  --python /Users/archerjohnson/behavior-edit/.venv/bin/python \
  --model /Users/archerjohnson/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
PYTHONPATH=. /Users/archerjohnson/behavior-edit/.venv/bin/python \
  evolution_lab/evidence/f027_cohort_content_sensitivity/run_f027_evaluations.py \
  --python /Users/archerjohnson/behavior-edit/.venv/bin/python \
  --model /Users/archerjohnson/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
PYTHONPATH=. /Users/archerjohnson/behavior-edit/.venv/bin/python \
  evolution_lab/evidence/f027_cohort_content_sensitivity/analyze_f027.py \
  --output evolution_lab/evidence/f027_cohort_content_sensitivity/summary.json
```

Evidence: preregistration, cohort manifests and exact examples, sequential training/evaluation campaign manifests, 12 adapter checkpoints, per-update traces, raw ledgers, regraded reports, checksums, and analysis are under `evidence/f027_cohort_content_sensitivity/`. Model revision is pinned to `7ae557604adf67be50417f59c2c2f167def9a775`; this run used CPU float32 because MPS was unavailable in this executor. It is not a fresh Apple Silicon GPU result.

**Next:** improve data-quality controls by measuring trusted-label validity and template/family leakage separately from cohort identity, then test a preregistered sampling intervention on fresh family-disjoint confirmation data with unrelated preservation probes. The exposed v3 development and retired confirmation pools cannot serve as fresh confirmation.

### F027 data-coverage addendum

The per-update traces show that task-ratio example sampling was with
replacement. Although each support had 16 phrase and 48 sentence examples,
the 64 updates saw only 9–11 unique phrase items and 29–33 unique sentence
items, or 40–42 unique examples total. Some examples appeared up to 3–4 times.
This makes "support set" and "examples actually used by updates" different
quantities.

The displayed answer position is balanced in the full completion-v3 dataset:
the trusted answer appears first/second 3,851/4,029 times in train, 200/228 in
development, and 206/226 in confirmation. The stored `options` array puts the
key first, but that array is not the prompt shown to the model; randomized text
choices are the relevant distribution. This is a check against a trivial
position cue, not proof of phrase-label quality.

### F028 — Example repetition versus complete support coverage

Status: completed exploratory development diagnostic; no candidate promoted.

F028 used one fixed 64-example train-split cohort (16 phrase, 48 sentence;
24 singular and 24 plural subjects) and crossed three initialization seeds
with two exposure policies. The model revision, support cohort, update ratio,
task-order seed 2603, optimizer, rank, learning rate, decoding, and evaluation
cases were held fixed. With replacement, the 64 updates saw 40, 42, and 42
distinct examples for seeds 42, 43, and 44 (phrase/sentence coverage 11/29,
10/32, and 9/33). Without replacement, all three saw the full support exactly
once (16/48). Task order positions matched exactly within each seed.

On the same previously inspected 428-case development set, with-replacement
adapter scores were 323/428, 332/428, and 340/428. Without-replacement scores
were 337/428, 333/428, and 336/428. Base and prompt scored 220/428 and 264/428
for every seed; norm-matched random controls scored 209–225/428. Adapter
formatting was 428/428 for all six runs. Paired family-bootstrap contrasts
for without minus with were +3.3 points for seed 42 (95% interval +0.5 to
+6.1), +0.2 for seed 43 (−3.0 to +3.3), and −0.9 for seed 44 (−4.4 to +2.1).
The tiny, inconsistent policy differences do not support a reliable exposure
benefit. All six adapters beat the prompt on this narrow task, but phrase
completion remained only 15–19/28 versus base 24/28; this is not broad
capability preservation.

F028 generated 5,136 new model responses and reused 5,136 checksum-verified
base/prompt responses, for 10,272 response records over zero new unique
questions. It used 384 training updates. The 428 development cases and 32
families were previously inspected, so this cannot establish independent
generalization, training-label validity, broad locality, a causal trait map,
or transfer to Astra. Keep all adapters quarantined. No final confirmation
data was consumed.

Reproduce the six CPU float32 training runs with
`python evolution_lab/evidence/f028_example_exposure/run_f028_training.py`
and the evaluation with
`python evolution_lab/evidence/f028_example_exposure/run_f028_evaluations.py`;
both runners accept `--python` and `--model` overrides. The analyzer is
`python evolution_lab/evidence/f028_example_exposure/analyze_f028.py --output evolution_lab/evidence/f028_example_exposure/summary.json`.
The preregistration, sampled-ID traces, checkpoints, raw generations, graded
reports, checksums, and summary are under `evidence/f028_example_exposure/`.
MPS was built but unavailable in this executor, so F028 is CPU evidence and
does not add Apple GPU compatibility results.
