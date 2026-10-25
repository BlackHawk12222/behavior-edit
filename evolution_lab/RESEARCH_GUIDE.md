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
