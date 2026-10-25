# Evolution Lab v0.3: controlled diagnostic study

Engineering and CPU experiments completed in the active session that began October 24, 2026 (America/Chicago). Run IDs use UTC.

## Outcome

Targeted LoRA produced a large, repeatable JSON-output effect on this small synthetic diagnostic, with capability-preservation failures in the frozen test split. No adapter met the full development-plus-test evidence gate. These results do not justify a hypernetwork, architecture-growth claim, or model promotion.

## Evidence census

2,092 model responses are recorded across 94 unique evaluation prompts and 131
training updates. Repeated prompts across seeds, conditions and devices are not
new independent questions. Software tests are counted separately. The raw
census is in `evolution_lab/evidence/pilot_v1/counts.json`.

The cumulative guide is `evolution_lab/RESEARCH_GUIDE.md`; the study index is
`evolution_lab/experiment_registry.json`.

## Frozen test results

Each target score is out of 12. Preservation is out of 16. These are the same public authored items across seeds; do not pool seeds as independent test items.

| Behavior | Seed | Base | System prompt | Random edit | LoRA | Preservation base → LoRA | Pass-to-fail losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| json_format | 42 | 0 | 0 | 0 | 12 | 11 → 9 | 2 |
| concise_content | 42 | 6 | 7 | 6 | 8 | 11 → 11 | 0 |
| exact_instruction | 42 | 3 | 1 | 3 | 4 | 11 → 11 | 0 |
| json_format | 43 | 0 | 0 | 0 | 12 | 11 → 10 | 1 |
| json_format | 44 | 0 | 0 | 0 | 12 | 11 → 10 | 1 |
| concise_content | 43 | 6 | 7 | 5 | 8 | 11 → 11 | 0 |
| exact_instruction | 43 | 3 | 1 | 3 | 1 | 11 → 11 | 0 |

## Stronger prompt-only follow-up

This retrospective control uses all 16 training examples as real conversation demonstrations, with no test answers included. Every plain-base output was regenerated and required to reproduce its frozen reference exactly. Longer context and adapter training have different costs; these are behavioral comparisons, not compute-efficiency comparisons.

| Behavior (seed-42 reference) | Base / 12 | 16-example prompt / 12 | LoRA / 12 | Prompt preservation / 16 |
|---|---:|---:|---:|---:|
| json_format | 0 | 0 | 12 | 10 |
| concise_content | 6 | 10 | 8 | 11 |
| exact_instruction | 3 | 3 | 4 | 12 |

## Fixed adapter mixture (development only)

The JSON and concise-content seed-42 adapters were combined at fixed 0.5 / 0.5 coefficients using exact effective-delta concatenation. Coefficients were not fitted. Two rank-4 sources produce rank 8, so this is not a rank-budget-matched comparison against a trained rank-8 adapter. The mixture was not tested on an independent mixture holdout.

| Probe group | Base | JSON source | Concise source | Equal mixture |
|---|---:|---:|---:|---:|
| json_format | 0/8 | 7/8 | 0/8 | 7/8 |
| concise_content | 4/8 | 7/8 | 5/8 | 5/8 |
| preservation | 10/16 | 10/16 | 10/16 | 10/16 |

## Protocol and validity

- Qwen/Qwen2.5-0.5B-Instruct, immutable revision `7ae557604adf67be50417f59c2c2f167def9a775`; CPU float32; greedy decoding with 48 new tokens. A separate MPS hardware-portability check is reported below.
- Each trained adapter: 16 updates on 16 separate behavior examples, rank 4, alpha 8, q/v projections, no dropout, AdamW at 0.0005. Training IDs, actual step order, losses, gradient norms and package versions are retained.
- Versioned suite: 48 training examples, 24 development target probes, 36 test target probes, and 16 preservation probes in each evaluation split. Target content/schema families differ across splits. This is manually authored family separation, not a proof of semantic independence or absence of pretraining contamination.
- Five development controls: base, explicit system instruction, zero-initialized adapter, conventional LoRA, and random adapter matched to the trained effective weight-delta norm per layer. Zero adapters reproduced all base responses exactly. Frozen tests reload the saved LoRA and random checkpoints without further training.
- Target scores are reported separately from preservation, with gained/lost passes, raw changed responses, per-category results, Wilson intervals and family-bootstrap intervals. Wilson intervals assume independent items; target family count is one per split, so its family-bootstrap interval is unavailable. Few families and repeated items severely limit inference.
- The diagnostic gate requires a target gain of at least 25 percentage points, strict superiority to system-prompt and random controls, and zero preservation pass-to-fail changes on both dev and test. It is retrospective and descriptive, not a preregistered significance test. The stronger 16-example prompt was an additional retrospective adversarial control.
- The first two development experiments predated checkpoint hashes at development completion. Their files were retained unchanged and hashed at frozen-test time; they are exploratory legacy evidence. Some exploratory runs recorded a dirty working tree. All recorded dirty-state flags and source revisions remain visible; they are not rewritten as clean provenance.
- One overlapping frozen test was explicitly interrupted due to memory pressure. Its failed manifest and partial output are preserved under `interruptions/`; it is excluded from results. Subsequent runs were sequential in separate processes. OMP/MKL thread settings and runtime versions are recorded where applicable.
- Exact-answer preservation combines task correctness and requested output format. Inspected JSON-adapter regressions included substantive errors: `47 + 11` changed from `58` to `68`; seed 42 also changed Canada’s capital from `Ottawa` to `Toronto`.
- Concise-content grading checks a word budget and required literal content. It does not assess open-ended usefulness, politeness, personality, intelligence, values or self-reported traits.
- Layer norms and a seven-row adapter/behavior catalog establish measured associations only. The catalog is far too small and correlated to fit or validate an edit predictor. Test outcomes must not become fitting/selection labels.

## Engineering delivered

Upstream research code is unchanged. The feature branch adds normalized train/evaluation leakage checks, strict JSON grading, stable assistant-only loss masking, explicit training precision, finite-loss/gradient/weight checks, checkpoint fingerprints, streamed observations, auditable hypothesis verification, resumable fixed studies, measured-feature catalogs, fixed exact-delta mixtures, and all-training-example prompt controls.

43 local tests passed: 39 dependency-free checks and four optional real-tensor integration tests. The tensor tests use a tiny random Llama without network downloads and verify frozen base weights, training updates, save/reload equivalence, random norm matching, exact mixture algebra, token masking, and demonstration chat roles. GitHub CI has separate pure-Python and CPU tensor jobs. A one-step real Qwen CPU smoke training run also completed; its two arithmetic scores stayed 2/2.

## Reproduce and audit

```sh
cd ~/behavior-edit
git pull --ff-only
source .venv/bin/activate
python -m pip install -r evolution_lab/requirements-train-mac.txt
python -m evolution_lab benchmark
python -m unittest discover -s evolution_lab/tests -v
python -m evolution_lab catalog evolution_lab/evidence/pilot_v1 --output-dir results/evolution_lab/rechecked_catalog
python -m evolution_lab experiment --behavior json_format --device auto --steps 1 --limit-per-role 2
```

After the short hardware check, use `--steps 16` without `--limit-per-role` for a full development run, then `test-checkpoint` on that saved run. The default training precision is now float32. The optional float16 frozen base retains float32 adapter parameters. The hardware checks below cover this host/package combination only.

For the fixed seven-trial matrix:

```sh
python -m evolution_lab study evolution_lab/studies/pilot_v1.json --device auto --output-dir results/evolution_lab/reproduction
```

Local reuse entries are cache hints; a fresh checkout trains those trials from scratch. Runtime, memory headroom and exact floating-point outputs can vary by hardware and package version. Use the recorded immutable model revision and inspect any failures rather than silently substituting a new checkpoint.

## Fresh MPS hardware verification

Direct GPU execution succeeded on host `Mac17,5`, Apple A18 Pro, 8 GB unified memory, macOS 27.2, Python 3.11.17, PyTorch 2.14.1, Transformers 4.57.6 and PEFT 0.18.1. This is a hardware-portability check after CPU results were inspected, not a new independent scientific holdout.

- Float32 MPS: full 16-step JSON development experiment and frozen checkpoint test completed. Test target 0/12 → 12/12; preservation 11/16 → 9/16; 2 pass-to-fail changes.
- Float16 frozen base with float32 adapter parameters: 2-step MPS smoke check completed with finite losses/gradients/weights and matched random-delta norms. It used two target and two preservation probes, so it provides no quality or long-run reliability claim.
- Both paths saved adapters and raw outputs. Zero-adapter equivalence and checkpoint integrity checks passed. Peak memory was not instrumented; no memory-headroom or throughput guarantee follows.

## Highest-value next work

1. Repeat the MPS checks on other package/OS combinations and longer runs, measuring peak memory rather than treating success on this host as a compatibility guarantee.
2. Expand independent task families and content before selecting adapters. Investigate preservation-aware training using a separate replay pool or KL anchor, tuning only on development probes. The currently inspected test set must be retired from final confirmation after such tuning.
3. Add stronger diverse prompt controls and a rank-budget-matched trained comparator before treating mixture composition as an improvement.
4. Collect many more independently validated adapters, then test simple conditional adapter selection against prompting, random selection and fixed mixtures. Keep a fresh final holdout; only consider a hypernetwork if those baselines leave a measured gap.

The draft PR remains unmerged. No model was promoted or deployed.
