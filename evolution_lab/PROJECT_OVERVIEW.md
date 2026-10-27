# AI Model Evolution Engine — plain-language overview

## What this project is trying to do

This project is a small, careful workshop for studying how a language model's
answers change when we make a small change to its internal weights. The long-
term aim is to learn how to make a model more reliably show a requested,
measurable behavior, and to learn which parts of the model contributed to that
change.

The project does **not** currently edit Astra or another huge model. The
experiments use a downloadable Qwen2.5-0.5B-Instruct model, which is small enough
to run locally. The methods and results may not transfer to a larger model.

## How one experiment works

1. Make questions with answers that can be checked independently. For example,
   an arithmetic question has an answer computed by normal integer arithmetic;
   a sentence blank has an answer determined by the written subject and verb.
2. Give the questions to the untouched model and save its answers.
3. Train a small **LoRA adapter** on a separate set of examples. A LoRA adapter
   is a compact set of learned weight changes; it is not a rewritten full
   model.
4. Ask the changed model the same test questions and have a deterministic
   grader compare answers with the trusted keys. The model does not grade
   itself.
5. Compare the result with the untouched model, a strong prompt-only control,
   and a random adapter with a similar weight-change size. This helps tell
   whether the improvement came from the learned change or from asking more
   clearly or merely perturbing the weights.
6. Check different kinds of questions separately. A model might improve one
   behavior while damaging another.

Training examples matter a great deal. If the adapter sees mostly one kind of
sentence, vocabulary, answer form, or phrase, it can learn a narrow bias. The
project records the actual examples sampled, their families and answer types,
and the sampling method. A future comparison should change one part of the
training data at a time when possible.

## Keep learning and testing separate

Each generated dataset has a training split for learning, a development split
for choosing an experiment, and a confirmation split reserved for one final
check. If a confirmation result is inspected, those exact questions are
retired from later tuning. Reusing the same question many times creates more
model responses, but not more unique questions.

The project reports the number of unique questions separately from model
generations, training updates, cached/reused responses, and software tests. A
large number of repeated answers is not the same as a large independent test.

## What is built so far

- A local command-line lab for validating tasks, generating questions,
  training LoRA and matched-random adapters, evaluating several controls, and
  saving raw answers and measured weight-delta summaries.
- A deterministic arithmetic generator with 100,000 unique addition cases and
  family-separated splits. Only a subset has been run through a model.
- Phrase and sentence completion benchmarks. The latest benchmark includes
  common phrases and sentences where a misleading nearby noun separates the
  true subject from the missing verb. Its training, development and
  confirmation sentence splits use different subject and verb vocabularies.
- A research guide and experiment registry that keep successful, mixed and
  failed results, with checkpoints and raw evidence.
- Existing upstream Behavior Editing code remains intact; the new research
  lab is developed separately on the feature branch.

## What recent results mean

In an earlier, separate arithmetic check, one frozen adapter scored 107/128
versus 78/128 for the prompt. That was a narrow result on one adapter and one
addition task, not evidence of broad reasoning ability.

On the phrase/sentence development benchmark, two task-balanced training
seeds both answered all plural-subject sentences correctly but had weak
singular-subject accuracy. A preregistered uniform-sampling run selected 1
phrase and 63 sentence examples and answered 358/400 sentences correctly,
including 159/200 singular and 199/200 plural. On the already-inspected
428-question development set it scored 376/428 overall versus 264/428 for the
explicit prompt (paired family-bootstrap 95% interval +20.8 to +31.3 points).
Phrase accuracy was 18/28, below the base model's 23/28. This is a strong
development result with a phrase trade-off, not independent confirmation. The
two schedules selected entirely different 64-example samples, so task ratio
and example identity are intertwined. The adapter remained quarantined until
the frozen confirmation below.

That check has now finished. On 432 new phrase/sentence questions, the frozen
adapter answered 409 correctly, including 390/400 sentences and 19/32 phrases.
It beat the explicit prompt (264/432) by 33.6 percentage points, with a
family-bootstrap 95% interval of +28.7 to +37.7. Its phrase score was still
below the untouched model (26/32), so this is strong evidence for a narrow
sentence-completion behavior with a phrase trade-off. A fixed mixture of the
balanced and uniform adapters scored 388/432; the uniform adapter was 4.9
points higher, but the mixture has twice the adapter rank and is not an
equal-capacity comparison.

On a separate 312-question addition check, the same adapter scored 99/312.
That was above the untouched model and matched-random edit (both 54/312), but
below the clear prompt control (155/312) by 17.9 points. It therefore does not
show broad capability preservation. The small phrase family count, synthetic
English tasks, and one candidate seed also limit what we can conclude. No
adapter has been promoted as a generally beneficial change; both F024
confirmation families are retired and cannot be reused for tuning.

Across archived answer logs through F024, the project has 29,864 response
records on 2,300 distinct normalized prompts. Many records repeat a question
under different arms or seeds. The arithmetic generator can create 100,000
questions, but generated questions are counted separately from model-tested
questions; this project has not asked the model millions of unique questions.

## Current limits

- The tested model is a small Qwen checkpoint, not Astra.
- Most new experiments here ran on CPU float32 because this session could not
  access MPS. Earlier MPS results are recorded separately and should not be
  confused with these CPU runs.
- The current text benchmark is synthetic and English-only. Its phrase list is
  culturally limited, and its grammar probe tests a narrow agreement rule.
- A measured relationship between an adapter and answers is not yet a general
  map from individual weights to human traits.

## Reproducing the latest text benchmark

From the repository root:

```sh
python -m evolution_lab generate-completions-v3 --validate-only \
  --output evolution_lab/evidence/completion_v3
python -m unittest discover -s evolution_lab/tests -q
```

The detailed experiment commands and the full research record are in
[`README.md`](README.md) and [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md).
