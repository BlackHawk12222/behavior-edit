# Experiment evidence

`pilot_v1/` preserves completed CPU experiments, frozen tests, adapter checkpoints,
stronger prompt controls, a fixed adapter-mixture diagnostic, MPS checks, and one
excluded interrupted run. Raw run manifests and adapter files are unchanged.
The study journal's directory links were relocated for repository portability;
recorded `source_run` strings remain historical metadata.

- `study.json`: portable links to seven full CPU development/test pairs.
- `adapter_behavior.jsonl` and `summary.json`: measured features and per-seed outcomes.
- `followups/`: fixed mixture and 16-demonstration prompt controls.
- `mps_validation/`: fresh float32 training/reload and float16-base smoke evidence.
- `counts.json`: recorded responses and unique-question accounting.
- `files.json`: SHA-256 and byte count for every raw evidence file except itself.

Adapters use safetensors; base model weights are not included. Checkpoints are
experimental candidates, not promoted models. Target gains do not erase the
recorded capability regressions. Read `../reports/pilot-v1.md` and
`../RESEARCH_GUIDE.md` before interpreting these files.

Audit the complete portable CPU registry from the repository root:

```sh
python -m evolution_lab catalog evolution_lab/evidence/pilot_v1 --output-dir results/evolution_lab/rechecked_catalog
```
