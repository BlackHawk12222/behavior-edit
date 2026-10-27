"""Sequential, paired LoRA controls using one resident base model.

No prompt baseline is described as a weight edit. Random controls are matched
per-layer to the trained effective delta norm and are not trained.
"""
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from .benchmark import load_suite, select_tasks
from .metrics import grade
from .provenance import environment, fingerprint, checkpoint_hashes
from .runner import LocalModel
from .statistics import compare_roles
from .train_adapter import BASE_SYSTEM, supervised_batch, lora_delta_stats

PROMPTS = {
    'json_format': 'You are a helpful assistant. Follow the requested JSON schema exactly. Output only valid JSON, without markdown or explanation.',
    'exact_instruction': 'You are a helpful assistant. Follow the requested output constraint exactly. Output only the requested answer without any additional text.',
    'concise_content': 'You are a helpful assistant. Give concise answers within the requested word limit and include all explicitly required content.',
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def adapter_parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in adapter_parameters(model).items()}


def restore(model, state, torch):
    params = adapter_parameters(model)
    if params.keys() != state.keys():
        raise ValueError('Adapter state keys mismatch')
    with torch.no_grad():
        for name, parameter in params.items():
            parameter.copy_(state[name].to(parameter.device))


def training_sequence(examples, steps, seed, sampling_strategy="uniform",
                     task_exposure_ratio=None, schedule_seed=None,
                     example_sampling="with-replacement"):
    """Return a deterministic training schedule for ablation-ready sampling."""
    if steps < 1 or not examples:
        raise ValueError("Training schedule requires examples and positive steps")
    if example_sampling not in {"with-replacement", "without-replacement"}:
        raise ValueError("Example sampling must be with-replacement or without-replacement")
    if sampling_strategy != "task-ratio" and example_sampling != "with-replacement":
        raise ValueError("Example sampling policy is only configurable for task-ratio sampling")
    rng = random.Random(seed)
    if sampling_strategy == "uniform":
        # Preserve the historical shuffled-epoch schedule exactly.
        order = list(examples)
        sequence = []
        for step in range(steps):
            if step % len(order) == 0:
                rng.shuffle(order)
            sequence.append(order[step % len(order)])
        return sequence
    if sampling_strategy != "task-balanced":
        if sampling_strategy != "task-ratio":
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

        groups = {}
        for example in examples:
            kind = example.get("task_type", "default")
            groups.setdefault(kind, []).append(example)
        if len(groups) < 2:
            raise ValueError("Task-ratio sampling requires at least two task types")
        if not isinstance(task_exposure_ratio, dict) or set(task_exposure_ratio) != set(groups):
            raise ValueError("Task-ratio weights must name every cohort task type exactly")
        if any(type(value) is not int or value < 0 for value in task_exposure_ratio.values()):
            raise ValueError("Task-ratio weights must be non-negative integers")
        if sum(task_exposure_ratio.values()) != steps or not all(task_exposure_ratio.values()):
            raise ValueError("Task-ratio counts must be positive and sum to the optimizer steps")

        # Keep each task's example stream independent of the task ratio. Across
        # paired conditions with the same seed, the same task-specific examples
        # are therefore consumed in the same order; the intervention changes
        # only the number and placement of exposures from each task.
        import hashlib
        task_streams = {}
        task_positions = {kind: 0 for kind in groups}
        for kind, group in groups.items():
            digest = hashlib.sha256(f"{seed}:{kind}".encode()).digest()
            stream_seed = int.from_bytes(digest[:8], "big")
            stream_rng = random.Random(stream_seed)
            if example_sampling == "with-replacement":
                task_streams[kind] = [group[stream_rng.randrange(len(group))]
                                      for _ in range(steps)]
            else:
                order = list(group)
                stream = []
                for position in range(steps):
                    if position % len(order) == 0:
                        stream_rng.shuffle(order)
                    stream.append(order[position % len(order)])
                task_streams[kind] = stream
        if schedule_seed is None:
            # Preserve the original task-ratio order exactly for old manifests
            # and commands that did not register a separate order seed.
            task_schedule = [kind for kind, count in sorted(task_exposure_ratio.items())
                             for _ in range(count)]
            random.Random(seed).shuffle(task_schedule)
        else:
            task_types = sorted(task_exposure_ratio)
            if len(task_types) == 2:
                # Common random positions couple paired ratios: the first task
                # receives a nested prefix of one seeded position permutation.
                # The model/data seed remains responsible for adapter init and
                # within-task example streams, independently of task ordering.
                positions = list(range(steps))
                random.Random(schedule_seed).shuffle(positions)
                first, second = task_types
                first_positions = set(positions[:task_exposure_ratio[first]])
                task_schedule = [first if position in first_positions else second
                                 for position in range(steps)]
            else:
                # Keep the API usable for future multi-task cohorts. The
                # two-task nested-position guarantee does not apply here.
                task_schedule = [kind for kind, count in sorted(task_exposure_ratio.items())
                                 for _ in range(count)]
                random.Random(schedule_seed).shuffle(task_schedule)
        sequence = []
        for kind in task_schedule:
            position = task_positions[kind]
            sequence.append(task_streams[kind][position])
            task_positions[kind] += 1
        return sequence

    groups = {}
    for example in examples:
        kind = example.get("task_type", "default")
        groups.setdefault(kind, []).append(example)
    if len(groups) < 2:
        raise ValueError("Task-balanced sampling requires at least two task types")
    task_types = sorted(groups)
    for group in groups.values():
        rng.shuffle(group)
    offsets = {kind: 0 for kind in task_types}
    sequence = []
    for step in range(steps):
        kind = task_types[step % len(task_types)]
        if offsets[kind] == len(groups[kind]):
            rng.shuffle(groups[kind])
            offsets[kind] = 0
        sequence.append(groups[kind][offsets[kind]])
        offsets[kind] += 1
    return sequence


def fit(local, examples, steps, learning_rate, max_length, seed,
        sampling_strategy="uniform", task_exposure_ratio=None,
        schedule_seed=None, example_sampling="with-replacement"):
    import torch
    params = list(adapter_parameters(local.model).values())
    for p in params:
        p.data = p.data.float()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, foreach=False)
    schedule = training_sequence(examples, steps, seed, sampling_strategy,
                                 task_exposure_ratio, schedule_seed,
                                 example_sampling)
    history = []
    local.model.config.use_cache = False
    local.model.train()
    try:
        for step in range(steps):
            ex = schedule[step]
            inputs = supervised_batch(local.tokenizer, ex, max_length, local.device)
            optimizer.zero_grad(set_to_none=True)
            loss = local.model(**inputs).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError(f'Non-finite loss at step {step+1}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(p).all().item() for p in params):
                raise RuntimeError(f'Non-finite weights at step {step+1}')
            history.append({'step':step+1,'example_id':ex['id'], 'task_type':ex.get('task_type','default'),
                            'loss':float(loss.detach()),
                            'gradient_norm_before_clip':float(norm)})
            print(f'train {step+1}/{steps}: loss={history[-1]["loss"]:.5f}', flush=True)
    finally:
        local.model.eval()
        local.model.config.use_cache = True
    return history


def norm_matched_random(model, torch, seed, adapter_name="default"):
    """Randomize one adapter while preserving its per-module effective ΔW norms."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if (not hasattr(module, 'lora_A')
                    or adapter_name not in module.lora_A
                    or adapter_name not in module.lora_B):
                continue
            a = module.lora_A[adapter_name].weight
            b = module.lora_B[adapter_name].weight
            ac, bc = a.detach().float().cpu(), b.detach().float().cpu()
            norm = torch.linalg.vector_norm(bc @ ac)
            ra = torch.randn(ac.shape, generator=generator, dtype=torch.float32)
            rb = torch.randn(bc.shape, generator=generator, dtype=torch.float32)
            denom = torch.linalg.vector_norm(rb @ ra)
            rb *= norm / denom.clamp_min(1e-30)
            a.copy_(ra.to(a.device)); b.copy_(rb.to(b.device))


def evaluate(local, tasks, condition, stream, system=BASE_SYSTEM, token_limit=48):
    rows = []
    for task in tasks:
        response, tokens, elapsed = local.generate(system, task['prompt'], token_limit)
        row = {'condition':condition, 'task_id':task['id'], 'task':task,
               'system_prompt':system, 'response':response, 'generated_tokens':tokens,
               'hit_token_limit':tokens >= token_limit,
               'latency_seconds':elapsed, 'metrics':grade(task,response)}
        rows.append(row)
        stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
        print(f'{condition:10} {task["id"]}: {row["metrics"]["passed"]}', flush=True)
    return rows


def run_experiment(args):
    if (args.steps < 1 or args.rank < 1 or args.max_length < 16 or args.max_new_tokens < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError('Invalid training settings')
    suite = load_suite(args.suite)
    tasks = select_tasks(suite, args.behavior, args.split, args.limit_per_role)
    examples = [r for r in suite['training'] if r['behavior'] == args.behavior]
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    random.seed(args.seed); torch.manual_seed(args.seed)
    local = LocalModel(args.model, revision=args.revision, device=args.device, precision=args.precision)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination = args.output_dir / run_id
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version':1,'run_id':run_id,'status':'running',
                'environment':environment(), 'model_id':args.model,
                'requested_revision':args.revision, 'resolved_revision':local.resolved_revision,
                'device':local.device,'precision':local.precision,'seed':args.seed,
                'suite_id':suite['suite_id'],'suite_sha256':fingerprint(suite),
                'training_sha256':fingerprint(examples),'evaluation_sha256':fingerprint(tasks),
                'behavior':args.behavior, 'split':args.split,'smoke_only':args.limit_per_role is not None,
                'settings':{'steps':args.steps,'rank':args.rank,'learning_rate':args.learning_rate,
                            'max_length':args.max_length,'max_new_tokens':args.max_new_tokens,
                            'decoding':'greedy','lora_alpha':2*args.rank,'lora_dropout':0.,
                            'target_modules':['q_proj','v_proj'],'optimizer':'AdamW',
                            'weight_decay':0.01,'clip_grad_norm':1.},
                'training_ids':[r['id'] for r in examples], 'tasks':tasks,
                'controls':['base','prompt','zero','lora','random'],
                'warning':'Diagnostic only. Test results must not be used to tune this benchmark. '
                          'No personality, intelligence, or generalized improvement claim.'}
    write_json(destination/'manifest.json',manifest)
    rows = {}
    try:
        with (destination/'observations.jsonl').open('w') as stream:
            rows['base'] = evaluate(local,tasks,'base',stream,token_limit=args.max_new_tokens)
            rows['prompt'] = evaluate(local,tasks,'prompt',stream,system=PROMPTS[args.behavior],token_limit=args.max_new_tokens)
            # Seed immediately before initialization, independent of previous generation calls.
            torch.manual_seed(args.seed)
            local.model = get_peft_model(local.model, LoraConfig(task_type=TaskType.CAUSAL_LM,
                r=args.rank, lora_alpha=2*args.rank,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none'))
            local.model.eval()
            zero = snapshot(local.model)
            rows['zero'] = evaluate(local,tasks,'zero',stream,token_limit=args.max_new_tokens)
            if any(a['response'] != b['response'] for a,b in zip(rows['base'],rows['zero'])):
                raise RuntimeError('Zero-initialized adapter changed outputs; controls are invalid')
            history = fit(local,examples,args.steps,args.learning_rate,args.max_length,args.seed)
            write_json(destination/'training.json',history)
            trained = snapshot(local.model)
            delta = lora_delta_stats(local.model,torch)
            write_json(destination/'delta_lora.json',delta)
            local.model.peft_config['default'].revision = local.resolved_revision
            local.model.save_pretrained(str(destination/'adapter_lora'),safe_serialization=True)
            rows['lora'] = evaluate(local,tasks,'lora',stream,token_limit=args.max_new_tokens)
            norm_matched_random(local.model,torch,args.seed+1000003)
            random_delta = lora_delta_stats(local.model,torch)
            write_json(destination/'delta_random.json',random_delta)
            for a,b in zip(delta['layers'],random_delta['layers']):
                if a['name'] != b['name'] or not math.isclose(a['delta_frobenius_norm'],b['delta_frobenius_norm'],rel_tol=1e-4,abs_tol=1e-8):
                    raise RuntimeError('Random control delta norm mismatch')
            local.model.save_pretrained(str(destination/'adapter_random'),safe_serialization=True)
            rows['random'] = evaluate(local,tasks,'random',stream,token_limit=args.max_new_tokens)
            restore(local.model,trained,torch)
        manifest['adapter_hashes'] = {arm:checkpoint_hashes(destination/f'adapter_{arm}') for arm in ('lora','random')}
        report = {condition:compare_roles(rows['base'],values,args.seed)
                  for condition,values in rows.items() if condition != 'base'}
        report['lora_vs_prompt'] = compare_roles(rows['prompt'], rows['lora'], args.seed)
        report['lora_vs_random'] = compare_roles(rows['random'], rows['lora'], args.seed)
        write_json(destination/'report.json',report)
        manifest.update(status='complete', zero_control_identical=True)
    except BaseException as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination/'manifest.json',manifest)
    print(f'Experiment saved: {destination}',flush=True)
    return 0
