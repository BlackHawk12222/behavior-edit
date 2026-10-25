"""Evaluate frozen saved adapters on untouched test families without retraining."""
import json
from datetime import datetime, timezone
from pathlib import Path
from .benchmark import load_suite, select_tasks
from .experiment import evaluate, PROMPTS, write_json
from .provenance import environment, fingerprint, checkpoint_hashes
from .runner import LocalModel
from .statistics import compare_roles


def evaluate_checkpoint(args):
    source=Path(args.run_dir)
    parent=json.loads((source/'manifest.json').read_text())
    if parent['status']!='complete' or parent['split']!='dev' or parent['smoke_only']:
        raise ValueError('Frozen test evaluation requires a complete, full development experiment')
    suite=load_suite(args.suite)
    if fingerprint(suite)!=parent['suite_sha256']:
        raise ValueError('Suite changed since adapter training')
    hashes={arm:checkpoint_hashes(source/f'adapter_{arm}') for arm in ('lora','random')}
    if 'adapter_hashes' in parent and hashes != parent['adapter_hashes']:
        raise ValueError('Saved adapter files changed since development evaluation')
    tasks=select_tasks(suite,parent['behavior'],'test')
    # Require the immutable resolved revision recorded during training.
    revision=parent['resolved_revision']
    if len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision.lower()):
        raise ValueError('Test evaluation requires an immutable resolved model commit SHA')
    local=LocalModel(parent['model_id'],revision=revision,device=args.device,
                     precision=parent['precision'].split('.')[-1])
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination=args.output_dir/run_id;destination.mkdir(parents=True,exist_ok=False)
    manifest={**parent,'run_id':run_id,'status':'running','split':'test',
              'source_run':str(source), 'source_manifest_sha256':fingerprint(parent),
              'environment':environment(),'device':local.device,
              'evaluation_sha256':fingerprint(tasks),'tasks':tasks,
              'adapter_hashes':hashes,
              'legacy_checkpoint_hashes_missing':'adapter_hashes' not in parent,
              'controls':['base','prompt','lora','random'],
              'protocol':'Frozen adapter test; no training or selection on this split.'}
    write_json(destination/'manifest.json',manifest)
    rows={}
    try:
        with (destination/'observations.jsonl').open('w') as stream:
            limit=parent['settings']['max_new_tokens']
            rows['base']=evaluate(local,tasks,'base',stream,token_limit=limit)
            rows['prompt']=evaluate(local,tasks,'prompt',stream,system=PROMPTS[parent['behavior']],token_limit=limit)
            from peft import PeftModel
            local.model=PeftModel.from_pretrained(local.model,str(source/'adapter_lora'),is_trainable=False)
            local.model.eval()
            rows['lora']=evaluate(local,tasks,'lora',stream,token_limit=limit)
            local.model.load_adapter(str(source/'adapter_random'),adapter_name='random',is_trainable=False)
            local.model.set_adapter('random');local.model.eval()
            rows['random']=evaluate(local,tasks,'random',stream,token_limit=limit)
        report={condition:compare_roles(rows['base'],values,parent['seed'])
                for condition,values in rows.items() if condition!='base'}
        report['lora_vs_prompt']=compare_roles(rows['prompt'],rows['lora'],parent['seed'])
        report['lora_vs_random']=compare_roles(rows['random'],rows['lora'],parent['seed'])
        write_json(destination/'report.json',report)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(destination/'manifest.json',manifest)
    print(f'Frozen test saved: {destination}',flush=True)
    return 0
