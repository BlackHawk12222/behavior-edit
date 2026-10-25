"""Stronger prompt-only control using all the adapter's training demonstrations.

This follow-up is retrospective and uses no test answers to construct the prompt.
"""
import json
from datetime import datetime, timezone
from .benchmark import load_suite
from .evidence import read_run
from .experiment import write_json, PROMPTS
from .metrics import grade
from .provenance import environment, fingerprint
from .runner import LocalModel
from .statistics import compare_roles


def evaluate_fewshot(args):
    source,reference=read_run(args.run_dir)
    suite=load_suite(args.suite)
    if fingerprint(suite)!=source['suite_sha256']:
        raise ValueError('Few-shot suite mismatch')
    examples=[row for row in suite['training'] if row['behavior']==source['behavior']]
    if fingerprint(examples)!=source['training_sha256']:
        raise ValueError('Few-shot examples differ from adapter training examples')
    local=LocalModel(source['model_id'],revision=source['resolved_revision'],device=args.device,
                     precision=source['precision'].split('.')[-1])
    if local.device!=source['device']:
        raise ValueError('Follow-up control must use the same device type as the reference run')
    destination=args.output_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'status':'running','kind':'all_training_examples_prompt_control',
              'source_manifest_sha256':fingerprint(source),'source_run':str(args.run_dir),
              'split':source['split'],'behavior':source['behavior'],
              'model_id':source['model_id'],'resolved_revision':local.resolved_revision,
              'device':local.device,'precision':local.precision,'environment':environment(),
              'demonstrations':examples,'tasks':source['tasks'],
              'max_new_tokens':source['settings']['max_new_tokens'],
              'warning':'Retrospective stronger prompt baseline. Uses all 16 training demonstrations '
                        'as actual user/assistant turns; no evaluation answers included. '
                        'Prompting changes input, not model weights. Base responses must reproduce reference.'}
    write_json(destination/'manifest.json',manifest)
    rows=[]
    try:
        old={r['task_id']:r for r in reference['base']}
        limit=source['settings']['max_new_tokens']
        with (destination/'observations.jsonl').open('w') as stream:
            for task in source['tasks']:
                # Establish an unchanged base reference in this process/device environment.
                response,_,_=local.generate('You are a helpful assistant.',task['prompt'],limit)
                if response!=old[task['id']]['response']:
                    raise RuntimeError('Base output differs from reference; historical comparison is invalid')
                response,tokens,elapsed=local.generate(PROMPTS[source['behavior']],task['prompt'],limit,
                                                       demonstrations=examples)
                row={'condition':'fewshot16','task_id':task['id'],'task':task,'response':response,
                     'system_prompt':PROMPTS[source['behavior']],
                     'demonstration_ids':[r['id'] for r in examples],
                     'generated_tokens':tokens,'hit_token_limit':tokens>=limit,
                     'latency_seconds':elapsed,'metrics':grade(task,response)}
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                print(f'fewshot16 {task["id"]}: {row["metrics"]["passed"]}',flush=True)
        report={'fewshot_vs_base':compare_roles(reference['base'],rows),
                'lora_vs_fewshot':compare_roles(rows,reference['lora']),
                'base_reference_reproduced':True}
        write_json(destination/'report.json',report)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:write_json(destination/'manifest.json',manifest)
    print(f'Few-shot control saved: {destination}',flush=True)
    return 0
