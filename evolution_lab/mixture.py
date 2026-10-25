"""A fixed equal-coefficient adapter mixture baseline, evaluated on development only."""
import json
from datetime import datetime, timezone
from .benchmark import load_suite, select_tasks
from .evidence import read_run
from .experiment import evaluate, write_json
from .provenance import checkpoint_hashes, environment, fingerprint
from .runner import LocalModel
from .statistics import compare_roles, paired_report
from .train_adapter import lora_delta_stats


def equal_mixture(model,names):
    model.add_weighted_adapter(names,[1/len(names)]*len(names),'mixture',combination_type='cat')
    model.set_adapter('mixture');model.eval()


def evaluate_mixture(args):
    if len(args.run_dirs)!=2:
        raise ValueError('This diagnostic supports exactly two distinct behavior adapters')
    sources=[read_run(p)[0] for p in args.run_dirs]
    first,second=sources
    if any(p['split']!='dev' for p in sources) or first['behavior']==second['behavior']:
        raise ValueError('Require two full development runs targeting distinct behaviors')
    for key in ('model_id','resolved_revision','suite_sha256','precision'):
        if first[key]!=second[key]:raise ValueError(f'Incompatible mixture sources: {key}')
    if first['settings']['max_new_tokens']!=second['settings']['max_new_tokens']:
        raise ValueError('Mixture decoding settings must match')
    suite=load_suite(args.suite)
    if fingerprint(suite)!=first['suite_sha256']:raise ValueError('Mixture suite mismatch')
    hashes=[checkpoint_hashes(p/'adapter_lora') for p in args.run_dirs]
    for source,current in zip(sources,hashes):
        if 'adapter_hashes' in source and current!=source['adapter_hashes']['lora']:
            raise ValueError('Mixture source checkpoint was modified')
    tasks={}
    for source in sources:
        for row in select_tasks(suite,source['behavior'],'dev'):tasks[row['id']]=row
    tasks=list(tasks.values())
    local=LocalModel(first['model_id'],revision=first['resolved_revision'],device=args.device,
                     precision=first['precision'].split('.')[-1])
    destination=args.output_dir/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    destination.mkdir(parents=True,exist_ok=False)
    manifest={'schema_version':1,'status':'running','split':'dev', 'kind':'fixed_equal_mixture',
              'source_manifests':[fingerprint(s) for s in sources],
              'source_adapter_hashes':hashes,'coefficients':[.5,.5],
              'source_ranks':[s['settings']['rank'] for s in sources],
              'mixture_rank':sum(s['settings']['rank'] for s in sources),
              'model_id':first['model_id'],'resolved_revision':local.resolved_revision,
              'precision':local.precision,'device':local.device,'environment':environment(),
              'suite_sha256':fingerprint(suite),'tasks':tasks,'evaluation_sha256':fingerprint(tasks),
              'warning':'Fixed nonlearned baseline on development data. Concatenation sums effective deltas exactly '
                        'but uses the sum of source ranks. No rank-budget-matched trained comparator, '
                        'independent mixture test, or claim of better generalization.'}
    write_json(destination/'manifest.json',manifest)
    rows={}
    try:
        with (destination/'observations.jsonl').open('w') as stream:
            limit=first['settings']['max_new_tokens']
            rows['base']=evaluate(local,tasks,'base',stream,token_limit=limit)
            from peft import PeftModel
            import torch
            local.model=PeftModel.from_pretrained(local.model,str(args.run_dirs[0]/'adapter_lora'),adapter_name='source0')
            local.model.load_adapter(str(args.run_dirs[1]/'adapter_lora'),adapter_name='source1')
            for name in ('source0','source1'):
                local.model.set_adapter(name);local.model.eval()
                rows[name]=evaluate(local,tasks,name,stream,token_limit=limit)
            equal_mixture(local.model,['source0','source1'])
            rows['mixture']=evaluate(local,tasks,'mixture',stream,token_limit=limit)
            write_json(destination/'delta_mixture.json',lora_delta_stats(local.model,torch,'mixture'))
            local.model.save_pretrained(str(destination/'adapter'),selected_adapters=['mixture'],safe_serialization=True)
        report={arm:{'roles':compare_roles(rows['base'],values),
                     'per_behavior':{behavior:paired_report(
                         [r for r in rows['base'] if r['task']['behavior']==behavior],
                         [r for r in values if r['task']['behavior']==behavior])
                         for behavior in sorted({r['behavior'] for r in tasks})}}
                for arm,values in rows.items() if arm!='base'}
        write_json(destination/'report.json',report)
        manifest['status']='complete'
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:write_json(destination/'manifest.json',manifest)
    print(f'Fixed mixture saved: {destination}',flush=True)
    return 0
