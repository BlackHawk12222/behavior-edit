"""Audit raw experiment records and verify an explicitly stated behavioral hypothesis.

A passing gate is evidence for one diagnostic, never automatic model promotion.
"""
import json
from pathlib import Path
from .metrics import grade
from .provenance import fingerprint, checkpoint_hashes
from .statistics import compare_roles


def read_run(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('status')!='complete' or manifest.get('smoke_only'):
        raise ValueError('Evidence requires complete non-smoke runs')
    if fingerprint(manifest['tasks'])!=manifest['evaluation_sha256']:
        raise ValueError('Task fingerprint mismatch')
    expected={r['id']:r for r in manifest['tasks']}
    conditions={name:[] for name in manifest['controls']}
    seen=set()
    for line in (directory/'observations.jsonl').read_text().splitlines():
        row=json.loads(line)
        condition,identifier=row['condition'],row['task_id']
        if condition not in conditions or identifier not in expected or (condition,identifier) in seen:
            raise ValueError('Unexpected or duplicate observation')
        seen.add((condition,identifier))
        if row['task']!=expected[identifier] or row['metrics']!=grade(row['task'],row['response']):
            raise ValueError('Task or objective score does not match raw response')
        conditions[condition].append(row)
    if len(seen)!=len(expected)*len(conditions):
        raise ValueError('Missing observations')
    return manifest,conditions


def verify_hypothesis(dev_dir,test_dir,min_gain=.25,max_regressions=0):
    if not 0 <= min_gain <= 1 or type(max_regressions) is not int or max_regressions < 0:
        raise ValueError('Invalid evidence gate')
    dev,dev_rows=read_run(dev_dir);test,test_rows=read_run(test_dir)
    if dev['split']!='dev' or test['split']!='test':
        raise ValueError('Require development and independent test splits')
    if test.get('source_manifest_sha256')!=fingerprint(dev):
        raise ValueError('Test does not refer to this frozen development experiment')
    if 'adapter_hashes' in test:
        actual={arm:checkpoint_hashes(Path(dev_dir)/f'adapter_{arm}') for arm in ('lora','random')}
        if actual != test['adapter_hashes']:
            raise ValueError('Checkpoint files changed since frozen test')
        if 'adapter_hashes' in dev and actual != dev['adapter_hashes']:
            raise ValueError('Checkpoint files changed since development evaluation')
    for condition in ('base','prompt','lora','random'):
        a={r.get('system_prompt') for r in dev_rows[condition]}
        b={r.get('system_prompt') for r in test_rows[condition]}
        if a != b or len(a)!=1:
            raise ValueError('System prompt changed between dev and test')
    for key in ('suite_sha256','model_id','resolved_revision','behavior','seed','settings'):
        if dev[key]!=test[key]:
            raise ValueError(f'Protocol mismatch: {key}')
    def assess(rows,seed):
        contrasts={name:compare_roles(rows[name],rows['lora'],seed)
                   for name in ('base','prompt','random')}
        gain=contrasts['base']['target']['pass_rate_delta']
        regressions=contrasts['base']['locality']['regressed']
        # Require strict superiority to each control on the targeted probes.
        exceeds=all(contrasts[name]['target']['pass_rate_delta']>0 for name in ('prompt','random'))
        return {'passes_gate':gain>=min_gain and regressions<=max_regressions and exceeds,
                'target_gain':gain,'locality_regressions':regressions,
                'beats_prompt_and_random':exceeds,'contrasts':contrasts}
    development=assess(dev_rows,dev['seed']);independent=assess(test_rows,test['seed'])
    return {'hypothesis':f"A {dev['behavior']} LoRA improves target pass rate by at least {min_gain:.0%}, "
            f"beats prompt and norm-matched random, and has at most {max_regressions} locality pass-to-fail changes.",
            'gate':{'min_target_gain':min_gain,'max_locality_regressions':max_regressions},
            'development':development,'independent_test':independent,
            'result':'replicated_on_this_diagnostic' if development['passes_gate'] and independent['passes_gate']
                     else 'not_verified',
            'behavior':dev['behavior'],'seed':dev['seed'],
            'dev_manifest_sha256':fingerprint(dev),'test_manifest_sha256':fingerprint(test),
            'caveat':'Retrospective diagnostic gate, not a preregistered significance test. '
                     'One seed and few authored families do not establish a predictive law, useful hypernetwork, '
                     'broad generalization, or permission to promote the adapter.'}
