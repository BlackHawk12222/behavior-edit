"""Fixed sequential experiment matrix with process isolation and resumable journaling."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .benchmark import load_suite
from .experiment import write_json
from .evidence import read_run, verify_hypothesis
from .provenance import fingerprint, environment


def execute_study(args):
    spec=json.loads(args.spec.read_text())
    if spec.get('schema_version')!=1 or not spec.get('trials'):
        raise ValueError('Study spec requires schema_version=1 and trials')
    suite=load_suite(args.suite)
    if spec['suite_sha256']!=fingerprint(suite):
        raise ValueError('Study suite fingerprint mismatch')
    keys=set()
    for trial in spec['trials']:
        key=(trial['behavior'],trial['seed'])
        if key in keys: raise ValueError('Duplicate behavior/seed trial')
        keys.add(key)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'study.json'
    if path.exists():
        journal=json.loads(path.read_text())
        if journal['spec_sha256']!=fingerprint(spec):
            raise ValueError('Cannot resume with a different study specification')
    else:
        journal={'schema_version':1,'spec':spec,'spec_sha256':fingerprint(spec),
                 'environment':environment(),'status':'running','trials':[]}
        write_json(path,journal)
    try:
        for i,trial in enumerate(spec['trials']):
            if i<len(journal['trials']) and journal['trials'][i]['status']=='complete':
                item=journal['trials'][i]
                # Reaudit existing evidence on resume rather than silently trusting status.
                verify_hypothesis(item['dev_dir'],item['test_dir'])
                continue
            trial_dir=args.output_dir/f'{i:02}_{trial["behavior"]}_{trial["seed"]}'
            trial_dir.mkdir(exist_ok=True)
            item={'behavior':trial['behavior'],'seed':trial['seed'],'status':'running'}
            if i<len(journal['trials']): item=journal['trials'][i];item['status']='running'
            else:journal['trials'].append(item)
            write_json(path,journal)
            dev_dir=item.get('dev_dir') or trial.get('reuse_dev')
            if dev_dir:
                manifest,_=read_run(dev_dir)
                if (manifest['split']!='dev' or manifest['behavior']!=trial['behavior']
                    or manifest['seed']!=trial['seed'] or manifest['settings']['steps']!=spec['steps']
                    or manifest['settings']['rank']!=spec['rank']
                    or manifest['settings']['learning_rate']!=spec['learning_rate']
                    or manifest['settings']['max_length']!=spec['max_length']
                    or manifest['settings']['max_new_tokens']!=spec['max_new_tokens']
                    or manifest['precision']!='torch.'+spec['precision']
                    or manifest['model_id']!=spec['model']
                    or manifest['resolved_revision']!=spec['revision']
                    or manifest['suite_sha256']!=spec['suite_sha256']):
                    raise ValueError('Reused development run does not match the study specification')
            else:
                command=[sys.executable,'-u','-m','evolution_lab','experiment','--suite',str(args.suite),
                         '--behavior',trial['behavior'],'--seed',str(trial['seed']),
                         '--model',spec['model'],'--revision',spec['revision'],'--device',args.device,
                         '--precision',spec['precision'],'--output-dir',str(trial_dir/'dev')]
                for name in ('steps','rank','learning_rate','max_length','max_new_tokens'):
                    command.extend(['--'+name.replace('_','-'),str(spec[name])])
                with (trial_dir/'dev.log').open('a') as stream:
                    subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
                candidates=[p.parent for p in (trial_dir/'dev').glob('*/manifest.json')
                            if json.loads(p.read_text())['status']=='complete']
                if len(candidates)!=1:raise ValueError('Expected exactly one completed development experiment')
                dev_dir=str(candidates[0])
            item['dev_dir']=str(dev_dir);write_json(path,journal)
            test_dir=item.get('test_dir')
            if not test_dir:
                with (trial_dir/'test.log').open('a') as stream:
                    subprocess.run([sys.executable,'-u','-m','evolution_lab','test-checkpoint',str(dev_dir),
                                    '--suite',str(args.suite),'--device',args.device,
                                    '--output-dir',str(trial_dir/'test')],
                                    check=True,stdout=stream,stderr=subprocess.STDOUT)
                candidates=[p.parent for p in (trial_dir/'test').glob('*/manifest.json')
                            if json.loads(p.read_text())['status']=='complete']
                if len(candidates)!=1:raise ValueError('Expected exactly one completed frozen test')
                test_dir=str(candidates[0])
                item['test_dir']=test_dir;write_json(path,journal)
            evidence=verify_hypothesis(dev_dir,test_dir)
            write_json(trial_dir/'hypothesis.json',evidence)
            item.update(status='complete',result=evidence['result'])
            write_json(path,journal)
            print(f'Study trial {i+1}/{len(spec["trials"])} complete: {trial["behavior"]} seed {trial["seed"]}: {item["result"]}',flush=True)
        journal['status']='complete'
    except BaseException as error:
        journal.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(path,journal)
    return 0
