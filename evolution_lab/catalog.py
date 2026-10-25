"""Export adapter/behavior observations for future, separately validated edit prediction."""
import json
from pathlib import Path
from .evidence import read_run, verify_hypothesis
from .statistics import compare_roles
from .provenance import fingerprint


def build_catalog(study_dir):
    journal=json.loads((Path(study_dir)/'study.json').read_text())
    if journal['status']!='complete' or any(t['status']!='complete' for t in journal['trials']):
        raise ValueError('Catalog requires a complete study; incomplete trials must not silently disappear')
    rows=[]
    for trial in journal['trials']:
        dev,development=read_run(trial['dev_dir']);test,independent=read_run(trial['test_dir'])
        evidence=verify_hypothesis(trial['dev_dir'],trial['test_dir'])
        delta=json.loads((Path(trial['dev_dir'])/'delta_lora.json').read_text())
        train=json.loads((Path(trial['dev_dir'])/'training.json').read_text())
        rows.append({'schema_version':1,'behavior':dev['behavior'],'seed':dev['seed'],
                     'model_id':dev['model_id'],'resolved_revision':dev['resolved_revision'],
                     'suite_sha256':dev['suite_sha256'],'training_sha256':dev['training_sha256'],
                     'dev_manifest_sha256':fingerprint(dev),'test_manifest_sha256':fingerprint(test),
                     'settings':dev['settings'],'device':dev['device'],'precision':dev['precision'],
                     'delta_global_frobenius_norm':delta['delta_global_frobenius_norm'],
                     'delta_layers':delta['layers'],'loss_first':train[0]['loss'],'loss_last':train[-1]['loss'],
                     'dev':{arm:compare_roles(development['base'],development[arm],dev['seed'])
                            for arm in ('prompt','lora','random')},
                     'test':{arm:compare_roles(independent['base'],independent[arm],dev['seed'])
                             for arm in ('prompt','lora','random')},
                     'hypothesis_result':evidence['result'],
                     'use_note':'Measured adapter features and diagnostic outcomes. '
                                'Test labels must never be used to fit or select an edit predictor. '
                                'This tiny correlated catalog is insufficient to train or validate a hypernetwork.'})
    return rows


def summarize_catalog(rows):
    result={}
    for behavior in sorted({r['behavior'] for r in rows}):
        selected=[r for r in rows if r['behavior']==behavior]
        result[behavior]={'seeds':[r['seed'] for r in selected],
                          'verified_diagnostics':sum(r['hypothesis_result']=='replicated_on_this_diagnostic' for r in selected),
                          'runs':[{ 'seed':r['seed'],
                                    'delta_norm':r['delta_global_frobenius_norm'],
                                    **{split:{arm:{role:{key:metric[key] for key in
                                           ('before_passed','after_passed','task_count','improved','regressed','response_changed')}
                                           for role,metric in r[split][arm].items()}
                                           for arm in ('prompt','lora','random')}
                                       for split in ('dev','test')}} for r in selected],
                          'warning':'Seeds reuse the same authored tasks; do not pool them as independent test items.'}
    return result
