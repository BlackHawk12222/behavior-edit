import copy
import json
import tempfile
import unittest
from pathlib import Path
from evolution_lab.evidence import read_run, verify_hypothesis
from evolution_lab.metrics import grade
from evolution_lab.provenance import fingerprint


def write_run(root,split,source=None):
    tasks=[{'id':split+'_'+role,'prompt':role,'kind':'exact','target':'ok',
            'role':role,'family':split+role} for role in ('target','locality')]
    manifest={'status':'complete','smoke_only':False,'split':split,'tasks':tasks,
              'suite_sha256':'suite','model_id':'model','resolved_revision':'a'*40,
              'behavior':'json_format','seed':42,'settings':{'steps':16},
              'controls':['base','prompt','random','lora'],
              'evaluation_sha256':fingerprint(tasks)}
    if source: manifest['source_manifest_sha256']=fingerprint(source)
    root.mkdir()
    (root/'manifest.json').write_text(json.dumps(manifest))
    rows=[]
    for condition in manifest['controls']:
        for task in tasks:
            response='ok' if condition=='lora' or task['role']=='locality' else 'bad'
            rows.append({'condition':condition,'task_id':task['id'],'task':task,
                         'response':response,'metrics':grade(task,response)})
    (root/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return manifest,rows


class EvidenceTests(unittest.TestCase):
    def test_independent_evidence_is_linked_to_frozen_dev(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            dev,_=write_run(root/'dev','dev')
            write_run(root/'test','test',dev)
            report=verify_hypothesis(root/'dev',root/'test')
            self.assertEqual(report['result'],'replicated_on_this_diagnostic')
            self.assertEqual(report['independent_test']['locality_regressions'],0)

    def test_forged_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run';_,rows=write_run(path,'dev')
            rows[0]['metrics']['passed']=True
            (path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'objective score'):
                read_run(path)

    def test_missing_and_duplicate_observations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run';_,rows=write_run(path,'dev')
            for bad,pattern in ((rows[:-1],'Missing'),(rows+[rows[0]],'duplicate')):
                (path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaisesRegex(ValueError,pattern):read_run(path)

    def test_wrong_source_run_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dev,_=write_run(root/'dev','dev')
            other=copy.deepcopy(dev);other['seed']=43
            write_run(root/'test','test',other)
            with self.assertRaisesRegex(ValueError,'frozen development'):
                verify_hypothesis(root/'dev',root/'test')

    def test_changed_system_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dev,_=write_run(root/'dev','dev')
            _,rows=write_run(root/'test','test',dev)
            for row in rows:row['system_prompt']='Changed prompt'
            (root/'test'/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'System prompt changed'):
                verify_hypothesis(root/'dev',root/'test')

    def test_changed_checkpoint_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dev,_=write_run(root/'dev','dev')
            test,_=write_run(root/'test','test',dev)
            test['adapter_hashes']={'lora':{'model.safetensors':'expected'},'random':{}}
            (root/'test'/'manifest.json').write_text(json.dumps(test))
            with self.assertRaisesRegex(ValueError,'Checkpoint files changed'):
                verify_hypothesis(root/'dev',root/'test')
