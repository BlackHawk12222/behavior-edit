import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
from evolution_lab.study import execute_study
from evolution_lab.benchmark import load_suite, SUITE_FILE
from evolution_lab.provenance import fingerprint


class StudyTests(unittest.TestCase):
    def spec(self):
        return {'schema_version':1,'suite_sha256':fingerprint(load_suite()),
                'model':'fixture','revision':'a'*40,'steps':1,'rank':2,'learning_rate':.001,
                'max_length':128,'max_new_tokens':16,'precision':'float32',
                'trials':[{'behavior':'json_format','seed':42}]}

    def test_child_failure_is_journaled(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'spec.json';p.write_text(json.dumps(self.spec()))
            args=SimpleNamespace(spec=p,suite=SUITE_FILE,device='cpu',output_dir=root/'study')
            with patch('evolution_lab.study.subprocess.run',side_effect=subprocess.CalledProcessError(1,['fixture'])):
                with self.assertRaises(subprocess.CalledProcessError):execute_study(args)
            journal=json.loads((args.output_dir/'study.json').read_text())
            self.assertEqual(journal['status'],'failed')
            self.assertIn('CalledProcessError',journal['error'])

    def test_changed_resume_spec_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'spec.json';spec=self.spec();p.write_text(json.dumps(spec))
            output=root/'study';output.mkdir()
            (output/'study.json').write_text(json.dumps({'spec_sha256':'different'}))
            with self.assertRaisesRegex(ValueError,'different study specification'):
                execute_study(SimpleNamespace(spec=p,suite=SUITE_FILE,device='cpu',output_dir=output))

    def test_duplicate_trials_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'spec.json';spec=self.spec();spec['trials']*=2;p.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                execute_study(SimpleNamespace(spec=p,suite=SUITE_FILE,device='cpu',output_dir=root/'study'))

    def test_resume_marks_running_before_relaunch_and_adopts_finished_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'spec.json';spec=self.spec();p.write_text(json.dumps(spec))
            output=root/'study';output.mkdir()
            journal={'spec_sha256':fingerprint(spec),'spec':spec,'status':'failed',
                     'error':'previous interruption','trials':[]}
            journal_path=output/'study.json';journal_path.write_text(json.dumps(journal))
            dev=output/'00_json_format_42'/'dev'/'completed';dev.mkdir(parents=True)
            (dev/'manifest.json').write_text(json.dumps({'status':'complete'}))
            manifest={'split':'dev','behavior':'json_format','seed':42,
                      'settings':{k:spec[k] for k in ('steps','rank','learning_rate','max_length','max_new_tokens')},
                      'precision':'torch.float32','model_id':spec['model'],
                      'resolved_revision':spec['revision'],'suite_sha256':spec['suite_sha256']}
            def launch(command,**kwargs):
                running=json.loads(journal_path.read_text())
                self.assertEqual(running['status'],'running')
                self.assertNotIn('error',running)
                self.assertIn('test-checkpoint',command)
                self.assertNotIn('experiment',command)
                raise subprocess.CalledProcessError(1,command)
            with patch('evolution_lab.study.read_run',return_value=(manifest,{})),patch('evolution_lab.study.subprocess.run',side_effect=launch):
                with self.assertRaises(subprocess.CalledProcessError):
                    execute_study(SimpleNamespace(spec=p,suite=SUITE_FILE,device='cpu',output_dir=output))
