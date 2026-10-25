import copy
import json
import tempfile
import unittest
from pathlib import Path
from evolution_lab.benchmark import load_suite, select_tasks
from evolution_lab.metrics import grade
from evolution_lab.statistics import paired_report, wilson


class BenchmarkTests(unittest.TestCase):
    def test_suite_covers_heldout_behavior_and_locality(self):
        suite = load_suite()
        rows = select_tasks(suite, 'json_format', 'test')
        self.assertEqual(sum(r['role']=='target' for r in rows),12)
        self.assertEqual(sum(r['role']=='locality' for r in rows),16)
        self.assertTrue(all(r['split']=='test' for r in rows))

    def test_smoke_limit_is_balanced(self):
        rows = select_tasks(load_suite(), 'exact_instruction', 'dev', 2)
        self.assertEqual([r['role'] for r in rows],['target','target','locality','locality'])
        with self.assertRaises(ValueError):
            select_tasks(load_suite(), 'json_format', 'dev', 0)

    def test_family_leakage_rejected(self):
        suite=copy.deepcopy(load_suite())
        suite['tasks'][0]['family']=suite['training'][0]['family']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'suite.json';path.write_text(json.dumps(suite))
            with self.assertRaisesRegex(ValueError,'family crosses'):
                load_suite(path)

    def test_brevity_requires_content(self):
        task={'kind':'bounded_contains','required':['water'],'max_words':4}
        self.assertTrue(grade(task,'Drink water regularly.')['passed'])
        for answer in ('','yes','watermelon','Drink some water each and every morning.'):
            self.assertFalse(grade(task,answer)['passed'])

    def test_wilson_small_sample_is_not_certainty(self):
        lo,hi=wilson(2,2)
        self.assertLess(lo,.5)
        self.assertAlmostEqual(hi,1.)

    def test_family_bootstrap_requires_multiple_families(self):
        def row(i,passed):
            return {'task_id':str(i),'task':{'family':'same'},'response':str(passed),
                    'metrics':{'passed':passed}}
        report=paired_report([row(1,False),row(2,True)],[row(1,True),row(2,False)])
        self.assertIsNone(report['paired_family_bootstrap95'])
        self.assertEqual(report['regressed'],1)
        self.assertEqual(report['improved'],1)

    def test_bootstrap_is_reproducible_and_paired(self):
        def row(i,passed):
            return {'task_id':str(i),'task':{'family':str(i)},'response':str(passed),
                    'metrics':{'passed':passed}}
        before=[row(i,False) for i in range(4)]
        after=[row(i,i<2) for i in range(4)]
        a=paired_report(before,after,seed=7)
        b=paired_report(before,list(reversed(after)),seed=7)
        self.assertEqual(a,b)
        self.assertEqual(a['pass_rate_delta'],.5)
