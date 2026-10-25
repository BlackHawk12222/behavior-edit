import unittest
from evolution_lab.metrics import grade
from evolution_lab.provenance import assert_disjoint, fingerprint


class IntegrityTests(unittest.TestCase):
    def test_json_rejects_boolean_number_confusion(self):
        self.assertFalse(grade({'kind': 'json', 'target': {'ok': True}}, '{"ok":1}')['passed'])

    def test_json_rejects_duplicate_keys_and_nonfinite_numbers(self):
        task = {'kind': 'json', 'target': {'n': 2}}
        for text in ('{"n":1,"n":2}', '{"n":NaN}', '{"n":Infinity}'):
            self.assertFalse(grade(task, text)['passed'])

    def test_normalized_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assert_disjoint([{'id':'train','prompt':'  Hello\nWORLD '}],
                            [{'id':'test','prompt':'hello world'}])

    def test_shared_ids_rejected_even_with_changed_prompt(self):
        with self.assertRaises(ValueError):
            assert_disjoint([{'id':'a','prompt':'x'}], [{'id':'a','prompt':'y'}])

    def test_fingerprint_is_order_independent_but_content_sensitive(self):
        self.assertEqual(fingerprint({'x':1,'y':2}), fingerprint({'y':2,'x':1}))
        self.assertNotEqual(fingerprint([1,2]), fingerprint([2,1]))
