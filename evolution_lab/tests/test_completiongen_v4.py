import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evolution_lab.case_eval import load_cases
from evolution_lab.completiongen import PHRASES as PHRASES_V1, SUBJECT_PAIRS as SUBJECTS_V1, VERB_PAIRS as VERBS_V1
from evolution_lab.completiongen_v2 import PHRASES as PHRASES_V2, SUBJECT_PAIRS as SUBJECTS_V2, VERB_PAIRS as VERBS_V2
from evolution_lab.completiongen_v3 import PHRASES as PHRASES_V3, SUBJECT_PAIRS as SUBJECTS_V3, VERB_PAIRS as VERBS_V3
from evolution_lab.completiongen_v4 import PHRASES, SUBJECT_PAIRS, VERB_PAIRS, generate, solve, validate_dataset, write_dataset


class CompletionGeneratorV4Tests(unittest.TestCase):
    def test_inventory_is_unique_and_disjoint_from_prior_versions(self):
        self.assertEqual(len(SUBJECT_PAIRS), len(set(SUBJECT_PAIRS)))
        self.assertEqual(len(VERB_PAIRS), len(set(VERB_PAIRS)))
        self.assertEqual(len({prefix for prefix, _, _ in PHRASES.values()}), len(PHRASES))
        self.assertEqual(len({(prefix.casefold(), answer.casefold())
                              for prefix, answer, _ in PHRASES.values()}), len(PHRASES))
        subject_roots = {root.casefold() for root, _ in SUBJECT_PAIRS}
        verb_roots = {root.casefold() for root, _ in VERB_PAIRS}
        phrase_tokens = {word.casefold() for prefix, answer, distractor in PHRASES.values()
                         for word in f"{prefix} {answer} {distractor}".split()}
        self.assertTrue(subject_roots.isdisjoint(phrase_tokens))
        for prior in (SUBJECTS_V1, SUBJECTS_V2, SUBJECTS_V3):
            prior_words = {word.casefold() for pair in prior for word in pair}
            self.assertTrue(subject_roots.isdisjoint(prior_words))
        for prior in (VERBS_V1, VERBS_V2, VERBS_V3):
            prior_words = {word.casefold() for pair in prior for word in pair}
            self.assertTrue(verb_roots.isdisjoint(prior_words))
        prior_phrases = []
        for prior in (PHRASES_V1, PHRASES_V2, PHRASES_V3):
            prior_phrases.extend((prefix.casefold(), answer.casefold())
                                 for prefix, answer, _ in prior.values())
        fresh_phrases = {(prefix.casefold(), answer.casefold())
                         for prefix, answer, _ in PHRASES.values()}
        self.assertTrue(fresh_phrases.isdisjoint(prior_phrases))

    def test_split_has_fresh_sentence_lexicons_and_trusted_keys(self):
        rows = generate(seed=84)
        family_splits = {}
        split_lexicons = {}
        for row in rows:
            self.assertEqual(row["answer"], solve(row))
            self.assertIn(row["answer"], row["options"])
            self.assertEqual(len(set(row["options"])), 2)
            self.assertEqual(row["prompt"].count("___"), 1)
            self.assertEqual(family_splits.setdefault(row["family"], row["split"]), row["split"])
            if row["task_type"] == "sentence_completion":
                keys = split_lexicons.setdefault(row["split"], {"subjects": set(), "verbs": set()})
                keys["subjects"].add(row["subject"].removesuffix("s"))
                keys["verbs"].add(row["base_verb"])
        for split in ("train", "dev", "confirmation"):
            for other in ({"train", "dev", "confirmation"} - {split}):
                self.assertTrue(split_lexicons[split]["subjects"].isdisjoint(split_lexicons[other]["subjects"]))
                self.assertTrue(split_lexicons[split]["verbs"].isdisjoint(split_lexicons[other]["verbs"]))
        counts = Counter((row["split"], row["task_type"]) for row in rows)
        self.assertEqual(counts[("train", "phrase_completion")], 33 * 8)
        self.assertEqual(counts[("dev", "phrase_completion")], 7 * 8)
        self.assertEqual(counts[("confirmation", "phrase_completion")], 8 * 8)
        self.assertEqual(counts[("confirmation", "sentence_completion")], 5 * 5 * 16)

    def test_dataset_round_trip_and_confirmation_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_dataset(root, seed=84)
            self.assertEqual(manifest["unique_cases"], len(generate(seed=84)))
            census = validate_dataset(root)
            self.assertEqual(census["unique_cases"], manifest["unique_cases"])
            source, confirmation = load_cases(root, "confirmation")
            self.assertEqual(source["manifest"]["generator"], "phrase-sentence-completion-v4-fresh-lexicon")
            self.assertEqual(len(confirmation), 464)
            self.assertEqual(Counter(row["task_type"] for row in confirmation),
                             {"phrase_completion": 64, "sentence_completion": 400})


if __name__ == "__main__":
    unittest.main()
