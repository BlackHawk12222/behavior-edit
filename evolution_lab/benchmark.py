"""Versioned diagnostic benchmark. Public synthetic tasks, not a capability leaderboard."""
import json
from pathlib import Path
from .metrics import validate_tasks
from .provenance import assert_disjoint, fingerprint, normalized_prompt

SUITE_FILE = Path(__file__).with_name('benchmark_v1.json')
BEHAVIORS = ('json_format', 'exact_instruction', 'concise_content')


def load_suite(path=SUITE_FILE):
    suite = json.loads(Path(path).read_text())
    if suite.get('schema_version') != 1:
        raise ValueError('Unsupported benchmark schema')
    train, tasks = suite['training'], validate_tasks(suite['tasks'])
    if not isinstance(train, list) or not train:
        raise ValueError('Training examples must be a nonempty list')
    for row in train:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                            for k in ('id', 'prompt', 'answer')):
            raise ValueError('Invalid training example')
    ids, prompts, families = set(), set(), {}
    for row in train + tasks:
        if row.get('behavior') not in (*BEHAVIORS, 'preservation'):
            raise ValueError('Unknown benchmark behavior')
        if not isinstance(row.get('family'), str) or not row['family']:
            raise ValueError('Every row needs a content family')
        split = row.get('split')
        if split not in {'train', 'dev', 'test'} or (row in train and split != 'train'):
            raise ValueError('Invalid split')
        if row in tasks and split == 'train':
            raise ValueError('Evaluation rows cannot use train split')
        if row in train and (not isinstance(row.get('answer'), str) or not row['answer']):
            raise ValueError('Training rows need answers')
        key = normalized_prompt(row['prompt'])
        if key in prompts or row['id'] in ids:
            raise ValueError('Duplicate prompt or ID in benchmark')
        ids.add(row['id']); prompts.add(key)
        old = families.setdefault(row['family'], split)
        if old != split:
            raise ValueError('Content family crosses splits')
    assert_disjoint(train, tasks)
    for behavior in BEHAVIORS:
        if not any(r['behavior'] == behavior for r in train):
            raise ValueError(f'Missing training behavior: {behavior}')
        for split in ('dev', 'test'):
            for role in (behavior, 'preservation'):
                if not any(r['behavior'] == role and r['split'] == split for r in tasks):
                    raise ValueError(f'Missing {split} evaluation: {role}')
    return suite


def select_tasks(suite, behavior, split, limit_per_role=None):
    if behavior not in BEHAVIORS or split not in {'dev', 'test'}:
        raise ValueError('Unknown behavior or evaluation split')
    if limit_per_role is not None and limit_per_role < 1:
        raise ValueError('Role limit must be positive')
    rows = []
    for role in (behavior, 'preservation'):
        chosen = [dict(r, role='target' if role == behavior else 'locality')
                  for r in suite['tasks'] if r['behavior'] == role and r['split'] == split]
        rows.extend(chosen[:limit_per_role] if limit_per_role else chosen)
    return rows


def describe(suite):
    return {'suite_id': suite['suite_id'], 'sha256': fingerprint(suite),
            'training_examples': len(suite['training']),
            'evaluation_tasks': len(suite['tasks']),
            'counts': {b: {s: sum(r['behavior'] == b and r['split'] == s
                                 for r in suite['training'] + suite['tasks'])
                           for s in ('train', 'dev', 'test')}
                       for b in (*BEHAVIORS, 'preservation')},
            'warning': 'Public synthetic diagnostic; family separation is manually authored, not semantic leakage proof.'}
