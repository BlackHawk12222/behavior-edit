"""Descriptive uncertainty for small diagnostic studies; no promotion claims."""
import math
import random
from collections import defaultdict
from .metrics import compare_before_after


def wilson(passed, count):
    if count < 1 or not 0 <= passed <= count:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = passed / count
    denom = 1 + z*z/count
    center = (p + z*z/(2*count))/denom
    radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denom
    return [max(0.,center-radius), min(1.,center+radius)]


def paired_report(before, after, seed=42, resamples=2000):
    comparison = compare_before_after(before, after)
    if resamples < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    old = {r['task_id']: r for r in before}
    new = {r['task_id']: r for r in after}
    groups = defaultdict(list)
    for key in sorted(old):
        family = old[key]['task'].get('family', key)
        groups[family].append(int(new[key]['metrics']['passed']) - int(old[key]['metrics']['passed']))
    ci = None
    if len(groups) >= 2:
        rng = random.Random(seed)
        clusters = list(groups.values())
        draws = []
        for _ in range(resamples):
            values = [v for group in rng.choices(clusters, k=len(clusters)) for v in group]
            draws.append(sum(values)/len(values))
        draws.sort()
        ci = [draws[int(.025*resamples)], draws[min(resamples-1, int(.975*resamples))]]
    n = len(old)
    return {**comparison,
            'before_wilson95': wilson(comparison['before_passed'], n),
            'after_wilson95': wilson(comparison['after_passed'], n),
            'paired_family_bootstrap95': ci, 'family_count': len(groups),
            'uncertainty_note': 'Wilson intervals assume independent items and may be too narrow. '
            'Bootstrap resamples authored content families, not model seeds; fewer than two families gives no interval. '
            'Public synthetic tasks and small family counts do not establish general capability gains.'}


def compare_roles(before, after, seed=42):
    return {role: paired_report([r for r in before if r['task']['role'] == role],
                                [r for r in after if r['task']['role'] == role], seed)
            for role in ('target', 'locality')}
