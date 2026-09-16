"""Evaluate Table 2: two embedding baselines, CLAY, eight probes and initial F0.

Uses the downloaded frozen Web evidence and CLAY score cache. All F1 cutoffs
are selected on original VQA Validation and saved before Test/Gallery evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'visual_analytics/pcp_analyze/web'
sys.path[:0] = [str(ROOT / 'probe_learning'), str(WEB / 'scripts')]

import numpy as np
from src.evaluation.retrieval_metrics import evaluation_metrics, fit_training_threshold
from report_initial_performance import METHODS, checked_current_scores
from run_clay_full_gallery import load_inputs, read_json, sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--web', type=Path, default=WEB, help='Web root containing the frozen asset bundle')
    parser.add_argument('--clay', type=Path, required=True, help='Frozen full-gallery CLAY score-cache directory')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    scope = read_json(ROOT / 'manifests/paper_main17.json')
    expected = {t['task_id']: t for t in scope['tasks']}
    web, clay = args.web.resolve(), args.clay.resolve()
    tasks, active, active_bytes, catalog_path = load_inputs(web, list(expected))
    if active['version'] != scope['initial_version']:
        raise ValueError('The paper evaluation requires the pinned initial publication')
    clay_manifest = read_json(clay / 'manifest.json')
    entries = {t['taskId']: t for t in clay_manifest['tasks']}
    if (not clay_manifest.get('complete') or clay_manifest['protocol'] != 'clay-full-gallery-current-f0-v1'
            or len(entries) != clay_manifest['taskCount'] or not set(expected) <= set(entries)
            or sha(clay / 'run-inputs.json') != clay_manifest['inputsSha256']):
        raise ValueError('CLAY cache is incomplete or has changed')
    if args.output.exists():
        raise FileExistsError('Use a new report directory')
    choices, prepared, audits = [], {}, []
    catalog_sha = sha(catalog_path)
    for task in tasks:
        tid = task['task_id']
        scores, audit = checked_current_scores(web, task, active, clay, entries[tid])
        if audit['f0ScoresSha256'] != expected[tid]['initial_scores_sha256']:
            raise ValueError(f'Published scores differ from the paper: {tid}')
        vr = np.asarray(task['val'], dtype=np.int64)
        labels = np.asarray(task['val_labels'], dtype=np.uint8)
        if set(labels.tolist()) != {0, 1} or len(vr) != len(labels):
            raise ValueError(f'Invalid fixed VQA Validation labels: {tid}')
        for method, vector in zip(METHODS, scores, strict=True):
            threshold = fit_training_threshold(labels, np.asarray(vector)[vr])['threshold']
            vm = evaluation_metrics(labels, np.asarray(vector)[vr], threshold)
            choices.append({'task_id': tid, 'method': method['methodId'], 'threshold': threshold,
                            'val_ap': vm['Ap'], 'val_f1': vm['F1'],
                            'score_sha256': hashlib.sha256(np.asarray(vector).tobytes()).hexdigest()})
        prepared[tid] = (task, scores)
        audits.append(audit)
        print(f'{tid}: 12 Validation cutoffs selected', flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    selection_path = args.output / 'val_selection.json'
    write_json(selection_path, {'cohort': 'main17', 'uses_test_labels': False, 'choices': choices,
                                'source_audit': audits, 'publication': active})
    selection_hash = sha(selection_path)
    lookup = {(r['task_id'], r['method']): r for r in choices}
    records = []
    for task, scores in prepared.values():
        truth = task['truth'][:, task['targets'].index('joint')]
        test = np.asarray(task['test'], dtype=bool)
        for method, vector in zip(METHODS, scores, strict=True):
            row = dict(lookup[task['task_id'], method['methodId']])
            for name, mask in [('test', test), ('gallery', np.ones(len(truth), dtype=bool))]:
                metrics = evaluation_metrics(truth[mask], np.asarray(vector)[mask], row['threshold'])
                if metrics['Ap'] is None:
                    raise ValueError('Main17 evaluation requires positives in each partition')
                row.update({f'{name}_ap': metrics['Ap'], f'{name}_f1': metrics['F1']})
            records.append(row)
    if ((web.parent / 'runtime/unified-initial/active.json').read_bytes() != active_bytes
            or sha(catalog_path) != catalog_sha or sha(selection_path) != selection_hash):
        raise ValueError('Frozen inputs/selection changed during evaluation')
    macro = []
    for method in METHODS:
        rows = [r for r in records if r['method'] == method['methodId']]
        if len(rows) != 17:
            raise ValueError('Every method must cover Main17 exactly once')
        macro.append({'method': method['methodId'], 'task_count': len(rows), **{
            metric: math.fsum(r[metric] for r in rows) / len(rows)
            for metric in ('test_ap', 'test_f1', 'gallery_ap', 'gallery_f1')}})
    write_json(args.output / 'results.json', {'task_count': 17, 'method_count': 12,
        'selection_sha256': selection_hash, 'macro': macro, 'records': records})
    print(json.dumps(macro, indent=2))


if __name__ == '__main__':
    main()
