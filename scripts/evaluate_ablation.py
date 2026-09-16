"""Reproduce Main17 component ablations from frozen, pre-feedback evidence.

The input index contains paths and SHA-256 values, not feature/model payloads.
All gate choices and F1 cutoffs are saved before Test/Gallery GT is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'probe_learning'))

import numpy as np
from src.evaluation.components import component_scores, select_on_validation, macro_metrics
from src.evaluation.retrieval_metrics import evaluation_metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_asset(root, spec):
    path = (root / spec['path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != spec['sha256']:
        raise ValueError(f"Asset path/checksum mismatch: {spec['path']}")
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def validate_inputs(data, contract, task):
    copy = dict(contract)
    fingerprint = copy.pop('fingerprint')
    if (digest(copy) != fingerprint or contract['taskId'] != task['task_id']
            or contract['protocol'] != 'task-common-val-isolation-v1'
            or [a['id'] for a in contract['attributes']] != task['attribute_ids']):
        raise ValueError('Frozen task/attribute contract mismatch')
    image_ids = data['image_ids'].tolist()
    if len(set(image_ids)) != len(image_ids) or digest(image_ids) != contract['imageIdsSha256']:
        raise ValueError('Image order differs from the frozen contract')
    z, e = data['probeFeatures'], data['embeddingFeatures']
    n, a = len(image_ids), len(task['attribute_ids'])
    if z.shape != (n, a, 8) or e.shape != (n, 2):
        raise ValueError('Expected aligned [N,A,8] probe scores and [N,2] holistic scores')
    for values in (z, e):
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Normalized evidence must be finite and in [0,1]')
    theta, temperature = data['theta'], data['temperature']
    if (theta.shape != (a,) or temperature.shape != (a,) or not np.isfinite(theta).all()
            or not np.isfinite(temperature).all() or np.any(temperature <= 0)):
        raise ValueError('Invalid reference gates')
    vr = np.asarray(contract['valRows'], dtype=np.int64)
    fit = contract['targets']['joint']
    test = data['test_mask']
    if (test.shape != (n,) or not np.isin(test, [0, 1]).all()
            or set(np.flatnonzero(test)) != set(contract['testRows'])
            or not np.array_equal(data['val_rows'], vr)
            or not np.array_equal(data['train_rows'], fit['fitRows'])
            or not np.array_equal(data['train_labels'], fit['fitLabels'])
            or not np.array_equal(data['developmentIndices'], contract['normalizationRows'])):
        raise ValueError('Snapshot partitions differ from the frozen contract')
    val = set(vr.tolist())
    blocked = set(contract['testRows']) | set(contract['queryRows']) | val
    if (len(val) != len(vr) or val & (set(contract['testRows']) | set(contract['queryRows']))
            or blocked & set(fit['fitRows']) or blocked & set(contract['normalizationRows'])
            or blocked & set(contract['trainPoolRows'])):
        raise ValueError('Fit, normalization or unlabeled pool overlaps a protected partition')
    if len(vr) != len(fit['valLabels']) or np.any(vr < 0) or np.any(vr >= n):
        raise ValueError('Invalid Validation rows/labels')
    return z.astype(np.float64).mean(2), e.astype(np.float64).mean(1), theta, temperature, vr


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True, help='Root of the separate evaluation asset package')
    parser.add_argument('--index', type=Path, default=ROOT / 'manifests/component_inputs.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/component_ablation.json')
    parser.add_argument('--output', type=Path, required=True, help='New report directory')
    args = parser.parse_args()
    config = read(args.config)
    scope = read(ROOT / config['cohort_manifest'])
    index = read(args.index)
    tasks = {t['task_id']: t for t in scope['tasks']}
    inputs = {t['task_id']: t for t in index['tasks']}
    if set(inputs) != set(tasks) or len(inputs) != len(index['tasks']):
        raise ValueError('Input index must contain exactly the 17 paper tasks')
    if (config['embedding_strength'] != .25 or config['temperature_cap'] != .30
            or len(config['variants']) != 4
            or {(v['id'], v['softgate'], v['embedding']) for v in config['variants']} != {
                ('full', True, True), ('no_softgate', False, True),
                ('no_embedding', True, False), ('neither', False, False)}):
        raise ValueError('Configuration must describe the four paper component ablations')
    if args.output.exists():
        raise FileExistsError('Use a new report directory to preserve previous results')
    prepared, selections = {}, []
    for task_id, task in tasks.items():
        entry = inputs[task_id]
        snapshot = checked_asset(args.assets, entry['snapshot'])
        contract = read(checked_asset(args.assets, entry['contract']))
        if contract['methods'] != scope['method_ids'] or contract['seeds'] != scope['seeds']:
            raise ValueError('Probe/seed order differs from the paper manifest')
        with np.load(snapshot, allow_pickle=False) as data:
            q, h, theta, temperature, vr = validate_inputs(data, contract, task)
        for variant in config['variants']:
            choice, candidates = select_on_validation(q[vr], h[vr], theta, temperature,
                np.asarray(contract['targets']['joint']['valLabels']), variant, config)
            selections.append({'task_id': task_id, 'variant': variant['id'],
                               **choice, 'candidates': candidates})
        prepared[task_id] = (snapshot, q, h, theta, temperature)
        print(f'{task_id}: Validation selection complete', flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    selection_path = args.output / 'val_selection.json'
    write_new(selection_path, {'config': config, 'input_index_sha256': sha(args.index),
                              'choices': selections, 'uses_test_labels': False})
    selection_hash = sha(selection_path)
    records = []
    variants = {v['id']: v for v in config['variants']}
    for choice in selections:
        task_id = choice['task_id']
        snapshot, q, h, theta, temperature = prepared[task_id]
        if sha(snapshot) != inputs[task_id]['snapshot']['sha256']:
            raise ValueError('Snapshot changed after Validation selection')
        scores = component_scores(q, h, theta, temperature, variants[choice['variant']], config,
                                  choice['theta_shift'], choice['temperature_scale'])
        with np.load(snapshot, allow_pickle=False) as data:
            truth, test = data['joint_gt'], data['test_mask'].astype(bool)
        row = {k: v for k, v in choice.items() if k != 'candidates'}
        row['score_sha256'] = hashlib.sha256(scores.tobytes()).hexdigest()
        for name, mask in [('test', test), ('gallery', np.ones(len(truth), dtype=bool))]:
            metrics = evaluation_metrics(truth[mask], scores[mask], choice['threshold'])
            if metrics['Ap'] is None:
                raise ValueError(f'No positive evaluation rows: {task_id}/{name}')
            row.update({f'{name}_ap': metrics['Ap'], f'{name}_f1': metrics['F1']})
        records.append(row)
    if sha(selection_path) != selection_hash:
        raise ValueError('Frozen Validation choices changed during evaluation')
    result = {'task_count': len(tasks), 'selection_sha256': selection_hash,
              'macro': macro_metrics(records, config['variants']), 'records': records}
    write_new(args.output / 'results.json', result)
    print(json.dumps(result['macro'], indent=2))


if __name__ == '__main__':
    main()
