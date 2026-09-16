"""Train the paper's eight probes on Main17 using frozen Validation contracts.

Without --execute, validate inputs only. --list prints the selected tasks without
opening datasets or importing the training environment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'visual_analytics/pcp_analyze/web/scripts/train_val_isolated_probes.py'


def selected_tasks(config_path: Path, task_ids: list[str] | None = None):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    cohort = json.loads((ROOT / config['cohort_manifest']).read_text(encoding='utf-8'))
    if (config['methods'] != cohort['method_ids'] or config['seeds'] != cohort['seeds']
            or config['backbone'] != 'siglip' or config['validation'] != 'task-common-val-isolation-v1'
            or type(config['epochs']) is not int or config['epochs'] < 1):
        raise ValueError('Training config differs from the eight-probe frozen-Validation protocol')
    known = {task['task_id'] for task in cohort['tasks']}
    if task_ids and (len(set(task_ids)) != len(task_ids) or not set(task_ids) <= known):
        raise ValueError('Task IDs must be unique members of Main17')
    return config, [t for t in cohort['tasks'] if not task_ids or t['task_id'] in task_ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paper_training.json')
    parser.add_argument('--directory', type=Path, help='Directory containing frozen per-task Val contracts')
    parser.add_argument('--task-id', action='append', help='Repeat to select a Main17 subset; default: all 17')
    parser.add_argument('--list', action='store_true', help='List tasks, method IDs and seeds without loading assets')
    parser.add_argument('--execute', action='store_true', help='Train after contract checks; default: preflight only')
    args = parser.parse_args()
    config, tasks = selected_tasks(args.config, args.task_id)
    if args.list:
        print(json.dumps({'methods': config['methods'], 'seeds': config['seeds'],
                          'epochs': config['epochs'], 'tasks': [t['task_id'] for t in tasks]}, indent=2))
        return
    if args.directory is None or not args.directory.is_dir():
        parser.error('--directory must point to the downloaded frozen Validation contracts')
    for task in tasks:
        command = [sys.executable, '-B', str(TRAINER), '--directory', str(args.directory.resolve()),
                   '--task-id', task['task_id'], '--epochs', str(config['epochs'])]
        if args.execute:
            command.append('--execute')
        print(f"{task['task_id']}: {'train' if args.execute else 'preflight'}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
