"""Protect the public paper scope and the frozen-evidence evaluation boundary."""
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'probe_learning'))
from src.methods.catalog import METHOD_IDS
from src.methods.gate_calibration import search_attribute_gate, joint_bce_from_gate_logits
from src.evaluation.components import component_scores, select_on_validation


def load_script(name):
    spec = importlib.util.spec_from_file_location('paper_' + name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING = load_script('train_probes')
ABLATION = load_script('evaluate_ablation')


class PaperEntryTests(unittest.TestCase):
    def test_default_training_is_exactly_main17_and_eight_probes(self):
        config, tasks = TRAINING.selected_tasks(ROOT / 'configs/paper_training.json')
        self.assertEqual(tuple(config['methods']), METHOD_IDS)
        self.assertEqual(config['seeds'], [0, 1, 2, 3, 4])
        self.assertEqual(len({t['task_id'] for t in tasks}), 17)
        self.assertEqual({d: sum(t['dataset'] == d for t in tasks) for d in ('cars', 'hico', 'celeba')},
                         {'cars': 7, 'hico': 8, 'celeba': 2})

    def test_unknown_and_duplicate_training_tasks_are_rejected(self):
        cfg = ROOT / 'configs/paper_training.json'
        for ids in [['059_hico_task_hico_hugging_cat_robust_test'],
                    ['001_cars_task_bmw_convertible'] * 2]:
            with self.assertRaises(ValueError):
                TRAINING.selected_tasks(cfg, ids)

    def test_invalid_training_settings_fail_before_loading_assets(self):
        config = json.loads((ROOT / 'configs/paper_training.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            config['epochs'] = 0
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                TRAINING.selected_tasks(path)

    def test_asset_hash_and_directory_boundary_are_enforced(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'evidence.txt'
            source.write_text('immutable')
            spec = {'path': source.name, 'sha256': ABLATION.sha(source)}
            self.assertEqual(ABLATION.checked_asset(root, spec), source)
            source.write_text('changed')
            with self.assertRaises(ValueError):
                ABLATION.checked_asset(root, spec)
            with self.assertRaises(ValueError):
                ABLATION.checked_asset(root, {'path': '../outside', 'sha256': '0' * 64})

    def test_ablation_without_both_uses_ungated_product(self):
        config = json.loads((ROOT / 'configs/component_ablation.json').read_text())
        variant = next(v for v in config['variants'] if v['id'] == 'neither')
        q = np.array([[.8, .25], [.2, .9]])
        out = component_scores(q, [.1, .99], [.5, .5], [.1, .1], variant, config)
        np.testing.assert_array_equal(out, np.array([.2, .18], dtype=np.float32))

    def test_val_selection_rejects_single_class(self):
        config = json.loads((ROOT / 'configs/component_ablation.json').read_text())
        with self.assertRaises(ValueError):
            select_on_validation(np.ones((2, 1)) * .5, [.5, .5], [.5], [.1],
                                 [1, 1], config['variants'][0], config)

    def test_reference_gate_ties_are_deterministic(self):
        result = search_attribute_gate(np.array([.5, .5]), np.array([0, 1]),
                                       temperatures=(.1, .3))
        self.assertEqual(result['theta'], .5)
        self.assertEqual(result['temperature'], .3)

    def test_joint_bce_remains_finite_for_extreme_logits(self):
        value = joint_bce_from_gate_logits(np.array([[-1000., 1000.], [-999., 900.]]),
                                          np.array([1, 0]))
        self.assertTrue(np.isfinite(value))


if __name__ == '__main__':
    unittest.main()
