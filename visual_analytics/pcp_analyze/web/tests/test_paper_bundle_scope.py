"""Exercise the public bundle selector without galleries or model assets."""
import unittest

import export_task_bundles as bundles


class PaperBundleScopeTest(unittest.TestCase):
    def test_default_export_selects_all_paper_tasks_including_celeba_052(self):
        tasks = bundles.select_tasks(bundles.MAIN17_ORDERS)
        self.assertEqual([task.task_id for task in tasks],
                         [row['task_id'] for row in bundles.PAPER_SCOPE['tasks']])
        self.assertEqual(len(tasks), 17)
        self.assertEqual({task.order for task in tasks if task.dataset == 'celeba'}, {6, 52})
        for task in tasks:
            self.assertTrue(bundles.EXPECTED_TASKS[task.order][2])
            self.assertTrue(bundles.TASK_LABELS[task.order])

    def test_historical_and_duplicate_orders_are_rejected(self):
        for orders in ([5], [12], [59], [1, 1]):
            with self.subTest(orders=orders), self.assertRaises(ValueError):
                bundles.select_tasks(orders)


if __name__ == '__main__':
    unittest.main()
