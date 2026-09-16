import copy
import unittest

from tests.test_addon_catalog_publish import CATALOG, core_catalog
from experimental_parent_overlay import TASK_ID, PROTOCOL


class ExperimentalCatalogTests(unittest.TestCase):
    def catalog(self):
        value = core_catalog()
        value["datasets"].append({"id":"hico", "label":"HICO", "tasks":[{
            "id":TASK_ID, "label":"Hugging a cat (robust test)",
            "catalogRole":"experimental", "dataRoot":f"/data/tasks/{TASK_ID}",
            "experimentalOverlay":{"protocol":PROTOCOL, "manifestSha256":"a"*64}}]})
        value["taskCount"] += 1
        return value

    def test_registered_experiment_preserved_without_changing_formal_entries(self):
        catalog = self.catalog()
        CATALOG.validate_catalog_shape(catalog)
        CATALOG.validate_formal_catalog_extension(core_catalog(), catalog)
        self.assertEqual(CATALOG.merge_published_addons(core_catalog(), catalog), catalog)
        self.assertEqual(catalog["datasets"][:1], core_catalog()["datasets"])

    def test_unknown_experimental_task_or_missing_identity_rejected(self):
        for field, value in (("id", "060_hico_unknown"), ("experimentalOverlay", {})):
            catalog = copy.deepcopy(self.catalog())
            catalog["datasets"][-1]["tasks"][0][field] = value
            with self.assertRaises(ValueError):
                CATALOG.validate_catalog_shape(catalog)
