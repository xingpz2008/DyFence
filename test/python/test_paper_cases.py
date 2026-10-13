import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_PATH = REPO_ROOT / "ae.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dyfence_paper_cases", AE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PaperCasesTest(unittest.TestCase):
    def test_manifest_has_every_reported_case(self):
        module = load_module()
        cases = module.load_paper_cases()

        self.assertEqual(
            {"I": 6, "II": 10, "III": 2, "IV": 10, "V": 3,
             "VI": 4, "VIII": 10, "IX": 6},
            {table: len(entries) for table, entries in cases["tables"].items()},
        )

    def test_table_iii_reuses_the_matching_table_ii_measurements(self):
        module = load_module()
        cases = module.load_paper_cases()["tables"]
        table_ii_ids = {entry["id"] for entry in cases["II"]}

        self.assertEqual(
            {"resnet18-cifar10", "vgg16-cifar10"},
            {entry["source_case"] for entry in cases["III"]},
        )
        self.assertTrue(all(entry["source_case"] in table_ii_ids for entry in cases["III"]))

    def test_accuracy_cases_cover_five_models_and_two_datasets(self):
        module = load_module()
        entries = module.load_paper_cases()["tables"]["IV"]

        self.assertEqual(
            {"resnet18", "resnet50", "resnet101", "vgg11", "vgg16"},
            {entry["model"] for entry in entries},
        )
        self.assertEqual({"cifar10", "imagenet"}, {entry["dataset"] for entry in entries})

    def test_sample_count_supports_bounded_and_full_dataset_runs(self):
        module = load_module()

        self.assertEqual(50, module.resolve_sample_count("50", "cifar10"))
        self.assertEqual(100, module.resolve_sample_count("100", "imagenet"))
        self.assertEqual(10000, module.resolve_sample_count("full", "cifar10"))
        self.assertEqual(50000, module.resolve_sample_count("full", "imagenet"))
        with self.assertRaises(ValueError):
            module.resolve_sample_count("49", "cifar10")

    def test_accuracy_matrix_expands_all_models_and_datasets(self):
        module = load_module()
        selected = module.select_accuracy_cases("all", "all")

        self.assertEqual(10, len(selected))
        self.assertEqual("resnet18-cifar10", selected[0]["id"])
        self.assertEqual("vgg16-imagenet", selected[-1]["id"])

    def test_integrated_cases_explicitly_enable_fusion_and_truncation(self):
        module = load_module()
        cases = module.load_paper_cases()["tables"]["VIII"]
        integrated = [case for case in cases if case["id"].startswith("integrated-")]

        self.assertEqual(3, len(integrated))
        self.assertTrue(all(case["arguments"][4:6] == ["1", "1"] for case in integrated))

    def test_every_measured_case_carries_its_paper_reference(self):
        module = load_module()
        tables = module.load_paper_cases()["tables"]
        for table in ("I", "II", "III", "V", "VI", "VIII", "IX"):
            for case in tables[table]:
                self.assertIn("reference", case, f"Table {table} case {case['id']}")

    def test_all_selector_preserves_shared_table_references(self):
        module = load_module()
        selected = module.select_paper_cases(module.load_paper_cases(), "all", "all")
        resnet = next(case for case in selected if case["kind"] == "model" and case["id"] == "resnet18-cifar10")

        self.assertEqual(["II", "III"], resnet["reported_in"])
        self.assertEqual({"II", "III"}, set(resnet["references"]))


if __name__ == "__main__":
    unittest.main()
