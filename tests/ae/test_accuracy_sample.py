import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AE_PATH = REPO_ROOT / "ae.py"
WEIGHTS_DIR = REPO_ROOT / "model_weights" / "cifar10"


def load_ae_module():
    spec = importlib.util.spec_from_file_location("dyfence_accuracy_ae", AE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AccuracySampleTest(unittest.TestCase):
    def test_accuracy_csv_records_both_accuracies_and_difference(self):
        module = load_ae_module()
        experiment = {
            "name": "resnet18-cifar10", "table": "IV", "status": "passed",
            "plaintext": {"percentage": 86.0}, "dyfence": {"percentage": 84.0},
            "difference_percentage_points": -2.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            module.write_results_csv(path, [experiment])
            text = path.read_text(encoding="utf-8")

        self.assertIn("plaintext_accuracy,dyfence_accuracy,difference_percentage_points", text)
        self.assertIn("86.0,84.0,-2.0", text)
    def test_plaintext_top1_accuracy_uses_dataset_labels(self):
        ground_truth_path = REPO_ROOT / "frontend" / "accuracy_metrics.py"
        spec = importlib.util.spec_from_file_location("dyfence_ground_truth", ground_truth_path)
        ground_truth = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(ground_truth)

        result = ground_truth.compute_top1_accuracy([1, 0, 0], [1, 0, 1])

        self.assertEqual(2, result["correct"])
        self.assertEqual(3, result["total"])
        self.assertAlmostEqual(66.6666667, result["percentage"])
    def test_accuracy_uses_only_real_paper_models(self):
        module = load_ae_module()
        entries = module.load_paper_cases()["tables"]["IV"]

        self.assertEqual(10, len(entries))
        self.assertFalse((REPO_ROOT / "samples" / "accuracy" / "tiny_model.py").exists())
        self.assertFalse((REPO_ROOT / "samples" / "accuracy" / "cases.json").exists())

    def test_cifar_model_constructors_never_download_default_weights(self):
        for path in (REPO_ROOT / "local_models").glob("*_cifar10.py"):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("weights='DEFAULT'", source, path.name)

    def test_manifest_checkpoint_hashes_match_packaged_files(self):
        module = load_ae_module()
        checkpoints = module.load_paper_cases()["cifar10_checkpoints"]
        for record in checkpoints.values():
            path = WEIGHTS_DIR / record["file"]
            self.assertTrue(path.is_file(), path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], digest, path.name)

    def test_accuracy_report_parser_accepts_real_dataset_accuracy(self):
        module = load_ae_module()
        report = "\x1b[1;32mZK Model Accuracy    : 84.00% (42/50)\x1b[0m"

        self.assertEqual({"percentage": 84.0, "correct": 42, "total": 50}, module.parse_accuracy_report(report, expected_total=50))
        with self.assertRaises(ValueError):
            module.parse_accuracy_report("ZK Model Accuracy : 84.00% (42/50)", expected_total=100)

    def test_plaintext_report_parser_reads_top1_result(self):
        module = load_ae_module()
        report = "Plaintext Top-1 Accuracy: 86.00% (43/50)"
        self.assertEqual(
            {"percentage": 86.0, "correct": 43, "total": 50},
            module.parse_plaintext_accuracy(report, expected_total=50),
        )


if __name__ == "__main__":
    unittest.main()
