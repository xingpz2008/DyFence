import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
if str(FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIR))

from backend_cpp_templates import TEMPLATE_CPP
from compiler import generate_cpp_code


class CompilerGenerationTest(unittest.TestCase):
    def test_weight_layer_without_bias_still_declares_empty_poly_bias(self):
        graph = [
            {"name": "x", "type": "input", "inputs": []},
            {
                "name": "conv",
                "type": "conv2d",
                "inputs": ["x"],
                "weight_shape": [2, 1, 1, 1],
                "stride": [1, 1],
                "padding": [0, 0],
            },
            {"name": "output", "type": "output", "inputs": ["conv"]},
        ]

        sections = generate_cpp_code(graph, "NoBiasNet")

        self.assertIn("PolyTensor conv_b;", sections["poly_fields"])
        self.assertIn("pw.conv_b = PolyTensor();", sections["inject_statements"])

    def test_generated_forward_explicitly_moves_owned_output(self):
        rendered = TEMPLATE_CPP.format(
            model_name="TinyNet",
            model_name_lower="tinynet",
            host_fields="",
            poly_fields="",
            load_statements="",
            inject_statements="",
            forward_statements="    PolyTensor tensor_out;",
            input_name="tensor_x",
            output_name="tensor_out",
            get_weights_cpp="",
        )

        self.assertIn("return std::move(tensor_out);", rendered)


if __name__ == "__main__":
    unittest.main()
