import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"

EXPECTED_COMMITS = {
    "emp-tool": "802b5d4fb7cc7fcaadd411cd6aa5e72ed4dd57fd",
    "emp-ot": "a603ca0c77fcda37b9d088bd692111f67a4bef96",
    "emp-zk": "73ab193a923d4c122be5a4f6bc1fe4f617966b02",
}


class BundledEmpSnapshotTest(unittest.TestCase):
    def test_contains_buildable_source_and_license_for_each_dependency(self):
        for name in EXPECTED_COMMITS:
            with self.subTest(name=name):
                root = THIRD_PARTY / name
                self.assertTrue(root.is_dir(), f"missing bundled dependency: {root}")
                self.assertTrue((root / "CMakeLists.txt").is_file())
                self.assertTrue((root / "LICENSE").is_file())
                self.assertTrue((root / "README.md").is_file())
                self.assertTrue((root / "cmake").is_dir())
                self.assertTrue((root / name).is_dir())

    def test_excludes_repository_and_generated_build_state(self):
        forbidden_names = {
            ".git",
            ".DS_Store",
            "CMakeCache.txt",
            "CMakeFiles",
            "Makefile",
            "bin",
            "build",
            "test",
        }
        found = []
        if THIRD_PARTY.exists():
            for path in THIRD_PARTY.rglob("*"):
                if path.name in forbidden_names:
                    found.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual([], found, f"generated or unrelated files bundled: {found}")

    def test_contains_dyfence_emp_zk_math_extension(self):
        extension = THIRD_PARTY / "emp-zk" / "emp-zk" / "emp-zk-math" / "ZKmath-global.h"
        self.assertTrue(extension.is_file(), f"missing DyFence EMP extension: {extension}")

    def test_emp_tool_cmake_sources_are_in_the_release(self):
        root = THIRD_PARTY / "emp-tool"
        cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
        sources = {
            line.strip()
            for line in cmake.splitlines()
            if line.strip().startswith("emp-tool/") and line.strip().endswith(".cpp")
        }
        self.assertTrue(sources, "EMP CMake source list is empty")
        missing = [
            f"third_party/emp-tool/{source}"
            for source in sorted(sources)
            if not (root / source).is_file()
            and not source.endswith(("aes_128.txt.cpp", "Keccak_f.txt.cpp"))
        ]
        self.assertEqual([], missing, f"CMake sources absent from the release: {missing}")

    def test_versions_records_exact_provenance_and_tree_checksums(self):
        versions_path = THIRD_PARTY / "VERSIONS.md"
        self.assertTrue(versions_path.is_file(), f"missing provenance file: {versions_path}")
        versions = versions_path.read_text(encoding="utf-8")

        for name, commit in EXPECTED_COMMITS.items():
            with self.subTest(name=name):
                self.assertIn(name, versions)
                self.assertIn(commit, versions)
                self.assertRegex(
                    versions,
                    rf"(?s){re.escape(name)}.*?SHA-256.*?[0-9a-f]{{64}}",
                )

        self.assertIn("DyFence-specific emp-zk-math extensions", versions)


if __name__ == "__main__":
    unittest.main()
