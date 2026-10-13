import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class BundledEmpCMakeTest(unittest.TestCase):
    def test_gnu_linker_does_not_inherit_outer_make_jobserver(self):
        with tempfile.TemporaryDirectory(prefix="dyfence-linker-test-") as tmp:
            build_dir = Path(tmp) / "build"
            configured = subprocess.run(
                ["cmake", "-S", str(REPO_ROOT), "-B", str(build_dir),
                 "-G", "Unix Makefiles", "-DTEST_SELECT=DEV",
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                capture_output=True, text=True,
            )
            self.assertEqual(0, configured.returncode, configured.stdout + configured.stderr)
            commands = json.loads((build_dir / "compile_commands.json").read_text())
            compiler = shlex.split(commands[0]["command"])[0]
            version = subprocess.check_output([compiler, "--version"], text=True)
            if "clang" in version.lower():
                self.skipTest("GNU-specific jobserver isolation")
            link = shlex.split((build_dir / "CMakeFiles/tensor_storage_test.dir/link.txt").read_text())
            launcher = link[:link.index(compiler)]
            environment = dict(os.environ, MAKEFLAGS="-j4 --jobserver-auth=9998,9999")
            result = subprocess.run(
                [*launcher, sys.executable, "-c",
                 "import os; print(os.environ.get('MAKEFLAGS', '<unset>'))"],
                env=environment, capture_output=True, text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertEqual("<unset>", result.stdout.strip())

    def test_configure_uses_repository_emp_instead_of_system_emp(self):
        with tempfile.TemporaryDirectory(prefix="dyfence-cmake-test-") as tmp:
            build_dir = Path(tmp) / "build"
            env = os.environ.copy()
            env.pop("CMAKE_PREFIX_PATH", None)

            result = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(REPO_ROOT),
                    "-B",
                    str(build_dir),
                    "-DDYFENCE_USE_BUNDLED_EMP=ON",
                    "-DTEST_SELECT=FUNC",
                    "-DUSE_JEMALLOC=OFF",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )

            output = result.stdout + result.stderr
            self.assertEqual(0, result.returncode, output)
            self.assertIn("DyFence EMP source: bundled", output)

            compile_commands = (build_dir / "compile_commands.json").read_text(encoding="utf-8")
            self.assertIn(str(REPO_ROOT / "third_party" / "emp-tool"), compile_commands)
            self.assertIn(str(REPO_ROOT / "third_party" / "emp-ot"), compile_commands)
            self.assertIn(str(REPO_ROOT / "third_party" / "emp-zk"), compile_commands)
            self.assertNotIn("/usr/local/include/emp-", compile_commands)


if __name__ == "__main__":
    unittest.main()
