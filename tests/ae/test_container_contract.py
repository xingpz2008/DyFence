import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ContainerContractTest(unittest.TestCase):
    def test_docker_image_is_reproducible_and_uses_bundled_emp(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

        self.assertRegex(
            dockerfile,
            r"^FROM --platform=linux/amd64 ubuntu:22\.04@sha256:[0-9a-f]{64}",
        )
        self.assertIn("RUN python3 ae.py build", dockerfile)
        ae_driver = (REPO_ROOT / "ae.py").read_text(encoding="utf-8")
        self.assertIn("-DDYFENCE_USE_BUNDLED_EMP=ON", ae_driver)
        self.assertNotRegex(dockerfile, re.compile(r"\bgit\s+clone\b", re.IGNORECASE))
        self.assertIn('ENTRYPOINT ["python3", "ae.py"]', dockerfile)

    def test_python_packages_are_exactly_pinned(self):
        requirements = (REPO_ROOT / "frontend" / "requirements.txt").read_text(
            encoding="utf-8"
        )
        active_lines = [
            line.strip()
            for line in requirements.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertGreater(len(active_lines), 0)
        for line in active_lines:
            if line.startswith("--"):
                continue
            self.assertRegex(line, r"^[A-Za-z0-9_.-]+==[^=\s]+$")

    def test_docker_context_excludes_local_outputs(self):
        ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
        entries = {line.strip() for line in ignored.splitlines() if line.strip()}
        for entry in (".git", "build*", "generated_model", "results", "frontend/mvzk_venv"):
            self.assertIn(entry, entries)

    def test_container_supports_claim_network_profiles(self):
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        throttle = (REPO_ROOT / "test" / "throttle.sh").read_text(encoding="utf-8")

        self.assertIn("iproute2", dockerfile)
        self.assertIn("TC=(tc)", throttle)
        self.assertIn('"${TC[@]}" qdisc', throttle)


if __name__ == "__main__":
    unittest.main()
