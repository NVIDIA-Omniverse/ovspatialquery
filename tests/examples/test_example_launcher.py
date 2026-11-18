# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "tools" / "examples" / "main.py"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("example_launcher", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load example launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The stable train's default paths; OVSPATIALQUERY_TRAIN=next selects separate ones.
@mock.patch.dict(os.environ, {"OVSPATIALQUERY_TRAIN": "stable"})
class ExampleLauncherTests(unittest.TestCase):
    def test_lesson_11_numbering_frames_wiring_and_language_restriction(self) -> None:
        launcher = _load_launcher()
        request = launcher.parse_arguments(["11"])
        self.assertFalse(request.headless)
        command, environment = launcher.build_command(
            request,
            project_root=PROJECT_ROOT,
            python_executable=Path(sys.executable),
            evidence=PROJECT_ROOT / "_build" / "evidence" / "lesson-11-test",
            platform_name="win32",
        )
        joined = "\n".join(command)
        self.assertIn("11_capability_showcase.py", joined)
        self.assertIn("11-capability-showcase", joined)
        self.assertIn("--frames\n3", joined)
        self.assertNotIn("--headless", command)
        self.assertNotIn("OVSPATIALQUERY_LIBRARY", environment)
        headless = launcher.parse_arguments(["11", "--headless", "--frames", "4"])
        headless_command, _environment = launcher.build_command(
            headless,
            project_root=PROJECT_ROOT,
            python_executable=Path(sys.executable),
            evidence=PROJECT_ROOT / "_build" / "evidence" / "lesson-11-test",
            platform_name="win32",
        )
        self.assertIn("--headless", headless_command)
        self.assertEqual(headless_command[headless_command.index("--frames") + 1], "4")
        with self.assertRaises(SystemExit):
            launcher.parse_arguments(["11", "--language", "native"])

    def test_short_command_defaults_to_python_and_exact_source_artifacts(self) -> None:
        launcher = _load_launcher()
        request = launcher.parse_arguments(["01", "--headless"])
        evidence = PROJECT_ROOT / "_build" / "evidence" / "launcher-test"
        command, environment = launcher.build_command(
            request,
            project_root=PROJECT_ROOT,
            python_executable=(
                PROJECT_ROOT / "_build" / "venv" / "ovspatialqueryapp" / "Scripts" / "python.exe"
            ),
            evidence=evidence,
            platform_name="win32",
        )
        joined = "\n".join(command)
        self.assertIn("--mode\npython", joined)
        self.assertIn("--include\n01_static_candidate_raycast", joined)
        self.assertIn("--expect-python-source", joined)
        self.assertIn(str(PROJECT_ROOT / "python" / "src"), joined)
        self.assertIn(
            str(
                PROJECT_ROOT
                / "_build"
                / "bootstrap"
                / "native-stable"
                / "install"
                / "bin"
                / "ovspatialquery.dll"
            ),
            joined,
        )
        self.assertNotIn("PYTHONPATH", environment)
        self.assertNotIn("OVSPATIALQUERY_LIBRARY", environment)

    def test_language_and_interactive_routing_remain_explicit(self) -> None:
        launcher = _load_launcher()
        request = launcher.parse_arguments(["10", "--language", "both", "--frames", "5"])
        command, _environment = launcher.build_command(
            request,
            project_root=PROJECT_ROOT,
            python_executable=Path(sys.executable),
            evidence=PROJECT_ROOT / "_build" / "evidence" / "launcher-test",
            platform_name="win32",
        )
        self.assertIn("both", command)
        self.assertIn("--interactive", command)
        self.assertIn("10_pair_reporter_deltas", command)
        self.assertIn("5", command)

    def test_launcher_preserves_source_venv_python_symlink_identity(self) -> None:
        launcher = _load_launcher()
        request = launcher.parse_arguments(["01", "--headless"])
        source_python = PROJECT_ROOT / "_build" / "venv" / "ovspatialqueryapp" / "bin" / "python"
        command, _environment = launcher.build_command(
            request,
            project_root=PROJECT_ROOT,
            python_executable=source_python,
            evidence=PROJECT_ROOT / "_build" / "evidence" / "launcher-test",
            platform_name="linux",
        )
        self.assertEqual(command[0], str(source_python))
        self.assertEqual(command[command.index("--python") + 1], str(source_python))

    def test_linux_facades_are_executable(self) -> None:
        launchers = ("dev.sh", "tools/packman/python.sh", "tools/packman/packman")
        if not (PROJECT_ROOT / ".git").exists():
            if os.name == "nt":
                self.skipTest("POSIX modes are checked in the source ZIP and Linux extraction")
            # An extracted archive has no index; its delivered modes must work.
            for launcher in launchers:
                self.assertTrue((PROJECT_ROOT / launcher).stat().st_mode & 0o111, launcher)
            return
        completed = subprocess.run(
            ["git", "ls-files", "--stage", "--", *launchers],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        records = {
            fields[3]: fields[0]
            for line in completed.stdout.splitlines()
            if len(fields := line.split(maxsplit=3)) == 4
        }
        self.assertEqual(records, {launcher: "100755" for launcher in launchers})


if __name__ == "__main__":
    unittest.main()
