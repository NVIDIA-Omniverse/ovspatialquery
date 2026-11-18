# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "examples" / "run_all.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("example_run_all", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load example runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunAllContractTests(unittest.TestCase):
    def test_runner_preserves_venv_python_symlink_identity(self) -> None:
        if not hasattr(os, "symlink"):
            self.skipTest("symlink support is required")
        runner = _load_runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base-python"
            link = root / "venv-python"
            base.write_text("", encoding="utf-8")
            try:
                link.symlink_to(base)
            except OSError as failure:
                self.skipTest(f"could not create symlink: {failure}")
            options = argparse.Namespace(
                python=link,
                frames=3,
                evidence=root / "evidence",
                sdk_root=(PROJECT_ROOT / "_build" / "bootstrap" / "native-stable" / "install"),
                library=None,
                expect_python_source=None,
                interactive=False,
            )

            command = runner._python_command(
                options,
                "01_static_candidate_raycast",
            )

            self.assertEqual(command[0], str(link))
            self.assertNotEqual(command[0], str(base))


if __name__ == "__main__":
    unittest.main()
