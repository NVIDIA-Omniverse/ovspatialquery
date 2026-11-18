# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "examples" / "python"


def _load_evidence():
    path = PYTHON_ROOT / "_evidence.py"
    spec = importlib.util.spec_from_file_location("example_evidence", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load example evidence module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class InteractiveOrbitContractTests(unittest.TestCase):
    def test_x11_close_filter_accepts_only_the_registered_protocol(self) -> None:
        evidence = _load_evidence()
        self.assertTrue(
            evidence._x11_event_requests_close(
                event_type=evidence.X11_CLIENT_MESSAGE,
                client_atom=17,
                wm_delete_atom=17,
                keysym=0,
            )
        )
        self.assertFalse(
            evidence._x11_event_requests_close(
                event_type=evidence.X11_CLIENT_MESSAGE,
                client_atom=23,
                wm_delete_atom=17,
                keysym=0,
            )
        )
        self.assertTrue(
            evidence._x11_event_requests_close(
                event_type=evidence.X11_DESTROY_NOTIFY,
                client_atom=0,
                wm_delete_atom=17,
                keysym=0,
            )
        )
        self.assertTrue(
            evidence._x11_event_requests_close(
                event_type=evidence.X11_KEY_PRESS,
                client_atom=0,
                wm_delete_atom=17,
                keysym=evidence.X11_ESCAPE_KEYSYM,
            )
        )


if __name__ == "__main__":
    unittest.main()
