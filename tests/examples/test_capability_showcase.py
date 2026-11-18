# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract for lesson 11's standalone installed capability showcase."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRESENTATION = PROJECT_ROOT / "examples" / "python" / "11_capability_showcase.py"
sys.path.insert(0, str(PRESENTATION.parent))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CapabilityShowcaseSourceTests(unittest.TestCase):
    def test_station_geometry_moves_with_native_evidence(self) -> None:
        presentation = _load_module("capability_showcase_data_driven", PRESENTATION)
        station_by_panel = {station.panel: station for station in presentation.STATIONS}
        payload = {
            "ovstage": {
                "source": {"translation": [12, 0, 0], "radius": 2},
                "accepted": {
                    "origin": [-4, 0, 0],
                    "position": [10, 0, 0],
                    "normal": [-1, 0, 0],
                },
            },
            "explicit": {
                "objects": [
                    {"object_id": 101, "translation": [8, 0, 0], "radius": 2},
                    {"object_id": 202, "translation": [10, 0, 0], "radius": 1},
                    {"object_id": 404, "translation": [0, 0, 0]},
                ],
                "candidate": {
                    "object_id": 101,
                    "origin": [-6, 0, 0],
                    "position": [2, 0, 0],
                    "normal": [-1, 0, 0],
                },
                "pair": [101, 202],
            },
            "convex_preparation": {
                "authored": {"points": [[0, 0, 0]]},
                "exact": {"position": [0, 1, 0]},
            },
        }
        ovstage = presentation._station_contents(station_by_panel["ovstage"], payload)
        explicit = presentation._station_contents(station_by_panel["explicit"], payload)
        self.assertIn("double3 xformOp:translate = (6.0, 0.0, -3.0)", ovstage)
        self.assertIn('def Xform "RejectedByBits"', ovstage)
        self.assertIn("double3 xformOp:translate = (4.0, 0.0, -3.0)", explicit)
        self.assertIn("double3 xformOp:translate = (-1.0, 0.0, -3.0)", explicit)

    def test_presentation_requires_real_cuda_family_evidence(self) -> None:
        presentation = _load_module("capability_showcase_presentation_cuda", PRESENTATION)
        payload = {
            "schema_version": 1,
            "status": "PASS",
            "cleanup": True,
            "lanes": {
                "core": "PASS",
                "ovstage": "PASS",
                "participation": "PASS",
                "explicit_cpu": "PASS",
                "cuda": "PASS",
            },
            "scenarios": [
                {
                    "panel": "ovstage",
                    "label": "Ovstage participation accept/reject",
                    "truth": "BOUNDS_CANDIDATE",
                },
                {
                    "panel": "explicit",
                    "label": "Explicit publication, candidate, and pair",
                    "truth": "BOUNDS_CANDIDATE",
                },
                {
                    "panel": "cuda",
                    "label": "CUDA exact family columns",
                    "truth": "EXACT_SURFACE",
                },
            ],
            "ovstage": {
                "publication": {"source_ordinal": 2, "object_count": 1},
                "accepted_query_bits": 0x40,
                "rejected_query_bits": 0x80,
                "source": {"translation": [2, 0, 0], "radius": 1},
                "accepted": {
                    "truth": "BOUNDS_CANDIDATE",
                    "source_ordinal": 2,
                    "object_id": 1,
                    "geometry_type": "SPHERE",
                    "geometry_value": 11,
                    "origin": [-3, 0, 0],
                    "position": [1, 0, 0],
                    "normal": [-1, 0, 0],
                    "distance": 4,
                },
            },
            "explicit": {
                "publication": {"source_ordinal": 1, "object_count": 3},
                "objects": [
                    {
                        "object_id": 101,
                        "geometry_type": "SPHERE",
                        "translation": [4, 0, 0],
                        "radius": 2,
                    },
                    {
                        "object_id": 202,
                        "geometry_type": "SPHERE",
                        "translation": [5, 0, 0],
                        "radius": 1,
                    },
                    {"object_id": 404, "geometry_type": "CONVEX", "translation": [0, 0, 0]},
                ],
                "candidate": {
                    "truth": "BOUNDS_CANDIDATE",
                    "source_ordinal": 1,
                    "object_id": 101,
                    "geometry_type": "SPHERE",
                    "geometry_value": 7,
                    "origin": [-2, 0, 0],
                    "position": [2, 0, 0],
                    "normal": [-1, 0, 0],
                    "distance": 4,
                },
                "pair": [101, 202],
            },
            "cuda": {
                "state": "PASS",
                "device": "kDLCUDA:0",
                "source_ordinal": 9,
                "families": {
                    name: {
                        "truth": "EXACT_SURFACE",
                        "row_status": "SUCCESS",
                        "stored_hit_count": 1,
                        "hit_offset": 0,
                        "hit_count": 1,
                        "truncated": 0,
                        "object_id": 303,
                        "geometry_type": "TRIANGLE_MESH",
                        "geometry_value": 88,
                        "origin": [0, 0, 0],
                        "position": [4, 0, 0],
                        "normal": [-1, 0, 0],
                        "distance": 4 if name == "raycast" else 3.5,
                    }
                    for name in ("raycast", "sphere_sweep", "overlap")
                },
            },
        }
        for name in ("raycast", "sphere_sweep"):
            payload["cuda"]["families"][name]["feature"] = {
                "kind": "TRIANGLE",
                "source_face": 0,
                "source_corners": [0, 1, 2],
                "barycentrics": [0.25, 0.25, 0.5],
            }
        payload["cuda"]["families"]["sphere_sweep"]["query_position"] = [3.5, 0, 0]
        payload["convex_preparation"] = {
            "authored": {
                "points": [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [0, 0, 0],
                ],
                "dent_position": [0, 0, 0],
            },
            "publication": {"source_ordinal": 1, "object_count": 3},
            "exact": {
                "truth": "EXACT_SURFACE",
                "source_ordinal": 1,
                "object_id": 404,
                "geometry_type": "CONVEX",
                "geometry_value": 91,
                "origin": [0, 3, 0],
                "position": [0, 1, 0],
                "normal": [0, 1, 0],
                "distance": 2,
                "feature": "NONE",
            },
        }
        self.assertIs(presentation._validate(payload, require_cuda=True), payload)
        unavailable = copy.deepcopy(payload)
        unavailable["lanes"]["cuda"] = "NOT_AVAILABLE"
        unavailable["scenarios"][2] = {
            "panel": "cuda",
            "label": "CUDA unavailable",
            "truth": "NOT_AVAILABLE",
        }
        unavailable["cuda"] = {
            "state": "NOT_AVAILABLE",
            "reason": "CUDA validation was not requested",
        }
        self.assertIs(presentation._validate(unavailable), unavailable)
        cuda_station = next(station for station in presentation.STATIONS if station.panel == "cuda")
        self.assertIn(
            'def Xform "Unavailable"',
            presentation._station_contents(cuda_station, unavailable),
        )
        with self.assertRaisesRegex(RuntimeError, "requires real CUDA"):
            presentation._validate(unavailable, require_cuda=True)
        payload["convex_preparation"]["exact"]["position"] = [0, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "convex envelope"):
            presentation._validate(payload)
        payload["convex_preparation"]["exact"]["position"] = [0, 1, 0]
        payload["cuda"]["families"]["overlap"]["stored_hit_count"] = 0
        with self.assertRaises(RuntimeError):
            presentation._validate(payload)

    def test_qualified_ctest_ignores_a_hostile_path(self) -> None:
        presentation = _load_module("capability_showcase_presentation_ctest", PRESENTATION)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk_root = root / "build" / "install"
            sdk_root.mkdir(parents=True)
            qualified = root / "qualified" / "ctest.exe"
            qualified.parent.mkdir()
            qualified.touch()
            hostile = root / "hostile"
            hostile.mkdir()
            (hostile / "cmake.exe").touch()
            (hostile / "ctest.exe").touch()
            (sdk_root.parent / "CMakeCache.txt").write_text(
                f"CMAKE_CTEST_COMMAND:INTERNAL={qualified}\n", encoding="utf-8"
            )
            with mock.patch.dict("os.environ", {"PATH": str(hostile)}, clear=False):
                self.assertEqual(presentation._qualified_ctest(sdk_root), qualified.resolve())

    def test_linux_arguments_select_portable_cpu_unless_cuda_is_required(self) -> None:
        presentation = _load_module("capability_showcase_linux_arguments", PRESENTATION)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = ["--sdk-root", str(root), "--library", str(root / "lib.so")]
            with mock.patch.object(presentation.sys, "platform", "linux"):
                self.assertFalse(presentation._arguments(common).require_cuda)
                self.assertTrue(presentation._arguments([*common, "--require-cuda"]).require_cuda)

    def test_presentation_rejects_a_library_outside_the_verified_install(self) -> None:
        presentation = _load_module("capability_showcase_presentation", PRESENTATION)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "bin" / "ovspatialquery.dll"
            expected.parent.mkdir()
            expected.touch()
            self.assertEqual(presentation._verified_library(root, "win32"), expected)
            with self.assertRaises(ValueError):
                presentation._require_verified_library(
                    root, root / "other" / "ovspatialquery.dll", "win32"
                )


if __name__ == "__main__":
    unittest.main()
