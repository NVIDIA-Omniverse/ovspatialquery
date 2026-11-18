# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
sys.path.insert(0, str(EXAMPLES_ROOT / "python"))

from _evidence import (  # noqa: E402
    EvidencePaths,
    compare_semantics,
    validate_evidence,
    write_semantic_json,
)
from _runtime import FRAME_DEVICE, frame_summary  # noqa: E402
import warp as wp  # noqa: E402


class EvidenceContractTests(unittest.TestCase):
    def test_frame_and_semantic_json_are_deterministic_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = EvidencePaths.create(root, "01_static_candidate_raycast", "python")
            pixels = wp.zeros((720, 1280, 4), dtype=wp.uint8, device=FRAME_DEVICE)
            pixels[:, :, 3].fill_(255)
            for channel, value in enumerate((32, 144, 220)):
                pixels[100:620, 200:1080, channel].fill_(value)

            image = frame_summary(pixels)
            # Warp counts the same on its CPU device as on the frame device.
            self.assertEqual(frame_summary(wp.clone(pixels, device="cpu")), image)
            self.assertEqual((image["nonblack_pixels"], image["unique_rgb_count"]), (520 * 880, 2))
            payload = {
                "schema_version": 2,
                "example": "01_static_candidate_raycast",
                "language": "python",
                "execution": {
                    "headless": True,
                    "frames": 3,
                    "resolution": [1280, 720],
                },
                "inputs": {"ray": {"origin": [0.0, 0.0, 0.0]}},
                "publication": {
                    "source_ordinal": 2,
                    "object_count": 1,
                    "excluded": 0,
                },
                "result": {
                    "has_hit": True,
                    "prim_path": "/World/Targets/Hit",
                },
                "native": {"ovspatialquery": {"runtime": [0, 1, 0]}},
                "evidence": image,
                "cleanup": {"clean": True, "errors": []},
                "errors": [],
            }
            write_semantic_json(paths.json, payload)

            validated = validate_evidence(paths.json)
            self.assertEqual(validated["evidence"]["width"], 1280)
            self.assertEqual(validated["evidence"]["height"], 720)
            self.assertGreater(validated["evidence"]["nonblack_pixels"], 0)
            self.assertEqual(
                json.loads(paths.json.read_text(encoding="utf-8")),
                payload,
            )
            for dimension in ("width", "height"):
                with self.subTest(dimension=dimension):
                    payload["evidence"] = {**image, dimension: 1}
                    write_semantic_json(paths.json, payload)
                    with self.assertRaisesRegex(ValueError, "frame dimensions"):
                        validate_evidence(paths.json)
            payload["evidence"] = image
            payload["schema_version"] = 1
            write_semantic_json(paths.json, payload)
            with self.assertRaisesRegex(ValueError, "PNG path and digest"):
                validate_evidence(paths.json)
            payload["schema_version"] = 99
            write_semantic_json(paths.json, payload)
            with self.assertRaisesRegex(ValueError, "schema version"):
                validate_evidence(paths.json)

    def test_language_and_artifact_identity_do_not_mask_semantic_drift(self) -> None:
        shared = {
            "schema_version": 1,
            "example": "01_static_candidate_raycast",
            "execution": {"headless": True, "frames": 3, "resolution": [1280, 720]},
            "inputs": {"ray": {"origin": [0.0, 0.0, 0.0]}},
            "publication": {
                "source_ordinal": 2,
                "object_count": 1,
                "excluded": 0,
            },
            "result": {
                "has_hit": True,
                "object_id": 17,
                "nested": [{"first_object_id": 18, "first_path": "/World/A"}],
                "prim_path": "/World/Targets/Hit",
            },
            "cleanup": {"clean": True, "errors": []},
            "errors": [],
        }
        python = {
            **shared,
            "language": "python",
            "schema_version": 2,
            "native": {"ovspatialquery": {"library_sha256": "a"}},
            "evidence": {"nonblack_pixels": 1, "unique_rgb_count": 2},
        }
        cpp = {
            **shared,
            "language": "cpp",
            "native": {"ovspatialquery": {"library_sha256": "c"}},
            "evidence": {"png": "cpp.png", "sha256": "d"},
        }
        cpp["result"] = {
            **cpp["result"],
            "object_id": 99,
            "nested": [{"first_object_id": 100, "first_path": "/World/A"}],
        }
        compare_semantics(python, cpp)
        cpp["result"] = {"has_hit": False, "prim_path": ""}
        with self.assertRaisesRegex(ValueError, "semantic mismatch"):
            compare_semantics(python, cpp)


if __name__ == "__main__":
    unittest.main()
