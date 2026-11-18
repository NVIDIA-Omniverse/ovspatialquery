# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import ctypes
from pathlib import Path
import re
import unittest
from ovspatialquery import raw

PUBLIC_HEADERS = Path(__file__).resolve().parents[2] / "include" / "ovspatialquery"
# The same declaration shape cmake/package/verify_install.cmake reads as the export set.
API_FUNCTION = re.compile(r"OVSPATIALQUERY_API\s+[A-Za-z_]\w*\s+(ovspatialquery_\w+)\s*\(")
# Pairs are bound with the core exports.
HEADER_EXPORT_GROUPS = {
    "ovspatialquery.h": "CORE",
    "ovspatialquery_pairs.h": "CORE",
    "ovspatialquery_explicit.h": "EXPLICIT",
    "ovspatialquery_ovstage.h": "OVSTAGE",
}


class RawProjectionTests(unittest.TestCase):
    def test_projects_every_public_header_function(self) -> None:
        expected: dict[str, set[str]] = {"CORE": set(), "EXPLICIT": set(), "OVSTAGE": set()}
        for header in PUBLIC_HEADERS.glob("*.h"):
            names = set(API_FUNCTION.findall(header.read_text(encoding="utf-8")))
            if names:
                self.assertIn(header.name, HEADER_EXPORT_GROUPS)
                expected[HEADER_EXPORT_GROUPS[header.name]] |= names
        for group, names in expected.items():
            with self.subTest(group=group):
                self.assertEqual(set(getattr(raw, f"{group}_EXPORT_SPECS")), names)

    def test_family_batch_columns_and_exports_match_authority(self) -> None:
        for batch, column in (
            (raw.SphereSweepBatch, "radii"),
            (raw.BoxSweepBatch, "half_extents"),
            (raw.OverlapBatch, "radii"),
        ):
            self.assertEqual(dict(batch._fields_)[column], ctypes.POINTER(raw.InputArray))
        self.assertEqual(raw.BoxSweepBatchResult, raw.SphereSweepBatchResult)
        self.assertEqual(raw.BoxSweepBatchReservation, raw.SphereSweepBatchReservation)
        pointer = ctypes.POINTER
        for family, batch, reservation, result in (
            ("raycast", raw.RaycastBatch, raw.RaycastBatchReservation, raw.RaycastBatchResult),
            (
                "sphere_sweep",
                raw.SphereSweepBatch,
                raw.SphereSweepBatchReservation,
                raw.SphereSweepBatchResult,
            ),
            (
                "box_sweep",
                raw.BoxSweepBatch,
                raw.BoxSweepBatchReservation,
                raw.BoxSweepBatchResult,
            ),
            ("overlap", raw.OverlapBatch, raw.OverlapBatchReservation, raw.OverlapBatchResult),
        ):
            self.assertEqual(
                raw.CORE_EXPORT_SPECS[f"ovspatialquery_enqueue_{family}_plan"].argtypes,
                (ctypes.c_uint64, ctypes.c_uint64, pointer(batch), pointer(reservation)),
            )
            self.assertEqual(
                raw.CORE_EXPORT_SPECS[f"ovspatialquery_fetch_{family}_batch"].argtypes,
                (ctypes.c_uint64, ctypes.c_uint64, pointer(result)),
            )

    def test_reporter_growth_signature_is_literal_header_shape(self) -> None:
        growth = raw.CORE_EXPORT_SPECS["ovspatialquery_grow_pair_reporter_capacity"]
        self.assertEqual(growth.argtypes, (ctypes.c_uint64,) * 4)
        self.assertIs(growth.restype, raw.Result)

    def test_geometry_definition_destruction_signature_is_literal(self) -> None:
        destruction = raw.EXPLICIT_EXPORT_SPECS["ovspatialquery_destroy_geometry_definition"]
        self.assertEqual(
            destruction.argtypes,
            (ctypes.c_uint64, raw.GeometryRef),
        )
        self.assertIs(destruction.restype, raw.Result)

    def test_explicit_adapter_signatures_are_literal(self) -> None:
        pointer = ctypes.POINTER
        explicit = raw.EXPLICIT_EXPORT_SPECS
        self.assertEqual(
            explicit["ovspatialquery_enqueue_explicit_population_set"].argtypes,
            (ctypes.c_uint64, ctypes.c_uint64, pointer(raw.ExplicitPopulationSet)),
        )
        self.assertEqual(
            explicit["ovspatialquery_enqueue_explicit_population_update"].argtypes,
            (
                ctypes.c_uint64,
                ctypes.c_uint64,
                pointer(raw.ExplicitPopulationUpdate),
            ),
        )
        self.assertEqual(
            explicit["ovspatialquery_create_explicit_view"].argtypes,
            (
                ctypes.c_uint64,
                ctypes.c_uint64,
                pointer(raw.ExplicitViewDesc),
                pointer(ctypes.c_uint64),
            ),
        )
        registration_layouts = {
            "ovspatialquery_enqueue_sphere_registration": raw.SphereDefinitionTable,
            "ovspatialquery_enqueue_box_registration": raw.BoxDefinitionTable,
            "ovspatialquery_enqueue_capsule_registration": raw.CapsuleDefinitionTable,
            "ovspatialquery_enqueue_convex_registration": raw.ConvexDefinitionTable,
            "ovspatialquery_enqueue_triangle_mesh_registration": (raw.TriangleMeshDefinitionTable),
        }
        for name, layout in registration_layouts.items():
            self.assertEqual(
                explicit[name].argtypes,
                (ctypes.c_uint64, pointer(layout)),
            )


if __name__ == "__main__":
    unittest.main()
