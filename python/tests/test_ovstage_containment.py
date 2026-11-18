# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed ingestion preserves the distinction between solids and mesh surfaces."""

import os
from pathlib import Path
import sys
import unittest

from tests.ovstage_population import populate
from ovspatialquery import Aabb, Library, OverlapQuery, QueryPrecision, Sphere

try:
    import ovstage
except ImportError:
    ovstage = None


@unittest.skipIf(ovstage is None, "Ovstage is an optional dependency")
class OvstageContainmentTests(unittest.TestCase):
    def test_analytical_cube_and_closed_mesh_have_distinct_exact_semantics(self):
        usda = """#usda 1.0
def Cube "Cube"
{
    double size = 2
}
def Mesh "Mesh"
{
    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    float3[] extent = [(-1, -1, -1), (1, 1, 1)]
    uniform token subdivisionScheme = "none"
}
"""
        source = ovstage.Stage("containment-test")
        populate(source, usda, 1, 0.0)
        source.advance_write_floor(ordinal=1).wait()
        source.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        source.advance_write_floor(ordinal=2).wait()
        # The test facade builds this install; an explicit environment also permits
        # qualification of a separately installed package without rebuilding it.
        install = Path(__file__).resolve().parents[2] / "_build/bootstrap/native-stable/install"
        artifact = install / (
            "bin/ovspatialquery.dll" if sys.platform == "win32" else "lib/libovspatialquery.so"
        )
        selected = os.environ.get("OVSPATIALQUERY_LIBRARY") or (
            artifact if artifact.is_file() else None
        )
        with (
            Library.open(selected, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            for path in ("/Cube", "/Mesh"):
                population = instance.ovstage(ovstage, source).create_population(
                    ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, (path,))])
                )
                try:
                    operation = population.enqueue_set(2)
                    try:
                        operation.wait()
                        self.assertEqual(operation.fetch_population_update().object_count, 1)
                    finally:
                        operation.close()
                    for shape_name, shape, intersects_surface in (
                        ("contained box", Aabb((-0.1,) * 3, (0.1,) * 3), False),
                        ("contained sphere", Sphere((0, 0, 0), 0.1), False),
                        ("crossing box", Aabb((0.9, -0.1, -0.1), (1.1, 0.1, 0.1)), True),
                        ("crossing sphere", Sphere((1, 0, 0), 0.1), True),
                        ("outside box", Aabb((2,) * 3, (3,) * 3), False),
                        ("outside sphere", Sphere((2, 0, 0), 0.1), False),
                    ):
                        for precision in (
                            QueryPrecision.EXACT,
                            QueryPrecision.BOUNDS_CANDIDATE,
                        ):
                            with self.subTest(target=path, shape=shape_name, precision=precision):
                                inside = shape_name.startswith("contained")
                                expected = intersects_surface or (
                                    inside
                                    and (
                                        path == "/Cube"
                                        or precision == QueryPrecision.BOUNDS_CANDIDATE
                                    )
                                )
                                self.assertEqual(
                                    bool(
                                        population.target.overlap_any(
                                            OverlapQuery(shape, precision=precision)
                                        ).hit
                                    ),
                                    expected,
                                )
                finally:
                    population.close()


if __name__ == "__main__":
    unittest.main()
