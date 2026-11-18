# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two copies of the installed library, loaded from two files in one process, work side by side:
each answers for its own objects although both return the same handle values, and one keeps
working after the other's instance is closed. On Linux the first copy is loaded into the global
symbol scope, as for an application linked to the SDK that also imports the wheel."""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

from ovspatialquery import (
    GeometryType,
    HitMode,
    Library,
    PairReporterConfig,
    RaycastPlanConfig,
    RaycastQuery,
)
from ovspatialquery.raw import Capability

LIBRARY: Path | None = None
RAY = RaycastQuery(origin=(-3.0, 0.0, 0.0), unit_direction=(1.0, 0.0, 0.0), max_distance=100.0)


def _publish(instance, sphere_count: int):
    """`sphere_count` unit spheres 4 apart on the x axis and, at x = 50, one triangle facing the
    ray, published as ordinal 0. The ray's exact hits build the triangle's PhysX query structure,
    so each copy uses its own PhysX foundation."""
    adapter = instance.explicit()
    adapter.register_spheres(
        definition_values=np.array([1], dtype=np.uint64), radii=np.array([1.0]), copy=True
    ).finish()
    adapter.register_triangle_meshes(
        definition_values=np.array([2], dtype=np.uint64),
        vertex_offsets=np.array([0], dtype=np.uint64),
        vertex_counts=np.array([3], dtype=np.uint64),
        triangle_offsets=np.array([0], dtype=np.uint64),
        triangle_counts=np.array([1], dtype=np.uint64),
        vertices=np.array([(0.0, -5.0, -5.0), (0.0, 5.0, -5.0), (0.0, 0.0, 5.0)], np.float32),
        triangle_indices=np.array([(0, 1, 2)], dtype=np.uint32),
        double_sided=np.array([1], dtype=np.uint8),
        copy=True,
    ).finish()
    count = sphere_count + 1
    population = adapter.create_population()
    with population.enqueue_set(
        0,
        object_values=np.arange(10, 10 + count, dtype=np.uint64),
        geometry_types=np.array(
            [GeometryType.SPHERE] * sphere_count + [GeometryType.TRIANGLE_MESH], dtype=np.int32
        ),
        geometry_values=np.array([1] * sphere_count + [2], dtype=np.uint64),
        translations=np.array(
            [(4.0 * index, 0.0, 0.0) for index in range(sphere_count)] + [(50.0, 0.0, 0.0)]
        ),
        rotations_xyzw=np.array([(0.0, 0.0, 0.0, 1.0)] * count, dtype=np.float32),
        copy=True,
    ) as publication:
        publication.wait()
    return population


def _identity(library: Library) -> int:
    """The address of `ovspatialquery_get_version` in the copy `library` loaded."""
    return ctypes.cast(library.exports["ovspatialquery_get_version"], ctypes.c_void_p).value


class TwoCopiesTest(unittest.TestCase):
    def setUp(self) -> None:
        assert LIBRARY is not None
        scratch = Path(tempfile.mkdtemp(prefix="ovspatialquery-two-copies-"))
        self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        paths = []
        for name in ("a", "b"):
            (scratch / name).mkdir()
            paths.append(Path(shutil.copy2(LIBRARY, scratch / name / LIBRARY.name)))
        if sys.platform != "win32":
            # Copy a in the global scope: copy b's calls to an exported name would resolve
            # there if the library bound them at run time.
            self.global_copy = ctypes.CDLL(str(paths[0]), mode=os.RTLD_GLOBAL)
        self.library_a = Library.open(paths[0], require_explicit=True)
        self.addCleanup(self.library_a.close)
        self.library_b = Library.open(paths[1], require_explicit=True)
        self.addCleanup(self.library_b.close)

    def test_two_copies_answer_for_their_own_objects(self) -> None:
        self.assertNotEqual(_identity(self.library_a), _identity(self.library_b))
        with self.library_a.instance() as instance_a, self.library_b.instance() as instance_b:
            # Different object counts tell the two populations apart.
            population_a = _publish(instance_a, 1)
            population_b = _publish(instance_b, 3)
            # Each copy numbers its handles the same way, so only the copy tells them apart.
            self.assertEqual(population_a.handle, population_b.handle)
            # The spheres plus the triangle.
            for instance, population, count in (
                (instance_a, population_a, 2),
                (instance_b, population_b, 4),
            ):
                with self.subTest(copy=instance.library.path.parent.name):
                    self.assertEqual(population.info().object_count, count)
                    self.assertTrue(instance.supports(Capability.RAYCAST_PLAN_EXACT))
                    hits = population.target.raycast_multiple(RAY, 8)
                    self.assertEqual(len(hits.hits), count)
                    reference = population.target.reference
                    instance.create_raycast_plan(
                        RaycastPlanConfig(reference, HitMode.CLOSEST, 1, 1, 1)
                    ).close()
                    instance.create_pair_reporter(
                        PairReporterConfig(reference, reference, 4, 4)
                    ).close()

    def test_a_copy_keeps_working_after_the_other_closes(self) -> None:
        with self.library_b.instance() as instance_b:
            population_b = _publish(instance_b, 2)
            with self.library_a.instance() as instance_a:
                population_a = _publish(instance_a, 1)
                # Both copies build their PhysX query structures before copy a's instance closes.
                self.assertEqual(len(population_a.target.raycast_multiple(RAY, 8).hits), 2)
                self.assertEqual(len(population_b.target.raycast_multiple(RAY, 8).hits), 3)
            self.assertEqual(population_b.info().object_count, 3)
            self.assertAlmostEqual(population_b.target.raycast_closest(RAY).hit.distance, 2.0)
            self.assertEqual(len(population_b.target.raycast_multiple(RAY, 8).hits), 3)


def main() -> int:
    global LIBRARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--dll", required=True, type=Path)
    arguments, remaining = parser.parse_known_args()
    LIBRARY = arguments.dll.resolve()
    program = unittest.main(argv=[sys.argv[0], *remaining], exit=False)
    return 0 if program.result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
