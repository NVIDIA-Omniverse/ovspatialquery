# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ovstage publication and bounded public multi-hit calls for the warehouse sample."""

from time import perf_counter
import numpy as np
import ovstage
from ovspatialquery import Aabb, OverlapQuery, PopulationConfig


def _prim_path(value):
    """An object ID's value as the path dictionary takes it: an int on Ovstage 0.3, a
    PrimPath on 0.4."""
    import ovstage

    handle = getattr(ovstage, "PrimPath", None)
    value = int(getattr(value, "raw_id", value))
    return value if handle is None else handle(value)


def path_filter(paths):
    return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, tuple(paths))])


def population(instance, stage, paths, ordinal=2, retain=2, categories=None, subsets=None):
    pop = instance.ovstage(ovstage, stage).create_population(
        path_filter(paths), PopulationConfig(retain)
    )
    pop.sample_subsets = {}
    for name, members in (subsets or {}).items():
        pop.sample_subsets[name] = pop.create_view(path_filter(members))
    if categories:
        pop.set_participation_map(0, categories)
    started = perf_counter()
    try:
        op = pop.enqueue_set(ordinal)
        try:
            op.wait()
            receipt = op.fetch_population_update()
        finally:
            op.close()
    except BaseException:
        close_population(pop)
        raise
    return pop, receipt, (perf_counter() - started) * 1000


def identities(pop, stage):
    # Enumerate identities through a real bounded query; an object ID's value is its prim path.
    found = pop.target.overlap_multiple(
        OverlapQuery(Aabb((-10000,) * 3, (10000,) * 3)), pop.info().object_count
    )
    if found.truncated:
        raise RuntimeError("Identity discovery was truncated")
    dictionary = ovstage.PathDictionary(stage)
    return {
        hit.object_id: str(dictionary.path_to_string(_prim_path(hit.object_id)))
        for hit in found.hits
    }


class QueryStage:
    """A query-only Ovstage source, also used by --sanity without constructing RTX."""

    def __init__(self, scene):
        self.stage = ovstage.Stage("warehouse.query-source")
        ovstage.population.open_usd_from_string_with_desc(
            self.stage,
            scene,
            1,
            0,
            descs=[
                ovstage.population.Desc(
                    # Ovstage 0.4 has no ALL domain; physics comes from selectors.
                    domains=getattr(
                        ovstage.PopulationDomain, "ALL", ovstage.PopulationDomain.RENDERING
                    ),
                    selectors=[
                        ovstage.population.Selector(
                            prim_predicate=ovstage.population.PrimPredicate(
                                ovstage.population.PrimPredicateKind.ALL
                            ),
                            property_predicate=ovstage.population.PropertyPredicate(
                                ovstage.population.PropertyPredicateKind.HAS_NAME,
                                ["physics:approximation"],
                            ),
                        )
                    ],
                )
            ],
        )
        self.stage.advance_write_floor(1).wait()
        self.stage.compute_hierarchy(1, 2, ovstage.HierarchyComputationModel.DEFAULT_CPU)
        self.stage.advance_write_floor(2).wait()
        self.ordinal = 2
        self.paths = ovstage.PathDictionary(self.stage)

    def transforms(self, paths, matrices):
        self.ordinal += 1
        path_list = self.paths.create_path_list_from_strings(paths)
        try:
            with self.stage.query_from_path_list(path_list) as query:
                value = np.ascontiguousarray(matrices, dtype=np.float64)
                dtype = ovstage.numpy_to_dldatatype(value.dtype, lanes=16)
                tensor = ovstage.make_dltensor(value, dtype=dtype, shape=[len(paths)], ndim=1)
                self.stage.write_attribute(
                    query,
                    self.paths.intern_token("omni:xform"),
                    self.ordinal,
                    tensor,
                    is_array=False,
                    semantic=ovstage.AttributeSemantic.MATRIX,
                ).wait()
        finally:
            self.paths.destroy_path_list(path_list)
        self.stage.advance_write_floor(self.ordinal).wait()
        self.stage.compute_hierarchy(
            self.ordinal, self.ordinal + 1, ovstage.HierarchyComputationModel.DEFAULT_CPU
        )
        self.ordinal += 1
        self.stage.advance_write_floor(self.ordinal).wait()
        return self.ordinal

    def close(self):
        self.stage.destroy()


def close_population(pop):
    for view in getattr(pop, "sample_subsets", {}).values():
        view.close()
    pop.close()
