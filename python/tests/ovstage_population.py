# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Populate a test stage the same way on Ovstage 0.3 and 0.4 (ADR-0069)."""

from __future__ import annotations


def populate(source, usda: str, ordinal: int = 1, time: float = 0.0, domains: str = "ALL") -> None:
    """Open `usda` into `source` with `domains` "ALL", "RENDERING" or "PHYSICS".

    Ovstage 0.3 has those population domains. 0.4 has only RENDERING and populates
    physics through a selector; there "PHYSICS" is every prim's `physics` and
    `material:binding` properties, the data the companion reads for convex hulls and
    physics materials, and "ALL" adds RENDERING.
    """
    import ovstage

    domain = ovstage.PopulationDomain
    if hasattr(domain, "ALL"):
        ovstage.population.open_usd_from_string(
            source, usda, ordinal, time, getattr(domain, domains)
        )
        return
    if domains == "RENDERING":
        ovstage.population.open_usd_from_string(source, usda, ordinal, time, domain.RENDERING)
        return
    population = ovstage.population
    physics = population.Selector(
        prim_predicate=population.PrimPredicate.all(),
        property_predicate=population.PropertyPredicate.in_namespace(
            "physics", "material:binding", "physxConvexHullCollision"
        ),
    )
    desc = population.Desc(
        domain.RENDERING if domains == "ALL" else domain.NONE, selectors=(physics,)
    )
    population.open_usd_from_string_with_desc(source, usda, ordinal, time, desc)
