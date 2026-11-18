/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// The two spellings a lesson needs to build against Ovstage 0.3 and 0.4: 0.4 nests the
// function table in `data_plane` and has no ALL population domain.

#include <ovstage/ovstage.h>

namespace example_ovstage
{

#if OVSTAGE_VERSION_MINOR >= 4
inline const ovstage_data_plane_vtable_t* table(ovstage_instance_t* stage) { return stage->data_plane.vtable; }
inline ovstage_data_plane_context_t*      context(ovstage_instance_t* stage) { return stage->data_plane.context; }
#else
inline const ovstage_vtable_t* table(ovstage_instance_t* stage) { return stage->vtable; }
inline ovstage_context_t*      context(ovstage_instance_t* stage) { return stage->context; }
#endif

// Opens USDA text with what ovspatialquery reads: 0.3's ALL domain, or on 0.4, which has no
// PHYSICS domain, RENDERING plus every prim's physics, physxConvexHullCollision and
// material:binding properties.
inline ovstage_population_enqueue_result_t openUsda(ovstage_instance_t* stage, ovx_string_t usda,
                                                    ovstage_ordinal_t ordinal, double time)
{
#if OVSTAGE_VERSION_MINOR >= 4
    static const ovx_string_t namespaces[] = {
        {"physics", 7u}, {"physxConvexHullCollision", 24u}, {"material:binding", 16u}};
    ovstage_population_selector_t physics{};
    physics.prim_predicate.kind               = OVSTAGE_POPULATION_PRIM_PREDICATE_ALL;
    physics.property_predicate.kind           = OVSTAGE_POPULATION_PROPERTY_PREDICATE_IN_NAMESPACE;
    physics.property_predicate.values.strings = namespaces;
    physics.property_predicate.value_count    = 3u;
    ovstage_population_desc_t desc{};
    desc.domains        = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    desc.selectors      = &physics;
    desc.selector_count = 1u;
    return ovstage_population_open_usd_from_string_with_desc(stage, usda, ordinal, time, &desc, 1u);
#else
    return ovstage_population_open_usd_from_string(stage, usda, ordinal, time, OVSTAGE_POPULATION_DOMAIN_ALL);
#endif
}

} // namespace example_ovstage
