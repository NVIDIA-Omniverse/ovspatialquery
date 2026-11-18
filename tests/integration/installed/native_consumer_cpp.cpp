/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <cstdint>
#include <type_traits>
#include <unordered_map>

static_assert(std::is_same<ovspatialquery_ordinal_t, int64_t>::value, "ordinal ABI");
static_assert(std::is_same<decltype(ovspatialquery_raycast_plan_desc_t::result_fields), uint32_t>::value,
              "raycast result-field mask");
static_assert(std::is_same<decltype(ovspatialquery_sphere_sweep_plan_desc_t::result_fields), uint32_t>::value,
              "sphere-sweep result-field mask");
static_assert(std::is_same<decltype(ovspatialquery_overlap_plan_desc_t::result_fields), uint32_t>::value,
              "overlap result-field mask");

using CreateInstance = ovspatialquery_result_t (*)(const ovspatialquery_instance_desc_t*,
                                                   ovspatialquery_instance_handle_t*);
using LastOpError    = ovspatialquery_string_t (*)(ovspatialquery_instance_handle_t, ovspatialquery_op_id_t);
using RaycastSync = ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t, const ovspatialquery_raycast_desc_t*,
                                                ovspatialquery_raycast_single_result_t*);
using CreateRaycastPlan        = ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t,
                                                      const ovspatialquery_raycast_plan_desc_t*,
                                                      ovspatialquery_batch_plan_handle_t*);
using CreateReporter           = ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t,
                                                   const ovspatialquery_pair_reporter_desc_t*,
                                                   ovspatialquery_pair_reporter_handle_t*);
using GrowReporter             = ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t,
                                                 ovspatialquery_pair_reporter_handle_t, uint64_t, uint64_t);
using EnqueueRaycastPlan       = ovspatialquery_enqueue_result_t (*)(ovspatialquery_instance_handle_t,
                                                               ovspatialquery_batch_plan_handle_t,
                                                               const ovspatialquery_raycast_batch_t*,
                                                               ovspatialquery_raycast_batch_reservation_t*);
using EnqueueSphereSweepPlan   = ovspatialquery_enqueue_result_t (*)(ovspatialquery_instance_handle_t,
                                                                   ovspatialquery_batch_plan_handle_t,
                                                                   const ovspatialquery_sphere_sweep_batch_t*,
                                                                   ovspatialquery_sphere_sweep_batch_reservation_t*);
using EnqueueOverlapPlan       = ovspatialquery_enqueue_result_t (*)(ovspatialquery_instance_handle_t,
                                                               ovspatialquery_batch_plan_handle_t,
                                                               const ovspatialquery_overlap_batch_t*,
                                                               ovspatialquery_overlap_batch_reservation_t*);
using EnqueueReporter          = ovspatialquery_enqueue_result_t (*)(ovspatialquery_instance_handle_t,
                                                            ovspatialquery_pair_reporter_handle_t,
                                                            const ovspatialquery_pair_reporter_advance_t*,
                                                            ovspatialquery_pair_reporter_reservation_t*);
using CreateExplicitPopulation = ovspatialquery_result_t (*)(ovspatialquery_instance_handle_t,
                                                             const ovspatialquery_population_desc_t*,
                                                             ovspatialquery_population_handle_t*);
using RegisterTriangleMesh     = ovspatialquery_enqueue_result_t (*)(
    ovspatialquery_instance_handle_t, const ovspatialquery_triangle_mesh_definition_table_t*);

static_assert(std::is_same<decltype(&ovspatialquery_create_instance), CreateInstance>::value, "instance signature");
static_assert(std::is_same<decltype(&ovspatialquery_get_last_op_error), LastOpError>::value, "op error signature");
static_assert(std::is_same<decltype(&ovspatialquery_raycast_closest_sync), RaycastSync>::value, "ray signature");
static_assert(std::is_same<decltype(&ovspatialquery_create_raycast_plan), CreateRaycastPlan>::value, "plan signature");
static_assert(std::is_same<decltype(&ovspatialquery_create_pair_reporter), CreateReporter>::value,
              "reporter signature");
static_assert(std::is_same<decltype(&ovspatialquery_grow_pair_reporter_capacity), GrowReporter>::value,
              "reporter growth signature");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_raycast_plan), EnqueueRaycastPlan>::value,
              "raycast enqueue reservation");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_sphere_sweep_plan), EnqueueSphereSweepPlan>::value,
              "sweep enqueue reservation");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_overlap_plan), EnqueueOverlapPlan>::value,
              "overlap enqueue reservation");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_pair_reporter_advance), EnqueueReporter>::value,
              "reporter enqueue reservation");
static_assert(std::is_same<decltype(&ovspatialquery_create_explicit_population), CreateExplicitPopulation>::value,
              "explicit population signature");
static_assert(std::is_same<decltype(&ovspatialquery_enqueue_triangle_mesh_registration), RegisterTriangleMesh>::value,
              "explicit mesh registration signature");

int main()
{
    uint32_t major = 0u;
    uint32_t minor = 0u;
    uint32_t patch = 0u;
    if (ovspatialquery_get_version(&major, &minor, &patch).status != OVSPATIALQUERY_API_SUCCESS)
    {
        return 1;
    }

    // Handles compare, order and hash by raw_id, and only against their own kind.
    const ovspatialquery_view_handle_t                    first{1u};
    const ovspatialquery_view_handle_t                    second{2u};
    std::unordered_map<ovspatialquery_view_handle_t, int> views{{first, 1}, {second, 2}};
    const ovspatialquery_op_id_t                          invalid = OVSPATIALQUERY_INVALID_OP;
    const bool handles = first == ovspatialquery_view_handle_t{1u} && first != second && first < second &&
                         invalid == OVSPATIALQUERY_INVALID_OP && !ovspatialquery_op_id_is_valid(invalid) &&
                         ovspatialquery_view_handle_is_valid(first) && views.size() == 2u && views.at(second) == 2 &&
                         ovspatialquery_pair_reporter_handle_t{3u} != OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    return handles ? 0 : 2;
}
