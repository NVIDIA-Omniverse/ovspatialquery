/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <ovstage/ovstage.h>

static int succeeded(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

static ovspatialquery_input_array_t copied_array(void* data, uint8_t dtype_code, uint8_t dtype_bits, int32_t rank,
                                                 int64_t* shape, int64_t* strides)
{
    ovspatialquery_input_array_t result = {0};
    result.tensor.data                  = data;
    result.tensor.device.device_type    = kDLCPU;
    result.tensor.device.device_id      = 0;
    result.tensor.ndim                  = rank;
    result.tensor.dtype.code            = dtype_code;
    result.tensor.dtype.bits            = dtype_bits;
    result.tensor.dtype.lanes           = 1u;
    result.tensor.shape                 = shape;
    result.tensor.strides               = strides;
    result.ownership                    = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return result;
}

static int wait_stage_op(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    ovstage_op_wait_result_t waited = {0};
    return operation.status == OVSTAGE_OK && operation.op_index.raw_id != OVSTAGE_INVALID_OP_ID.raw_id &&
           stage->data_plane.vtable->wait_op(stage->data_plane.context, operation.op_index, OVSTAGE_TIMEOUT_INFINITE,
                                             &waited) == OVSTAGE_OK &&
           stage->data_plane.vtable->release_op(stage->data_plane.context, operation.op_index) == OVSTAGE_OK;
}

static int finish_update(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t operation,
                         ovspatialquery_ordinal_t expected_ordinal)
{
    ovspatialquery_op_wait_result_t           waited = {0};
    ovspatialquery_population_update_result_t result = {0};
    return operation.status == OVSPATIALQUERY_API_SUCCESS && ovspatialquery_op_id_is_valid(operation.op_id) &&
           succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
           waited.state == OVSPATIALQUERY_OP_COMPLETED &&
           succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &result)) &&
           result.published_ordinal == expected_ordinal && result.object_count == 0u &&
           result.excluded_object_count == 0u && succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)) &&
           !ovspatialquery_op_id_is_valid(operation.op_id);
}

int main(void)
{
    ovstage_instance_t*           stage      = NULL;
    const ovstage_instance_desc_t stage_desc = {"ovspatialquery-ovstage-installed-c"};
    if (ovstage_create_instance(&stage_desc, &stage) != OVSTAGE_OK || stage == NULL)
    {
        return 1;
    }

    ovspatialquery_instance_desc_t instance_desc = {0};
    instance_desc.operation_capacity             = 2u;
    instance_desc.max_cpu_concurrency            = 1u;
    instance_desc.host_byte_budget               = UINT64_C(16) << 20u;
    ovspatialquery_instance_handle_t instance    = OVSPATIALQUERY_INVALID_INSTANCE;
    if (!succeeded(ovspatialquery_create_instance(&instance_desc, &instance)) ||
        !ovspatialquery_instance_handle_is_valid(instance))
    {
        (void)ovstage_destroy_instance(stage);
        return 2;
    }

    ovspatialquery_ovstage_population_desc_t population_desc = {0};
    population_desc.stage                                    = stage;
    population_desc.common.retained_ordinal_capacity         = 1u;
    ovspatialquery_population_handle_t population            = OVSPATIALQUERY_INVALID_POPULATION;
    const ovspatialquery_result_t      population_created =
        ovspatialquery_create_ovstage_population(instance, &population_desc, &population);
    if (!succeeded(population_created))
    {
        return 3;
    }

    ovspatialquery_ovstage_view_desc_t view_desc = {0};
    ovspatialquery_view_handle_t       view      = OVSPATIALQUERY_INVALID_VIEW;
    if (!succeeded(ovspatialquery_create_ovstage_view(instance, population, &view_desc, &view)))
    {
        return 4;
    }

    {
        const ovx_primpath_t                         path  = OVX_INVALID_PRIMPATH;
        const ovspatialquery_participation_bits_t    bits  = {{UINT64_C(1), 0u}};
        ovspatialquery_ovstage_participation_table_t table = {0};
        table.default_bits.words[0]                        = UINT64_C(1);
        if (!succeeded(ovspatialquery_set_ovstage_participation_map(instance, population, &table)))
        {
            return 12;
        }
        table.paths = &path;
        table.bits  = &bits;
        table.count = 1u;
        if (ovspatialquery_set_ovstage_participation_map(instance, population, &table).status !=
            OVSPATIALQUERY_API_INVALID_ARGUMENT)
        {
            return 12;
        }
    }

    ovstage_write_floor_desc_t floor = {1u, OVSTAGE_SCOPE_ALL, NULL, 0u};
    if (!wait_stage_op(stage, stage->data_plane.vtable->advance_write_floor(stage->data_plane.context, &floor)))
    {
        return 5;
    }
    ovspatialquery_enqueue_result_t operation = ovspatialquery_enqueue_ovstage_population_set(instance, population, 1);
    if (!finish_update(instance, operation, 1))
    {
        return 6;
    }

    if (!succeeded(ovspatialquery_set_ovstage_participation_map(instance, population, NULL)))
    {
        return 13;
    }

    ovspatialquery_overlap_desc_t overlap              = {0};
    overlap.common.target.target.kind                  = OVSPATIALQUERY_TARGET_VIEW;
    overlap.common.target.target.handle.view           = view;
    overlap.common.target.ordinal.kind                 = OVSPATIALQUERY_ORDINAL_HEAD;
    overlap.common.precision                           = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    overlap.shape.kind                                 = OVSPATIALQUERY_OVERLAP_AABB;
    overlap.shape.data.aabb.min_corner[0]              = -1.0;
    overlap.shape.data.aabb.min_corner[1]              = -1.0;
    overlap.shape.data.aabb.min_corner[2]              = -1.0;
    overlap.shape.data.aabb.max_corner[0]              = 1.0;
    overlap.shape.data.aabb.max_corner[1]              = 1.0;
    overlap.shape.data.aabb.max_corner[2]              = 1.0;
    ovspatialquery_overlap_any_result_t overlap_result = {0};
    if (!succeeded(ovspatialquery_overlap_any_sync(instance, &overlap, &overlap_result)) ||
        overlap_result.has_hit != 0u || overlap_result.source_ordinal != 1)
    {
        return 7;
    }

    floor.ordinal = 2u;
    if (!wait_stage_op(stage, stage->data_plane.vtable->advance_write_floor(stage->data_plane.context, &floor)))
    {
        return 8;
    }
    /* An explicit ON_HOST execution is the host call itself (ADR-0047). */
    const ovspatialquery_dynamic_execution_t on_host = {{kDLCPU, 0}, {0u, 0u, 0u}, OVSPATIALQUERY_PREPARE_ON_HOST};
    operation = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, 1, 2, &on_host);
    if (!finish_update(instance, operation, 2))
    {
        return 9;
    }

    overlap.common.target.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    overlap.common.target.ordinal.source_ordinal = 1;
    overlap_result.source_ordinal                = -1;
    if (!succeeded(ovspatialquery_overlap_any_sync(instance, &overlap, &overlap_result)) ||
        overlap_result.has_hit != 0u || overlap_result.source_ordinal != 1)
    {
        return 14;
    }

    {
        const DLDevice cpu               = {kDLCPU, 0};
        uint64_t       before_plan_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &before_plan_bytes)))
        {
            return 15;
        }
        {
            ovspatialquery_raycast_plan_desc_t invalid_ray  = {0};
            invalid_ray.target.kind                         = OVSPATIALQUERY_TARGET_VIEW;
            invalid_ray.target.handle.view                  = view;
            invalid_ray.hit_mode                            = OVSPATIALQUERY_HIT_MODE_ANY;
            invalid_ray.precision                           = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
            invalid_ray.device.device_type                  = kDLCUDA;
            invalid_ray.maximum_query_count                 = 1u;
            invalid_ray.maximum_hits_per_query              = 1u;
            invalid_ray.hit_capacity_per_slot               = 1u;
            invalid_ray.result_slot_count                   = 1u;
            invalid_ray.result_fields                       = UINT32_C(0x80000000);
            ovspatialquery_batch_plan_handle_t invalid_plan = {UINT64_C(99)};
            if (ovspatialquery_create_raycast_plan(instance, &invalid_ray, &invalid_plan).status !=
                    OVSPATIALQUERY_API_INVALID_ARGUMENT ||
                ovspatialquery_batch_plan_handle_is_valid(invalid_plan))
            {
                return 37;
            }

            ovspatialquery_sphere_sweep_plan_desc_t invalid_sweep = {0};
            invalid_sweep.target                                  = invalid_ray.target;
            invalid_sweep.precision                               = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
            invalid_sweep.radius                                  = 1.0;
            invalid_sweep.device.device_type                      = kDLCUDA;
            invalid_sweep.maximum_query_count                     = 1u;
            invalid_sweep.result_slot_count                       = 1u;
            invalid_sweep.result_fields                           = UINT32_C(0x80000000);
            invalid_plan                                          = (ovspatialquery_batch_plan_handle_t){UINT64_C(99)};
            if (ovspatialquery_create_sphere_sweep_plan(instance, &invalid_sweep, &invalid_plan).status !=
                    OVSPATIALQUERY_API_INVALID_ARGUMENT ||
                ovspatialquery_batch_plan_handle_is_valid(invalid_plan))
            {
                return 38;
            }

            ovspatialquery_overlap_plan_desc_t invalid_overlap = {0};
            invalid_overlap.target                             = invalid_ray.target;
            invalid_overlap.hit_mode                           = (ovspatialquery_hit_mode_t)99;
            invalid_overlap.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
            invalid_overlap.shape.kind                         = OVSPATIALQUERY_OVERLAP_SPHERE;
            invalid_overlap.shape.data.sphere.radius           = 1.0;
            invalid_overlap.device.device_type                 = kDLCUDA;
            invalid_overlap.maximum_query_count                = 1u;
            invalid_overlap.maximum_hits_per_query             = 1u;
            invalid_overlap.hit_capacity_per_slot              = 1u;
            invalid_overlap.result_slot_count                  = 1u;
            invalid_plan                                       = (ovspatialquery_batch_plan_handle_t){UINT64_C(99)};
            if (ovspatialquery_create_overlap_plan(instance, &invalid_overlap, &invalid_plan).status !=
                    OVSPATIALQUERY_API_INVALID_ARGUMENT ||
                ovspatialquery_batch_plan_handle_is_valid(invalid_plan))
            {
                return 39;
            }

            ovspatialquery_pair_reporter_desc_t invalid_reporter          = {0};
            invalid_reporter.first                                        = invalid_ray.target;
            invalid_reporter.second                                       = invalid_ray.target;
            invalid_reporter.active_pair_capacity                         = 1u;
            invalid_reporter.delta_pair_capacity                          = 1u;
            invalid_reporter.result_device.device_type                    = kDLCUDA;
            ovspatialquery_pair_reporter_handle_t invalid_reporter_handle = {UINT64_C(99)};
            if (ovspatialquery_create_pair_reporter(instance, &invalid_reporter, &invalid_reporter_handle).status !=
                    OVSPATIALQUERY_API_INVALID_ARGUMENT ||
                ovspatialquery_pair_reporter_handle_is_valid(invalid_reporter_handle))
            {
                return 40;
            }
        }
        ovspatialquery_overlap_plan_desc_t plan_desc     = {0};
        plan_desc.target.kind                            = OVSPATIALQUERY_TARGET_VIEW;
        plan_desc.target.handle.view                     = view;
        plan_desc.hit_mode                               = OVSPATIALQUERY_HIT_MODE_ANY;
        plan_desc.precision                              = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
        plan_desc.shape.kind                             = OVSPATIALQUERY_OVERLAP_AABB;
        plan_desc.shape.data.aabb.half_extents[0]        = 1.0;
        plan_desc.shape.data.aabb.half_extents[1]        = 1.0;
        plan_desc.shape.data.aabb.half_extents[2]        = 1.0;
        plan_desc.device.device_type                     = kDLCPU;
        plan_desc.maximum_query_count                    = 1u;
        plan_desc.maximum_hits_per_query                 = 1u;
        plan_desc.result_slot_count                      = 1u;
        plan_desc.hit_capacity_per_slot                  = UINT64_C(1) << 20u;
        ovspatialquery_batch_plan_handle_t rejected_plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        const ovspatialquery_result_t      rejected_plan_result =
            ovspatialquery_create_overlap_plan(instance, &plan_desc, &rejected_plan);
        uint64_t after_rejected_plan_bytes = UINT64_MAX;
        if (rejected_plan_result.status != OVSPATIALQUERY_API_HOST_BUDGET ||
            ovspatialquery_batch_plan_handle_is_valid(rejected_plan) ||
            !succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_rejected_plan_bytes)) ||
            after_rejected_plan_bytes != before_plan_bytes)
        {
            return 33;
        }
        plan_desc.hit_capacity_per_slot         = 1u;
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        if (!succeeded(ovspatialquery_create_overlap_plan(instance, &plan_desc, &plan)))
        {
            return 16;
        }
        uint64_t plan_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &plan_bytes)) ||
            plan_bytes <= before_plan_bytes)
        {
            return 17;
        }

        double                       translations[1][3]     = {{0.0, 0.0, 0.0}};
        int64_t                      translation_shape[2]   = {1, 3};
        int64_t                      translation_strides[2] = {3, 1};
        ovspatialquery_input_array_t translation_array =
            copied_array(translations, kDLFloat, 64u, 2, translation_shape, translation_strides);
        ovspatialquery_overlap_batch_t batch             = {0};
        batch.ordinal.kind                               = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
        batch.ordinal.source_ordinal                     = 1;
        batch.query_count                                = 1u;
        batch.translations                               = &translation_array;
        ovspatialquery_overlap_batch_reservation_t early = {0};
        ovspatialquery_enqueue_result_t launch = ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &early);
        if (launch.status != OVSPATIALQUERY_API_SUCCESS || !ovspatialquery_op_id_is_valid(launch.op_id) ||
            early.common.query_count != 1u || early.common.source_ordinal != 1 ||
            ovspatialquery_destroy_batch_plan(instance, plan).status != OVSPATIALQUERY_API_BUSY)
        {
            return 18;
        }
        uint64_t operation_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &operation_bytes)) ||
            operation_bytes <= plan_bytes)
        {
            return 19;
        }
        ovspatialquery_op_wait_result_t       waited  = {0};
        ovspatialquery_overlap_batch_result_t fetched = {0};
        if (!succeeded(ovspatialquery_wait_op(instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
            waited.state != OVSPATIALQUERY_OP_COMPLETED || early.columns.hit_ranges.query_status.data == NULL ||
            early.columns.hit_ranges.query_status.ndim != 1 || early.columns.hit_ranges.query_status.shape == NULL ||
            early.columns.hit_ranges.query_status.shape[0] != 1 || early.columns.object_ids.data == NULL ||
            early.columns.object_ids.ndim != 1 || early.columns.object_ids.shape == NULL ||
            early.columns.object_ids.shape[0] != 1 ||
            !succeeded(ovspatialquery_fetch_overlap_batch(instance, launch.op_id, &fetched)) ||
            fetched.common.query_count != 1u || fetched.common.source_ordinal != 1 ||
            fetched.columns.hit_ranges.query_status.ndim != 1 ||
            fetched.columns.hit_ranges.query_status.shape == NULL ||
            fetched.columns.hit_ranges.query_status.shape[0] != 1 || fetched.columns.object_ids.ndim != 1 ||
            fetched.columns.object_ids.shape == NULL || fetched.columns.object_ids.shape[0] != 0 ||
            !succeeded(ovspatialquery_destroy_op(instance, &launch.op_id)) ||
            ovspatialquery_op_id_is_valid(launch.op_id))
        {
            return 20;
        }
        uint64_t after_operation_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_operation_bytes)) ||
            after_operation_bytes != plan_bytes || !succeeded(ovspatialquery_destroy_batch_plan(instance, plan)))
        {
            return 21;
        }
        uint64_t after_plan_bytes = UINT64_MAX;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_plan_bytes)) ||
            after_plan_bytes != before_plan_bytes)
        {
            return 22;
        }

        plan_desc.hit_capacity_per_slot                     = UINT64_C(1) << 20u;
        ovspatialquery_batch_plan_handle_t constrained_plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        const ovspatialquery_result_t      constrained_plan_result =
            ovspatialquery_create_overlap_plan(instance, &plan_desc, &constrained_plan);
        uint64_t after_constrained_plan_bytes = UINT64_MAX;
        if (constrained_plan_result.status != OVSPATIALQUERY_API_HOST_BUDGET ||
            ovspatialquery_batch_plan_handle_is_valid(constrained_plan) ||
            !succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_constrained_plan_bytes)) ||
            after_constrained_plan_bytes != before_plan_bytes)
        {
            return 36;
        }
    }

    {
        const DLDevice cpu                   = {kDLCPU, 0};
        uint64_t       before_reporter_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &before_reporter_bytes)))
        {
            return 23;
        }
        ovspatialquery_pair_reporter_desc_t reporter_desc       = {0};
        reporter_desc.first.kind                                = OVSPATIALQUERY_TARGET_POPULATION;
        reporter_desc.first.handle.population                   = population;
        reporter_desc.second                                    = reporter_desc.first;
        reporter_desc.delta_pair_capacity                       = 1u;
        reporter_desc.result_slot_count                         = 1u;
        reporter_desc.result_device.device_type                 = kDLCPU;
        reporter_desc.active_pair_capacity                      = UINT64_C(1) << 21u;
        ovspatialquery_pair_reporter_handle_t rejected_reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
        const ovspatialquery_result_t         rejected_reporter_result =
            ovspatialquery_create_pair_reporter(instance, &reporter_desc, &rejected_reporter);
        uint64_t after_rejected_reporter_bytes = UINT64_MAX;
        if (rejected_reporter_result.status != OVSPATIALQUERY_API_HOST_BUDGET ||
            ovspatialquery_pair_reporter_handle_is_valid(rejected_reporter) ||
            !succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_rejected_reporter_bytes)) ||
            after_rejected_reporter_bytes != before_reporter_bytes)
        {
            return 34;
        }
        reporter_desc.active_pair_capacity             = 1u;
        ovspatialquery_pair_reporter_handle_t reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
        if (!succeeded(ovspatialquery_create_pair_reporter(instance, &reporter_desc, &reporter)))
        {
            return 24;
        }
        uint64_t reporter_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &reporter_bytes)) ||
            reporter_bytes <= before_reporter_bytes)
        {
            return 25;
        }
        ovspatialquery_pair_reporter_advance_t advance   = {0};
        advance.first_ordinal.kind                       = OVSPATIALQUERY_ORDINAL_HEAD;
        advance.second_ordinal.kind                      = OVSPATIALQUERY_ORDINAL_HEAD;
        ovspatialquery_pair_reporter_reservation_t early = {0};
        ovspatialquery_enqueue_result_t            launch =
            ovspatialquery_enqueue_pair_reporter_advance(instance, reporter, &advance, &early);
        if (launch.status != OVSPATIALQUERY_API_SUCCESS || !ovspatialquery_op_id_is_valid(launch.op_id) ||
            ovspatialquery_destroy_pair_reporter(instance, reporter).status != OVSPATIALQUERY_API_BUSY)
        {
            return 26;
        }
        uint64_t reporter_operation_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &reporter_operation_bytes)) ||
            reporter_operation_bytes <= reporter_bytes)
        {
            return 27;
        }
        ovspatialquery_op_wait_result_t       waited  = {0};
        ovspatialquery_pair_reporter_result_t fetched = {0};
        if (!succeeded(ovspatialquery_wait_op(instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
            waited.state != OVSPATIALQUERY_OP_COMPLETED ||
            !succeeded(ovspatialquery_fetch_pair_reporter(instance, launch.op_id, &fetched)) ||
            fetched.outcome != OVSPATIALQUERY_PAIR_REPORTER_ADVANCED || fetched.first_source_ordinal != 2 ||
            fetched.second_source_ordinal != 2 || fetched.entered_pairs.ndim != 2 ||
            fetched.entered_pairs.shape == NULL || fetched.entered_pairs.shape[0] != 0 ||
            fetched.entered_pairs.shape[1] != 2 || fetched.exited_pairs.ndim != 2 ||
            fetched.exited_pairs.shape == NULL || fetched.exited_pairs.shape[0] != 0 ||
            fetched.exited_pairs.shape[1] != 2 || !succeeded(ovspatialquery_destroy_op(instance, &launch.op_id)))
        {
            return 28;
        }
        uint64_t after_reporter_operation_bytes = 0u;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_reporter_operation_bytes)) ||
            after_reporter_operation_bytes != reporter_bytes ||
            !succeeded(ovspatialquery_destroy_pair_reporter(instance, reporter)))
        {
            return 29;
        }
        uint64_t after_reporter_bytes = UINT64_MAX;
        if (!succeeded(ovspatialquery_get_instance_memory_usage(instance, cpu, &after_reporter_bytes)) ||
            after_reporter_bytes != before_reporter_bytes)
        {
            return 30;
        }
    }

    if (!succeeded(ovspatialquery_destroy_view(instance, view)) ||
        !succeeded(ovspatialquery_destroy_population(instance, population)) ||
        !succeeded(ovspatialquery_destroy_instance(instance)) || ovstage_destroy_instance(stage) != OVSTAGE_OK)
    {
        return 32;
    }
    return 0;
}
