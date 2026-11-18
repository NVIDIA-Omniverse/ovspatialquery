/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All
 * rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Publish one box and cast a ray at it through the C API. */

#include <dlpack/dlpack.h>
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <stdint.h>
#include <stdio.h>

/* The shape and packed strides of a one-row column. */
typedef struct
{
    int32_t ndim;
    int64_t shape[2];
    int64_t strides[2];
} layout_t;

static layout_t SCALAR = {1, {1}, {1}};
static layout_t VEC3 = {2, {1, 3}, {3, 1}};
static layout_t VEC4 = {2, {1, 4}, {4, 1}};

/* Every input is a DLPack tensor on the CPU, copied when the call is admitted. */
static ovspatialquery_input_array_t input(void* data, DLDataType dtype, layout_t* layout)
{
    ovspatialquery_input_array_t array = {0};
    array.tensor.data = data;
    array.tensor.device = (DLDevice){kDLCPU, 0};
    array.tensor.ndim = layout->ndim;
    array.tensor.dtype = dtype;
    array.tensor.shape = layout->shape;
    array.tensor.strides = layout->strides;
    array.ownership = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return array;
}

static const DLDataType U64 = {kDLUInt, 64, 1};
static const DLDataType I32 = {kDLInt, 32, 1};
static const DLDataType F64 = {kDLFloat, 64, 1};
static const DLDataType F32 = {kDLFloat, 32, 1};

/* Print why the last call on this thread failed. */
static int failed(void)
{
    ovspatialquery_string_t error = ovspatialquery_get_last_error();
    fprintf(stderr, "failed: %.*s\n", (int)error.length, error.ptr);
    return 0;
}

static int ok(ovspatialquery_result_t result)
{
    return result.status == OVSPATIALQUERY_API_SUCCESS || failed();
}

/* Wait for asynchronous work, then release its operation ID. */
static int finish(ovspatialquery_instance_handle_t instance,
                  ovspatialquery_enqueue_result_t admitted)
{
    if (admitted.status != OVSPATIALQUERY_API_SUCCESS)
        return failed();
    ovspatialquery_op_wait_result_t waited = {0};
    if (!ok(ovspatialquery_wait_op(instance, admitted.op_id,
                                   OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)))
        return 0;
    if (waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        ovspatialquery_string_t error =
            ovspatialquery_get_last_op_error(instance, admitted.op_id);
        fprintf(stderr, "operation failed: %.*s\n", (int)error.length, error.ptr);
        ovspatialquery_destroy_op(instance, &admitted.op_id);
        return 0;
    }
    return ok(ovspatialquery_destroy_op(instance, &admitted.op_id));
}

int main(void)
{
    /* 1. One instance holds the budgets for everything created through it. */
    ovspatialquery_instance_desc_t desc = {
        .operation_capacity = 8,
        .max_cpu_concurrency = 2,
        .host_byte_budget = UINT64_C(64) << 20,
    };
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (!ok(ovspatialquery_create_instance(&desc, &instance)))
        return 1;

    /* 2. Register definition 1: a box with half extents 0.5. */
    uint64_t definition = 1;
    double half_extent[3] = {0.5, 0.5, 0.5};

    ovspatialquery_input_array_t ids = input(&definition, U64, &SCALAR);
    ovspatialquery_input_array_t extents = input(half_extent, F64, &VEC3);
    const ovspatialquery_box_definition_table_t boxes = {&ids, &extents};
    if (!finish(instance, ovspatialquery_enqueue_box_registration(instance, &boxes)))
        return 1;

    /* 3. Place object 10 at x = 4 and publish it as ordinal 1. */
    const ovspatialquery_population_desc_t population_desc = {
        .retained_ordinal_capacity = 2,
    };
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    if (!ok(ovspatialquery_create_explicit_population(instance, &population_desc,
                                                      &population)))
        return 1;

    uint64_t object = 10;
    int32_t type = OVSPATIALQUERY_GEOMETRY_BOX;
    double translation[3] = {4.0, 0.0, 0.0};
    float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f}; /* x, y, z, w */

    ovspatialquery_input_array_t objects = input(&object, U64, &SCALAR);
    ovspatialquery_input_array_t types = input(&type, I32, &SCALAR);
    ovspatialquery_input_array_t geometries = input(&definition, U64, &SCALAR);
    ovspatialquery_input_array_t translations = input(translation, F64, &VEC3);
    ovspatialquery_input_array_t rotations = input(rotation, F32, &VEC4);
    const ovspatialquery_explicit_population_set_t set = {
        .target_ordinal = 1,
        .instances = {&objects, &types, &geometries, &translations, &rotations},
    };
    if (!finish(instance, ovspatialquery_enqueue_explicit_population_set(
                              instance, population, &set)))
        return 1;

    /* 4. Cast a ray along +x at the head, asking for the exact surface. */
    ovspatialquery_raycast_desc_t ray = {
        .common.target.target.kind = OVSPATIALQUERY_TARGET_POPULATION,
        .common.target.target.handle.population = population,
        .common.target.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD,
        .common.precision = OVSPATIALQUERY_QUERY_EXACT,
        .ray.unit_direction = {1.0f, 0.0f, 0.0f},
        .ray.max_distance = 20.0,
    };
    ovspatialquery_raycast_single_result_t found = {0};
    if (!ok(ovspatialquery_raycast_closest_sync(instance, &ray, &found)))
        return 1;

    if (found.has_hit && found.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE)
    {
        const ovspatialquery_raycast_exact_hit_t* hit = &found.hit.data.exact;
        printf("ray hit %llu at distance %.3f, ordinal %lld\n",
               (unsigned long long)hit->identity.object_id.value, hit->distance,
               (long long)found.source_ordinal);
    }

    /* 5. Destroy in reverse order; each call returns BUSY while something still
     * uses the object. */
    const ovspatialquery_geometry_ref_t box = {OVSPATIALQUERY_GEOMETRY_BOX, 1};
    ovspatialquery_destroy_population(instance, population);
    ovspatialquery_destroy_geometry_definition(instance, box);
    ovspatialquery_destroy_instance(instance);
    return 0;
}
