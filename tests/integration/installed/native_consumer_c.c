/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <dlpack/dlpack.h>
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <stddef.h>
#include <stdint.h>

_Static_assert(sizeof(ovspatialquery_op_id_t) == 8u, "operation ID ABI");
_Static_assert(sizeof(ovspatialquery_instance_handle_t) == 8u, "instance handle");
_Static_assert(sizeof(ovspatialquery_population_handle_t) == 8u, "population handle");
_Static_assert(sizeof(ovspatialquery_view_handle_t) == 8u, "view handle");
_Static_assert(sizeof(ovspatialquery_batch_plan_handle_t) == 8u, "plan handle");
_Static_assert(sizeof(ovspatialquery_pair_reporter_handle_t) == 8u, "reporter handle");
_Static_assert(_Generic(((ovspatialquery_raycast_batch_columns_t*)0)->contact.candidate.bounds_positions, DLTensor : 1,
                        default : 0),
               "raycast candidate positions are a by-value DLTensor");
_Static_assert(_Generic(((ovspatialquery_raycast_batch_columns_t*)0)->contact.exact.features,
                        ovspatialquery_exact_feature_columns_t : 1, default : 0),
               "raycast exact features are one atomic column group");
_Static_assert(_Generic(((ovspatialquery_sphere_sweep_batch_columns_t*)0)->contact.exact.features,
                        ovspatialquery_exact_feature_columns_t : 1, default : 0),
               "sphere sweep exact features are one atomic column group");

static ovspatialquery_input_array_t copied_input(void* data, int32_t ndim, uint8_t code, uint8_t bits, int64_t* shape,
                                                 int64_t* strides)
{
    ovspatialquery_input_array_t input = {0};
    input.tensor.data                  = data;
    input.tensor.device                = (DLDevice){kDLCPU, 0};
    input.tensor.ndim                  = ndim;
    input.tensor.dtype                 = (DLDataType){code, bits, 1u};
    input.tensor.shape                 = shape;
    input.tensor.strides               = strides;
    input.ownership                    = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    return input;
}

static int wait_for_success(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    ovspatialquery_op_wait_result_t wait = {0};
    return ovspatialquery_op_id_is_valid(operation) &&
           ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
               OVSPATIALQUERY_API_SUCCESS &&
           wait.state == OVSPATIALQUERY_OP_COMPLETED;
}

int main(void)
{
    ovspatialquery_enqueue_result_t (*sphere_registration)(ovspatialquery_instance_handle_t,
                                                           const ovspatialquery_sphere_definition_table_t*) =
        &ovspatialquery_enqueue_sphere_registration;
    (void)sphere_registration;
    uint32_t major = 0u;
    uint32_t minor = 0u;
    uint32_t patch = 0u;
    if (ovspatialquery_get_version(&major, &minor, &patch).status != OVSPATIALQUERY_API_SUCCESS)
    {
        return 1;
    }
    ovspatialquery_instance_desc_t desc       = {0};
    desc.operation_capacity                   = 4u;
    desc.max_cpu_concurrency                  = 2u;
    desc.host_byte_budget                     = UINT64_C(4) << 20;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    if (ovspatialquery_create_instance(&desc, &instance).status != OVSPATIALQUERY_API_SUCCESS ||
        !ovspatialquery_instance_handle_is_valid(instance))
    {
        return 2;
    }
    uint64_t       bytes = UINT64_MAX;
    const DLDevice cpu   = {kDLCPU, 0};
    if (ovspatialquery_get_instance_memory_usage(instance, cpu, &bytes).status != OVSPATIALQUERY_API_SUCCESS ||
        bytes == 0u || bytes > desc.host_byte_budget)
    {
        return 3;
    }
    if (ovspatialquery_get_last_op_error(instance, OVSPATIALQUERY_INVALID_OP).length != 0u)
    {
        return 4;
    }

    int64_t                      scalar_shape[1]  = {1};
    int64_t                      scalar_stride[1] = {1};
    uint64_t                     definition_value = 7u;
    double                       radius           = 1.0;
    ovspatialquery_input_array_t definition_values =
        copied_input(&definition_value, 1, kDLUInt, 64u, scalar_shape, scalar_stride);
    ovspatialquery_input_array_t radii = copied_input(&radius, 1, kDLFloat, 64u, scalar_shape, scalar_stride);
    const ovspatialquery_sphere_definition_table_t sphere = {&definition_values, &radii};
    ovspatialquery_enqueue_result_t admission = ovspatialquery_enqueue_sphere_registration(instance, &sphere);
    if (admission.status != OVSPATIALQUERY_API_SUCCESS || !wait_for_success(instance, admission.op_id))
    {
        return 5;
    }
    if (ovspatialquery_destroy_op(instance, &admission.op_id).status != OVSPATIALQUERY_API_SUCCESS ||
        ovspatialquery_op_id_is_valid(admission.op_id))
    {
        return 6;
    }

    const ovspatialquery_population_desc_t population_desc = {1u, 1u};
    ovspatialquery_population_handle_t     population      = OVSPATIALQUERY_INVALID_POPULATION;
    if (ovspatialquery_create_explicit_population(instance, &population_desc, &population).status !=
            OVSPATIALQUERY_API_SUCCESS ||
        !ovspatialquery_population_handle_is_valid(population))
    {
        return 7;
    }
    const ovspatialquery_population_handle_t same_population = population;
    if (!ovspatialquery_population_handle_equal(population, same_population) ||
        ovspatialquery_population_handle_equal(population, OVSPATIALQUERY_INVALID_POPULATION))
    {
        return 13;
    }

    uint64_t                     object_value           = 11u;
    int32_t                      geometry_type          = OVSPATIALQUERY_GEOMETRY_SPHERE;
    double                       translation[3]         = {0.0, 0.0, 0.0};
    float                        rotation[4]            = {0.0f, 0.0f, 0.0f, 1.0f};
    int64_t                      translation_shape[2]   = {1, 3};
    int64_t                      translation_strides[2] = {3, 1};
    int64_t                      rotation_shape[2]      = {1, 4};
    int64_t                      rotation_strides[2]    = {4, 1};
    ovspatialquery_input_array_t object_values =
        copied_input(&object_value, 1, kDLUInt, 64u, scalar_shape, scalar_stride);
    ovspatialquery_input_array_t geometry_types =
        copied_input(&geometry_type, 1, kDLInt, 32u, scalar_shape, scalar_stride);
    ovspatialquery_input_array_t geometry_values =
        copied_input(&definition_value, 1, kDLUInt, 64u, scalar_shape, scalar_stride);
    ovspatialquery_input_array_t translations =
        copied_input(translation, 2, kDLFloat, 64u, translation_shape, translation_strides);
    ovspatialquery_input_array_t rotations = copied_input(rotation, 2, kDLFloat, 32u, rotation_shape, rotation_strides);
    const ovspatialquery_explicit_population_set_t set = {
        0, {&object_values, &geometry_types, &geometry_values, &translations, &rotations, NULL, NULL, NULL}, NULL, 0u};
    admission = ovspatialquery_enqueue_explicit_population_set(instance, population, &set);
    if (admission.status != OVSPATIALQUERY_API_SUCCESS || !wait_for_success(instance, admission.op_id))
    {
        return 8;
    }
    ovspatialquery_population_update_result_t publication = {0};
    if (ovspatialquery_fetch_population_update(instance, admission.op_id, &publication).status !=
            OVSPATIALQUERY_API_SUCCESS ||
        publication.published_ordinal != 0 || publication.object_count != 1u)
    {
        return 9;
    }
    if (ovspatialquery_destroy_op(instance, &admission.op_id).status != OVSPATIALQUERY_API_SUCCESS)
    {
        return 10;
    }

    ovspatialquery_raycast_desc_t ray          = {0};
    ray.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    ray.common.target.target.handle.population = population;
    ray.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
    ray.common.precision                       = OVSPATIALQUERY_QUERY_EXACT;
    ray.common.query_bits.words[0]             = UINT64_MAX;
    ray.common.query_bits.words[1]             = UINT64_MAX;
    ray.ray.origin[0]                          = -3.0;
    ray.ray.unit_direction[0]                  = 1.0f;
    ray.ray.max_distance                       = 10.0;
    ovspatialquery_raycast_single_result_t hit = {0};
    if (ovspatialquery_raycast_closest_sync(instance, &ray, &hit).status != OVSPATIALQUERY_API_SUCCESS ||
        hit.has_hit == 0u || hit.hit.data.exact.identity.object_id.value != object_value ||
        hit.hit.data.exact.distance != 2.0)
    {
        return 11;
    }
    if (ovspatialquery_destroy_population(instance, population).status != OVSPATIALQUERY_API_SUCCESS ||
        ovspatialquery_destroy_geometry_definition(
            instance, (ovspatialquery_geometry_ref_t){OVSPATIALQUERY_GEOMETRY_SPHERE, definition_value})
                .status != OVSPATIALQUERY_API_SUCCESS)
    {
        return 12;
    }
    return ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS ? 0 : 13;
}
