/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// One installed-library capability showcase. Query truth is
// written below; the Python companion only renders the resulting JSON.

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <ovstage/ovstage.h>

#include "example_ovstage.h"

#include "cuda_driver.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

constexpr char     kStageScene[] = R"usda(#usda 1.0
def Xform "World"
{
    def Sphere "StageSphere"
    {
        double radius = 1
        double3 xformOp:translate = (2, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
})usda";
constexpr char     kStagePath[]  = "/World/StageSphere";
constexpr uint64_t kConvexValue  = 91u;
constexpr uint64_t kConvexObject = 404u;
// Object 202's sphere, published beside object 101 so the two form the explicit pair.
constexpr uint64_t              kPairedSphereValue = 8u;
constexpr std::array<float, 27> kAuthoredConvexPoints{
    -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,  -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f,  1.0f,  -1.0f, 1.0f, 1.0f,  1.0f,  1.0f, -1.0f, 1.0f,  1.0f,  0.0f, 0.0f,  0.0f,
};

[[noreturn]] void fail(const char* message) { throw std::runtime_error(message); }

void require(bool value, const char* message)
{
    if (!value)
    {
        fail(message);
    }
}

bool ok(ovspatialquery_result_t result) { return result.status == OVSPATIALQUERY_API_SUCCESS; }

bool finish(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t admission,
            ovspatialquery_population_update_result_t* publication = nullptr)
{
    if (admission.status != OVSPATIALQUERY_API_SUCCESS || admission.op_id == OVSPATIALQUERY_INVALID_OP)
    {
        return false;
    }
    ovspatialquery_op_wait_result_t waited{};
    if (!ok(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        return false;
    }
    if (publication != nullptr && !ok(ovspatialquery_fetch_population_update(instance, admission.op_id, publication)))
    {
        return false;
    }
    return ok(ovspatialquery_destroy_op(instance, &admission.op_id)) && admission.op_id == OVSPATIALQUERY_INVALID_OP;
}

bool waitStage(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    if (operation.status != OVSTAGE_OK || operation.op_index == OVSTAGE_INVALID_OP_ID)
    {
        return false;
    }
    ovstage_op_wait_result_t result{};
    return example_ovstage::table(stage)->wait_op(example_ovstage::context(stage), operation.op_index,
                                                  OVSTAGE_TIMEOUT_INFINITE, &result) == OVSTAGE_OK &&
           result.error_op_id_count == 0u &&
           example_ovstage::table(stage)->release_op(example_ovstage::context(stage), operation.op_index) == OVSTAGE_OK;
}

bool waitOpen(ovstage_instance_t* stage, ovstage_population_enqueue_result_t operation)
{
    if (operation.status != OVSTAGE_OK || operation.op_index == OVSTAGE_POPULATION_INVALID_OP_ID)
    {
        return false;
    }
    ovstage_population_op_wait_result_t result{};
    return ovstage_population_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result) == OVSTAGE_OK &&
           result.error_op_id_count == 0u;
}

struct Input
{
    template <class T>
    Input(T* data, int64_t rows, int64_t columns, uint8_t code,
          ovspatialquery_input_ownership_t ownership = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION)
    {
        shape                = {rows, columns};
        strides              = {columns, 1};
        value.tensor.data    = data;
        value.tensor.device  = {kDLCPU, 0};
        value.tensor.ndim    = columns == 1 ? 1 : 2;
        value.tensor.dtype   = {code, static_cast<uint8_t>(sizeof(T) * 8u), 1u};
        value.tensor.shape   = shape.data();
        value.tensor.strides = strides.data();
        value.ownership      = ownership;
    }

    std::array<int64_t, 2>       shape{};
    std::array<int64_t, 2>       strides{};
    ovspatialquery_input_array_t value{};
};

struct CudaInput
{
    template <class T>
    CudaInput(CudaDriver& cuda, CudaDriver::Stream stream, const T* data, int64_t rows, int64_t columns, uint8_t code)
    {
        const size_t                    byte_count = static_cast<size_t>(rows * columns) * sizeof(T);
        const CudaDriver::DevicePointer pointer    = cuda.allocate(byte_count);
        cuda.uploadAsync(pointer, data, byte_count, stream);
        shape                = {rows, columns};
        strides              = {columns, 1};
        value.tensor.data    = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
        value.tensor.device  = {kDLCUDA, 0};
        value.tensor.ndim    = columns == 1 ? 1 : 2;
        value.tensor.dtype   = {code, static_cast<uint8_t>(sizeof(T) * 8u), 1u};
        value.tensor.shape   = shape.data();
        value.tensor.strides = strides.data();
        value.ownership      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    }

    std::array<int64_t, 2>       shape{};
    std::array<int64_t, 2>       strides{};
    ovspatialquery_input_array_t value{};
};

CudaDriver::DevicePointer devicePointer(const DLTensor& tensor)
{
    return static_cast<CudaDriver::DevicePointer>(reinterpret_cast<uintptr_t>(tensor.data)) + tensor.byte_offset;
}

bool destroyPlanWhenRetired(ovspatialquery_instance_handle_t instance, ovspatialquery_batch_plan_handle_t plan)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do
    {
        const ovspatialquery_api_status_t status = ovspatialquery_destroy_batch_plan(instance, plan).status;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            return true;
        }
        if (status != OVSPATIALQUERY_API_BUSY)
        {
            return false;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

struct CudaFamilyEvidence
{
    uint32_t                row_status       = UINT32_MAX;
    uint64_t                stored_hit_count = UINT64_MAX;
    uint64_t                hit_offset       = UINT64_MAX;
    uint64_t                hit_count        = UINT64_MAX;
    uint8_t                 truncated        = UINT8_MAX;
    uint64_t                object_id        = 0u;
    int32_t                 geometry_type    = 0;
    uint64_t                geometry_value   = 0u;
    uint32_t                feature_kind     = OVSPATIALQUERY_GEOMETRY_FEATURE_NONE;
    uint64_t                source_face      = 0u;
    std::array<uint32_t, 3> source_corners{};
    std::array<float, 3>    barycentrics{};
    std::array<double, 3>   position{};
    std::array<float, 3>    normal{};
    std::array<double, 3>   query_position{};
    double                  distance = 0.0;
};

struct CudaContactTensors
{
    const DLTensor* position       = nullptr;
    const DLTensor* normal         = nullptr;
    const DLTensor* query_position = nullptr;
    const DLTensor* distance       = nullptr;
};

template <typename Result>
CudaFamilyEvidence consumeCudaResult(CudaDriver& cuda, CudaDriver::Stream consumer,
                                     ovspatialquery_instance_handle_t instance,
                                     ovspatialquery_enqueue_result_t operation, const Result& result,
                                     const ovspatialquery_exact_feature_columns_t* features,
                                     const CudaContactTensors*                     contacts)
{
    require(result.common.execution_device.device_type == kDLCUDA && result.common.execution_device.device_id == 0 &&
                result.common.result_ready.value != 0u,
            "CUDA result did not expose one real device-ready event");
    cuda.wait(consumer, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    CudaFamilyEvidence evidence{};
    cuda.downloadAsync(&evidence.row_status, devicePointer(result.columns.hit_ranges.query_status),
                       sizeof(evidence.row_status), consumer);
    cuda.downloadAsync(&evidence.stored_hit_count, devicePointer(result.columns.hit_ranges.stored_hit_count),
                       sizeof(evidence.stored_hit_count), consumer);
    cuda.downloadAsync(&evidence.hit_offset, devicePointer(result.columns.hit_ranges.hit_offsets),
                       sizeof(evidence.hit_offset), consumer);
    cuda.downloadAsync(&evidence.hit_count, devicePointer(result.columns.hit_ranges.hit_counts),
                       sizeof(evidence.hit_count), consumer);
    cuda.downloadAsync(&evidence.truncated, devicePointer(result.columns.hit_ranges.truncated),
                       sizeof(evidence.truncated), consumer);
    cuda.downloadAsync(&evidence.object_id, devicePointer(result.columns.object_ids), sizeof(evidence.object_id),
                       consumer);
    cuda.downloadAsync(&evidence.geometry_type, devicePointer(result.columns.geometry.types),
                       sizeof(evidence.geometry_type), consumer);
    cuda.downloadAsync(&evidence.geometry_value, devicePointer(result.columns.geometry.values),
                       sizeof(evidence.geometry_value), consumer);
    if (contacts != nullptr)
    {
        cuda.downloadAsync(evidence.position.data(), devicePointer(*contacts->position), sizeof(evidence.position),
                           consumer);
        cuda.downloadAsync(evidence.normal.data(), devicePointer(*contacts->normal), sizeof(evidence.normal), consumer);
        cuda.downloadAsync(&evidence.distance, devicePointer(*contacts->distance), sizeof(evidence.distance), consumer);
        if (contacts->query_position != nullptr)
        {
            cuda.downloadAsync(evidence.query_position.data(), devicePointer(*contacts->query_position),
                               sizeof(evidence.query_position), consumer);
        }
    }
    if (features != nullptr)
    {
        cuda.downloadAsync(&evidence.feature_kind, devicePointer(features->kinds), sizeof(evidence.feature_kind),
                           consumer);
        cuda.downloadAsync(&evidence.source_face, devicePointer(features->source_face_indices),
                           sizeof(evidence.source_face), consumer);
        cuda.downloadAsync(evidence.source_corners.data(), devicePointer(features->source_corner_indices),
                           sizeof(evidence.source_corners), consumer);
        cuda.downloadAsync(evidence.barycentrics.data(), devicePointer(features->barycentrics),
                           sizeof(evidence.barycentrics), consumer);
    }
    const CudaDriver::Event consumer_done = cuda.createEvent();
    cuda.record(consumer_done, consumer);
    cuda.synchronize(consumer_done);
    ovspatialquery_cuda_event_t   consumer_done_value{reinterpret_cast<uintptr_t>(consumer_done)};
    const ovspatialquery_result_t retirement =
        ovspatialquery_destroy_op_after_cuda(instance, &operation.op_id, &consumer_done_value);
    const bool consumer_done_transferred = ok(retirement);
    const bool consumer_done_forgotten   = consumer_done_transferred && cuda.forgetTransferredEvent(consumer_done);
    require(consumer_done_transferred && consumer_done_forgotten && operation.op_id == OVSPATIALQUERY_INVALID_OP &&
                consumer_done_value.value == 0u,
            "CUDA consumer_done retirement failed");
    require(evidence.row_status == OVSPATIALQUERY_BATCH_ROW_SUCCESS && evidence.stored_hit_count == 1u &&
                evidence.hit_offset == 0u && evidence.hit_count == 1u && evidence.truncated == 0u &&
                evidence.object_id == 303u && evidence.geometry_type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH &&
                evidence.geometry_value == 88u,
            "CUDA semantic result columns were not exact");
    if (features != nullptr)
    {
        require(evidence.feature_kind == OVSPATIALQUERY_GEOMETRY_FEATURE_TRIANGLE && evidence.source_face == 0u &&
                    evidence.source_corners == std::array<uint32_t, 3>{0u, 1u, 2u} &&
                    std::abs(evidence.barycentrics[0] - 0.25f) < 1.0e-5f &&
                    std::abs(evidence.barycentrics[1] - 0.25f) < 1.0e-5f &&
                    std::abs(evidence.barycentrics[2] - 0.5f) < 1.0e-5f,
                "CUDA triangle source feature columns were not truthful");
    }
    return evidence;
}

void publishExplicit(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                     uint64_t sphere_value, ovspatialquery_population_update_result_t* out_publication)
{
    uint64_t objects[3]         = {101u, 202u, kConvexObject};
    int32_t  geometry_types[3]  = {OVSPATIALQUERY_GEOMETRY_SPHERE, OVSPATIALQUERY_GEOMETRY_SPHERE,
                                   OVSPATIALQUERY_GEOMETRY_CONVEX};
    uint64_t geometry_values[3] = {sphere_value, kPairedSphereValue, kConvexValue};
    double   translations[9]    = {4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    float    rotations[12]      = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t bits[6]            = {1u, 0u, 2u, 0u, 4u, 0u}; // uint64[N,2], bits 0-63 first
    Input    object_input(objects, 3, 1, kDLUInt);
    Input    type_input(geometry_types, 3, 1, kDLInt);
    Input    value_input(geometry_values, 3, 1, kDLUInt);
    Input    translation_input(translations, 3, 3, kDLFloat);
    Input    rotation_input(rotations, 3, 4, kDLFloat);
    Input    bit_input(bits, 3, 2, kDLUInt);
    const ovspatialquery_explicit_population_set_t set{1,
                                                       {&object_input.value, &type_input.value, &value_input.value,
                                                        &translation_input.value, &rotation_input.value, nullptr,
                                                        &bit_input.value, nullptr},
                                                       nullptr,
                                                       0u};
    require(
        finish(instance, ovspatialquery_enqueue_explicit_population_set(instance, population, &set), out_publication),
        "explicit publication failed");
}

void registerPreparedConvex(ovspatialquery_instance_handle_t instance)
{
    uint64_t                                       value  = kConvexValue;
    uint64_t                                       offset = 0u;
    uint64_t                                       count  = kAuthoredConvexPoints.size() / 3u;
    std::array<float, 27>                          points = kAuthoredConvexPoints;
    Input                                          values(&value, 1, 1, kDLUInt);
    Input                                          offsets(&offset, 1, 1, kDLUInt);
    Input                                          counts(&count, 1, 1, kDLUInt);
    Input                                          vertices(points.data(), static_cast<int64_t>(count), 3, kDLFloat);
    const ovspatialquery_convex_definition_table_t convexes{
        &values.value, &offsets.value, &counts.value, nullptr, nullptr, &vertices.value,
        nullptr,       nullptr,        nullptr,       0.0,     0u};
    require(finish(instance, ovspatialquery_enqueue_convex_registration(instance, &convexes)),
            "convex preparation registration failed");
}

void publishCudaMesh(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t* out_population,
                     ovspatialquery_population_update_result_t* out_publication)
{
    uint64_t geometry_value = 88u;
    uint64_t zero           = 0u;
    uint64_t vertex_count   = 3u;
    uint64_t triangle_count = 1u;
    float    vertices[9]    = {
        0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
    };
    uint32_t                                              triangles[3] = {0u, 1u, 2u};
    uint8_t                                               double_sided = 1u;
    Input                                                 values(&geometry_value, 1, 1, kDLUInt);
    Input                                                 vertex_offsets(&zero, 1, 1, kDLUInt);
    Input                                                 vertex_counts(&vertex_count, 1, 1, kDLUInt);
    Input                                                 triangle_offsets(&zero, 1, 1, kDLUInt);
    Input                                                 triangle_counts(&triangle_count, 1, 1, kDLUInt);
    Input                                                 vertex_data(vertices, 3, 3, kDLFloat);
    Input                                                 triangle_data(triangles, 1, 3, kDLUInt);
    Input                                                 sidedness(&double_sided, 1, 1, kDLUInt);
    const ovspatialquery_triangle_mesh_definition_table_t meshes{&values.value,
                                                                 &vertex_offsets.value,
                                                                 &vertex_counts.value,
                                                                 &triangle_offsets.value,
                                                                 &triangle_counts.value,
                                                                 &vertex_data.value,
                                                                 &triangle_data.value,
                                                                 &sidedness.value,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr};
    require(finish(instance, ovspatialquery_enqueue_triangle_mesh_registration(instance, &meshes)),
            "CUDA showcase mesh registration failed");

    const ovspatialquery_population_desc_t population_desc{1u, 1u};
    require(ok(ovspatialquery_create_explicit_population(instance, &population_desc, out_population)),
            "CUDA showcase population create failed");
    uint64_t                                       object           = 303u;
    int32_t                                        geometry_type    = OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
    double                                         translation[3]   = {4.0, 0.0, 0.0};
    float                                          rotation[4]      = {0.0f, 0.0f, 0.0f, 1.0f};
    uint64_t                                       participation[2] = {UINT64_C(0x20), 0u};
    Input                                          objects(&object, 1, 1, kDLUInt);
    Input                                          types(&geometry_type, 1, 1, kDLInt);
    Input                                          geometry_values(&geometry_value, 1, 1, kDLUInt);
    Input                                          positions(translation, 1, 3, kDLFloat);
    Input                                          rotations(rotation, 1, 4, kDLFloat);
    Input                                          bits(participation, 1, 2, kDLUInt);
    const ovspatialquery_explicit_population_set_t set{9,
                                                       {&objects.value, &types.value, &geometry_values.value,
                                                        &positions.value, &rotations.value, nullptr, &bits.value,
                                                        nullptr},
                                                       nullptr,
                                                       0u};
    require(finish(instance, ovspatialquery_enqueue_explicit_population_set(instance, *out_population, &set),
                   out_publication),
            "CUDA showcase population publication failed");
}

void waitCompleted(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation)
{
    ovspatialquery_op_wait_result_t waited{};
    require(ok(ovspatialquery_wait_op(instance, operation, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                waited.state == OVSPATIALQUERY_OP_COMPLETED,
            "CUDA family operation did not complete");
}

struct CudaShowcaseEvidence
{
    ovspatialquery_ordinal_t source_ordinal = 0;
    CudaFamilyEvidence       raycast{};
    CudaFamilyEvidence       sphere_sweep{};
    CudaFamilyEvidence       overlap{};
};

struct CpuRayEvidence
{
    ovspatialquery_ordinal_t source_ordinal = 0;
    uint64_t                 object_id      = 0u;
    int32_t                  geometry_type  = 0;
    uint64_t                 geometry_value = 0u;
    std::array<double, 3>    origin{};
    std::array<double, 3>    position{};
    std::array<float, 3>     normal{};
    double                   distance     = 0.0;
    uint32_t                 feature_kind = OVSPATIALQUERY_GEOMETRY_FEATURE_NONE;
};

struct OvstageShowcaseEvidence
{
    ovspatialquery_ordinal_t source_ordinal      = 0;
    uint64_t                 object_count        = 0u;
    uint64_t                 accepted_query_bits = 0u;
    uint64_t                 rejected_query_bits = 0u;
    std::array<double, 3>    source_translation{2.0, 0.0, 0.0};
    double                   source_radius = 1.0;
    CpuRayEvidence           accepted{};
};

struct ExplicitShowcaseEvidence
{
    ovspatialquery_ordinal_t source_ordinal = 0;
    uint64_t                 object_count   = 0u;
    std::array<double, 3>    sphere_translation{4.0, 0.0, 0.0};
    double                   sphere_radius = 2.0;
    std::array<double, 3>    paired_translation{5.0, 0.0, 0.0};
    double                   paired_radius = 1.0;
    CpuRayEvidence           candidate{};
    std::array<uint64_t, 2>  pair{};
};

struct ConvexPreparationEvidence
{
    ovspatialquery_ordinal_t source_ordinal = 0;
    uint64_t                 object_count   = 0u;
    CpuRayEvidence           exact{};
};

CpuRayEvidence boundsEvidence(const ovspatialquery_raycast_single_result_t& result)
{
    const ovspatialquery_raycast_bounds_candidate_t& hit = result.hit.data.bounds_candidate;
    CpuRayEvidence                                   evidence{};
    evidence.source_ordinal = result.source_ordinal;
    evidence.object_id      = hit.identity.object_id.value;
    evidence.geometry_type  = hit.identity.geometry.type;
    evidence.geometry_value = hit.identity.geometry.value;
    for (size_t axis = 0u; axis != 3u; ++axis)
    {
        evidence.position[axis] = hit.bounds_position[axis];
        evidence.normal[axis]   = hit.bounds_normal[axis];
    }
    evidence.distance = hit.distance;
    return evidence;
}

CpuRayEvidence exactEvidence(const ovspatialquery_raycast_single_result_t& result)
{
    const ovspatialquery_raycast_exact_hit_t& hit = result.hit.data.exact;
    CpuRayEvidence                            evidence{};
    evidence.source_ordinal = result.source_ordinal;
    evidence.object_id      = hit.identity.object_id.value;
    evidence.geometry_type  = hit.identity.geometry.type;
    evidence.geometry_value = hit.identity.geometry.value;
    for (size_t axis = 0u; axis != 3u; ++axis)
    {
        evidence.position[axis] = hit.position[axis];
        evidence.normal[axis]   = hit.normal[axis];
    }
    evidence.distance     = hit.distance;
    evidence.feature_kind = hit.feature.kind;
    return evidence;
}

CudaFamilyEvidence runCudaRaycast(CudaDriver& cuda, CudaDriver::Stream producer, CudaDriver::Stream consumer,
                                  ovspatialquery_instance_handle_t   instance,
                                  ovspatialquery_population_handle_t population,
                                  ovspatialquery_ordinal_t           source_ordinal)
{
    ovspatialquery_raycast_plan_desc_t plan_desc{};
    plan_desc.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    plan_desc.target.handle.population = population;
    plan_desc.hit_mode                 = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    plan_desc.precision                = OVSPATIALQUERY_QUERY_EXACT;
    plan_desc.device                   = {kDLCUDA, 0};
    plan_desc.maximum_query_count      = 1u;
    plan_desc.maximum_hits_per_query   = 1u;
    plan_desc.hit_capacity_per_slot    = 1u;
    plan_desc.result_slot_count        = 1u;
    plan_desc.result_fields            = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                              OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE |
                              OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    require(ok(ovspatialquery_create_raycast_plan(instance, &plan_desc, &plan)), "CUDA raycast plan creation failed");

    double                         origin[3]        = {0.0, 0.0, 0.0};
    float                          direction[3]     = {1.0f, 0.0f, 0.0f};
    double                         maximum_distance = 10.0;
    uint64_t                       query_bits[2]    = {UINT64_C(0x20), 0u};
    CudaInput                      origins(cuda, producer, origin, 1, 3, kDLFloat);
    CudaInput                      directions(cuda, producer, direction, 1, 3, kDLFloat);
    CudaInput                      maximum_distances(cuda, producer, &maximum_distance, 1, 1, kDLFloat);
    CudaInput                      bits(cuda, producer, query_bits, 1, 2, kDLUInt);
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.origins              = &origins.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximum_distances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP &&
                reservation.common.source_ordinal == source_ordinal &&
                reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                reservation.common.result_ready.value != 0u,
            "CUDA raycast admission metadata was not truthful");
    waitCompleted(instance, operation.op_id);
    ovspatialquery_raycast_batch_result_t result{};
    require(ok(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &result)), "CUDA raycast fetch failed");
    const CudaContactTensors contacts{&result.columns.contact.exact.positions, &result.columns.contact.exact.normals,
                                      nullptr, &result.columns.contact.exact.distances};
    const CudaFamilyEvidence evidence = consumeCudaResult(cuda, consumer, instance, operation, result,
                                                          &result.columns.contact.exact.features, &contacts);
    require(std::abs(evidence.position[0] - 4.0) < 1.0e-9 && std::abs(evidence.position[1]) < 1.0e-9 &&
                std::abs(evidence.position[2]) < 1.0e-9 && std::abs(evidence.normal[0] + 1.0f) < 1.0e-5f &&
                std::abs(evidence.normal[1]) < 1.0e-5f && std::abs(evidence.normal[2]) < 1.0e-5f &&
                std::abs(evidence.distance - 4.0) < 1.0e-9,
            "CUDA raycast contact columns were not exact");
    require(destroyPlanWhenRetired(instance, plan), "CUDA raycast plan destroy failed");
    return evidence;
}

CudaFamilyEvidence runCudaSweep(CudaDriver& cuda, CudaDriver::Stream producer, CudaDriver::Stream consumer,
                                ovspatialquery_instance_handle_t   instance,
                                ovspatialquery_population_handle_t population, ovspatialquery_ordinal_t source_ordinal)
{
    ovspatialquery_sphere_sweep_plan_desc_t plan_desc{};
    plan_desc.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    plan_desc.target.handle.population = population;
    plan_desc.precision                = OVSPATIALQUERY_QUERY_EXACT;
    plan_desc.radius                   = 0.5;
    plan_desc.device                   = {kDLCUDA, 0};
    plan_desc.maximum_query_count      = 1u;
    plan_desc.result_slot_count        = 1u;
    plan_desc.result_fields            = OVSPATIALQUERY_BATCH_RESULT_POSITION | OVSPATIALQUERY_BATCH_RESULT_NORMAL |
                              OVSPATIALQUERY_BATCH_RESULT_DISTANCE | OVSPATIALQUERY_BATCH_RESULT_QUERY_POSITION |
                              OVSPATIALQUERY_BATCH_RESULT_SOURCE_FEATURE |
                              OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
    plan_desc.hit_capacity_per_slot         = 1u;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    require(ok(ovspatialquery_create_sphere_sweep_plan(instance, &plan_desc, &plan)),
            "CUDA sweep plan creation failed");

    double                              center[3]        = {0.0, 0.0, 0.0};
    float                               direction[3]     = {1.0f, 0.0f, 0.0f};
    double                              maximum_distance = 10.0;
    uint64_t                            query_bits[2]    = {UINT64_C(0x20), 0u};
    CudaInput                           centers(cuda, producer, center, 1, 3, kDLFloat);
    CudaInput                           directions(cuda, producer, direction, 1, 3, kDLFloat);
    CudaInput                           maximum_distances(cuda, producer, &maximum_distance, 1, 1, kDLFloat);
    CudaInput                           bits(cuda, producer, query_bits, 1, 2, kDLUInt);
    ovspatialquery_sphere_sweep_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.centers              = &centers.value;
    batch.unit_directions      = &directions.value;
    batch.max_distances        = &maximum_distances.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_sphere_sweep_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t                 operation =
        ovspatialquery_enqueue_sphere_sweep_plan(instance, plan, &batch, &reservation);
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP &&
                reservation.common.source_ordinal == source_ordinal &&
                reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                reservation.common.result_ready.value != 0u,
            "CUDA sweep admission metadata was not truthful");
    waitCompleted(instance, operation.op_id);
    ovspatialquery_sphere_sweep_batch_result_t result{};
    require(ok(ovspatialquery_fetch_sphere_sweep_batch(instance, operation.op_id, &result)), "CUDA sweep fetch failed");
    const CudaContactTensors contacts{&result.columns.contact.exact.positions, &result.columns.contact.exact.normals,
                                      &result.columns.contact.exact.query_positions,
                                      &result.columns.contact.exact.distances};
    const CudaFamilyEvidence evidence = consumeCudaResult(cuda, consumer, instance, operation, result,
                                                          &result.columns.contact.exact.features, &contacts);
    require(std::abs(evidence.position[0] - 4.0) < 1.0e-9 && std::abs(evidence.position[1]) < 1.0e-9 &&
                std::abs(evidence.position[2]) < 1.0e-9 && std::abs(evidence.normal[0] + 1.0f) < 1.0e-5f &&
                std::abs(evidence.normal[1]) < 1.0e-5f && std::abs(evidence.normal[2]) < 1.0e-5f &&
                std::abs(evidence.query_position[0] - 3.5) < 1.0e-9 && std::abs(evidence.query_position[1]) < 1.0e-9 &&
                std::abs(evidence.query_position[2]) < 1.0e-9 && std::abs(evidence.distance - 3.5) < 1.0e-9,
            "CUDA sphere-sweep contact columns were not exact");
    require(destroyPlanWhenRetired(instance, plan), "CUDA sweep plan destroy failed");
    return evidence;
}

CudaFamilyEvidence runCudaOverlap(CudaDriver& cuda, CudaDriver::Stream producer, CudaDriver::Stream consumer,
                                  ovspatialquery_instance_handle_t   instance,
                                  ovspatialquery_population_handle_t population,
                                  ovspatialquery_ordinal_t           source_ordinal)
{
    ovspatialquery_overlap_plan_desc_t plan_desc{};
    plan_desc.target.kind                   = OVSPATIALQUERY_TARGET_POPULATION;
    plan_desc.target.handle.population      = population;
    plan_desc.hit_mode                      = OVSPATIALQUERY_HIT_MODE_ANY;
    plan_desc.precision                     = OVSPATIALQUERY_QUERY_EXACT;
    plan_desc.shape.kind                    = OVSPATIALQUERY_OVERLAP_SPHERE;
    plan_desc.shape.data.sphere.radius      = 0.5;
    plan_desc.device                        = {kDLCUDA, 0};
    plan_desc.maximum_query_count           = 1u;
    plan_desc.maximum_hits_per_query        = 1u;
    plan_desc.hit_capacity_per_slot         = 1u;
    plan_desc.result_slot_count             = 1u;
    plan_desc.result_fields                 = OVSPATIALQUERY_BATCH_RESULT_GEOMETRY_IDENTITY;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    require(ok(ovspatialquery_create_overlap_plan(instance, &plan_desc, &plan)), "CUDA overlap plan creation failed");

    double                         translation[3] = {4.0, 0.0, 0.0};
    uint64_t                       query_bits[2]  = {UINT64_C(0x20), 0u};
    CudaInput                      translations(cuda, producer, translation, 1, 3, kDLFloat);
    CudaInput                      bits(cuda, producer, query_bits, 1, 2, kDLUInt);
    ovspatialquery_overlap_batch_t batch{};
    batch.ordinal.kind         = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count          = 1u;
    batch.query_bits           = &bits.value;
    batch.translations         = &translations.value;
    batch.cuda_sync.has_stream = 1u;
    batch.cuda_sync.stream     = reinterpret_cast<uintptr_t>(producer);
    ovspatialquery_overlap_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            operation =
        ovspatialquery_enqueue_overlap_plan(instance, plan, &batch, &reservation);
    require(operation.status == OVSPATIALQUERY_API_SUCCESS && operation.op_id != OVSPATIALQUERY_INVALID_OP &&
                reservation.common.source_ordinal == source_ordinal &&
                reservation.common.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                reservation.common.result_ready.value != 0u,
            "CUDA overlap admission metadata was not truthful");
    waitCompleted(instance, operation.op_id);
    ovspatialquery_overlap_batch_result_t result{};
    require(ok(ovspatialquery_fetch_overlap_batch(instance, operation.op_id, &result)), "CUDA overlap fetch failed");
    const CudaFamilyEvidence evidence =
        consumeCudaResult(cuda, consumer, instance, operation, result, nullptr, nullptr);
    require(destroyPlanWhenRetired(instance, plan), "CUDA overlap plan destroy failed");
    return evidence;
}

CudaShowcaseEvidence runCuda(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population,
                             ovspatialquery_ordinal_t source_ordinal)
{
    CudaDriver               cuda(CudaDriver::Ownership::tracked);
    const CudaDriver::Stream producer = cuda.createStream();
    const CudaDriver::Stream consumer = cuda.createStream();
    CudaShowcaseEvidence     evidence{};
    evidence.source_ordinal = source_ordinal;
    evidence.raycast        = runCudaRaycast(cuda, producer, consumer, instance, population, source_ordinal);
    evidence.sphere_sweep   = runCudaSweep(cuda, producer, consumer, instance, population, source_ordinal);
    evidence.overlap        = runCudaOverlap(cuda, producer, consumer, instance, population, source_ordinal);
    return evidence;
}

OvstageShowcaseEvidence runOvstage(ovspatialquery_instance_handle_t instance)
{
    const std::string            root = std::string(OVSTAGE_PACKAGE_ROOT) + "/bin";
    const ovstage_config_entry_t config_entry =
        ovstage_config_entry_binary_package_root_path({root.data(), root.size()});
    const ovstage_config_t config{&config_entry, 1u};
    require(ovstage_initialize(&config) == OVSTAGE_OK, "ovstage init failed");
    ovstage_instance_t*     stage = nullptr;
    OvstageShowcaseEvidence evidence{};
    // Value-initialized: Ovstage 0.4 adds `implementation`, which 0.3 lacks.
    ovstage_instance_desc_t desc{};
    desc.name = "ovspatialquery.lesson11.capabilities";
    require(ovstage_create_instance(&desc, &stage) == OVSTAGE_OK && stage != nullptr, "ovstage instance failed");
    try
    {
        require(waitOpen(stage, example_ovstage::openUsda(stage, {kStageScene, sizeof(kStageScene) - 1u}, 1u, 0.0)),
                "ovstage open failed");
        const ovstage_write_floor_desc_t floor_one{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        require(waitStage(stage, ovstage_advance_write_floor(stage, &floor_one)), "ovstage floor failed");
        require(
            waitStage(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)),
            "ovstage hierarchy failed");
        const ovstage_write_floor_desc_t floor_two{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        require(waitStage(stage, ovstage_advance_write_floor(stage, &floor_two)), "ovstage seal failed");
        const ovx_string_t  path{kStagePath, sizeof(kStagePath) - 1u};
        ovstage_predicate_t predicate{};
        predicate.attribute.string = {"usd-path", sizeof("usd-path") - 1u};
        predicate.op               = OVSTAGE_FILTER_OP_IN;
        predicate.values           = &path;
        predicate.value_count      = 1u;
        ovspatialquery_ovstage_population_desc_t population_desc{};
        population_desc.stage                            = stage;
        population_desc.filter                           = {&predicate, 1u};
        population_desc.common.retained_ordinal_capacity = 1u;
        ovspatialquery_population_handle_t population    = OVSPATIALQUERY_INVALID_POPULATION;
        require(ok(ovspatialquery_create_ovstage_population(instance, &population_desc, &population)),
                "ovstage population failed");
        // A table with no entries gives every object default_bits.
        const ovspatialquery_ovstage_participation_table_t participation{{{UINT64_C(0x40), 0u}}, nullptr, nullptr, 0u};
        require(ok(ovspatialquery_set_ovstage_participation_map(instance, population, &participation)),
                "ovstage participation table failed");
        ovspatialquery_population_update_result_t publication{};
        require(
            finish(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, 2), &publication) &&
                publication.object_count == 1u,
            "ovstage publication failed");
        evidence.source_ordinal = publication.published_ordinal;
        evidence.object_count   = publication.object_count;
        ovspatialquery_raycast_desc_t ray{};
        ray.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
        ray.common.target.target.handle.population = population;
        ray.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
        ray.common.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
        ray.common.query_bits                      = {{UINT64_C(0x40), 0u}};
        ray.ray.origin[0]                          = -3.0;
        ray.ray.unit_direction[0]                  = 1.0f;
        ray.ray.max_distance                       = 10.0;
        ovspatialquery_raycast_single_result_t hit{};
        require(ok(ovspatialquery_raycast_closest_sync(instance, &ray, &hit)) && hit.has_hit != 0u &&
                    hit.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                    hit.hit.data.bounds_candidate.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_SPHERE,
                "ovstage candidate ray failed");
        evidence.accepted_query_bits = ray.common.query_bits.words[0];
        evidence.accepted            = boundsEvidence(hit);
        evidence.accepted.origin     = {ray.ray.origin[0], ray.ray.origin[1], ray.ray.origin[2]};
        ray.common.query_bits        = {{UINT64_C(0x80), 0u}};
        ovspatialquery_raycast_single_result_t rejected{};
        require(ok(ovspatialquery_raycast_closest_sync(instance, &ray, &rejected)) && rejected.has_hit == 0u,
                "ovstage participation table did not reject a disjoint query bit");
        evidence.rejected_query_bits = ray.common.query_bits.words[0];
        require(ok(ovspatialquery_destroy_population(instance, population)), "ovstage population destroy failed");
    }
    catch (...)
    {
        ovstage_destroy_instance(stage);
        ovstage_shutdown();
        throw;
    }
    require(ovstage_destroy_instance(stage) == OVSTAGE_OK, "ovstage destroy failed");
    require(ovstage_shutdown() == OVSTAGE_OK, "ovstage shutdown failed");
    return evidence;
}

template <typename Value>
Value canonicalZero(Value value)
{
    return value == static_cast<Value>(0) ? static_cast<Value>(0) : value;
}

void appendCudaFamilyJson(std::ostringstream& output, const char* name, const CudaFamilyEvidence& evidence,
                          bool include_feature, bool include_query_position)
{
    output << '\"' << name
           << "\":{\"truth\":\"EXACT_SURFACE\","
              "\"row_status\":\"SUCCESS\",\"stored_hit_count\":"
           << evidence.stored_hit_count << ",\"hit_offset\":" << evidence.hit_offset
           << ",\"hit_count\":" << evidence.hit_count << ",\"truncated\":" << static_cast<uint32_t>(evidence.truncated)
           << ",\"object_id\":" << evidence.object_id
           << ",\"geometry_type\":\"TRIANGLE_MESH\",\"geometry_value\":" << evidence.geometry_value;
    if (include_feature)
    {
        output << ",\"origin\":[0,0,0],\"position\":[" << canonicalZero(evidence.position[0]) << ','
               << canonicalZero(evidence.position[1]) << ',' << canonicalZero(evidence.position[2]) << "],\"normal\":["
               << canonicalZero(evidence.normal[0]) << ',' << canonicalZero(evidence.normal[1]) << ','
               << canonicalZero(evidence.normal[2]) << "],\"distance\":" << canonicalZero(evidence.distance);
        if (include_query_position)
        {
            output << ",\"query_position\":[" << canonicalZero(evidence.query_position[0]) << ','
                   << canonicalZero(evidence.query_position[1]) << ',' << canonicalZero(evidence.query_position[2])
                   << ']';
        }
    }
    if (include_feature)
    {
        output << ",\"feature\":{\"kind\":\"TRIANGLE\",\"source_face\":" << evidence.source_face
               << ",\"source_corners\":[" << evidence.source_corners[0] << ',' << evidence.source_corners[1] << ','
               << evidence.source_corners[2] << "],\"barycentrics\":[" << canonicalZero(evidence.barycentrics[0]) << ','
               << canonicalZero(evidence.barycentrics[1]) << ',' << canonicalZero(evidence.barycentrics[2]) << "]}";
    }
    output << '}';
}

void appendCpuRayJson(std::ostringstream& output, const CpuRayEvidence& evidence, const char* truth,
                      const char* geometry_type, bool include_feature = false)
{
    output << "{\"truth\":\"" << truth << "\",\"source_ordinal\":" << evidence.source_ordinal
           << ",\"object_id\":" << evidence.object_id << ",\"geometry_type\":\"" << geometry_type
           << "\",\"geometry_value\":" << evidence.geometry_value << ",\"origin\":["
           << canonicalZero(evidence.origin[0]) << ',' << canonicalZero(evidence.origin[1]) << ','
           << canonicalZero(evidence.origin[2]) << "],\"position\":[" << canonicalZero(evidence.position[0]) << ','
           << canonicalZero(evidence.position[1]) << ',' << canonicalZero(evidence.position[2]) << "],\"normal\":["
           << canonicalZero(evidence.normal[0]) << ',' << canonicalZero(evidence.normal[1]) << ','
           << canonicalZero(evidence.normal[2]) << "],\"distance\":" << canonicalZero(evidence.distance);
    if (include_feature)
    {
        require(evidence.feature_kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE, "convex exact feature must be NONE");
        output << ",\"feature\":\"NONE\"";
    }
    output << '}';
}

std::string json(const CudaShowcaseEvidence* cuda, const std::string& cuda_unavailable_reason,
                 const OvstageShowcaseEvidence& ovstage, const ExplicitShowcaseEvidence& explicit_cpu,
                 const ConvexPreparationEvidence& convex)
{
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << "{\"schema_version\":1,\"status\":\"PASS\","
              "\"lanes\":{\"core\":\"PASS\",\"ovstage\":\"PASS\","
              "\"participation\":\"PASS\","
              "\"explicit_cpu\":\"PASS\",\"cuda\":\""
           << (cuda != nullptr ? "PASS" : "NOT_AVAILABLE")
           << "\"},"
              "\"scenarios\":[{\"panel\":\"ovstage\",\"label\":\"Ovstage participation accept/reject\","
              "\"truth\":\"BOUNDS_CANDIDATE\"},{\"panel\":\"explicit\","
              "\"label\":\"Explicit publication, candidate, and pair\","
              "\"truth\":\"BOUNDS_CANDIDATE\"},{\"panel\":\"cuda\",";
    if (cuda != nullptr)
    {
        output << "\"label\":\"CUDA exact family columns\",\"truth\":\"EXACT_SURFACE\"}";
    }
    else
    {
        output << "\"label\":\"CUDA unavailable\",\"truth\":\"NOT_AVAILABLE\"}";
    }
    output << "],\"ovstage\":{\"publication\":{"
              "\"source_ordinal\":"
           << ovstage.source_ordinal << ",\"object_count\":" << ovstage.object_count
           << "},\"accepted_query_bits\":" << ovstage.accepted_query_bits
           << ",\"rejected_query_bits\":" << ovstage.rejected_query_bits << ",\"source\":{\"translation\":["
           << ovstage.source_translation[0] << ',' << ovstage.source_translation[1] << ','
           << ovstage.source_translation[2] << "],\"radius\":" << ovstage.source_radius << "},\"accepted\":";
    appendCpuRayJson(output, ovstage.accepted, "BOUNDS_CANDIDATE", "SPHERE");
    output << "},\"explicit\":{\"publication\":{\"source_ordinal\":" << explicit_cpu.source_ordinal
           << ",\"object_count\":" << explicit_cpu.object_count
           << "},\"objects\":[{\"object_id\":101,"
              "\"geometry_type\":\"SPHERE\",\"translation\":["
           << explicit_cpu.sphere_translation[0] << ',' << explicit_cpu.sphere_translation[1] << ','
           << explicit_cpu.sphere_translation[2] << "],\"radius\":" << explicit_cpu.sphere_radius
           << "},{\"object_id\":202,"
              "\"geometry_type\":\"SPHERE\",\"translation\":["
           << explicit_cpu.paired_translation[0] << ',' << explicit_cpu.paired_translation[1] << ','
           << explicit_cpu.paired_translation[2] << "],\"radius\":" << explicit_cpu.paired_radius
           << "},{\"object_id\":" << kConvexObject
           << ",\"geometry_type\":\"CONVEX\","
              "\"translation\":[0,0,0]}],\"candidate\":";
    appendCpuRayJson(output, explicit_cpu.candidate, "BOUNDS_CANDIDATE", "SPHERE");
    output << ",\"pair\":[" << explicit_cpu.pair[0] << ',' << explicit_cpu.pair[1] << "]},\"cuda\":{";
    if (cuda != nullptr)
    {
        output << "\"state\":\"PASS\",\"device\":\"kDLCUDA:0\","
                  "\"source_ordinal\":"
               << cuda->source_ordinal << ",\"families\":{";
        appendCudaFamilyJson(output, "raycast", cuda->raycast, true, false);
        output << ',';
        appendCudaFamilyJson(output, "sphere_sweep", cuda->sphere_sweep, true, true);
        output << ',';
        appendCudaFamilyJson(output, "overlap", cuda->overlap, false, false);
        output << "}}";
    }
    else
    {
        output << "\"state\":\"NOT_AVAILABLE\",\"reason\":\"" << cuda_unavailable_reason << "\"}";
    }
    output << ",\"convex_preparation\":{\"authored\":{\"points\":[";
    for (size_t point = 0u; point != kAuthoredConvexPoints.size() / 3u; ++point)
    {
        if (point != 0u)
        {
            output << ',';
        }
        output << '[' << kAuthoredConvexPoints[point * 3u] << ',' << kAuthoredConvexPoints[point * 3u + 1u] << ','
               << kAuthoredConvexPoints[point * 3u + 2u] << ']';
    }
    output << "],\"dent_position\":[0,0,0]},\"publication\":{"
              "\"source_ordinal\":"
           << convex.source_ordinal << ",\"object_count\":" << convex.object_count << "},\"exact\":";
    appendCpuRayJson(output, convex.exact, "EXACT_SURFACE", "CONVEX", true);
    output << "},\"cleanup\":true}\n";
    return output.str();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        std::string json_path;
        bool        require_cuda = false;
        for (int index = 1; index < argc; ++index)
        {
            if (std::string(argv[index]) == "--json" && index + 1 < argc)
            {
                json_path = argv[++index];
            }
            else if (std::string(argv[index]) == "--require-cuda")
            {
                require_cuda = true;
            }
        }
        require(!json_path.empty(), "--json evidence path is required");
        bool        run_cuda                = false;
        std::string cuda_unavailable_reason = "CUDA validation was not requested by the portable CPU contract";
        if (require_cuda)
        {
            try
            {
                CudaDriver cuda_probe(CudaDriver::Ownership::tracked);
                (void)cuda_probe;
                run_cuda = true;
            }
            catch (const std::runtime_error& error)
            {
                cuda_unavailable_reason = error.what();
            }
        }
        const ovspatialquery_device_budget_t cuda_budget{{kDLCUDA, 0}, UINT64_C(64) << 20u};
        const ovspatialquery_instance_desc_t instance_desc{16u, 2u, UINT64_C(64) << 20u,
                                                           run_cuda ? &cuda_budget : nullptr, run_cuda ? 1u : 0u};
        ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
        require(ok(ovspatialquery_create_instance(&instance_desc, &instance)), "instance create failed");
        ovspatialquery_population_handle_t explicit_population = OVSPATIALQUERY_INVALID_POPULATION;
        ovspatialquery_population_handle_t cuda_population     = OVSPATIALQUERY_INVALID_POPULATION;
        try
        {
            uint64_t sphere_value          = 7u;
            uint64_t sphere_definitions[2] = {sphere_value, kPairedSphereValue};
            double   sphere_radii[2]       = {2.0, 1.0};
            Input    sphere_values(sphere_definitions, 2, 1, kDLUInt, OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL);
            Input    radii(sphere_radii, 2, 1, kDLFloat, OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL);
            const ovspatialquery_sphere_definition_table_t sphere{&sphere_values.value, &radii.value};
            require(finish(instance, ovspatialquery_enqueue_sphere_registration(instance, &sphere)),
                    "sphere registration failed");
            registerPreparedConvex(instance);
            const ovspatialquery_population_desc_t population_desc{1u, 1u};
            require(ok(ovspatialquery_create_explicit_population(instance, &population_desc, &explicit_population)),
                    "explicit population create failed");
            ovspatialquery_population_update_result_t publication{};
            publishExplicit(instance, explicit_population, sphere_value, &publication);
            CudaShowcaseEvidence cuda_evidence{};
            if (run_cuda)
            {
                ovspatialquery_population_update_result_t cuda_publication{};
                publishCudaMesh(instance, &cuda_population, &cuda_publication);
                cuda_evidence = runCuda(instance, cuda_population, cuda_publication.published_ordinal);
            }
            const OvstageShowcaseEvidence ovstage_evidence = runOvstage(instance);
            ovspatialquery_raycast_desc_t candidate{};
            candidate.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
            candidate.common.target.target.handle.population = explicit_population;
            candidate.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
            candidate.common.precision                       = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
            candidate.common.query_bits                      = {{1u, 0u}};
            candidate.ray.origin[0]                          = -2.0;
            candidate.ray.unit_direction[0]                  = 1.0f;
            candidate.ray.max_distance                       = 20.0;
            ovspatialquery_raycast_single_result_t candidate_result{};
            require(ok(ovspatialquery_raycast_closest_sync(instance, &candidate, &candidate_result)) &&
                        candidate_result.has_hit != 0u &&
                        candidate_result.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE &&
                        candidate_result.hit.data.bounds_candidate.identity.object_id.value == 101u &&
                        candidate_result.hit.data.bounds_candidate.identity.geometry.type ==
                            OVSPATIALQUERY_GEOMETRY_SPHERE &&
                        candidate_result.hit.data.bounds_candidate.identity.geometry.value == sphere_value,
                    "explicit candidate ray failed");
            ExplicitShowcaseEvidence explicit_evidence{};
            explicit_evidence.source_ordinal   = publication.published_ordinal;
            explicit_evidence.object_count     = publication.object_count;
            explicit_evidence.candidate        = boundsEvidence(candidate_result);
            explicit_evidence.candidate.origin = {candidate.ray.origin[0], candidate.ray.origin[1],
                                                  candidate.ray.origin[2]};
            ovspatialquery_raycast_desc_t convex_ray{};
            convex_ray.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
            convex_ray.common.target.target.handle.population = explicit_population;
            convex_ray.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
            convex_ray.common.precision                       = OVSPATIALQUERY_QUERY_EXACT;
            convex_ray.common.query_bits                      = {{4u, 0u}};
            convex_ray.ray.origin[1]                          = 3.0;
            convex_ray.ray.unit_direction[1]                  = -1.0f;
            convex_ray.ray.max_distance                       = 10.0;
            ovspatialquery_raycast_single_result_t convex_result{};
            require(
                ok(ovspatialquery_raycast_closest_sync(instance, &convex_ray, &convex_result)) &&
                    convex_result.has_hit != 0u && convex_result.source_ordinal == publication.published_ordinal &&
                    convex_result.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE &&
                    convex_result.hit.data.exact.identity.object_id.value == kConvexObject &&
                    convex_result.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX &&
                    convex_result.hit.data.exact.identity.geometry.value == kConvexValue &&
                    convex_result.hit.data.exact.position[0] == 0.0 &&
                    convex_result.hit.data.exact.position[1] == 1.0 &&
                    convex_result.hit.data.exact.position[2] == 0.0 && convex_result.hit.data.exact.normal[0] == 0.0f &&
                    convex_result.hit.data.exact.normal[1] == 1.0f && convex_result.hit.data.exact.normal[2] == 0.0f &&
                    convex_result.hit.data.exact.distance == 2.0 &&
                    convex_result.hit.data.exact.feature.kind == OVSPATIALQUERY_GEOMETRY_FEATURE_NONE,
                "prepared convex exact ray failed");
            ConvexPreparationEvidence convex_evidence{};
            convex_evidence.source_ordinal = publication.published_ordinal;
            convex_evidence.object_count   = publication.object_count;
            convex_evidence.exact          = exactEvidence(convex_result);
            convex_evidence.exact.origin   = {convex_ray.ray.origin[0], convex_ray.ray.origin[1],
                                              convex_ray.ray.origin[2]};
            ovspatialquery_pair_t               pairs[1]{};
            ovspatialquery_pair_buffer_t        pair_buffer{pairs, 1u};
            ovspatialquery_pair_snapshot_desc_t pair_desc{};
            pair_desc.first.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
            pair_desc.first.target.handle.population = explicit_population;
            pair_desc.first.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
            pair_desc.second                         = pair_desc.first;
            ovspatialquery_pair_snapshot_result_t pair_result{};
            require(ok(ovspatialquery_query_pairs_sync(instance, &pair_desc, &pair_buffer, &pair_result)) &&
                        pair_result.stored_pair_count == 1u && pairs[0].first.value == 101u &&
                        pairs[0].second.value == 202u,
                    "explicit pair failed");
            explicit_evidence.pair = {pairs[0].first.value, pairs[0].second.value};
            if (run_cuda)
            {
                require(ok(ovspatialquery_destroy_population(instance, cuda_population)),
                        "CUDA showcase population destroy failed");
                cuda_population = OVSPATIALQUERY_INVALID_POPULATION;
                require(ok(ovspatialquery_destroy_geometry_definition(instance,
                                                                      {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, 88u})),
                        "CUDA showcase mesh destroy failed");
            }
            require(ok(ovspatialquery_destroy_population(instance, explicit_population)),
                    "explicit population destroy failed");
            explicit_population = OVSPATIALQUERY_INVALID_POPULATION;
            require(ok(ovspatialquery_destroy_geometry_definition(instance,
                                                                  {OVSPATIALQUERY_GEOMETRY_CONVEX, kConvexValue})),
                    "convex definition destroy failed");
            require(ok(ovspatialquery_destroy_geometry_definition(instance,
                                                                  {OVSPATIALQUERY_GEOMETRY_SPHERE, sphere_value})),
                    "sphere destroy failed");
            require(ok(ovspatialquery_destroy_geometry_definition(
                        instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kPairedSphereValue})),
                    "paired sphere destroy failed");
            require(ok(ovspatialquery_destroy_instance(instance)), "instance destroy failed");
            instance                            = OVSPATIALQUERY_INVALID_INSTANCE;
            const std::string           payload = json(run_cuda ? &cuda_evidence : nullptr, cuda_unavailable_reason,
                                             ovstage_evidence, explicit_evidence, convex_evidence);
            const std::filesystem::path json_output(json_path);
            if (!json_output.parent_path().empty())
            {
                std::error_code directory_error;
                std::filesystem::create_directories(json_output.parent_path(), directory_error);
                require(!directory_error, "cannot create JSON evidence directory");
            }
            std::ofstream output(json_output, std::ios::binary | std::ios::trunc);
            require(output.good(), "cannot open JSON evidence");
            output << payload;
            require(output.good(), "cannot write JSON evidence");
            std::cout << "PASS core/ovstage/explicit installed-public showcase\n";
            if (run_cuda)
            {
                std::cout << "PASS CUDA ray/sweep/overlap real device result columns\n";
            }
            else
            {
                std::cout << "NOT_AVAILABLE CUDA: " << cuda_unavailable_reason << '\n';
            }
            std::cout << payload;
            return require_cuda && !run_cuda ? 2 : 0;
        }
        catch (...)
        {
            if (cuda_population != OVSPATIALQUERY_INVALID_POPULATION)
            {
                ovspatialquery_destroy_population(instance, cuda_population);
            }
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH, 88u});
            if (explicit_population != OVSPATIALQUERY_INVALID_POPULATION)
            {
                ovspatialquery_destroy_population(instance, explicit_population);
            }
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_CONVEX, kConvexValue});
            if (instance != OVSPATIALQUERY_INVALID_INSTANCE)
            {
                ovspatialquery_destroy_instance(instance);
            }
            throw;
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "FAIL capability showcase: " << error.what() << '\n';
        return 1;
    }
}
