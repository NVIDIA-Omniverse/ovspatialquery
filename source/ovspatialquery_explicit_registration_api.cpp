/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_instance_state.h"

#include "ovspatialquery_explicit_cuda_capture.h"
#include "ovspatialquery_input_layout.h"
#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <iterator>
#include <limits>
#include <new>
#include <string>

using namespace ovspatialquery::detail;

namespace
{

// An ownership outside the enum is INVALID_ARGUMENT before the tensor's dtype,
// device or rank is looked at; PreparedInputArray checks those first.
ovspatialquery_api_status_t explicitPreparedArray(const ovspatialquery_input_array_t* source, const char* column,
                                                  uint8_t code, uint8_t bits, int32_t rank, int64_t trailing,
                                                  ovspatialquery::detail::PreparedInputArray& out) noexcept
{
    if (source != nullptr && source->ownership > 1u)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    return ovspatialquery::detail::PreparedInputArray::prepare(source, code, bits, rank, trailing, column, out);
}

ovspatialquery_api_status_t explicitPreparedRealArray(const ovspatialquery_input_array_t* source, const char* column,
                                                      int32_t rank, int64_t trailing, PreparedInputArray& out) noexcept
{
    if (source == nullptr)
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    const uint8_t bits = source->tensor.dtype.bits;
    if (bits != 32u && bits != 64u)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    return explicitPreparedArray(source, column, kDLFloat, bits, rank, trailing, out);
}

std::vector<float> explicitReadVertices(const PreparedInputArray& source, size_t offset, size_t count)
{
    std::vector<float> result(count);
    for (size_t index = 0u; index < count; ++index)
    {
        result[index] = readPrimitive<float>(source, offset + index);
    }
    return result;
}

double explicitReadReal(const ovspatialquery::detail::PreparedInputArray& source, size_t index, uint8_t bits) noexcept
{
    if (bits == 32u)
    {
        return static_cast<double>(readPrimitive<float>(source, index));
    }
    return readPrimitive<double>(source, index);
}

ovspatialquery_api_status_t prepareExplicitInstances(
    const ovspatialquery_explicit_instance_table_t& table, const ovspatialquery_dynamic_execution_t* execution,
    bool posesOnly, bool rowBits, std::vector<ovspatialquery::detail::ExplicitInstanceInput>& outInstances,
    std::vector<ovspatialquery::detail::PreparedInputArray>& outLeases) noexcept
{
    if (execution != nullptr && (execution->device.device_type != kDLCPU || execution->device.device_id != 0 ||
                                 execution->cuda_sync.has_stream != 0u || execution->cuda_sync.wait_event != 0u ||
                                 execution->preparation != OVSPATIALQUERY_PREPARE_ON_HOST))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    // Poses-only rows carry no binding columns, and bits only with
    // POSES_AND_BITS; the caller fills the rest from the base.
    const bool bindings = !posesOnly;
    if (table.object_values == nullptr ||
        (bindings && (table.geometry_types == nullptr || table.geometry_values == nullptr)) ||
        (!bindings && (table.geometry_types != nullptr || table.geometry_values != nullptr ||
                       table.material_ids != nullptr || (table.participation_bits != nullptr) != rowBits)) ||
        table.translations == nullptr || table.rotations_xyzw == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::vector<ovspatialquery::detail::PreparedInputArray> leases;
        leases.reserve(8u);
        ovspatialquery::detail::PreparedInputArray objects;
        ovspatialquery::detail::PreparedInputArray types;
        ovspatialquery::detail::PreparedInputArray geometry;
        ovspatialquery::detail::PreparedInputArray translations;
        ovspatialquery::detail::PreparedInputArray rotations;
        ovspatialquery::detail::PreparedInputArray scales;
        ovspatialquery::detail::PreparedInputArray participation;
        ovspatialquery::detail::PreparedInputArray materials;
        ovspatialquery_api_status_t                status =
            explicitPreparedArray(table.object_values, "instances.object_values", kDLUInt, 64u, 1, 0, objects);
        if (status == OVSPATIALQUERY_API_SUCCESS && bindings)
        {
            status = explicitPreparedArray(table.geometry_types, "instances.geometry_types", kDLInt, 32u, 1, 0, types);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && bindings)
        {
            status =
                explicitPreparedArray(table.geometry_values, "instances.geometry_values", kDLUInt, 64u, 1, 0, geometry);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = explicitPreparedRealArray(table.translations, "instances.translations", 2, 3, translations);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = explicitPreparedRealArray(table.rotations_xyzw, "instances.rotations_xyzw", 2, 4, rotations);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && table.uniform_scales != nullptr)
        {
            status = explicitPreparedRealArray(table.uniform_scales, "instances.uniform_scales", 1, 0, scales);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && table.participation_bits != nullptr)
        {
            status = explicitPreparedArray(table.participation_bits, "instances.participation_bits", kDLUInt, 64u, 2, 2,
                                           participation);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && table.material_ids != nullptr)
        {
            status = explicitPreparedArray(table.material_ids, "instances.material_ids", kDLUInt, 64u, 1, 0, materials);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        const uint8_t translationBits = table.translations->tensor.dtype.bits;
        const uint8_t scaleBits       = table.uniform_scales == nullptr ? 64u : table.uniform_scales->tensor.dtype.bits;
        const int64_t count           = objects.shape(0u);
        if ((bindings && (types.shape(0u) != count || geometry.shape(0u) != count)) ||
            translations.shape(0u) != count || rotations.shape(0u) != count ||
            (table.uniform_scales != nullptr && scales.shape(0u) != count) ||
            (table.participation_bits != nullptr && participation.shape(0u) != count) ||
            (table.material_ids != nullptr && materials.shape(0u) != count))
        {
            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
        }
        std::vector<ovspatialquery::detail::ExplicitInstanceInput> instances;
        instances.reserve(static_cast<size_t>(count));
        for (size_t row = 0u; row < static_cast<size_t>(count); ++row)
        {
            ovspatialquery::detail::ExplicitInstanceInput instance{};
            instance.objectValue = readPrimitive<uint64_t>(objects, row);
            if (bindings)
            {
                const int32_t publicType = readPrimitive<int32_t>(types, row);
                // USER is reserved: no definition of it can exist.
                if (publicType < kGeometryTypeFirstBuiltin || publicType > kGeometryTypeLast)
                {
                    return publicType == OVSPATIALQUERY_GEOMETRY_USER ? OVSPATIALQUERY_API_NOT_SUPPORTED
                                                                      : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                instance.geometry = {
                    static_cast<ovspatialquery::detail::ExactGeometryFamily>(publicType - kGeometryTypeFirstBuiltin),
                    readPrimitive<uint64_t>(geometry, row)};
            }
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                instance.translation[axis] = explicitReadReal(translations, row * 3u + axis, translationBits);
            }
            for (uint32_t component = 0u; component < 4u; ++component)
            {
                instance.rotationXyzw[component] =
                    explicitReadReal(rotations, row * 4u + component, table.rotations_xyzw->tensor.dtype.bits);
            }
            instance.uniformScale = table.uniform_scales == nullptr ? 1.0 : explicitReadReal(scales, row, scaleBits);
            if (table.participation_bits != nullptr)
            {
                instance.participationBits = {{readPrimitive<uint64_t>(participation, row * 2u),
                                               readPrimitive<uint64_t>(participation, row * 2u + 1u)}};
            }
            if (table.material_ids != nullptr)
            {
                instance.materialId = readPrimitive<uint64_t>(materials, row);
            }
            instances.push_back(instance);
        }
        leases.push_back(std::move(objects));
        if (bindings)
        {
            leases.push_back(std::move(types));
            leases.push_back(std::move(geometry));
        }
        leases.push_back(std::move(translations));
        leases.push_back(std::move(rotations));
        if (table.uniform_scales != nullptr)
        {
            leases.push_back(std::move(scales));
        }
        if (table.participation_bits != nullptr)
        {
            leases.push_back(std::move(participation));
        }
        if (table.material_ids != nullptr)
        {
            leases.push_back(std::move(materials));
        }
        outInstances = std::move(instances);
        outLeases    = std::move(leases);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

// The registration owns copies of every definition it read, so no input table
// is retained past enqueue.
ovspatialquery_enqueue_result_t submitExplicitRegistration(
    InstanceState* instance, ovspatialquery::detail::ExplicitPreparedRegistration registration) noexcept
{
    try
    {
        const std::vector<uint64_t>& definitionBytes = registration.definitionResidentByteCounts();
        std::vector<std::shared_ptr<ovspatialquery::detail::HostLease>> definitionLeases;
        definitionLeases.reserve(definitionBytes.size());
        ovspatialquery_api_status_t leaseStatus = OVSPATIALQUERY_API_SUCCESS;
        for (uint64_t bytes : definitionBytes)
        {
            std::shared_ptr<ovspatialquery::detail::HostLease> lease;
            leaseStatus = reserveSharedHostLease(instance, bytes, lease);
            if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
            {
                return {leaseStatus, OVSPATIALQUERY_INVALID_OP};
            }
            definitionLeases.push_back(std::move(lease));
        }
        // The meshes' query structures and the build scratch of the unbuilt ones
        // (ADR-0075).
        leaseStatus =
            ovspatialquery::detail::admitTriangleMeshStructures(instance->hostLedger, registration.triangleMeshes());
        if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return {leaseStatus, OVSPATIALQUERY_INVALID_OP};
        }
        std::shared_ptr<ovspatialquery::detail::HostLease> claimLease;
        leaseStatus = reserveSharedHostLease(instance, registration.claimResidentByteCount(), claimLease);
        if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return {leaseStatus, OVSPATIALQUERY_INVALID_OP};
        }
        std::shared_ptr<ovspatialquery::detail::HostLease> transientLease;
        leaseStatus = reserveSharedHostLease(instance, registration.transientResidentByteCount(), transientLease);
        if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return {leaseStatus, OVSPATIALQUERY_INVALID_OP};
        }
        std::lock_guard<std::mutex> lock(instance->explicitRuntimeMutex);
        if (instance->explicitRegistrationOwner == nullptr)
        {
            const ovspatialquery_api_status_t created =
                ovspatialquery::detail::ExplicitRegistrationOperationOwner::create(instance->publicHandle,
                                                                                   instance->explicitRegistrationOwner);
            if (created != OVSPATIALQUERY_API_SUCCESS)
            {
                return {created, OVSPATIALQUERY_INVALID_OP};
            }
        }
        const ovspatialquery::detail::ExplicitRegistrationAdmission admission =
            instance->explicitRegistrationOwner->enqueuePrepared(instance->explicitPreparationRegistry,
                                                                 std::move(registration), std::move(definitionLeases),
                                                                 std::move(claimLease), std::move(transientLease));
        return {admission.outcome, admission.operation.value};
    }
    catch (const std::bad_alloc&)
    {
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, OVSPATIALQUERY_INVALID_OP};
    }
    catch (...)
    {
        return {OVSPATIALQUERY_API_INTERNAL, OVSPATIALQUERY_INVALID_OP};
    }
}

// One 64-bit real column of a primitive definition table and the shape it
// must have: `rank` 1 with `trailing` 0 for one value per row, rank 2 with
// trailing 3 for one triple per row.
struct ExplicitPrimitiveColumn
{
    const ovspatialquery_input_array_t* source;
    const char*                         name;
    int32_t                             rank;
    int64_t                             trailing;
};

// The sphere, box, capsule, convex and triangle-mesh registrations share one
// admission shell: `build(table, prepared, arrays, diagnostic)` reads the table,
// through the input arrays it prepares into `arrays`, into owned definitions and
// may name what it refuses; the shell prepares and submits them and drops the arrays.
template <typename Table, typename Build>
ovspatialquery_enqueue_result_t admitDefinitionRegistration(ovspatialquery_instance_handle_t instanceHandle,
                                                            const Table* definitions, Build build,
                                                            std::string& diagnostic)
{
    // Columns are read before their statuses are checked; the first refused
    // layout keeps the diagnostic until the entry point reports it.
    clearInputLayoutDiagnostic();
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return {OVSPATIALQUERY_API_INVALID_HANDLE, OVSPATIALQUERY_INVALID_OP};
    }
    if (definitions == nullptr)
    {
        return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    }
    try
    {
        std::vector<ovspatialquery::detail::ExplicitDefinitionInput> prepared;
        std::vector<ovspatialquery::detail::PreparedInputArray>      arrays;
        ovspatialquery::detail::ExplicitPreparedRegistration         registration;
        ovspatialquery_api_status_t status = build(*definitions, prepared, arrays, diagnostic);
        std::string                 dropReport;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = instance->explicitPreparationRegistry->prepareRegistration(
                prepared, registration, &diagnostic, &dropReport, instance->hostLedger.total());
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        const ovspatialquery_enqueue_result_t result = submitExplicitRegistration(instance, std::move(registration));
        // A successful registration that dropped triangles without area names them (ADR-0073).
        if (result.status == OVSPATIALQUERY_API_SUCCESS)
        {
            diagnostic = std::move(dropReport);
        }
        return result;
    }
    catch (const std::bad_alloc&)
    {
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, OVSPATIALQUERY_INVALID_OP};
    }
    catch (...)
    {
        return {OVSPATIALQUERY_API_INTERNAL, OVSPATIALQUERY_INVALID_OP};
    }
}

// Every return of a registration entry point sets the calling thread's diagnostic,
// a success too: empty, or the triangles a triangle-mesh registration dropped.
template <typename Table, typename Build>
ovspatialquery_enqueue_result_t enqueueDefinitionRegistration(ovspatialquery_instance_handle_t instanceHandle,
                                                              const Table* definitions, Build build)
{
    std::string                           diagnostic;
    const ovspatialquery_enqueue_result_t result =
        admitDefinitionRegistration(instanceHandle, definitions, build, diagnostic);
    return sourcePublicEnqueueResult(result.status, result.op_id, diagnostic.empty() ? nullptr : diagnostic.c_str());
}

// Sphere, box and capsule tables: the definition-value column and real columns
// with matching row counts, one definition per row. Only the column list and
// the per-row definition differ, so the entry points pass those in.
template <size_t ColumnCount, typename Table, typename Columns, typename Definition>
ovspatialquery_enqueue_result_t enqueuePrimitiveRegistration(ovspatialquery_instance_handle_t            instanceHandle,
                                                             const Table*                                definitions,
                                                             ovspatialquery::detail::ExactGeometryFamily family,
                                                             Columns columns, Definition definition)
{
    return enqueueDefinitionRegistration(
        instanceHandle, definitions,
        [family, &columns, &definition](
            const Table& table, std::vector<ovspatialquery::detail::ExplicitDefinitionInput>& prepared,
            std::vector<ovspatialquery::detail::PreparedInputArray>&, std::string&) -> ovspatialquery_api_status_t
        {
            const std::array<ExplicitPrimitiveColumn, ColumnCount>              specs = columns(table);
            ovspatialquery::detail::PreparedInputArray                          values;
            std::array<ovspatialquery::detail::PreparedInputArray, ColumnCount> arrays;
            ovspatialquery_api_status_t                                         status =
                explicitPreparedArray(table.definition_values, "definition_values", kDLUInt, 64u, 1, 0, values);
            for (size_t column = 0u; status == OVSPATIALQUERY_API_SUCCESS && column < ColumnCount; ++column)
            {
                status = explicitPreparedArray(specs[column].source, specs[column].name, kDLFloat, 64u,
                                               specs[column].rank, specs[column].trailing, arrays[column]);
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            if (values.shape(0u) <= 0)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            for (const ovspatialquery::detail::PreparedInputArray& array : arrays)
            {
                if (array.shape(0u) != values.shape(0u))
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
            }
            const size_t count = static_cast<size_t>(values.shape(0u));
            prepared.reserve(count);
            for (size_t row = 0u; row < count; ++row)
            {
                prepared.push_back({{family, readPrimitive<uint64_t>(values, row)}, definition(arrays, row)});
            }
            return OVSPATIALQUERY_API_SUCCESS;
        });
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_sphere_registration(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_sphere_definition_table_t* definitions)
{
    using ovspatialquery::detail::PreparedInputArray;
    return enqueuePrimitiveRegistration<1u>(
        instanceHandle, definitions, ovspatialquery::detail::ExactGeometryFamily::sphere,
        [](const ovspatialquery_sphere_definition_table_t& table) {
            return std::array<ExplicitPrimitiveColumn, 1u>{ExplicitPrimitiveColumn{table.radii, "radii", 1, 0}};
        },
        [](const std::array<PreparedInputArray, 1u>& columns, size_t row)
        { return ovspatialquery::detail::ExplicitSphereDefinition{readPrimitive<double>(columns[0], row)}; });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_box_registration(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_box_definition_table_t* definitions)
{
    using ovspatialquery::detail::PreparedInputArray;
    return enqueuePrimitiveRegistration<1u>(
        instanceHandle, definitions, ovspatialquery::detail::ExactGeometryFamily::box,
        [](const ovspatialquery_box_definition_table_t& table) {
            return std::array<ExplicitPrimitiveColumn, 1u>{
                ExplicitPrimitiveColumn{table.half_extents, "half_extents", 2, 3}};
        },
        [](const std::array<PreparedInputArray, 1u>& columns, size_t row)
        {
            return ovspatialquery::detail::ExplicitBoxDefinition{{readPrimitive<double>(columns[0], row * 3u),
                                                                  readPrimitive<double>(columns[0], row * 3u + 1u),
                                                                  readPrimitive<double>(columns[0], row * 3u + 2u)}};
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_capsule_registration(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_capsule_definition_table_t* definitions)
{
    using ovspatialquery::detail::PreparedInputArray;
    return enqueuePrimitiveRegistration<2u>(
        instanceHandle, definitions, ovspatialquery::detail::ExactGeometryFamily::capsule,
        [](const ovspatialquery_capsule_definition_table_t& table)
        {
            return std::array<ExplicitPrimitiveColumn, 2u>{
                ExplicitPrimitiveColumn{table.radii, "radii", 1, 0},
                ExplicitPrimitiveColumn{table.half_segments, "half_segments", 1, 0}};
        },
        [](const std::array<PreparedInputArray, 2u>& columns, size_t row)
        {
            return ovspatialquery::detail::ExplicitCapsuleDefinition{readPrimitive<double>(columns[0], row),
                                                                     readPrimitive<double>(columns[1], row)};
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_convex_registration(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_convex_definition_table_t* definitions)
{
    return enqueueDefinitionRegistration(
        instanceHandle, definitions,
        [](const ovspatialquery_convex_definition_table_t&               table,
           std::vector<ovspatialquery::detail::ExplicitDefinitionInput>& prepared,
           std::vector<ovspatialquery::detail::PreparedInputArray>&      arrays,
           std::string&                                                  diagnostic) -> ovspatialquery_api_status_t
        {
            // One hull vertex limit and one enclosure tolerance cover every definition in the
            // call (ADR-0078); zero keeps today's exact hull and the 2^-20 tolerance.
            ovspatialquery::detail::ConvexHullCookingOptions options;
            const double                                     tolerance = table.enclosure_tolerance;
            if (tolerance != 0.0 && !(tolerance >= ovspatialquery::detail::kDefaultEnclosureTolerance &&
                                      tolerance <= ovspatialquery::detail::kMaximumEnclosureTolerance))
            {
                char text[128];
                std::snprintf(text, sizeof(text),
                              "convex registration: enclosure tolerance %.9g is not zero or between 2^-20 and 2^-16",
                              tolerance);
                diagnostic = text;
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            if (tolerance != 0.0)
                options.enclosureTolerance = tolerance;
            const uint32_t limit = table.hull_vertex_limit;
            if (limit != 0u && (limit < ovspatialquery::detail::kMinimumHullVertexLimit ||
                                limit > ovspatialquery::detail::kMaximumHullVertexLimit))
            {
                diagnostic = "convex registration: hull vertex limit " + std::to_string(limit) +
                             " is not zero or between 8 and 64";
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            options.vertexLimit = limit;
            arrays.resize(9u);
            const bool   topologyColumns[5] = {table.face_offsets != nullptr, table.face_counts != nullptr,
                                               table.face_vertex_offsets != nullptr, table.face_vertex_counts != nullptr,
                                               table.face_vertex_indices != nullptr};
            const size_t topologyColumnCount =
                static_cast<size_t>(std::count(std::begin(topologyColumns), std::end(topologyColumns), true));
            if (topologyColumnCount != 0u && topologyColumnCount != 5u)
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            const bool                        hasSourceTopology   = topologyColumnCount == 5u;
            const ovspatialquery_api_status_t requiredStatuses[4] = {
                explicitPreparedArray(table.definition_values, "definition_values", kDLUInt, 64u, 1, 0, arrays[0]),
                explicitPreparedArray(table.vertex_offsets, "vertex_offsets", kDLUInt, 64u, 1, 0, arrays[1]),
                explicitPreparedArray(table.vertex_counts, "vertex_counts", kDLUInt, 64u, 1, 0, arrays[2]),
                explicitPreparedArray(table.vertices, "vertices", kDLFloat, 32u, 2, 3, arrays[5])};
            for (const ovspatialquery_api_status_t status : requiredStatuses)
            {
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    return status;
                }
            }
            if (hasSourceTopology)
            {
                const ovspatialquery_api_status_t topologyStatuses[5] = {
                    explicitPreparedArray(table.face_offsets, "face_offsets", kDLUInt, 64u, 1, 0, arrays[3]),
                    explicitPreparedArray(table.face_counts, "face_counts", kDLUInt, 64u, 1, 0, arrays[4]),
                    explicitPreparedArray(table.face_vertex_offsets, "face_vertex_offsets", kDLUInt, 64u, 1, 0,
                                          arrays[6]),
                    explicitPreparedArray(table.face_vertex_counts, "face_vertex_counts", kDLUInt, 32u, 1, 0,
                                          arrays[7]),
                    explicitPreparedArray(table.face_vertex_indices, "face_vertex_indices", kDLUInt, 32u, 1, 0,
                                          arrays[8])};
                for (const ovspatialquery_api_status_t status : topologyStatuses)
                {
                    if (status != OVSPATIALQUERY_API_SUCCESS)
                    {
                        return status;
                    }
                }
            }
            const int64_t definitionCount = arrays[0].shape(0u);
            if (definitionCount <= 0 || arrays[1].shape(0u) != definitionCount ||
                arrays[2].shape(0u) != definitionCount ||
                (hasSourceTopology &&
                 (arrays[3].shape(0u) != definitionCount || arrays[4].shape(0u) != definitionCount)))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            const uint64_t vertexTotal = static_cast<uint64_t>(arrays[5].shape(0u));
            const uint64_t faceTotal   = hasSourceTopology ? static_cast<uint64_t>(arrays[6].shape(0u)) : 0u;
            const uint64_t indexTotal  = hasSourceTopology ? static_cast<uint64_t>(arrays[8].shape(0u)) : 0u;
            if (hasSourceTopology && arrays[7].shape(0u) != arrays[6].shape(0u))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            prepared.reserve(static_cast<size_t>(definitionCount));
            for (size_t row = 0u; row < static_cast<size_t>(definitionCount); ++row)
            {
                const uint64_t vertexOffset = readPrimitive<uint64_t>(arrays[1], row);
                const uint64_t vertexCount  = readPrimitive<uint64_t>(arrays[2], row);
                const uint64_t faceOffset   = hasSourceTopology ? readPrimitive<uint64_t>(arrays[3], row) : 0u;
                const uint64_t faceCount    = hasSourceTopology ? readPrimitive<uint64_t>(arrays[4], row) : 0u;
                if (vertexOffset > vertexTotal || vertexCount > vertexTotal - vertexOffset || faceOffset > faceTotal ||
                    faceCount > faceTotal - faceOffset)
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                ovspatialquery::detail::ExplicitConvexDefinition convex;
                convex.options  = options;
                convex.vertices = explicitReadVertices(arrays[5], static_cast<size_t>(vertexOffset) * 3u,
                                                       static_cast<size_t>(vertexCount) * 3u);
                convex.faceVertexCounts.reserve(static_cast<size_t>(faceCount));
                for (uint64_t face = 0u; face < faceCount; ++face)
                {
                    const size_t   faceRow    = static_cast<size_t>(faceOffset + face);
                    const uint64_t loopOffset = readPrimitive<uint64_t>(arrays[6], faceRow);
                    const uint32_t loopCount  = readPrimitive<uint32_t>(arrays[7], faceRow);
                    if (loopOffset > indexTotal || static_cast<uint64_t>(loopCount) > indexTotal - loopOffset)
                    {
                        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                    }
                    convex.faceVertexCounts.push_back(loopCount);
                    for (uint32_t corner = 0u; corner < loopCount; ++corner)
                    {
                        const uint32_t index =
                            readPrimitive<uint32_t>(arrays[8], static_cast<size_t>(loopOffset + corner));
                        if (index >= vertexCount)
                        {
                            return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                        }
                        convex.faceVertexIndices.push_back(index);
                    }
                }
                prepared.push_back(
                    {{ovspatialquery::detail::ExactGeometryFamily::convex, readPrimitive<uint64_t>(arrays[0], row)},
                     std::move(convex)});
            }
            return OVSPATIALQUERY_API_SUCCESS;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_triangle_mesh_registration(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_triangle_mesh_definition_table_t* definitions)
{
    return enqueueDefinitionRegistration(
        instanceHandle, definitions,
        [](const ovspatialquery_triangle_mesh_definition_table_t&        table,
           std::vector<ovspatialquery::detail::ExplicitDefinitionInput>& prepared,
           std::vector<ovspatialquery::detail::PreparedInputArray>& arrays, std::string&) -> ovspatialquery_api_status_t
        {
            arrays.resize(11u);
            const bool hasSourceFaces = table.source_face_indices != nullptr;
            if (hasSourceFaces != (table.source_face_corner_indices != nullptr))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            ovspatialquery_api_status_t status =
                explicitPreparedArray(table.definition_values, "definition_values", kDLUInt, 64u, 1, 0, arrays[0]);
            const ovspatialquery_api_status_t required[6] = {
                explicitPreparedArray(table.vertex_offsets, "vertex_offsets", kDLUInt, 64u, 1, 0, arrays[1]),
                explicitPreparedArray(table.vertex_counts, "vertex_counts", kDLUInt, 64u, 1, 0, arrays[2]),
                explicitPreparedArray(table.triangle_offsets, "triangle_offsets", kDLUInt, 64u, 1, 0, arrays[3]),
                explicitPreparedArray(table.triangle_counts, "triangle_counts", kDLUInt, 64u, 1, 0, arrays[4]),
                explicitPreparedArray(table.vertices, "vertices", kDLFloat, 32u, 2, 3, arrays[5]),
                explicitPreparedArray(table.triangle_indices, "triangle_indices", kDLUInt, 32u, 2, 3, arrays[6])};
            for (const ovspatialquery_api_status_t item : required)
            {
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    status = item;
                }
            }
            const bool hasDoubleSided = table.double_sided != nullptr;
            if (status == OVSPATIALQUERY_API_SUCCESS && hasDoubleSided)
            {
                status = explicitPreparedArray(table.double_sided, "double_sided", kDLUInt, 8u, 1, 0, arrays[7]);
            }
            const bool hasFaceMaterials = table.source_face_material_ids != nullptr;
            if (status == OVSPATIALQUERY_API_SUCCESS && hasFaceMaterials)
            {
                status = explicitPreparedArray(table.source_face_material_ids, "source_face_material_ids", kDLUInt, 64u,
                                               1, 0, arrays[8]);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && hasSourceFaces)
            {
                status = explicitPreparedArray(table.source_face_indices, "source_face_indices", kDLUInt, 64u, 1, 0,
                                               arrays[9]);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && hasSourceFaces)
            {
                status = explicitPreparedArray(table.source_face_corner_indices, "source_face_corner_indices", kDLUInt,
                                               32u, 2, 3, arrays[10]);
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            const int64_t definitionCount = arrays[0].shape(0u);
            if (definitionCount <= 0 || arrays[1].shape(0u) != definitionCount ||
                arrays[2].shape(0u) != definitionCount || arrays[3].shape(0u) != definitionCount ||
                arrays[4].shape(0u) != definitionCount || (hasDoubleSided && arrays[7].shape(0u) != definitionCount) ||
                (hasFaceMaterials && arrays[8].shape(0u) != arrays[6].shape(0u)) ||
                (hasSourceFaces &&
                 (arrays[9].shape(0u) != arrays[6].shape(0u) || arrays[10].shape(0u) != arrays[6].shape(0u))))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            const uint64_t vertexTotal   = static_cast<uint64_t>(arrays[5].shape(0u));
            const uint64_t triangleTotal = static_cast<uint64_t>(arrays[6].shape(0u));
            prepared.reserve(static_cast<size_t>(definitionCount));
            for (size_t row = 0u; row < static_cast<size_t>(definitionCount); ++row)
            {
                const uint64_t vertexOffset   = readPrimitive<uint64_t>(arrays[1], row);
                const uint64_t vertexCount    = readPrimitive<uint64_t>(arrays[2], row);
                const uint64_t triangleOffset = readPrimitive<uint64_t>(arrays[3], row);
                const uint64_t triangleCount  = readPrimitive<uint64_t>(arrays[4], row);
                if (vertexOffset > vertexTotal || vertexCount > vertexTotal - vertexOffset ||
                    triangleOffset > triangleTotal || triangleCount > triangleTotal - triangleOffset)
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                ovspatialquery::detail::ExplicitTriangleMeshDefinition mesh;
                mesh.doubleSided = hasDoubleSided && readPrimitive<uint8_t>(arrays[7], row) != 0u;
                mesh.vertices    = explicitReadVertices(arrays[5], static_cast<size_t>(vertexOffset) * 3u,
                                                        static_cast<size_t>(vertexCount) * 3u);
                mesh.triangleVertexIndices.reserve(static_cast<size_t>(triangleCount) * 3u);
                for (uint64_t triangle = 0u; triangle < triangleCount; ++triangle)
                {
                    for (uint32_t corner = 0u; corner < 3u; ++corner)
                    {
                        mesh.triangleVertexIndices.push_back(readPrimitive<uint32_t>(
                            arrays[6], static_cast<size_t>((triangleOffset + triangle) * 3u + corner)));
                    }
                }
                if (hasSourceFaces)
                {
                    mesh.hasSourceFaces = true;
                    mesh.sourceFaces.reserve(static_cast<size_t>(triangleCount));
                    mesh.sourceCorners.reserve(static_cast<size_t>(triangleCount) * 3u);
                    for (uint64_t triangle = 0u; triangle < triangleCount; ++triangle)
                    {
                        const size_t source = static_cast<size_t>(triangleOffset + triangle);
                        mesh.sourceFaces.push_back(readPrimitive<uint64_t>(arrays[9], source));
                        for (uint32_t corner = 0u; corner < 3u; ++corner)
                        {
                            mesh.sourceCorners.push_back(readPrimitive<uint32_t>(arrays[10], source * 3u + corner));
                        }
                    }
                }
                if (hasFaceMaterials)
                {
                    // A table of zeros reports the instance material everywhere, as no table does.
                    std::vector<uint64_t> faceMaterials(static_cast<size_t>(triangleCount));
                    bool                  anyMaterial = false;
                    for (uint64_t triangle = 0u; triangle < triangleCount; ++triangle)
                    {
                        faceMaterials[static_cast<size_t>(triangle)] =
                            readPrimitive<uint64_t>(arrays[8], static_cast<size_t>(triangleOffset + triangle));
                        anyMaterial = anyMaterial || faceMaterials[static_cast<size_t>(triangle)] != 0u;
                    }
                    if (anyMaterial)
                    {
                        mesh.sourceFaceMaterialIds =
                            std::make_shared<const std::vector<uint64_t>>(std::move(faceMaterials));
                    }
                }
                prepared.push_back({{ovspatialquery::detail::ExactGeometryFamily::triangleMesh,
                                     readPrimitive<uint64_t>(arrays[0], row)},
                                    std::move(mesh)});
            }
            return OVSPATIALQUERY_API_SUCCESS;
        });
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_explicit_population(
    ovspatialquery_instance_handle_t instanceHandle, const ovspatialquery_population_desc_t* desc,
    ovspatialquery_population_handle_t* outPopulation)
{
    if (outPopulation == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outPopulation             = OVSPATIALQUERY_INVALID_POPULATION;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (desc == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    ovspatialquery::detail::PopulationSourceBinding binding = instance->explicitPreparationRegistry->sourceBinding();
    try
    {
        binding.deviceState = std::make_shared<ovspatialquery::detail::DevicePreparedState>();
    }
    catch (const std::bad_alloc&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    const ovspatialquery_api_status_t status =
        createEmptyPopulationForSource(instanceHandle, std::move(binding), desc->retained_ordinal_capacity, 0u,
                                       *outPopulation, desc->pending_publication_capacity);
    return apiResult(status);
}

namespace
{

// `diagnostic` names a refusal more precisely than its status; the entry point
// sets it, or the status message, as the calling thread's diagnostic.
ovspatialquery_enqueue_result_t admitExplicitPopulationSet(ovspatialquery_instance_handle_t   instanceHandle,
                                                           ovspatialquery_population_handle_t population,
                                                           const ovspatialquery_explicit_population_set_t* set,
                                                           const char*&                                    diagnostic)
{
    clearInputLayoutDiagnostic();
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return {OVSPATIALQUERY_API_INVALID_HANDLE, OVSPATIALQUERY_INVALID_OP};
    }
    if (set == nullptr || set->target_ordinal < 0)
    {
        return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    }
    try
    {
        ovspatialquery::detail::PopulationKey           key{};
        ovspatialquery::detail::PopulationSourceBinding source{};
        ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, source);
        const ovspatialquery::detail::PopulationSourceBinding expected =
            instance->explicitPreparationRegistry->sourceBinding();
        if (status != OVSPATIALQUERY_API_SUCCESS || source.sourceState != expected.sourceState)
        {
            return {OVSPATIALQUERY_API_INVALID_HANDLE, OVSPATIALQUERY_INVALID_OP};
        }
        // A slot capacity sizes device tables; a host-prepared set has none to size.
        const bool setOnDevice = set->dynamic_execution != nullptr &&
                                 set->dynamic_execution->device.device_type == kDLCUDA &&
                                 set->dynamic_execution->preparation != OVSPATIALQUERY_PREPARE_ON_HOST;
        if (set->slot_capacity != 0u && !setOnDevice)
        {
            return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
        }
        if (set->dynamic_execution != nullptr && set->dynamic_execution->device.device_type == kDLCUDA)
        {
            if (const char* unavailable = ovspatialquery::detail::cudaBackendUnavailableReason(
                    set->dynamic_execution->device.device_id,
                    instance->cudaModulePins(set->dynamic_execution->device.device_id)))
            {
                diagnostic = unavailable;
                return {OVSPATIALQUERY_API_NOT_SUPPORTED, OVSPATIALQUERY_INVALID_OP};
            }
            std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture> capture;
            const ovspatialquery::detail::ExplicitCudaCaptureContext captureContext{instanceHandle, source.deviceState};
            status = ovspatialquery::detail::prepareExplicitCudaCapture(
                set->instances, nullptr, *set->dynamic_execution, 0, set->target_ordinal, false, false,
                set->slot_capacity, instance->explicitPreparationRegistry, captureContext, capture);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return {status, OVSPATIALQUERY_INVALID_OP};
            }
            ovspatialquery::detail::PreparedPopulationInput input;
            status = ovspatialquery::detail::PreparedPopulationInput::prepareCopiedSet(
                set->target_ordinal, ovspatialquery::detail::ObjectTable(), input);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                input.restrictToUnreadyPopulation();
                input.enforceObjectValueNoReuse();
                input.adoptCudaCapture(std::move(capture));
            }
            ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = submitPopulationInput(instance, key, std::move(input), operation);
            }
            return {status, status == OVSPATIALQUERY_API_SUCCESS ? operation : OVSPATIALQUERY_INVALID_OP};
        }
        std::vector<ovspatialquery::detail::ExplicitInstanceInput> instances;
        std::vector<ovspatialquery::detail::PreparedInputArray>    leases;
        status = prepareExplicitInstances(set->instances, set->dynamic_execution, false, false, instances, leases);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        ovspatialquery::detail::ExplicitPreparedUpdate prepared;
        status = instance->explicitPreparationRegistry->prepareSet(instances, prepared);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        ovspatialquery::detail::PreparedPopulationInput input;
        status = ovspatialquery::detail::PreparedPopulationInput::prepareCopiedSet(set->target_ordinal,
                                                                                   prepared.releaseObjects(), input);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        input.restrictToUnreadyPopulation();
        input.enforceObjectValueNoReuse();
        for (ovspatialquery::detail::PreparedInputArray& lease : leases)
        {
            input.adoptInputLease(std::move(lease));
        }
        ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
        status                           = submitPopulationInput(instance, key, std::move(input), operation);
        return {status, status == OVSPATIALQUERY_API_SUCCESS ? operation : OVSPATIALQUERY_INVALID_OP};
    }
    catch (const std::bad_alloc&)
    {
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, OVSPATIALQUERY_INVALID_OP};
    }
    catch (...)
    {
        return {OVSPATIALQUERY_API_INTERNAL, OVSPATIALQUERY_INVALID_OP};
    }
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_set(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_population_handle_t population,
    const ovspatialquery_explicit_population_set_t* set)
{
    const char*                           diagnostic = nullptr;
    const ovspatialquery_enqueue_result_t result =
        admitExplicitPopulationSet(instanceHandle, population, set, diagnostic);
    return sourcePublicEnqueueResult(result.status, result.op_id, diagnostic);
}

namespace
{

// `diagnostic` names a refusal more precisely than its status; the entry point
// sets it, or the status message, as the calling thread's diagnostic.
ovspatialquery_enqueue_result_t admitExplicitPopulationUpdate(ovspatialquery_instance_handle_t   instanceHandle,
                                                              ovspatialquery_population_handle_t population,
                                                              const ovspatialquery_explicit_population_update_t* update,
                                                              const char*& diagnostic)
{
    clearInputLayoutDiagnostic();
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return {OVSPATIALQUERY_API_INVALID_HANDLE, OVSPATIALQUERY_INVALID_OP};
    }
    if (update == nullptr || update->base_ordinal < 0 || update->removed_object_values == nullptr)
    {
        return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    }
    if (update->target_ordinal <= update->base_ordinal)
    {
        return {OVSPATIALQUERY_API_STALE_ORDINAL, OVSPATIALQUERY_INVALID_OP};
    }
    if (update->kind != OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS &&
        update->kind != OVSPATIALQUERY_EXPLICIT_UPDATE_POSES &&
        update->kind != OVSPATIALQUERY_EXPLICIT_UPDATE_POSES_AND_BITS)
    {
        return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
    }
    const bool posesOnly = update->kind != OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS;
    const bool rowBits   = update->kind == OVSPATIALQUERY_EXPLICIT_UPDATE_POSES_AND_BITS;
    try
    {
        ovspatialquery::detail::PopulationKey           key{};
        ovspatialquery::detail::PopulationSourceBinding source{};
        ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, key, source);
        const ovspatialquery::detail::PopulationSourceBinding expected =
            instance->explicitPreparationRegistry->sourceBinding();
        if (status != OVSPATIALQUERY_API_SUCCESS || source.sourceState != expected.sourceState)
        {
            return {OVSPATIALQUERY_API_INVALID_HANDLE, OVSPATIALQUERY_INVALID_OP};
        }
        // A version prepared on the device stays there: a host-prepared
        // successor would need the head on the host.
        const bool deviceHead = source.deviceState != nullptr && source.deviceState->current() != nullptr;
        const bool onDevice   = update->dynamic_execution != nullptr &&
                              update->dynamic_execution->device.device_type == kDLCUDA &&
                              update->dynamic_execution->preparation != OVSPATIALQUERY_PREPARE_ON_HOST;
        if (deviceHead && !onDevice)
        {
            return {OVSPATIALQUERY_API_NOT_SUPPORTED, OVSPATIALQUERY_INVALID_OP};
        }
        if (onDevice && !deviceHead)
        {
            // The first complete set defines the membership an update is checked
            // against, so an update waits for it (BUSY) rather than chaining on it.
            bool pending = false;
            if (instance->populations.getUpdatePending(key, pending) == OVSPATIALQUERY_API_SUCCESS && pending)
            {
                return {OVSPATIALQUERY_API_BUSY, OVSPATIALQUERY_INVALID_OP};
            }
        }
        if (update->dynamic_execution != nullptr && update->dynamic_execution->device.device_type == kDLCUDA)
        {
            if (const char* unavailable = ovspatialquery::detail::cudaBackendUnavailableReason(
                    update->dynamic_execution->device.device_id,
                    instance->cudaModulePins(update->dynamic_execution->device.device_id)))
            {
                diagnostic = unavailable;
                return {OVSPATIALQUERY_API_NOT_SUPPORTED, OVSPATIALQUERY_INVALID_OP};
            }
            std::shared_ptr<ovspatialquery::detail::CudaPopulationCapture> capture;
            const ovspatialquery::detail::ExplicitCudaCaptureContext captureContext{instanceHandle, source.deviceState};
            status = ovspatialquery::detail::prepareExplicitCudaCapture(
                update->changed_instances, update->removed_object_values, *update->dynamic_execution,
                update->base_ordinal, update->target_ordinal, posesOnly, rowBits, 0u,
                instance->explicitPreparationRegistry, captureContext, capture);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return {status, OVSPATIALQUERY_INVALID_OP};
            }
            ovspatialquery::detail::PreparedPopulationInput input;
            status = ovspatialquery::detail::PreparedPopulationInput::prepareCopiedCompleteUpdate(
                update->base_ordinal, update->target_ordinal, {}, input);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                input.enforceObjectValueNoReuse();
                input.adoptCudaCapture(std::move(capture));
                if (onDevice)
                {
                    input.chainPendingPublications(); // ADR-0021: base may be the newest admitted update
                }
            }
            ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = submitPopulationInput(instance, key, std::move(input), operation);
            }
            return {status, status == OVSPATIALQUERY_API_SUCCESS ? operation : OVSPATIALQUERY_INVALID_OP};
        }
        std::vector<ovspatialquery::detail::ExplicitInstanceInput> changed;
        std::vector<ovspatialquery::detail::PreparedInputArray>    leases;
        status = prepareExplicitInstances(update->changed_instances, update->dynamic_execution, posesOnly, rowBits,
                                          changed, leases);
        ovspatialquery::detail::PreparedInputArray removed;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = explicitPreparedArray(update->removed_object_values, "removed_object_values", kDLUInt, 64u, 1, 0,
                                           removed);
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        std::vector<uint64_t> removedValues;
        removedValues.reserve(static_cast<size_t>(removed.shape(0u)));
        for (size_t row = 0u; row < static_cast<size_t>(removed.shape(0u)); ++row)
        {
            removedValues.push_back(readPrimitive<uint64_t>(removed, row));
        }
        if (posesOnly && !removedValues.empty())
        {
            return {OVSPATIALQUERY_API_INVALID_ARGUMENT, OVSPATIALQUERY_INVALID_OP};
        }
        ovspatialquery::detail::GenerationLease base;
        status = instance->populations.pinExact(key, update->base_ordinal, base);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        if (posesOnly)
        {
            status = ovspatialquery::detail::completePoseRows(base, update->changed_instances.uniform_scales != nullptr,
                                                              rowBits, changed);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return {status, OVSPATIALQUERY_INVALID_OP};
            }
        }
        ovspatialquery::detail::ExplicitPreparedUpdate prepared;
        status = instance->explicitPreparationRegistry->prepareUpdate(base, update->base_ordinal, changed,
                                                                      removedValues, prepared);
        base.reset();
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        ovspatialquery::detail::PreparedPopulationInput input;
        const std::optional<uint64_t>                   exactGeometryCharge = prepared.exactGeometryCharge();
        status = ovspatialquery::detail::PreparedPopulationInput::prepareCopiedCompleteUpdate(
            update->base_ordinal, update->target_ordinal, prepared.releaseObjects(), input, exactGeometryCharge,
            prepared.releaseChanges());
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return {status, OVSPATIALQUERY_INVALID_OP};
        }
        input.enforceObjectValueNoReuse();
        leases.push_back(std::move(removed));
        for (ovspatialquery::detail::PreparedInputArray& lease : leases)
        {
            input.adoptInputLease(std::move(lease));
        }
        ovspatialquery_op_id_t operation = OVSPATIALQUERY_INVALID_OP;
        status                           = submitPopulationInput(instance, key, std::move(input), operation);
        return {status, status == OVSPATIALQUERY_API_SUCCESS ? operation : OVSPATIALQUERY_INVALID_OP};
    }
    catch (const std::bad_alloc&)
    {
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, OVSPATIALQUERY_INVALID_OP};
    }
    catch (...)
    {
        return {OVSPATIALQUERY_API_INTERNAL, OVSPATIALQUERY_INVALID_OP};
    }
}

} // namespace

extern "C" OVSPATIALQUERY_API ovspatialquery_enqueue_result_t ovspatialquery_enqueue_explicit_population_update(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_population_handle_t population,
    const ovspatialquery_explicit_population_update_t* update)
{
    const char*                           diagnostic = nullptr;
    const ovspatialquery_enqueue_result_t result =
        admitExplicitPopulationUpdate(instanceHandle, population, update, diagnostic);
    return sourcePublicEnqueueResult(result.status, result.op_id, diagnostic);
}

extern "C" OVSPATIALQUERY_API ovspatialquery_result_t ovspatialquery_create_explicit_view(
    ovspatialquery_instance_handle_t instanceHandle, ovspatialquery_population_handle_t population,
    const ovspatialquery_explicit_view_desc_t* desc, ovspatialquery_view_handle_t* outView)
{
    clearInputLayoutDiagnostic();
    if (outView == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    *outView                   = OVSPATIALQUERY_INVALID_VIEW;
    InstanceReference instance = instanceState(instanceHandle);
    if (instance == nullptr)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
    }
    if (desc == nullptr || desc->object_ids == nullptr || desc->object_ids->ownership != 1u)
    {
        return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
    }
    try
    {
        ovspatialquery::detail::PopulationKey           parent{};
        ovspatialquery::detail::PopulationSourceBinding source{};
        ovspatialquery_api_status_t status = instance->handles.resolvePopulationSource(population, parent, source);
        const ovspatialquery::detail::PopulationSourceBinding expected =
            instance->explicitPreparationRegistry->sourceBinding();
        if (status != OVSPATIALQUERY_API_SUCCESS || source.sourceState != expected.sourceState)
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_HANDLE);
        }
        if (source.deviceState != nullptr && source.deviceState->current() != nullptr)
        {
            return apiResult(OVSPATIALQUERY_API_NOT_SUPPORTED); // no host objects to select from (ADR-0024)
        }
        ovspatialquery::detail::PreparedInputArray ids;
        status = explicitPreparedArray(desc->object_ids, "object_ids", kDLUInt, 64u, 1, 0, ids);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            const ovspatialquery_result_t refused = apiResult(status, inputLayoutDiagnostic(status));
            clearInputLayoutDiagnostic();
            return refused;
        }
        std::vector<uint64_t> values;
        values.reserve(static_cast<size_t>(ids.shape(0u)));
        for (size_t row = 0u; row < static_cast<size_t>(ids.shape(0u)); ++row)
        {
            values.push_back(readPrimitive<uint64_t>(ids, row));
        }
        std::sort(values.begin(), values.end());
        if ((!values.empty() && values.front() == 0u) ||
            std::adjacent_find(values.begin(), values.end()) != values.end())
        {
            return apiResult(OVSPATIALQUERY_API_INVALID_ARGUMENT);
        }
        uint64_t charge = 512u;
        if (values.size() > (std::numeric_limits<uint64_t>::max() - charge) / sizeof(uint64_t))
        {
            return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
        }
        charge += static_cast<uint64_t>(values.size()) * sizeof(uint64_t);
        std::shared_ptr<ovspatialquery::detail::HostLease> lease;
        const ovspatialquery_api_status_t leaseStatus = reserveSharedHostLease(instance, charge, lease);
        if (leaseStatus != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(leaseStatus);
        }
        const std::unique_lock<std::mutex> completionLock = instance->acquirePopulationCompletion();
        status = instance->handles.createView(parent, std::move(values), *outView);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return apiResult(status);
        }
        try
        {
            instance->viewCharges.emplace(*outView, std::move(lease));
        }
        catch (...)
        {
            (void)instance->handles.invalidateView(*outView);
            *outView = OVSPATIALQUERY_INVALID_VIEW;
            throw;
        }
        return apiResult(OVSPATIALQUERY_API_SUCCESS);
    }
    catch (const std::bad_alloc&)
    {
        return apiResult(OVSPATIALQUERY_API_OUT_OF_MEMORY);
    }
    catch (...)
    {
        return apiResult(OVSPATIALQUERY_API_INTERNAL);
    }
}
