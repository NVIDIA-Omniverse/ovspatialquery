/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_TEST_PUBLIC_API_INPUTS_H
#define OVSPATIALQUERY_TEST_PUBLIC_API_INPUTS_H

// Helpers shared by native tests of the public C API: target builders,
// participation bits, an input array that owns its shape and strides, explicit instance rows and
// populations, a wait-and-destroy for operations that must complete, a
// rejected-admission check, sphere registration and host usage.
// Failures report the line in this header and exit nonzero.

#include "test_failure.h"

#include <ovspatialquery/ovspatialquery.h>
#include <ovspatialquery/ovspatialquery_explicit.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <vector>

inline ovspatialquery_target_ref_t populationRef(ovspatialquery_population_handle_t population)
{
    ovspatialquery_target_ref_t target{};
    target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
    target.handle.population = population;
    return target;
}

inline ovspatialquery_target_ref_t viewRef(ovspatialquery_view_handle_t view)
{
    ovspatialquery_target_ref_t target{};
    target.kind        = OVSPATIALQUERY_TARGET_VIEW;
    target.handle.view = view;
    return target;
}

inline ovspatialquery_query_target_t populationTarget(ovspatialquery_population_handle_t population)
{
    ovspatialquery_query_target_t target{};
    target.target       = populationRef(population);
    target.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    return target;
}

inline ovspatialquery_query_target_t viewTarget(ovspatialquery_view_handle_t view)
{
    ovspatialquery_query_target_t target{};
    target.target       = viewRef(view);
    target.ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    return target;
}

// Participation bits with `low` in bits 0-63 and `high` in bits 64-127.
inline ovspatialquery_participation_bits_t participationBits(uint64_t low, uint64_t high = 0u) { return {{low, high}}; }

// A copied-on-submission input array whose shape and strides live inside the
// object, so it must stay put while the library reads it. Rows are host
// memory unless the caller names another device. A zero-row array carries
// null data.
struct InputArray
{
    InputArray()                             = default;
    InputArray(const InputArray&)            = delete;
    InputArray& operator=(const InputArray&) = delete;

    template <typename Value>
    InputArray(const Value* values, int64_t count, uint8_t code, DLDevice device = {kDLCPU, 0})
    {
        set(const_cast<Value*>(values), count, code, device);
    }

    template <typename Value>
    InputArray(const Value* values, int64_t rows, int64_t columns, uint8_t code, DLDevice device = {kDLCPU, 0})
    {
        set(const_cast<Value*>(values), rows, columns, code, device);
    }

    template <typename Value>
    void set(Value* values, int64_t count, uint8_t code, DLDevice device = {kDLCPU, 0})
    {
        static_assert(std::is_arithmetic<Value>::value, "pass element pointers");
        shape   = {count, 0};
        strides = {1, 0};
        fill(values, 1, code, static_cast<uint8_t>(sizeof(Value) * 8u), device);
    }

    template <typename Value>
    void set(Value* values, int64_t rows, int64_t columns, uint8_t code, DLDevice device = {kDLCPU, 0})
    {
        static_assert(std::is_arithmetic<Value>::value, "pass element pointers");
        shape   = {rows, columns};
        strides = {columns, 1};
        fill(values, 2, code, static_cast<uint8_t>(sizeof(Value) * 8u), device);
    }

    void fill(void* values, int32_t ndim, uint8_t code, uint8_t bits, DLDevice device)
    {
        value                = {};
        value.tensor.data    = shape[0] == 0 ? nullptr : values;
        value.tensor.device  = device;
        value.tensor.ndim    = ndim;
        value.tensor.dtype   = {code, bits, 1u};
        value.tensor.shape   = shape.data();
        value.tensor.strides = strides.data();
        value.ownership      = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    }

    std::array<int64_t, 2u>      shape{};
    std::array<int64_t, 2u>      strides{};
    ovspatialquery_input_array_t value{};
};

// Explicit-instance rows added one object at a time, each with an identity
// rotation. table() binds the columns, so call it after the last add() and
// keep the rows alive while the library reads them. Zero rows publish an
// ordinal bump with no objects.
struct InstanceRows
{
    void add(uint64_t object, int32_t type, uint64_t definition, double x, double y, double z)
    {
        objects.push_back(object);
        types.push_back(type);
        geometry.push_back(definition);
        translations.insert(translations.end(), {x, y, z});
        rotations.insert(rotations.end(), {0.0f, 0.0f, 0.0f, 1.0f});
    }

    ovspatialquery_explicit_instance_table_t table()
    {
        const int64_t count = static_cast<int64_t>(objects.size());
        objectArray.set(objects.data(), count, kDLUInt);
        typeArray.set(types.data(), count, kDLInt);
        geometryArray.set(geometry.data(), count, kDLUInt);
        translationArray.set(translations.data(), count, 3, kDLFloat);
        rotationArray.set(rotations.data(), count, 4, kDLFloat);
        if (!materials.empty())
        {
            materialArray.set(materials.data(), count, kDLUInt);
        }
        return {&objectArray.value,
                &typeArray.value,
                &geometryArray.value,
                &translationArray.value,
                &rotationArray.value,
                nullptr,
                nullptr,
                materials.empty() ? nullptr : &materialArray.value};
    }

    std::vector<uint64_t> objects;
    std::vector<int32_t>  types;
    std::vector<uint64_t> geometry;
    std::vector<double>   translations;
    std::vector<float>    rotations;
    // Optional: one material per row when nonempty (ADR-0066).
    std::vector<uint64_t> materials;
    InputArray            objectArray;
    InputArray            typeArray;
    InputArray            geometryArray;
    InputArray            translationArray;
    InputArray            rotationArray;
    InputArray            materialArray;
};

// An explicit population with no head yet and one pending publication.
inline ovspatialquery_population_handle_t createExplicitPopulation(ovspatialquery_instance_handle_t instance,
                                                                   uint32_t retainedOrdinalCapacity)
{
    const ovspatialquery_population_desc_t desc{retainedOrdinalCapacity, 1u};
    ovspatialquery_population_handle_t     population = OVSPATIALQUERY_INVALID_POPULATION;
    REQUIRE(ovspatialquery_create_explicit_population(instance, &desc, &population).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(population != OVSPATIALQUERY_INVALID_POPULATION);
    return population;
}

// Waits for an admitted operation that must complete, then destroys it.
inline void waitAndDestroy(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t admission)
{
    if (admission.status != OVSPATIALQUERY_API_SUCCESS)
    {
        std::fprintf(stderr, "operation admission returned %u\n", static_cast<unsigned>(admission.status));
    }
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(admission.op_id != OVSPATIALQUERY_INVALID_OP);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(admission.op_id == OVSPATIALQUERY_INVALID_OP);
}

// An admission refused with `status`, which leaves no operation behind.
inline void requireRejected(ovspatialquery_enqueue_result_t admission, ovspatialquery_api_status_t status)
{
    if (admission.status != status)
    {
        std::fprintf(stderr, "rejection status %u, expected %u\n", static_cast<unsigned>(admission.status),
                     static_cast<unsigned>(status));
    }
    REQUIRE(admission.status == status);
    REQUIRE(admission.op_id == OVSPATIALQUERY_INVALID_OP);
}

// Registers one sphere definition and waits for the registration.
inline void registerSphere(ovspatialquery_instance_handle_t instance, uint64_t value, double radius)
{
    InputArray                                     values(&value, 1, kDLUInt);
    InputArray                                     radii(&radius, 1, kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&values.value, &radii.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
}

// Two polygons the caller triangulated, double-sided in the plane x = 5: a quad, source
// face 7, split into corners {0, 1, 2} and {0, 2, 3}, and a pentagon, source face 3,
// split not as a fan into {0, 1, 4}, {1, 2, 3} and {1, 3, 4}. A ray along +X through
// kPolygonRayYz[t] (y, z) crosses submitted triangle t only. table() registers the
// source faces and corners unless `withSourceFaces` is false; the quad carries
// material 2000 and the pentagon none unless the caller changes `materials`.
struct PolygonMesh
{
    static constexpr uint32_t kTriangleCount = 5u;

    ovspatialquery_triangle_mesh_definition_table_t table(bool withSourceFaces = true)
    {
        valueArray.set(&value, 1, kDLUInt);
        offsetArray.set(&zero, 1, kDLUInt);
        vertexCountArray.set(&vertexCount, 1, kDLUInt);
        triangleCountArray.set(&triangleCount, 1, kDLUInt);
        vertexArray.set(vertices, 9, 3, kDLFloat);
        triangleArray.set(triangles, kTriangleCount, 3, kDLUInt);
        sidednessArray.set(&doubleSided, 1, kDLUInt);
        materialArray.set(materials, kTriangleCount, kDLUInt);
        faceArray.set(faces, kTriangleCount, kDLUInt);
        cornerArray.set(&corners[0][0], kTriangleCount, 3, kDLUInt);
        return {&valueArray.value,
                &offsetArray.value,
                &vertexCountArray.value,
                &offsetArray.value,
                &triangleCountArray.value,
                &vertexArray.value,
                &triangleArray.value,
                &sidednessArray.value,
                &materialArray.value,
                withSourceFaces ? &faceArray.value : nullptr,
                withSourceFaces ? &cornerArray.value : nullptr};
    }

    uint64_t value         = 0u;
    uint64_t zero          = 0u;
    uint64_t vertexCount   = 9u;
    uint64_t triangleCount = kTriangleCount;
    uint8_t  doubleSided   = 1u;
    // Quad corners 0-3 are vertices 0-3; pentagon corners 0-4 are vertices 4-8.
    float    vertices[27] = {5.0f,  -3.0f, -1.0f, 5.0f,  -1.0f, -1.0f, 5.0f, -1.0f, 1.0f, 5.0f, -3.0f, 1.0f, 5.0f, 1.0f,
                             -1.0f, 5.0f,  3.0f,  -1.0f, 5.0f,  4.0f,  0.0f, 5.0f,  3.0f, 1.0f, 5.0f,  1.0f, 1.0f};
    uint32_t triangles[kTriangleCount * 3u] = {0u, 1u, 2u, 0u, 2u, 3u, 4u, 5u, 8u, 5u, 6u, 7u, 5u, 7u, 8u};
    uint64_t materials[kTriangleCount]      = {2000u, 2000u, 0u, 0u, 0u};
    uint64_t faces[kTriangleCount]          = {7u, 7u, 3u, 3u, 3u};
    uint32_t corners[kTriangleCount][3]     = {{0u, 1u, 2u}, {0u, 2u, 3u}, {0u, 1u, 4u}, {1u, 2u, 3u}, {1u, 3u, 4u}};

    InputArray valueArray;
    InputArray offsetArray;
    InputArray vertexCountArray;
    InputArray triangleCountArray;
    InputArray vertexArray;
    InputArray triangleArray;
    InputArray sidednessArray;
    InputArray materialArray;
    InputArray faceArray;
    InputArray cornerArray;
};

constexpr double kPolygonRayYz[PolygonMesh::kTriangleCount][2] = {
    {-1.5, -0.5}, {-2.5, 0.5}, {1.5, -0.5}, {3.4, 0.0}, {2.6, 0.5}};

// The instance's current host memory usage.
inline uint64_t hostUsage(ovspatialquery_instance_handle_t instance)
{
    uint64_t bytes = UINT64_MAX;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, {kDLCPU, 0}, &bytes).status ==
            OVSPATIALQUERY_API_SUCCESS);
    return bytes;
}

#endif
