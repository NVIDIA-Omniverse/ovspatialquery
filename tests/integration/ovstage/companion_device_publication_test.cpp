/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ovstage publications prepared on the device (ADR-0047), against the real Ovstage runtime
// and a real CUDA device: a device set, poses-only updates through a parent and a prim,
// membership changes caught by the host comparison (an added, a removed and a deleted prim,
// and a definition edit), independent cube scale changes, exact Capsule segment
// containment and anisotropic mesh reshape recovery, chained updates with a CUDA plan selecting the admitted one, the
// authored world matrix of prims without a Fabric one, and the side rules. On a runtime without the placement extension
// (Ovstage 0.2), or without CUDA, the device calls must return NOT_SUPPORTED and the host calls keep working.
// First, while nothing else in the process has used CUDA: every CUDA user of an instance shares one module per device.

#define NOMINMAX
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include <ovstage/ovstage.h>
#include <ovstage/ovstage_api/experimental/ovstage_placement_extension.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include "../../support/native/cuda_public_test_helpers.h"
#include "../../support/native/ovstage_companion_fixtures.h"
#include "../../support/native/public_api_inputs.h"
#include "cuda_driver.h"
#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_typed_geometry.h"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

extern "C" void     ovspatialquery_test_set_cuda_failure_point(uint32_t point);
extern "C" uint32_t ovspatialquery_test_get_cuda_failure_point();
extern "C" void     ovspatialquery_test_get_cuda_dispatch_evidence(uint64_t* deviceToHostBytes,
                                                                   uint64_t* deviceToDeviceBytes, uint64_t* moduleLoads,
                                                                   uint64_t* moduleUnloads);

namespace
{

bool gCudaFixtureQueries = false;

// A parent Xform with one sphere under it, a second sphere, a cube stretched along x,
// a rotated cube and a capsule along x, all on the x axis, so a downward ray at a
// prim's (x, y) meets that prim's top first.
const char kStage[] = R"(#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "World"
{
    def Xform "Group"
    {
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "s0"
        {
            double radius = 0.5
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def Sphere "s1"
    {
        double radius = 0.5
        double3 xformOp:translate = (4, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    def Cube "c0"
    {
        double size = 1
        double3 xformOp:translate = (8, 0, 0)
        double3 xformOp:scale = (2, 1, 1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
    def Cube "c1"
    {
        double size = 1
        double3 xformOp:translate = (12, 0, 0)
        float3 xformOp:rotateXYZ = (0, 0, 30)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }
    def Capsule "cap"
    {
        double radius = 0.25
        double height = 1
        uniform token axis = "X"
        double3 xformOp:translate = (16, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)";

const char kAdded[] = R"(#usda 1.0
(
    defaultPrim = "A"
)
def Sphere "A"
{
    double radius = 0.25
    double3 xformOp:translate = (20, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)";

struct Stage
{
    ovstage_instance_t*         stage      = nullptr;
    path_dictionary_instance_t* dictionary = nullptr;
    ovstage_ordinal_t           sealed     = 0u;
};

bool waitPopulation(ovstage_instance_t* stage, ovstage_population_enqueue_result_t operation)
{
    return operation.status == OVSTAGE_OK &&
           ovstage_population_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, nullptr) == OVSTAGE_OK;
}

// Derives world matrices into `ordinal` and seals it: the protocol the change feed needs.
bool computeAndSeal(Stage& s, ovstage_ordinal_t ordinal)
{
    const ovstage_write_floor_desc_t floor{ordinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    if (!waitStageOperation(s.stage, ovstage_compute_hierarchy(s.stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU,
                                                               ordinal, ordinal)) ||
        !waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floor)))
    {
        return false;
    }
    s.sealed = ordinal;
    return true;
}

// Writes a prim's local transform: rows 0 to 2 are its scaled axes, row 3 the translation.
bool writeXform(Stage& s, const char* path, ovstage_ordinal_t ordinal, double yawDegrees, double x, double y,
                double scaleX = 1.0)
{
    const double c = std::cos(yawDegrees * 3.14159265358979323846 / 180.0);
    const double n = std::sin(yawDegrees * 3.14159265358979323846 / 180.0);
    double       matrix[16]{c * scaleX, n * scaleX, 0.0, 0.0, -n, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, 0.0, 1.0};
    return writeAttribute(s.stage, s.dictionary, path, std::strlen(path), "omni:xform", ordinal, matrix,
                          DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX);
}

uint64_t pathValue(const Stage& s, const char* path)
{
    const ovx_string_t text{path, std::strlen(path)};
    ovx_primpath_t     value = OVX_INVALID_PRIMPATH;
    return dictionaryCallSucceeded(
               s.dictionary, s.dictionary->vtable->create_paths_from_strings(s.dictionary->context, &text, 1u, &value))
               ? handleValue(value)
               : 0u;
}

// Waits for a publication and checks what it published, naming a refusal.
bool published(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t operation,
               ovspatialquery_ordinal_t ordinal, uint64_t objects)
{
    if (operation.status != OVSPATIALQUERY_API_SUCCESS)
    {
        const ovspatialquery_string_t error = ovspatialquery_get_last_error();
        std::fprintf(stderr, "publication %lld refused: %u %.*s\n", static_cast<long long>(ordinal),
                     static_cast<unsigned>(operation.status), static_cast<int>(error.length),
                     error.ptr != nullptr ? error.ptr : "");
        return false;
    }
    if (finishPopulationUpdate(instance, operation, ordinal, objects, 0u))
        return true;
    ovspatialquery_op_status_t                status{};
    ovspatialquery_population_update_result_t result{};
    (void)ovspatialquery_query_op_status(instance, operation.op_id, &status);
    (void)ovspatialquery_fetch_population_update(instance, operation.op_id, &result);
    std::fprintf(stderr, "publication %lld: terminal %u, published %lld, objects %llu\n",
                 static_cast<long long>(ordinal), static_cast<unsigned>(status.terminal_status),
                 static_cast<long long>(result.published_ordinal),
                 static_cast<unsigned long long>(result.object_count));
    (void)ovspatialquery_destroy_op(instance, &operation.op_id);
    return false;
}

// The first object a downward ray at (x, y) from z = 10 meets in the head.
struct DownHit
{
    bool     hit      = false;
    uint64_t id       = 0u;
    double   distance = 0.0;
};

DownHit downRay(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population, double x,
                double y, ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = precision;
    ray.ray.origin[0]         = x;
    ray.ray.origin[1]         = y;
    ray.ray.origin[2]         = 10.0;
    ray.ray.unit_direction[2] = -1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t result{};
    DownHit                                hit;
    if (succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &result)) && result.has_hit != 0u)
    {
        hit.hit = true;
        if (result.hit.truth == OVSPATIALQUERY_HIT_BOUNDS_CANDIDATE)
        {
            hit.id       = result.hit.data.bounds_candidate.identity.object_id.value;
            hit.distance = result.hit.data.bounds_candidate.distance;
        }
        else
        {
            hit.id       = result.hit.data.exact.identity.object_id.value;
            hit.distance = result.hit.data.exact.distance;
        }
    }
    return hit;
}

bool hits(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population, double x, double y,
          uint64_t id, double top)
{
    const DownHit hit = downRay(instance, population, x, y);
    if (!hit.hit || hit.id != id || std::abs(hit.distance - (10.0 - top)) > 1.0e-9)
    {
        std::fprintf(stderr, "ray at (%g, %g): hit %d id %llu distance %.12g, expected id %llu top %g\n", x, y,
                     static_cast<int>(hit.hit), static_cast<unsigned long long>(hit.id), hit.distance,
                     static_cast<unsigned long long>(id), top);
        return false;
    }
    return true;
}

bool misses(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population, double x,
            double y)
{
    return !downRay(instance, population, x, y).hit;
}

ovspatialquery_ovstage_population_desc_t populationDesc(const Stage& s, ovstage_predicate_t& predicate,
                                                        ovx_string_t (&types)[3], uint32_t   pending)
{
    types[0]                   = ovx_string_t{"Sphere", 6u};
    types[1]                   = ovx_string_t{"Cube", 4u};
    types[2]                   = ovx_string_t{"Capsule", 7u};
    predicate                  = ovstage_predicate_t{};
    predicate.attribute.string = ovx_string_t{"usd-prim-type", 13u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = types;
    predicate.value_count      = 3u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = ovstage_filter_t{&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = pending;
    return desc;
}

// Without the placement extension or a CUDA device the device path refuses before
// admission and the host path is unchanged.
int testRefusal(ovspatialquery_instance_handle_t instance, Stage& s)
{
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 1u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    const ovspatialquery_dynamic_execution_t onDevice{{kDLCUDA, 0}, {1u, 1u, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    const ovspatialquery_enqueue_result_t    refused =
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, &onDevice);
    CHECK(refused.status == OVSPATIALQUERY_API_NOT_SUPPORTED && refused.op_id == OVSPATIALQUERY_INVALID_OP);
    CHECK(published(
        instance, ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, nullptr),
        s.sealed, 5u));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    return 0;
}

bool fixtureRay(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population, double x,
                double y, double z, float directionX, float directionZ, ovspatialquery_raycast_single_result_t& result,
                double                           distanceTolerance = 1.0e-10,
                ovspatialquery_query_precision_t precision         = OVSPATIALQUERY_QUERY_EXACT);

int testDevicePublications(ovspatialquery_instance_handle_t instance, Stage& s)
{
    CudaDriver                               cuda;
    const CudaDriver::Stream                 stream      = cuda.createStream();
    const uintptr_t                          streamValue = reinterpret_cast<uintptr_t>(stream);
    const ovspatialquery_dynamic_execution_t readable{
        {kDLCUDA, 0}, {1u, streamValue, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE};

    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 2u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));

    // Malformed executions: ON_HOST names no CUDA device, and the device path needs a stream.
    const ovspatialquery_dynamic_execution_t hostOnCuda{
        {kDLCUDA, 0}, {1u, streamValue, 0u}, OVSPATIALQUERY_PREPARE_ON_HOST};
    CHECK(ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, &hostOnCuda)
              .status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    const ovspatialquery_dynamic_execution_t noStream{{kDLCUDA, 0}, {0u, 0u, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    CHECK(ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, &noStream)
              .status == OVSPATIALQUERY_API_INVALID_ARGUMENT);

    const uint64_t s0 = pathValue(s, "/World/Group/s0"), s1 = pathValue(s, "/World/s1"), c0 = pathValue(s, "/World/c0"),
                   c1 = pathValue(s, "/World/c1"), cap = pathValue(s, "/World/cap"),
                   added = pathValue(s, "/World/added");

    // The device set: every family, the stretched cube's half extents (1, 0.5, 0.5) and the
    // capsule's axis along x.
    ovspatialquery_ordinal_t head = s.sealed;
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, head, &readable),
                    head, 5u));
    CHECK(hits(instance, population, 1.0, 0.0, s0, 0.5));
    CHECK(hits(instance, population, 4.0, 0.0, s1, 0.5));
    CHECK(hits(instance, population, 8.9, 0.0, c0, 0.5));
    CHECK(hits(instance, population, 16.7, 0.0, cap, 0.25));
    CHECK(misses(instance, population, 6.0, 0.0));
    // Exact geometry comes from the readable version's host objects.
    const DownHit exact = downRay(instance, population, 1.0, 0.0, OVSPATIALQUERY_QUERY_EXACT);
    CHECK(exact.hit && exact.id == s0 && std::abs(exact.distance - 9.5) < 1.0e-9);

    // Device versions belong to the device: the host calls, a view and an ON_HOST update refuse.
    CHECK(ovspatialquery_enqueue_ovstage_population_update(instance, population, head, head + 1).status ==
          OVSPATIALQUERY_API_NOT_SUPPORTED);
    ovspatialquery_ovstage_view_desc_t viewDesc{};
    viewDesc.filter                   = desc.filter;
    ovspatialquery_view_handle_t view = OVSPATIALQUERY_INVALID_VIEW;
    CHECK(ovspatialquery_create_ovstage_view(instance, population, &viewDesc, &view).status ==
          OVSPATIALQUERY_API_NOT_SUPPORTED);

    // A parent move reaches its child through the change feed.
    CHECK(writeXform(s, "/World/Group", s.sealed + 1u, 0.0, 0.0, 5.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 5u));
    head = s.sealed;
    CHECK(hits(instance, population, 1.0, 5.0, s0, 0.5));
    CHECK(misses(instance, population, 1.0, 0.0));
    CHECK(hits(instance, population, 4.0, 0.0, s1, 0.5));

    // A prim's own move, with a new yaw.
    CHECK(writeXform(s, "/World/c1", s.sealed + 1u, 60.0, 12.0, 3.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 5u));
    head = s.sealed;
    CHECK(hits(instance, population, 12.0, 3.0, c1, 0.5));
    CHECK(misses(instance, population, 12.0, 0.0));

    // An added prim is a membership change the host comparison catches.
    ovstage_population_usd_reference_handle_t reference{};
    CHECK(waitPopulation(
        s.stage, ovstage_population_add_usd_reference_from_string(s.stage, ovx_string_t{kAdded, sizeof(kAdded) - 1u},
                                                                  ovx_string_t{"/World/added", 12u}, &reference)));
    CHECK(waitPopulation(s.stage, ovstage_population_apply_usd_changes(s.stage, s.sealed + 1u)) &&
          computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 6u));
    head = s.sealed;
    CHECK(hits(instance, population, 20.0, 0.0, added, 0.25));
    CHECK(hits(instance, population, 1.0, 5.0, s0, 0.5));

    // The stretched cube changes only its X extent; Z remains one half.
    CHECK(writeXform(s, "/World/c0", s.sealed + 1u, 0.0, 8.0, 0.0, 3.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 6u));
    head = s.sealed;
    CHECK(hits(instance, population, 9.25, 0.0, c0, 0.5));

    // So are a removed prim and a deleted one.
    CHECK(waitPopulation(s.stage, ovstage_population_remove_usd_reference(s.stage, reference)));
    CHECK(waitPopulation(s.stage, ovstage_population_apply_usd_changes(s.stage, s.sealed + 1u)) &&
          computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 5u));
    head = s.sealed;
    CHECK(misses(instance, population, 20.0, 0.0));
    {
        const ovx_string_t     path{"/World/c0", 9u};
        ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
        ovx_primpath_list_t    list  = OVX_INVALID_PRIMPATH_LIST;
        CHECK(withPathQuery(s.stage, s.dictionary, path, query, list));
        CHECK(waitStageOperation(s.stage, ovstage_delete_attributes(s.stage, query, nullptr, 0u, s.sealed + 1u)));
        CHECK(releasePathQuery(s.stage, s.dictionary, query, list));
    }
    CHECK(computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 4u));
    head = s.sealed;
    CHECK(misses(instance, population, 8.9, 0.0));

    // A definition edit writes no world matrix; the host comparison catches it too.
    double radius = 0.75;
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/s1", 9u, "radius", s.sealed + 1u, &radius,
                         DLDataType{kDLFloat, 64u, 1u}) &&
          computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 4u));
    head = s.sealed;
    CHECK(hits(instance, population, 4.7, 0.0, s1, 0.75));

    // A prepared box follows independent axis scales without a reshape exclusion.
    CHECK(writeXform(s, "/World/c1", s.sealed + 1u, 60.0, 12.0, 3.0, 3.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 4u));
    head = s.sealed;
    CHECK(hits(instance, population, 12.625, 3.0 + 0.625 * std::sqrt(3.0), c1, 0.5));
    ovspatialquery_raycast_single_result_t scaledBox{};
    CHECK(fixtureRay(instance, population, 12.625, 3.0 + 0.625 * std::sqrt(3.0), 2.0, 0.0f, -1.0f, scaledBox, 1.0e-10,
                     OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE));
    CHECK(scaledBox.has_hit != 0u && scaledBox.hit.data.bounds_candidate.identity.object_id.value == c1 &&
          std::abs(scaledBox.hit.data.bounds_candidate.distance - 1.5) <= 1.0e-10);
    CHECK(writeXform(s, "/World/c1", s.sealed + 1u, 60.0, 12.0, 3.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 4u));
    head = s.sealed;
    CHECK(hits(instance, population, 12.0, 3.0, c1, 0.5));

    // Chained updates: the second builds on the first while it is still pending.
    CHECK(writeXform(s, "/World/s1", s.sealed + 1u, 0.0, 4.0, 2.0) && computeAndSeal(s, s.sealed + 1u));
    const ovspatialquery_ordinal_t  first = s.sealed;
    ovspatialquery_enqueue_result_t firstUpdate =
        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head, first, &readable);
    CHECK(firstUpdate.status == OVSPATIALQUERY_API_SUCCESS);
    CHECK(writeXform(s, "/World/s1", s.sealed + 1u, 0.0, 4.0, 4.0) && computeAndSeal(s, s.sealed + 1u));
    ovspatialquery_enqueue_result_t secondUpdate = ovspatialquery_enqueue_ovstage_population_update_with_execution(
        instance, population, first, s.sealed, &readable);
    CHECK(published(instance, firstUpdate, first, 4u));
    CHECK(published(instance, secondUpdate, s.sealed, 4u));
    CHECK(hits(instance, population, 4.0, 4.0, s1, 0.75));
    CHECK(misses(instance, population, 4.0, 2.0));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    return 0;
}

// The CUDA modules every Driver of the library loaded and unloaded so far.
struct ModuleCounts
{
    uint64_t loads   = 0u;
    uint64_t unloads = 0u;
};

ModuleCounts moduleCounts()
{
    ModuleCounts counts;
    ovspatialquery_test_get_cuda_dispatch_evidence(nullptr, nullptr, &counts.loads, &counts.unloads);
    return counts;
}

ovspatialquery_batch_plan_handle_t createDevicePlan(ovspatialquery_instance_handle_t   instance,
                                                    ovspatialquery_population_handle_t population, int32_t ordinal)
{
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                             = populationRef(population);
    desc.hit_mode                           = OVSPATIALQUERY_HIT_MODE_ANY;
    desc.precision                          = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    desc.device                             = {kDLCUDA, ordinal};
    desc.maximum_query_count                = 1u;
    desc.maximum_hits_per_query             = 1u;
    desc.hit_capacity_per_slot              = 1u;
    desc.result_slot_count                  = 1u;
    desc.result_fields                      = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    return succeeded(ovspatialquery_create_raycast_plan(instance, &desc, &plan)) ? plan
                                                                                 : OVSPATIALQUERY_INVALID_BATCH_PLAN;
}

// Every CUDA user of an instance takes its device's one module (the bundled PTX
// loaded into the primary context): the availability probe loads it and the
// instance keeps it, and an Ovstage device publication (its gather and its device
// box population), a plan, a plan created after the last one was destroyed, and a
// pair reporter load nothing more. With the driver's JIT cache unusable, every
// load compiles the PTX again, for seconds. A second device loads its own module.
// The modules unload with the instance. Runs before anything else in the process
// uses CUDA.
int testOneModulePerDevice(Stage& s, bool placement)
{
    CHECK(moduleCounts().loads == 0u);
    const ovspatialquery_device_budget_t budgets[2]{{{kDLCUDA, 0}, UINT64_C(64) << 20u},
                                                    {{kDLCUDA, 1}, UINT64_C(64) << 20u}};
    ovspatialquery_instance_desc_t       desc = commonInstanceDesc(4u);
    desc.device_budgets                       = budgets;
    desc.device_budget_count                  = 2u;
    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;
    CHECK(succeeded(ovspatialquery_create_instance(&desc, &instance)));
    const auto available = [instance](int32_t ordinal)
    {
        return succeeded(ovspatialquery_query_capability(instance, DLDevice{kDLCUDA, ordinal},
                                                         OVSPATIALQUERY_CAPABILITY_RAYCAST_PLAN));
    };
    if (!placement || !available(0))
    {
        std::printf("one module per device: skipped without CUDA or the placement extension\n");
        CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
        return 0;
    }
    CHECK(moduleCounts().loads == 1u && moduleCounts().unloads == 0u);
    const uint64_t devices = available(1) ? 2u : 1u;
    std::printf("one module per device: %llu CUDA device(s)\n", static_cast<unsigned long long>(devices));
    CHECK(moduleCounts().loads == devices);

    CudaDriver                               cuda;
    const ovspatialquery_dynamic_execution_t onDevice{
        {kDLCUDA, 0}, {1u, reinterpret_cast<uintptr_t>(cuda.createStream()), 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          populationDescription = populationDesc(s, predicate, types, 1u);
    auto                population            = OVSPATIALQUERY_INVALID_POPULATION;
    auto                hostPopulation        = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &populationDescription, &population)));
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &populationDescription, &hostPopulation)));
    CHECK(published(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, &onDevice),
        s.sealed, 5u));
    CHECK(published(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, hostPopulation, s.sealed, nullptr),
        s.sealed, 5u));
    ovspatialquery_batch_plan_handle_t plan = createDevicePlan(instance, population, 0);
    CHECK(plan != OVSPATIALQUERY_INVALID_BATCH_PLAN && destroyPlanWhenRetired(instance, plan));
    plan = createDevicePlan(instance, population, 0);
    CHECK(plan != OVSPATIALQUERY_INVALID_BATCH_PLAN);
    ovspatialquery_pair_reporter_desc_t reporterDesc{};
    reporterDesc.first                             = populationRef(population);
    reporterDesc.second                            = populationRef(population);
    reporterDesc.active_pair_capacity              = 16u;
    reporterDesc.delta_pair_capacity               = 16u;
    reporterDesc.result_slot_count                 = 1u;
    reporterDesc.result_device                     = DLDevice{kDLCUDA, 0};
    ovspatialquery_pair_reporter_handle_t reporter = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    CHECK(succeeded(ovspatialquery_create_pair_reporter(instance, &reporterDesc, &reporter)));
    ovspatialquery_batch_plan_handle_t secondPlan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    if (devices == 2u)
    {
        secondPlan = createDevicePlan(instance, hostPopulation, 1);
        CHECK(secondPlan != OVSPATIALQUERY_INVALID_BATCH_PLAN);
    }
    CHECK(moduleCounts().loads == devices && moduleCounts().unloads == 0u);

    CHECK(succeeded(ovspatialquery_destroy_pair_reporter(instance, reporter)));
    CHECK(destroyPlanWhenRetired(instance, plan));
    CHECK(secondPlan == OVSPATIALQUERY_INVALID_BATCH_PLAN || destroyPlanWhenRetired(instance, secondPlan));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, hostPopulation)));
    CHECK(moduleCounts().unloads == 0u);
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(moduleCounts().loads == devices && moduleCounts().unloads == devices);
    return 0;
}

// A plain ON_DEVICE population: CPU readers refuse it, and a CUDA plan reads it on the
// device, here selecting a publication before it completes.
int testCudaPlan(ovspatialquery_instance_handle_t instance, Stage& s)
{
    CudaDriver                               cuda;
    const CudaDriver::Stream                 stream      = cuda.createStream();
    const uintptr_t                          streamValue = reinterpret_cast<uintptr_t>(stream);
    const ovspatialquery_dynamic_execution_t onDevice{
        {kDLCUDA, 0}, {1u, streamValue, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 2u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    const uint64_t s1 = pathValue(s, "/World/s1");
    CHECK(published(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, &onDevice),
        s.sealed, 4u));
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    ray.ray.unit_direction[2] = -1.0f;
    ray.ray.max_distance      = 20.0;
    ovspatialquery_raycast_single_result_t refused{};
    CHECK(ovspatialquery_raycast_closest_sync(instance, &ray, &refused).status == OVSPATIALQUERY_API_NOT_SUPPORTED);

    ovspatialquery_raycast_plan_desc_t planDesc{};
    planDesc.target                         = populationRef(population);
    planDesc.hit_mode                       = OVSPATIALQUERY_HIT_MODE_ANY;
    planDesc.precision                      = OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE;
    planDesc.device                         = {kDLCUDA, 0};
    planDesc.maximum_query_count            = 1u;
    planDesc.maximum_hits_per_query         = 1u;
    planDesc.hit_capacity_per_slot          = 1u;
    planDesc.result_slot_count              = 1u;
    planDesc.result_fields                  = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    CHECK(succeeded(ovspatialquery_create_raycast_plan(instance, &planDesc, &plan)));

    // Move s1 and publish; the plan selects that publication at once.
    const ovspatialquery_ordinal_t base = s.sealed;
    CHECK(writeXform(s, "/World/s1", s.sealed + 1u, 0.0, 4.0, 7.0) && computeAndSeal(s, s.sealed + 1u));
    ovspatialquery_enqueue_result_t update = ovspatialquery_enqueue_ovstage_population_update_with_execution(
        instance, population, base, s.sealed, &onDevice);
    CHECK(update.status == OVSPATIALQUERY_API_SUCCESS);

    double                          origin[3]    = {4.0, 7.0, 10.0};
    float                           direction[3] = {0.0f, 0.0f, -1.0f};
    double                          distance     = 20.0;
    const CudaDriver::DevicePointer origins      = cuda.allocate(sizeof(origin));
    const CudaDriver::DevicePointer directions   = cuda.allocate(sizeof(direction));
    const CudaDriver::DevicePointer distances    = cuda.allocate(sizeof(distance));
    cuda.uploadAsync(origins, origin, sizeof(origin), stream);
    cuda.uploadAsync(directions, direction, sizeof(direction), stream);
    cuda.uploadAsync(distances, &distance, sizeof(distance), stream);
    const CudaDriver::Event inputsReady = cuda.createEvent();
    cuda.record(inputsReady, stream);
    const DLDevice onCuda{kDLCUDA, 0};
    InputArray originInput(reinterpret_cast<const double*>(static_cast<uintptr_t>(origins)), 1, 3, kDLFloat, onCuda);
    InputArray directionInput(reinterpret_cast<const float*>(static_cast<uintptr_t>(directions)), 1, 3, kDLFloat,
                              onCuda);
    InputArray distanceInput(reinterpret_cast<const double*>(static_cast<uintptr_t>(distances)), 1, kDLFloat, onCuda);
    for (InputArray* input : {&originInput, &directionInput, &distanceInput})
        input->value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind           = OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT;
    batch.ordinal.source_ordinal = s.sealed;
    batch.query_count            = 1u;
    batch.origins                = &originInput.value;
    batch.unit_directions        = &directionInput.value;
    batch.max_distances          = &distanceInput.value;
    batch.cuda_sync              = {1u, streamValue, reinterpret_cast<uintptr_t>(inputsReady)};
    ovspatialquery_raycast_batch_reservation_t reservation{};
    ovspatialquery_enqueue_result_t launch = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    CHECK(launch.status == OVSPATIALQUERY_API_SUCCESS && reservation.common.source_ordinal == s.sealed);
    ovspatialquery_op_wait_result_t waited{};
    CHECK(succeeded(ovspatialquery_wait_op(instance, launch.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)));
    ovspatialquery_raycast_batch_result_t result{};
    CHECK(succeeded(ovspatialquery_fetch_raycast_batch(instance, launch.op_id, &result)));
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(result.common.result_ready.value));
    uint64_t stored = 0u, id = 0u;
    double   hitDistance = 0.0;
    cuda.downloadAsync(&stored, deviceAddress(result.columns.hit_ranges.stored_hit_count), sizeof(stored), stream);
    cuda.downloadAsync(&id, deviceAddress(result.columns.object_ids), sizeof(id), stream);
    cuda.downloadAsync(&hitDistance, deviceAddress(result.columns.contact.candidate.distances), sizeof(hitDistance),
                       stream);
    const CudaDriver::Event downloaded = cuda.createEvent();
    cuda.record(downloaded, stream);
    cuda.synchronize(downloaded);
    CHECK(stored == 1u && id == s1 && std::abs(hitDistance - 9.25) < 1.0e-9);
    // The downloads were the consumers; the library takes the event.
    ovspatialquery_cuda_event_t consumerDone{reinterpret_cast<uintptr_t>(downloaded)};
    CHECK(succeeded(ovspatialquery_destroy_op_after_cuda(instance, &launch.op_id, &consumerDone)));
    CHECK(published(instance, update, s.sealed, 4u));
    // The plan stays BUSY until the library has retired the launch's storage behind the event.
    CHECK(destroyPlanWhenRetired(instance, plan));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    return 0;
}

// Writes a prim's authored `worldMatrix` (an unrotated pose at x, y), the matrix the
// host and device paths read for prims without a Fabric world matrix.
bool writeAuthoredMatrix(Stage& s, const char* path, ovstage_ordinal_t ordinal, double x, double y)
{
    double matrix[16]{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, 0.0, 1.0};
    return writeAttribute(s.stage, s.dictionary, path, std::strlen(path), "worldMatrix", ordinal, matrix,
                          DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX);
}

// A prim without a Fabric world matrix takes its authored one; a prim with both keeps
// the Fabric pose, on a membership change and on poses-only updates alike.
int testAuthoredMatrix(ovspatialquery_instance_handle_t instance, Stage& s)
{
    CudaDriver                               cuda;
    const CudaDriver::Stream                 stream = cuda.createStream();
    const ovspatialquery_dynamic_execution_t readable{
        {kDLCUDA, 0}, {1u, reinterpret_cast<uintptr_t>(stream), 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE};
    const uint64_t s0 = pathValue(s, "/World/Group/s0"), raw = pathValue(s, "/World/raw");
    const auto     seal = [&s](ovstage_ordinal_t ordinal)
    {
        const ovstage_write_floor_desc_t floor{ordinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        s.sealed = ordinal;
        return waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floor));
    };
    // A sphere written through the write API, with an authored world matrix only.
    const ovx_string_t sphereName{"Sphere", 6u};
    ovx_token_t        sphere = OVX_INVALID_TOKEN;
    double             radius = 0.25;
    double             bounds[6]{29.75, -0.25, -0.25, 30.25, 0.25, 0.25};
    CHECK(dictionaryCallSucceeded(s.dictionary, s.dictionary->vtable->create_tokens_from_strings(
                                                    s.dictionary->context, &sphereName, 1u, &sphere)));
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/raw", 10u, "usd-prim-type", s.sealed + 1u, &sphere,
                         DLDataType{kDLUInt, 64u, 1u}) &&
          writeAttribute(s.stage, s.dictionary, "/World/raw", 10u, "radius", s.sealed + 1u, &radius,
                         DLDataType{kDLFloat, 64u, 1u}) &&
          writeBounds(s.stage, s.dictionary, "/World/raw", 10u, s.sealed + 1u, bounds));
    CHECK(writeAuthoredMatrix(s, "/World/raw", s.sealed + 1u, 30.0, 0.0) &&
          writeAuthoredMatrix(s, "/World/Group/s0", s.sealed + 1u, 40.0, 0.0) && seal(s.sealed + 1u));

    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 1u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    ovspatialquery_ordinal_t              head = s.sealed;
    const ovspatialquery_enqueue_result_t set =
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, head, &readable);
    if (set.status == OVSPATIALQUERY_API_NOT_SUPPORTED)
    {
        // Ovstage 0.3's placed read refuses the Fabric matrices while a matched prim has none.
        std::printf("authored world matrix: the runtime refuses the device read\n");
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
        return 0;
    }
    CHECK(published(instance, set, head, 5u));
    CHECK(hits(instance, population, 30.0, 0.0, raw, 0.25));
    CHECK(hits(instance, population, 1.0, 5.0, s0, 0.5));
    CHECK(misses(instance, population, 40.0, 0.0));

    CHECK(writeAuthoredMatrix(s, "/World/raw", s.sealed + 1u, 30.0, 5.0) &&
          writeAuthoredMatrix(s, "/World/Group/s0", s.sealed + 1u, 40.0, 5.0) && seal(s.sealed + 1u));
    CHECK(published(instance,
                    ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, head,
                                                                                    s.sealed, &readable),
                    s.sealed, 5u));
    CHECK(hits(instance, population, 30.0, 5.0, raw, 0.25));
    CHECK(misses(instance, population, 30.0, 0.0));
    CHECK(hits(instance, population, 1.0, 5.0, s0, 0.5));
    CHECK(misses(instance, population, 40.0, 5.0));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    return 0;
}

// A host-prepared population keeps its side: a device update on it refuses.
int testHostHead(ovspatialquery_instance_handle_t instance, Stage& s)
{
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 1u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(published(instance, ovspatialquery_enqueue_ovstage_population_set(instance, population, s.sealed), s.sealed,
                    5u));
    const ovspatialquery_dynamic_execution_t onDevice{{kDLCUDA, 0}, {1u, 1u, 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
    CHECK(computeAndSeal(s, s.sealed + 1u));
    CHECK(ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, s.sealed - 1u, s.sealed,
                                                                          &onDevice)
              .status == OVSPATIALQUERY_API_NOT_SUPPORTED);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    return 0;
}

// A fresh real stage keeps the numerical fixtures independent of membership edits above.
bool openTransformFixture(Stage& s, const char* usda)
{
    // Value-initialized: Ovstage 0.4 adds `implementation`, which 0.3 lacks.
    ovstage_instance_desc_t desc{};
    desc.name = "ovspatialquery-transform-regression";
    if (ovstage_create_instance(&desc, &s.stage) != OVSTAGE_OK || s.stage == nullptr)
        return false;
    s.dictionary = ovstage_get_path_dictionary(s.stage);
    if (s.dictionary == nullptr ||
        !waitPopulation(s.stage, openUsdWithPhysics(s.stage, ovx_string_t{usda, std::strlen(usda)}, 1u, 0.0)))
        return false;
    const ovstage_write_floor_desc_t floor{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    s.sealed = 2u;
    return waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floor)) &&
           waitStageOperation(
               s.stage, ovstage_compute_hierarchy(s.stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)) &&
           waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floorTwo));
}

bool publishedRows(ovspatialquery_instance_handle_t instance, ovspatialquery_enqueue_result_t operation,
                   ovspatialquery_ordinal_t ordinal, uint64_t objects, uint64_t excludedId = 0u,
                   ovspatialquery_ovstage_exclusion_reason_t reason = OVSPATIALQUERY_OVSTAGE_EXCLUSION_RESHAPE_PENDING)
{
    if (operation.status != OVSPATIALQUERY_API_SUCCESS)
    {
        const auto error = ovspatialquery_get_last_error();
        std::fprintf(stderr, "fixture publication enqueue=%u %.*s\n", unsigned(operation.status), int(error.length),
                     error.ptr ? error.ptr : "");
        return false;
    }
    ovspatialquery_op_wait_result_t           waited{};
    ovspatialquery_population_update_result_t result{};
    if (!succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
        waited.state != OVSPATIALQUERY_OP_COMPLETED ||
        !succeeded(ovspatialquery_fetch_population_update(instance, operation.op_id, &result)))
    {
        ovspatialquery_op_status_t status{};
        (void)ovspatialquery_query_op_status(instance, operation.op_id, &status);
        std::fprintf(stderr, "fixture publication wait=%u terminal=%u\n", unsigned(waited.state),
                     unsigned(status.terminal_status));
        return false;
    }
    const int64_t rows = excludedId == 0u ? 0 : 1;
    if (result.published_ordinal != ordinal || result.object_count != objects ||
        result.excluded_object_count != static_cast<uint64_t>(rows) || result.excluded_objects.ndim != 1 ||
        result.exclusion_reasons.ndim != 1 || result.excluded_objects.shape == nullptr ||
        result.exclusion_reasons.shape == nullptr || result.excluded_objects.shape[0] != rows ||
        result.exclusion_reasons.shape[0] != rows || result.excluded_objects.device.device_type != kDLCPU ||
        result.exclusion_reasons.device.device_type != kDLCPU || result.excluded_objects.dtype.code != kDLUInt ||
        result.excluded_objects.dtype.bits != 64u || result.exclusion_reasons.dtype.code != kDLInt ||
        result.exclusion_reasons.dtype.bits != 32u)
    {
        std::fprintf(stderr, "fixture rows ordinal=%llu count=%llu exclusions=%llu reason=%d\n",
                     static_cast<unsigned long long>(result.published_ordinal),
                     static_cast<unsigned long long>(result.object_count),
                     static_cast<unsigned long long>(result.excluded_object_count),
                     result.excluded_object_count ? *static_cast<const int32_t*>(result.exclusion_reasons.data) : 0);
        return false;
    }
    if (rows != 0 && (result.excluded_objects.data == nullptr || result.exclusion_reasons.data == nullptr ||
                      *static_cast<const uint64_t*>(result.excluded_objects.data) != excludedId ||
                      *static_cast<const int32_t*>(result.exclusion_reasons.data) != static_cast<int32_t>(reason)))
        return false;
    return finishPopulationUpdate(instance, operation, ordinal, objects, static_cast<uint64_t>(rows));
}

bool fixtureRay(ovspatialquery_instance_handle_t instance, ovspatialquery_population_handle_t population, double x,
                double y, double z, float directionX, float directionZ, ovspatialquery_raycast_single_result_t& result,
                double distanceTolerance, ovspatialquery_query_precision_t precision)
{
    ovspatialquery_raycast_desc_t ray{};
    ray.common.target         = populationTarget(population);
    ray.common.precision      = precision;
    ray.ray.origin[0]         = x;
    ray.ray.origin[1]         = y;
    ray.ray.origin[2]         = z;
    ray.ray.unit_direction[0] = directionX;
    ray.ray.unit_direction[2] = directionZ;
    ray.ray.max_distance      = 4.0;
    if (!succeeded(ovspatialquery_raycast_closest_sync(instance, &ray, &result)))
        return false;
    if (!gCudaFixtureQueries)
        return true;

    CudaDriver                         cuda(CudaDriver::Ownership::tracked);
    const auto                         stream = cuda.createStream();
    ovspatialquery_raycast_plan_desc_t desc{};
    desc.target                 = populationRef(population);
    desc.hit_mode               = OVSPATIALQUERY_HIT_MODE_CLOSEST;
    desc.precision              = precision;
    desc.device                 = {kDLCUDA, 0};
    desc.maximum_query_count    = 1u;
    desc.maximum_hits_per_query = 1u;
    desc.hit_capacity_per_slot  = 1u;
    desc.result_slot_count      = 1u;
    desc.result_fields          = OVSPATIALQUERY_BATCH_RESULT_DISTANCE;
    auto       plan             = OVSPATIALQUERY_INVALID_BATCH_PLAN;
    const auto creation         = ovspatialquery_create_raycast_plan(instance, &desc, &plan);
    if (!succeeded(creation))
    {
        std::fprintf(stderr, "fixture plan creation status=%u\n", static_cast<unsigned>(creation.status));
        return false;
    }
    double     origin[3]{x, y, z};
    float      direction[3]{directionX, 0.0f, directionZ};
    double     distance   = 4.0;
    const auto origins    = cuda.allocate(sizeof(origin));
    const auto directions = cuda.allocate(sizeof(direction));
    const auto distances  = cuda.allocate(sizeof(distance));
    cuda.uploadAsync(origins, origin, sizeof(origin), stream);
    cuda.uploadAsync(directions, direction, sizeof(direction), stream);
    cuda.uploadAsync(distances, &distance, sizeof(distance), stream);
    const auto ready = cuda.createEvent();
    cuda.record(ready, stream);
    const DLDevice device{kDLCUDA, 0};
    InputArray originInput(reinterpret_cast<const double*>(static_cast<uintptr_t>(origins)), 1, 3, kDLFloat, device);
    InputArray directionInput(reinterpret_cast<const float*>(static_cast<uintptr_t>(directions)), 1, 3, kDLFloat,
                              device);
    InputArray distanceInput(reinterpret_cast<const double*>(static_cast<uintptr_t>(distances)), 1, kDLFloat, device);
    for (InputArray* input : {&originInput, &directionInput, &distanceInput})
        input->value.ownership = OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL;
    ovspatialquery_raycast_batch_t batch{};
    batch.ordinal.kind    = OVSPATIALQUERY_ORDINAL_HEAD;
    batch.query_count     = 1u;
    batch.origins         = &originInput.value;
    batch.unit_directions = &directionInput.value;
    batch.max_distances   = &distanceInput.value;
    batch.cuda_sync       = {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)};
    ovspatialquery_raycast_batch_reservation_t reservation{};
    auto operation = ovspatialquery_enqueue_raycast_plan(instance, plan, &batch, &reservation);
    ovspatialquery_op_wait_result_t       waited{};
    ovspatialquery_raycast_batch_result_t gpu{};
    if (operation.status != OVSPATIALQUERY_API_SUCCESS ||
        !succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) ||
        !succeeded(ovspatialquery_fetch_raycast_batch(instance, operation.op_id, &gpu)))
    {
        std::fprintf(stderr, "fixture enqueue status=%u wait=%u\n", static_cast<unsigned>(operation.status),
                     static_cast<unsigned>(waited.state));
        return false;
    }
    cuda.wait(stream, reinterpret_cast<CudaDriver::Event>(gpu.common.result_ready.value));
    uint32_t rowStatus = OVSPATIALQUERY_BATCH_ROW_FAILED;
    uint64_t stored = 0u, count = 0u, id = 0u;
    double   gpuDistance = 0.0;
    cuda.downloadAsync(&rowStatus, deviceAddress(gpu.columns.hit_ranges.query_status), sizeof(rowStatus), stream);
    cuda.downloadAsync(&stored, deviceAddress(gpu.columns.hit_ranges.stored_hit_count), sizeof(stored), stream);
    cuda.downloadAsync(&count, deviceAddress(gpu.columns.hit_ranges.hit_counts), sizeof(count), stream);
    const auto downloaded = cuda.createEvent();
    cuda.record(downloaded, stream);
    cuda.synchronize(downloaded);
    if (stored != 0u)
    {
        cuda.downloadAsync(&id, deviceAddress(gpu.columns.object_ids), sizeof(id), stream);
        cuda.downloadAsync(&gpuDistance,
                           deviceAddress(precision == OVSPATIALQUERY_QUERY_EXACT
                                             ? gpu.columns.contact.exact.distances
                                             : gpu.columns.contact.candidate.distances),
                           sizeof(gpuDistance), stream);
        cuda.record(downloaded, stream);
        cuda.synchronize(downloaded);
    }
    const uint64_t scalarId       = result.has_hit == 0u ? 0u
                                    : precision == OVSPATIALQUERY_QUERY_EXACT
                                        ? result.hit.data.exact.identity.object_id.value
                                        : result.hit.data.bounds_candidate.identity.object_id.value;
    const double   scalarDistance = result.has_hit == 0u                      ? 0.0
                                    : precision == OVSPATIALQUERY_QUERY_EXACT ? result.hit.data.exact.distance
                                                                              : result.hit.data.bounds_candidate.distance;
    const bool     same           = gpu.common.source_ordinal == result.source_ordinal &&
                      rowStatus == OVSPATIALQUERY_BATCH_ROW_SUCCESS && stored == result.has_hit && count == stored &&
                      (stored == 0u || (id == scalarId && std::abs(gpuDistance - scalarDistance) <= distanceTolerance));
    if (!same)
        std::fprintf(
            stderr,
            "CUDA fixture ray disagreement: row=%u count=%llu scalar=%u gpuDistance=%.17g scalarDistance=%.17g\n",
            rowStatus, static_cast<unsigned long long>(stored), result.has_hit, gpuDistance, scalarDistance);
    ovspatialquery_cuda_event_t consumerDone{reinterpret_cast<uintptr_t>(downloaded)};
    return succeeded(ovspatialquery_destroy_op_after_cuda(instance, &operation.op_id, &consumerDone)) &&
           cuda.forgetTransferredEvent(downloaded) && destroyPlanWhenRetired(instance, plan) && same;
}

// Unchanged placed geometry may be retained, but each placement and size
// change must still move the public surface, with current set/update parity.
int testCubePlacementUpdates(ovspatialquery_instance_handle_t          instance,
                             const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
 def Cube "Shape"
 {
     double size = 1
     double3 xformOp:translate = (0, 0, 0)
     uniform token[] xformOpOrder = ["xformOp:translate"]
 }
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 2u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 1u));
    struct Step
    {
        double                         x, angle, scale, size, top;
        double                         shear = 0.0;
        ovspatialquery_geometry_type_t type  = OVSPATIALQUERY_GEOMETRY_BOX;
    };
    const Step                    steps[] = {{0, 0, 1, 1, 0.5},
                                             {3, 0, 1, 1, 0.5},
                                             {3, 45, 1, 1, std::sqrt(0.5)},
                                             {3, 0, 1, 1, 0.5},
                                             {3, 0, 2, 1, 1},
                                             {3, 0, 2, 2, 2},
                                             {0, 0, 1, 1, 0.5},
                                             {0, 0, 1, 1, 0.5, 2.1e-6, OVSPATIALQUERY_GEOMETRY_CONVEX},
                                             {0, 0, 1, 1, 0.5, 1.8e-6, OVSPATIALQUERY_GEOMETRY_BOX},
                                             {0, 0, 1, 1, 0.5, 2.1e-6, OVSPATIALQUERY_GEOMETRY_CONVEX}};
    const uint64_t                id      = pathValue(s, "/Shape");
    ovspatialquery_geometry_ref_t convex{};
    for (const auto& step : steps)
    {
        // Host acquisition chooses BOX/CONVEX anew even when K is retained.
        // Device pose updates retain their recorded family until reshape recovery.
        // fixtureRay also checks CUDA queries over these host publications.
        if (execution && step.shear != 0.0)
            continue;
        const auto   base = s.sealed;
        const double c    = std::cos(step.angle * 3.14159265358979323846 / 180.0);
        const double n    = std::sin(step.angle * 3.14159265358979323846 / 180.0);
        double       matrix[16]{c * step.scale, 0, -n * step.scale, 0, 0,      step.scale, 0, 0,
                                n * step.scale, 0, c * step.scale,  0, step.x, 0,          0, 1};
        matrix[4]   = step.shear;
        double size = step.size;
        CHECK(writeAttribute(s.stage, s.dictionary, "/Shape", 6u, "omni:xform", base + 1u, matrix,
                             DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX));
        CHECK(writeAttribute(s.stage, s.dictionary, "/Shape", 6u, "size", base + 1u, &size,
                             DLDataType{kDLFloat, 64u, 1u}));
        CHECK(waitStageOperation(
            s.stage,
            ovstage_compute_hierarchy(s.stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, base + 1u, base + 1u)));
        // compute_hierarchy leaves the original _worldExtent stale. Both paths
        // must publish the exact geometry box at this new world matrix.
        const ovstage_write_floor_desc_t floor{base + 1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floor)));
        s.sealed = base + 1u;
        if (!execution && step.type == OVSPATIALQUERY_GEOMETRY_CONVEX && convex.type == OVSPATIALQUERY_GEOMETRY_INVALID)
        {
            ovspatialquery::detail::failNextPlacedConvexForTest();
            const auto refused = ovspatialquery_enqueue_ovstage_population_update(instance, population, base, s.sealed);
            CHECK(refused.status == OVSPATIALQUERY_API_OUT_OF_MEMORY && refused.op_id == OVSPATIALQUERY_INVALID_OP);
            ovspatialquery_raycast_single_result_t retained{};
            CHECK(fixtureRay(instance, population, 0, 0, 3, 0, -1, retained));
            CHECK(retained.source_ordinal == base && retained.has_hit != 0u &&
                  retained.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_BOX);
        }
        CHECK(publishedRows(instance,
                            ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                            s.sealed, execution),
                            s.sealed, 1u));
        auto fresh = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &fresh)));
        CHECK(publishedRows(
            instance,
            ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, fresh, s.sealed, execution),
            s.sealed, 1u));
        for (auto target : {population, fresh})
        {
            ovspatialquery_raycast_single_result_t result{};
            CHECK(fixtureRay(instance, target, step.x, 0, 3, 0, -1, result));
            CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
                  std::abs(result.hit.data.exact.distance - (3.0 - step.top)) <= 1.0e-9);
            CHECK(result.hit.data.exact.identity.geometry.type == step.type);
            ovspatialquery_raycast_single_result_t candidate{};
            CHECK(fixtureRay(instance, target, step.x, 0, 3, 0, -1, candidate, 1.0e-9,
                             OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE));
            CHECK(candidate.has_hit != 0u && candidate.hit.data.bounds_candidate.identity.object_id.value == id &&
                  std::abs(candidate.hit.data.bounds_candidate.distance - (3.0 - step.top)) <= 1.0e-9);
            if (target == population && step.type == OVSPATIALQUERY_GEOMETRY_CONVEX)
            {
                // Returning to the same retained shape names the same definition.
                if (convex.type == OVSPATIALQUERY_GEOMETRY_INVALID)
                    convex = result.hit.data.exact.identity.geometry;
                else
                    CHECK(result.hit.data.exact.identity.geometry.value == convex.value);
            }
        }
        CHECK(succeeded(ovspatialquery_destroy_population(instance, fresh)));
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

// A submission error before the completion event is recorded still owns
// its borrowed stage groups and scratch until the accepted gather is fenced.
int testPartialGatherFailure(ovspatialquery_instance_handle_t          instance,
                             const ovspatialquery_dynamic_execution_t* execution)
{
    Stage s;
    CHECK(openTransformFixture(s, kStage));
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 2u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 5u));
    const auto base = s.sealed;
    CHECK(computeAndSeal(s, base + 1u));
    CudaDriver cuda(CudaDriver::Ownership::tracked);
    const auto producer = cuda.createStream();
    const auto ready    = cuda.createEvent();
    struct Gate
    {
        std::mutex              mutex;
        std::condition_variable condition;
        bool                    entered = false, released = false, wrote = false;
    } gate;
    cuda.enqueueHostFunction(
        producer,
        [](void* pointer)
        {
            auto&                        g = *static_cast<Gate*>(pointer);
            std::unique_lock<std::mutex> lock(g.mutex);
            g.entered = true;
            g.condition.notify_all();
            g.condition.wait_for(lock, std::chrono::seconds(10), [&] { return g.released; });
        },
        &gate);
    cuda.record(ready, producer);
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        CHECK(gate.condition.wait_for(lock, std::chrono::seconds(10), [&] { return gate.entered; }));
    }
    auto delayed                 = *execution;
    delayed.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(ready);
    ovspatialquery_test_set_cuda_failure_point(ovspatialquery::detail::kCudaFailureOvstageGatherAfterKernel);
    ovspatialquery_enqueue_result_t failed{};
    std::thread                     submit(
        [&]
        {
            failed = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                                         s.sealed, &delayed);
        });
    // Consumption proves a real kernel was accepted behind the held producer.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (ovspatialquery_test_get_cuda_failure_point() != 0u && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool consumed = ovspatialquery_test_get_cuda_failure_point() == 0u;
    double     matrix[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1};
    const auto write = [&]
    {
        return writeAttribute(s.stage, s.dictionary, "/World/Group/s0", 15u, "omni:fabric:worldMatrix", base + 2u,
                              matrix, DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX);
    };
    bool        wrote = false;
    std::thread writer(
        [&]
        {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            do
            {
                wrote = write();
                if (!wrote)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } while (!wrote && std::chrono::steady_clock::now() < until);
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.wrote = true;
            gate.condition.notify_all();
        });
    bool wroteEarly;
    {
        std::unique_lock<std::mutex> lock(gate.mutex);
        wroteEarly    = gate.condition.wait_for(lock, std::chrono::milliseconds(500), [&] { return gate.wrote; });
        gate.released = true;
        gate.condition.notify_all();
    }
    submit.join();
    writer.join();
    cuda.synchronize(ready);
    ovspatialquery_test_set_cuda_failure_point(0u);
    if (!(consumed && !(wroteEarly && wrote)))
        std::fprintf(stderr, "partial gather consumed=%u wrote=%u early=%u enqueue=%u\n", unsigned(consumed),
                     unsigned(wrote), unsigned(wroteEarly), unsigned(failed.status));
    CHECK(consumed && !(wroteEarly && wrote));
    CHECK(write());
    CHECK(failed.status == OVSPATIALQUERY_API_INTERNAL && failed.op_id == OVSPATIALQUERY_INVALID_OP);
    CHECK(computeAndSeal(s, base + 2u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 5u));
    CHECK(hits(instance, population, 1.0, 0.0, pathValue(s, "/World/Group/s0"), 0.5));
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

int testCapsuleContainment(ovspatialquery_instance_handle_t          instance,
                           const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Capsule "Shape"
    {
        double radius = 1
        double height = 2048
        uniform token axis = "Y"
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 1u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 1u));
    const uint64_t id = pathValue(s, "/World/Shape");
    for (int sign : {-1, 1})
    {
        ovspatialquery_raycast_single_result_t result{};
        CHECK(
            fixtureRay(instance, population, 2.0 * sign, 1024.0 * sign, 0.0, static_cast<float>(-sign), 0.0f, result));
        CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
              std::abs(result.hit.data.exact.distance - 1.0) <= 1.0e-10);
    }
    constexpr double e = 0x1p-22;
    double           matrix[16]{1.0, e, 0.0, 0.0, e, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const ovspatialquery_ordinal_t base = s.sealed;
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "omni:xform", s.sealed + 1u, matrix,
                         DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX) &&
          computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 1u));
    const double n               = std::sqrt(1.0 + e * e);
    const double authoredSupport = 1024.0 * e + n;
    const double y               = 1024.0 + 2.0 * e / n;
    for (int sign : {-1, 1})
    {
        ovspatialquery_raycast_single_result_t result{};
        CHECK(fixtureRay(instance, population, 2.0 * sign, y * sign, 0.0, static_cast<float>(-sign), 0.0f, result));
        CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
              result.hit.data.exact.distance <= 2.0 - authoredSupport + 1.0e-10);
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

int testAnisotropicRecovery(ovspatialquery_instance_handle_t          instance,
                            const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Mesh "Shape"
    {
        uniform bool doubleSided = 1
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
        point3f[] points = [(0,0,0),(281474976710656,0,0),(0,0.000000059604644775390625,0)]
        float3[] extent = [(0,0,0),(281474976710656,0.000000059604644775390625,0)]
        matrix4d xformOp:transform = ((0.000000000000003552713678800500929355621337890625,0,0,0),(0,16777216,0,0),(0,0,16777216,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    const ovx_string_t  type{"Mesh", 4u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = {"usd-prim-type", 13u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &type;
    predicate.value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = {&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = 2u;
    auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 1u));
    const uint64_t                         id = pathValue(s, "/World/Shape");
    ovspatialquery_raycast_single_result_t result{};
    CHECK(fixtureRay(instance, population, 0.75, 0.9, 2.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit == 0u);
    double matrix[16]{0x1p-48, 0x1p-48, 0.0, 0.0, 0.0, 0x1p24, 0.0, 0.0, 0.0, 0.0, 0x1p24, 0.0, 0.0, 0.0, 0.0, 1.0};
    ovspatialquery_ordinal_t base = s.sealed;
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "omni:xform", s.sealed + 1u, matrix,
                         DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX) &&
          computeAndSeal(s, s.sealed + 1u));
    const bool device = execution != nullptr && execution->preparation != OVSPATIALQUERY_PREPARE_ON_HOST;
    if (device)
    {
        // Capture each current sealed target. A completed first publication may
        // be recovered at the second capture; no completion timing is assumed.
        const ovspatialquery_ordinal_t reshape = s.sealed;
        auto first = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                     reshape, execution);
        CHECK(computeAndSeal(s, s.sealed + 1u));
        auto second = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, reshape,
                                                                                      s.sealed, execution);
        if (first.status != OVSPATIALQUERY_API_SUCCESS || second.status != OVSPATIALQUERY_API_SUCCESS)
        {
            const auto error = ovspatialquery_get_last_error();
            std::fprintf(stderr, "reshape chain first=%u second=%u %.*s\n", unsigned(first.status),
                         unsigned(second.status), int(error.length), error.ptr ? error.ptr : "");
        }
        CHECK(first.status == OVSPATIALQUERY_API_SUCCESS && second.status == OVSPATIALQUERY_API_SUCCESS);
        for (auto* operation : {&first, &second})
        {
            ovspatialquery_op_wait_result_t           waited{};
            ovspatialquery_population_update_result_t publication{};
            CHECK(succeeded(
                      ovspatialquery_wait_op(instance, operation->op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
                  waited.state == OVSPATIALQUERY_OP_COMPLETED &&
                  succeeded(ovspatialquery_fetch_population_update(instance, operation->op_id, &publication)));
            CHECK(publication.published_ordinal == (operation == &first ? reshape : s.sealed));
            CHECK(publication.object_count <= 1u && (operation != &first || publication.object_count == 0u));
            const uint64_t rows = 1u - publication.object_count;
            CHECK(publication.excluded_object_count == rows && publication.excluded_objects.ndim == 1 &&
                  publication.exclusion_reasons.ndim == 1 && publication.excluded_objects.shape != nullptr &&
                  publication.exclusion_reasons.shape != nullptr &&
                  publication.excluded_objects.shape[0] == static_cast<int64_t>(rows) &&
                  publication.exclusion_reasons.shape[0] == static_cast<int64_t>(rows));
            if (rows != 0u)
                CHECK(publication.excluded_objects.data != nullptr && publication.exclusion_reasons.data != nullptr &&
                      *static_cast<const uint64_t*>(publication.excluded_objects.data) == id &&
                      *static_cast<const int32_t*>(publication.exclusion_reasons.data) ==
                          OVSPATIALQUERY_OVSTAGE_EXCLUSION_RESHAPE_PENDING);
            // Destroying the first operation after the second is admitted must
            // leave the population's completed publications available to recovery.
            CHECK(succeeded(ovspatialquery_destroy_op(instance, &operation->op_id)));
        }
        base = s.sealed;
        CHECK(computeAndSeal(s, s.sealed + 1u));
        CHECK(publishedRows(instance,
                            ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                            s.sealed, execution),
                            s.sealed, 1u));
    }
    else
        CHECK(publishedRows(instance,
                            ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                            s.sealed, execution),
                            s.sealed, 1u));
    CHECK(fixtureRay(instance, population, 0.75, 0.9, 2.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
          std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-9);
    if (device)
    {
        // Completion observation must preserve pending rows even without a fetch.
        matrix[1] = 0.0;
        base      = s.sealed;
        CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "omni:xform", s.sealed + 1u, matrix,
                             DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX) &&
              computeAndSeal(s, s.sealed + 1u));
        auto operation = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                         s.sealed, execution);
        CHECK(operation.status == OVSPATIALQUERY_API_SUCCESS);
        ovspatialquery_op_wait_result_t waited{};
        ovspatialquery_op_status_t      status{};
        CHECK(succeeded(ovspatialquery_wait_op(instance, operation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
              waited.state == OVSPATIALQUERY_OP_COMPLETED &&
              succeeded(ovspatialquery_query_op_status(instance, operation.op_id, &status)) &&
              status.terminal_status == OVSPATIALQUERY_API_SUCCESS);
        CHECK(succeeded(ovspatialquery_destroy_op(instance, &operation.op_id)));
        CHECK(fixtureRay(instance, population, 0.2, 0.2, 2.0, 0.0f, -1.0f, result));
        CHECK(result.has_hit == 0u);
        base = s.sealed;
        CHECK(computeAndSeal(s, s.sealed + 1u));
        // Retiring the first recovery fork must not drain the stream behind a
        // later producer event. The producer stays blocked until enqueue returns,
        // or a bounded watchdog releases it so a regression can fail without hanging.
        CudaDriver gateCuda(CudaDriver::Ownership::tracked);
        const auto producer = gateCuda.createStream();
        const auto ready    = gateCuda.createEvent();
        struct Gate
        {
            std::mutex              mutex;
            std::condition_variable condition;
            bool                    entered = false, returned = false, released = false, nonblocking = false;
        } gate;
        gateCuda.enqueueHostFunction(
            producer,
            [](void* pointer)
            {
                auto&                        g = *static_cast<Gate*>(pointer);
                std::unique_lock<std::mutex> lock(g.mutex);
                g.entered = true;
                g.condition.notify_all();
                g.condition.wait_for(lock, std::chrono::seconds(10), [&] { return g.released; });
            },
            &gate);
        gateCuda.record(ready, producer);
        {
            std::unique_lock<std::mutex> lock(gate.mutex);
            CHECK(gate.condition.wait_for(lock, std::chrono::seconds(10), [&] { return gate.entered; }));
        }
        std::thread watchdog(
            [&]
            {
                std::unique_lock<std::mutex> lock(gate.mutex);
                gate.nonblocking =
                    gate.condition.wait_for(lock, std::chrono::seconds(2), [&] { return gate.returned; });
                gate.released = true;
                gate.condition.notify_all();
            });
        auto delayed                 = *execution;
        delayed.cuda_sync.wait_event = reinterpret_cast<uintptr_t>(ready);
        auto recovered = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                         s.sealed, &delayed);
        {
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.returned = true;
            gate.condition.notify_all();
        }
        watchdog.join();
        gateCuda.synchronize(ready);
        CHECK(publishedRows(instance, recovered, s.sealed, 1u));
        CHECK(gate.nonblocking);
        CHECK(fixtureRay(instance, population, 0.2, 0.2, 2.0, 0.0f, -1.0f, result));
        CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
              std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-9);
        CHECK(fixtureRay(instance, population, 0.75, 0.9, 2.0, 0.0f, -1.0f, result));
        CHECK(result.has_hit == 0u);
    }
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

// The oracles use authored world points, independently of the adapter's polar split.
int testAffineFamilies(ovspatialquery_instance_handle_t instance, const ovspatialquery_dynamic_execution_t* execution)
{
    struct Ray
    {
        double x, y, z;
        float  directionZ;
        bool   hit;
    };
    struct Fixture
    {
        const char* usda;
        const char* type;
        size_t      typeLength;
        Ray         rays[2];
        size_t      rayCount;
        bool        removeConvexSchema = false;
    };
    const Fixture fixtures[] = {
        {R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Mesh "Shape"
    {
        uniform bool doubleSided = 0
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
        point3f[] points = [(0,0,0),(1,0,0),(0,1,1)]
        float3[] extent = [(0,0,0),(1,1,1)]
        matrix4d xformOp:transform = ((-1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)",
         "Mesh",
         4u,
         {{-0.2, 0.6, 2.6, -1.0f, true}, {-0.2, 0.6, -1.4, 1.0f, false}},
         2u},
        {R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Mesh "Shape" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {
        uniform token physics:approximation = "convexHull"
        uniform bool doubleSided = 0
        uniform token orientation = "rightHanded"
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3,3,3,3]
        int[] faceVertexIndices = [0,2,1,0,1,3,0,3,2,1,2,3]
        point3f[] points = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
        float3[] extent = [(0,0,0),(1,1,1)]
        matrix4d xformOp:transform = ((-1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)",
         "Mesh",
         4u,
         {{-0.2, 0.2, 2.6, -1.0f, true}, {}},
         1u,
         true},
        {R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Cube "Shape"
    {
        double size = 2
        matrix4d xformOp:transform = ((1,0,0,0),(.5,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)",
         "Cube",
         4u,
         {{1.2, 0.8, 3.0, -1.0f, true}, {1.2, -0.8, 3.0, -1.0f, false}},
         2u},
    };
    // Mirrored triangle: barycentrics (.2,.6) give world (-.2,.6,.6), with its
    // authored front preserved. Its reverse ray must be rejected as a back face.
    // Mirrored tetrahedron: at x=-.2,y=.2 its upper face is z=1-.2-.2=.6.
    // Sheared Cube: x'=x+.5y, so y=.8 gives x in [-.6,1.4], with top z=1.
    // At y=-.8, x=1.2 is outside the surface but inside its world AABB.
    for (const Fixture& fixture : fixtures)
    {
        Stage s;
        CHECK(openTransformFixture(s, fixture.usda));
        const ovx_string_t  type{fixture.type, fixture.typeLength};
        ovstage_predicate_t predicate{};
        predicate.attribute.string = {"usd-prim-type", 13u};
        predicate.op               = OVSTAGE_FILTER_OP_IN;
        predicate.values           = &type;
        predicate.value_count      = 1u;
        ovspatialquery_ovstage_population_desc_t desc{};
        desc.stage                               = s.stage;
        desc.filter                              = {&predicate, 1u};
        desc.common.retained_ordinal_capacity    = 2u;
        desc.common.pending_publication_capacity = 1u;
        auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
        CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
        CHECK(publishedRows(
            instance,
            ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
            s.sealed, 1u));
        const uint64_t id = pathValue(s, "/World/Shape");
        for (size_t row = 0u; row < fixture.rayCount; ++row)
        {
            const Ray&                             ray = fixture.rays[row];
            ovspatialquery_raycast_single_result_t result{};
            CHECK(fixtureRay(instance, population, ray.x, ray.y, ray.z, 0.0f, ray.directionZ, result, 1.0e-6));
            CHECK((result.has_hit != 0u) == ray.hit);
            if (ray.hit)
                CHECK(result.hit.data.exact.identity.object_id.value == id &&
                      std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-6);
        }
        if (fixture.removeConvexSchema)
        {
            ovspatialquery_raycast_single_result_t result{};
            CHECK(fixtureRay(instance, population, -.2, .2, 2.6, 0, -1, result, 1.0e-6));
            CHECK(result.has_hit != 0u &&
                  result.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_CONVEX);
            uint64_t empty = 0;
            auto     base  = s.sealed;
            CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "usd-schemas", s.sealed + 1u, &empty,
                                 DLDataType{kDLUInt, 64u, 1u}, 0u, OVSTAGE_SEMANTIC_NONE, true));
            const ovstage_write_floor_desc_t floor{s.sealed + 1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
            CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floor)));
            ++s.sealed;
            CHECK(publishedRows(instance,
                                ovspatialquery_enqueue_ovstage_population_update_with_execution(
                                    instance, population, base, s.sealed, execution),
                                s.sealed, 1u));
            CHECK(fixtureRay(instance, population, -.2, .2, 2.6, 0, -1, result, 1.0e-6));
            CHECK(result.has_hit != 0u &&
                  result.hit.data.exact.identity.geometry.type == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH);
        }
        CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
        CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    }
    return 0;
}

int testAffineSurface(ovspatialquery_instance_handle_t instance, const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Mesh "Shape"
    {
        uniform bool doubleSided = 1
        uniform token subdivisionScheme = "none"
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0,1,2]
        point3f[] points = [(0,0,0),(1,0,0),(0,1,1)]
        float3[] extent = [(0,0,0),(1,1,1)]
        matrix4d xformOp:transform = ((2,0,0,0),(.5,3,0,0),(0,0,1,0),(10,20,30,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    const ovx_string_t  type{"Mesh", 4u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = {"usd-prim-type", 13u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &type;
    predicate.value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = {&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = 1u;
    auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 1u));
    // Barycentric weights (.2,.6) give local (.2,.6,.6), world (10.7,21.8,30.6).
    ovspatialquery_raycast_single_result_t result{};
    CHECK(fixtureRay(instance, population, 10.7, 21.8, 32.6, 0.0f, -1.0f, result, 1.0e-6));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == pathValue(s, "/World/Shape") &&
          std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-6);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

// Borrowed C result tensors remain valid until their own operation is destroyed,
// even when a later publication returns a different exclusion list.
int testExclusionOwnership(ovspatialquery_instance_handle_t          instance,
                           const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Sphere "A"
    {
        double radius = 1
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
    def Sphere "B"
    {
        double radius = 1
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(10,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    const ovx_string_t  type{"Sphere", 6u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = {"usd-prim-type", 13u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &type;
    predicate.value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = {&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = 1u;
    auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 2u));
    const uint64_t idA = pathValue(s, "/World/A"), idB = pathValue(s, "/World/B");
    auto           base = s.sealed;
    CHECK(writeXform(s, "/World/A", s.sealed + 1u, 0.0, 0.0, 0.0, 2.0) && computeAndSeal(s, s.sealed + 1u));
    auto firstOperation = ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                          s.sealed, execution);
    CHECK(firstOperation.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t           waited{};
    ovspatialquery_population_update_result_t first{};
    CHECK(succeeded(ovspatialquery_wait_op(instance, firstOperation.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &waited)) &&
          waited.state == OVSPATIALQUERY_OP_COMPLETED &&
          succeeded(ovspatialquery_fetch_population_update(instance, firstOperation.op_id, &first)));
    CHECK(first.published_ordinal == s.sealed && first.object_count == 1u && first.excluded_object_count == 1u &&
          first.excluded_objects.data != nullptr && first.exclusion_reasons.data != nullptr &&
          first.excluded_objects.shape != nullptr && first.exclusion_reasons.shape != nullptr &&
          first.excluded_objects.shape[0] == 1 && first.exclusion_reasons.shape[0] == 1);
    const auto* firstIds     = static_cast<const uint64_t*>(first.excluded_objects.data);
    const auto* firstReasons = static_cast<const int32_t*>(first.exclusion_reasons.data);
    CHECK(firstIds[0] == idA && firstReasons[0] == OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE);

    base = s.sealed;
    CHECK(writeXform(s, "/World/A", s.sealed + 1u, 0.0, 0.0, 0.0) &&
          writeXform(s, "/World/B", s.sealed + 1u, 0.0, 10.0, 0.0, 0.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 1u, idB, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY));
    CHECK(firstIds[0] == idA && firstReasons[0] == OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE &&
          first.excluded_objects.shape[0] == 1 && first.exclusion_reasons.shape[0] == 1);
    base = s.sealed;
    CHECK(writeXform(s, "/World/B", s.sealed + 1u, 0.0, 10.0, 0.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 2u));
    CHECK(firstIds[0] == idA && firstReasons[0] == OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE &&
          first.excluded_objects.shape[0] == 1 && first.exclusion_reasons.shape[0] == 1);
    ovspatialquery_population_update_result_t repeated{};
    CHECK(succeeded(ovspatialquery_fetch_population_update(instance, firstOperation.op_id, &repeated)));
    CHECK(repeated.published_ordinal == first.published_ordinal && repeated.object_count == 1u &&
          repeated.excluded_object_count == 1u && repeated.excluded_objects.data != nullptr &&
          repeated.exclusion_reasons.data != nullptr &&
          *static_cast<const uint64_t*>(repeated.excluded_objects.data) == idA &&
          *static_cast<const int32_t*>(repeated.exclusion_reasons.data) ==
              OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE);
    CHECK(succeeded(ovspatialquery_destroy_op(instance, &firstOperation.op_id)));

    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

// Changing only a selected prim's type must retry an unsupported-type exclusion.
int testUnsupportedTypeRecovery(ovspatialquery_instance_handle_t          instance,
                                const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Cube "Shape"
    {
        double size = 2
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    const ovx_string_t names[2]{{"Cylinder", 8u}, {"Cube", 4u}};
    ovx_token_t        types[2]{};
    CHECK(dictionaryCallSucceeded(
        s.dictionary, s.dictionary->vtable->create_tokens_from_strings(s.dictionary->context, names, 2u, types)));
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "usd-prim-type", 3u, &types[0],
                         DLDataType{kDLUInt, 64u, 1u}));
    const ovstage_write_floor_desc_t firstFloor{3u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &firstFloor)));
    const ovx_string_t  path{"/World/Shape", 12u};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = {"usd-path", 8u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = &path;
    predicate.value_count      = 1u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = {&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = 1u;
    auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance, ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, 3u, execution), 3u,
        0u, pathValue(s, "/World/Shape"), OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE));
    const ovstage_write_floor_desc_t unchangedFloor{4u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &unchangedFloor)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, 3u, 4u, execution), 4u,
        0u, pathValue(s, "/World/Shape"), OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE));
    // Keep the same path, bounds, transform and definition attributes.
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "usd-prim-type", 5u, &types[1],
                         DLDataType{kDLUInt, 64u, 1u}));
    const ovstage_write_floor_desc_t secondFloor{5u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &secondFloor)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, 4u, 5u, execution), 5u,
        1u));
    ovspatialquery_raycast_single_result_t result{};
    CHECK(fixtureRay(instance, population, 0, 0, 3, 0, -1, result));
    CHECK(result.has_hit && result.hit.data.exact.identity.object_id.value == pathValue(s, "/World/Shape") &&
          std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-10);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

int testUnsupportedSphere(ovspatialquery_instance_handle_t          instance,
                          const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Sphere "Shape"
    {
        double radius = 1
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    ovstage_predicate_t predicate{};
    ovx_string_t        types[3]{};
    const auto          desc       = populationDesc(s, predicate, types, 1u);
    auto                population = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 1u));
    const uint64_t                         id = pathValue(s, "/World/Shape");
    ovspatialquery_raycast_single_result_t result{};
    CHECK(fixtureRay(instance, population, 0.0, 0.0, 2.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
          std::abs(result.hit.data.exact.distance - 1.0) <= 1.0e-10);
    auto             base    = s.sealed;
    constexpr double stretch = 0x1p-22;
    double           nearUniform[16]{1.0 - stretch, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1.0 + stretch, 0, 0, 0, 0, 1};
    CHECK(writeAttribute(s.stage, s.dictionary, "/World/Shape", 12u, "omni:xform", s.sealed + 1u, nearUniform,
                         DLDataType{kDLFloat, 64u, 16u}, 1u, OVSTAGE_SEMANTIC_MATRIX) &&
          computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 1u));
    for (auto precision : {OVSPATIALQUERY_QUERY_EXACT, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE})
    {
        CHECK(fixtureRay(instance, population, 0, 0, 2, 0, -1, result, 1.0e-10, precision));
        const double distance = precision == OVSPATIALQUERY_QUERY_EXACT ? result.hit.data.exact.distance
                                                                        : result.hit.data.bounds_candidate.distance;
        CHECK(result.has_hit != 0u && std::abs(distance - (1.0 - stretch)) <= 1.0e-10);
    }
    base = s.sealed;
    CHECK(writeXform(s, "/World/Shape", s.sealed + 1u, 0.0, 0.0, 0.0, 2.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 0u, id, OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE));
    CHECK(fixtureRay(instance, population, 0.0, 0.0, 2.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit == 0u);
    base = s.sealed;
    CHECK(writeXform(s, "/World/Shape", s.sealed + 1u, 0.0, 0.0, 0.0) && computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 1u));
    CHECK(fixtureRay(instance, population, 0.0, 0.0, 2.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == id &&
          std::abs(result.hit.data.exact.distance - 1.0) <= 1.0e-10);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

// A filter that names its prims publishes those that exist, on the device as on the host:
// a listed path with no prim, and one whose prim is deleted later, are left out.
int testListedPathWithoutPrim(ovspatialquery_instance_handle_t          instance,
                              const ovspatialquery_dynamic_execution_t* execution)
{
    const char usda[] = R"(#usda 1.0
(defaultPrim = "World")
def Xform "World"
{
    def Cube "A"
    {
        double size = 2
    }
    def Cube "B"
    {
        double size = 2
        double3 xformOp:translate = (10, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)";
    Stage      s;
    CHECK(openTransformFixture(s, usda));
    const ovx_string_t  paths[3]{{"/World/A", 8u}, {"/World/B", 8u}, {"/World/Nope", 11u}};
    ovstage_predicate_t predicate{};
    predicate.attribute.string = {"usd-path", 8u};
    predicate.op               = OVSTAGE_FILTER_OP_IN;
    predicate.values           = paths;
    predicate.value_count      = 3u;
    ovspatialquery_ovstage_population_desc_t desc{};
    desc.stage                               = s.stage;
    desc.filter                              = {&predicate, 1u};
    desc.common.retained_ordinal_capacity    = 2u;
    desc.common.pending_publication_capacity = 1u;
    auto population                          = OVSPATIALQUERY_INVALID_POPULATION;
    CHECK(succeeded(ovspatialquery_create_ovstage_population(instance, &desc, &population)));
    CHECK(publishedRows(
        instance,
        ovspatialquery_enqueue_ovstage_population_set_with_execution(instance, population, s.sealed, execution),
        s.sealed, 2u));
    ovspatialquery_raycast_single_result_t result{};
    CHECK(fixtureRay(instance, population, 10.0, 0.0, 3.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == pathValue(s, "/World/B") &&
          std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-10);
    const auto base = s.sealed;
    {
        ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
        ovx_primpath_list_t    list  = OVX_INVALID_PRIMPATH_LIST;
        CHECK(withPathQuery(s.stage, s.dictionary, paths[1], query, list));
        CHECK(waitStageOperation(s.stage, ovstage_delete_attributes(s.stage, query, nullptr, 0u, s.sealed + 1u)));
        CHECK(releasePathQuery(s.stage, s.dictionary, query, list));
    }
    CHECK(computeAndSeal(s, s.sealed + 1u));
    CHECK(publishedRows(instance,
                        ovspatialquery_enqueue_ovstage_population_update_with_execution(instance, population, base,
                                                                                        s.sealed, execution),
                        s.sealed, 1u));
    CHECK(fixtureRay(instance, population, 10.0, 0.0, 3.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit == 0u);
    CHECK(fixtureRay(instance, population, 0.0, 0.0, 3.0, 0.0f, -1.0f, result));
    CHECK(result.has_hit != 0u && result.hit.data.exact.identity.object_id.value == pathValue(s, "/World/A") &&
          std::abs(result.hit.data.exact.distance - 2.0) <= 1.0e-10);
    CHECK(succeeded(ovspatialquery_destroy_population(instance, population)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return 0;
}

} // namespace

int main()
{
    Stage                         s;
    const ovstage_instance_desc_t stageDesc{"ovspatialquery-device-publication-test"};
    CHECK(ovstage_create_instance(&stageDesc, &s.stage) == OVSTAGE_OK && s.stage != nullptr);
    s.dictionary = ovstage_get_path_dictionary(s.stage);
    CHECK(s.dictionary != nullptr);
    CHECK(waitPopulation(s.stage, openUsdWithPhysics(s.stage, ovx_string_t{kStage, sizeof(kStage) - 1u}, 1u, 0.0)));
    const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floorOne)));
    CHECK(waitStageOperation(
        s.stage, ovstage_compute_hierarchy(s.stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u)));
    const ovstage_write_floor_desc_t floorTwo{2u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
    CHECK(waitStageOperation(s.stage, ovstage_advance_write_floor(s.stage, &floorTwo)));
    s.sealed                                          = 2u;
    const ovstage_placement_extension_v1_t* extension = nullptr;
    const bool placement = ovstage_query_placement_extension(s.stage, &extension) == OVSTAGE_OK;
    if (const int moduleStatus = testOneModulePerDevice(s, placement))
        return moduleStatus;

    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, UINT64_C(256) << 20u};
    ovspatialquery_instance_desc_t       instanceDesc = commonInstanceDesc(8u);
    instanceDesc.device_budgets                       = &budget;
    instanceDesc.device_budget_count                  = 1u;
    ovspatialquery_instance_handle_t instance         = OVSPATIALQUERY_INVALID_INSTANCE;
    // The device path needs this build's CUDA backend and a device it can use, not only a driver.
    bool cudaAvailable = ovspatialquery::detail::cudaBackendUnavailableReason(0, nullptr) == nullptr;
    try
    {
        CudaDriver probe;
    }
    catch (...)
    {
        cudaAvailable = false;
    }
    if (!cudaAvailable || !succeeded(ovspatialquery_create_instance(&instanceDesc, &instance)))
    {
        cudaAvailable                                 = false;
        const ovspatialquery_instance_desc_t hostOnly = commonInstanceDesc(8u);
        CHECK(succeeded(ovspatialquery_create_instance(&hostOnly, &instance)));
    }
    gCudaFixtureQueries = cudaAvailable;
    // The capability query predicts what the device path below does on this runtime: the
    // core answer covers the CUDA device, and only the Ovstage answer adds the extension.
    const DLDevice cudaDevice{kDLCUDA, 0};
    CHECK(ovspatialquery_query_capability(instance, cudaDevice, OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE).status ==
          (cudaAvailable ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_NOT_SUPPORTED));
    CHECK(ovspatialquery_query_ovstage_capability(instance, s.stage, cudaDevice,
                                                  OVSPATIALQUERY_CAPABILITY_PREPARE_ON_DEVICE)
              .status == (cudaAvailable && placement ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_NOT_SUPPORTED));
    CHECK(ovspatialquery_query_ovstage_capability(instance, s.stage, cudaDevice, OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN)
              .status == (cudaAvailable ? OVSPATIALQUERY_API_SUCCESS : OVSPATIALQUERY_API_NOT_SUPPORTED));
    CHECK(ovspatialquery_query_ovstage_capability(instance, nullptr, cudaDevice, OVSPATIALQUERY_CAPABILITY_OVERLAP_PLAN)
              .status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    int status = testCubePlacementUpdates(instance, nullptr);
    if (status == 0)
        status = testCapsuleContainment(instance, nullptr);
    if (status == 0)
        status = testAnisotropicRecovery(instance, nullptr);
    if (status == 0)
        status = testAffineFamilies(instance, nullptr);
    if (status == 0)
        status = testAffineSurface(instance, nullptr);
    if (status == 0)
        status = testExclusionOwnership(instance, nullptr);
    if (status == 0)
        status = testUnsupportedSphere(instance, nullptr);
    if (status == 0)
        status = testUnsupportedTypeRecovery(instance, nullptr);
    if (status == 0)
        status = testListedPathWithoutPrim(instance, nullptr);
    if (status != 0)
        return status;
    if (!placement || !cudaAvailable)
    {
        std::printf("device path unavailable (placement extension %s, CUDA %s): checking refusals\n",
                    placement ? "present" : "absent", cudaAvailable ? "present" : "absent");
        status = testRefusal(instance, s);
    }
    else
    {
        CudaDriver                               cuda;
        const ovspatialquery_dynamic_execution_t readable{{kDLCUDA, 0},
                                                          {1u, reinterpret_cast<uintptr_t>(cuda.createStream()), 0u},
                                                          OVSPATIALQUERY_PREPARE_ON_DEVICE_HOST_READABLE};
        status = testCubePlacementUpdates(instance, &readable);
        if (status == 0)
            status = testPartialGatherFailure(instance, &readable);
        if (status == 0)
            status = testCapsuleContainment(instance, &readable);
        if (status == 0)
            status = testAnisotropicRecovery(instance, &readable);
        if (status == 0)
            status = testAffineFamilies(instance, &readable);
        if (status == 0)
            status = testAffineSurface(instance, &readable);
        if (status == 0)
            status = testExclusionOwnership(instance, &readable);
        if (status == 0)
            status = testUnsupportedSphere(instance, &readable);
        if (status == 0)
            status = testUnsupportedTypeRecovery(instance, &readable);
        if (status == 0)
            status = testListedPathWithoutPrim(instance, &readable);
        if (status == 0)
            status = testHostHead(instance, s);
        if (status == 0)
            status = testDevicePublications(instance, s);
        if (status == 0)
            status = testCudaPlan(instance, s);
        if (status == 0)
            status = testAuthoredMatrix(instance, s);
    }
    CHECK(succeeded(ovspatialquery_destroy_instance(instance)));
    CHECK(ovstage_destroy_instance(s.stage) == OVSTAGE_OK);
    return status;
}
