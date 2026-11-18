/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../support/native/public_api_inputs.h"
#include "../../support/native/test_failure.h"
#include <ovspatialquery/ovspatialquery_explicit.h>
#include <ovspatialquery/ovspatialquery_pairs.h>

#include "ovspatialquery_pair_reporter_operation.h"

#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
#include "cuda_driver.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_cuda_pair_reporter.h"
#include "ovspatialquery_instance_operation_runtime.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <thread>
#include <utility>
#include <vector>

// Reporter advances patched from per-version change records (ADR-0019),
// driven through the public API by randomized explicit publications: moves,
// participation changes, insertions after and between existing identities,
// and removals, with advances that span several versions, select retained
// ordinals or fall behind retention. Views, query bits, tolerance and capacity
// outcomes are covered. Every result is checked against snapshots of both
// versions, and in this build the library also recomputes every patched
// advance completely and exits on any difference.
//
// Built with OVSPATIALQUERY_TEST_CUDA_RESULTS, the same scenarios run CUDA
// reporters (ADR-0019 stage 3): their device rows are read back for the
// snapshot checks, each advance retires after a consumer event, and the
// library compares every CUDA advance with a CPU recomputation. That build
// adds mixed object sizes, far coordinates and a wide tolerance for the device
// grid; a population published through CUDA, whose complete set's device
// arrays the reporter borrows and whose updates' copies it derives from their
// records; reporters sharing one population's device copies under different
// views; a failed admission; a device copy that does not fit the device
// budget; one copy kept per population whatever the retention; a result
// slot reused only after its consumer finished; an advance that loses the
// device while it launches or completes; and a retirement whose restore of
// the caller's context fails or reports device loss.

#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
extern "C" void ovspatialquery_test_set_cuda_failure_point(uint32_t point);
#endif

namespace
{

using PairSet = std::set<std::pair<uint64_t, uint64_t>>;

// Advances per outcome: advanced, active capacity, delta capacity.
std::array<uint64_t, 3u> gOutcomeCounts{};

#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
CudaDriver*        gCuda = nullptr;
CudaDriver::Stream gCudaStream{};

DLDevice resultDevice() { return DLDevice{kDLCUDA, 0}; }

uint64_t patchedAdvanceCount() { return ovspatialquery::detail::patchedCudaPairReporterAdvanceCountForTest(); }
#else
DLDevice resultDevice() { return DLDevice{kDLCPU, 0}; }

uint64_t patchedAdvanceCount() { return ovspatialquery::detail::patchedPairReporterAdvanceCountForTest(); }
#endif

// Sphere definitions: unit and half spheres, and for mixed sizes a twentieth
// of a unit to hundreds, so the device grid meets objects wider than its cells.
constexpr uint64_t               kUnitSphere  = 1u;
constexpr uint64_t               kHalfSphere  = 2u;
constexpr uint64_t               kSmallSphere = 3u;
constexpr uint64_t               kWideSphere  = 4u;
constexpr uint64_t               kHugeSphere  = 5u;
constexpr std::array<double, 5u> kRadii{1.0, 0.5, 0.05, 4.0, 300.0};

void registerSpheres(ovspatialquery_instance_handle_t instance, std::vector<uint64_t> values, std::vector<double> radii)
{
    InputArray valueArray(values.data(), static_cast<int64_t>(values.size()), kDLUInt);
    InputArray radiusArray(radii.data(), static_cast<int64_t>(radii.size()), kDLFloat);
    const ovspatialquery_sphere_definition_table_t spheres{&valueArray.value, &radiusArray.value};
    waitAndDestroy(instance, ovspatialquery_enqueue_sphere_registration(instance, &spheres));
}

ovspatialquery_instance_handle_t createInstance(uint64_t deviceBudget = UINT64_C(1) << 31)
{
    const ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, deviceBudget};
    const bool                           cuda = resultDevice().device_type == kDLCUDA;
    const ovspatialquery_instance_desc_t desc{cuda ? 16u : 1u, 1u, UINT64_C(1) << 30, cuda ? &budget : nullptr,
                                              cuda ? 1u : 0u};
    ovspatialquery_instance_handle_t     instance = OVSPATIALQUERY_INVALID_INSTANCE;
    REQUIRE(ovspatialquery_create_instance(&desc, &instance).status == OVSPATIALQUERY_API_SUCCESS);
    registerSpheres(instance, {kUnitSphere, kHalfSphere, kSmallSphere, kWideSphere, kHugeSphere},
                    std::vector<double>(kRadii.begin(), kRadii.end()));
    return instance;
}

void destroyInstance(ovspatialquery_instance_handle_t instance)
{
    for (uint64_t definition = kUnitSphere; definition <= kHugeSphere; ++definition)
    {
        REQUIRE(
            ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, definition}).status ==
            OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_instance(instance).status == OVSPATIALQUERY_API_SUCCESS);
}

struct Body
{
    std::array<double, 3u> center{};
    uint64_t               definition = kUnitSphere;
    uint64_t               bits       = 1u;
};

using Bodies = std::map<uint64_t, Body>;

// Categories 1 and 2 of a body or a reporter role as participation bits 0 and 127, so
// every pair filter reads both words.
ovspatialquery_participation_bits_t categoryBits(uint64_t categories)
{
    return participationBits(categories & 1u, (categories & 2u) != 0u ? UINT64_C(1) << 63u : 0u);
}

// Rows and removed identities from host memory or, for a population
// published through CUDA, copied to device memory and read on the stream.
struct BodyRows
{
    BodyRows(const Bodies& bodies, bool cuda, std::vector<uint64_t> removed = {}) : removedIds(std::move(removed))
    {
        for (const std::pair<const uint64_t, Body>& entry : bodies)
        {
            ids.push_back(entry.first);
            types.push_back(OVSPATIALQUERY_GEOMETRY_SPHERE);
            geometry.push_back(entry.second.definition);
            translations.insert(translations.end(), entry.second.center.begin(), entry.second.center.end());
            rotations.insert(rotations.end(), {0.0f, 0.0f, 0.0f, 1.0f});
            const ovspatialquery_participation_bits_t words = categoryBits(entry.second.bits);
            bits.insert(bits.end(), words.words, words.words + 2u);
        }
        const int64_t count = static_cast<int64_t>(ids.size());
        bind(idArray, ids.data(), count, 0, kDLUInt, cuda);
        bind(typeArray, types.data(), count, 0, kDLInt, cuda);
        bind(geometryArray, geometry.data(), count, 0, kDLUInt, cuda);
        bind(translationArray, translations.data(), count, 3, kDLFloat, cuda);
        bind(rotationArray, rotations.data(), count, 4, kDLFloat, cuda);
        bind(bitArray, bits.data(), count, 2, kDLUInt, cuda);
        bind(removedArray, removedIds.data(), static_cast<int64_t>(removedIds.size()), 0, kDLUInt, cuda);
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
        if (cuda)
        {
            ready = gCuda->createEvent();
            gCuda->record(ready, gCudaStream);
            execution = {{kDLCUDA, 0},
                         {1u, reinterpret_cast<uintptr_t>(gCudaStream), reinterpret_cast<uintptr_t>(ready)},
                         OVSPATIALQUERY_PREPARE_ON_HOST};
            dynamic   = &execution;
        }
#endif
    }

    BodyRows(const BodyRows&)            = delete;
    BodyRows& operator=(const BodyRows&) = delete;

    // The publication that read the rows has completed.
    ~BodyRows()
    {
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
        if (ready != nullptr)
        {
            gCuda->synchronize(ready);
            gCuda->destroyEvent(ready);
        }
        for (const CudaDriver::DevicePointer allocation : allocations)
        {
            gCuda->free(allocation);
        }
#endif
    }

    template <typename Value>
    void bind(InputArray& array, Value* values, int64_t count, int64_t columns, uint8_t code, bool cuda)
    {
        DLDevice rowDevice{kDLCPU, 0};
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
        const size_t bytes =
            static_cast<size_t>(count) * static_cast<size_t>(columns == 0 ? 1 : columns) * sizeof(Value);
        if (cuda)
        {
            rowDevice = {kDLCUDA, 0};
        }
        if (cuda && bytes != 0u)
        {
            const CudaDriver::DevicePointer device = gCuda->allocate(bytes);
            allocations.push_back(device);
            gCuda->uploadAsync(device, values, bytes, gCudaStream);
            values = reinterpret_cast<Value*>(static_cast<uintptr_t>(device));
        }
#endif
        (void)cuda;
        if (columns == 0)
        {
            array.set(values, count, code, rowDevice);
        }
        else
        {
            array.set(values, count, columns, code, rowDevice);
        }
    }

    ovspatialquery_explicit_instance_table_t table() const
    {
        return {&idArray.value,       &typeArray.value, &geometryArray.value, &translationArray.value,
                &rotationArray.value, nullptr,          &bitArray.value,      nullptr};
    }

    std::vector<uint64_t>                     ids;
    std::vector<int32_t>                      types;
    std::vector<uint64_t>                     geometry;
    std::vector<double>                       translations;
    std::vector<float>                        rotations;
    std::vector<uint64_t>                     bits;
    std::vector<uint64_t>                     removedIds;
    InputArray                                idArray;
    InputArray                                typeArray;
    InputArray                                geometryArray;
    InputArray                                translationArray;
    InputArray                                rotationArray;
    InputArray                                bitArray;
    InputArray                                removedArray;
    const ovspatialquery_dynamic_execution_t* dynamic = nullptr;
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
    std::vector<CudaDriver::DevicePointer> allocations;
    CudaDriver::Event                      ready = nullptr;
    ovspatialquery_dynamic_execution_t     execution{};
#endif
};

// One explicit population, its current bodies, and its published ordinals
// that are still retained (head last). Bodies are placed within `extent`;
// mixed sizes add small, wide and huge bodies and a few far away.
struct Population
{
    ovspatialquery_population_handle_t handle             = OVSPATIALQUERY_INVALID_POPULATION;
    uint32_t                           retainedPriorHeads = 3u;
    double                             extent             = 14.0;
    bool                               mixed              = false;
    // Versions are published from device memory through the CUDA explicit path.
    bool                                 cuda = false;
    Bodies                               bodies;
    std::deque<ovspatialquery_ordinal_t> retained;
    uint64_t                             nextId = 0u;
    // Identities are never reused, including removed ones.
    std::set<uint64_t> used;
};

Body randomBody(std::mt19937_64& random, const Population& population)
{
    std::uniform_real_distribution<double> coordinate(0.0, population.extent);
    Body                                   body;
    body.center = {coordinate(random), coordinate(random), coordinate(random)};
    if (!population.mixed)
    {
        body.definition = (random() & 1u) != 0u ? kHalfSphere : kUnitSphere;
    }
    else
    {
        // Mostly small and half bodies, some wide, a few far beyond the
        // grid's other cells or wider than any cell.
        const uint32_t kind = static_cast<uint32_t>(random() % 400u);
        body.definition =
            kind == 0u ? kHugeSphere : (kind < 20u ? kWideSphere : (kind < 200u ? kHalfSphere : kSmallSphere));
        if (kind == 1u || kind == 2u)
        {
            body.center[0] += 1.0e7;
        }
    }
    body.bits = 1u + random() % 3u;
    return body;
}

// Creates the population with `count` bodies at even identities; odd ones are
// reserved for insertions between them. The first version is not published.
void createPopulation(ovspatialquery_instance_handle_t instance, Population& population, size_t count,
                      std::mt19937_64& random)
{
    population.handle = createExplicitPopulation(instance, population.retainedPriorHeads);
    for (size_t index = 0u; index < count; ++index)
    {
        population.bodies[2u * index + 2u] = randomBody(random, population);
        population.used.insert(2u * index + 2u);
    }
    population.nextId = 2u * count + 2u;
}

void publishFirst(ovspatialquery_instance_handle_t instance, Population& population)
{
    BodyRows                                       rows(population.bodies, population.cuda);
    const ovspatialquery_explicit_population_set_t set{1, rows.table(), rows.dynamic, 0u};
    waitAndDestroy(instance, ovspatialquery_enqueue_explicit_population_set(instance, population.handle, &set));
    population.retained.push_back(1);
}

void publishUpdate(ovspatialquery_instance_handle_t instance, Population& population, const Bodies& changed,
                   std::vector<uint64_t> removed)
{
    const ovspatialquery_ordinal_t                    base = population.retained.back();
    BodyRows                                          rows(changed, population.cuda, removed);
    const ovspatialquery_explicit_population_update_t update{
        base,        base + 1, OVSPATIALQUERY_EXPLICIT_UPDATE_COMPLETE_ROWS, rows.table(), &rows.removedArray.value,
        rows.dynamic};
    ovspatialquery_enqueue_result_t admission =
        ovspatialquery_enqueue_explicit_population_update(instance, population.handle, &update);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            wait.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_population_update_result_t published{};
    REQUIRE(ovspatialquery_fetch_population_update(instance, admission.op_id, &published).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    for (const std::pair<const uint64_t, Body>& entry : changed)
    {
        population.bodies[entry.first] = entry.second;
        population.used.insert(entry.first);
    }
    for (const uint64_t id : removed)
    {
        population.bodies.erase(id);
    }
    // Whichever memory held the rows, the new head holds exactly these bodies.
    REQUIRE(published.object_count == population.bodies.size());
    population.retained.push_back(base + 1);
    if (population.retained.size() > population.retainedPriorHeads + 1u)
    {
        population.retained.pop_front();
    }
}

uint64_t randomExisting(std::mt19937_64& random, const Bodies& bodies)
{
    Bodies::const_iterator entry = bodies.begin();
    std::advance(entry, static_cast<long>(random() % bodies.size()));
    return entry->first;
}

// Replaces `count` random bodies with new ones, nothing else.
void replaceSome(ovspatialquery_instance_handle_t instance, Population& population, std::mt19937_64& random,
                 size_t count)
{
    Bodies changed;
    for (size_t index = 0u; index < count; ++index)
    {
        changed[randomExisting(random, population.bodies)] = randomBody(random, population);
    }
    publishUpdate(instance, population, changed, {});
}

// One randomized publication: mostly a few moves (small steps or jumps), or
// `moves` of them, sometimes many moves, participation changes, insertions
// after the largest identity, insertions between identities, or removals.
void mutate(ovspatialquery_instance_handle_t instance, Population& population, std::mt19937_64& random,
            size_t moves = 0u)
{
    Bodies                                 changed;
    std::vector<uint64_t>                  removed;
    std::uniform_real_distribution<double> jitter(-0.4, 0.4);
    const uint32_t                         kind = static_cast<uint32_t>(random() % 20u);
    const size_t                           moveCount =
        moves != 0u ? moves : (kind == 0u ? population.bodies.size() / 2u : 1u + static_cast<size_t>(random() % 4u));
    for (size_t index = 0u; index < moveCount; ++index)
    {
        const uint64_t id   = randomExisting(random, population.bodies);
        Body           body = population.bodies[id];
        if ((random() & 3u) == 0u)
        {
            body.center = randomBody(random, population).center;
        }
        else
        {
            for (double& value : body.center)
            {
                value += jitter(random);
            }
        }
        changed[id] = body;
    }
    if (kind == 1u || kind == 2u)
    {
        const uint64_t id   = randomExisting(random, population.bodies);
        Body           body = changed.count(id) != 0u ? changed[id] : population.bodies[id];
        body.bits           = body.bits == 1u ? 2u : 1u;
        changed[id]         = body;
    }
    if (kind == 3u || kind == 4u)
    {
        changed[population.nextId] = randomBody(random, population);
        population.nextId += 2u;
    }
    if (kind == 5u)
    {
        // Odd identities are never used otherwise, so this one lands between
        // existing identities and renumbers the positions after it.
        const uint64_t below = randomExisting(random, population.bodies) | 1u;
        if (population.used.count(below) == 0u)
        {
            changed[below] = randomBody(random, population);
        }
    }
    if (kind == 6u && population.bodies.size() > 8u)
    {
        const uint64_t id = randomExisting(random, population.bodies);
        if (changed.count(id) == 0u)
        {
            removed.push_back(id);
        }
    }
    publishUpdate(instance, population, changed, removed);
}

ovspatialquery_query_target_t exactTarget(const ovspatialquery_target_ref_t& target, ovspatialquery_ordinal_t ordinal)
{
    ovspatialquery_query_target_t selected{};
    selected.target                 = target;
    selected.ordinal.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
    selected.ordinal.source_ordinal = ordinal;
    return selected;
}

struct ReporterCase
{
    ovspatialquery_target_ref_t first{};
    uint64_t                    firstBits       = 0u;
    Population*                 firstPopulation = nullptr;
    ovspatialquery_target_ref_t second{};
    uint64_t                    secondBits       = 0u;
    Population*                 secondPopulation = nullptr;
    double                      tolerance        = 0.0;
    uint64_t                    activeCapacity   = 1u << 16;
    uint64_t                    deltaCapacity    = 1u << 16;
    DLDevice                    device           = resultDevice();
    // Whether advances are checked against snapshots; a reporter whose pairs
    // overflow the snapshot buffer relies on the library's recomputation.
    bool                                  checked        = true;
    ovspatialquery_pair_reporter_handle_t handle         = OVSPATIALQUERY_INVALID_PAIR_REPORTER;
    bool                                  hasAccepted    = false;
    ovspatialquery_ordinal_t              acceptedFirst  = 0;
    ovspatialquery_ordinal_t              acceptedSecond = 0;
    uint64_t                              acceptedCount  = 0u;
    PairSet                               accepted;
};

ReporterCase reporterCase(ovspatialquery_target_ref_t first, uint64_t firstBits, Population* firstPopulation,
                          ovspatialquery_target_ref_t second, uint64_t secondBits, Population* secondPopulation,
                          double tolerance, uint64_t activeCapacity = 1u << 16, uint64_t deltaCapacity = 1u << 16)
{
    ReporterCase reporter;
    reporter.first            = first;
    reporter.firstBits        = firstBits;
    reporter.firstPopulation  = firstPopulation;
    reporter.second           = second;
    reporter.secondBits       = secondBits;
    reporter.secondPopulation = secondPopulation;
    reporter.tolerance        = tolerance;
    reporter.activeCapacity   = activeCapacity;
    reporter.deltaCapacity    = deltaCapacity;
    return reporter;
}

// A CUDA reporter gets two result slots unless told otherwise.
void createReporter(ovspatialquery_instance_handle_t instance, ReporterCase& reporter, uint32_t slots = 0u)
{
    const bool                          cuda = reporter.device.device_type == kDLCUDA;
    ovspatialquery_pair_reporter_desc_t desc{};
    desc.first                = reporter.first;
    desc.first_query_bits     = categoryBits(reporter.firstBits);
    desc.second               = reporter.second;
    desc.second_query_bits    = categoryBits(reporter.secondBits);
    desc.active_pair_capacity = reporter.activeCapacity;
    desc.delta_pair_capacity  = reporter.deltaCapacity;
    desc.result_slot_count    = slots != 0u ? slots : (cuda ? 2u : 1u);
    desc.contact_tolerance    = reporter.tolerance;
    desc.result_device        = reporter.device;
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &reporter.handle).status ==
            OVSPATIALQUERY_API_SUCCESS);
}

ovspatialquery_view_handle_t createView(ovspatialquery_instance_handle_t instance, const Population& population,
                                        std::vector<uint64_t> ids)
{
    InputArray                                idArray(ids.data(), static_cast<int64_t>(ids.size()), kDLUInt);
    const ovspatialquery_explicit_view_desc_t desc{&idArray.value};
    ovspatialquery_view_handle_t              view = OVSPATIALQUERY_INVALID_VIEW;
    REQUIRE(ovspatialquery_create_explicit_view(instance, population.handle, &desc, &view).status ==
            OVSPATIALQUERY_API_SUCCESS);
    return view;
}

PairSet snapshot(ovspatialquery_instance_handle_t instance, const ReporterCase& reporter,
                 ovspatialquery_ordinal_t firstOrdinal, ovspatialquery_ordinal_t secondOrdinal)
{
    ovspatialquery_pair_snapshot_desc_t desc{};
    desc.first             = exactTarget(reporter.first, firstOrdinal);
    desc.first_query_bits  = categoryBits(reporter.firstBits);
    desc.second            = exactTarget(reporter.second, secondOrdinal);
    desc.second_query_bits = categoryBits(reporter.secondBits);
    desc.contact_tolerance = reporter.tolerance;
    std::vector<ovspatialquery_pair_t>    rows(1u << 16);
    ovspatialquery_pair_buffer_t          buffer{rows.data(), rows.size()};
    ovspatialquery_pair_snapshot_result_t result{};
    REQUIRE(ovspatialquery_query_pairs_sync(instance, &desc, &buffer, &result).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.truncated == 0u && result.first_source_ordinal == firstOrdinal &&
            result.second_source_ordinal == secondOrdinal);
    PairSet pairs;
    for (uint64_t index = 0u; index < result.stored_pair_count; ++index)
    {
        pairs.insert({rows[index].first.value, rows[index].second.value});
    }
    return pairs;
}

// Reads result rows from the host or the device; no row repeats.
PairSet tensorPairs(const DLTensor& tensor)
{
    PairSet       pairs;
    const int64_t rowStride    = tensor.strides != nullptr ? tensor.strides[0] : 2;
    const int64_t columnStride = tensor.strides != nullptr ? tensor.strides[1] : 1;
    REQUIRE(tensor.ndim == 2 && tensor.shape[1] == 2 && tensor.dtype.bits == 64u);
    const uint64_t* data = static_cast<const uint64_t*>(tensor.data) + tensor.byte_offset / sizeof(uint64_t);
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
    std::vector<uint64_t> host(static_cast<size_t>(tensor.shape[0] * rowStride));
    if (tensor.device.device_type == kDLCUDA)
    {
        if (!host.empty())
        {
            const CudaDriver::Event done = gCuda->createEvent();
            gCuda->downloadAsync(host.data(),
                                 reinterpret_cast<CudaDriver::DevicePointer>(tensor.data) + tensor.byte_offset,
                                 host.size() * sizeof(uint64_t), gCudaStream);
            gCuda->record(done, gCudaStream);
            gCuda->synchronize(done);
            gCuda->destroyEvent(done);
        }
        data = host.data();
    }
#endif
    for (int64_t row = 0; row < tensor.shape[0]; ++row)
    {
        const uint64_t* base = data + row * rowStride;
        REQUIRE(pairs.insert({base[0], base[columnStride]}).second);
    }
    return pairs;
}

// Destroys an advance: a CUDA reporter's fetched device views retire only
// after a consumer event, and plain destruction refuses them.
void destroyAdvance(ovspatialquery_instance_handle_t instance, ovspatialquery_op_id_t operation, bool cuda)
{
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
    if (cuda)
    {
        ovspatialquery_op_id_t copy = operation;
        REQUIRE(ovspatialquery_destroy_op(instance, &copy).status == OVSPATIALQUERY_API_BUSY);
        const CudaDriver::Event     consumer = gCuda->createEvent();
        ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumer)};
        gCuda->record(consumer, gCudaStream);
        REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(operation == OVSPATIALQUERY_INVALID_OP && done.value == 0u);
        return;
    }
#endif
    (void)cuda;
    REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
}

// A CUDA reporter stays busy until its last advance's consumers finished.
void destroyReporter(ovspatialquery_instance_handle_t instance, ovspatialquery_pair_reporter_handle_t reporter)
{
    for (uint32_t attempt = 0u;; ++attempt)
    {
        const ovspatialquery_api_status_t status = ovspatialquery_destroy_pair_reporter(instance, reporter).status;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            return;
        }
        REQUIRE(status == OVSPATIALQUERY_API_BUSY && attempt < 1000000u);
        std::this_thread::yield();
    }
}

PairSet difference(const PairSet& first, const PairSet& second)
{
    PairSet result;
    std::set_difference(first.begin(), first.end(), second.begin(), second.end(), std::inserter(result, result.end()));
    return result;
}

ovspatialquery_pair_reporter_advance_t headSelection()
{
    ovspatialquery_pair_reporter_advance_t selection{};
    selection.first_ordinal.kind  = OVSPATIALQUERY_ORDINAL_HEAD;
    selection.second_ordinal.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    return selection;
}

// A retired CUDA advance holds its result slot until its consumer finished.
ovspatialquery_enqueue_result_t enqueue(ovspatialquery_instance_handle_t instance, const ReporterCase& reporter,
                                        const ovspatialquery_pair_reporter_advance_t& selection,
                                        ovspatialquery_pair_reporter_reservation_t&   reservation)
{
    // The deliberately delayed consumer runs on the GPU: CPU retry counts
    // do not bound how long it has had to finish (especially on Windows).
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;)
    {
        const ovspatialquery_enqueue_result_t admission =
            ovspatialquery_enqueue_pair_reporter_advance(instance, reporter.handle, &selection, &reservation);
        if (admission.status != OVSPATIALQUERY_API_RESULT_CAPACITY || reporter.device.device_type != kDLCUDA)
        {
            return admission;
        }
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// Selects the head or, sometimes, an older retained ordinal that is still no
// older than the accepted one.
ovspatialquery_ordinal_selection_t select(std::mt19937_64& random, const Population& population, bool hasAccepted,
                                          ovspatialquery_ordinal_t accepted)
{
    ovspatialquery_ordinal_selection_t selection{};
    selection.kind = OVSPATIALQUERY_ORDINAL_HEAD;
    if (population.retained.size() > 1u && random() % 4u == 0u)
    {
        const ovspatialquery_ordinal_t older = population.retained[population.retained.size() - 2u];
        if (!hasAccepted || older >= accepted)
        {
            selection.kind           = OVSPATIALQUERY_ORDINAL_RETAINED_EXACT;
            selection.source_ordinal = older;
        }
    }
    return selection;
}

// Advances to the heads or, sometimes, to older retained ordinals, and checks
// the result against snapshots of the accepted and the selected versions.
// Returns the outcome, or -1 when the selected ordinals were already accepted.
int advance(ovspatialquery_instance_handle_t instance, ReporterCase& reporter, std::mt19937_64& random,
            uint64_t& admittedCount, bool headOnly = false)
{
    ovspatialquery_pair_reporter_advance_t selection = headSelection();
    if (!headOnly)
    {
        selection.first_ordinal =
            select(random, *reporter.firstPopulation, reporter.hasAccepted, reporter.acceptedFirst);
        selection.second_ordinal = selection.first_ordinal;
        if (reporter.secondPopulation != reporter.firstPopulation)
        {
            selection.second_ordinal =
                select(random, *reporter.secondPopulation, reporter.hasAccepted, reporter.acceptedSecond);
        }
    }
    const auto resolve = [](const ovspatialquery_ordinal_selection_t& value, const Population& population)
    { return value.kind == OVSPATIALQUERY_ORDINAL_HEAD ? population.retained.back() : value.source_ordinal; };
    const ovspatialquery_ordinal_t firstOrdinal  = resolve(selection.first_ordinal, *reporter.firstPopulation);
    const ovspatialquery_ordinal_t secondOrdinal = resolve(selection.second_ordinal, *reporter.secondPopulation);

    const bool                                 cuda = reporter.device.device_type == kDLCUDA;
    ovspatialquery_pair_reporter_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      admission = enqueue(instance, reporter, selection, reservation);
    if (reporter.hasAccepted && firstOrdinal == reporter.acceptedFirst && secondOrdinal == reporter.acceptedSecond)
    {
        REQUIRE(admission.status == OVSPATIALQUERY_API_STALE_ORDINAL);
        return -1;
    }
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(reservation.result_device.device_type == reporter.device.device_type &&
            (reservation.result_ready.value != 0u) == cuda);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(wait.state == OVSPATIALQUERY_OP_COMPLETED);
    ovspatialquery_pair_reporter_result_t result{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(result.first_source_ordinal == firstOrdinal && result.second_source_ordinal == secondOrdinal);
    REQUIRE(result.result_ready.value == reservation.result_ready.value);
    REQUIRE(result.entered_pairs.device.device_type == reporter.device.device_type);
    PairSet current;
    if (reporter.checked)
    {
        current               = snapshot(instance, reporter, firstOrdinal, secondOrdinal);
        const PairSet entered = difference(current, reporter.accepted);
        const PairSet exited  = difference(reporter.accepted, current);
        const auto    expected =
            current.size() > reporter.activeCapacity
                   ? OVSPATIALQUERY_PAIR_REPORTER_ACTIVE_CAPACITY
                   : (entered.size() + exited.size() > reporter.deltaCapacity ? OVSPATIALQUERY_PAIR_REPORTER_DELTA_CAPACITY
                                                                              : OVSPATIALQUERY_PAIR_REPORTER_ADVANCED);
        REQUIRE(result.outcome == expected);
        if (expected == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED)
        {
            REQUIRE(tensorPairs(result.entered_pairs) == entered && tensorPairs(result.exited_pairs) == exited);
            REQUIRE(result.accepted_active_pair_count == current.size());
        }
    }
    REQUIRE(result.outcome < gOutcomeCounts.size());
    ++gOutcomeCounts[result.outcome];
    if (result.outcome == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED)
    {
        reporter.hasAccepted    = true;
        reporter.acceptedFirst  = firstOrdinal;
        reporter.acceptedSecond = secondOrdinal;
        reporter.acceptedCount  = result.accepted_active_pair_count;
        reporter.accepted       = std::move(current);
    }
    else
    {
        REQUIRE(tensorPairs(result.entered_pairs).empty() && tensorPairs(result.exited_pairs).empty() &&
                result.accepted_active_pair_count == reporter.acceptedCount);
    }
    ++admittedCount;
    destroyAdvance(instance, admission.op_id, cuda);
    return static_cast<int>(result.outcome);
}

// A patched advance needs the records of the selected version and the versions
// in between, not the accepted version: with no prior version retained every
// advance after the first patches, and with two retained a reporter three
// publications behind patches while one four behind recomputes. Rows resent
// unchanged are no change.
void requirePatchesWithoutAcceptedVersion(uint64_t& admittedCount)
{
    std::mt19937_64                        random(5u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    for (const uint32_t retainedPriorHeads : {0u, 2u})
    {
        Population population;
        population.retainedPriorHeads = retainedPriorHeads;
        population.extent             = 8.0;
        createPopulation(instance, population, 64u, random);
        publishFirst(instance, population);
        const ovspatialquery_target_ref_t target = populationRef(population.handle);
        ReporterCase reporter                    = reporterCase(target, 0u, &population, target, 0u, &population, 0.1);
        createReporter(instance, reporter);
        advance(instance, reporter, random, admittedCount, true);
        const std::array<uint32_t, 3u> lags =
            retainedPriorHeads == 0u ? std::array<uint32_t, 3u>{1u, 1u, 1u} : std::array<uint32_t, 3u>{3u, 4u, 3u};
        for (const uint32_t lag : lags)
        {
            for (uint32_t publication = 0u; publication < lag; ++publication)
            {
                replaceSome(instance, population, random, 2u);
            }
            const uint64_t before = patchedAdvanceCount();
            advance(instance, reporter, random, admittedCount, true);
            const uint64_t expected = lag <= retainedPriorHeads + 1u ? 1u : 0u;
            REQUIRE(patchedAdvanceCount() - before == expected);
        }
        // A caller resending every row, two of them moved, still patches.
        Bodies all = population.bodies;
        for (uint32_t index = 0u; index < 2u; ++index)
        {
            all[randomExisting(random, all)] = randomBody(random, population);
        }
        publishUpdate(instance, population, all, {});
        const uint64_t before = patchedAdvanceCount();
        advance(instance, reporter, random, admittedCount, true);
        REQUIRE(patchedAdvanceCount() - before == 1u);
        destroyReporter(instance, reporter.handle);
        REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    }
    destroyInstance(instance);
}

// A definition only a replaced object used can be destroyed once no retained
// version references it, although the other objects of that version's
// storage are still shared by the next version.
void requireReplacedDefinitionsRelease()
{
    const ovspatialquery_instance_handle_t instance            = createInstance();
    constexpr uint64_t                     kReplacedDefinition = kHugeSphere + 1u;
    registerSpheres(instance, {kReplacedDefinition}, {0.75});
    std::mt19937_64 random(9u);
    Population      population;
    population.retainedPriorHeads = 0u;
    population.extent             = 8.0;
    createPopulation(instance, population, 64u, random);
    // The first body uses the extra definition in the first, contiguous
    // version; the update rebinds it, and the version's other chunk stays
    // shared by the update.
    population.bodies[2u].definition = kReplacedDefinition;
    publishFirst(instance, population);
    Body rebound       = population.bodies[2u];
    rebound.definition = kUnitSphere;
    publishUpdate(instance, population, {{2u, rebound}}, {});
    REQUIRE(ovspatialquery_destroy_geometry_definition(instance, {OVSPATIALQUERY_GEOMETRY_SPHERE, kReplacedDefinition})
                .status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
using ovspatialquery::detail::kCudaFailureConsumerTransferRestore;
using ovspatialquery::detail::kCudaFailureConsumerTransferRestoreDeviceLost;
using ovspatialquery::detail::kCudaFailurePairReporterAdmission;
using ovspatialquery::detail::kCudaFailurePairReporterComplete;
using ovspatialquery::detail::kCudaFailurePairReporterLaunch;

uint64_t deviceUsage(ovspatialquery_instance_handle_t instance)
{
    uint64_t bytes = UINT64_MAX;
    REQUIRE(ovspatialquery_get_instance_memory_usage(instance, {kDLCUDA, 0}, &bytes).status ==
            OVSPATIALQUERY_API_SUCCESS);
    return bytes;
}

// An active pair capacity too large for the device identity index fails to
// create or grow a CUDA reporter instead of sizing the index forever.
void requireHugeCapacityFails()
{
    std::mt19937_64                        random(61u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    createPopulation(instance, population, 4u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t   target   = populationRef(population.handle);
    ReporterCase                        reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.0);
    ovspatialquery_pair_reporter_desc_t desc{target, {}, target, {}, UINT64_MAX, 1u, 2u, 0.0, resultDevice()};
    REQUIRE(ovspatialquery_create_pair_reporter(instance, &desc, &reporter.handle).status ==
            OVSPATIALQUERY_API_OUT_OF_MEMORY);
    createReporter(instance, reporter);
    REQUIRE(ovspatialquery_grow_pair_reporter_capacity(instance, reporter.handle, UINT64_MAX, reporter.deltaCapacity)
                .status == OVSPATIALQUERY_API_OUT_OF_MEMORY);
    destroyReporter(instance, reporter.handle);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

// Mixed sizes, far coordinates and wide tolerances on a population published
// from host memory and one published through CUDA, with reporters sharing
// their device copies under different views. Their pairs overflow the snapshot
// buffer, so the library's recomputation of every CUDA advance checks them;
// two CPU reporters keep the CPU's patched path under the same inputs. An
// admission that fails after planning its device copies leaves them as they
// were: the first reporter's copy would have been rewritten in place, the view
// reporter's would have gained a membership.
void requireMixedSizes()
{
    std::mt19937_64                        random(17u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             host;
    Population                             device;
    for (Population* population : {&host, &device})
    {
        population->extent = 120.0;
        population->mixed  = true;
    }
    device.cuda               = true;
    device.retainedPriorHeads = 2u;
    createPopulation(instance, host, 3000u, random);
    publishFirst(instance, host);
    createPopulation(instance, device, 1200u, random);
    publishFirst(instance, device);
    std::vector<uint64_t> viewIds;
    for (const std::pair<const uint64_t, Body>& entry : host.bodies)
    {
        if ((entry.first / 2u) % 3u != 0u)
        {
            viewIds.push_back(entry.first);
        }
    }
    const ovspatialquery_view_handle_t view = createView(instance, host, viewIds);

    const ovspatialquery_target_ref_t hostTarget   = populationRef(host.handle);
    const ovspatialquery_target_ref_t deviceTarget = populationRef(device.handle);
    const ovspatialquery_target_ref_t viewed       = viewRef(view);
    const auto add = [](ovspatialquery_target_ref_t first, uint64_t firstBits, Population* firstPopulation,
                        ovspatialquery_target_ref_t second, uint64_t secondBits, Population* secondPopulation,
                        double tolerance, uint64_t activeCapacity = 1u << 20, uint64_t deltaCapacity = 1u << 18)
    {
        ReporterCase reporter = reporterCase(first, firstBits, firstPopulation, second, secondBits, secondPopulation,
                                             tolerance, activeCapacity, deltaCapacity);
        reporter.checked      = false;
        return reporter;
    };
    std::vector<ReporterCase> reporters;
    reporters.push_back(add(hostTarget, 0u, &host, hostTarget, 0u, &host, 0.0));
    reporters.push_back(add(hostTarget, 0u, &host, hostTarget, 0u, &host, 40.0));
    reporters.push_back(add(hostTarget, 1u, &host, hostTarget, 2u, &host, 0.25));
    reporters.push_back(add(viewed, 0u, &host, hostTarget, 3u, &host, 0.1));
    reporters.push_back(add(viewed, 0u, &host, viewed, 0u, &host, 0.0));
    reporters.push_back(add(hostTarget, 0u, &host, deviceTarget, 0u, &device, 0.0));
    reporters.push_back(add(deviceTarget, 0u, &device, deviceTarget, 0u, &device, 0.1));
    reporters.push_back(add(deviceTarget, 2u, &device, hostTarget, 0u, &host, 1.5));
    // Narrow capacities reach both capacity outcomes.
    reporters.push_back(add(hostTarget, 0u, &host, hostTarget, 0u, &host, 0.0, 400u, 12u));
    for (const bool firstIsHost : {true, false})
    {
        ReporterCase cpu = firstIsHost ? add(hostTarget, 0u, &host, hostTarget, 0u, &host, 40.0)
                                       : add(hostTarget, 0u, &host, deviceTarget, 0u, &device, 0.0);
        cpu.device       = {kDLCPU, 0};
        reporters.push_back(cpu);
    }
    for (ReporterCase& reporter : reporters)
    {
        createReporter(instance, reporter);
    }
    const std::array<uint64_t, 3u> outcomesBefore = gOutcomeCounts;
    const uint64_t                 patchedBefore  = patchedAdvanceCount();
    uint64_t                       admittedCount  = 0u;
    for (uint32_t step = 0u; step < 24u; ++step)
    {
        const double fraction = step % 6u == 5u ? 0.4 : 0.01 + 0.02 * static_cast<double>(step % 3u);
        mutate(instance, host, random, static_cast<size_t>(fraction * static_cast<double>(host.bodies.size())));
        // The first advances borrow the device arrays of the CUDA set; its
        // updates publish none, so later copies derive from their records.
        if (step % 2u == 1u)
        {
            mutate(instance, device, random, static_cast<size_t>(fraction * static_cast<double>(device.bodies.size())));
        }
        if (step == 10u)
        {
            // Removing the first body renumbers every position, so a
            // membership a failed admission left behind would not match.
            publishUpdate(instance, host, {}, {host.bodies.begin()->first});
        }
        if (step == 7u || step == 15u)
        {
            // Removals from device memory: one from the middle, then the first body.
            const uint64_t id = step == 7u ? randomExisting(random, device.bodies) : device.bodies.begin()->first;
            publishUpdate(instance, device, {}, {id});
        }
        for (size_t index = 0u; index < reporters.size(); ++index)
        {
            if (step == 10u && (index == 0u || index == 3u))
            {
                ovspatialquery_test_set_cuda_failure_point(kCudaFailurePairReporterAdmission);
                ovspatialquery_pair_reporter_reservation_t reservation{};
                REQUIRE(enqueue(instance, reporters[index], headSelection(), reservation).status ==
                        OVSPATIALQUERY_API_OUT_OF_MEMORY);
            }
            advance(instance, reporters[index], random, admittedCount, true);
        }
    }
    for (size_t outcome = 0u; outcome < gOutcomeCounts.size(); ++outcome)
    {
        REQUIRE(gOutcomeCounts[outcome] != outcomesBefore[outcome]);
    }
    REQUIRE(patchedAdvanceCount() != patchedBefore);
    for (const ReporterCase& reporter : reporters)
    {
        destroyReporter(instance, reporter.handle);
    }
    REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    for (Population* population : {&host, &device})
    {
        REQUIRE(ovspatialquery_destroy_population(instance, population->handle).status == OVSPATIALQUERY_API_SUCCESS);
    }
    destroyInstance(instance);
}

// A population keeps one device copy whatever its retention: the copy of the
// version selected last is rewritten in place, so device usage stays flat.
// A copy that does not fit the device budget fails the advance at admission
// and leaves nothing charged.
void requireCopiesStayBounded()
{
    std::mt19937_64                        random(23u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.retainedPriorHeads = 8u;
    population.extent             = 120.0;
    population.mixed              = true;
    createPopulation(instance, population, 4000u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t target   = populationRef(population.handle);
    ReporterCase                      reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.0);
    createReporter(instance, reporter);
    uint64_t              admittedCount = 0u;
    std::vector<uint64_t> usage;
    for (uint32_t step = 0u; step < 12u; ++step)
    {
        replaceSome(instance, population, random, 40u);
        REQUIRE(advance(instance, reporter, random, admittedCount, true) == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED);
        // Let the reaper retire the advance, so only the copy and the
        // reporter's own storage remain charged.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        usage.push_back(deviceUsage(instance));
    }
    for (size_t index = 2u; index < usage.size(); ++index)
    {
        REQUIRE(usage[index] == usage[1u]);
    }
    destroyReporter(instance, reporter.handle);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);

    // 40,000 objects need 2.5 MB of device copy; the budget holds the
    // reporter but not the copy.
    const ovspatialquery_instance_handle_t tight = createInstance(UINT64_C(2) << 20);
    Population                             large;
    large.retainedPriorHeads = 1u;
    large.extent             = 120.0;
    large.mixed              = true;
    createPopulation(tight, large, 40000u, random);
    publishFirst(tight, large);
    ReporterCase narrow = reporterCase(populationRef(large.handle), 0u, &large, populationRef(large.handle), 0u, &large,
                                       0.0, 1000u, 100u);
    createReporter(tight, narrow, 1u);
    const uint64_t                             before = deviceUsage(tight);
    ovspatialquery_pair_reporter_reservation_t reservation{};
    const ovspatialquery_enqueue_result_t      admission = enqueue(tight, narrow, headSelection(), reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_DEVICE_BUDGET && admission.op_id == OVSPATIALQUERY_INVALID_OP);
    REQUIRE(deviceUsage(tight) == before);
    destroyReporter(tight, narrow.handle);
    REQUIRE(ovspatialquery_destroy_population(tight, large.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(tight);
}

// A view destroyed and replaced, likely at the same address, never lends its
// membership on a version's device copy to the new view.
void requireViewsNotConfused()
{
    std::mt19937_64                        random(37u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.retainedPriorHeads = 2u;
    population.extent             = 120.0;
    population.mixed              = true;
    createPopulation(instance, population, 2000u, random);
    publishFirst(instance, population);
    uint64_t admittedCount = 0u;
    for (const uint64_t modulus : {2u, 3u, 5u})
    {
        // Each view reads the same version, whose copy stays attached.
        std::vector<uint64_t> ids;
        for (const std::pair<const uint64_t, Body>& entry : population.bodies)
        {
            if ((entry.first / 2u) % modulus == 0u)
            {
                ids.push_back(entry.first);
            }
        }
        const ovspatialquery_view_handle_t view = createView(instance, population, ids);
        ReporterCase                       reporter =
            reporterCase(viewRef(view), 0u, &population, populationRef(population.handle), 0u, &population, 0.0);
        createReporter(instance, reporter);
        REQUIRE(advance(instance, reporter, random, admittedCount, true) == OVSPATIALQUERY_PAIR_REPORTER_ADVANCED);
        destroyReporter(instance, reporter.handle);
        REQUIRE(ovspatialquery_destroy_view(instance, view).status == OVSPATIALQUERY_API_SUCCESS);
    }
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

// A result slot is reused only after the consumer the caller handed over has
// finished; until then the next advance reports RESULT_CAPACITY. Once that
// consumer has finished, the slot is free when destroy_op_after_cuda returns:
// the next advance and a capacity grow succeed at once.
void requireSlotWaitsForConsumer()
{
    std::mt19937_64                        random(31u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.retainedPriorHeads = 2u;
    population.extent             = 120.0;
    population.mixed              = true;
    createPopulation(instance, population, 500u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t target = populationRef(population.handle);
    ReporterCase reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.0, 1u << 16, 1u << 14);
    createReporter(instance, reporter, 1u);
    ovspatialquery_pair_reporter_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            admission = enqueue(instance, reporter, headSelection(), reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_pair_reporter_result_t result{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    // Returned device views refuse plain destruction.
    ovspatialquery_op_id_t copy = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op(instance, &copy).status == OVSPATIALQUERY_API_BUSY);
    // A consumer still running holds the slot.
    const CudaDriver::Stream consumerStream = gCuda->createStream();
    const CudaDriver::Event  consumer       = gCuda->createEvent();
    gCuda->enqueueDelay(consumerStream, UINT64_C(400000000));
    gCuda->record(consumer, consumerStream);
    ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumer)};
    ovspatialquery_op_id_t      operation = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status == OVSPATIALQUERY_API_SUCCESS);
    replaceSome(instance, population, random, 10u);
    const ovspatialquery_pair_reporter_advance_t selection = headSelection();
    REQUIRE(ovspatialquery_enqueue_pair_reporter_advance(instance, reporter.handle, &selection, &reservation).status ==
            OVSPATIALQUERY_API_RESULT_CAPACITY);
    REQUIRE(ovspatialquery_destroy_pair_reporter(instance, reporter.handle).status == OVSPATIALQUERY_API_BUSY);
    admission = enqueue(instance, reporter, selection, reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    // An advance whose views were never fetched is destroyed plainly.
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);

    // A consumer that has finished holds nothing. Each round publishes the next
    // head before retiring, as a frame loop does, so nothing but the retired
    // slot stands between the retirement and the next advance or grow. The
    // reaper is held back, so the reporter must reclaim the slot itself.
    REQUIRE(ovspatialquery::detail::setInstanceCudaReaperPausedForTest(instance, true) == OVSPATIALQUERY_API_SUCCESS);
    replaceSome(instance, population, random, 10u);
    for (uint32_t round = 0u; round < 20u; ++round)
    {
        admission = ovspatialquery_enqueue_pair_reporter_advance(instance, reporter.handle, &selection, &reservation);
        REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(ovspatialquery_fetch_pair_reporter(instance, admission.op_id, &result).status ==
                OVSPATIALQUERY_API_SUCCESS);
        replaceSome(instance, population, random, 10u);
        const CudaDriver::Event finished = gCuda->createEvent();
        gCuda->record(finished, gCudaStream);
        gCuda->synchronize(finished);
        ovspatialquery_cuda_event_t finishedDone{reinterpret_cast<uintptr_t>(finished)};
        REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &admission.op_id, &finishedDone).status ==
                OVSPATIALQUERY_API_SUCCESS);
        if (round % 2u == 1u)
        {
            reporter.activeCapacity += 1u;
            reporter.deltaCapacity += 1u;
            REQUIRE(ovspatialquery_grow_pair_reporter_capacity(instance, reporter.handle, reporter.activeCapacity,
                                                               reporter.deltaCapacity)
                        .status == OVSPATIALQUERY_API_SUCCESS);
        }
    }
    admission = ovspatialquery_enqueue_pair_reporter_advance(instance, reporter.handle, &selection, &reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(ovspatialquery::detail::setInstanceCudaReaperPausedForTest(instance, false) == OVSPATIALQUERY_API_SUCCESS);
    destroyReporter(instance, reporter.handle);
    gCuda->destroyStream(consumerStream);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

// A supplied stream 0 is CUDA's legacy default stream, not "no stream"
// (ADR-0060): the advance runs behind the work already queued there. A
// has_stream other than 0 or 1 is refused.
void requireAdvanceRunsOnTheLegacyDefaultStream()
{
    std::mt19937_64                        random(59u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.extent = 8.0;
    createPopulation(instance, population, 64u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t target   = populationRef(population.handle);
    ReporterCase                      reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.1);
    createReporter(instance, reporter);
    ovspatialquery_pair_reporter_reservation_t reservation{};
    ovspatialquery_pair_reporter_advance_t     selection = headSelection();
    selection.cuda_sync.has_stream                       = 2u;
    REQUIRE(enqueue(instance, reporter, selection, reservation).status == OVSPATIALQUERY_API_INVALID_ARGUMENT);
    selection.cuda_sync.has_stream = 1u; // and stream 0
    const CudaDriver::Event held   = gCuda->createEvent();
    gCuda->enqueueDelay(nullptr, UINT64_C(400000000));
    gCuda->record(held, nullptr);
    ovspatialquery_enqueue_result_t admission = enqueue(instance, reporter, selection, reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    // Poll the device event: waiting for the operation reads its outcome back on the host,
    // which waits for the legacy stream whichever stream the advance ran on.
    const auto ready    = reinterpret_cast<CudaDriver::Event>(reservation.result_ready.value);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!gCuda->isComplete(ready))
    {
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // On the reporter's own stream the advance would finish while the delay still holds the legacy stream.
    REQUIRE(gCuda->isComplete(held));
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            wait.state == OVSPATIALQUERY_OP_COMPLETED);
    REQUIRE(ovspatialquery_destroy_op(instance, &admission.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    destroyReporter(instance, reporter.handle);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

// Device loss detected while an advance launches or completes ends it with a
// terminal DEVICE_LOST failure: typed fetch repeatedly reports
// OPERATION_FAILED and leaves its output alone, the advance is destroyed
// without a consumer event, and the reporter admits no later advance.
void requireDeviceLossEndsAdvance(uint32_t failurePoint)
{
    std::mt19937_64                        random(41u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.extent = 8.0;
    createPopulation(instance, population, 64u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t target   = populationRef(population.handle);
    ReporterCase                      reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.1);
    createReporter(instance, reporter);
    ovspatialquery_test_set_cuda_failure_point(failurePoint);
    ovspatialquery_pair_reporter_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            lost = enqueue(instance, reporter, headSelection(), reservation);
    REQUIRE(lost.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, lost.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
                OVSPATIALQUERY_API_SUCCESS &&
            wait.state == OVSPATIALQUERY_OP_FAILURE);
    ovspatialquery_op_status_t status{};
    REQUIRE(ovspatialquery_query_op_status(instance, lost.op_id, &status).status == OVSPATIALQUERY_API_SUCCESS);
    REQUIRE(status.state == OVSPATIALQUERY_OP_FAILURE && status.terminal_status == OVSPATIALQUERY_API_DEVICE_LOST);
    for (uint32_t fetch = 0u; fetch < 2u; ++fetch)
    {
        ovspatialquery_pair_reporter_result_t result{};
        result.accepted_active_pair_count = UINT64_MAX;
        REQUIRE(ovspatialquery_fetch_pair_reporter(instance, lost.op_id, &result).status ==
                OVSPATIALQUERY_API_OPERATION_FAILED);
        REQUIRE(result.accepted_active_pair_count == UINT64_MAX);
    }
    REQUIRE(ovspatialquery_destroy_op(instance, &lost.op_id).status == OVSPATIALQUERY_API_SUCCESS);
    replaceSome(instance, population, random, 2u);
    const ovspatialquery_enqueue_result_t later = enqueue(instance, reporter, headSelection(), reservation);
    REQUIRE(later.status == OVSPATIALQUERY_API_DEVICE_LOST && later.op_id == OVSPATIALQUERY_INVALID_OP);
    destroyReporter(instance, reporter.handle);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}

// Retiring an advance after its consumer restores the caller's context and
// checks the restore: a failed restore fails the retirement and leaves the
// advance and the consumer event with the caller, and a restore that reports
// device loss also ends the reporter's device work.
void requireRetirementChecksContextRestore(uint32_t failurePoint)
{
    std::mt19937_64                        random(53u);
    const ovspatialquery_instance_handle_t instance = createInstance();
    Population                             population;
    population.extent = 8.0;
    createPopulation(instance, population, 64u, random);
    publishFirst(instance, population);
    const ovspatialquery_target_ref_t target   = populationRef(population.handle);
    ReporterCase                      reporter = reporterCase(target, 0u, &population, target, 0u, &population, 0.1);
    createReporter(instance, reporter);
    ovspatialquery_pair_reporter_reservation_t reservation{};
    ovspatialquery_enqueue_result_t            admission = enqueue(instance, reporter, headSelection(), reservation);
    REQUIRE(admission.status == OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_op_wait_result_t wait{};
    REQUIRE(ovspatialquery_wait_op(instance, admission.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait).status ==
            OVSPATIALQUERY_API_SUCCESS);
    ovspatialquery_pair_reporter_result_t result{};
    REQUIRE(ovspatialquery_fetch_pair_reporter(instance, admission.op_id, &result).status ==
            OVSPATIALQUERY_API_SUCCESS);
    const CudaDriver::Event consumer = gCuda->createEvent();
    gCuda->record(consumer, gCudaStream);
    ovspatialquery_cuda_event_t done{reinterpret_cast<uintptr_t>(consumer)};
    ovspatialquery_test_set_cuda_failure_point(failurePoint);
    ovspatialquery_op_id_t operation = admission.op_id;
    REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status ==
            OVSPATIALQUERY_API_INVALID_STATE);
    REQUIRE(operation == admission.op_id && done.value != 0u);
    if (failurePoint == kCudaFailureConsumerTransferRestore)
    {
        // The device was not lost, so the retirement succeeds when retried.
        REQUIRE(ovspatialquery_destroy_op_after_cuda(instance, &operation, &done).status == OVSPATIALQUERY_API_SUCCESS);
        REQUIRE(operation == OVSPATIALQUERY_INVALID_OP && done.value == 0u);
    }
    else
    {
        // A lost advance is destroyed plainly, and the reporter admits no later one.
        REQUIRE(ovspatialquery_destroy_op(instance, &operation).status == OVSPATIALQUERY_API_SUCCESS);
        replaceSome(instance, population, random, 2u);
        REQUIRE(enqueue(instance, reporter, headSelection(), reservation).status == OVSPATIALQUERY_API_DEVICE_LOST);
    }
    destroyReporter(instance, reporter.handle);
    REQUIRE(ovspatialquery_destroy_population(instance, population.handle).status == OVSPATIALQUERY_API_SUCCESS);
    destroyInstance(instance);
}
#endif

} // namespace

int main()
{
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
    CudaDriver cuda;
    gCuda       = &cuda;
    gCudaStream = cuda.createStream();
#endif
    uint64_t admittedCount = 0u;
    requirePatchesWithoutAcceptedVersion(admittedCount);
    requireReplacedDefinitionsRelease();
    for (const uint64_t seed : {11u, 29u, 47u})
    {
        std::mt19937_64                        random(seed);
        const ovspatialquery_instance_handle_t instance = createInstance();

        Population first;
        Population second;
        for (Population* population : {&first, &second})
        {
            createPopulation(instance, *population, population == &first ? 180u : 120u, random);
            publishFirst(instance, *population);
        }

        // Views name identities that do not exist yet, so later insertions
        // join them.
        std::array<ovspatialquery_view_handle_t, 2u> views{};
        for (size_t viewIndex = 0u; viewIndex < views.size(); ++viewIndex)
        {
            std::vector<uint64_t> ids;
            for (uint64_t id = 2u; id < first.nextId + 40u; id += 2u)
            {
                if ((id / 2u + viewIndex) % 3u != 0u)
                {
                    ids.push_back(id);
                }
            }
            views[viewIndex] = createView(instance, first, ids);
        }

        const ovspatialquery_target_ref_t firstTarget  = populationRef(first.handle);
        const ovspatialquery_target_ref_t secondTarget = populationRef(second.handle);
        std::vector<ReporterCase>         reporters(8u);
        // One population against itself, without and with tolerance.
        reporters[0] = reporterCase(firstTarget, 0u, &first, firstTarget, 0u, &first, 0.0);
        reporters[1] = reporterCase(firstTarget, 0u, &first, firstTarget, 0u, &first, 0.25);
        // Overlapping views, and one view against itself.
        reporters[2] = reporterCase(viewRef(views[0]), 0u, &first, viewRef(views[1]), 0u, &first, 0.1);
        reporters[3] = reporterCase(viewRef(views[0]), 0u, &first, viewRef(views[0]), 0u, &first, 0.0);
        // One population under two participation masks.
        reporters[4] = reporterCase(firstTarget, 1u, &first, firstTarget, 2u, &first, 0.0);
        // Distinct populations, with a view and tolerance.
        reporters[5] = reporterCase(firstTarget, 0u, &first, secondTarget, 0u, &second, 0.0);
        reporters[6] = reporterCase(secondTarget, 3u, &second, viewRef(views[1]), 0u, &first, 0.3);
        // Narrow capacities: this reporter reaches both capacity outcomes from
        // an empty baseline, since it never accepts one; that a capacity outcome
        // keeps an accepted baseline is candidate_pairs_test's smallDelta proof.
        reporters[7] = reporterCase(firstTarget, 0u, &first, secondTarget, 0u, &second, 0.2, 200u, 6u);
        for (ReporterCase& reporter : reporters)
        {
            createReporter(instance, reporter);
        }

        for (uint32_t step = 0u; step < 70u; ++step)
        {
            // Up to three publications per step, so an advance may span
            // several versions or fall behind the retained ones.
            const uint32_t publications = static_cast<uint32_t>(random() % 3u);
            for (uint32_t index = 0u; index <= publications; ++index)
            {
                mutate(instance, (random() % 3u) != 0u ? first : second, random);
            }
            for (ReporterCase& reporter : reporters)
            {
                if (random() % 5u != 0u)
                {
                    advance(instance, reporter, random, admittedCount);
                }
            }
            if (step % 10u == 9u)
            {
                ReporterCase& narrow = reporters[7];
                narrow.activeCapacity += 30u;
                narrow.deltaCapacity += 3u;
                ovspatialquery_api_status_t grown = OVSPATIALQUERY_API_BUSY;
                // A CUDA reporter is busy until its last advance's consumers finished.
                for (uint32_t attempt = 0u; grown == OVSPATIALQUERY_API_BUSY; ++attempt)
                {
                    REQUIRE(attempt < 100000u);
                    grown = ovspatialquery_grow_pair_reporter_capacity(instance, narrow.handle, narrow.activeCapacity,
                                                                       narrow.deltaCapacity)
                                .status;
                    std::this_thread::yield();
                }
                REQUIRE(grown == OVSPATIALQUERY_API_SUCCESS);
            }
        }

        for (ReporterCase& reporter : reporters)
        {
            destroyReporter(instance, reporter.handle);
        }
        for (ovspatialquery_view_handle_t handle : views)
        {
            REQUIRE(ovspatialquery_destroy_view(instance, handle).status == OVSPATIALQUERY_API_SUCCESS);
        }
        for (Population* value : {&first, &second})
        {
            REQUIRE(ovspatialquery_destroy_population(instance, value->handle).status == OVSPATIALQUERY_API_SUCCESS);
        }
        destroyInstance(instance);
    }
    const uint64_t patched = patchedAdvanceCount();
    std::printf("pair reporter patch test: %llu advances (%llu advanced, %llu active and %llu delta capacity), "
                "%llu patched\n",
                static_cast<unsigned long long>(admittedCount), static_cast<unsigned long long>(gOutcomeCounts[0]),
                static_cast<unsigned long long>(gOutcomeCounts[1]), static_cast<unsigned long long>(gOutcomeCounts[2]),
                static_cast<unsigned long long>(patched));
    REQUIRE(gOutcomeCounts[1] != 0u && gOutcomeCounts[2] != 0u);
    // Most advances are patched; first advances, removals, insertions between
    // identities, large moves and lagging reporters recompute.
    REQUIRE(patched * 2u > admittedCount && patched < admittedCount);
#if defined(OVSPATIALQUERY_TEST_CUDA_RESULTS)
    requireHugeCapacityFails();
    requireMixedSizes();
    requireCopiesStayBounded();
    requireViewsNotConfused();
    requireSlotWaitsForConsumer();
    requireAdvanceRunsOnTheLegacyDefaultStream();
    requireDeviceLossEndsAdvance(kCudaFailurePairReporterLaunch);
    requireDeviceLossEndsAdvance(kCudaFailurePairReporterComplete);
    requireRetirementChecksContextRestore(kCudaFailureConsumerTransferRestore);
    requireRetirementChecksContextRestore(kCudaFailureConsumerTransferRestoreDeviceLost);
    // Device copies found attached, borrowed from a CUDA publication, rewritten
    // in place, derived by copying another, and uploaded whole.
    const std::array<uint64_t, 5u> copies = ovspatialquery::detail::CudaPairReporterDevice::copyCountsForTest();
    std::printf("device copies found %llu, borrowed %llu, rewritten %llu, derived %llu, uploaded %llu\n",
                static_cast<unsigned long long>(copies[0]), static_cast<unsigned long long>(copies[1]),
                static_cast<unsigned long long>(copies[2]), static_cast<unsigned long long>(copies[3]),
                static_cast<unsigned long long>(copies[4]));
    REQUIRE(std::all_of(copies.begin(), copies.end(), [](uint64_t count) { return count != 0u; }));
#endif
    std::puts("pair reporter patch test passed");
    return 0;
}
