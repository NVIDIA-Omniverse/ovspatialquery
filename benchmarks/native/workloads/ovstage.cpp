/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Stage authoring through the pinned Ovstage SDK, the Ovstage and explicit publication lanes, parity and timing.
#include "ovstage.h"

#include "../installed_adapter.h"
#include "../measurement.h"
#include "cuda_driver.h"
#include "ovstage_support.h"
#if defined(_WIN32)
// <windows.h>, through cuda_driver.h, defines near and far as empty macros.
#undef near
#undef far
#endif

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace bench
{
namespace
{
// The stage is populated at ordinal 1 and its hierarchy derived into ordinal 2, the first sealed read.
constexpr ovstage_ordinal_t initialOrdinal         = 2;
constexpr char              worldMatrixAttribute[] = "omni:fabric:worldMatrix";
constexpr char              worldExtentAttribute[] = "_worldExtent";
constexpr size_t            paritySteps            = 3;

void stageRelease(ovstage_instance_t* stage, ovstage_enqueue_result_t op, const char* action) noexcept
{
    try
    {
        stageOp(stage, op, action);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << " during teardown\n";
        std::exit(EXIT_FAILURE);
    }
}
// USD row-vector layout: the first three rows are the scaled basis images, the last the translation.
// Ovstage keeps omni:fabric:worldMatrix in this same 16-double order. The quaternion is
// renormalized in double first: the installed sphere and capsule rule rejects a rotation
// that is not orthonormal within 1e-12, and a float32 quaternion leaves ~1e-7.
std::array<double, 16> usdMatrix(const Object& ob)
{
    std::array<double, 4> q{ob.q[0], ob.q[1], ob.q[2], ob.q[3]};
    const double          length = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (double& component : q)
        component /= length;
    const auto             m     = rotationMatrix(q);
    const Vec              scale = ob.sphere ? Vec{1, 1, 1} : Vec{2 * boxHalf[0], 2 * boxHalf[1], 2 * boxHalf[2]};
    std::array<double, 16> r{};
    for (size_t axis = 0; axis < 3; ++axis)
        for (size_t k = 0; k < 3; ++k)
            r[axis * 4 + k] = scale[axis] * m[k][axis];
    r[12] = ob.p[0];
    r[13] = ob.p[1];
    r[14] = ob.p[2];
    r[15] = 1;
    return r;
}
// Spheres carry the fixture radius; cubes have unit size and take the half extents from the matrix scale.
std::string usda(const Fixture& f)
{
    std::string out = "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
                      "def Xform \"World\"\n{\n";
    for (const auto& ob : f.objects)
    {
        out += ob.sphere ? "    def Sphere \"p" : "    def Cube \"p";
        out += std::to_string(ob.id);
        out += "\"\n    {\n";
        if (ob.sphere)
        {
            out += "        double radius = ";
            appendNumber(out, sphereRadius);
            out += '\n';
        }
        else
            out += "        double size = 1\n";
        out += "        matrix4d xformOp:transform = (";
        const auto m = usdMatrix(ob);
        for (size_t row = 0; row < 4; ++row)
        {
            out += row ? ", (" : "(";
            for (size_t column = 0; column < 4; ++column)
            {
                if (column)
                    out += ", ";
                appendNumber(out, m[row * 4 + column]);
            }
            out += ')';
        }
        out += ")\n        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n    }\n";
    }
    out += "}\n";
    return out;
}
// The host side: a populated, hierarchy-derived stage whose first `moved` prims are rewritten and
// resealed every step. Synthetic analogue of a cosim host; no cosim source revision was available.
class Stage
{
  public:
    ovstage_instance_t*                    stage      = nullptr;
    path_dictionary_instance_t*            dictionary = nullptr;
    std::vector<ovx_primpath_t>            paths;    // Fixture order; these are the population's object identities.
    std::unordered_map<uint64_t, uint64_t> idByPath; // Prim path value to fixture object identity.
    ovstage_ordinal_t                      sealed = initialOrdinal;
    double                                 openMs = 0, hierarchyMs = 0;
    Stage(const Fixture& f, size_t moved)
    {
        // Value-initialized: Ovstage 0.4 adds `implementation`, which 0.3 lacks.
        ovstage_instance_desc_t desc{};
        desc.name = "ovspatialquery-scene-benchmark";
        require(ovstage_create_instance(&desc, &stage) == OVSTAGE_OK && stage != nullptr, "create ovstage instance");
        dictionary = ovstage_get_path_dictionary(stage);
        require(dictionary != nullptr, "ovstage path dictionary");
        const std::string                         source    = usda(f);
        const auto                                openStart = Clock::now();
        const ovstage_population_enqueue_result_t population =
            openUsdWithPhysics(stage, ovx_string_t{source.data(), source.size()}, 1u, 0.0);
        require(population.status == OVSTAGE_OK, "populate stage from usda");
        ovstage_population_op_wait_result_t populated{};
        require(ovstage_population_wait_op(stage, population.op_index, OVSTAGE_TIMEOUT_INFINITE, &populated) ==
                        OVSTAGE_OK &&
                    populated.error_op_id_count == 0,
                "stage population failed");
        const ovstage_write_floor_desc_t floorOne{1u, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floorOne), "seal ordinal 1");
        const auto hierarchyStart = Clock::now();
        stageOp(stage, ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1u, 2u),
                "compute hierarchy");
        hierarchyMs = elapsed(hierarchyStart);
        const ovstage_write_floor_desc_t floorTwo{initialOrdinal, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floorTwo), "seal ordinal 2");
        openMs = elapsed(openStart);
        std::vector<std::string>  names;
        std::vector<ovx_string_t> strings;
        names.reserve(f.objects.size());
        strings.reserve(f.objects.size());
        for (const auto& ob : f.objects)
        {
            names.push_back("/World/p" + std::to_string(ob.id));
            strings.push_back({names.back().data(), names.back().size()});
        }
        paths.assign(f.objects.size(), OVX_INVALID_PRIMPATH);
        dictionaryOk(dictionary,
                     dictionary->vtable->create_paths_from_strings(dictionary->context, strings.data(), strings.size(),
                                                                   paths.data()),
                     "intern prim paths");
        for (size_t i = 0; i < paths.size(); ++i)
        {
            require(paths[i] != OVX_INVALID_PRIMPATH, "invalid interned prim path");
            require(idByPath.emplace(paths[i].raw_id, f.objects[i].id).second, "duplicate prim path");
        }
        require(moved <= paths.size(), "moved subset exceeds the stage");
        dictionaryOk(
            dictionary,
            dictionary->vtable->create_path_list_from_paths(dictionary->context, paths.data(), moved, &movedList),
            "moved prim path list");
        require(ovstage_query_from_path_list(stage, movedList, &movedQuery) == OVSTAGE_OK &&
                    movedQuery != OVSTAGE_INVALID_QUERY_HANDLE,
                "moved prim query");
        matrices.resize(16 * moved);
        extents.resize(6 * moved);
    }
    ~Stage()
    {
        if (movedQuery != OVSTAGE_INVALID_QUERY_HANDLE)
            stageRelease(stage, ovstage_release_query(stage, movedQuery), "release moved query");
        if (movedList != OVX_INVALID_PRIMPATH_LIST)
        {
            const ovx_api_result_t released =
                dictionary->vtable->release_path_list_reference(dictionary->context, movedList);
            if (released.error.ptr != nullptr)
                dictionary->vtable->release_error(dictionary->context, released.error);
            if (released.status != OVX_API_SUCCESS)
            {
                std::cerr << "release moved path list failed during teardown\n";
                std::exit(EXIT_FAILURE);
            }
        }
        if (stage != nullptr && ovstage_destroy_instance(stage) != OVSTAGE_OK)
        {
            std::cerr << "destroy ovstage instance failed during teardown\n";
            std::exit(EXIT_FAILURE);
        }
    }
    Stage(const Stage&) = delete;
    // Writes the moved prims' world matrices and conservative world extents at `target` and seals it.
    // The world extent is the host's duty on this pin: the adapter reads it as the indexing bound and
    // checks that it encloses the exact geometry, exactly as the integration tests author it.
    void author(const Fixture& f, size_t moved, ovstage_ordinal_t target)
    {
        for (size_t i = 0; i < moved; ++i)
        {
            const auto m = usdMatrix(f.objects[i]);
            std::copy(m.begin(), m.end(), matrices.begin() + ptrdiff_t(16 * i));
            const Aabb bounds = worldBounds(f.objects[i]);
            std::copy(bounds.minimum.begin(), bounds.minimum.end(), extents.begin() + ptrdiff_t(6 * i));
            std::copy(bounds.maximum.begin(), bounds.maximum.end(), extents.begin() + ptrdiff_t(6 * i + 3));
        }
        int64_t  rows = int64_t(moved);
        DLTensor matrixTensor{}, extentTensor{};
        matrixTensor.data               = matrices.data();
        matrixTensor.device.device_type = kDLCPU;
        matrixTensor.ndim               = 1;
        matrixTensor.dtype              = {kDLFloat, 64, 16};
        matrixTensor.shape              = &rows;
        extentTensor                    = matrixTensor;
        extentTensor.data               = extents.data();
        extentTensor.dtype.lanes        = 6;
        ovstage_attribute_write_t writes[2]{};
        writes[0].attribute.string  = text(worldMatrixAttribute);
        writes[0].data.tensors      = &matrixTensor;
        writes[0].data.tensor_count = 1;
        writes[0].data.semantic     = OVSTAGE_SEMANTIC_MATRIX;
        writes[1].attribute.string  = text(worldExtentAttribute);
        writes[1].data.tensors      = &extentTensor;
        writes[1].data.tensor_count = 1;
        writes[1].data.semantic     = OVSTAGE_SEMANTIC_NONE;
        stageOp(stage, ovstage_write_attributes(stage, movedQuery, writes, 2, target, OVSTAGE_PRIM_MODE_UPSERT),
                "write moved poses");
        const ovstage_write_floor_desc_t floor{target, OVSTAGE_SCOPE_ALL, nullptr, 0u};
        stageOp(stage, ovstage_advance_write_floor(stage, &floor), "seal ordinal");
        sealed = target;
    }

  private:
    ovx_primpath_list_t    movedList  = OVX_INVALID_PRIMPATH_LIST;
    ovstage_query_handle_t movedQuery = OVSTAGE_INVALID_QUERY_HANDLE;
    std::vector<double>    matrices, extents;
};
struct ExplicitStep
{
    double      pack = 0; // Row packing the host does before the enqueue.
    Publication publication;
};
ExplicitStep explicitStep(Spatial& s, const Fixture& f, size_t moved)
{
    ExplicitStep                                e;
    const auto                                  packStart = Clock::now();
    InstanceRows                                rows(f, moved);
    Input                                       removed(nullptr, 0, 0, kDLUInt, 64);
    ovspatialquery_explicit_population_update_t change{};
    change.removed_object_values = &removed.value;
    change.base_ordinal          = s.ordinal;
    change.target_ordinal        = s.ordinal + 1;
    change.changed_instances     = rows.table;
    e.pack                       = elapsed(packStart);
    const auto enqueueStart      = Clock::now();
    e.publication =
        finishPublication(s, ovspatialquery_enqueue_explicit_population_update(s.instance, s.population, &change),
                          enqueueStart, f.objects.size(), "explicit update");
    return e;
}
// The explicit device lane (ADR-0024): the same objects as CUDA rows in an ON_DEVICE population, and per
// step the moved rows' IDs and poses uploaded and published as a POSES update. Packing the rows on the
// host and queuing their upload is `pack_ms`; a GPU producer would already hold them on the device.
class ExplicitDeviceLane
{
  public:
    Spatial                   spatial;
    CudaDriver                cuda{CudaDriver::Ownership::tracked};
    CudaDriver::Stream        stream = nullptr;
    CudaDriver::Event         ready  = nullptr;
    CudaDriver::DevicePointer ids = 0, definitions = 0, types = 0, bits = 0, positions = 0, rotations = 0, removed = 0;
    double                    publicationMs = 0;
    explicit ExplicitDeviceLane(const Fixture& f) : spatial(f)
    {
        // Spatial registered the definitions and a host population with a view; the device one replaces both.
        if (spatial.view != OVSPATIALQUERY_INVALID_VIEW)
            check(ovspatialquery_destroy_view(spatial.instance, spatial.view), "destroy host view");
        spatial.view = OVSPATIALQUERY_INVALID_VIEW;
        check(ovspatialquery_destroy_population(spatial.instance, spatial.population), "destroy host population");
        spatial.population                           = OVSPATIALQUERY_INVALID_POPULATION;
        const auto                             start = Clock::now();
        const ovspatialquery_population_desc_t desc{2u, 1u};
        check(ovspatialquery_create_explicit_population(spatial.instance, &desc, &spatial.population),
              "create device population");
        const size_t n = f.objects.size();
        stream         = cuda.createStream();
        ready          = cuda.createEvent();
        ids            = cuda.allocate(n * sizeof(uint64_t));
        definitions    = cuda.allocate(n * sizeof(uint64_t));
        types          = cuda.allocate(n * sizeof(int32_t));
        bits           = cuda.allocate(n * 2u * sizeof(uint64_t));
        positions      = cuda.allocate(n * sizeof(Vec));
        rotations      = cuda.allocate(n * sizeof(Quat));
        removed        = cuda.allocate(sizeof(uint64_t));
        InstanceRows rows(f, n);
        cuda.uploadAsync(ids, rows.ids.data(), n * sizeof(uint64_t), stream);
        cuda.uploadAsync(definitions, rows.definitions.data(), n * sizeof(uint64_t), stream);
        cuda.uploadAsync(types, rows.types.data(), n * sizeof(int32_t), stream);
        cuda.uploadAsync(bits, rows.bits.data(), n * 2u * sizeof(uint64_t), stream);
        cuda.uploadAsync(positions, rows.positions.data(), n * sizeof(Vec), stream);
        cuda.uploadAsync(rotations, rows.rotations.data(), n * sizeof(Quat), stream);
        cuda.record(ready, stream);
        const DLDevice device{kDLCUDA, 0};
        Input          idInput(pointer(ids), n, 0, kDLUInt, 64, device),
            defInput(pointer(definitions), n, 0, kDLUInt, 64, device),
            typeInput(pointer(types), n, 0, kDLInt, 32, device), bitInput(pointer(bits), n, 2, kDLUInt, 64, device),
            posInput(pointer(positions), n, 3, kDLFloat, 64, device),
            rotInput(pointer(rotations), n, 4, kDLFloat, 32, device);
        const ovspatialquery_dynamic_execution_t execution = this->execution();
        ovspatialquery_explicit_population_set_t set{};
        set.target_ordinal               = 1;
        set.instances.object_values      = &idInput.value;
        set.instances.geometry_types     = &typeInput.value;
        set.instances.geometry_values    = &defInput.value;
        set.instances.translations       = &posInput.value;
        set.instances.rotations_xyzw     = &rotInput.value;
        set.instances.participation_bits = &bitInput.value;
        set.dynamic_execution            = &execution;
        const auto enqueueStart          = Clock::now();
        finishPublication(spatial,
                          ovspatialquery_enqueue_explicit_population_set(spatial.instance, spatial.population, &set),
                          enqueueStart, n, "explicit device set");
        publicationMs = elapsed(start);
    }
    ExplicitDeviceLane(const ExplicitDeviceLane&) = delete;
    Publication step(const Fixture& f, size_t moved, size_t objects, double& packMs)
    {
        const auto   packStart = Clock::now();
        InstanceRows rows(f, moved);
        cuda.uploadAsync(ids, rows.ids.data(), moved * sizeof(uint64_t), stream);
        cuda.uploadAsync(positions, rows.positions.data(), moved * sizeof(Vec), stream);
        cuda.uploadAsync(rotations, rows.rotations.data(), moved * sizeof(Quat), stream);
        cuda.record(ready, stream);
        const DLDevice device{kDLCUDA, 0};
        Input          idInput(pointer(ids), moved, 0, kDLUInt, 64, device),
            posInput(pointer(positions), moved, 3, kDLFloat, 64, device),
            rotInput(pointer(rotations), moved, 4, kDLFloat, 32, device),
            removedInput(nullptr, 0, 0, kDLUInt, 64, device);
        const ovspatialquery_dynamic_execution_t    execution = this->execution();
        ovspatialquery_explicit_population_update_t update{};
        update.base_ordinal                     = spatial.ordinal;
        update.target_ordinal                   = spatial.ordinal + 1;
        update.kind                             = OVSPATIALQUERY_EXPLICIT_UPDATE_POSES;
        update.changed_instances.object_values  = &idInput.value;
        update.changed_instances.translations   = &posInput.value;
        update.changed_instances.rotations_xyzw = &rotInput.value;
        update.removed_object_values            = &removedInput.value;
        update.dynamic_execution                = &execution;
        packMs                                  = elapsed(packStart);
        const auto enqueueStart                 = Clock::now();
        return finishPublication(
            spatial, ovspatialquery_enqueue_explicit_population_update(spatial.instance, spatial.population, &update),
            enqueueStart, objects, "explicit device update");
    }

  private:
    static void* pointer(CudaDriver::DevicePointer value)
    {
        return reinterpret_cast<void*>(static_cast<uintptr_t>(value));
    }
    ovspatialquery_dynamic_execution_t execution() const
    {
        return {{kDLCUDA, 0},
                {1u, reinterpret_cast<uintptr_t>(stream), reinterpret_cast<uintptr_t>(ready)},
                OVSPATIALQUERY_PREPARE_ON_DEVICE};
    }
};
// The Ovstage lane: a population over every Sphere and Cube prim of the stage, republished at each sealed ordinal.
class OvstageLane
{
  public:
    Stage       stage;
    Spatial     spatial;
    double      publicationMs = 0;
    Publication initial;
    // Device-prepared publications (ADR-0047): poses through Ovstage's placed read on this stream.
    std::unique_ptr<CudaDriver>               cuda;
    CudaDriver::Stream                        stream = nullptr;
    ovspatialquery_dynamic_execution_t        execution{};
    const ovspatialquery_dynamic_execution_t* onDevice() const { return cuda ? &execution : nullptr; }
    OvstageLane(const Fixture& f, size_t moved, bool device = false) : stage(f, moved), spatial(Spatial::InstanceOnly{})
    {
        if (device)
        {
            cuda      = std::make_unique<CudaDriver>(CudaDriver::Ownership::tracked);
            stream    = cuda->createStream();
            execution = {{kDLCUDA, 0}, {1u, reinterpret_cast<uintptr_t>(stream), 0u}, OVSPATIALQUERY_PREPARE_ON_DEVICE};
        }
        static const ovx_string_t types[] = {text("Sphere"), text("Cube")};
        ovstage_predicate_t       predicate{};
        predicate.attribute.string = text("usd-prim-type");
        predicate.op               = OVSTAGE_FILTER_OP_IN;
        predicate.values           = types;
        predicate.value_count      = 2;
        ovspatialquery_ovstage_population_desc_t desc{};
        desc.stage                            = stage.stage;
        desc.filter                           = ovstage_filter_t{&predicate, 1};
        desc.common.retained_ordinal_capacity = 2;
        const auto start                      = Clock::now();
        check(ovspatialquery_create_ovstage_population(spatial.instance, &desc, &spatial.population),
              "create ovstage population");
        const auto enqueueStart = Clock::now();
        initial                 = finishPublication(spatial,
                                                    ovspatialquery_enqueue_ovstage_population_set_with_execution(
                                        spatial.instance, spatial.population, initialOrdinal, onDevice()),
                                                    enqueueStart, f.objects.size(), "ovstage set");
        publicationMs           = elapsed(start);
    }
    Publication step(const Fixture& f, size_t moved, double& authoringMs)
    {
        const ovstage_ordinal_t target = stage.sealed + 1;
        const auto              start  = Clock::now();
        stage.author(f, moved, target);
        authoringMs             = elapsed(start);
        const auto enqueueStart = Clock::now();
        return finishPublication(spatial,
                                 ovspatialquery_enqueue_ovstage_population_update_with_execution(
                                     spatial.instance, spatial.population, spatial.ordinal, target, onDevice()),
                                 enqueueStart, f.objects.size(), "ovstage update");
    }
    // Ovstage object identities are prim paths and geometry values are adapter-assigned
    // references; parity compares fixture identities and the geometry family.
    void mapIdentities(const Fixture& f, Results& results) const
    {
        for (auto& row : results)
            for (auto& hit : row.hits)
            {
                const auto found = stage.idByPath.find(hit.id);
                require(found != stage.idByPath.end(), "hit identity is not a stage prim path");
                hit.id         = found->second;
                const auto& ob = f.objects.at(size_t(hit.id - 1));
                require(hit.geometryType == ob.geometryType(), "hit geometry family differs from the fixture");
                hit.geometryValue = ob.geometryValue();
            }
    }
};
// Every moved object must answer a vertical ray at its new centre with the top of its own surface:
// an analytic expectation independent of both PhysX and the explicit lane. Rotations are about z,
// so the top of each object's bounds is the top of its surface and a device lane's bounds plan
// answers it too.
void analyticMovedProbe(const Fixture& f, OvstageLane& lane, size_t moved)
{
    Fixture probe = f;
    probe.queries.clear();
    for (size_t i = 0; i < moved; ++i)
    {
        Query q;
        q.p = {f.objects[i].p[0], f.objects[i].p[1], 4};
        probe.queries.push_back(q);
    }
    Results rows(moved);
    if (lane.cuda)
    {
        Spatial::BatchPlan plan(lane.spatial, probe, moved, DLDevice{kDLCUDA, 0},
                                OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
        plan.batch(probe, rows);
    }
    else
        lane.spatial.scalar(probe, rows);
    lane.mapIdentities(f, rows);
    for (size_t i = 0; i < moved; ++i)
    {
        const auto& ob = f.objects[i];
        require(rows[i].hits.size() == 1, "moved prim probe missed");
        const auto& hit = rows[i].hits[0];
        require(hit.id == ob.id && near(hit.distance, 4 - (ob.p[2] + 0.5)) && near(hit.position[0], ob.p[0]) &&
                    near(hit.position[1], ob.p[1]),
                "moved prim probe hit the wrong surface at " + std::to_string(hit.position[0]));
    }
}
void ovstageParity(const Options& o, Fixture& f, size_t moved)
{
    Direct                                direct(f);
    Spatial                               explicitLane(f);
    OvstageLane                           lane(f, moved);
    std::vector<ovspatialquery_ordinal_t> published{lane.initial.result.published_ordinal};
    lane.spatial.makePlan(f);
    // The device lanes, where CUDA is: both derive each object's bounds on the device from its
    // pose and definition, so their bounds rays agree with each other (ADR-0047).
    std::string                         cudaMessage;
    const bool                          cuda = cudaPlansAvailable(cudaMessage);
    std::unique_ptr<ExplicitDeviceLane> explicitDevice;
    std::unique_ptr<OvstageLane>        ovstageDevice;
    if (cuda)
    {
        explicitDevice = std::make_unique<ExplicitDeviceLane>(f);
        ovstageDevice  = std::make_unique<OvstageLane>(f, moved, true);
        explicitDevice->spatial.makePlan(f, true, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
        ovstageDevice->spatial.makePlan(f, true, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
    }
    const auto compareDevice = [&]
    {
        if (!cuda)
            return;
        Results explicitRows(f.queries.size()), ovstageRows(f.queries.size());
        explicitDevice->spatial.batch(f, explicitRows);
        ovstageDevice->spatial.batch(f, ovstageRows);
        ovstageDevice->mapIdentities(f, ovstageRows);
        compare(f, explicitRows, ovstageRows, "explicit-device/ovstage-device bounds rays");
    };
    compareDevice();
    const auto compareAll = [&]
    {
        compareAdapters(f, direct, explicitLane);
        Results expected(f.queries.size()), scalar(f.queries.size()), batch(f.queries.size());
        direct.query(f, expected);
        lane.spatial.scalar(f, scalar);
        lane.spatial.batch(f, batch);
        lane.mapIdentities(f, scalar);
        lane.mapIdentities(f, batch);
        compare(f, expected, scalar, "PhysX/ovstage scalar rays");
        compare(f, expected, batch, "PhysX/ovstage batch rays");
    };
    compareAll();
    for (size_t step = 1; step <= paritySteps; ++step)
    {
        move(f, moved, step);
        direct.update(f, moved);
        explicitLane.update(f, moved);
        double     authoring = 0;
        const auto p         = lane.step(f, moved, authoring);
        require(p.result.published_ordinal == ovspatialquery_ordinal_t(initialOrdinal + step),
                "ovstage published ordinal does not follow the sealed ordinal");
        published.push_back(p.result.published_ordinal);
        compareAll();
        analyticMovedProbe(f, lane, moved);
        if (cuda)
        {
            double     pack = 0;
            const auto e    = explicitDevice->step(f, moved, f.objects.size(), pack);
            const auto d    = ovstageDevice->step(f, moved, authoring);
            require(e.result.published_ordinal == ovspatialquery_ordinal_t(1 + step) &&
                        d.result.published_ordinal == ovspatialquery_ordinal_t(initialOrdinal + step),
                    "device lanes did not publish the next ordinal");
            compareDevice();
            analyticMovedProbe(f, *ovstageDevice, moved);
        }
    }
    uint32_t major = 0, minor = 0, patch = 0;
    ovstage_get_version(lane.stage.stage, &major, &minor, &patch);
    std::cout << "{\"backend\":\"parity\",\"workload\":\"ovstage\",\"objects\":" << o.objects
              << ",\"queries\":" << o.queries << ",\"moved_per_step\":" << moved << ",\"steps\":" << paritySteps
              << ",\"parity\":true,\"analytic_moved_probe\":true,\"device_parity\":\""
              << (cuda ? "parity" : "unavailable: " + cudaMessage) << "\",\"ovstage_version\":[" << major << ','
              << minor << ',' << patch << "],\"published_ordinals\":[";
    for (size_t i = 0; i < published.size(); ++i)
        std::cout << (i ? "," : "") << published[i];
    std::cout << "],\"identity_mapping\":\"prim path to fixture identity through the stage path dictionary\"}\n";
}
} // namespace

int ovstageRun(const Options& o)
{
    Options rayOptions  = o;
    rayOptions.workload = "rays";
    Fixture      f      = fixture(rayOptions);
    const size_t moved  = changedCount(f, o.updatePercent);
    if (o.backend == "parity")
    {
        ovstageParity(o, f, moved);
        return 0;
    }
    // The device lanes (ADR-0047) prepare versions on the device, which a CUDA plan reads as bounds.
    const bool device = o.backend == "explicit-device" || o.backend == "ovstage-device";
    if (device)
    {
        std::string message;
        require(cudaPlansAvailable(message), "the " + o.backend + " lane needs CUDA: " + message);
    }
    const auto                          memoryBaseline = memorySnapshot(OVSPATIALQUERY_INVALID_INSTANCE);
    const auto                          setupStart     = Clock::now();
    std::unique_ptr<Direct>             direct;
    std::unique_ptr<Spatial>            explicitLane;
    std::unique_ptr<ExplicitDeviceLane> explicitDevice;
    std::unique_ptr<OvstageLane>        lane;
    if (o.backend == "physx")
        direct = std::make_unique<Direct>(f);
    else if (o.backend == "explicit")
        explicitLane = std::make_unique<Spatial>(f);
    else if (o.backend == "explicit-device")
        explicitDevice = std::make_unique<ExplicitDeviceLane>(f);
    else
        lane = std::make_unique<OvstageLane>(f, moved, o.backend == "ovstage-device");
    Spatial*     spatial     = explicitLane     ? explicitLane.get()
                               : explicitDevice ? &explicitDevice->spatial
                               : lane           ? &lane->spatial
                                                : nullptr;
    const double setupMs     = elapsed(setupStart);
    const auto   instance    = spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE;
    const auto   memoryReady = memorySnapshot(instance);
    double       planMs      = 0;
    if (spatial)
    {
        const auto planStart = Clock::now();
        if (device)
            spatial->makePlan(f, true, OVSPATIALQUERY_QUERY_BOUNDS_CANDIDATE);
        else
            spatial->makePlan(f);
        planMs = elapsed(planStart);
    }
    const auto  memoryPlanReady = memorySnapshot(instance);
    Results     result(o.queries);
    Observation observation(o.queries);
    for (auto& row : result)
        row.hits.reserve(1);
    const auto query = [&]
    {
        if (direct)
            direct->query(f, result);
        else
            spatial->batch(f, result);
    };
    // First use: the cold ray column against the freshly published head.
    const auto firstUseStart = Clock::now();
    query();
    const double firstUseQueryMs = elapsed(firstUseStart);
    observation.consume(f, result, false);
    const double firstUseCycleMs   = elapsed(firstUseStart);
    const double firstUseChecksum  = observation.checksum;
    observation                    = Observation(o.queries);
    const auto memoryFirstUseReady = memorySnapshot(instance);
    struct StepSample
    {
        double authoring = 0, pack = 0, capture = 0, readiness = 0, update = 0, query = 0, observation = 0;
    };
    std::vector<StepSample> samples;
    for (size_t frame = 0; frame < o.frames + o.warmup; ++frame)
    {
        StepSample s;
        move(f, moved, frame);
        if (direct)
        {
            const auto start = Clock::now();
            direct->update(f, moved);
            s.update = elapsed(start);
        }
        else if (explicitLane)
        {
            const auto e = explicitStep(*explicitLane, f, moved);
            s.pack       = e.pack;
            s.capture    = e.publication.capture;
            s.readiness  = e.publication.readiness;
            s.update     = s.pack + s.capture + s.readiness;
        }
        else if (explicitDevice)
        {
            const auto p = explicitDevice->step(f, moved, f.objects.size(), s.pack);
            s.capture    = p.capture;
            s.readiness  = p.readiness;
            s.update     = s.pack + s.capture + s.readiness;
        }
        else
        {
            const auto p = lane->step(f, moved, s.authoring);
            s.capture    = p.capture;
            s.readiness  = p.readiness;
            s.update     = s.capture + s.readiness;
        }
        auto start = Clock::now();
        query();
        s.query = elapsed(start);
        start   = Clock::now();
        observation.consume(f, result, false);
        s.observation = elapsed(start);
        if (frame >= o.warmup)
            samples.push_back(s);
    }
    const auto memorySteady = memorySnapshot(instance);
    uint32_t   major = 0, minor = 0, patch = 0;
    check(ovspatialquery_get_version(&major, &minor, &patch), "runtime version");
    uint32_t stageMajor = 0, stageMinor = 0, stagePatch = 0;
    if (lane)
        ovstage_get_version(lane->stage.stage, &stageMajor, &stageMinor, &stagePatch);
    std::cout << std::setprecision(12) << "{\"backend\":\"" << o.backend
              << "\",\"workload\":\"ovstage\",\"objects\":" << o.objects << ",\"stage_prims\":" << o.objects
              << ",\"queries\":" << o.queries << ",\"frames\":" << o.frames << ",\"warmup\":" << o.warmup
              << ",\"update_percent\":" << o.updatePercent << ",\"moved_per_step\":" << moved
              << ",\"synthetic_analogue\":true,\"physx_version\":" << PX_PHYSICS_VERSION << ",\"physx_config\":\""
              << BENCH_PHYSX_CONFIG << "\",\"spatial_version\":[" << major << ',' << minor << ',' << patch
              << "],\"ovstage_version\":";
    if (lane)
        std::cout << '[' << stageMajor << ',' << stageMinor << ',' << stagePatch << ']';
    else
        std::cout << "null";
    std::cout
        << ",\"host_budget_bytes\":" << hostBudget << ",\"query_threads\":1,\"setup_ms\":" << setupMs
        << ",\"stage_open_ms\":" << (lane ? std::to_string(lane->stage.openMs) : "null")
        << ",\"stage_hierarchy_ms\":" << (lane ? std::to_string(lane->stage.hierarchyMs) : "null")
        << ",\"geometry_setup_ms\":"
        << (direct           ? direct->geometrySetupMs
            : explicitLane   ? explicitLane->geometrySetupMs
            : explicitDevice ? explicitDevice->spatial.geometrySetupMs
                             : 0)
        << ",\"publication_ms\":"
        << (direct           ? direct->publicationMs
            : explicitLane   ? explicitLane->publicationMs
            : explicitDevice ? explicitDevice->publicationMs
                             : lane->publicationMs)
        << ",\"initial_capture_ms\":" << (lane ? std::to_string(lane->initial.capture) : "null")
        << ",\"initial_readiness_ms\":" << (lane ? std::to_string(lane->initial.readiness) : "null")
        << ",\"plan_ms\":" << planMs << ",\"first_use_cycle_ms\":" << firstUseCycleMs
        << ",\"first_use_query_ms\":" << firstUseQueryMs << ",\"first_use_checksum\":" << firstUseChecksum
        << ",\"update_scope\":\""
        << (direct         ? "setGlobalPose on the moved actors and flushQueryUpdates"
            : explicitLane ? "pack_ms packs the moved rows; capture_ms is the explicit update enqueue; readiness_ms "
                             "is wait, fetch and operation destruction until the head is published"
            : explicitDevice
                ? "pack_ms packs the moved rows and queues their upload; capture_ms is the ON_DEVICE POSES "
                  "update enqueue; readiness_ms is wait, fetch and operation destruction until the head is "
                  "published; query_ms is a CUDA bounds ray plan"
            : lane->cuda
                ? "capture_ms is the ON_DEVICE update enqueue, which compares the membership on the host and "
                  "reads every world matrix at the target on the device; readiness_ms is wait, fetch and operation "
                  "destruction until the head is published; query_ms is a CUDA bounds ray plan; authoring_ms is "
                  "the host's stage write and seal and is outside frame_ms"
                : "capture_ms is the enqueue call that reads the sealed ordinal and acquires typed geometry "
                  "synchronously; readiness_ms is wait, fetch and operation destruction until the head is "
                  "published; authoring_ms is the host's stage write and seal and is outside frame_ms")
        << "\",\"memory\":";
    printMemoryRecord(spatial != nullptr, lane ? "the process also holds the Ovstage runtime and stage" : "",
                      memoryBaseline,
                      {{"ready", memoryReady, true},
                       {"plan_ready", memoryPlanReady, true},
                       {"first_use_ready", memoryFirstUseReady, false},
                       {"steady", memorySteady, true}});
    std::cout << ",\"checksum\":" << observation.checksum << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& s = samples[i];
        std::cout << (i ? "," : "") << "{\"authoring_ms\":" << s.authoring << ",\"pack_ms\":" << s.pack
                  << ",\"capture_ms\":" << s.capture << ",\"readiness_ms\":" << s.readiness
                  << ",\"update_ms\":" << s.update << ",\"query_ms\":" << s.query
                  << ",\"observation_ms\":" << s.observation << ",\"frame_ms\":" << s.update + s.query + s.observation
                  << '}';
    }
    std::cout << "]}\n";
    return 0;
}
} // namespace bench
