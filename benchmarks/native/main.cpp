/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Native scene-query comparison against the verified installed library.
// Isaac: closest sensor rays, occupancy box ANY, proximity box MULTIPLE.
// Actor: four-probe grounding, immediate support, dependent overlap/sweep avoidance.
// Mesh terrain: shared/unique triangle definitions, grounding and sphere sweeps,
// with explicit sidedness, edge, clipping, overlap and update parity probes.
// Synthetic point cloud: coherent angular tiles, mixed geometry, compact float32 XYZ.
// Pairs: two populations on their own cadence, AABB pair snapshots and reporter deltas (co-simulation workload).
// Ovstage: one sealed stage republished per step through the public Ovstage population API (host analogue).
// Deform: fixed-topology sheets deforming per step, republished through Ovstage, with direct PhysX recook/refit.
// Adapters and measurement code are shared; no Kit, live actor, dynamics or renderer.
// This file owns option dispatch, the timed frame loop and the JSON record.
#include "installed_adapter.h"
#include "measurement.h"
#include "workloads/actor.h"
#include "workloads/camera.h"
#include "workloads/convex.h"
#include "workloads/deform.h"
#include "workloads/isaac.h"
#include "workloads/mesh.h"
#include "workloads/ovstage.h"
#include "workloads/pairs.h"
#include "workloads/sensor.h"
#include "workloads/statics.h"
#include "workloads/warehouse.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifndef _WIN32
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#endif

namespace bench
{
int run(int argc, char** argv)
{
    const Options o = options(argc, argv);
    registerMeshSourceFaces(o.meshSourceFaces != 0);
    require(o.backend != "cuda" || !actorWorkload(o), "the cuda backend covers reusable batch-plan workloads only");
    if (o.workload.rfind("pairs", 0) == 0)
        return pairsRun(o);
    if (o.workload == "ovstage")
        return ovstageRun(o);
    if (deformWorkload(o))
        return deformRun(o);
    if (o.workload == "mesh-parity")
    {
        meshSemanticSmoke();
        return 0;
    }
    auto f = warehouseWorkload(o)                 ? warehouseFixture(o)
             : cameraWorkload(o)                  ? cameraFixture(o)
             : o.workload == "sensor-point-cloud" ? sensorFixture(o)
             : o.workload.find("convex-") == 0    ? convexFixture(o)
             : staticsWorkload(o)                 ? staticsFixture(o)
             : actorWorkload(o)
                 ? actorFixture(o)
                 : (o.workload == "mesh-sweeps" || o.workload == "mesh-box-sweeps" ? meshFixture(o) : fixture(o));
    if (o.workload == "mesh-sweeps" || o.workload == "mesh-box-sweeps")
        meshQueries(f, 0);
    if (o.backend == "parity" && f.pointCloud)
    {
        sensorParity(o, f);
        return 0;
    }
    if (o.backend == "parity" && actorWorkload(o))
    {
        Direct  direct(f);
        Spatial spatial(f);
        Options directOptions = o, scalarOptions = o, batchOptions = o;
        directOptions.backend = "physx";
        scalarOptions.backend = "scalar";
        batchOptions.backend  = "batch";
        ActorCycle d(directOptions, &direct, nullptr), s(scalarOptions, nullptr, &spatial),
            b(batchOptions, nullptr, &spatial);
        for (size_t frame = 0; frame < 5; ++frame)
        {
            const size_t count = changedCount(f, o.updatePercent);
            move(f, count, frame);
            direct.update(f, count);
            spatial.update(f, count);
            d.prepare(f, frame);
            s.prepare(f, frame);
            b.prepare(f, frame);
            d.execute(f, true);
            s.execute(f, true);
            b.execute(f, true);
            d.compareWith(s);
            d.compareWith(b);
            d.commitFrame();
            s.commitFrame();
            b.commitFrame();
        }
        std::cout << "{\"backend\":\"parity\",\"parity\":true,\"strict_nominal_contact_parity\":true,\"actor_frames\":5"
                  << "}\n";
        return 0;
    }
    if (o.backend == "parity" && (warehouseWorkload(o) || cameraWorkload(o)))
    {
        Direct  direct(f);
        Spatial spatial(f);
        // Direct PhysX is compared on identity and distance (its float contacts are approximate on
        // these meshes); the installed CPU paths and the CUDA plan are compared with each other in
        // full, box contacts through the independent solve of each reported face, and every ray
        // contact is checked against the analytic point on the ray.
        const std::string kind     = kindName(f.kind);
        const bool        baseline = !f.approximateBaselineContacts;
        Results           expected(f.queries.size()), scalar(f.queries.size()), cpu(f.queries.size());
        CompareStats      physxScalar, physxBatch, scalarBatch, physxCuda, cpuCuda;
        direct.query(f, expected);
        spatial.scalar(f, scalar);
        spatial.makePlan(f);
        spatial.batch(f, cpu);
        compare(f, expected, scalar, "PhysX/scalar " + kind, false, baseline, &physxScalar, true);
        compare(f, expected, cpu, "PhysX/batch " + kind, false, baseline, &physxBatch, true);
        compare(f, scalar, cpu, "scalar/batch " + kind, false, true, &scalarBatch);
        require(scalarBatch.missedByFirst == 0, "the installed scalar sweep missed a contact the CPU plan found");
        std::string cudaParity;
        if (cudaPlansAvailable(cudaParity))
        {
            Results gpu(f.queries.size());
            spatial.makePlan(f, true);
            spatial.batch(f, gpu);
            compare(f, expected, gpu, "PhysX/CUDA " + kind, false, baseline, &physxCuda, true);
            compare(f, cpu, gpu, "CPU batch/CUDA " + kind, false, true, &cpuCuda);
            require(cpuCuda.missedByFirst == 0, "the installed CPU sweep missed a contact the CUDA plan found");
            // The kernel's double-precision ray contacts must match the analytic face evaluation, every
            // hit row included, and lie on the ray regardless of the tolerance declared for PhysX float
            // contacts; `compare` already holds installed sphere and box sweep contacts to their oracles.
            if (f.kind == QueryKind::rays)
            {
                size_t hitRows = 0;
                // Only mesh hits have an analytic face; the camera box is compared above.
                for (const auto& row : gpu)
                    hitRows += !row.hits.empty() && row.hits[0].geometryType == OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH;
                require(cpuCuda.maxOffRaySecond <= 1e-4 + 1e-6 * 15, "CUDA ray contact left the ray");
                require(cpuCuda.analyticSecond == hitRows && cpuCuda.maxDistanceErrorSecond <= 2e-5 + 2e-6 * 15,
                        "CUDA contact distance is not analytic: largest error " +
                            std::to_string(cpuCuda.maxDistanceErrorSecond) + " over " +
                            std::to_string(cpuCuda.analyticSecond) + " of " + std::to_string(hitRows) + " hit rows");
            }
            cudaParity = "parity";
        }
        else
            cudaParity = "unavailable: " + cudaParity;
        size_t definitionTriangles = 0;
        for (const auto& mesh : f.meshes)
            definitionTriangles += mesh.triangles.size();
        std::cout << "{\"backend\":\"parity\",\"workload\":\"" << o.workload << "\",\"instances\":" << o.objects
                  << ",\"queries\":" << o.queries << ",\"mesh_definitions\":" << f.meshes.size()
                  << ",\"definition_triangles\":" << definitionTriangles
                  << ",\"instanced_triangles\":" << warehouseInstancedTriangles(f)
                  << ",\"parity\":true,\"sweep_relative_tolerance\":" << f.sweepRelativeTolerance
                  << ",\"contact_tolerance\":" << f.contactTolerance;
        if (f.kind == QueryKind::boxSweep)
            std::cout << ",\"off_ray_max_m\":null,\"analytic_distance_error_max_m\":{\"physx\":null,\"cpu_scalar\":"
                      << physxScalar.maxDistanceErrorSecond << ",\"cpu_batch\":" << physxBatch.maxDistanceErrorSecond
                      << ",\"cuda\":" << cpuCuda.maxDistanceErrorSecond << "},\"box_baseline_early_max_m\":"
                      << std::max(
                             {physxScalar.maxBaselineEarly, physxBatch.maxBaselineEarly, physxCuda.maxBaselineEarly})
                      << ",\"box_baseline_early_tolerance\":" << f.boxSweepBaselineEarlyTolerance;
        else
            std::cout << ",\"off_ray_max_m\":{\"physx\":"
                      << std::max(physxScalar.maxOffRayFirst, physxCuda.maxOffRayFirst)
                      << ",\"cpu_scalar\":" << physxScalar.maxOffRaySecond
                      << ",\"cpu_batch\":" << physxBatch.maxOffRaySecond << ",\"cuda\":" << cpuCuda.maxOffRaySecond
                      << "},\"analytic_distance_error_max_m\":{\"physx\":"
                      << std::max(physxScalar.maxDistanceErrorFirst, physxCuda.maxDistanceErrorFirst)
                      << ",\"cpu_scalar\":" << physxScalar.maxDistanceErrorSecond
                      << ",\"cpu_batch\":" << physxBatch.maxDistanceErrorSecond
                      << ",\"cuda\":" << cpuCuda.maxDistanceErrorSecond
                      << "},\"analytic_rows\":{\"cpu_scalar\":" << physxScalar.analyticSecond
                      << ",\"cpu_batch\":" << physxBatch.analyticSecond << ",\"cuda\":" << cpuCuda.analyticSecond
                      << "}";
        std::cout << ",\"missed_contacts\":{\"physx_vs_cpu\":" << physxBatch.missedByFirst
                  << ",\"physx_vs_cuda\":" << physxCuda.missedByFirst << ",\"cpu_vs_cuda\":" << cpuCuda.missedByFirst
                  << "}"
                  << ",\"baseline_contact_fields_compared\":" << (f.approximateBaselineContacts ? "false" : "true")
                  << ",\"cuda_parity\":\"" << cudaParity << "\",\"physx_version\":" << PX_PHYSICS_VERSION
                  << ",\"physx_config\":\"" << BENCH_PHYSX_CONFIG << "\"}\n";
        return 0;
    }
    if (o.backend == "parity")
    {
        // The CUDA plan, where one can be created, is compared in full with a CPU plan on the same
        // instance at every published ordinal; both plans are kept across the updates.
        std::string cudaParity;
        const bool  cuda = cudaPlansAvailable(cudaParity);
        cudaParity       = cuda ? "parity" : "unavailable: " + cudaParity;
        {
            Direct                              direct(f);
            Spatial                             spatial(f);
            std::unique_ptr<Spatial::BatchPlan> cpuPlan, cudaPlan;
            if (cuda)
            {
                cpuPlan  = std::make_unique<Spatial::BatchPlan>(spatial, f, f.queries.size());
                cudaPlan = std::make_unique<Spatial::BatchPlan>(spatial, f, f.queries.size(), DLDevice{kDLCUDA, 0});
            }
            auto compareCuda = [&]
            {
                if (!cuda)
                    return;
                Results cpu(f.queries.size()), gpu(f.queries.size());
                cpuPlan->batch(f, cpu);
                cudaPlan->batch(f, gpu);
                compare(f, cpu, gpu, "CPU batch/CUDA " + std::string(kindName(f.kind)));
            };
            compareAdapters(f, direct, spatial);
            compareCuda();
            const size_t count = changedCount(f, o.updatePercent ? o.updatePercent : 10);
            for (size_t frame = 0; frame < std::min(o.frames, size_t(3)); ++frame)
            {
                move(f, count, frame);
                direct.update(f, count);
                spatial.update(f, count);
                compareAdapters(f, direct, spatial);
                compareCuda();
            }
        }
        semanticSmoke();
        std::cout << "{\"backend\":\"parity\",\"workload\":\"" << o.workload << "\",\"objects\":" << o.objects
                  << ",\"queries\":" << o.queries << ",\"parity\":true,\"cuda_parity\":\"" << cudaParity
                  << "\",\"edge_smoke\":true,\"physx_version\":" << PX_PHYSICS_VERSION << ",\"physx_config\":\""
                  << BENCH_PHYSX_CONFIG
                  << "\",\"ray_parity_scope\":\"ray origins and near planes exterior to "
                     "primitives\",\"inside_origin_probes\":[";
        const char* names[] = {"near_inside_box", "near_inside_sphere", "origin_inside_box", "origin_inside_sphere"};
        for (size_t i = 0; i < insideProbes.size(); ++i)
        {
            if (i)
                std::cout << ',';
            std::cout << "{\"case\":\"" << names[i] << "\",\"physx\":";
            printHit(insideProbes[i].direct);
            std::cout << ",\"spatial_min_distance\":";
            printHit(insideProbes[i].spatial);
            std::cout << ",\"spatial_shifted_origin\":";
            printHit(insideProbes[i].shifted);
            std::cout << ",\"spatial_point_composition\":";
            printHit(insideProbes[i].composed);
            std::cout << ",\"scalar_batch_equal\":true,\"point_composition_parity\":true}";
        }
        std::cout << "]}\n";
        return 0;
    }
    const MemorySnapshot     memoryBaseline = memorySnapshot(OVSPATIALQUERY_INVALID_INSTANCE);
    const auto               setupStart     = Clock::now();
    std::unique_ptr<Direct>  direct;
    std::unique_ptr<Spatial> spatial;
    if (o.backend == "physx")
        direct = std::make_unique<Direct>(f);
    else
        spatial = std::make_unique<Spatial>(f);
    const double         setupMs     = elapsed(setupStart);
    const MemorySnapshot memoryReady = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
    double               planMs      = 0;
    std::optional<MemorySnapshot> memoryPlanReady;
    if ((o.backend == "batch" || o.backend == "cuda") && !actorWorkload(o))
    {
        const auto start = Clock::now();
        spatial->makePlan(f, o.backend == "cuda");
        planMs          = elapsed(start);
        memoryPlanReady = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
    }
    std::unique_ptr<ActorCycle> actor;
    if (actorWorkload(o))
        actor = std::make_unique<ActorCycle>(o, direct.get(), spatial.get());
    Results result(o.queries);
    for (auto& row : result)
        row.hits.reserve(f.kind == QueryKind::zones ? std::min(o.objects, size_t(128)) : 1);
    Observation                       observation(o.queries);
    std::optional<PointCloud>         cloud;
    std::vector<std::array<float, 3>> localBeams;
    if (f.pointCloud)
    {
        cloud.emplace(o.queries);
        localBeams.reserve(o.queries);
    }
    std::vector<Sample>           samples;
    std::optional<double>         firstUseCycleMs, firstUseQueryMs, firstUseChecksum;
    std::optional<MemorySnapshot> memoryBeforeFirstUse, memoryFirstUseReady;
    if (o.workload == "grounding-mesh" || !f.convexes.empty() || warehouseWorkload(o))
    {
        // Measure first query readiness before warming any backend caches or reusable plans.
        // Capture the complete first cycle separately from steady-state warmup.
        memoryBeforeFirstUse     = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
        const auto firstUseStart = Clock::now();
        if (actor)
        {
            actor->prepare(f, 0);
            actor->execute(f);
            firstUseQueryMs  = actor->overlapMs; // Ray stage includes lazy plan creation.
            firstUseChecksum = actor->checksum;
            actor->checksum  = 0;
        }
        else
        {
            if (direct)
                direct->query(f, result);
            else if (o.backend == "scalar")
                spatial->scalar(f, result);
            else
                spatial->batch(f, result);
            firstUseQueryMs = elapsed(firstUseStart);
            if (cloud)
            {
                cloud->compact(result);
                firstUseChecksum = cloud->checksum;
                cloud->checksum  = 0;
            }
            else
            {
                observation.consume(f, result, false);
                firstUseChecksum = observation.checksum;
                observation      = Observation(o.queries);
            }
        }
        firstUseCycleMs = elapsed(firstUseStart);
        if (cloud)
        {
            cloud->qualify(f, result);
            require(cloud->xyz.size() < o.queries, "sensor fixture has no misses");
            for (size_t hits : cloud->familyHits)
                require(hits > 0, "sensor fixture failed to exercise a geometry family");
        }
        memoryFirstUseReady = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
    }
    const size_t count = changedCount(f, o.updatePercent);
    for (size_t frame = 0; frame < o.frames + o.warmup; ++frame)
    {
        Sample s;
        if (count)
        {
            move(f, count, frame);
            const auto start = Clock::now();
            if (direct)
                direct->update(f, count);
            else
                spatial->update(f, count);
            s.update = elapsed(start);
        }
        if (cloud)
        {
            const auto frameStart = Clock::now();
            for (size_t iteration = 0; iteration < o.iterations; ++iteration)
            {
                auto start = Clock::now();
                sensorBeams(localBeams, o.queries);
                s.beamGeneration += elapsed(start);
                start = Clock::now();
                sensorPose(f, localBeams);
                s.poseTransform += elapsed(start);
                start = Clock::now();
                if (o.backend == "batch" || o.backend == "cuda")
                    spatial->defaultPlan->pack(f);
                s.pack += elapsed(start);
                start = Clock::now();
                if (direct)
                    direct->query(f, result);
                else if (o.backend == "scalar")
                    spatial->scalar(f, result, spatial->ordinal);
                else
                    spatial->batch(f, result, spatial->ordinal);
                s.query += elapsed(start);
                start = Clock::now();
                cloud->compact(result);
                s.observation += elapsed(start);
            }
            s.completeFrame = elapsed(frameStart) / double(o.iterations);
            s.beamGeneration /= double(o.iterations);
            s.poseTransform /= double(o.iterations);
            s.pack /= double(o.iterations);
            s.query /= double(o.iterations);
            s.observation /= double(o.iterations);
            observation.checksum = cloud->checksum;
            if (frame >= o.warmup)
                samples.push_back(s);
            continue;
        }
        // A phase block amortizes clock overhead even for a single-ray fixture.
        const auto queryStart = Clock::now();
        for (size_t iteration = 0; iteration < o.iterations; ++iteration)
        {
            if (actor)
            {
                const auto packStart = Clock::now();
                actor->prepare(f, frame);
                s.pack += elapsed(packStart);
                actor->execute(f);
                s.overlap += actor->overlapMs;
                s.sweep += actor->sweepMs;
                s.ray += actor->rayMs;
                s.reduction += actor->reductionMs;
                s.overlapHits = actor->overlapHits;
                s.sweeps      = actor->sweepCount;
                s.decisions   = actor->decisionCount;
                s.retries += actor->retries;
            }
            else if (direct)
                direct->query(f, result);
            else if (o.backend == "scalar")
                spatial->scalar(f, result);
            else
            {
                spatial->batch(f, result);
                s.cudaEnqueue += spatial->defaultPlan->last.enqueue;
                s.cudaWait += spatial->defaultPlan->last.wait;
                s.cudaFetch += spatial->defaultPlan->last.fetch;
                s.cudaReadback += spatial->defaultPlan->last.readback;
            }
        }
        s.query = elapsed(queryStart) / double(o.iterations);
        s.cudaEnqueue /= double(o.iterations);
        s.cudaWait /= double(o.iterations);
        s.cudaFetch /= double(o.iterations);
        s.cudaReadback /= double(o.iterations);
        const auto observationStart = Clock::now();
        for (size_t iteration = 0; !actor && iteration < o.iterations; ++iteration)
            observation.consume(f, result, iteration == 0);
        s.observation = actor ? 0 : elapsed(observationStart) / double(o.iterations);
        if (actor)
        {
            actor->commitFrame();
            s.overlap /= double(o.iterations);
            s.sweep /= double(o.iterations);
            s.ray /= double(o.iterations);
            s.reduction /= double(o.iterations);
            s.pack /= double(o.iterations);
            observation.checksum = actor->checksum;
        }
        if (actor && o.backend == "batch" && frame + 1 == o.warmup)
            memoryPlanReady = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
        if (frame >= o.warmup)
            samples.push_back(s);
    }
    const MemorySnapshot memorySteady = memorySnapshot(spatial ? spatial->instance : OVSPATIALQUERY_INVALID_INSTANCE);
    size_t               convexSourceVertices = 0;
    for (const auto& vertices : f.convexes)
        convexSourceVertices += vertices.size();
    size_t definitionTriangles = 0;
    for (const auto& mesh : f.meshes)
        definitionTriangles += mesh.triangles.size();
    uint32_t major = 0, minor = 0, patch = 0;
    check(ovspatialquery_get_version(&major, &minor, &patch), "runtime version");
    std::cout << std::setprecision(12) << "{\"backend\":\"" << o.backend << "\",\"workload\":\"" << o.workload
              << "\",\"objects\":" << o.objects << ",\"queries\":" << o.queries << ",\"frames\":" << o.frames
              << ",\"convex_definitions\":" << f.convexes.size()
              << ",\"convex_source_vertices\":" << convexSourceVertices << ",\"mesh_definitions\":" << f.meshes.size()
              << ",\"triangles_per_definition\":" << (f.meshes.empty() ? 0 : f.meshes[0].triangles.size())
              << ",\"definition_triangles\":" << definitionTriangles
              << ",\"instanced_triangles\":" << (f.meshes.empty() ? size_t(0) : warehouseInstancedTriangles(f))
              << ",\"mesh_side\":" << o.meshSide << ",\"unique_meshes\":" << o.uniqueMeshes
              << ",\"mesh_source_faces\":" << o.meshSourceFaces << ",\"zone_hit_capacity\":" << o.zoneHitCapacity
              << ",\"actor_hit_capacity\":" << o.actorHitCapacity << ",\"reuse_plans\":" << o.reusePlans
              << ",\"radius_groups\":" << o.radiusGroups << ",\"memory\":";
    std::optional<MemoryDelta> firstUseDelta;
    if (memoryFirstUseReady && memoryBeforeFirstUse)
        firstUseDelta = memoryDelta(*memoryFirstUseReady, *memoryBeforeFirstUse);
    printMemoryRecord(spatial != nullptr, nullptr, memoryBaseline,
                      {{"ready", memoryReady, true},
                       {"plan_ready", memoryPlanReady, true},
                       {"before_first_use", memoryBeforeFirstUse, false},
                       {"first_use_ready", memoryFirstUseReady, false},
                       {"steady", memorySteady, true}},
                      firstUseDelta);
    std::cout << ",\"block_iterations\":" << o.iterations
              << ",\"physx_performance_warnings\":" << (direct ? direct->warningCount() : 0)
              << ",\"geometry_instances\":" << f.objects.size() << ",\"warmup\":" << o.warmup << ",\"subset\":\""
              << o.subset << "\",\"updated_objects\":" << count << ",\"physx_version\":" << PX_PHYSICS_VERSION
              << ",\"physx_config\":\"" << BENCH_PHYSX_CONFIG << "\",\"spatial_version\":[" << major << ',' << minor
              << ',' << patch << "]"
              << ",\"host_budget_bytes\":" << hostBudget << ",\"query_threads\":1,\"setup_ms\":" << setupMs
              << ",\"geometry_setup_ms\":" << (direct ? direct->geometrySetupMs : spatial->geometrySetupMs)
              << ",\"publication_ms\":" << (direct ? direct->publicationMs : spatial->publicationMs)
              << ",\"plan_ms\":" << (actor ? "null" : std::to_string(planMs)) << ",\"first_use_cycle_ms\":";
    printOptionalValue(firstUseCycleMs);
    std::cout << ",\"first_use_query_ms\":";
    printOptionalValue(firstUseQueryMs);
    std::cout << ",\"first_use_checksum\":";
    printOptionalValue(firstUseChecksum);
    std::cout << ",\"first_use_scope\":"
              << (firstUseCycleMs
                      ? "\"first cycle before warmup, including any lazy plans/cooking, query and observation\""
                      : "null")
              << ",\"checksum\":" << observation.checksum;
    if (cloud)
    {
        std::cout
            << ",\"synthetic_geometric_analogue\":true,\"angular_tiles\":6,\"near_distance\":0.25,"
               "\"far_distance\":10,\"sensor_pose\":\"translation(2,-3,1), Z rotation 180 degrees\","
               "\"requested_fields\":[\"position\",\"distance\"],\"point_count\":"
            << cloud->xyz.size() << ",\"miss_count\":" << o.queries - cloud->xyz.size()
            << ",\"family_order\":[\"triangle_mesh\",\"box\",\"sphere\",\"capsule\",\"convex\"],\"family_hits\":[";
        for (size_t i = 0; i < cloud->familyHits.size(); ++i)
            std::cout << (i ? "," : "") << cloud->familyHits[i];
        std::cout << "],\"point_cloud_storage\":\"reused uint8 mask and reserved float32 XYZ in beam order\","
                     "\"query_cost_scope\":\"adapter calls and shared Results vector materialization; scalar returns "
                     "full contact\","
                     "\"beam_policy\":\"regenerate local angular lattice and transform each iteration\"";
    }
    std::cout << ",\"samples\":[";
    for (size_t i = 0; i < samples.size(); ++i)
    {
        const auto& s = samples[i];
        if (i)
            std::cout << ',';
        std::cout << "{\"update_ms\":" << s.update << ",\"query_ms\":" << s.query
                  << ",\"observation_ms\":" << s.observation
                  << ",\"frame_ms\":" << (cloud ? s.completeFrame : s.update + s.query + s.observation)
                  << ",\"pack_ms\":" << s.pack << ",\"overlap_ms\":" << s.overlap << ",\"sweep_ms\":" << s.sweep
                  << ",\"ray_ms\":" << s.ray << ",\"reduction_ms\":" << s.reduction
                  << ",\"cuda_enqueue_ms\":" << s.cudaEnqueue << ",\"cuda_wait_ms\":" << s.cudaWait
                  << ",\"cuda_fetch_ms\":" << s.cudaFetch << ",\"cuda_readback_ms\":" << s.cudaReadback
                  << ",\"overlap_hits\":" << s.overlapHits << ",\"sweeps\":" << s.sweeps
                  << ",\"decisions\":" << s.decisions << ",\"retries\":" << s.retries;
        if (cloud)
            std::cout << ",\"beam_generation_ms\":" << s.beamGeneration << ",\"pose_transform_ms\":" << s.poseTransform
                      << ",\"compaction_ms\":" << s.observation;
        std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
}
#ifndef _WIN32
// A crash, often at exit while Ovstage runtime threads are still alive, otherwise leaves CTest
// only "Exception: SegFault". Print the crashing thread's frames as module+offset (resolve with
// addr2line against the same build), then re-raise so the test still fails on the signal.
// backtrace is not async-signal-safe: SA_RESETHAND makes a second fault in it fatal, and the
// alarm ends a handler that deadlocks in the unwinder or loader instead of hanging CTest.
void crashed(int number)
{
    alarm(10);
    char banner[] = "scene_query_benchmark: fatal signal 00, backtrace:\n";
    banner[36]    = char('0' + number / 10 % 10);
    banner[37]    = char('0' + number % 10);
    (void)!write(STDERR_FILENO, banner, sizeof(banner) - 1);
    void*     frames[64];
    const int count = backtrace(frames, 64);
    backtrace_symbols_fd(frames, count, STDERR_FILENO);
    raise(number);
}

void reportCrashes()
{
    void* warm;
    backtrace(&warm, 1); // loads libgcc_s now; the handler must not allocate
    struct sigaction action = {};
    action.sa_handler       = crashed;
    action.sa_flags         = SA_RESETHAND;
    for (int number : {SIGSEGV, SIGBUS, SIGABRT, SIGFPE, SIGILL})
        sigaction(number, &action, nullptr);
}
#endif
} // namespace bench
int main(int argc, char** argv)
{
#ifndef _WIN32
    bench::reportCrashes();
#endif
    // Each result line reaches CTest as written, so a crash in teardown still shows it.
    std::cout << std::unitbuf;
    try
    {
        return bench::run(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::cerr << "scene_query_benchmark: " << error.what() << '\n';
        return 1;
    }
}
