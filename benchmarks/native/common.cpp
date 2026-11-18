/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Command-line option parsing and validation.
#include "common.h"

#include "workloads/statics.h"

#include <cmath>

namespace bench
{
Options options(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; i += 2)
    {
        require(i + 1 < argc, "every option requires a value");
        const std::string key = argv[i], value = argv[i + 1];
        if (key == "--backend")
            o.backend = value;
        else if (key == "--workload")
            o.workload = value;
        else if (key == "--subset")
            o.subset = value;
        else if (key == "--warehouse-meshes")
            o.warehouseMeshes = value;
        else
        {
            require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                    "count must be an unsigned integer");
            const size_t n = std::stoull(value);
            if (key == "--mesh-side")
                o.meshSide = n;
            else if (key == "--unique-meshes")
                o.uniqueMeshes = n;
            else if (key == "--objects")
                o.objects = n;
            else if (key == "--queries")
                o.queries = n;
            else if (key == "--frames")
                o.frames = n;
            else if (key == "--warmup")
                o.warmup = n;
            else if (key == "--iterations")
                o.iterations = n;
            else if (key == "--update-percent")
                o.updatePercent = n;
            else if (key == "--actor-hit-capacity")
                o.actorHitCapacity = n;
            else if (key == "--radius-groups")
                o.radiusGroups = n;
            else if (key == "--reuse-plans")
                o.reusePlans = n;
            else if (key == "--zone-hit-capacity")
                o.zoneHitCapacity = n;
            else if (key == "--pair-tolerance-milli")
                o.pairToleranceMilli = n;
            else if (key == "--pair-capacity")
                o.pairCapacity = n;
            else if (key == "--mesh-source-faces")
                o.meshSourceFaces = n;
            else
                throw std::runtime_error("unknown option: " + key);
        }
    }
    const bool pairs = o.workload == "pairs" || o.workload == "pairs-self" || o.workload == "pairs-views" ||
                       o.workload == "pairs-split";
    const bool ovstage   = o.workload == "ovstage";
    const bool warehouse = o.workload.rfind("warehouse-", 0) == 0;
    const bool deform    = o.workload.rfind("deform-", 0) == 0;
    const bool statics   = staticsWorkload(o);
    require(o.backend == "physx" || o.backend == "parity" ||
                (pairs     ? (o.backend == "snapshot" || o.backend == "reporter" || o.backend == "reporter-cuda" ||
                          o.backend == "reporter-gpu" || o.backend == "abp")
                 : ovstage ? (o.backend == "explicit" || o.backend == "ovstage" || o.backend == "explicit-device" ||
                              o.backend == "ovstage-device")
                 : deform  ? (o.backend == "physx-refit" || o.backend == "physx-clone" || o.backend == "ovstage" ||
                             o.backend == "ovstage-cuda")
                           : (o.backend == "scalar" || o.backend == "batch" || o.backend == "cuda")),
            "unknown backend");
    require(o.workload == "rays" || o.workload == "occupancy" || o.workload == "zones" || o.workload == "grounding" ||
                o.workload == "support" || o.workload == "avoidance" || o.workload == "avoidance-mixed" ||
                o.workload == "actor-frame" || o.workload == "actor-frame-bits" || o.workload == "grounding-mesh" ||
                o.workload == "mesh-sweeps" || o.workload == "mesh-box-sweeps" || o.workload == "mesh-parity" ||
                o.workload == "convex-rays" || o.workload == "convex-sweeps" || o.workload == "convex-overlaps" ||
                o.workload == "sensor-point-cloud" || o.workload == "warehouse-rays" ||
                o.workload == "warehouse-sweeps" || o.workload == "warehouse-box-sweeps" ||
                o.workload == "warehouse-overlaps" || pairs || ovstage || o.workload == "deform-none" ||
                o.workload == "deform-rigid" || o.workload == "deform-vertex" || o.workload == "deform-fold" ||
                o.workload == "deform-crumple" || o.workload == "deform-mixed" || o.workload == "camera-mesh" ||
                o.workload == "camera-box" || statics,
            "unknown workload");
    require(o.meshSide >= 2 && o.meshSide <= 224 && o.uniqueMeshes <= 1 && o.meshSourceFaces <= 1,
            "invalid mesh size/sharing option");
    require(o.subset == "all" || o.subset == "even", "subset must be all or even");
    // A camera batch with no objects measures the empty scene.
    require((o.objects || o.workload == "camera-mesh") && o.objects <= 100000 && o.queries && o.queries <= 262144 &&
                o.frames && o.frames <= 10000 && o.warmup <= 100 && o.iterations && o.iterations <= 10000,
            "counts outside bounded benchmark limits");
    require(o.updatePercent <= 100, "update-percent must be in [0,100]");
    if (o.workload == "occupancy")
    {
        const size_t side = size_t(std::sqrt(double(o.queries)));
        require(side * side == o.queries, "occupancy queries must be a perfect square cell count");
    }
    require(o.zoneHitCapacity == 0 || (o.workload == "zones" && o.zoneHitCapacity <= o.objects),
            "zone-hit-capacity requires zones and cannot exceed the object count");
    if (o.workload == "zones")
        require(o.queries <= 256, "zones limit is 256");
    require(o.reusePlans <= 1 && o.radiusGroups <= 256, "invalid plan reuse/radius group count");
    if (o.workload == "grounding" || o.workload == "grounding-mesh")
        require(o.queries % 4 == 0, "grounding needs four rays per actor");
    if (o.workload == "support")
        require(o.queries == 1, "support is one immediate ray");
    if (o.workload.find("avoidance") == 0 || o.workload.rfind("actor-frame", 0) == 0)
        require(o.queries <= 256, "avoidance actor limit is 256");
    if (pairs)
    {
        require(o.subset == "all" && o.updatePercent > 0,
                "pairs takes all objects and a nonzero per-step update percent");
        // The cell workloads repurpose the query count as bodies per cell.
        require(o.workload == "pairs" ? o.queries == 1
                                      : o.queries == 2 || o.queries == 4 || o.queries == 16 || o.queries == 64,
                "pairs takes one query column; the cell workloads take 2, 4, 16 or 64 bodies per cell");
        require(o.pairToleranceMilli <= 10000 && (o.pairCapacity == 0 || o.pairCapacity >= o.objects),
                "pair tolerance is at most 10 m and the pair capacity at least the objects per side");
    }
    else
        require(o.pairToleranceMilli == 0 && o.pairCapacity == 0, "pair options require the pairs workload");
    if (ovstage)
        require(o.subset == "all" && o.updatePercent > 0 && o.queries <= 4096,
                "ovstage takes all objects, a nonzero per-step update percent and at most 4096 rays");
    if (o.workload == "sensor-point-cloud")
    {
        require(o.objects == 5 && o.subset == "all" && o.updatePercent == 0,
                "sensor-point-cloud requires five objects, all selection, and no object updates");
        require(o.queries == 16384 || o.queries == 65536 || o.queries == 259200,
                "sensor-point-cloud ray count must be 16384, 65536, or 259200");
        require(o.backend != "scalar" || o.queries <= 16384, "scalar sensor anchor is limited to 16384 rays");
    }
    if (warehouse)
        require(o.subset == "all", "warehouse workloads require subset all");
    if (deform)
        require(o.subset == "all" && o.objects <= 512 && o.queries <= 16384 &&
                    ((o.workload == "deform-none") == (o.updatePercent == 0)),
                "deform takes all cells (at most 512), at most 16384 rays, and a nonzero changed share except for "
                "deform-none");
    return o;
}
} // namespace bench
