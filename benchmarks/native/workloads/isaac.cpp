/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Isaac-derived semantic smoke over the shared adapters.
#include "isaac.h"

#include "../installed_adapter.h"

#include <algorithm>
#include <iterator>

namespace bench
{
std::vector<InsideProbe> insideProbes;
void                     semanticSmoke()
{
    // Bounded rays cover box/sphere, rotations, misses, clipping and participation.
    {
        Options o;
        o.objects                 = 8;
        o.queries                 = 16;
        auto f                    = fixture(o);
        f.queries[0].farDistance  = 2;    // object surface is outside far plane
        f.queries[1].nearDistance = 4.25; // whole primitive behind near plane
        f.queries[2].nearDistance = 1;
        f.queries[2].farDistance  = 3.5;
        f.queries[3].bits         = 2;
        f.queries[5].bits         = 1; // sphere ID 2 participates only in bit 2
        Direct  direct(f);
        Spatial spatial(f);
        compareAdapters(f, direct, spatial);
        Results clipped(f.queries.size());
        direct.query(f, clipped);
        require(clipped[0].hits.empty() && clipped[1].hits.empty() && clipped[2].hits.size() == 1 &&
                    clipped[5].hits.empty(),
                "range/participation expectations");
        f.even = true;
        compareAdapters(f, direct, spatial);
        f.even = false;
        f.objects[0].p[0] += 8;
        direct.update(f, 1);
        spatial.update(f, 1);
        compareAdapters(f, direct, spatial);
        f.queries[0].p            = f.objects[0].p;
        f.queries[0].p[2]         = 4;
        f.queries[0].nearDistance = 3.75;
        f.queries[0].farDistance  = 8;
        f.queries[1].p            = f.objects[1].p;
        f.queries[1].p[2]         = 4;
        f.queries[1].nearDistance = 3.75;
        f.queries[1].farDistance  = 8;
        f.queries[2].p            = f.objects[0].p;
        f.queries[2].nearDistance = 0;
        f.queries[3].p            = f.objects[1].p;
        f.queries[3].nearDistance = 0;
        f.queries.resize(4);
        Results insideDirect(4), insideScalar(4), insideBatch(4), shiftedScalar(4), shiftedBatch(4);
        direct.query(f, insideDirect);
        spatial.scalar(f, insideScalar);
        spatial.makePlan(f);
        spatial.batch(f, insideBatch);
        compare(f, insideScalar, insideBatch, "inside scalar/batch");
        auto shifted = f;
        for (auto& q : shifted.queries)
        {
            for (size_t axis = 0; axis < 3; ++axis)
                q.p[axis] += q.nearDistance * q.direction[axis];
            q.farDistance -= q.nearDistance;
            q.nearDistance = 0;
        }
        spatial.scalar(shifted, shiftedScalar);
        spatial.makePlan(shifted);
        spatial.batch(shifted, shiftedBatch);
        compare(shifted, shiftedScalar, shiftedBatch, "shifted inside scalar/batch");
        Results composed(4);
        for (size_t i = 0; i < 4; ++i)
        {
            // A degenerate AABB is an exact point query. Keep target and immutable
            // ordinal identical to the ray when composing initial-overlap semantics.
            ovspatialquery_overlap_desc_t point{};
            point.common     = spatial.common(shifted, shifted.queries[i], spatial.ordinal);
            point.shape.kind = OVSPATIALQUERY_OVERLAP_AABB;
            std::copy(shifted.queries[i].p.begin(), shifted.queries[i].p.end(), point.shape.data.aabb.min_corner);
            std::copy(shifted.queries[i].p.begin(), shifted.queries[i].p.end(), point.shape.data.aabb.max_corner);
            ovspatialquery_overlap_any_result_t containment{};
            check(ovspatialquery_overlap_any_sync(spatial.instance, &point, &containment), "inside point containment");
            require(containment.has_hit && containment.source_ordinal == spatial.ordinal &&
                        containment.hit.truth == OVSPATIALQUERY_HIT_EXACT_SURFACE,
                    "point composition identity/version");
            Hit h;
            h.id            = containment.hit.identity.object_id.value;
            h.geometryType  = containment.hit.identity.geometry.type;
            h.geometryValue = containment.hit.identity.geometry.value;
            h.position      = shifted.queries[i].p;
            h.distance      = f.queries[i].nearDistance;
            for (size_t axis = 0; axis < 3; ++axis)
                h.normal[axis] = -f.queries[i].direction[axis];
            composed[i].hits.push_back(h);
            require(insideDirect[i].hits.size() == 1 && insideScalar[i].hits.size() == 1 &&
                        shiftedScalar[i].hits.size() == 1,
                    "inside probes must hit known primitive");
            shiftedScalar[i].hits[0].distance += f.queries[i].nearDistance;
            insideProbes.push_back({insideDirect[i].hits[0], insideScalar[i].hits[0], shiftedScalar[i].hits[0], h});
        }
        compare(f, insideDirect, composed, "PhysX/point containment composition");
        // A clipped start inside a solid reports what a start moved there reports (ADR-0018).
        compare(f, insideScalar, shiftedScalar, "clipped start/moved origin");
    }
    {
        // Dedicated coincident geometry admits either tied identity, but identical contact.
        Options tieOptions;
        tieOptions.objects    = 2;
        tieOptions.queries    = 1;
        auto tie              = fixture(tieOptions);
        tie.objects[0].sphere = false;
        tie.objects[0].q      = identity;
        tie.objects[1].sphere = false;
        tie.objects[1].q      = identity;
        tie.objects[1].p      = tie.objects[0].p;
        Direct  tieDirect(tie);
        Spatial tieSpatial(tie);
        compareAdapters(tie, tieDirect, tieSpatial, true);
    }
    {
        // Exactly representable touching box faces, a just-separated box and sphere.
        Options boundaryOptions;
        boundaryOptions.objects  = 2;
        boundaryOptions.queries  = 4;
        boundaryOptions.workload = "occupancy";
        auto boundary            = fixture(boundaryOptions);
        boundary.objects[0].p    = {0, 0, 0};
        boundary.objects[0].q    = identity;
        boundary.objects[1].p    = {4, 0, 0};
        boundary.half            = {0.5, 0.5, 0.5};
        boundary.queries[0].p    = {1, 0, 0};
        boundary.queries[1].p    = {1.0009765625, 0, 0};
        boundary.queries[2].p    = {5, 0, 0};
        boundary.queries[3].p    = {5.0009765625, 0, 0};
        Direct  boundaryDirect(boundary);
        Spatial boundarySpatial(boundary);
        compareAdapters(boundary, boundaryDirect, boundarySpatial);
        Results touching(4);
        boundaryDirect.query(boundary, touching);
        require(touching[0].hits.size() == 1 && touching[1].hits.empty() && touching[2].hits.size() == 1 &&
                    touching[3].hits.empty(),
                "touching boundary expectations");
    }
    {
        // Thin-band occupancy cells must include the floor with the minimum Z extent.
        Options gridOptions;
        gridOptions.objects  = 8;
        gridOptions.queries  = 64;
        gridOptions.workload = "occupancy";
        auto grid            = fixture(gridOptions);
        require(grid.half[2] == std::max(0.125, grid.half[0]), "minimum occupancy height rule");
        Direct  gridDirect(grid);
        Spatial gridSpatial(grid);
        compareAdapters(grid, gridDirect, gridSpatial);
    }
    {
        // An AABB candidate at a rotated box corner must fail the exact overlap.
        Options rotatedOptions;
        rotatedOptions.objects  = 1;
        rotatedOptions.queries  = 1;
        rotatedOptions.workload = "zones";
        auto rotated            = fixture(rotatedOptions);
        rotated.half            = {0.03125, 0.03125, 0.03125};
        rotated.objects[0].p    = {0, 0, 0};
        rotated.objects[0].q    = rotation(0.7853981634f);
        rotated.queries[0].p    = {0.55, 0.55, 0};
        Direct  rotatedDirect(rotated);
        Spatial rotatedSpatial(rotated);
        compareAdapters(rotated, rotatedDirect, rotatedSpatial);
        Results exactCorner(1);
        rotatedDirect.query(rotated, exactCorner);
        require(exactCorner[0].hits.empty(), "rotated-box AABB corner must not be an exact hit");
    }
    {
        // Three collider hits map to two owners. Equal-count replacement must emit transitions.
        Options zoneOptions;
        zoneOptions.objects    = 4;
        zoneOptions.queries    = 1;
        zoneOptions.workload   = "zones";
        auto zones             = fixture(zoneOptions);
        zones.half             = {2, 2, 1};
        zones.queries[0].p     = {0, 0, 0.5};
        zones.objects[0].p     = {-0.5, 0, 0.5};
        zones.objects[0].owner = 1;
        zones.objects[1].p     = {0.5, 0, 0.5};
        zones.objects[1].owner = 1;
        zones.objects[2].p     = {1.5, 0, 0.5};
        zones.objects[2].owner = 2;
        zones.objects[3].p     = {10, 0, 0.5};
        zones.objects[3].owner = 3;
        Direct  zoneDirect(zones);
        Spatial zoneSpatial(zones);
        compareAdapters(zones, zoneDirect, zoneSpatial);
        Results before(1), after(1), retained(1);
        zoneDirect.query(zones, before);
        require(owners(zones, before[0]) == std::vector<uint64_t>({1, 2}), "initial zone owners");
        ovspatialquery_overlap_hit_t        shortHit{};
        ovspatialquery_overlap_hit_buffer_t shortBuffer{&shortHit, 1};
        ovspatialquery_multiple_result_t    truncated{};
        auto                                descriptor = zoneSpatial.overlap(zones, zones.queries[0]);
        check(ovspatialquery_overlap_multiple_sync(zoneSpatial.instance, &descriptor, &shortBuffer, &truncated),
              "short overlap buffer");
        require(truncated.truncated && truncated.stored_hit_count == 1, "truncation was not reported");
        zones.objects[2].p[0] = 10;
        zones.objects[3].p[0] = 1.5;
        zoneDirect.update(zones, 4);
        zoneSpatial.update(zones, 4);
        compareAdapters(zones, zoneDirect, zoneSpatial);
        zoneDirect.query(zones, after);
        require(owners(zones, after[0]) == std::vector<uint64_t>({1, 3}), "replacement zone owners");
        const auto beforeFiltered = owners(zones, before[0], 1), afterFiltered = owners(zones, after[0], 1);
        require(beforeFiltered == std::vector<uint64_t>({2}) && afterFiltered == std::vector<uint64_t>({3}),
                "exclusion/equal-count replacement");
        std::vector<uint64_t> entered, exited;
        std::set_difference(afterFiltered.begin(), afterFiltered.end(), beforeFiltered.begin(), beforeFiltered.end(),
                            std::back_inserter(entered));
        std::set_difference(beforeFiltered.begin(), beforeFiltered.end(), afterFiltered.begin(), afterFiltered.end(),
                            std::back_inserter(exited));
        require(entered == std::vector<uint64_t>({3}) && exited == std::vector<uint64_t>({2}),
                "complete-set transitions");
        // Retry after HEAD changed must read the original ordinal, never mix versions.
        zoneSpatial.scalar(zones, retained, truncated.source_ordinal);
        compare(zones, before, retained, "same-ordinal scalar retry");
        zoneSpatial.batch(zones, retained, truncated.source_ordinal);
        compare(zones, before, retained, "retained batch version");
        zones.even = true;
        compareAdapters(zones, zoneDirect, zoneSpatial);
    }
}
} // namespace bench
