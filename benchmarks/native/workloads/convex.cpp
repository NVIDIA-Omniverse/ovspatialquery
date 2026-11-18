/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Convex fixture generation.
#include "convex.h"

#include <cmath>

namespace bench
{
Fixture convexFixture(const Options& o)
{
    Fixture f;
    f.kind   = o.workload == "convex-rays"     ? QueryKind::rays
               : o.workload == "convex-sweeps" ? QueryKind::sweep
                                               : QueryKind::zones;
    f.even   = o.subset == "even";
    f.radius = .125;
    f.half   = {.2, .15, .2};
    f.convexes.emplace_back();
    for (float z : {-1.f, 1.f})
        for (float y : {-1.f, 1.f})
            for (float x : {-1.f, 1.f})
                f.convexes[0].push_back({x, y, z});
    if (o.objects > 1)
    {
        f.convexes.emplace_back();
        for (float z : {-1.f, 1.f})
            for (size_t side = 0; side < 16; ++side)
            {
                const double angle = double(side) * std::acos(-1.0) / 8;
                f.convexes[1].push_back({float(std::cos(angle)), float(std::sin(angle)), z});
            }
    }
    // Both hulls are centered with unit maximum extent. The installed baseline
    // normalizes convex cooking to that domain, so both adapters cook the same points.
    const size_t width = size_t(std::ceil(std::sqrt(double(o.objects))));
    for (size_t i = 0; i < o.objects; ++i)
        f.objects.push_back({i + 1,
                             i / 2 + 1,
                             false,
                             {double(i % width) * 4, double(i / width) * 4, 0},
                             i % 3 ? identity : rotation(.25f),
                             1,
                             0,
                             i % f.convexes.size() + 1});
    for (size_t i = 0; i < o.queries; ++i)
    {
        const auto& object = f.objects[(i * 37) % f.objects.size()];
        Query       q;
        q.p            = {object.p[0] + .17, object.p[1] + .13, f.kind == QueryKind::zones ? 1.06 : 3};
        q.q            = rotation(.17f);
        q.nearDistance = .125;
        q.farDistance  = 5;
        if (i % 8 == 0)
            q.p[0] += 1.5; // Exterior miss for either hull, including pose updates.
        else if (f.kind == QueryKind::zones)
        {
            if (i % 8 == 1)
                q.p[2] = 1.5; // Separated overlap.
        }
        else if (i % 8 == 1)
            q.farDistance = 1; // Far clip before the top face.
        else if (i % 8 == 2)
            q.nearDistance = 4.25; // Near plane beyond the entire hull.
        f.queries.push_back(q);
    }
    return f;
}
} // namespace bench
