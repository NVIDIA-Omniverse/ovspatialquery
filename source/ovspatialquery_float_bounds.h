/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_FLOAT_BOUNDS_H
#define OVSPATIALQUERY_FLOAT_BOUNDS_H

#include <cmath>
#include <cstdint>
#include <limits>

namespace ovspatialquery
{
namespace detail
{

// The input domain of ovspatialquery.h (ADR-0032): every caller coordinate,
// extent, radius, distance and tolerance, and every published world AABB, lies
// within +/-2^60 caller units, and every uniform placement scale within
// [2^-60, 2^60]. Inside it a local-frame value stays below 2^122, so every
// float evaluator input is finite. A NaN is outside both.
constexpr double kInputDomainBound    = 1152921504606846976.0; // 2^60
constexpr double kMinimumUniformScale = 8.673617379884035e-19; // 2^-60
inline bool      inInputDomain(double value) { return std::abs(value) <= kInputDomainBound; }
inline bool      inScaleDomain(double scale) { return scale >= kMinimumUniformScale && scale <= kInputDomainBound; }

inline double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// Normalises quaternionXyzw by the caller-supplied inverseLength (each
// caller computes lengthSquared/inverseLength itself, since callers differ
// in how strictly/where they validate it) and writes the equivalent 3x3
// rotation matrix. Keeping the multiply-by-inverseLength and matrix-build
// steps here, rather than also recomputing lengthSquared, means every call
// site reproduces its own prior bit-for-bit result.
inline void quaternionRotationMatrix(const double quaternionXyzw[4], double inverseLength, double rotation[3][3])
{
    const double x            = quaternionXyzw[0] * inverseLength;
    const double y            = quaternionXyzw[1] * inverseLength;
    const double z            = quaternionXyzw[2] * inverseLength;
    const double w            = quaternionXyzw[3] * inverseLength;
    const double result[3][3] = {
        {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)},
        {2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)},
        {2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)},
    };
    for (uint32_t row = 0u; row < 3u; ++row)
    {
        for (uint32_t column = 0u; column < 3u; ++column)
        {
            rotation[row][column] = result[row][column];
        }
    }
}

// Ericson's closest-point-on-triangle region test (Real-Time Collision
// Detection, section 5.1.5). `triangle[0]` is the barycentric origin;
// `outU`/`outV` are the barycentric weights of `triangle[1] - triangle[0]`
// and `triangle[2] - triangle[0]` respectively, so
// closest == triangle[0] + outU * (triangle[1] - triangle[0])
//                        + outV * (triangle[2] - triangle[0]).
// Shared by ovspatialquery_geometry.cpp (which also consumes outU/outV) and
// ovspatialquery_convex_geometry.cpp (which calls it directly and only
// consumes `closest`) so the two files cannot drift onto different region
// tests again.
inline void closestPointOnTriangle(const double triangle[3][3], const double point[3], double closest[3], double& outU,
                                   double& outV)
{
    double first[3]     = {};
    double second[3]    = {};
    double fromFirst[3] = {};
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        first[axis]     = triangle[1][axis] - triangle[0][axis];
        second[axis]    = triangle[2][axis] - triangle[0][axis];
        fromFirst[axis] = point[axis] - triangle[0][axis];
    }
    const double firstProjection  = dot3(first, fromFirst);
    const double secondProjection = dot3(second, fromFirst);
    if (firstProjection <= 0.0 && secondProjection <= 0.0)
    {
        outU = 0.0;
        outV = 0.0;
    }
    else
    {
        double fromSecond[3] = {};
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            fromSecond[axis] = point[axis] - triangle[1][axis];
        }
        const double thirdProjection  = dot3(first, fromSecond);
        const double fourthProjection = dot3(second, fromSecond);
        if (thirdProjection >= 0.0 && fourthProjection <= thirdProjection)
        {
            outU = 1.0;
            outV = 0.0;
        }
        else
        {
            const double firstEdgeRegion = firstProjection * fourthProjection - thirdProjection * secondProjection;
            if (firstEdgeRegion <= 0.0 && firstProjection >= 0.0 && thirdProjection <= 0.0)
            {
                outU = firstProjection / (firstProjection - thirdProjection);
                outV = 0.0;
            }
            else
            {
                double fromThird[3] = {};
                for (uint32_t axis = 0u; axis < 3u; ++axis)
                {
                    fromThird[axis] = point[axis] - triangle[2][axis];
                }
                const double fifthProjection = dot3(first, fromThird);
                const double sixthProjection = dot3(second, fromThird);
                if (sixthProjection >= 0.0 && fifthProjection <= sixthProjection)
                {
                    outU = 0.0;
                    outV = 1.0;
                }
                else
                {
                    const double secondEdgeRegion =
                        fifthProjection * secondProjection - firstProjection * sixthProjection;
                    if (secondEdgeRegion <= 0.0 && secondProjection >= 0.0 && sixthProjection <= 0.0)
                    {
                        outU = 0.0;
                        outV = secondProjection / (secondProjection - sixthProjection);
                    }
                    else
                    {
                        const double thirdEdgeRegion =
                            thirdProjection * sixthProjection - fifthProjection * fourthProjection;
                        if (thirdEdgeRegion <= 0.0 && fourthProjection - thirdProjection >= 0.0 &&
                            fifthProjection - sixthProjection >= 0.0)
                        {
                            outV = (fourthProjection - thirdProjection) /
                                   ((fourthProjection - thirdProjection) + (fifthProjection - sixthProjection));
                            outU = 1.0 - outV;
                        }
                        else
                        {
                            const double inverse = 1.0 / (firstEdgeRegion + secondEdgeRegion + thirdEdgeRegion);
                            outU                 = secondEdgeRegion * inverse;
                            outV                 = firstEdgeRegion * inverse;
                        }
                    }
                }
            }
        }
    }
    for (uint32_t axis = 0u; axis < 3u; ++axis)
    {
        closest[axis] = std::fma(outV, second[axis], std::fma(outU, first[axis], triangle[0][axis]));
    }
}

inline float conservativeFloatBound(double value, bool lower)
{
    const double maximum = static_cast<double>(std::numeric_limits<float>::max());
    if (value <= -maximum)
    {
        return -std::numeric_limits<float>::max();
    }
    if (value >= maximum)
    {
        return std::numeric_limits<float>::max();
    }
    float converted = static_cast<float>(value);
    if (lower && static_cast<double>(converted) > value)
    {
        converted = std::nextafter(converted, -std::numeric_limits<float>::infinity());
    }
    else if (!lower && static_cast<double>(converted) < value)
    {
        converted = std::nextafter(converted, std::numeric_limits<float>::infinity());
    }
    return converted;
}

} // namespace detail
} // namespace ovspatialquery

#endif
