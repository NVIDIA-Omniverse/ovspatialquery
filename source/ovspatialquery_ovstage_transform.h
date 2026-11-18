// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#ifndef OVSPATIALQUERY_OVSTAGE_TRANSFORM_H
#define OVSPATIALQUERY_OVSTAGE_TRANSFORM_H
#if !defined(__CUDACC__)
#include <cmath>
#endif
#if defined(__CUDACC__)
#define OVSTAGE_MATH __host__ __device__
#else
#define OVSTAGE_MATH
#endif
namespace ovspatialquery
{
namespace detail
{
namespace ovstage_transform
{
#if !defined(__CUDACC__)
using std::fabs;
using std::fmax;
using std::frexp;
using std::isfinite;
using std::ldexp;
using std::nextafter;
using std::sqrt;
#endif
constexpr double tolerance = 0x1p-19;
// Ovstage supplies column-major matrices; host acquisition transposes them.
OVSTAGE_MATH inline bool normalizeAffine(double* m, bool columnMajor)
{
    if (m[columnMajor ? 3 : 12] != 0.0 || m[columnMajor ? 7 : 13] != 0.0 || m[columnMajor ? 11 : 14] != 0.0 ||
        !isfinite(m[15]) || m[15] == 0.0)
        return false;
    const double w = m[15];
    if (w != 1.0)
        for (unsigned i = 0; i < 16; ++i)
            m[i] /= w;
    return true;
}
OVSTAGE_MATH inline double determinant(const double* a)
{
    return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6]);
}
OVSTAGE_MATH inline bool inverse(const double* a, double* b)
{
    const double d = determinant(a);
    if (!isfinite(d) || d == 0.0)
        return false;
    b[0] = (a[4] * a[8] - a[5] * a[7]) / d;
    b[1] = (a[2] * a[7] - a[1] * a[8]) / d;
    b[2] = (a[1] * a[5] - a[2] * a[4]) / d;
    b[3] = (a[5] * a[6] - a[3] * a[8]) / d;
    b[4] = (a[0] * a[8] - a[2] * a[6]) / d;
    b[5] = (a[2] * a[3] - a[0] * a[5]) / d;
    b[6] = (a[3] * a[7] - a[4] * a[6]) / d;
    b[7] = (a[1] * a[6] - a[0] * a[7]) / d;
    b[8] = (a[0] * a[4] - a[1] * a[3]) / d;
    return true;
}
OVSTAGE_MATH inline bool rigid(const double* a)
{
    if (!(determinant(a) > 0.0))
        return false;
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = i; j < 3; ++j)
        {
            const double d = (a[i] * a[j] + a[3 + i] * a[3 + j]) + a[6 + i] * a[6 + j] - (i == j ? 1.0 : 0.0);
            if (!isfinite(d) || fabs(d) > tolerance)
                return false;
        }
    return true;
}
OVSTAGE_MATH inline void candidate(const double* l, const double* ki, double s, double* a)
{
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            a[i * 3 + j] = ((l[i * 3] * ki[j] + l[i * 3 + 1] * ki[3 + j]) + l[i * 3 + 2] * ki[6 + j]) / s;
}
OVSTAGE_MATH inline void quaternion(const double* r, double* q)
{
    const double trace = r[0] + r[4] + r[8];
    if (trace > 0.0)
    {
        const double v = 2.0 * sqrt(trace + 1.0);
        q[3]           = v * 0.25;
        q[0]           = (r[7] - r[5]) / v;
        q[1]           = (r[2] - r[6]) / v;
        q[2]           = (r[3] - r[1]) / v;
    }
    else
    {
        unsigned     i = r[0] > r[4] ? (r[0] > r[8] ? 0 : 2) : (r[4] > r[8] ? 1 : 2);
        unsigned     j = (i + 1) % 3, k = (j + 1) % 3;
        const double v = 2.0 * sqrt(1.0 + r[i * 3 + i] - r[j * 3 + j] - r[k * 3 + k]);
        q[i]           = v * 0.25;
        q[j]           = (r[i * 3 + j] + r[j * 3 + i]) / v;
        q[k]           = (r[i * 3 + k] + r[k * 3 + i]) / v;
        q[3]           = (r[k * 3 + j] - r[j * 3 + k]) / v;
    }
    const double n = sqrt((q[0] * q[0] + q[1] * q[1]) + (q[2] * q[2] + q[3] * q[3]));
    for (unsigned i = 0; i < 4; ++i)
        q[i] /= n;
}
OVSTAGE_MATH inline void rotation(const double* q, double* r)
{
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    r[0] = 1 - 2 * (y * y + z * z);
    r[1] = 2 * (x * y - z * w);
    r[2] = 2 * (x * z + y * w);
    r[3] = 2 * (x * y + z * w);
    r[4] = 1 - 2 * (x * x + z * z);
    r[5] = 2 * (y * z - x * w);
    r[6] = 2 * (x * z - y * w);
    r[7] = 2 * (y * z + x * w);
    r[8] = 1 - 2 * (x * x + y * y);
}
OVSTAGE_MATH inline void rotateSymmetric(double& a, double& b, double& v, double& x, double& y)
{
    const double tau = (b - a) / (2 * v);
    const double t   = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1 + tau * tau));
    const double c = 1 / sqrt(1 + t * t), sn = t * c;
    a -= t * v;
    b += t * v;
    v                 = 0;
    const double oldX = x;
    x                 = c * oldX - sn * y;
    y                 = sn * oldX + c * y;
}
OVSTAGE_MATH inline double largestSingular(const double* l, double scale)
{
    int exponent;
    (void)frexp(scale, &exponent);
    scale = ldexp(1.0, exponent); // exact power-of-two normalization
    double sym[9];
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            sym[i * 3 + j] = ((l[i] / scale) * (l[j] / scale) + (l[3 + i] / scale) * (l[3 + j] / scale)) +
                             (l[6 + i] / scale) * (l[6 + j] / scale);
    const double common = sym[0];
    double       a = 0, b = sym[4] - common, c = sym[8] - common, x = sym[1], y = sym[2], z = sym[5];
    for (unsigned it = 0; it < 12; ++it)
    {
        const double ax = fabs(x), ay = fabs(y), az = fabs(z);
        if (fmax(ax, fmax(ay, az)) <= 0x1p-56)
            break;
        if (ax >= ay && ax >= az)
            rotateSymmetric(a, b, x, y, z);
        else if (ay >= az)
            rotateSymmetric(a, c, y, x, z);
        else
            rotateSymmetric(b, c, z, x, y);
    }
    const double upper = fmax(a + fabs(x) + fabs(y), fmax(b + fabs(x) + fabs(z), c + fabs(y) + fabs(z)));
    // The remaining off-diagonal terms bound the eigenvalue from above.
    // Cover dot-product and rotation roundoff before rounding the radius out.
    const double value = scale * sqrt((common + upper) * (1.0 + 0x1p-48));
    return nextafter(value, 0x1.fffffffffffffp1023);
}
} // namespace ovstage_transform
} // namespace detail
} // namespace ovspatialquery
#undef OVSTAGE_MATH
#endif
