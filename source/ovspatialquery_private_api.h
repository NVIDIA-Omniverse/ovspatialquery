/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_PRIVATE_API_H
#define OVSPATIALQUERY_PRIVATE_API_H

/*
 * Engine-private vocabulary shared by the source/ translation units.
 *
 * The public headers are the contract; this header adds only what the engine
 * needs internally: the AABB, the source kind and the participation test. Overlap queries carry the public
 * ovspatialquery_overlap_shape_t directly. The operation runtime and its
 * owners report ovspatialquery_api_status_t directly.
 * Nothing declared here is installed or exported.
 */

#include <ovspatialquery/ovspatialquery.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct ovspatialquery_aabb_t
    {
        /* Finite world-space corners with minimum[i] <= maximum[i]. */
        double minimum[3];
        double maximum[3];
    } ovspatialquery_aabb_t;

    /* Identity and acquisition domain fixed when an instance is created. */
    typedef enum ovspatialquery_source_kind_t
    {
        /* Object IDs are caller-defined numeric values. */
        OVSPATIALQUERY_SOURCE_EXPLICIT = 0,
        /* Object IDs are path tokens from a caller-owned ovstage dictionary. */
        OVSPATIALQUERY_SOURCE_OVSTAGE = 1
    } ovspatialquery_source_kind_t;

#ifdef __cplusplus
}

namespace ovspatialquery
{
namespace detail
{

using ParticipationBits = ovspatialquery_participation_bits_t;

/* What an object has when its source assigns no bits. */
constexpr ParticipationBits kAllParticipationBits{{UINT64_MAX, UINT64_MAX}};

inline bool participationIsZero(const ParticipationBits& bits) noexcept
{
    return (bits.words[0] | bits.words[1]) == 0u;
}

inline bool participationEqual(const ParticipationBits& first, const ParticipationBits& second) noexcept
{
    return first.words[0] == second.words[0] && first.words[1] == second.words[1];
}

/* Zero query bits accept every object; otherwise any shared set bit passes. */
inline bool participationAccepts(const ParticipationBits& query, const ParticipationBits& object) noexcept
{
    return participationIsZero(query) ||
           ((query.words[0] & object.words[0]) | (query.words[1] & object.words[1])) != 0u;
}

} // namespace detail
} // namespace ovspatialquery
#endif

#endif
