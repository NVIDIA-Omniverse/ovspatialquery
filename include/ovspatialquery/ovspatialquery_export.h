/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_EXPORT_H
#define OVSPATIALQUERY_EXPORT_H

#if defined(_WIN32)
#if defined(OVSPATIALQUERY_BUILD_DLL)
#define OVSPATIALQUERY_API __declspec(dllexport)
#elif defined(OVSPATIALQUERY_USE_DLL)
#define OVSPATIALQUERY_API __declspec(dllimport)
#else
#define OVSPATIALQUERY_API
#endif
#elif defined(OVSPATIALQUERY_BUILD_DLL) && (defined(__GNUC__) || defined(__clang__))
#define OVSPATIALQUERY_API __attribute__((visibility("default")))
#else
#define OVSPATIALQUERY_API
#endif

/*
 * Marks an entry point that a newer one replaces, placed first in its
 * declaration. The message names the replacement and the release that removes
 * the old entry point, a major release at least one minor release later, so
 * the compiler warning tells the caller what to change and by when:
 * OVSPATIALQUERY_DEPRECATED("Use ovspatialquery_x_v2 instead; removed in 2.0.0").
 * It is the library's local form of the shared OV_DEPRECATED macro.
 */
#if defined(_MSC_VER)
#define OVSPATIALQUERY_DEPRECATED(message) __declspec(deprecated(message))
#elif defined(__GNUC__) || defined(__clang__)
#define OVSPATIALQUERY_DEPRECATED(message) __attribute__((deprecated(message)))
#else
#define OVSPATIALQUERY_DEPRECATED(message)
#endif

/*
 * Marks an experimental function, placed first in its declaration. An
 * experimental declaration may change or be removed in any minor release, also
 * from 1.0, without a deprecation window; a patch release leaves it unchanged.
 * Experimental enumerators and struct members, which no attribute marks
 * portably, begin their comment with "Experimental:". An experimental member
 * of a stable struct may change meaning in a minor release, but from 1.0 it
 * stays in place until a major release, because removing it would change the
 * struct's layout. A caller that defines
 * OVSPATIALQUERY_WARN_EXPERIMENTAL before including any ovspatialquery header
 * gets a deprecation warning at every use of an experimental function, so it
 * can list what it depends on or refuse it with -Werror.
 */
#if defined(OVSPATIALQUERY_WARN_EXPERIMENTAL)
#define OVSPATIALQUERY_EXPERIMENTAL OVSPATIALQUERY_DEPRECATED("experimental: may change in any minor release")
#else
#define OVSPATIALQUERY_EXPERIMENTAL
#endif

#endif /* OVSPATIALQUERY_EXPORT_H */
