/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The commit a library was built from, as one plain string that `strings` or
// `findstr` finds in a loose library or an extracted SDK. No code reads it and it
// is not exported. CMake defines OVSPATIALQUERY_SOURCE_REVISION for this file
// alone, so a new commit recompiles one translation unit and relinks.

#if !defined(OVSPATIALQUERY_SOURCE_REVISION)
#error "CMake defines OVSPATIALQUERY_SOURCE_REVISION for this file"
#endif

#if defined(_MSC_VER)
// /INCLUDE keeps the unreferenced definition through /OPT:REF and /LTCG.
#pragma comment(linker, "/INCLUDE:ovspatialquerySourceRevision")
#define OVSPATIALQUERY_KEEP_UNREFERENCED
#else
// `used` keeps the definition through optimization and LTO. CMake makes it an
// explicit ELF linker root; GCC 11 ARM can report `retain` but then ignore it.
#define OVSPATIALQUERY_KEEP_UNREFERENCED __attribute__((used))
#endif

extern "C" OVSPATIALQUERY_KEEP_UNREFERENCED const char ovspatialquerySourceRevision[] =
    "ovspatialquery-source: " OVSPATIALQUERY_SOURCE_REVISION;
