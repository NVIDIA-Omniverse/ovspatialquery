/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_V03_LAYOUT_H
#define OVSPATIALQUERY_OVSTAGE_V03_LAYOUT_H

/* The Ovstage 0.3 read slots, which the 0.4 headers that build the companion no longer
 * declare. A 0.3 fetch_read_next returns one group per call and the group holds its one
 * attribute's column inline; release_group releases one group. Every other function-table
 * slot and struct the companion reads has the same layout in 0.3 and 0.4. Handles are
 * spelled uint64_t, which 0.3 used and which passes like 0.4's one-member handle structs.
 * Only types both releases declare appear here, so tools/ci/ovstage_layout_probe.cpp
 * compiles this header against the real 0.3 headers and checks it field by field. */

#include <ovstage/ovstage_api/ovstage_api_types.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct ovspatialquery_ovstage_v03_read_group_t
    {
        uint64_t                     read_group_id;
        uint64_t                     attribute;
        ovstage_ordinal_t            ordinal;
        bool                         is_delete;
        bool                         is_array;
        ovstage_attribute_semantic_t semantic;
        ovstage_prim_group_t         prims;
        ovstage_data_t               data;
        ovstage_attribute_meta_t     meta;
    } ovspatialquery_ovstage_v03_read_group_t;

    typedef ovstage_api_status_t (*ovspatialquery_ovstage_v03_fetch_read_next_t)(
        void* context, uint64_t read_handle, ovstage_timeout_ns_t timeout,
        ovspatialquery_ovstage_v03_read_group_t* out_group);

    typedef ovstage_api_status_t (*ovspatialquery_ovstage_v03_release_group_t)(
        void* context, const ovspatialquery_ovstage_v03_read_group_t* group);

#ifdef __cplusplus
}
#endif

#endif
