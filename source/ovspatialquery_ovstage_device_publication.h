/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_DEVICE_PUBLICATION_H
#define OVSPATIALQUERY_OVSTAGE_DEVICE_PUBLICATION_H

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovstage/ovstage_api/experimental/ovstage_placement_extension.h>

#include <string>

namespace ovspatialquery
{
namespace detail
{

// The stage's placement extension that device-prepared publications read poses
// through, or null with `outDiagnostic` naming what is missing. The publication
// calls and ovspatialquery_query_ovstage_capability share this one check.
const ovstage_placement_extension_v1_t* ovstagePlacementExtension(ovstage_instance_t* stage,
                                                                  std::string&        outDiagnostic);

// Admits an Ovstage publication prepared on the device (ADR-0047): a complete set
// when `baseOrdinal` is null, else an update. `execution` asks for ON_DEVICE or
// ON_DEVICE_HOST_READABLE; the caller handles ON_HOST. `outDiagnostic` names a
// refusal more precisely than its status. It throws what it allocates; the caller
// maps exceptions to statuses.
ovspatialquery_api_status_t submitOvstageDevicePublication(ovspatialquery_instance_handle_t          instance,
                                                           ovspatialquery_population_handle_t        population,
                                                           const ovspatialquery_ordinal_t*           baseOrdinal,
                                                           ovspatialquery_ordinal_t                  targetOrdinal,
                                                           const ovspatialquery_dynamic_execution_t& execution,
                                                           ovspatialquery_op_id_t&                   outOperation,
                                                           std::string&                              outDiagnostic);

} // namespace detail
} // namespace ovspatialquery

#endif
