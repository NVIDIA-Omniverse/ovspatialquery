/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ovstage SDK helpers shared by the ovstage and deform workloads: waited stage operations, path
// dictionary results and one installed Ovstage publication split into capture and readiness.
#ifndef BENCH_WORKLOADS_OVSTAGE_SUPPORT_H
#define BENCH_WORKLOADS_OVSTAGE_SUPPORT_H
#include "../installed_adapter.h"
#include "../measurement.h"

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace bench
{
inline ovx_string_t text(const char* value) { return {value, std::strlen(value)}; }
// Ovstage 0.4 has no PHYSICS domain: RENDERING plus every prim's physics,
// physxConvexHullCollision and material:binding properties, which ovspatialquery reads for
// convex hulls, their vertex limits and physics materials.
inline ovstage_population_enqueue_result_t openUsdWithPhysics(ovstage_instance_t* stage, ovx_string_t usda,
                                                              ovstage_ordinal_t ordinal, double time)
{
    static const ovx_string_t kNamespaces[] = {
        {"physics", 7u}, {"physxConvexHullCollision", 24u}, {"material:binding", 16u}};
    ovstage_population_selector_t selector{};
    selector.prim_predicate.kind               = OVSTAGE_POPULATION_PRIM_PREDICATE_ALL;
    selector.property_predicate.kind           = OVSTAGE_POPULATION_PROPERTY_PREDICATE_IN_NAMESPACE;
    selector.property_predicate.values.strings = kNamespaces;
    selector.property_predicate.value_count    = 3u;
    ovstage_population_desc_t desc{};
    desc.domains        = OVSTAGE_POPULATION_DOMAIN_RENDERING;
    desc.selectors      = &selector;
    desc.selector_count = 1u;
    return ovstage_population_open_usd_from_string_with_desc(stage, usda, ordinal, time, &desc, 1u);
}
// Appends a number to authored USDA text with every digit a double carries.
inline void appendNumber(std::string& out, double value)
{
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    out += buffer;
}
inline std::string opError(ovstage_instance_t* stage, ovstage_op_id_t op)
{
    const ovx_string_t message = ovstage_get_last_op_error(stage, op);
    return std::string(message.ptr ? message.ptr : "", message.length);
}
// Waits for one stage operation and releases it; a stage failure ends the run.
inline void stageOp(ovstage_instance_t* stage, ovstage_enqueue_result_t op, const char* action)
{
    require(op.status == OVSTAGE_OK && op.op_index != OVSTAGE_INVALID_OP_ID,
            std::string(action) + " was not admitted, status " + std::to_string(op.status));
    ovstage_op_wait_result_t   result{};
    const ovstage_api_status_t waited   = ovstage_wait_op(stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const bool                 ok       = waited == OVSTAGE_OK && result.error_op_id_count == 0;
    const std::string          error    = ok ? std::string() : opError(stage, op.op_index);
    const ovstage_api_status_t released = ovstage_release_op(stage, op.op_index);
    require(ok, std::string(action) + " failed: " + error);
    require(released == OVSTAGE_OK, std::string(action) + " release");
}
inline void dictionaryOk(path_dictionary_instance_t* dictionary, const ovx_api_result_t& result, const char* action)
{
    const bool ok = result.status == OVX_API_SUCCESS;
    if (result.error.ptr != nullptr)
        dictionary->vtable->release_error(dictionary->context, result.error);
    require(ok, action);
}

struct Publication
{
    double capture = 0, readiness = 0; // The synchronous enqueue span; then wait, fetch and destroy.
    ovspatialquery_population_update_result_t result{};
};
// Both workloads' installed lanes publish through this so their spans are split identically.
inline Publication finishPublication(Spatial& s, ovspatialquery_enqueue_result_t admitted,
                                     Clock::time_point enqueueStart, size_t objects, const char* action)
{
    Publication p;
    p.capture = elapsed(enqueueStart);
    check({admitted.status}, action);
    const auto                      waitStart = Clock::now();
    ovspatialquery_op_wait_result_t wait{};
    check(ovspatialquery_wait_op(s.instance, admitted.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &wait), "wait");
    if (wait.state != OVSPATIALQUERY_OP_COMPLETED)
    {
        const auto                 error = ovspatialquery_get_last_op_error(s.instance, admitted.op_id);
        const std::string          message(error.ptr ? error.ptr : "", size_t(error.length));
        ovspatialquery_op_status_t status{};
        check(ovspatialquery_query_op_status(s.instance, admitted.op_id, &status), "query failed op status");
        uint64_t hostBytes = 0;
        check(ovspatialquery_get_instance_memory_usage(s.instance, DLDevice{kDLCPU, 0}, &hostBytes), "host usage");
        check(ovspatialquery_destroy_op(s.instance, &admitted.op_id), "destroy failed op");
        throw std::runtime_error(std::string(action) + " did not complete: " + message + " (terminal status " +
                                 std::to_string(status.terminal_status) + ", host ledger " + std::to_string(hostBytes) +
                                 " bytes of " + std::to_string(hostBudget) + ")");
    }
    check(ovspatialquery_fetch_population_update(s.instance, admitted.op_id, &p.result), "fetch publication");
    check(ovspatialquery_destroy_op(s.instance, &admitted.op_id), "destroy op");
    p.readiness = elapsed(waitStart);
    require(p.result.object_count == objects && p.result.excluded_object_count == 0,
            std::string(action) + " published " + std::to_string(p.result.object_count) + " objects, " +
                std::to_string(p.result.excluded_object_count) + " excluded");
    s.ordinal = p.result.published_ordinal;
    return p;
}
} // namespace bench
#endif
