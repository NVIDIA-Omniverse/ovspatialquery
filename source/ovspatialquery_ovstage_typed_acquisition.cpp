/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_backend.h"
#include "ovspatialquery_float_bounds.h"
#include "ovspatialquery_ovstage_filter.h"
#include "ovspatialquery_ovstage_mesh_geometry.h"
#include "ovspatialquery_ovstage_transform.h"
#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

namespace
{

struct OvstageSplit
{
    std::shared_ptr<const OvstageShapeTransform> shape;
    std::array<double, 3>                        lengths{};
    double                                       scale    = 1.0;
    double                                       largest  = 1.0;
    bool                                         uniform  = false;
    bool                                         diagonal = false;
    bool                                         retained = false;
};

struct OvstageSplitCache
{
    bool                         valid    = false;
    ExactGeometryFamily          family   = ExactGeometryFamily::user;
    const OvstageShapeTransform* previous = nullptr;
    std::array<double, 9>        linear{}, pose{};
    OvstageSplit                 value;
};

bool splitOvstageTransform(std::array<double, 16>& matrix, const PopulationObject* previous, ExactGeometryFamily family,
                           OvstageSplit& out, std::array<OvstageSplitCache, 2>& cache, size_t& nextCache)
{
    namespace math = ovstage_transform;
    for (double x : matrix)
        if (!inInputDomain(x))
            return false;
    double l[9];
    for (unsigned r = 0; r < 3; ++r)
        for (unsigned c = 0; c < 3; ++c)
            l[r * 3 + c] = matrix[r * 4 + c];
    const auto priorShape = previous ? previous->ovstageShape.get() : nullptr;
    // Bucket rows often share a linear transform. Cache only exact inputs,
    // including the recorded shape that determines the rigidity predicate.
    for (const auto& entry : cache)
        if (entry.valid && entry.family == family && entry.previous == priorShape &&
            std::memcmp(entry.linear.data(), l, sizeof(l)) == 0)
        {
            out = entry.value;
            for (unsigned r = 0; r < 3; ++r)
                for (unsigned c = 0; c < 3; ++c)
                    matrix[r * 4 + c] = entry.pose[r * 3 + c];
            return true;
        }
    const double det = math::determinant(l);
    if (!std::isfinite(det) || det == 0)
        return false;
    out.scale = std::cbrt(std::abs(det));
    if (!inScaleDomain(out.scale))
        return false;
    double a[9];
    for (unsigned i = 0; i < 9; ++i)
        a[i] = l[i] / out.scale;
    double metric[9];
    out.diagonal = true;
    for (unsigned i = 0; i < 3; ++i)
    {
        out.lengths[i] = std::sqrt((l[i] * l[i] + l[3 + i] * l[3 + i]) + l[6 + i] * l[6 + i]);
        if (!(out.lengths[i] > 0))
            return false;
    }
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
        {
            metric[i * 3 + j] = (a[i] * a[j] + a[3 + i] * a[3 + j]) + a[6 + i] * a[6 + j];
            if (i != j &&
                std::abs((l[i] * l[j] + l[3 + i] * l[3 + j]) + l[6 + i] * l[6 + j]) / out.lengths[i] / out.lengths[j] >
                    math::tolerance)
                out.diagonal = false;
        }
    out.uniform = true;
    for (unsigned i = 0; i < 9; ++i)
        if (std::abs(metric[i] - (i % 4 == 0 ? 1.0 : 0.0)) > math::tolerance)
            out.uniform = false;
    // Retention tests the candidate placement, including for an extremely anisotropic shape.
    if (previous && previous->ovstageShape)
    {
        math::candidate(l, previous->ovstageShape->inverse.data(), out.scale, a);
        if (math::rigid(a))
        {
            out.shape    = previous->ovstageShape;
            out.retained = true;
        }
    }
    if (!out.retained)
    {
        const double sign  = det < 0 ? -1.0 : 1.0;
        auto         shape = std::shared_ptr<OvstageShapeTransform>{};
        if (out.uniform)
        {
            static const auto positive = std::make_shared<const OvstageShapeTransform>();
            static const auto negative = []()
            {
                auto value      = std::make_shared<OvstageShapeTransform>();
                value->mirrored = true;
                for (unsigned i = 0; i < 9; ++i)
                    value->k[i] = value->inverse[i] = (i % 4 == 0 ? -1.0 : 0.0);
                return std::shared_ptr<const OvstageShapeTransform>(value);
            }();
            out.shape = det < 0 ? negative : positive;
            for (unsigned i = 0; i < 9; ++i)
                a[i] = l[i] / out.scale * sign;
        }
        else
        {
            shape           = std::make_shared<OvstageShapeTransform>();
            shape->mirrored = det < 0;
            double r[9];
            for (unsigned i = 0; i < 9; ++i)
                r[i] = l[i] / out.scale;
            if (out.diagonal)
            {
                for (unsigned i = 0; i < 3; ++i)
                    for (unsigned j = 0; j < 3; ++j)
                        r[i * 3 + j] = l[i * 3 + j] / out.lengths[j];
            }
            else
            {
                // Scaled Newton polar iteration avoids squaring the condition number.
                for (unsigned iteration = 0; iteration < 64; ++iteration)
                {
                    double inv[9];
                    if (!math::inverse(r, inv))
                        return false;
                    double nr = 0, ni = 0;
                    for (unsigned i = 0; i < 9; ++i)
                    {
                        nr += r[i] * r[i];
                        ni += inv[i] * inv[i];
                    }
                    const double gamma  = std::sqrt(std::sqrt(ni / nr));
                    double       change = 0;
                    for (unsigned i = 0; i < 3; ++i)
                        for (unsigned j = 0; j < 3; ++j)
                        {
                            const unsigned ix = i * 3 + j;
                            const double   v  = 0.5 * (gamma * r[ix] + inv[j * 3 + i] / gamma);
                            change            = std::max(change, std::abs(v - r[ix]));
                            r[ix]             = v;
                        }
                    if (change < 1e-15)
                        break;
                }
            }
            for (unsigned i = 0; i < 3; ++i)
                for (unsigned j = 0; j < 3; ++j)
                {
                    shape->k[i * 3 + j] =
                        sign * ((r[i] * l[j] + r[3 + i] * l[3 + j]) + r[6 + i] * l[6 + j]) / out.scale;
                    shape->metric[i * 3 + j] = metric[i * 3 + j];
                }
            if (!math::inverse(shape->k.data(), shape->inverse.data()))
                return false;
            math::candidate(l, shape->inverse.data(), out.scale, a);
        }
        if (shape)
            out.shape = std::move(shape);
    }
    double q[4], rotation[9];
    math::quaternion(a, q);
    math::rotation(q, rotation);
    for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j)
            matrix[i * 4 + j] = rotation[i * 3 + j] * out.scale;
    out.largest = out.scale;
    if (out.uniform && (family == ExactGeometryFamily::sphere || family == ExactGeometryFamily::capsule))
        out.largest = ovstage_transform::largestSingular(l, out.scale);
    auto& entry    = cache[nextCache];
    nextCache      = (nextCache + 1) % cache.size();
    entry.valid    = true;
    entry.family   = family;
    entry.previous = priorShape;
    std::copy(l, l + 9, entry.linear.begin());
    for (unsigned r = 0; r < 3; ++r)
        for (unsigned c = 0; c < 3; ++c)
            entry.pose[r * 3 + c] = matrix[r * 4 + c];
    entry.value = out;
    return true;
}

void applyOvstageShape(std::vector<float>& points, const OvstageShapeTransform& shape)
{
    for (size_t i = 0; i < points.size(); i += 3)
    {
        const double p[3] = {points[i], points[i + 1], points[i + 2]};
        for (unsigned r = 0; r < 3; ++r)
            points[i + r] =
                static_cast<float>((shape.k[r * 3] * p[0] + shape.k[r * 3 + 1] * p[1]) + shape.k[r * 3 + 2] * p[2]);
    }
}

/* Waits for an enqueued Ovstage operation that produces `handle`; a completed operation
 * that left the handle invalid failed. The caller releases the handle either way. */
template <typename Handle>
ovspatialquery_api_status_t awaitHandle(ovstage_instance_t* stage, const ovstage_enqueue_result_t& enqueued,
                                        const Handle& handle, Handle invalid)
{
    if (enqueued.status != OVSTAGE_OK)
    {
        return mapStageStatus(enqueued.status);
    }
    const ovspatialquery_api_status_t status = waitAndRelease(stage, enqueued.op_index);
    return status == OVSPATIALQUERY_API_SUCCESS && handle == invalid ? OVSPATIALQUERY_API_OPERATION_FAILED : status;
}

/* Evaluates the filter's _worldExtent query and waits for its handle; the caller releases it. */
ovspatialquery_api_status_t enqueueFilterQuery(ovstage_instance_t* stage, const OwnedFilter& filter,
                                               ovstage_query_handle_t& queryHandle)
{
    return awaitHandle(
        stage, stageVtable(stage)->query(stageContext(stage), &filter.filter, &filter.worldExtent, 1u, &queryHandle),
        queryHandle, OVSTAGE_INVALID_QUERY_HANDLE);
}

/* Calls `visit` with the path of each prim that has a row of `attribute` through the query,
 * whatever the row holds; a failed read, an exception included, visits what it reached and
 * still releases its group and read. */
template <typename Visit>
ovspatialquery_api_status_t visitAttributePrims(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                                ovstage_query_handle_t queryHandle, ovx_token_t attribute,
                                                ovspatialquery_ordinal_t targetOrdinal, Visit visit)
{
    const ovstage_ordinal_range_t range{0u, static_cast<ovstage_ordinal_t>(targetOrdinal), false};
    ovstage_read_handle_t         readHandle = OVSTAGE_INVALID_READ_HANDLE;
    ovspatialquery_api_status_t   status     = awaitHandle(
        stage,
        stageVtable(stage)->read_attributes(stageContext(stage), queryHandle, &attribute, 1u, range, &readHandle),
        readHandle, OVSTAGE_INVALID_READ_HANDLE);
    ReadCursor cursor(stage);
    while (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ReadGroup                  group{};
        const ovstage_api_status_t fetchStatus = cursor.next(readHandle, OVSTAGE_TIMEOUT_INFINITE, group);
        if (fetchStatus != OVSTAGE_OK)
        {
            status = fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION ? status : mapStageStatus(fetchStatus);
            break;
        }
        try
        {
            std::vector<ovx_primpath_t> paths;
            status = resolveGroupPaths(group, dictionary, paths);
            for (size_t row = 0u; status == OVSPATIALQUERY_API_SUCCESS && row < paths.size(); ++row)
            {
                visit(paths[row]);
            }
        }
        catch (...)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        (void)releaseReadGroup(stage, group);
    }
    const ovspatialquery_api_status_t releaseStatus = releaseRead(stage, readHandle);
    return status == OVSPATIALQUERY_API_SUCCESS ? releaseStatus : status;
}

/* Every PointInstancer's prototype targets, authored and RENDERING's copies alike, read from
 * whichever of the instancer's three prototype columns the runtime writes, and every prim
 * carrying `_instancerPath`, which RENDERING writes on each copy; a stage without instancers
 * costs one counted query. */
ovspatialquery_api_status_t readInstancerPrototypes(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                                    const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                    std::vector<ovx_primpath_t>& outTargets)
{
    const ovx_string_t  instancerType{"PointInstancer", sizeof("PointInstancer") - 1u};
    ovstage_predicate_t predicate{};
    predicate.attribute.token = filter.usdPrimType;
    predicate.op              = OVSTAGE_FILTER_OP_IN;
    predicate.values          = &instancerType;
    predicate.value_count     = 1u;
    const ovstage_filter_t      instancerFilter{&predicate, 1u};
    const auto&                 columns     = filter.instancerPrototypes;
    ovstage_query_handle_t      queryHandle = OVSTAGE_INVALID_QUERY_HANDLE;
    ovstage_read_handle_t       readHandle  = OVSTAGE_INVALID_READ_HANDLE;
    ovspatialquery_api_status_t status      = awaitHandle(
        stage,
        stageVtable(stage)->query(stageContext(stage), &instancerFilter, columns.data(), columns.size(), &queryHandle),
        queryHandle, OVSTAGE_INVALID_QUERY_HANDLE);
    uint64_t instancerCount = 0u;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ovstage_query_result_t result{};
        status = mapStageStatus(stageVtable(stage)->fetch_query_result(stageContext(stage), queryHandle,
                                                                       OVSTAGE_TIMEOUT_INFINITE, &result));
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            instancerCount = result.total_prim_count;
            status         = mapStageStatus(stageVtable(stage)->release_query_result(stageContext(stage), &result));
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS && instancerCount != 0u)
    {
        const ovstage_ordinal_range_t range{0u, static_cast<ovstage_ordinal_t>(targetOrdinal), false};
        status = awaitHandle(stage,
                             stageVtable(stage)->read_attributes(stageContext(stage), queryHandle, columns.data(),
                                                                 columns.size(), range, &readHandle),
                             readHandle, OVSTAGE_INVALID_READ_HANDLE);
    }
    ReadCursor cursor(stage);
    while (status == OVSPATIALQUERY_API_SUCCESS && readHandle != OVSTAGE_INVALID_READ_HANDLE)
    {
        ReadGroup                  group{};
        const ovstage_api_status_t fetchStatus = cursor.next(readHandle, OVSTAGE_TIMEOUT_INFINITE, group);
        if (fetchStatus != OVSTAGE_OK)
        {
            status = fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION ? status : mapStageStatus(fetchStatus);
            break;
        }
        /* A relationship reads as one array of path values per instancer. */
        if (!group.is_delete &&
            (!group.is_array || group.data.tensor_count != group.prims.count || group.data.tensors == nullptr))
        {
            status = OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        for (uint32_t row = 0u; status == OVSPATIALQUERY_API_SUCCESS && !group.is_delete && row < group.prims.count;
             ++row)
        {
            const DLTensor& tensor = group.data.tensors[row];
            if (tensor.device.device_type != kDLCPU || tensor.dtype.code != kDLUInt || tensor.dtype.bits != 64u ||
                tensor.dtype.lanes != 1u || tensor.ndim != 1 || tensor.shape == nullptr || tensor.shape[0] < 0 ||
                (tensor.strides != nullptr && tensor.strides[0] != 1) ||
                (tensor.shape[0] != 0 && tensor.data == nullptr))
            {
                status = OVSPATIALQUERY_API_NOT_SUPPORTED;
                break;
            }
            const auto* values = reinterpret_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
            for (int64_t index = 0; index < tensor.shape[0]; ++index)
            {
                ovx_primpath_t target = OVX_INVALID_PRIMPATH;
                std::memcpy(&target, values + static_cast<size_t>(index) * sizeof(target), sizeof(target));
                if (target != OVX_INVALID_PRIMPATH)
                {
                    outTargets.push_back(target);
                }
            }
        }
        (void)releaseReadGroup(stage, group);
    }
    const ovspatialquery_api_status_t readReleaseStatus = releaseRead(stage, readHandle);
    status = status == OVSPATIALQUERY_API_SUCCESS ? readReleaseStatus : status;
    const ovspatialquery_api_status_t queryReleaseStatus = releaseQuery(stage, queryHandle);
    status = status == OVSPATIALQUERY_API_SUCCESS ? queryReleaseStatus : status;
    if (status != OVSPATIALQUERY_API_SUCCESS || instancerCount == 0u)
    {
        return status;
    }
    ovstage_predicate_t copyPredicate{};
    copyPredicate.attribute.token = filter.instancerPath;
    copyPredicate.op              = OVSTAGE_FILTER_OP_HAS;
    const ovstage_filter_t copyFilter{&copyPredicate, 1u};
    status = awaitHandle(
        stage, stageVtable(stage)->query(stageContext(stage), &copyFilter, &filter.instancerPath, 1u, &queryHandle),
        queryHandle, OVSTAGE_INVALID_QUERY_HANDLE);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = visitAttributePrims(stage, dictionary, queryHandle, filter.instancerPath, targetOrdinal,
                                     [&](ovx_primpath_t path) { outTargets.push_back(path); });
    }
    const ovspatialquery_api_status_t copyReleaseStatus = releaseQuery(stage, queryHandle);
    return status == OVSPATIALQUERY_API_SUCCESS ? copyReleaseStatus : status;
}

/* The roots of the prims no population publishes as themselves. Ovstage keeps a copy of each
 * native-instancing prototype under a root that every domain marks with `PrototypeRootAPI` in
 * `usd-schemas`, so one filter query finds them (ADR-0050). A PointInstancer's prototype
 * prims, and the copies RENDERING makes of them under the instancer, are drawn only by its
 * instances, so each target is a root too. Each root is kept as its path's elements. A
 * stage without either has none. */
class PrototypeRoots final : public DroppedRows
{
  public:
    ovspatialquery_api_status_t read(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                     const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal)
    {
        ovspatialquery_api_status_t status = readCopies(stage, dictionary, filter, targetOrdinal);
        std::vector<ovx_primpath_t> targets;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readInstancerPrototypes(stage, dictionary, filter, targetOrdinal, targets);
        }
        /* RENDERING's copies are both targets and `_instancerPath` carriers, and instancers
         * may share prototypes; each root is kept once. */
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        for (size_t first = 0u; status == OVSPATIALQUERY_API_SUCCESS && first < targets.size(); first += kBatch)
        {
            status = forEachPathElements(dictionary, targets.data() + first, std::min(kBatch, targets.size() - first),
                                         [&](size_t, const ovx_token_t* elements, size_t count) {
                                             roots_.push_back({{elements, elements + count}, kDroppedPointPrototype});
                                         });
        }
        return status;
    }

    bool empty() const noexcept { return roots_.empty(); }

    /* Flags each of `paths` with the kind of the first root it is or lies under, 0 for none. */
    ovspatialquery_api_status_t mark(path_dictionary_instance_t* dictionary, const std::vector<ovx_primpath_t>& paths,
                                     std::vector<uint8_t>& outCopy) const override
    {
        outCopy.assign(paths.size(), 0u);
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        for (size_t first = 0u; status == OVSPATIALQUERY_API_SUCCESS && first < paths.size(); first += kBatch)
        {
            status = forEachPathElements(dictionary, paths.data() + first, std::min(kBatch, paths.size() - first),
                                         [&](size_t index, const ovx_token_t* elements, size_t count)
                                         {
                                             for (const Root& root : roots_)
                                             {
                                                 if (root.elements.size() <= count &&
                                                     std::equal(root.elements.begin(), root.elements.end(), elements))
                                                 {
                                                     outCopy[first + index] = root.kind;
                                                     break;
                                                 }
                                             }
                                         });
        }
        return status;
    }

  private:
    struct Root
    {
        std::vector<ovx_token_t> elements;
        uint8_t                  kind;
    };

    ovspatialquery_api_status_t readCopies(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                           const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal)
    {
        ovx_string_t        schema{"PrototypeRootAPI", sizeof("PrototypeRootAPI") - 1u};
        ovstage_predicate_t predicate{};
        predicate.attribute.token = filter.usdSchemas;
        predicate.op              = OVSTAGE_FILTER_OP_CONTAINS;
        predicate.values          = &schema;
        predicate.value_count     = 1u;
        const ovstage_filter_t      rootFilter{&predicate, 1u};
        ovstage_query_handle_t      queryHandle = OVSTAGE_INVALID_QUERY_HANDLE;
        std::vector<ovx_primpath_t> paths;
        ovspatialquery_api_status_t status = awaitHandle(
            stage, stageVtable(stage)->query(stageContext(stage), &rootFilter, &filter.usdSchemas, 1u, &queryHandle),
            queryHandle, OVSTAGE_INVALID_QUERY_HANDLE);
        /* A stage without native instances, the common case, matches no root, so its rows
         * are read only when the count says there are some. */
        uint64_t rootCount = 0u;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            ovstage_query_result_t result{};
            status = mapStageStatus(stageVtable(stage)->fetch_query_result(stageContext(stage), queryHandle,
                                                                           OVSTAGE_TIMEOUT_INFINITE, &result));
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                rootCount = result.total_prim_count;
                status    = mapStageStatus(stageVtable(stage)->release_query_result(stageContext(stage), &result));
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && rootCount != 0u)
        {
            status = visitAttributePrims(stage, dictionary, queryHandle, filter.usdSchemas, targetOrdinal,
                                         [&](ovx_primpath_t path) { paths.push_back(path); });
        }
        const ovspatialquery_api_status_t releaseStatus = releaseQuery(stage, queryHandle);
        status                                          = status == OVSPATIALQUERY_API_SUCCESS ? releaseStatus : status;
        for (size_t first = 0u; status == OVSPATIALQUERY_API_SUCCESS && first < paths.size(); first += kBatch)
        {
            status = forEachPathElements(dictionary, paths.data() + first, std::min(kBatch, paths.size() - first),
                                         [&](size_t, const ovx_token_t* elements, size_t count) {
                                             roots_.push_back({{elements, elements + count}, kDroppedPrototypeCopy});
                                         });
        }
        return status;
    }

    static constexpr size_t kBatch = 1024u;

    std::vector<Root> roots_;
};

/* Ovstage's count of instance-proxy rows on the stage; zero when it was populated without
 * the RENDERING domain, which alone writes them (ADR-0050). */
ovspatialquery_api_status_t countProxyRows(ovstage_instance_t* stage, const OwnedFilter& filter, uint64_t& outCount)
{
    ovstage_predicate_t predicate{};
    predicate.attribute.token = filter.protoPrimType;
    predicate.op              = OVSTAGE_FILTER_OP_HAS;
    const ovstage_filter_t      proxyFilter{&predicate, 1u};
    ovstage_query_handle_t      queryHandle = OVSTAGE_INVALID_QUERY_HANDLE;
    ovspatialquery_api_status_t status      = awaitHandle(
        stage, stageVtable(stage)->query(stageContext(stage), &proxyFilter, &filter.protoPrimType, 1u, &queryHandle),
        queryHandle, OVSTAGE_INVALID_QUERY_HANDLE);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ovstage_query_result_t result{};
        status = mapStageStatus(stageVtable(stage)->fetch_query_result(stageContext(stage), queryHandle,
                                                                       OVSTAGE_TIMEOUT_INFINITE, &result));
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            outCount = result.total_prim_count;
            status   = mapStageStatus(stageVtable(stage)->release_query_result(stageContext(stage), &result));
        }
    }
    const ovspatialquery_api_status_t releaseStatus = releaseQuery(stage, queryHandle);
    return status == OVSPATIALQUERY_API_SUCCESS ? releaseStatus : status;
}

/* A query over the prototype prims instance proxies take their definitions from (ADR-0050):
 * ovstage queries an interned path list, which must outlive the query, so both are released
 * together, the query first. */
class PrototypeQuery
{
  public:
    PrototypeQuery(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary) noexcept
        : stage_(stage), dictionary_(dictionary)
    {}
    PrototypeQuery(const PrototypeQuery&)            = delete;
    PrototypeQuery& operator=(const PrototypeQuery&) = delete;
    ~PrototypeQuery() { (void)close(); }

    ovspatialquery_api_status_t open(const std::vector<ovx_primpath_t>& paths)
    {
        if (stageVtable(stage_)->query_from_path_list == nullptr ||
            dictionary_->vtable->create_path_list_from_paths == nullptr ||
            dictionary_->vtable->release_path_list_reference == nullptr)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        ovspatialquery_api_status_t status = mapDictionaryResult(
            dictionary_,
            dictionary_->vtable->create_path_list_from_paths(dictionary_->context, paths.data(), paths.size(), &list_));
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = mapStageStatus(stageVtable(stage_)->query_from_path_list(stageContext(stage_), list_, &query_));
        }
        return status == OVSPATIALQUERY_API_SUCCESS && query_ == OVSTAGE_INVALID_QUERY_HANDLE
                   ? OVSPATIALQUERY_API_OPERATION_FAILED
                   : status;
    }

    ovstage_query_handle_t handle() const noexcept { return query_; }

    ovspatialquery_api_status_t close() noexcept
    {
        ovspatialquery_api_status_t status = releaseQuery(stage_, query_);
        if (list_ != OVX_INVALID_PRIMPATH_LIST)
        {
            const ovspatialquery_api_status_t listStatus = mapDictionaryResult(
                dictionary_, dictionary_->vtable->release_path_list_reference(dictionary_->context, list_));
            list_  = OVX_INVALID_PRIMPATH_LIST;
            status = status == OVSPATIALQUERY_API_SUCCESS ? listStatus : status;
        }
        return status;
    }

  private:
    ovstage_instance_t*         stage_;
    path_dictionary_instance_t* dictionary_;
    ovx_primpath_list_t         list_  = OVX_INVALID_PRIMPATH_LIST;
    ovstage_query_handle_t      query_ = OVSTAGE_INVALID_QUERY_HANDLE;
};

constexpr const char* kNoProxyRowsRule =
    "it is ovstage's copy of a native-instancing prototype, and the stage has no instance-proxy rows: populate the "
    "RENDERING or ALL domain to publish its instances";

// STALE_ORDINAL unless _worldExtent's write floor is exactly the target ordinal.
ovspatialquery_api_status_t checkWriteFloor(ovstage_instance_t* stage, const OwnedFilter& filter,
                                            ovspatialquery_ordinal_t targetOrdinal)
{
    ovstage_ordinal_query_handle_t                 ordinalHandle = OVSTAGE_INVALID_ORDINAL_QUERY_HANDLE;
    const decltype(ovstage_predicate_t::attribute) floorAttribute{filter.worldExtent, ovx_string_t{nullptr, 0u}};
    ovspatialquery_api_status_t                    status = awaitHandle(
        stage, stageVtable(stage)->get_attribute_write_floor(stageContext(stage), floorAttribute, &ordinalHandle),
        ordinalHandle, OVSTAGE_INVALID_ORDINAL_QUERY_HANDLE);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ovstage_ordinal_t observedFloor = 0u;
        status = mapStageStatus(stageVtable(stage)->fetch_ordinal(stageContext(stage), ordinalHandle,
                                                                  OVSTAGE_TIMEOUT_INFINITE, &observedFloor));
        if (status == OVSPATIALQUERY_API_SUCCESS && observedFloor != static_cast<ovstage_ordinal_t>(targetOrdinal))
        {
            status = OVSPATIALQUERY_API_STALE_ORDINAL;
        }
    }
    const ovspatialquery_api_status_t releaseStatus = releaseOrdinalQuery(stage, ordinalHandle);
    return status == OVSPATIALQUERY_API_SUCCESS ? releaseStatus : status;
}

/* Evaluates the filter once at the sealed target ordinal and reads _worldExtent
 * through it into the matched prims' objects. With outHasBounds every matched prim
 * is kept, flagged bounded (1) or carrying the package's missing-bounds sentinel (0);
 * without it only identities are read. With outRetainedQuery the query handle stays
 * open on success so the caller can read further attributes of the same matched set;
 * the caller then releases it. outPointPrototypes counts the matched PointInstancer
 * prototype prims and copies the capture dropped. */
ovspatialquery_api_status_t acquireFilterSnapshot(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                                  const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                  std::vector<PopulationObject>& outObjects, uint64_t& outMatchedCount,
                                                  std::vector<uint8_t>*   outHasBounds       = nullptr,
                                                  ovstage_query_handle_t* outRetainedQuery   = nullptr,
                                                  PrimRefusal*            outRefusal         = nullptr,
                                                  uint64_t*               outPointPrototypes = nullptr)
{
    if (outRetainedQuery != nullptr)
    {
        *outRetainedQuery = OVSTAGE_INVALID_QUERY_HANDLE;
    }

    ovstage_query_handle_t      queryHandle = OVSTAGE_INVALID_QUERY_HANDLE;
    ovstage_read_handle_t       readHandle  = OVSTAGE_INVALID_READ_HANDLE;
    ovstage_query_result_t      queryResult{};
    bool                        hasQueryResult = false;
    size_t                      matchedCount   = 0u;
    ovspatialquery_api_status_t status         = OVSPATIALQUERY_API_SUCCESS;

    status = checkWriteFloor(stage, filter, targetOrdinal);

    /* A filter that names its prims queries them from its interned path list instead of
     * having Ovstage derive `usd-path` per value. The reads below return a row for every
     * listed path that has a prim with bounds state, so when every path has one the
     * matched count is the list's size; otherwise it is the number of listed prims that
     * exist, read through the same query. */
    const bool exactPaths =
        filter.usdPathList != OVX_INVALID_PRIMPATH_LIST && stageVtable(stage)->query_from_path_list != nullptr;
    if (status == OVSPATIALQUERY_API_SUCCESS && exactPaths)
    {
        status = mapStageStatus(
            stageVtable(stage)->query_from_path_list(stageContext(stage), filter.usdPathList, &queryHandle));
        if (status == OVSPATIALQUERY_API_SUCCESS && queryHandle == OVSTAGE_INVALID_QUERY_HANDLE)
        {
            status = OVSPATIALQUERY_API_OPERATION_FAILED;
        }
        matchedCount = filter.usdPathCount;
    }
    else if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = enqueueFilterQuery(stage, filter, queryHandle);
    }
    if (status == OVSPATIALQUERY_API_SUCCESS && !exactPaths)
    {
        const ovstage_api_status_t fetchStatus = stageVtable(stage)->fetch_query_result(
            stageContext(stage), queryHandle, OVSTAGE_TIMEOUT_INFINITE, &queryResult);
        status         = mapStageStatus(fetchStatus);
        hasQueryResult = fetchStatus == OVSTAGE_OK;
        if (hasQueryResult)
        {
            matchedCount = queryResult.total_prim_count;
            if (queryResult.query_result_id == OVSTAGE_INVALID_QUERY_RESULT_ID || queryResult.all_handle != queryHandle)
            {
                status = OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        const ovstage_ordinal_range_t range{0u, static_cast<ovstage_ordinal_t>(targetOrdinal), false};
        status = awaitHandle(stage,
                             stageVtable(stage)->read_attributes(stageContext(stage), queryHandle, &filter.worldExtent,
                                                                 1u, range, &readHandle),
                             readHandle, OVSTAGE_INVALID_READ_HANDLE);
    }

    std::unordered_set<uint64_t>  identities;
    std::vector<PopulationObject> objects;
    std::vector<uint8_t>          hasBounds;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        try
        {
            identities.reserve(matchedCount);
            objects.reserve(matchedCount);
            if (outHasBounds != nullptr)
            {
                hasBounds.reserve(matchedCount);
            }
        }
        catch (const std::bad_alloc&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (const std::length_error&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    /* Ovstage's copies of native-instancing prototypes (ADR-0050) and a PointInstancer's
     * prototype prims and their copies are never published as themselves: they leave the
     * matched set before every other rule, their bounds included, uncounted. */
    PrototypeRoots  roots;
    DroppedRowCount dropped;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        try
        {
            status = roots.read(stage, dictionary, filter, targetOrdinal);
        }
        catch (const std::bad_alloc&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (const std::length_error&)
        {
            status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    ReadCursor cursor(stage);
    while (status == OVSPATIALQUERY_API_SUCCESS)
    {
        ReadGroup                  group{};
        const ovstage_api_status_t fetchStatus = cursor.next(readHandle, OVSTAGE_TIMEOUT_INFINITE, group);
        if (fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION)
        {
            break;
        }
        if (fetchStatus != OVSTAGE_OK)
        {
            status = mapStageStatus(fetchStatus);
            break;
        }
        ovspatialquery_api_status_t groupStatus = OVSPATIALQUERY_API_SUCCESS;
        try
        {
            const DroppedRows* copies = roots.empty() ? nullptr : &roots;
            groupStatus               = outHasBounds != nullptr
                                            ? appendGroup(group, filter.worldExtent, targetOrdinal, dictionary, identities, objects,
                                                          hasBounds, copies, &dropped)
                                            : appendIdentityGroup(group, filter.worldExtent, targetOrdinal, dictionary, identities,
                                                                  objects, copies, &dropped);
        }
        catch (const std::bad_alloc&)
        {
            groupStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (const std::length_error&)
        {
            groupStatus = OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
        catch (...)
        {
            groupStatus = OVSPATIALQUERY_API_INTERNAL;
        }
        const ovstage_api_status_t releaseStatus = releaseReadGroup(stage, group);
        if (groupStatus == OVSPATIALQUERY_API_SUCCESS)
        {
            groupStatus = mapStageStatus(releaseStatus);
        }
        status = groupStatus;
    }
    uint64_t droppedRows   = dropped.count;
    uint64_t droppedCopies = dropped.copies;
    uint64_t firstCopy     = dropped.first;
    try
    {
        if (status == OVSPATIALQUERY_API_SUCCESS && exactPaths && objects.size() + droppedRows != matchedCount)
        {
            /* A listed path without a row: count the listed prims that exist through the same
             * query, one row each, instead of having Ovstage evaluate the filter, which costs
             * its prims times the listed paths. A path listed twice counts once. */
            std::unordered_set<uint64_t> listed;
            listed.reserve(matchedCount);
            status       = visitAttributePrims(stage, dictionary, queryHandle, filter.usdPath, targetOrdinal,
                                               [&](ovx_primpath_t path) { listed.insert(handleValue(path)); });
            matchedCount = listed.size();
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            matchedCount = matchedCount >= droppedRows ? matchedCount - droppedRows : 0u;
        }
        /* A dropped prim without a _worldExtent row (a copy on a stage populated without
         * RENDERING, an authored PointInstancer prototype) is in the matched count but was not
         * read; enumerating every matched prim finds them all. */
        if (status == OVSPATIALQUERY_API_SUCCESS && !roots.empty() && objects.size() != matchedCount)
        {
            std::vector<ovx_primpath_t> paths;
            std::vector<uint8_t>        copies;
            status = visitAttributePrims(stage, dictionary, queryHandle, filter.usdPath, targetOrdinal,
                                         [&](ovx_primpath_t path) { paths.push_back(path); });
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = roots.mark(dictionary, paths, copies);
            }
            uint64_t allDropped = 0u, allCopies = 0u;
            for (size_t index = 0u; status == OVSPATIALQUERY_API_SUCCESS && index < paths.size(); ++index)
            {
                if (copies[index] != 0u)
                {
                    ++allDropped;
                    identities.insert(handleValue(paths[index]));
                }
                if (copies[index] == kDroppedPrototypeCopy)
                {
                    firstCopy = allCopies++ == 0u && firstCopy == 0u ? handleValue(paths[index]) : firstCopy;
                }
            }
            matchedCount += droppedRows;
            matchedCount  = matchedCount >= allDropped ? matchedCount - allDropped : 0u;
            droppedRows   = allDropped;
            droppedCopies = allCopies;
        }
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    /* Without proxy rows the copies are all the stage holds of its instances, so dropping
     * them would publish the stage without its instances. */
    if (status == OVSPATIALQUERY_API_SUCCESS && outHasBounds != nullptr && droppedCopies != 0u)
    {
        uint64_t proxyRows = 0u;
        status             = countProxyRows(stage, filter, proxyRows);
        if (status == OVSPATIALQUERY_API_SUCCESS && proxyRows == 0u)
        {
            status = OVSPATIALQUERY_API_NOT_SUPPORTED;
            if (outRefusal != nullptr)
            {
                outRefusal->note(firstCopy, OVX_INVALID_TOKEN, kNoProxyRowsRule);
            }
        }
    }
    if (status == OVSPATIALQUERY_API_SUCCESS && objects.size() > matchedCount)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    if (status == OVSPATIALQUERY_API_SUCCESS && outHasBounds != nullptr && objects.size() != matchedCount)
    {
        status = visitAttributePrims(stage, dictionary, queryHandle, filter.usdPath, targetOrdinal,
                                     [&](ovx_primpath_t path)
                                     {
                                         if (identities.insert(handleValue(path)).second)
                                         {
                                             PopulationObject object{};
                                             object.id = objectIdFromValue(handleValue(path));
                                             objects.push_back(object);
                                             hasBounds.push_back(0);
                                         }
                                     });
    }
    const ovspatialquery_api_status_t readReleaseStatus = releaseRead(stage, readHandle);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = readReleaseStatus;
    }
    if (hasQueryResult)
    {
        const ovstage_api_status_t releaseStatus =
            stageVtable(stage)->release_query_result(stageContext(stage), &queryResult);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = mapStageStatus(releaseStatus);
        }
    }
    if (status != OVSPATIALQUERY_API_SUCCESS || outRetainedQuery == nullptr)
    {
        const ovspatialquery_api_status_t queryReleaseStatus = releaseQuery(stage, queryHandle);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = queryReleaseStatus;
        }
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    if (outRetainedQuery != nullptr)
    {
        *outRetainedQuery = queryHandle;
    }
    outObjects      = std::move(objects);
    outMatchedCount = matchedCount;
    if (outPointPrototypes != nullptr)
    {
        *outPointPrototypes = droppedRows - droppedCopies;
    }
    if (outHasBounds != nullptr)
    {
        *outHasBounds = std::move(hasBounds);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

// Prepares the acquisition's authored meshes, sharing any preparation the population
// already holds for identical content. The hash selects candidates; a full comparison
// confirms the match, so a collision cannot alias two meshes. Drop expired entries
// before acquisition, including old content hashes that will never be queried again.
ovspatialquery_api_status_t prepareSharedAuthoredMeshes(
    OwnedFilter& filter, const TriangleGeometrySource& source,
    std::vector<std::shared_ptr<const TriangleObjectGeometry>>& outObjects)
{
    {
        std::lock_guard<std::mutex> lock(filter.preparedMeshMutex);
        for (auto entry = filter.preparedMeshes.begin(); entry != filter.preparedMeshes.end();)
        {
            if (entry->second.expired())
            {
                entry = filter.preparedMeshes.erase(entry);
            }
            else
            {
                ++entry;
            }
        }
    }
    std::vector<std::shared_ptr<const PreparedTriangleMesh>> meshes;
    meshes.reserve(source.meshes.size());
    for (const TriangleMeshSource& mesh : source.meshes)
    {
        const PreparedOvstageMeshKey key{triangleMeshContentHash(mesh), mesh.ovstageMetric, mesh.ovstageMirrored};
        std::shared_ptr<const PreparedTriangleMesh> prepared;
        {
            std::lock_guard<std::mutex> lock(filter.preparedMeshMutex);
            auto                        range = filter.preparedMeshes.equal_range(key);
            for (auto entry = range.first; entry != range.second;)
            {
                std::shared_ptr<const PreparedTriangleMesh> held = entry->second.lock();
                if (held == nullptr)
                {
                    entry = filter.preparedMeshes.erase(entry);
                    continue;
                }
                if (prepared == nullptr && preparedTriangleMeshMatches(*held, mesh))
                {
                    prepared = std::move(held);
                }
                ++entry;
            }
        }
        if (prepared == nullptr)
        {
            const ovspatialquery_api_status_t status = prepareTriangleMesh(mesh, prepared);
            if (status != OVSPATIALQUERY_API_SUCCESS && status != OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                status != OVSPATIALQUERY_API_INVALID_STATE && status != OVSPATIALQUERY_API_NOT_SUPPORTED)
                return status;
            if (prepared != nullptr)
            {
                std::lock_guard<std::mutex> lock(filter.preparedMeshMutex);
                filter.preparedMeshes.emplace(key, prepared);
            }
        }
        meshes.push_back(std::move(prepared));
    }
    std::vector<std::shared_ptr<const TriangleObjectGeometry>> objects;
    objects.reserve(source.objects.size());
    for (const TriangleObjectSource& object : source.objects)
    {
        if (object.meshIndex >= meshes.size())
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        if (meshes[static_cast<size_t>(object.meshIndex)] == nullptr)
        {
            objects.push_back(nullptr);
            continue;
        }
        std::shared_ptr<const TriangleObjectGeometry> bound;
        const ovspatialquery_api_status_t             status = bindPreparedTriangleGeometry(
            meshes[static_cast<size_t>(object.meshIndex)], object.worldFromLocal, object.doubleSided, bound);
        if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT || status == OVSPATIALQUERY_API_INVALID_STATE ||
            status == OVSPATIALQUERY_API_NOT_SUPPORTED)
        {
            objects.push_back(nullptr);
            continue;
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        objects.push_back(std::move(bound));
    }
    outObjects = std::move(objects);
    return OVSPATIALQUERY_API_SUCCESS;
}

/* The rule a primitive's or mesh's world transform broke, by the status its qualification returned. */
const char* transformRule(ovspatialquery_api_status_t status, ExactGeometryFamily family)
{
    if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT)
    {
        return "its world transform is not finite or lies outside the input domain of ovspatialquery.h";
    }
    switch (family)
    {
    case ExactGeometryFamily::box:
        return "ovstage populations publish a Cube only under an affine world transform with orthogonal axes and "
               "positive scales";
    case ExactGeometryFamily::sphere:
    case ExactGeometryFamily::capsule:
        return "ovstage populations publish a Sphere or Capsule only under an affine world transform with positive "
               "uniform scale";
    default:
        return "ovstage populations publish a Mesh only under a right-handed rigid world transform with positive "
               "uniform scale";
    }
}

// See acquireTypedSnapshot; notes the first prim it refuses in `refusal`.
/* Replaces `query`, the filter's evaluation, with a query over the prims the scope keeps,
 * so that every later read sees only those prims' rows: a scoped-out prim whose attributes
 * have a layout the adapter refuses then fails nothing, as if the filter had rejected it.
 * Unchanged when the scope keeps every prim. */
ovspatialquery_api_status_t narrowToScope(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                          const std::vector<uint64_t>& ids, const std::vector<uint8_t>& inScope,
                                          ovstage_query_handle_t& query)
{
    if (std::find(inScope.begin(), inScope.end(), uint8_t{0u}) == inScope.end() ||
        stageVtable(stage)->query_from_path_list == nullptr ||
        dictionary->vtable->create_path_list_from_paths == nullptr ||
        dictionary->vtable->release_path_list_reference == nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    std::vector<ovx_primpath_t> kept;
    kept.reserve(ids.size());
    for (size_t i = 0u; i < ids.size(); ++i)
        if (inScope[i] != 0u)
            kept.push_back(makeHandle<ovx_primpath_t>(ids[i]));
    ovx_primpath_list_t         list   = OVX_INVALID_PRIMPATH_LIST;
    ovspatialquery_api_status_t status = mapDictionaryResult(
        dictionary,
        dictionary->vtable->create_path_list_from_paths(dictionary->context, kept.data(), kept.size(), &list));
    ovstage_query_handle_t narrowed = OVSTAGE_INVALID_QUERY_HANDLE;
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = mapStageStatus(stageVtable(stage)->query_from_path_list(stageContext(stage), list, &narrowed));
        if (status == OVSPATIALQUERY_API_SUCCESS && narrowed == OVSTAGE_INVALID_QUERY_HANDLE)
            status = OVSPATIALQUERY_API_OPERATION_FAILED;
    }
    if (list != OVX_INVALID_PRIMPATH_LIST)
    {
        const ovspatialquery_api_status_t released =
            mapDictionaryResult(dictionary, dictionary->vtable->release_path_list_reference(dictionary->context, list));
        status = status == OVSPATIALQUERY_API_SUCCESS ? released : status;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        if (narrowed != OVSTAGE_INVALID_QUERY_HANDLE)
            (void)releaseQuery(stage, narrowed);
        return status;
    }
    status = releaseQuery(stage, query);
    query  = narrowed;
    return status;
}

/* Material identity of an Ovstage population (ADR-0066), resolved on the host after the
 * objects are acquired. An object takes the first target of `material:binding:physics` on
 * its prim or the nearest ancestor that has one; failing that, of `material:binding` the
 * same way, as omni.physx's Ovstage source resolves a collider's physics material. A
 * triangle mesh also takes, per source face, the binding of its face GeomSubset children of
 * the `materialBind` family: the subset's own physics binding, else its own all-purpose
 * one. Overlapping subsets resolve in path order, the later winning; a face in no bound
 * subset keeps the object's material. Collection bindings, binding strength and
 * instance-proxy bindings are not modelled. Material is optional: a binding or subset
 * attribute the reads cannot take, such as `int64` indices, contributes nothing instead of
 * failing the publication, and a subset index outside its mesh's source faces is ignored.
 * One stage-side `HAS` filter per relationship counts the prims that author it, so a stage
 * without any costs two filter evaluations and nothing else. A relationship on few prims is
 * read through that filter; one on more than twice the population's object and ancestor
 * paths is read for those paths only, so the cost and charge follow the population, not the
 * stage. Face subsets are found by enumerating the paths of the stage's GeomSubsets, nothing
 * else of them, so no subset of a published mesh is ever missed; a subset belongs to the
 * mesh its path's parent names, through the path dictionary, and only those are read
 * beyond their paths. Every scratch
 * allocation is charged to the host budget before it is made and released when resolution
 * ends; a refused charge fails the publication with HOST_BUDGET (ADR-0009), as a refused
 * convex-hull or scope-cache charge does. A stage without a function-table slot the resolver
 * calls, which object acquisition does not need, leaves every material zero. */
class OvstageMaterialResolver
{
  public:
    OvstageMaterialResolver(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                            const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal)
        : mStage(stage), mDictionary(dictionary), mFilter(filter), mTargetOrdinal(targetOrdinal)
    {}

    ovspatialquery_api_status_t resolve(std::vector<PopulationObject>& objects)
    {
        uint32_t major = 0u, minor = 0u, patch = 0u;
        stageVtable(mStage)->get_version(stageContext(mStage), &major, &minor, &patch);
        if (objects.empty() || (major == 0u && minor < 3u) || stageVtable(mStage)->query_from_path_list == nullptr ||
            mDictionary->vtable->create_path_list_from_paths == nullptr ||
            mDictionary->vtable->release_path_list_reference == nullptr ||
            mDictionary->vtable->create_paths_from_tokens == nullptr ||
            mDictionary->vtable->get_strings_from_tokens == nullptr)
            return OVSPATIALQUERY_API_SUCCESS;
        ovspatialquery_api_status_t status = internTokens();
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        /* How many prims of the stage author each binding, through one stage-side filter
         * each that reads no row; with none bound, every material stays zero and nothing
         * else is read. */
        uint64_t physicsCount = 0u, generalCount = 0u;
        status = countBound(mPhysicsBinding, physicsCount);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = countBound(mBinding, generalCount);
        if (status != OVSPATIALQUERY_API_SUCCESS || (physicsCount == 0u && generalCount == 0u))
            return status;
        /* Every object's chain of prim paths, the object first, then each ancestor below "/". */
        status =
            charge(objects.size() * (sizeof(ovx_primpath_t) + sizeof(std::vector<ovx_primpath_t>) + sizeof(uint8_t)));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<std::vector<ovx_primpath_t>> chains(objects.size());
        std::vector<ovx_primpath_t>              ids(objects.size());
        std::vector<uint8_t>                     physicsBound(objects.size(), 0u);
        for (size_t i = 0u; i < objects.size(); ++i)
            ids[i] = makeHandle<ovx_primpath_t>(objects[i].id.value);
        status = ancestorChains(ids, chains);
        /* A relationship bound on few prims is read through its stage-side filter. One
         * bound on more prims than twice the population's object and ancestor paths (a
         * render binding on every prim of a large stage) is read only for those paths, so
         * its cost and host charge follow the population, not the stage. Face subsets are
         * resolved separately, below, and never depend on this choice. */
        uint64_t links = 0u;
        for (const std::vector<ovx_primpath_t>& chain : chains)
            links += chain.size();
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = charge(links * (sizeof(ovx_primpath_t) + kHashEntryBytes));
        std::vector<ovx_primpath_t> unique;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            unique.reserve(static_cast<size_t>(links));
            std::unordered_set<uint64_t> seen;
            for (const std::vector<ovx_primpath_t>& chain : chains)
                for (ovx_primpath_t path : chain)
                    if (seen.insert(handleValue(path)).second)
                        unique.push_back(path);
        }
        const uint64_t                         limit = 2u * static_cast<uint64_t>(unique.size());
        std::unordered_map<uint64_t, uint64_t> physics;
        std::unordered_map<uint64_t, uint64_t> general;
        for (const auto& [relationship, count, bindings] : {std::make_tuple(mPhysicsBinding, physicsCount, &physics),
                                                            std::make_tuple(mBinding, generalCount, &general)})
        {
            if (status != OVSPATIALQUERY_API_SUCCESS || count == 0u)
                continue;
            if (count <= limit)
            {
                status = boundPrims(relationship, *bindings);
                continue;
            }
            status = readPaths(unique, relationship, *bindings);
        }
        for (size_t i = 0u; status == OVSPATIALQUERY_API_SUCCESS && i < objects.size(); ++i)
        {
            objects[i].materialId = firstBinding(chains[i], physics);
            physicsBound[i]       = objects[i].materialId != 0u ? 1u : 0u;
            if (objects[i].materialId == 0u)
                objects[i].materialId = firstBinding(chains[i], general);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = resolveFaceMaterials(objects, physicsBound);
        return status;
    }

    /* The charge of the face material tables `resolve` left in the objects, which must
     * last until the publication's own staging reservation counts those tables. */
    std::vector<std::shared_ptr<void>> takeRetainedCharges() { return std::move(mRetainedCharges); }

  private:
    /* An unordered container's node, bucket and allocator overhead beyond its value. */
    static constexpr uint64_t kHashEntryBytes = 6u * sizeof(void*);
    /* Paths decomposed per forEachPathElements call (as PrototypeRoots does). */
    static constexpr size_t kPathBatch = 1024u;

    struct Subset
    {
        uint64_t             path        = 0u;
        uint64_t             elementType = 0u;
        uint64_t             familyName  = 0u;
        size_t               mesh        = 0u; // the owning mesh's object index
        std::vector<int32_t> indices;
        bool                 hasIndices = false;
    };

    ovspatialquery_api_status_t internTokens()
    {
        static const char* const     names[] = {"material:binding:physics",
                                                "material:binding",
                                                "elementType",
                                                "familyName",
                                                "indices",
                                                "face",
                                                "materialBind"};
        std::array<ovx_string_t, 7u> strings{};
        for (size_t i = 0u; i < strings.size(); ++i)
            strings[i] = ovx_string_t{names[i], std::strlen(names[i])};
        std::array<ovx_token_t, 7u>       tokens{};
        const ovspatialquery_api_status_t status =
            mapDictionaryResult(mDictionary, mDictionary->vtable->create_tokens_from_strings(
                                                 mDictionary->context, strings.data(), strings.size(), tokens.data()));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        mPhysicsBinding = tokens[0];
        mBinding        = tokens[1];
        mElementType    = tokens[2];
        mFamilyName     = tokens[3];
        mIndices        = tokens[4];
        mFace           = tokens[5];
        mMaterialBind   = tokens[6];
        return OVSPATIALQUERY_API_SUCCESS;
    }

    /* Each path's chain: itself, then its ancestors up to the first element below "/". */
    ovspatialquery_api_status_t ancestorChains(const std::vector<ovx_primpath_t>&        paths,
                                               std::vector<std::vector<ovx_primpath_t>>& outChains)
    {
        /* A counting pass sizes the prefixes, so that they are charged before they exist. */
        uint64_t                    elementCount = 0u, prefixElementCount = 0u;
        ovspatialquery_api_status_t status = forEachElements(paths,
                                                             [&](size_t, const ovx_token_t*, size_t count)
                                                             {
                                                                 elementCount += count;
                                                                 prefixElementCount +=
                                                                     static_cast<uint64_t>(count) * (count + 1u) / 2u;
                                                             });
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status =
                charge(prefixElementCount * sizeof(ovx_token_t) +
                       elementCount * (sizeof(size_t) + 2u * sizeof(ovx_primpath_t)) + paths.size() * sizeof(size_t));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<ovx_token_t> prefixTokens;
        std::vector<size_t>      prefixCounts;
        std::vector<size_t>      chainLengths(paths.size(), 0u);
        prefixTokens.reserve(static_cast<size_t>(prefixElementCount));
        prefixCounts.reserve(static_cast<size_t>(elementCount));
        status = forEachElements(paths,
                                 [&](size_t index, const ovx_token_t* elements, size_t count)
                                 {
                                     chainLengths[index] = count;
                                     for (size_t length = count; length > 0u; --length)
                                     {
                                         prefixTokens.insert(prefixTokens.end(), elements, elements + length);
                                         prefixCounts.push_back(length);
                                     }
                                 });
        if (status != OVSPATIALQUERY_API_SUCCESS || prefixCounts.empty())
            return status;
        std::vector<ovx_primpath_t> prefixes(prefixCounts.size(), OVX_INVALID_PRIMPATH);
        status      = mapDictionaryResult(mDictionary, mDictionary->vtable->create_paths_from_tokens(
                                                      mDictionary->context, prefixTokens.data(), prefixCounts.data(),
                                                      prefixCounts.size(), prefixes.data()));
        size_t next = 0u;
        for (size_t i = 0u; status == OVSPATIALQUERY_API_SUCCESS && i < paths.size(); ++i)
        {
            outChains[i].assign(prefixes.begin() + static_cast<std::ptrdiff_t>(next),
                                prefixes.begin() + static_cast<std::ptrdiff_t>(next + chainLengths[i]));
            next += chainLengths[i];
        }
        return status;
    }

    /* forEachPathElements over `paths` in batches of kPathBatch, so its token buffer and
     * per-path arrays stay bounded whatever the stage holds; that bounded scratch is
     * charged once, before the first batch. */
    template <typename Visit>
    ovspatialquery_api_status_t forEachElements(const std::vector<ovx_primpath_t>& paths, Visit visit)
    {
        ovspatialquery_api_status_t status = OVSPATIALQUERY_API_SUCCESS;
        if (!mBatchCharged && !paths.empty())
        {
            status        = charge(kPathBatch * (17u * sizeof(ovx_token_t) + sizeof(ovx_token_t*) + sizeof(size_t)));
            mBatchCharged = status == OVSPATIALQUERY_API_SUCCESS;
        }
        for (size_t first = 0u; status == OVSPATIALQUERY_API_SUCCESS && first < paths.size(); first += kPathBatch)
        {
            status = forEachPathElements(mDictionary, paths.data() + first, std::min(kPathBatch, paths.size() - first),
                                         [&](size_t index, const ovx_token_t* elements, size_t count)
                                         { visit(first + index, elements, count); });
        }
        return status;
    }

    /* Releases a resolver-local query and path list when it leaves scope, so an exception
     * between opening and releasing them (an allocation or a charge that throws) does not
     * leak stage resources. Explicit releases reset the handles, which makes this a no-op
     * on the normal path. */
    struct HandleGuard
    {
        OvstageMaterialResolver& resolver;
        ovstage_query_handle_t&  query;
        ovx_primpath_list_t*     list = nullptr;
        ~HandleGuard()
        {
            (void)releaseQuery(resolver.mStage, query);
            if (list != nullptr)
                resolver.releasePathList(*list);
        }
    };

    /* A query over exactly `paths`, without a stage-side filter evaluation. */
    ovspatialquery_api_status_t queryPaths(const std::vector<ovx_primpath_t>& paths, ovstage_query_handle_t& outQuery,
                                           ovx_primpath_list_t& outList)
    {
        if (stageVtable(mStage)->query_from_path_list == nullptr)
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        ovspatialquery_api_status_t status =
            mapDictionaryResult(mDictionary, mDictionary->vtable->create_path_list_from_paths(
                                                 mDictionary->context, paths.data(), paths.size(), &outList));
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status =
                mapStageStatus(stageVtable(mStage)->query_from_path_list(stageContext(mStage), outList, &outQuery));
        if (status == OVSPATIALQUERY_API_SUCCESS && outQuery == OVSTAGE_INVALID_QUERY_HANDLE)
            status = OVSPATIALQUERY_API_OPERATION_FAILED;
        return status;
    }

    void releasePathList(ovx_primpath_list_t& list)
    {
        if (list != OVX_INVALID_PRIMPATH_LIST)
            (void)mDictionary->vtable->release_path_list_reference(mDictionary->context, list);
        list = OVX_INVALID_PRIMPATH_LIST;
    }

    /* Reads one optional attribute: what the read cannot take contributes nothing. */
    template <typename Consumer>
    ovspatialquery_api_status_t readOptional(ovstage_query_handle_t query, ovx_token_t attribute,
                                             ovstage_attribute_semantic_t semantic, uint8_t code, uint8_t bits,
                                             bool allowNoneSemantic, Consumer consume)
    {
        return readTriangleAttribute(
            mStage, query, attribute, mTargetOrdinal, mDictionary, semantic, code, bits, 1u, allowNoneSemantic, true,
            consume, []() { return OVSPATIALQUERY_API_SUCCESS; }, true);
    }

    /* The first nonzero target of a relationship for every queried prim that has one. */
    ovspatialquery_api_status_t readFirstTargets(ovstage_query_handle_t query, ovx_token_t relationship,
                                                 std::unordered_map<uint64_t, uint64_t>& out)
    {
        return readOptional(query, relationship, OVSTAGE_SEMANTIC_RELATIONSHIP_PATH_ID, kDLUInt, 64u, true,
                            [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                            {
                                for (size_t offset = 0u; offset + 8u <= byteCount; offset += 8u)
                                {
                                    uint64_t target = 0u;
                                    std::memcpy(&target, bytes + offset, sizeof(target));
                                    if (target != 0u)
                                    {
                                        out[id] = target;
                                        break;
                                    }
                                }
                                return OVSPATIALQUERY_API_SUCCESS;
                            });
    }

    /* A stage-side filter for the prims that author `relationship`; the caller releases
     * `outQuery`. */
    ovspatialquery_api_status_t queryBound(ovx_token_t& relationship, ovstage_query_handle_t& outQuery,
                                           uint64_t& outCount)
    {
        ovstage_predicate_t predicate{};
        predicate.attribute.token = relationship;
        predicate.op              = OVSTAGE_FILTER_OP_HAS;
        const ovstage_filter_t      filter{&predicate, 1u};
        ovspatialquery_api_status_t status =
            awaitHandle(mStage, stageVtable(mStage)->query(stageContext(mStage), &filter, &relationship, 1u, &outQuery),
                        outQuery, OVSTAGE_INVALID_QUERY_HANDLE);
        outCount = 0u;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            ovstage_query_result_t result{};
            status = mapStageStatus(stageVtable(mStage)->fetch_query_result(stageContext(mStage), outQuery,
                                                                            OVSTAGE_TIMEOUT_INFINITE, &result));
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                outCount = result.total_prim_count;
                status   = mapStageStatus(stageVtable(mStage)->release_query_result(stageContext(mStage), &result));
            }
        }
        return status;
    }

    /* How many prims of the stage author `relationship`; no row is read or charged. */
    ovspatialquery_api_status_t countBound(ovx_token_t relationship, uint64_t& outCount)
    {
        ovstage_query_handle_t            query = OVSTAGE_INVALID_QUERY_HANDLE;
        const HandleGuard                 guard{*this, query};
        const ovspatialquery_api_status_t status   = queryBound(relationship, query, outCount);
        const ovspatialquery_api_status_t released = releaseQuery(mStage, query);
        return status == OVSPATIALQUERY_API_SUCCESS ? released : status;
    }

    /* The first target of `relationship` on every prim of the stage that authors it, read
     * through the stage-side filter and charged before it is kept. The caller uses it only
     * when the count is within its bound. */
    ovspatialquery_api_status_t boundPrims(ovx_token_t relationship, std::unordered_map<uint64_t, uint64_t>& out)
    {
        ovstage_query_handle_t      query = OVSTAGE_INVALID_QUERY_HANDLE;
        const HandleGuard           guard{*this, query};
        uint64_t                    count  = 0u;
        ovspatialquery_api_status_t status = queryBound(relationship, query, count);
        if (status == OVSPATIALQUERY_API_SUCCESS && count != 0u)
        {
            status = charge(count * (2u * sizeof(uint64_t) + kHashEntryBytes));
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                out.reserve(static_cast<size_t>(count));
                status = readFirstTargets(query, relationship, out);
            }
        }
        const ovspatialquery_api_status_t released = releaseQuery(mStage, query);
        return status == OVSPATIALQUERY_API_SUCCESS ? released : status;
    }

    /* The first target of `relationship` on each of `paths` that authors it, through a
     * query over exactly those paths; charged for every path before it is read. */
    ovspatialquery_api_status_t readPaths(const std::vector<ovx_primpath_t>& paths, ovx_token_t relationship,
                                          std::unordered_map<uint64_t, uint64_t>& out)
    {
        ovspatialquery_api_status_t status = charge(paths.size() * (2u * sizeof(uint64_t) + kHashEntryBytes));
        ovstage_query_handle_t      query  = OVSTAGE_INVALID_QUERY_HANDLE;
        ovx_primpath_list_t         list   = OVX_INVALID_PRIMPATH_LIST;
        const HandleGuard           guard{*this, query, &list};
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = queryPaths(paths, query, list);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readFirstTargets(query, relationship, out);
        const ovspatialquery_api_status_t released = releaseQuery(mStage, query);
        releasePathList(list);
        return status == OVSPATIALQUERY_API_SUCCESS ? released : status;
    }

    static uint64_t firstBinding(const std::vector<ovx_primpath_t>&            chain,
                                 const std::unordered_map<uint64_t, uint64_t>& bindings)
    {
        for (ovx_primpath_t path : chain)
        {
            const auto found = bindings.find(handleValue(path));
            if (found != bindings.end())
                return found->second;
        }
        return 0u;
    }

    /* Charges `bytes` of resolution scratch to the host budget before it is allocated; the
     * charge lasts until the resolver is gone. */
    ovspatialquery_api_status_t charge(uint64_t bytes, std::vector<std::shared_ptr<void>>* into = nullptr)
    {
        const ConvexHullPreparationBudget& budget = mFilter.preparation.convexHullBudget;
        if (bytes == 0u || budget.acquire == nullptr)
            return OVSPATIALQUERY_API_SUCCESS;
        std::shared_ptr<void>              owner;
        const ConvexHullPreparationOutcome outcome = budget.acquire(budget.state, bytes, owner);
        if (outcome != ConvexHullPreparationOutcome::success)
            return convexHullPreparationStatus(outcome);
        (into != nullptr ? *into : mCharges).push_back(std::move(owner));
        return OVSPATIALQUERY_API_SUCCESS;
    }

    /* Every path's text, one after another in `outText`, path `i` ending at `outEnds[i]`. A
     * counting pass sizes the text, so that it is charged before it exists. */
    ovspatialquery_api_status_t pathTexts(const std::vector<ovx_primpath_t>& paths, std::string& outText,
                                          std::vector<size_t>& outEnds)
    {
        std::vector<ovx_string_t>   strings;
        ovspatialquery_api_status_t stringStatus = OVSPATIALQUERY_API_SUCCESS;
        const auto                  visit        = [&](bool write)
        {
            return [&, write](size_t, const ovx_token_t* elements, size_t count)
            {
                if (stringStatus != OVSPATIALQUERY_API_SUCCESS)
                    return;
                strings.resize(count);
                if (count != 0u)
                    stringStatus =
                        mapDictionaryResult(mDictionary, mDictionary->vtable->get_strings_from_tokens(
                                                             mDictionary->context, elements, count, strings.data()));
                size_t length = 0u;
                for (size_t i = 0u; stringStatus == OVSPATIALQUERY_API_SUCCESS && i < count; ++i)
                {
                    if (write)
                        outText.append("/").append(strings[i].ptr, strings[i].length);
                    length += 1u + strings[i].length;
                }
                if (write)
                    outEnds.push_back(outText.size());
                else
                    outEnds.push_back(length);
            };
        };
        ovspatialquery_api_status_t status = charge(paths.size() * sizeof(size_t));
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            outEnds.reserve(paths.size());
            status = forEachElements(paths, visit(false));
        }
        status            = status == OVSPATIALQUERY_API_SUCCESS ? stringStatus : status;
        uint64_t textSize = 0u;
        for (size_t length : outEnds)
            textSize += length;
        outEnds.clear();
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = charge(textSize);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        outText.reserve(static_cast<size_t>(textSize));
        status = forEachElements(paths, visit(true));
        return status == OVSPATIALQUERY_API_SUCCESS ? stringStatus : status;
    }

    /* The GeomSubset prims whose parent is one of `meshes` (object id to object index), with
     * that mesh. The stage-wide query reads only each subset's prim type, so a subset of
     * another prim costs its path id and path elements, charged before they are kept, and
     * nothing of its payload. A subset's parent comes from its path elements through the path
     * dictionary, as an object's ancestors do, not from `usd-parent`, which not every
     * populator maintains (ADR-0066). */
    ovspatialquery_api_status_t subsetsOfMeshes(const std::unordered_map<uint64_t, size_t>& meshes,
                                                std::vector<ovx_primpath_t>& outPaths, std::vector<size_t>& outMeshes)
    {
        ovx_string_t        subsetType{"GeomSubset", sizeof("GeomSubset") - 1u};
        ovstage_predicate_t predicate{};
        predicate.attribute.token = mFilter.usdPrimType;
        predicate.op              = OVSTAGE_FILTER_OP_IN;
        predicate.values          = &subsetType;
        predicate.value_count     = 1u;
        const ovstage_filter_t      subsetFilter{&predicate, 1u};
        ovstage_query_handle_t      query = OVSTAGE_INVALID_QUERY_HANDLE;
        const HandleGuard           guard{*this, query};
        ovspatialquery_api_status_t status = awaitHandle(
            mStage, stageVtable(mStage)->query(stageContext(mStage), &subsetFilter, &mFilter.usdPrimType, 1u, &query),
            query, OVSTAGE_INVALID_QUERY_HANDLE);
        uint64_t subsetCount = 0u;
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            ovstage_query_result_t result{};
            status = mapStageStatus(stageVtable(mStage)->fetch_query_result(stageContext(mStage), query,
                                                                            OVSTAGE_TIMEOUT_INFINITE, &result));
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                subsetCount = result.total_prim_count;
                status      = mapStageStatus(stageVtable(mStage)->release_query_result(stageContext(mStage), &result));
            }
        }
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = charge(subsetCount * sizeof(ovx_primpath_t));
        std::vector<ovx_primpath_t> paths;
        bool                        overflow = false;
        if (status == OVSPATIALQUERY_API_SUCCESS && subsetCount != 0u)
        {
            paths.reserve(static_cast<size_t>(subsetCount));
            status = visitAttributePrims(mStage, mDictionary, query, mFilter.usdPrimType, mTargetOrdinal,
                                         [&](ovx_primpath_t path)
                                         {
                                             if (paths.size() < paths.capacity())
                                                 paths.push_back(path);
                                             else
                                                 overflow = true;
                                         });
        }
        const ovspatialquery_api_status_t released = releaseQuery(mStage, query);
        status                                     = status == OVSPATIALQUERY_API_SUCCESS ? released : status;
        if (status == OVSPATIALQUERY_API_SUCCESS && overflow)
            status = OVSPATIALQUERY_API_INTERNAL; // more rows than the stage's own count
        if (status != OVSPATIALQUERY_API_SUCCESS || paths.empty())
            return status;

        /* Each subset's parent: its path's elements but the last. A counting pass sizes the
         * parent elements, so that they are charged before they exist; a subset directly
         * under "/" has no mesh parent. */
        uint64_t parentElementCount = 0u, parented = 0u;
        status = forEachElements(paths,
                                 [&](size_t, const ovx_token_t*, size_t count)
                                 {
                                     if (count > 1u)
                                     {
                                         parentElementCount += count - 1u;
                                         ++parented;
                                     }
                                 });
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = charge(parentElementCount * sizeof(ovx_token_t) +
                            parented * (2u * sizeof(size_t) + sizeof(ovx_primpath_t)));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<ovx_token_t> parentTokens;
        std::vector<size_t>      parentCounts;
        std::vector<size_t>      children;
        parentTokens.reserve(static_cast<size_t>(parentElementCount));
        parentCounts.reserve(static_cast<size_t>(parented));
        children.reserve(static_cast<size_t>(parented));
        status = forEachElements(paths,
                                 [&](size_t index, const ovx_token_t* elements, size_t count)
                                 {
                                     if (count > 1u)
                                     {
                                         parentTokens.insert(parentTokens.end(), elements, elements + count - 1u);
                                         parentCounts.push_back(count - 1u);
                                         children.push_back(index);
                                     }
                                 });
        if (status != OVSPATIALQUERY_API_SUCCESS || children.empty())
            return status;
        std::vector<ovx_primpath_t> parents(children.size(), OVX_INVALID_PRIMPATH);
        status = mapDictionaryResult(mDictionary, mDictionary->vtable->create_paths_from_tokens(
                                                      mDictionary->context, parentTokens.data(), parentCounts.data(),
                                                      parentCounts.size(), parents.data()));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        size_t kept = 0u;
        for (ovx_primpath_t parent : parents)
            kept += meshes.count(handleValue(parent));
        status = charge(kept * (sizeof(ovx_primpath_t) + sizeof(size_t)));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        outPaths.reserve(kept);
        outMeshes.reserve(kept);
        for (size_t i = 0u; i < parents.size(); ++i)
        {
            const auto mesh = meshes.find(handleValue(parents[i]));
            if (mesh != meshes.end())
            {
                outPaths.push_back(paths[children[i]]);
                outMeshes.push_back(mesh->second);
            }
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }

    /* Each mesh object's per-source-face table from its face GeomSubset children. A subset
     * resolves as its own prim would: its physics binding, else the mesh's physics binding
     * (which every face already reports), else its all-purpose binding. */
    ovspatialquery_api_status_t resolveFaceMaterials(std::vector<PopulationObject>& objects,
                                                     const std::vector<uint8_t>&    physicsBound)
    {
        size_t meshCount = 0u;
        for (const PopulationObject& object : objects)
            meshCount += object.triangleGeometry != nullptr ? 1u : 0u;
        if (meshCount == 0u)
            return OVSPATIALQUERY_API_SUCCESS;
        /* Per mesh: an entry in the mesh, face-count, table-size and table maps. */
        ovspatialquery_api_status_t status = charge(meshCount * (4u * kHashEntryBytes + sizeof(std::vector<uint64_t>)));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::unordered_map<uint64_t, size_t> meshes;
        meshes.reserve(meshCount);
        for (size_t i = 0u; i < objects.size(); ++i)
            if (objects[i].triangleGeometry != nullptr)
                meshes.emplace(objects[i].id.value, i);
        std::vector<ovx_primpath_t> subsetPaths;
        std::vector<size_t>         subsetMeshes;
        status = subsetsOfMeshes(meshes, subsetPaths, subsetMeshes);
        if (status != OVSPATIALQUERY_API_SUCCESS || subsetPaths.empty())
            return status;

        /* Only the meshes' own subsets are read, through a query over exactly their paths:
         * first their element type, family and bindings, then the indices of those that can
         * give a face a material, so a large selection or partition subset costs nothing.
         * Per subset: its record and one binding of each kind; its indices as they arrive. */
        status = charge(subsetPaths.size() * (sizeof(Subset) + 3u * kHashEntryBytes));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::unordered_map<uint64_t, Subset> subsets;
        subsets.reserve(subsetPaths.size());
        for (size_t i = 0u; i < subsetPaths.size(); ++i)
        {
            Subset& subset = subsets[handleValue(subsetPaths[i])];
            subset.path    = handleValue(subsetPaths[i]);
            subset.mesh    = subsetMeshes[i];
        }
        std::unordered_map<uint64_t, uint64_t> physics;
        std::unordered_map<uint64_t, uint64_t> general;
        ovstage_query_handle_t                 query = OVSTAGE_INVALID_QUERY_HANDLE;
        ovx_primpath_list_t                    list  = OVX_INVALID_PRIMPATH_LIST;
        const HandleGuard                      guard{*this, query, &list};
        status               = queryPaths(subsetPaths, query, list);
        const auto readToken = [&](ovx_token_t attribute, uint64_t Subset::*member)
        {
            return readOptional(query, attribute, OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u, true,
                                [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                {
                                    const auto subset = subsets.find(id);
                                    if (subset != subsets.end() && byteCount >= sizeof(uint64_t))
                                        std::memcpy(&(subset->second.*member), bytes, sizeof(uint64_t));
                                    return OVSPATIALQUERY_API_SUCCESS;
                                });
        };
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readToken(mElementType, &Subset::elementType);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readToken(mFamilyName, &Subset::familyName);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readFirstTargets(query, mPhysicsBinding, physics);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readFirstTargets(query, mBinding, general);
        ovspatialquery_api_status_t released = releaseQuery(mStage, query);
        releasePathList(list);
        status = status == OVSPATIALQUERY_API_SUCCESS ? released : status;
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;

        /* The subsets that can give a face a material: a face subset of the materialBind
         * family (each checked when the populator published it) with its own physics
         * binding, or its all-purpose one on a mesh without a physics binding. */
        const auto contributes = [&](const Subset& subset)
        {
            return (subset.elementType == 0u || subset.elementType == handleValue(mFace)) &&
                   (subset.familyName == 0u || subset.familyName == handleValue(mMaterialBind)) &&
                   (physics.count(subset.path) != 0u ||
                    (general.count(subset.path) != 0u && physicsBound[subset.mesh] == 0u));
        };
        status = charge(subsets.size() * sizeof(ovx_primpath_t));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<ovx_primpath_t> contributing;
        contributing.reserve(subsets.size());
        for (const auto& entry : subsets)
            if (contributes(entry.second))
                contributing.push_back(makeHandle<ovx_primpath_t>(entry.second.path));
        if (contributing.empty())
            return OVSPATIALQUERY_API_SUCCESS;
        query  = OVSTAGE_INVALID_QUERY_HANDLE;
        list   = OVX_INVALID_PRIMPATH_LIST;
        status = queryPaths(contributing, query, list);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = readOptional(query, mIndices, OVSTAGE_SEMANTIC_NONE, kDLInt, 32u, false,
                                  [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                  {
                                      const auto subset = subsets.find(id);
                                      if (subset == subsets.end())
                                          return OVSPATIALQUERY_API_SUCCESS;
                                      const size_t                      count   = byteCount / sizeof(int32_t);
                                      const ovspatialquery_api_status_t charged = charge(count * sizeof(int32_t));
                                      if (charged != OVSPATIALQUERY_API_SUCCESS)
                                          return charged;
                                      subset->second.hasIndices = true;
                                      subset->second.indices.assign(count, 0);
                                      if (count != 0u)
                                          std::memcpy(subset->second.indices.data(), bytes, count * sizeof(int32_t));
                                      return OVSPATIALQUERY_API_SUCCESS;
                                  });
        released = releaseQuery(mStage, query);
        releasePathList(list);
        status = status == OVSPATIALQUERY_API_SUCCESS ? released : status;
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;

        /* Qualifying subsets with a material, in path order, with their parent mesh. */
        size_t     qualifying = 0u;
        const auto qualifies  = [&](const Subset& subset) { return subset.hasIndices && contributes(subset); };
        for (const auto& entry : subsets)
            qualifying += qualifies(entry.second) ? 1u : 0u;
        using Ordered = std::pair<std::string_view, std::pair<uint64_t, size_t>>; // path text, (subset, mesh)
        status        = charge(qualifying * (sizeof(ovx_primpath_t) + sizeof(Ordered)));
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<ovx_primpath_t> paths;
        paths.reserve(qualifying);
        for (const auto& entry : subsets)
            if (qualifies(entry.second))
                paths.push_back(makeHandle<ovx_primpath_t>(entry.second.path));
        std::string         subsetText;
        std::vector<size_t> subsetEnds;
        status = pathTexts(paths, subsetText, subsetEnds);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::vector<Ordered> ordered;
        ordered.reserve(paths.size());
        for (size_t i = 0u; i < paths.size(); ++i)
        {
            const size_t           begin = i == 0u ? 0u : subsetEnds[i - 1u];
            const std::string_view text(subsetText.data() + begin, subsetEnds[i] - begin);
            ordered.push_back({text, {handleValue(paths[i]), subsets.at(handleValue(paths[i])).mesh}});
        }
        std::sort(ordered.begin(), ordered.end());
        /* Each mesh's source face count, its largest prepared source face plus one: an
         * index at or past it can never match a hit, so it is ignored, as an invalid
         * subset index is, and cannot size the table. */
        std::unordered_map<size_t, uint64_t> faceCounts;
        const auto                           faceCount = [&](size_t mesh)
        {
            const auto found = faceCounts.find(mesh);
            if (found != faceCounts.end())
                return found->second;
            TriangleGeometryDescriptionView description{};
            uint64_t                        count = 0u;
            if (exportTriangleGeometryDescription(*objects[mesh].triangleGeometry, description) &&
                description.sourceFaces != nullptr)
                for (uint64_t triangle = 0u; triangle < description.triangleCount; ++triangle)
                    count = std::max(count, description.sourceFaces[triangle] + 1u);
            faceCounts.emplace(mesh, count);
            return count;
        };
        const auto materialOf = [&](const Ordered& entry)
        {
            const uint64_t subset = entry.second.first;
            const auto     found  = physics.find(subset);
            if (found != physics.end())
                return found->second;
            const auto all = general.find(subset);
            return physicsBound[entry.second.second] == 0u && all != general.end() ? all->second : uint64_t{0u};
        };
        /* Every table's size first, so that the tables are charged before they exist. */
        std::unordered_map<size_t, uint64_t> tableSizes;
        uint64_t                             tableBytes = 0u;
        for (const Ordered& entry : ordered)
        {
            if (materialOf(entry) == 0u)
                continue;
            const uint64_t faces = faceCount(entry.second.second);
            uint64_t&      size  = tableSizes[entry.second.second];
            for (int32_t face : subsets.at(entry.second.first).indices)
                if (face >= 0 && static_cast<uint64_t>(face) < faces)
                    size = std::max(size, static_cast<uint64_t>(face) + 1u);
        }
        for (const auto& entry : tableSizes)
            tableBytes += entry.second * sizeof(uint64_t);
        /* The tables outlive the resolver in the published objects, so their charge does
         * too: takeRetainedCharges hands it to the snapshot. */
        status = charge(tableBytes, &mRetainedCharges);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        std::unordered_map<size_t, std::vector<uint64_t>> tables;
        for (const auto& entry : tableSizes)
            if (entry.second != 0u)
                tables[entry.first].assign(static_cast<size_t>(entry.second), 0u);
        for (const Ordered& entry : ordered)
        {
            const uint64_t material = materialOf(entry);
            const auto     table    = tables.find(entry.second.second);
            if (material == 0u || table == tables.end())
                continue;
            for (int32_t face : subsets.at(entry.second.first).indices)
                if (face >= 0 && static_cast<uint64_t>(face) < table->second.size())
                    table->second[static_cast<size_t>(face)] = material;
        }
        for (auto& entry : tables)
            objects[entry.first].faceMaterialIds =
                std::make_shared<const std::vector<uint64_t>>(std::move(entry.second));
        return OVSPATIALQUERY_API_SUCCESS;
    }

    ovstage_instance_t*                mStage;
    path_dictionary_instance_t*        mDictionary;
    const OwnedFilter&                 mFilter;
    ovspatialquery_ordinal_t           mTargetOrdinal;
    ovx_token_t                        mPhysicsBinding = OVX_INVALID_TOKEN;
    ovx_token_t                        mBinding        = OVX_INVALID_TOKEN;
    ovx_token_t                        mElementType    = OVX_INVALID_TOKEN;
    ovx_token_t                        mFamilyName     = OVX_INVALID_TOKEN;
    ovx_token_t                        mIndices        = OVX_INVALID_TOKEN;
    ovx_token_t                        mFace           = OVX_INVALID_TOKEN;
    ovx_token_t                        mMaterialBind   = OVX_INVALID_TOKEN;
    std::vector<std::shared_ptr<void>> mCharges;
    std::vector<std::shared_ptr<void>> mRetainedCharges;
    bool                               mBatchCharged = false;
};

ovspatialquery_api_status_t acquireTypedRows(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                             const std::shared_ptr<OwnedFilter>&   filter,
                                             const std::vector<SourceViewRequest>& viewRequests,
                                             ovspatialquery_ordinal_t targetOrdinal, const SourceAcquisitionBase* base,
                                             CopiedSourceSnapshot& outSnapshot, PrimRefusal& refusal,
                                             std::unordered_map<uint64_t, int32_t>& excluded)
{
    ovstage_query_handle_t queryHandle = OVSTAGE_INVALID_QUERY_HANDLE;
    try
    {
        /* Mesh prims that dropped triangles without area (ADR-0073), named on success. */
        std::vector<std::pair<uint64_t, std::shared_ptr<const PreparedTriangleMesh>>> droppingMeshes;
        /* One filter evaluation serves the identities, the bounds and every attribute read below. */
        std::vector<PopulationObject> matched;
        uint64_t                      matchedCount = 0u;
        std::vector<uint8_t>          hasBounds;
        ovspatialquery_api_status_t   status =
            acquireFilterSnapshot(stage, dictionary, *filter, targetOrdinal, matched, matchedCount, &hasBounds,
                                  &queryHandle, &refusal, &outSnapshot.droppedPointPrototypes);
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        /* A prim outside the population's scope is not selected, as if the filter had
         * rejected it (ADR-0055): it leaves before any row is read, and is no exclusion. */
        std::vector<uint8_t> inScope;
        if (filter->scope != nullptr)
        {
            std::vector<uint64_t> ids(matched.size());
            for (size_t i = 0u; i < matched.size(); ++i)
                ids[i] = matched[i].id.value;
            status = filter->scope->contains(dictionary, ids.data(), ids.size(), inScope);
            if (status == OVSPATIALQUERY_API_SUCCESS)
                status = narrowToScope(stage, dictionary, ids, inScope, queryHandle);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                (void)releaseQuery(stage, queryHandle);
                return status;
            }
        }

        size_t retainedCount = 0u;
        bool   missingBounds = false;
        for (size_t i = 0u; i < matched.size(); ++i)
        {
            if ((!inScope.empty() && inScope[i] == 0u) || excluded.count(matched[i].id.value))
                continue;
            missingBounds = missingBounds || hasBounds[i] == 0u;
            if (retainedCount != i)
                matched[retainedCount] = std::move(matched[i]);
            hasBounds[retainedCount++] = hasBounds[i];
        }
        matched.resize(retainedCount);
        hasBounds.resize(retainedCount);
        struct Row
        {
            PopulationObject                          object{};
            OvstageSplit                              split;
            std::array<double, 3>                     authoredCapsuleAxis{};
            std::array<double, 16u>                   worldFromLocal{};
            ovx_token_t                               primType          = OVX_INVALID_TOKEN;
            ovx_token_t                               capsuleAxis       = OVX_INVALID_TOKEN;
            double                                    radius            = 0.0;
            double                                    size              = 0.0;
            double                                    height            = 0.0;
            ExactGeometryFamily                       family            = ExactGeometryFamily::sphere;
            bool                                      hasPrimType       = false;
            bool                                      hasCapsuleAxis    = false;
            bool                                      hasRadius         = false;
            bool                                      hasSize           = false;
            bool                                      hasHeight         = false;
            bool                                      hasWorldFromLocal = false;
            std::unique_ptr<GeometryDefinitionValue>  definition;
            std::shared_ptr<const PreparedConvexHull> preparedConvexHull;
            /* Set when the mesh content is unchanged since the base generation
             * and its prepared mesh and reference are carried over. */
            GeometryReference reusedReference{};
            bool              hasBounds = false;
            /* An instance proxy (ADR-0050) takes its definition from the prim its
             * `_protoPath` names; every other row from itself. */
            bool        proxy            = false;
            bool        hasProtoPrimType = false;
            bool        hasProtoPath     = false;
            ovx_token_t protoPrimType    = OVX_INVALID_TOKEN;
            uint64_t    protoPath        = 0u;
            uint64_t    definitionId     = 0u;
        };
        std::vector<Row>                     rows(matched.size());
        std::unordered_map<uint64_t, size_t> rowById;
        rowById.reserve(rows.size());
        for (size_t rowIndex = 0u; rowIndex < rows.size(); ++rowIndex)
        {
            rows[rowIndex].object       = matched[rowIndex];
            rows[rowIndex].hasBounds    = hasBounds[rowIndex] != 0u;
            rows[rowIndex].definitionId = rows[rowIndex].object.id.value;
            rowById.emplace(rows[rowIndex].object.id.value, rowIndex);
        }
        const auto findRow = [&](uint64_t id) -> Row*
        {
            const std::unordered_map<uint64_t, size_t>::const_iterator found = rowById.find(id);
            return found == rowById.end() ? nullptr : &rows[found->second];
        };

        // One value per matched row; a second value or one of another size is INVALID_STATE.
        const auto readColumn = [&](ovx_token_t attribute, ovstage_attribute_semantic_t semantic, uint8_t code,
                                    bool allowNoneSemantic, bool optional, auto valueMember, bool Row::*presenceMember)
        {
            return readTriangleAttribute(stage, queryHandle, attribute, targetOrdinal, dictionary, semantic, code, 64u,
                                         1u, allowNoneSemantic, optional,
                                         [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                         {
                                             Row* row = findRow(id);
                                             if (row == nullptr)
                                             {
                                                 return OVSPATIALQUERY_API_SUCCESS;
                                             }
                                             if (row->*presenceMember || byteCount != sizeof(row->*valueMember))
                                             {
                                                 return OVSPATIALQUERY_API_INVALID_STATE;
                                             }
                                             std::memcpy(&(row->*valueMember), bytes, sizeof(row->*valueMember));
                                             row->*presenceMember = true;
                                             return OVSPATIALQUERY_API_SUCCESS;
                                         });
        };
        status =
            readColumn(filter->usdPrimType, OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt,
                       true /* The pinned built-in reports SEMANTIC_NONE. */, false, &Row::primType, &Row::hasPrimType);
        if (status == OVSPATIALQUERY_API_SUCCESS && missingBounds)
        {
            size_t kept = 0u;
            for (size_t i = 0u; i < rows.size(); ++i)
            {
                Row& row = rows[i];
                if (!row.hasBounds)
                {
                    ExactGeometryFamily family    = ExactGeometryFamily::user;
                    const bool          supported = row.hasPrimType && (filter->familyOf(row.primType, family) ||
                                                               filter->proxyFamilyOf(row.primType, family));
                    excluded.emplace(row.object.id.value, supported
                                                              ? OVSPATIALQUERY_OVSTAGE_EXCLUSION_NO_SPATIAL_SOURCE
                                                              : OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE);
                    continue;
                }
                if (kept != i)
                    rows[kept] = std::move(row);
                ++kept;
            }
            rows.resize(kept);
            rowById.clear();
            for (size_t i = 0u; i < rows.size(); ++i)
                rowById.emplace(rows[i].object.id.value, i);
        }
        /* An instance proxy's family comes from its `_protoPrimType` and its definition from
         * the prim its `_protoPath` names (ADR-0050); ovstage flattens nested instances, so
         * that prim is never itself a proxy. Both columns are read only when a proxy matched. */
        /* One pass finds the proxies and which definition columns each query must read:
         * radius for spheres and capsules, size for cubes, height and axis for capsules,
         * indexed by whether the rows are proxies. */
        bool                                 anyProxy = false;
        std::array<std::array<bool, 3u>, 2u> columns{};
        for (Row& row : rows)
        {
            ExactGeometryFamily family = ExactGeometryFamily::user;
            if (!row.hasPrimType)
            {
                continue;
            }
            if (!filter->familyOf(row.primType, family))
            {
                row.proxy = filter->proxyFamilyOf(row.primType, family);
                if (!row.proxy)
                {
                    continue;
                }
            }
            if (row.proxy)
            {
                row.family = family;
                anyProxy   = true;
            }
            std::array<bool, 3u>& needed = columns[row.proxy ? 1u : 0u];
            needed[0] = needed[0] || family == ExactGeometryFamily::sphere || family == ExactGeometryFamily::capsule;
            needed[1] = needed[1] || family == ExactGeometryFamily::box;
            needed[2] = needed[2] || family == ExactGeometryFamily::capsule;
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && anyProxy)
        {
            status = readColumn(filter->protoPrimType, OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, true, true,
                                &Row::protoPrimType, &Row::hasProtoPrimType);
        }
        if (status == OVSPATIALQUERY_API_SUCCESS && anyProxy)
        {
            status = readColumn(filter->protoPath, OVSTAGE_SEMANTIC_RELATIONSHIP_PATH_ID, kDLUInt, true, true,
                                &Row::protoPath, &Row::hasProtoPath);
        }
        /* The proxies naming each prototype prim; every other row defines itself and is
         * found through rowById. */
        std::unordered_map<uint64_t, std::vector<size_t>> rowsByDefinition;
        std::vector<ovx_primpath_t>                       prototypePaths;
        for (size_t rowIndex = 0u; status == OVSPATIALQUERY_API_SUCCESS && rowIndex < rows.size(); ++rowIndex)
        {
            Row& row = rows[rowIndex];
            if (row.proxy)
            {
                ExactGeometryFamily prototypeFamily = ExactGeometryFamily::user;
                if (!row.hasProtoPath || !row.hasProtoPrimType || row.protoPath == 0u)
                {
                    refusal.note(row.object.id.value, row.primType,
                                 "ovstage gave this instance proxy no _protoPath, so its prototype is unknown");
                    status = OVSPATIALQUERY_API_NOT_SUPPORTED;
                    break;
                }
                if (!filter->familyOf(row.protoPrimType, prototypeFamily) || prototypeFamily != row.family)
                {
                    refusal.note(row.object.id.value, row.primType,
                                 "its _protoPrimType is not the type its instance type names");
                    status = OVSPATIALQUERY_API_INVALID_STATE;
                    break;
                }
                row.definitionId             = row.protoPath;
                std::vector<size_t>& sharing = rowsByDefinition[row.definitionId];
                if (sharing.empty())
                {
                    prototypePaths.push_back(makeHandle<ovx_primpath_t>(row.protoPath));
                }
                sharing.push_back(rowIndex);
            }
        }
        PrototypeQuery prototypes(stage, dictionary);
        if (status == OVSPATIALQUERY_API_SUCCESS && !prototypePaths.empty())
        {
            status = prototypes.open(prototypePaths);
        }
        /* Each prototype prim must exist with the type its proxies' `_protoPrimType` names;
         * a prim that is itself a proxy would need a second hop, which ovstage never asks for. */
        if (status == OVSPATIALQUERY_API_SUCCESS && !prototypePaths.empty())
        {
            std::unordered_map<uint64_t, ovx_token_t> prototypeTypes;
            status = readTriangleAttribute(stage, prototypes.handle(), filter->usdPrimType, targetOrdinal, dictionary,
                                           OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt, 64u, 1u, true, false,
                                           [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                                           {
                                               ovx_token_t type = OVX_INVALID_TOKEN;
                                               if (byteCount != sizeof(type))
                                               {
                                                   return OVSPATIALQUERY_API_INVALID_STATE;
                                               }
                                               std::memcpy(&type, bytes, sizeof(type));
                                               prototypeTypes[id] = type;
                                               return OVSPATIALQUERY_API_SUCCESS;
                                           });
            for (size_t index = 0u; status == OVSPATIALQUERY_API_SUCCESS && index < prototypePaths.size(); ++index)
            {
                const Row& proxy = rows[rowsByDefinition.at(handleValue(prototypePaths[index])).front()];
                const std::unordered_map<uint64_t, ovx_token_t>::const_iterator type =
                    prototypeTypes.find(handleValue(prototypePaths[index]));
                ExactGeometryFamily ignored = ExactGeometryFamily::user;
                if (type == prototypeTypes.end() || filter->proxyFamilyOf(type->second, ignored))
                {
                    refusal.note(proxy.object.id.value, proxy.primType,
                                 "its _protoPath names no prototype prim with geometry ovstage populations publish");
                    status = OVSPATIALQUERY_API_NOT_SUPPORTED;
                }
                else if (type->second != proxy.protoPrimType)
                {
                    refusal.note(proxy.object.id.value, proxy.primType,
                                 "the prim its _protoPath names is not of its _protoPrimType");
                    status = OVSPATIALQUERY_API_INVALID_STATE;
                }
            }
        }
        /* A definition column is read through the filter's query for the rows that define
         * themselves and through the prototype query for proxies; the filter's query may still
         * match a prototype prim, whose dropped row each pass skips. One value per definition
         * prim, a second one or one of another size is INVALID_STATE. */
        const auto readDefinitionColumn = [&](bool proxies, ovstage_query_handle_t query, ovx_token_t attribute,
                                              ovstage_attribute_semantic_t semantic, uint8_t code, auto valueMember,
                                              bool Row::*presenceMember)
        {
            return readTriangleAttribute(
                stage, query, attribute, targetOrdinal, dictionary, semantic, code, 64u, 1u, false, true,
                [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                {
                    const auto assign = [&](Row& row)
                    {
                        if (row.*presenceMember || byteCount != sizeof(row.*valueMember))
                        {
                            return OVSPATIALQUERY_API_INVALID_STATE;
                        }
                        std::memcpy(&(row.*valueMember), bytes, sizeof(row.*valueMember));
                        row.*presenceMember = true;
                        return OVSPATIALQUERY_API_SUCCESS;
                    };
                    if (!proxies)
                    {
                        Row* row = findRow(id);
                        return row == nullptr || row->proxy ? OVSPATIALQUERY_API_SUCCESS : assign(*row);
                    }
                    const auto found = rowsByDefinition.find(id);
                    if (found == rowsByDefinition.end())
                    {
                        return OVSPATIALQUERY_API_SUCCESS;
                    }
                    for (size_t rowIndex : found->second)
                    {
                        const ovspatialquery_api_status_t assignmentStatus = assign(rows[rowIndex]);
                        if (assignmentStatus != OVSPATIALQUERY_API_SUCCESS)
                        {
                            return assignmentStatus;
                        }
                    }
                    return OVSPATIALQUERY_API_SUCCESS;
                });
        };
        /* Through a `usd-path IN` population's path-list query Ovstage charges a read about
         * 0.1 us per listed prim, whether or not the prim carries the attribute (1 ms per
         * column at 10,000 prims), so read only the columns the matched prim types use. */
        for (const bool proxies : {false, true})
        {
            const std::array<bool, 3u>&  needed             = columns[proxies ? 1u : 0u];
            const bool                   hasSphereOrCapsule = needed[0];
            const bool                   hasCube            = needed[1];
            const bool                   hasCapsule         = needed[2];
            const ovstage_query_handle_t query              = proxies ? prototypes.handle() : queryHandle;
            if (status == OVSPATIALQUERY_API_SUCCESS && hasSphereOrCapsule)
            {
                status = readDefinitionColumn(proxies, query, filter->radius, OVSTAGE_SEMANTIC_NONE, kDLFloat,
                                              &Row::radius, &Row::hasRadius);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && hasCube)
            {
                status = readDefinitionColumn(proxies, query, filter->size, OVSTAGE_SEMANTIC_NONE, kDLFloat, &Row::size,
                                              &Row::hasSize);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && hasCapsule)
            {
                status = readDefinitionColumn(proxies, query, filter->height, OVSTAGE_SEMANTIC_NONE, kDLFloat,
                                              &Row::height, &Row::hasHeight);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && hasCapsule)
            {
                status = readDefinitionColumn(proxies, query, filter->axis, OVSTAGE_SEMANTIC_TOKEN_ID, kDLUInt,
                                              &Row::capsuleAxis, &Row::hasCapsuleAxis);
            }
        }
        const auto readMatrix = [&](ovx_token_t attribute, bool allowNoneSemantic)
        {
            return readTriangleAttribute(
                stage, queryHandle, attribute, targetOrdinal, dictionary, OVSTAGE_SEMANTIC_MATRIX, kDLFloat, 64u, 16u,
                allowNoneSemantic, true,
                [&](uint64_t id, const uint8_t* bytes, size_t byteCount)
                {
                    Row* row = findRow(id);
                    if (row == nullptr || row->hasWorldFromLocal || byteCount == 0u)
                    {
                        return OVSPATIALQUERY_API_SUCCESS;
                    }
                    if (byteCount != sizeof(double) * 16u)
                    {
                        refusal.note(id, row->primType, "its world matrix is not one 4x4 float64 matrix");
                        return OVSPATIALQUERY_API_NOT_SUPPORTED;
                    }
                    double source[16]{};
                    std::memcpy(source, bytes, sizeof(source));
                    for (uint32_t matrixRow = 0u; matrixRow < 4u; ++matrixRow)
                    {
                        for (uint32_t column = 0u; column < 4u; ++column)
                        {
                            row->worldFromLocal[matrixRow * 4u + column] = source[column * 4u + matrixRow];
                        }
                    }
                    row->hasWorldFromLocal = true;
                    return OVSPATIALQUERY_API_SUCCESS;
                });
        };
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = readMatrix(filter->fabricWorldMatrix, true /* The pinned built-in reports SEMANTIC_NONE. */);
        }
        /* The authored world matrix only fills rows the Fabric matrix left without one. */
        const bool worldMatrixColumnRead =
            std::any_of(rows.begin(), rows.end(), [](const Row& row) { return !row.hasWorldFromLocal; });
        if (status == OVSPATIALQUERY_API_SUCCESS && worldMatrixColumnRead)
        {
            status = readMatrix(filter->worldMatrix, false);
        }

        std::array<OvstageSplitCache, 2> transformCache;
        size_t                           nextTransform = 0;
        const bool                       alignedPrior = base && base->objects && base->objects->size() == rows.size() &&
                                  std::equal(rows.begin(), rows.end(), base->objects->begin(),
                                             [](const Row& row, const PopulationObject& object)
                                             { return row.object.id.value == object.id.value; });
        std::unordered_map<uint64_t, const PopulationObject*> priorObjects;
        if (!alignedPrior && base && base->objects)
        {
            priorObjects.reserve(base->objects->size());
            for (const auto& object : *base->objects)
                priorObjects.emplace(object.id.value, &object);
        }
        const auto previousObject = [&](const Row& row) -> const PopulationObject*
        {
            if (alignedPrior)
            {
                const auto& previous = (*base->objects)[static_cast<size_t>(&row - rows.data())];
                return previous.id.value == row.object.id.value ? &previous : nullptr;
            }
            const auto previous = priorObjects.find(row.object.id.value);
            return previous == priorObjects.end() ? nullptr : previous->second;
        };
        const auto shapeLess = [](const OvstageShapeTransform* a, const OvstageShapeTransform* b)
        {
            int order = std::memcmp(a->k.data(), b->k.data(), sizeof(a->k));
            if (!order)
                order = std::memcmp(a->inverse.data(), b->inverse.data(), sizeof(a->inverse));
            if (!order)
                order = std::memcmp(a->metric.data(), b->metric.data(), sizeof(a->metric));
            return order ? order < 0 : a->mirrored < b->mirrored;
        };
        using MeshShapeKey = std::pair<uint64_t, const OvstageShapeTransform*>;
        struct MeshShapeHash
        {
            size_t operator()(const MeshShapeKey& key) const noexcept
            {
                return std::hash<uint64_t>{}(key.first) ^ (std::hash<const OvstageShapeTransform*>{}(key.second) << 1u);
            }
        };
        std::map<const OvstageShapeTransform*, std::shared_ptr<const OvstageShapeTransform>, decltype(shapeLess)>
            shapes(shapeLess);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            for (size_t index = 0; index < rows.size(); ++index)
            {
                Row&    row    = rows[index];
                int32_t reason = 0;
                if (!row.hasPrimType || !row.hasWorldFromLocal)
                    reason = 4;
                else if (!(row.proxy ? filter->proxyFamilyOf(row.primType, row.family)
                                     : filter->familyOf(row.primType, row.family)))
                    reason = 2;
                else if (row.family == ExactGeometryFamily::sphere && (!row.hasRadius || !std::isfinite(row.radius) ||
                                                                       row.radius <= 0.0 || !inInputDomain(row.radius)))
                    reason = OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY;
                else if (row.family == ExactGeometryFamily::box &&
                         (!row.hasSize || !std::isfinite(row.size) || row.size <= 0.0 || !inInputDomain(row.size)))
                    reason = OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY;
                else if (row.family == ExactGeometryFamily::capsule &&
                         (!row.hasRadius || !row.hasHeight || !row.hasCapsuleAxis || !std::isfinite(row.radius) ||
                          !std::isfinite(row.height) || row.radius <= 0.0 || row.height <= 0.0 ||
                          !inInputDomain(row.radius) || !inInputDomain(row.height)))
                    reason = OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY;
                else if (row.family == ExactGeometryFamily::capsule &&
                         std::find(filter->principalAxes.begin(), filter->principalAxes.end(), row.capsuleAxis) ==
                             filter->principalAxes.end())
                    reason = OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_SHAPE;
                else if (!ovstage_transform::normalizeAffine(row.worldFromLocal.data(), false))
                    reason = 4;
                else
                {
                    if (row.family == ExactGeometryFamily::capsule)
                        for (unsigned axis = 0; axis < 3; ++axis)
                            if (row.capsuleAxis == filter->principalAxes[axis])
                                for (unsigned world = 0; world < 3; ++world)
                                    row.authoredCapsuleAxis[world] = row.worldFromLocal[world * 4 + axis];
                    const PopulationObject* previous = previousObject(row);
                    if (!splitOvstageTransform(row.worldFromLocal, previous, row.family, row.split, transformCache,
                                               nextTransform))
                        reason = 4;
                    else if ((row.family == ExactGeometryFamily::sphere ||
                              row.family == ExactGeometryFamily::capsule) &&
                             !row.split.uniform)
                        reason = 3;
                    else
                    {
                        if (!row.split.retained && !row.split.uniform)
                            row.split.shape = shapes.try_emplace(row.split.shape.get(), row.split.shape).first->second;
                        row.object.ovstageShape = row.split.shape;

                        if (row.family == ExactGeometryFamily::box && previous && row.split.retained && row.hasSize &&
                            row.size == previous->ovstageCubeSize &&
                            previous->geometryReference.family ==
                                (row.split.diagonal ? ExactGeometryFamily::box : ExactGeometryFamily::convex))
                            row.reusedReference = previous->geometryReference;
                    }
                }
                if (reason)
                    excluded.emplace(row.object.id.value, reason);
            }
            if (!excluded.empty())
            {
                const auto kept = std::remove_if(
                    rows.begin(), rows.end(), [&](const Row& row) { return excluded.count(row.object.id.value) != 0; });
                if (kept != rows.end())
                {
                    rows.erase(kept, rows.end());
                    rowById.clear();
                    for (size_t i = 0; i < rows.size(); ++i)
                        rowById.emplace(rows[i].object.id.value, i);
                }
            }
        }
        std::vector<PopulationObject> meshObjects;
        std::vector<size_t>           meshRows, proxyMeshRows;
        if (status == OVSPATIALQUERY_API_SUCCESS)
            for (size_t i = 0; i < rows.size(); ++i)
            {
                const Row& row = rows[i];
                if (row.family == ExactGeometryFamily::triangleMesh)
                {
                    (row.proxy ? proxyMeshRows : meshRows).push_back(i);
                    if (!row.proxy)
                        meshObjects.push_back(row.object);
                }
            }
        /* Convex detection reads the schema and approximation columns; only Mesh prims can carry them. */
        std::unordered_set<uint64_t> authoredConvexIds;
        if (status == OVSPATIALQUERY_API_SUCCESS && !meshObjects.empty())
        {
            status =
                collectAuthoredConvexIds(stage, queryHandle, *filter, targetOrdinal, dictionary, authoredConvexIds);
            for (size_t meshRow : meshRows)
            {
                if (authoredConvexIds.find(rows[meshRow].object.id.value) != authoredConvexIds.end())
                {
                    rows[meshRow].family = ExactGeometryFamily::convex;
                }
            }
        }
        /* A proxy's convex detection reads its prototype prim's schema and approximation. */
        std::unordered_set<uint64_t> prototypeConvexIds;
        if (status == OVSPATIALQUERY_API_SUCCESS && !proxyMeshRows.empty())
        {
            status = collectAuthoredConvexIds(stage, prototypes.handle(), *filter, targetOrdinal, dictionary,
                                              prototypeConvexIds);
            for (size_t meshRow : proxyMeshRows)
            {
                if (prototypeConvexIds.find(rows[meshRow].definitionId) != prototypeConvexIds.end())
                {
                    rows[meshRow].family = ExactGeometryFamily::convex;
                }
            }
        }

        /* A triangle mesh whose authored content is identical to what the base
         * generation already holds keeps that generation's prepared mesh and geometry
         * reference under its current pose; a mesh whose points alone changed keeps
         * that mesh's topology and takes only its new points; only the remaining
         * meshes are read and prepared in full. The comparison reads the stage, so it
         * holds whichever way the content changed, through the host or through the
         * stage's own population and playback layers. */
        std::vector<PopulationObject>                         readObjects;
        std::vector<size_t>                                   readRows;
        std::vector<size_t>                                   deformedRows;
        std::unordered_map<uint64_t, DeformedMeshPoints>      deformed;
        std::unordered_map<uint64_t, const PopulationObject*> previousById;
        if (status == OVSPATIALQUERY_API_SUCCESS && !meshObjects.empty())
        {
            if (base != nullptr && base->objects != nullptr)
            {
                previousById.reserve(base->objects->size());
                for (const PopulationObject& previous : *base->objects)
                {
                    if (previous.triangleGeometry != nullptr && previous.triangleGeometry->mesh != nullptr &&
                        previous.geometryReference.value != 0u)
                    {
                        previousById.emplace(previous.id.value, &previous);
                    }
                }
            }
            /* The registry retains each base mesh's authored arrays with its definition,
             * and the base generation keeps that definition alive while this runs. */
            std::unordered_map<uint64_t, const MeshDefinitionValue*> definitions;
            for (size_t meshRow : meshRows)
            {
                const Row&                                                                  row = rows[meshRow];
                const std::unordered_map<uint64_t, const PopulationObject*>::const_iterator previous =
                    row.family == ExactGeometryFamily::triangleMesh && row.split.retained
                        ? previousById.find(row.object.id.value)
                        : previousById.end();
                if (previous != previousById.end())
                {
                    definitions.emplace(previous->second->geometryReference.value, nullptr);
                }
            }
            std::unordered_map<uint64_t, const MeshDefinitionValue*> candidates;
            if (!definitions.empty() && base->geometryIdentities != nullptr)
            {
                base->geometryIdentities->findMeshDefinitions(definitions);
                for (size_t meshRow : meshRows)
                {
                    const Row&                                                                  row = rows[meshRow];
                    const std::unordered_map<uint64_t, const PopulationObject*>::const_iterator previous =
                        row.family == ExactGeometryFamily::triangleMesh && row.split.retained
                            ? previousById.find(row.object.id.value)
                            : previousById.end();
                    if (previous != previousById.end())
                    {
                        candidates.emplace(row.object.id.value,
                                           definitions.at(previous->second->geometryReference.value));
                    }
                }
            }
            std::unordered_set<uint64_t> unchanged;
            if (!candidates.empty())
            {
                status = findUnchangedAuthoredMeshes(stage, queryHandle, *filter, targetOrdinal, dictionary, candidates,
                                                     unchanged, &deformed);
            }
            readObjects.reserve(meshObjects.size());
            readRows.reserve(meshRows.size());
            for (size_t meshIndex = 0u; status == OVSPATIALQUERY_API_SUCCESS && meshIndex < meshRows.size();
                 ++meshIndex)
            {
                Row& row = rows[meshRows[meshIndex]];
                const std::unordered_map<uint64_t, const PopulationObject*>::const_iterator previous =
                    unchanged.find(row.object.id.value) != unchanged.end() ? previousById.find(row.object.id.value)
                                                                           : previousById.end();
                if (previous == previousById.end())
                {
                    if (deformed.find(row.object.id.value) != deformed.end())
                    {
                        deformedRows.push_back(meshRows[meshIndex]);
                        continue;
                    }
                    readObjects.push_back(meshObjects[meshIndex]);
                    readRows.push_back(meshRows[meshIndex]);
                    continue;
                }
                const TriangleObjectGeometry&                 kept = *previous->second->triangleGeometry;
                std::shared_ptr<const TriangleObjectGeometry> bound;
                status = qualifyRigidMeshTransform(row.worldFromLocal);
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    refusal.note(row.object.id.value, row.primType, transformRule(status, row.family));
                }
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    status = bindPreparedTriangleGeometry(kept.mesh, row.worldFromLocal, kept.doubleSided, bound);
                }
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    row.object.triangleGeometry = bound;
                    status                      = createTriangleMeshExactGeometry(bound, row.object.exactGeometry);
                }
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    break;
                }
                row.reusedReference = previous->second->geometryReference;
            }
        }
        AcquiredTypedMeshGeometry meshSource;
        if (status == OVSPATIALQUERY_API_SUCCESS && !readObjects.empty())
        {
            std::vector<std::array<double, 16u>> readPoses;
            readPoses.reserve(readRows.size());
            for (size_t readRow : readRows)
            {
                readPoses.push_back(rows[readRow].worldFromLocal);
            }
            status =
                acquireAuthoredMeshGeometry(stage, queryHandle, *filter, targetOrdinal, dictionary, readObjects,
                                            readPoses, worldMatrixColumnRead, authoredConvexIds, meshSource, refusal);
            const Row* refused = refusal.noted ? findRow(refusal.prim) : nullptr;
            if (refused != nullptr && refusal.primType == OVX_INVALID_TOKEN)
            {
                refusal.primType = refused->primType;
            }
        }
        /* The deformed meshes join the read set behind the fully read ones: the base
         * generation's prepared mesh lends its topology, the new points arrive from the
         * comparison, and the definition that identifies the content is assembled from
         * both without another stage read or a triangle expansion. */
        for (size_t deformedIndex = 0u; status == OVSPATIALQUERY_API_SUCCESS && deformedIndex < deformedRows.size();
             ++deformedIndex)
        {
            Row&                    row      = rows[deformedRows[deformedIndex]];
            const PopulationObject& previous = *previousById.at(row.object.id.value);
            DeformedMeshPoints&     moved    = deformed.at(row.object.id.value);
            status                           = qualifyRigidMeshTransform(row.worldFromLocal);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                refusal.note(row.object.id.value, row.primType, transformRule(status, row.family));
                break;
            }
            /* The new points match the retained definition's count, and the base mesh was
             * prepared from that definition's arrays, so its topology serves the new points. */
            MeshDefinitionValue definition;
            definition                   = *moved.authored;
            definition.points            = moved.points;
            definition.faceVertexCounts  = moved.authored->faceVertexCounts;
            definition.faceVertexIndices = moved.authored->faceVertexIndices;
            definition.sourceFaces       = moved.authored->sourceFaces;
            definition.sourceCorners     = moved.authored->sourceCorners;
            definition.doubleSided       = moved.authored->doubleSided;

            TriangleMeshSource mesh;
            mesh.key          = row.object.id.value;
            mesh.vertices     = std::move(moved.points);
            mesh.topologyBase = previous.triangleGeometry->mesh;
            TriangleObjectSource object;
            object.meshIndex      = static_cast<uint64_t>(meshSource.triangles.meshes.size());
            object.worldFromLocal = row.worldFromLocal;
            object.doubleSided    = definition.doubleSided;
            meshSource.triangles.meshes.push_back(std::move(mesh));
            meshSource.triangles.objects.push_back(object);
            readRows.push_back(deformedRows[deformedIndex]);
            meshSource.triangleObjectRows.push_back(readRows.size() - 1u);
            meshSource.triangleDefinitions.push_back(std::move(definition));
        }

        /* The proxies of one Mesh prototype prim share its definition: it is compared with
         * the base generation, read and prepared once, and bound under each proxy's own pose
         * (ADR-0050). A convex prototype's points are kept here and its hull prepared once
         * below with the other hulls. */
        struct PrototypeConvex
        {
            uint64_t              prototype = 0u;
            std::vector<float>    vertices;
            ConvexDefinitionValue definition;
        };
        std::unordered_map<uint64_t, GeometryDefinitionValue> sharedDefinitions;
        std::vector<PrototypeConvex>                          prototypeConvexes;
        std::vector<uint64_t>                                 meshPrototypes;
        std::unordered_map<uint64_t, std::vector<size_t>>     meshRowsByPrototype;
        for (size_t meshRow : proxyMeshRows)
        {
            std::vector<size_t>& group = meshRowsByPrototype[rows[meshRow].definitionId];
            if (group.empty())
            {
                meshPrototypes.push_back(rows[meshRow].definitionId);
            }
            group.push_back(meshRow);
        }
        const auto appendProxyObjects = [&](const std::vector<size_t>& group, uint64_t meshIndex, bool doubleSided)
        {
            for (size_t rowIndex : group)
            {
                const Row&                        row             = rows[rowIndex];
                const ovspatialquery_api_status_t transformStatus = qualifyRigidMeshTransform(row.worldFromLocal);
                if (transformStatus != OVSPATIALQUERY_API_SUCCESS)
                {
                    refusal.note(row.object.id.value, row.primType, transformRule(transformStatus, row.family));
                    return transformStatus;
                }
                TriangleObjectSource object;
                object.meshIndex      = meshIndex;
                object.worldFromLocal = row.worldFromLocal;
                object.doubleSided    = doubleSided;
                meshSource.triangles.objects.push_back(object);
                readRows.push_back(rowIndex);
                meshSource.triangleObjectRows.push_back(readRows.size() - 1u);
                meshSource.triangleDefinitions.push_back(MeshDefinitionValue{});
            }
            return OVSPATIALQUERY_API_SUCCESS;
        };
        if (status == OVSPATIALQUERY_API_SUCCESS && !meshPrototypes.empty())
        {
            /* A prototype's base definition is that of any proxy the base generation held. */
            std::unordered_map<uint64_t, const PopulationObject*>    previousByPrototype;
            std::unordered_map<uint64_t, const MeshDefinitionValue*> candidates;
            if (base != nullptr && base->objects != nullptr && base->geometryIdentities != nullptr)
            {
                std::unordered_map<uint64_t, const PopulationObject*> previousMeshes;
                for (const PopulationObject& previous : *base->objects)
                {
                    if (previous.triangleGeometry != nullptr && previous.triangleGeometry->mesh != nullptr &&
                        previous.geometryReference.value != 0u)
                    {
                        previousMeshes.emplace(previous.id.value, &previous);
                    }
                }
                std::unordered_map<uint64_t, const MeshDefinitionValue*> definitions;
                for (uint64_t prototype : meshPrototypes)
                {
                    for (size_t rowIndex : meshRowsByPrototype.at(prototype))
                    {
                        const auto previous = previousMeshes.find(rows[rowIndex].object.id.value);
                        if (rows[rowIndex].family == ExactGeometryFamily::triangleMesh &&
                            previous != previousMeshes.end())
                        {
                            previousByPrototype.emplace(prototype, previous->second);
                            definitions.emplace(previous->second->geometryReference.value, nullptr);
                            break;
                        }
                    }
                }
                if (!definitions.empty())
                {
                    base->geometryIdentities->findMeshDefinitions(definitions);
                    for (const auto& previous : previousByPrototype)
                    {
                        candidates.emplace(previous.first, definitions.at(previous.second->geometryReference.value));
                    }
                }
            }
            std::unordered_set<uint64_t>                     unchangedPrototypes;
            std::unordered_map<uint64_t, DeformedMeshPoints> deformedPrototypes;
            if (!candidates.empty())
            {
                status = findUnchangedAuthoredMeshes(stage, prototypes.handle(), *filter, targetOrdinal, dictionary,
                                                     candidates, unchangedPrototypes, &deformedPrototypes);
            }
            std::vector<PopulationObject> prototypeObjects;
            for (size_t index = 0u; status == OVSPATIALQUERY_API_SUCCESS && index < meshPrototypes.size(); ++index)
            {
                const uint64_t             prototype = meshPrototypes[index];
                const std::vector<size_t>& group     = meshRowsByPrototype.at(prototype);
                if (unchangedPrototypes.find(prototype) != unchangedPrototypes.end() &&
                    std::all_of(group.begin(), group.end(),
                                [&](size_t i)
                                {
                                    return rows[i].split.retained &&
                                           std::memcmp(rows[i].split.shape->metric.data(),
                                                       previousByPrototype.at(prototype)->ovstageShape->metric.data(),
                                                       sizeof(rows[i].split.shape->metric)) == 0 &&
                                           rows[i].split.shape->mirrored ==
                                               previousByPrototype.at(prototype)->ovstageShape->mirrored;
                                }))
                {
                    const PopulationObject&       previous = *previousByPrototype.at(prototype);
                    const TriangleObjectGeometry& kept     = *previous.triangleGeometry;
                    for (size_t rowIndex : group)
                    {
                        Row&                    row   = rows[rowIndex];
                        const PopulationObject* prior = previousObject(row);
                        if (prior && prior->triangleGeometry && prior->triangleGeometry->mesh == kept.mesh &&
                            prior->triangleGeometry->doubleSided == kept.doubleSided &&
                            prior->triangleGeometry->worldFromLocal == row.worldFromLocal)
                        {
                            row.object.triangleGeometry = prior->triangleGeometry;
                            row.object.exactGeometry    = prior->exactGeometry;
                            row.reusedReference         = prior->geometryReference;
                            continue;
                        }
                        status = qualifyRigidMeshTransform(row.worldFromLocal);
                        if (status != OVSPATIALQUERY_API_SUCCESS)
                        {
                            refusal.note(row.object.id.value, row.primType, transformRule(status, row.family));
                            break;
                        }
                        std::shared_ptr<const TriangleObjectGeometry> bound;
                        status = bindPreparedTriangleGeometry(kept.mesh, row.worldFromLocal, kept.doubleSided, bound);
                        if (status == OVSPATIALQUERY_API_SUCCESS)
                        {
                            row.object.triangleGeometry = bound;
                            status = createTriangleMeshExactGeometry(bound, row.object.exactGeometry);
                        }
                        if (status != OVSPATIALQUERY_API_SUCCESS)
                        {
                            break;
                        }
                        row.reusedReference = previous.geometryReference;
                    }
                    continue;
                }
                const auto moved = deformedPrototypes.find(prototype);
                if (moved != deformedPrototypes.end() &&
                    std::all_of(group.begin(), group.end(),
                                [&](size_t i)
                                {
                                    return rows[i].split.retained &&
                                           rows[i].split.shape->mirrored ==
                                               previousByPrototype.at(prototype)->ovstageShape->mirrored;
                                }))
                {
                    MeshDefinitionValue definition;
                    definition                   = *moved->second.authored;
                    definition.points            = moved->second.points;
                    definition.faceVertexCounts  = moved->second.authored->faceVertexCounts;
                    definition.faceVertexIndices = moved->second.authored->faceVertexIndices;
                    definition.sourceFaces       = moved->second.authored->sourceFaces;
                    definition.sourceCorners     = moved->second.authored->sourceCorners;
                    definition.doubleSided       = moved->second.authored->doubleSided;
                    TriangleMeshSource mesh;
                    mesh.key                 = prototype;
                    mesh.vertices            = std::move(moved->second.points);
                    mesh.topologyBase        = previousByPrototype.at(prototype)->triangleGeometry->mesh;
                    const uint64_t meshIndex = static_cast<uint64_t>(meshSource.triangles.meshes.size());
                    meshSource.triangles.meshes.push_back(std::move(mesh));
                    status = appendProxyObjects(group, meshIndex, definition.doubleSided);
                    sharedDefinitions.emplace(prototype, std::move(definition));
                    continue;
                }
                PopulationObject object{};
                object.id = objectIdFromValue(prototype);
                prototypeObjects.push_back(object);
            }
            if (status == OVSPATIALQUERY_API_SUCCESS && !prototypeObjects.empty())
            {
                /* The prototype prims are read in their own frame; each proxy's pose is
                 * qualified when it is bound. */
                const std::array<double, 16u> identity{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
                AcquiredTypedMeshGeometry     prototypeSource;
                status = acquireAuthoredMeshGeometry(
                    stage, prototypes.handle(), *filter, targetOrdinal, dictionary, prototypeObjects,
                    std::vector<std::array<double, 16u>>(prototypeObjects.size(), identity), true, prototypeConvexIds,
                    prototypeSource, refusal);
                /* A refused prototype is named by its first proxy, the prim the caller selected. */
                const auto refusedGroup =
                    refusal.noted ? meshRowsByPrototype.find(refusal.prim) : meshRowsByPrototype.end();
                if (refusedGroup != meshRowsByPrototype.end())
                {
                    if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT || status == OVSPATIALQUERY_API_INVALID_STATE ||
                        status == OVSPATIALQUERY_API_NOT_SUPPORTED)
                        for (size_t rowIndex : refusedGroup->second)
                            excluded.emplace(rows[rowIndex].object.id.value,
                                             OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                    refusal.prim     = rows[refusedGroup->second.front()].object.id.value;
                    refusal.primType = rows[refusedGroup->second.front()].primType;
                }
                for (size_t index = 0u;
                     status == OVSPATIALQUERY_API_SUCCESS && index < prototypeSource.triangleObjectRows.size(); ++index)
                {
                    const uint64_t prototype = prototypeObjects[prototypeSource.triangleObjectRows[index]].id.value;
                    const TriangleObjectSource& object    = prototypeSource.triangles.objects[index];
                    const uint64_t              meshIndex = static_cast<uint64_t>(meshSource.triangles.meshes.size());
                    meshSource.triangles.meshes.push_back(
                        std::move(prototypeSource.triangles.meshes[static_cast<size_t>(object.meshIndex)]));
                    status = appendProxyObjects(meshRowsByPrototype.at(prototype), meshIndex, object.doubleSided);
                    sharedDefinitions.emplace(prototype, std::move(prototypeSource.triangleDefinitions[index]));
                }
                for (size_t index = 0u;
                     status == OVSPATIALQUERY_API_SUCCESS && index < prototypeSource.convexObjectRows.size(); ++index)
                {
                    PrototypeConvex convex;
                    convex.prototype = prototypeObjects[prototypeSource.convexObjectRows[index]].id.value;
                    convex.vertices  = std::move(prototypeSource.convexes[index].vertices);
                    convex.definition =
                        std::move(std::get<ConvexDefinitionValue>(prototypeSource.convexDefinitions[index]));
                    prototypeConvexes.push_back(std::move(convex));
                }
            }
        }

        const ovspatialquery_api_status_t queryReleaseStatus = releaseQuery(stage, queryHandle);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = queryReleaseStatus;
        }
        const ovspatialquery_api_status_t prototypeReleaseStatus = prototypes.close();
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = prototypeReleaseStatus;
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }

        // Bake the symmetric shape into each local mesh; preparation shares byte-identical shapes.
        std::vector<TriangleMeshSource> shapedMeshes;
        shapedMeshes.reserve(meshSource.triangleObjectRows.size());
        std::unordered_map<MeshShapeKey, uint64_t, MeshShapeHash> bakedMeshIndices;
        bakedMeshIndices.reserve(meshSource.triangleObjectRows.size());
        for (size_t i = 0; i < meshSource.triangleObjectRows.size(); ++i)
        {
            Row&       row    = rows[readRows[meshSource.triangleObjectRows[i]]];
            auto&      object = meshSource.triangles.objects[i];
            const auto baked  = bakedMeshIndices.try_emplace({object.meshIndex, row.split.shape.get()},
                                                             static_cast<uint64_t>(shapedMeshes.size()));
            if (!baked.second)
            {
                object.meshIndex = baked.first->second;
                continue;
            }
            auto& definition = std::get<MeshDefinitionValue>(meshSource.triangleDefinitions[i]);
            if (row.proxy)
                definition = std::get<MeshDefinitionValue>(sharedDefinitions.at(row.definitionId));
            definition.shapeMetric  = row.split.shape->metric;
            definition.mirrored     = row.split.shape->mirrored;
            TriangleMeshSource mesh = meshSource.triangles.meshes[object.meshIndex];
            mesh.key                = row.object.id.value;
            mesh.ovstageMetric      = row.split.shape->metric;
            mesh.ovstageMirrored    = row.split.shape->mirrored;
            // A refit base already includes K and mirror winding.
            applyOvstageShape(mesh.vertices, *row.split.shape);
            if (mesh.topologyBase == nullptr && row.split.shape->mirrored)
                for (size_t t = 0; t < mesh.triangles.size(); t += 3)
                {
                    std::swap(mesh.triangles[t + 1], mesh.triangles[t + 2]);
                    std::swap(mesh.sourceCorners[t + 1], mesh.sourceCorners[t + 2]);
                }
            object.meshIndex = shapedMeshes.size();
            shapedMeshes.push_back(std::move(mesh));
        }
        meshSource.triangles.meshes = std::move(shapedMeshes);
        for (size_t i = 0; i < meshSource.convexObjectRows.size(); ++i)
        {
            Row&  row              = rows[readRows[meshSource.convexObjectRows[i]]];
            auto& definition       = std::get<ConvexDefinitionValue>(meshSource.convexDefinitions[i]);
            definition.shapeMetric = row.split.shape->metric;
            definition.mirrored    = row.split.shape->mirrored;
            applyOvstageShape(meshSource.convexes[i].vertices, *row.split.shape);
        }
        std::vector<std::shared_ptr<const TriangleObjectGeometry>> preparedMeshes;
        if (!meshSource.triangleObjectRows.empty())
        {
            status = prepareSharedAuthoredMeshes(*filter, meshSource.triangles, preparedMeshes);
            if (status != OVSPATIALQUERY_API_SUCCESS || preparedMeshes.size() != meshSource.triangleObjectRows.size())
            {
                return status == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_INTERNAL : status;
            }
            std::unordered_set<MeshShapeKey, MeshShapeHash> definitionMeshes;
            if (!meshRowsByPrototype.empty())
                definitionMeshes.reserve(meshSource.triangleObjectRows.size());
            for (size_t meshIndex = 0u; meshIndex < meshSource.triangleObjectRows.size(); ++meshIndex)
            {
                const size_t meshObjectRow = meshSource.triangleObjectRows[meshIndex];
                if (meshObjectRow >= readRows.size())
                {
                    return OVSPATIALQUERY_API_INTERNAL;
                }
                Row& row = rows[readRows[meshObjectRow]];
                if (preparedMeshes[meshIndex] == nullptr)
                {
                    excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                    row.hasBounds = false;
                    continue;
                }
                row.object.triangleGeometry = preparedMeshes[meshIndex];
                status = createTriangleMeshExactGeometry(row.object.triangleGeometry, row.object.exactGeometry);
                if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT || status == OVSPATIALQUERY_API_INVALID_STATE ||
                    status == OVSPATIALQUERY_API_NOT_SUPPORTED)
                {
                    excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                    row.hasBounds = false;
                    status        = OVSPATIALQUERY_API_SUCCESS;
                    continue;
                }
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    return status;
                }
                if (meshIndex >= meshSource.triangleDefinitions.size())
                {
                    return OVSPATIALQUERY_API_INTERNAL;
                }
                if (!row.proxy || definitionMeshes.emplace(row.definitionId, row.split.shape.get()).second)
                    row.definition =
                        std::make_unique<GeometryDefinitionValue>(std::move(meshSource.triangleDefinitions[meshIndex]));
                if (preparedTriangleMeshDroppedCount(*row.object.triangleGeometry->mesh) != 0u)
                    droppingMeshes.emplace_back(row.object.id.value, row.object.triangleGeometry->mesh);
            }
        }
        if (meshSource.convexes.size() != meshSource.convexObjectRows.size())
        {
            return OVSPATIALQUERY_API_INTERNAL;
        }
        /* A convexHull Mesh cooks under its hull vertex limit, as PhysX simulates it (ADR-0078). */
        const auto convexCookingOptions = [](const ConvexDefinitionValue& definition)
        {
            ConvexHullCookingOptions options;
            options.vertexLimit = definition.hullVertexLimit;
            return options;
        };
        /* Prims with the same authored points share one prepared hull, which is also the
         * owner that keeps their definition in the identity registry. */
        const auto shapeBitsLess = [](const ConvexDefinitionValue* first, const ConvexDefinitionValue* second)
        {
            const int metricOrder = std::memcmp(first->shapeMetric.data(), second->shapeMetric.data(),
                                                first->shapeMetric.size() * sizeof(double));
            if (metricOrder != 0)
                return metricOrder < 0;
            if (first->mirrored != second->mirrored)
                return first->mirrored < second->mirrored;
            if (first->hullVertexLimit != second->hullVertexLimit)
                return first->hullVertexLimit < second->hullVertexLimit;
            return first->points.size() != second->points.size()
                       ? first->points.size() < second->points.size()
                       : !first->points.empty() && std::memcmp(first->points.data(), second->points.data(),
                                                               first->points.size() * sizeof(float)) < 0;
        };
        std::map<const ConvexDefinitionValue*, std::shared_ptr<const PreparedConvexHull>, decltype(shapeBitsLess)>
            preparedByPoints(shapeBitsLess);
        for (size_t convexIndex = 0u; convexIndex < meshSource.convexes.size(); ++convexIndex)
        {
            const size_t meshObjectRow = meshSource.convexObjectRows[convexIndex];
            if (meshObjectRow >= readRows.size())
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            if (convexIndex >= meshSource.convexDefinitions.size())
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
            Row& row = rows[readRows[meshObjectRow]];
            row.definition =
                std::make_unique<GeometryDefinitionValue>(std::move(meshSource.convexDefinitions[convexIndex]));
            std::shared_ptr<const PreparedConvexHull>& shared =
                preparedByPoints[&std::get<ConvexDefinitionValue>(*row.definition)];
            const ConvexHullPreparationOutcome preparationOutcome =
                shared != nullptr
                    ? ConvexHullPreparationOutcome::success
                    : prepareConvexHull(filter->preparation.backendRuntime, meshSource.convexes[convexIndex].vertices,
                                        filter->preparation.maximumResidentBytes, shared,
                                        &filter->preparation.convexHullBudget,
                                        convexCookingOptions(std::get<ConvexDefinitionValue>(*row.definition)));
            row.preparedConvexHull = shared;
            status                 = convexHullPreparationStatus(preparationOutcome);
            if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT)
            {
                excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                row.hasBounds = false;
                status        = OVSPATIALQUERY_API_SUCCESS;
                continue;
            }
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            const ConvexConstructionOutcome outcome = createConvexExactGeometry(
                row.preparedConvexHull, meshSource.convexes[convexIndex].worldFromLocal, row.object.exactGeometry);
            if (outcome == ConvexConstructionOutcome::malformed)
            {
                excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                row.hasBounds = false;
                continue;
            }
            if (outcome == ConvexConstructionOutcome::outOfMemory)
            {
                return OVSPATIALQUERY_API_OUT_OF_MEMORY;
            }
            if (outcome != ConvexConstructionOutcome::success)
            {
                return OVSPATIALQUERY_API_INTERNAL;
            }
        }
        for (PrototypeConvex& convex : prototypeConvexes)
            for (size_t rowIndex : meshRowsByPrototype.at(convex.prototype))
            {
                Row& row               = rows[rowIndex];
                auto definition        = convex.definition;
                definition.shapeMetric = row.split.shape->metric;
                definition.mirrored    = row.split.shape->mirrored;
                row.definition         = std::make_unique<GeometryDefinitionValue>(std::move(definition));
                auto vertices          = convex.vertices;
                applyOvstageShape(vertices, *row.split.shape);
                auto&      shared = preparedByPoints[&std::get<ConvexDefinitionValue>(*row.definition)];
                const auto outcome =
                    shared != nullptr
                        ? ConvexHullPreparationOutcome::success
                        : prepareConvexHull(filter->preparation.backendRuntime, vertices,
                                            filter->preparation.maximumResidentBytes, shared,
                                            &filter->preparation.convexHullBudget,
                                            convexCookingOptions(std::get<ConvexDefinitionValue>(*row.definition)));
                row.preparedConvexHull = shared;
                status                 = convexHullPreparationStatus(outcome);
                if (status == OVSPATIALQUERY_API_INVALID_ARGUMENT)
                {
                    excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                    row.hasBounds = false;
                    status        = OVSPATIALQUERY_API_SUCCESS;
                    continue;
                }
                if (status != OVSPATIALQUERY_API_SUCCESS)
                    return status;
                const auto placed =
                    createConvexExactGeometry(row.preparedConvexHull, row.worldFromLocal, row.object.exactGeometry);
                if (placed == ConvexConstructionOutcome::malformed)
                {
                    excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                    row.hasBounds = false;
                    continue;
                }
                if (placed != ConvexConstructionOutcome::success)
                    return placed == ConvexConstructionOutcome::outOfMemory ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                                            : OVSPATIALQUERY_API_INTERNAL;
            }

        for (Row& row : rows)
        {
            if (!row.hasBounds)
                continue;
            /* Meshes and convexes carry their geometry from above; a missing one is caught below. */
            if (row.family != ExactGeometryFamily::triangleMesh && row.family != ExactGeometryFamily::convex)
            {
                // The split has validated the input and constructed a proper
                // rotation with positive uniform scale. Use that frame directly.
                QualifiedPrimitiveTransform transform{};
                transform.center = {row.worldFromLocal[3], row.worldFromLocal[7], row.worldFromLocal[11]};
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    transform.scales[axis] =
                        row.family == ExactGeometryFamily::box ? row.split.lengths[axis] : row.split.largest;
                    for (unsigned world = 0; world < 3; ++world)
                        transform.axes[axis][world] = row.worldFromLocal[world * 4 + axis] / row.split.scale;
                }
                if (row.family == ExactGeometryFamily::sphere)
                {
                    SphereGeometrySource sphere{};
                    sphere.center = transform.center;
                    sphere.radius = row.radius * transform.scales[0];
                    status        = createSphereExactGeometry(sphere, row.object.exactGeometry);
                }
                else if (row.family == ExactGeometryFamily::box)
                {
                    row.object.ovstageCubeSize = row.size;
                    BoxGeometrySource box{};
                    box.center = transform.center;
                    box.axes   = transform.axes;
                    for (uint32_t axis = 0u; axis < 3u; ++axis)
                    {
                        box.halfExtents[axis] = row.size * 0.5 * transform.scales[axis];
                    }
                    if (row.split.diagonal)
                    {
                        // An identical placed box is immutable across retained versions.
                        const PopulationObject* previous = previousObject(row);
                        if (row.reusedReference.value != 0u && previous && previous->exactGeometry)
                        {
                            ExactGeometryDescription description;
                            if (previous->exactGeometry->exportDescription(description))
                                if (const auto* kept = std::get_if<BoxGeometrySource>(&description))
                                    if (kept->center == box.center && kept->axes == box.axes &&
                                        kept->halfExtents == box.halfExtents)
                                        row.object.exactGeometry = previous->exactGeometry;
                        }
                        if (!row.object.exactGeometry)
                            status = createBoxExactGeometry(box, row.object.exactGeometry);
                    }
                    else
                    {
                        ConvexDefinitionValue definition;
                        for (unsigned corner = 0; corner < 8; ++corner)
                            for (unsigned axis = 0; axis < 3; ++axis)
                                definition.points.push_back(
                                    static_cast<float>((corner & (1u << axis) ? 1.0 : -1.0) * row.size * 0.5));
                        auto vertices = definition.points;
                        applyOvstageShape(vertices, *row.split.shape);
                        definition.shapeMetric = row.split.shape->metric;
                        definition.mirrored    = row.split.shape->mirrored;
                        status                 = convexHullPreparationStatus(prepareConvexHull(
                            filter->preparation.backendRuntime, vertices, filter->preparation.maximumResidentBytes,
                            row.preparedConvexHull, &filter->preparation.convexHullBudget));
                        if (status == OVSPATIALQUERY_API_SUCCESS)
                        {
                            const auto outcome = createConvexExactGeometry(row.preparedConvexHull, row.worldFromLocal,
                                                                           row.object.exactGeometry);
                            status = outcome == ConvexConstructionOutcome::success ? OVSPATIALQUERY_API_SUCCESS
                                     : outcome == ConvexConstructionOutcome::outOfMemory
                                         ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                         : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                        }
                        row.definition = std::make_unique<GeometryDefinitionValue>(std::move(definition));
                    }
                }
                else
                {
                    const std::array<ovx_token_t, 3u>::const_iterator axis =
                        std::find(filter->principalAxes.begin(), filter->principalAxes.end(), row.capsuleAxis);
                    const size_t          axisIndex = static_cast<size_t>(axis - filter->principalAxes.begin());
                    CapsuleGeometrySource capsule{};
                    capsule.radius = row.radius * transform.scales[axisIndex];
                    for (uint32_t worldAxis = 0u; worldAxis < 3u; ++worldAxis)
                    {
                        const double offset          = row.authoredCapsuleAxis[worldAxis] * row.height * 0.5;
                        capsule.endpointA[worldAxis] = transform.center[worldAxis] - offset;
                        capsule.endpointB[worldAxis] = transform.center[worldAxis] + offset;
                    }
                    status = createCapsuleExactGeometry(capsule, row.object.exactGeometry);
                }
            }
            if (status != OVSPATIALQUERY_API_SUCCESS || row.object.exactGeometry == nullptr)
            {
                if (status != OVSPATIALQUERY_API_SUCCESS && status != OVSPATIALQUERY_API_INVALID_ARGUMENT &&
                    status != OVSPATIALQUERY_API_INVALID_STATE && status != OVSPATIALQUERY_API_NOT_SUPPORTED)
                    return status;
                excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                row.hasBounds = false;
                status        = OVSPATIALQUERY_API_SUCCESS;
                continue;
            }
            const ovspatialquery_aabb_t& exact = row.object.exactGeometry->worldBounds();
            bool                         valid = true;
            for (uint32_t axis = 0u; axis < 3u; ++axis)
            {
                if (!inInputDomain(exact.minimum[axis]) || !inInputDomain(exact.maximum[axis]))
                {
                    valid = false;
                    break;
                }
            }
            if (!valid)
            {
                excluded.emplace(row.object.id.value, OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
                row.hasBounds = false;
                continue;
            }
            row.object.bounds = exact;
        }

        std::vector<PopulationObject>                          result;
        std::vector<GeometryDefinitionValue>                   definitions;
        std::vector<std::shared_ptr<const void>>               definitionOwners;
        std::vector<GeometryReference>                         reusedReferences;
        std::vector<std::shared_ptr<const PreparedConvexHull>> preparedHulls;
        const bool                                             anyReused = std::any_of(
            rows.begin(), rows.end(), [](const Row& row) { return row.hasBounds && row.reusedReference.value != 0u; });
        /* Proxies of one prototype prim stage one definition (ADR-0050). */
        std::vector<size_t>                                     definitionSources;
        std::vector<uint64_t>                                   publishedPrototypes;
        std::unordered_set<uint64_t>                            seenPrototypes;
        std::unordered_map<MeshShapeKey, size_t, MeshShapeHash> proxyMeshDefinitions;
        if (!meshRowsByPrototype.empty())
            proxyMeshDefinitions.reserve(meshSource.triangleObjectRows.size());
        result.reserve(rows.size());
        const auto definitionCount = static_cast<size_t>(std::count_if(
            rows.begin(), rows.end(), [](const Row& row) { return row.hasBounds && row.reusedReference.value == 0u; }));
        definitions.reserve(definitionCount);
        definitionOwners.reserve(definitionCount);
        if (anyReused)
        {
            reusedReferences.reserve(rows.size());
        }
        preparedHulls.reserve(meshSource.convexes.size());
        for (Row& row : rows)
        {
            if (!row.hasBounds)
            {
                continue;
            }
            if (row.preparedConvexHull != nullptr)
            {
                preparedHulls.push_back(row.preparedConvexHull);
            }
            /* A triangle mesh or convex definition lives in the registry while its prepared
             * mesh or hull does; the fixed-size primitive definitions stay. */
            size_t source = std::numeric_limits<size_t>::max();
            if (row.reusedReference.value == 0u)
            {
                const auto proxyKey = std::make_pair(row.definitionId, row.split.shape.get());
                const auto shared   = row.proxy && row.family == ExactGeometryFamily::triangleMesh
                                          ? proxyMeshDefinitions.find(proxyKey)
                                          : proxyMeshDefinitions.end();
                if (shared != proxyMeshDefinitions.end())
                    source = shared->second;
                else
                {
                    source = definitions.size();
                    GeometryDefinitionValue definition;
                    if (row.definition)
                        definition = std::move(*row.definition);
                    else if (row.family == ExactGeometryFamily::sphere)
                        definition = SphereDefinitionValue{row.radius};
                    else if (row.family == ExactGeometryFamily::box)
                        definition = BoxDefinitionValue{row.size};
                    else if (row.family == ExactGeometryFamily::capsule)
                    {
                        const auto axis =
                            std::find(filter->principalAxes.begin(), filter->principalAxes.end(), row.capsuleAxis);
                        definition = CapsuleDefinitionValue{
                            row.radius, row.height, static_cast<CapsuleAxis>(axis - filter->principalAxes.begin())};
                    }
                    else
                        return OVSPATIALQUERY_API_INTERNAL;
                    std::visit(
                        [&](auto& value)
                        {
                            value.shapeMetric = row.split.shape->metric;
                            value.mirrored    = row.split.shape->mirrored;
                        },
                        definition);
                    definitions.push_back(std::move(definition));
                    definitionOwners.push_back(row.object.triangleGeometry != nullptr
                                                   ? std::shared_ptr<const void>(row.object.triangleGeometry->mesh)
                                                   : std::shared_ptr<const void>(row.preparedConvexHull));
                    if (row.proxy && row.family == ExactGeometryFamily::triangleMesh)
                        proxyMeshDefinitions.emplace(proxyKey, source);
                }
            }
            if (row.proxy && seenPrototypes.insert(row.definitionId).second)
            {
                publishedPrototypes.push_back(row.definitionId);
            }
            if (!prototypePaths.empty())
            {
                definitionSources.push_back(source);
            }
            result.push_back(std::move(row.object));
            if (anyReused)
            {
                reusedReferences.push_back(row.reusedReference);
            }
        }
        /* An automatic view selects the indexed prims its own filter matches. */
        std::vector<CopiedSourceSnapshot::ViewMembership> memberships;
        memberships.reserve(viewRequests.size());
        for (const SourceViewRequest& request : viewRequests)
        {
            std::vector<PopulationObject> viewObjects;
            uint64_t                      viewMatchedCount = 0u;
            status =
                acquireFilterSnapshot(stage, dictionary, *std::static_pointer_cast<OwnedFilter>(request.sourceState),
                                      targetOrdinal, viewObjects, viewMatchedCount);
            if (status != OVSPATIALQUERY_API_SUCCESS)
            {
                return status;
            }
            CopiedSourceSnapshot::ViewMembership membership;
            membership.view = request.view;
            membership.selectedIds.reserve(viewObjects.size());
            for (const PopulationObject& object : viewObjects)
            {
                const std::unordered_map<uint64_t, size_t>::const_iterator found = rowById.find(object.id.value);
                if (found != rowById.end() && rows[found->second].hasBounds)
                {
                    membership.selectedIds.push_back(object.id.value);
                }
            }
            /* The view's scope narrows what its filter selected (ADR-0055). */
            const std::shared_ptr<PathScope>& viewScope =
                std::static_pointer_cast<OwnedFilter>(request.sourceState)->scope;
            if (viewScope != nullptr)
            {
                std::vector<uint8_t> inView;
                status = viewScope->contains(dictionary, membership.selectedIds.data(), membership.selectedIds.size(),
                                             inView);
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    return status;
                }
                size_t kept = 0u;
                for (size_t i = 0u; i < membership.selectedIds.size(); ++i)
                    if (inView[i] != 0u)
                        membership.selectedIds[kept++] = membership.selectedIds[i];
                membership.selectedIds.resize(kept);
            }
            memberships.push_back(std::move(membership));
        }
        if (outSnapshot.resolveMaterials)
        {
            OvstageMaterialResolver resolver(stage, dictionary, *filter, targetOrdinal);
            status                      = resolver.resolve(result);
            outSnapshot.retainedCharges = resolver.takeRetainedCharges();
        }
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        outSnapshot.objects             = std::move(result);
        outSnapshot.geometryDefinitions = std::move(definitions);
        outSnapshot.definitionSources   = std::move(definitionSources);
        outSnapshot.prototypePaths      = std::move(publishedPrototypes);
        outSnapshot.definitionOwners    = std::move(definitionOwners);
        outSnapshot.reusedReferences    = std::move(reusedReferences);
        outSnapshot.preparedConvexHulls = std::move(preparedHulls);
        outSnapshot.viewMemberships     = std::move(memberships);
        outSnapshot.indexedCount        = outSnapshot.objects.size();
        outSnapshot.exclusions.objects.clear();
        outSnapshot.exclusions.reasons.clear();
        for (const auto& entry : excluded)
        {
            outSnapshot.exclusions.objects.push_back(entry.first);
            outSnapshot.exclusions.reasons.push_back(entry.second);
        }
        outSnapshot.excludedCount = excluded.size();
        outSnapshot.droppedTriangles.clear();
        outSnapshot.droppedTriangleCounts = {};
        std::sort(droppingMeshes.begin(), droppingMeshes.end(),
                  [](const auto& first, const auto& second) { return first.first < second.first; });
        constexpr size_t kNamedMeshes = 4u;
        size_t           named        = 0u;
        size_t           unnamed      = 0u;
        for (const auto& [prim, mesh] : droppingMeshes)
        {
            if (excluded.count(prim) != 0u)
                continue;
            outSnapshot.droppedTriangleCounts.objects.push_back(prim);
            outSnapshot.droppedTriangleCounts.counts.push_back(preparedTriangleMeshDroppedCount(*mesh));
            if (named == kNamedMeshes)
            {
                ++unnamed;
                continue;
            }
            const std::string path = primPathText(dictionary, prim);
            outSnapshot.droppedTriangles += (named++ == 0u ? "Mesh " : "; Mesh ") +
                                            (path.empty() ? std::string("prim") : "\"" + path + "\"") + " " +
                                            describeDroppedTriangles(*mesh, "face");
        }
        if (unnamed != 0u)
            outSnapshot.droppedTriangles +=
                "; and " + std::to_string(unnamed) + " more Mesh prims dropped triangles without area";
        (void)matchedCount;
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        (void)releaseQuery(stage, queryHandle);
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        (void)releaseQuery(stage, queryHandle);
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        (void)releaseQuery(stage, queryHandle);
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

} // namespace

namespace
{

/* Appends one group's payload, row by row in logical order, as a byte count then the bytes
 * packed into words. A fixed-size row selects its data row through `data.index_map`; an
 * array attribute carries one tensor per logical row. */
ovspatialquery_api_status_t appendGroupPayload(const ReadGroup& group, std::vector<uint64_t>& out)
{
    const auto appendBytes = [&out](const uint8_t* bytes, uint64_t count)
    {
        out.push_back(count);
        const size_t first = out.size();
        out.resize(first + static_cast<size_t>((count + 7u) / 8u), 0u);
        if (count != 0u)
        {
            std::memcpy(out.data() + first, bytes, static_cast<size_t>(count));
        }
    };
    const auto tensorBytes = [](const DLTensor& tensor, uint64_t& outRows, uint64_t& outRowBytes)
    {
        if (tensor.device.device_type != kDLCPU || tensor.ndim < 1 || tensor.shape == nullptr || tensor.shape[0] < 0)
        {
            return false;
        }
        uint64_t elements = static_cast<uint64_t>(tensor.dtype.lanes) * (tensor.dtype.bits / 8u);
        for (int32_t axis = 1; axis < tensor.ndim; ++axis)
        {
            elements *= static_cast<uint64_t>(tensor.shape[axis]);
        }
        outRows     = static_cast<uint64_t>(tensor.shape[0]);
        outRowBytes = elements;
        return true;
    };
    if (group.is_array)
    {
        if (group.data.tensor_count != group.prims.count)
        {
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        for (uint32_t row = 0u; row < group.data.tensor_count; ++row)
        {
            const DLTensor& tensor = group.data.tensors[row];
            uint64_t        rows = 0u, rowBytes = 0u;
            if (!tensorBytes(tensor, rows, rowBytes))
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            appendBytes(static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset, rows * rowBytes);
        }
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (group.data.tensor_count != 1u)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const DLTensor& tensor = group.data.tensors[0];
    uint64_t        rows = 0u, rowBytes = 0u;
    if (!tensorBytes(tensor, rows, rowBytes))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const uint8_t* data = static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
    for (uint32_t row = 0u; row < group.prims.count; ++row)
    {
        const uint64_t dataRow = group.data.index_map != nullptr ? group.data.index_map[row] : row;
        if (dataRow >= rows)
        {
            return OVSPATIALQUERY_API_INVALID_STATE;
        }
        appendBytes(data + dataRow * rowBytes, rowBytes);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

/* Only the missing-bounds sentinel affects membership. Other extent values are
 * deliberately ignored: the published box comes from geometry and transform. */
ovspatialquery_api_status_t appendExtentPresence(const ReadGroup& group, std::vector<uint64_t>& out)
{
    if (group.is_array || group.data.tensor_count != 1u || group.data.tensors == nullptr)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    const DLTensor& tensor = group.data.tensors[0];
    const bool      laneRows =
        tensor.dtype.lanes == 6u && tensor.ndim == 1 && (tensor.strides == nullptr || tensor.strides[0] == 1);
    const bool componentRows = tensor.dtype.lanes == 1u && tensor.ndim == 2 && tensor.shape != nullptr &&
                               tensor.shape[1] == 6 &&
                               (tensor.strides == nullptr || (tensor.strides[0] == 6 && tensor.strides[1] == 1));
    if (tensor.device.device_type != kDLCPU || tensor.dtype.code != kDLFloat || tensor.dtype.bits != 64u ||
        !(laneRows || componentRows) || tensor.shape == nullptr || tensor.shape[0] < 0 || tensor.data == nullptr)
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    const uint64_t rowCount  = static_cast<uint64_t>(tensor.shape[0]);
    const double   emptyMin  = static_cast<double>(std::numeric_limits<float>::max());
    const double   emptyMax  = -emptyMin;
    const auto*    data      = static_cast<const uint8_t*>(tensor.data) + tensor.byte_offset;
    const size_t   firstWord = out.size();
    out.resize(firstWord + (static_cast<size_t>(group.prims.count) + 63u) / 64u, 0u);
    for (uint32_t row = 0u; row < group.prims.count; ++row)
    {
        const uint64_t dataRow = group.data.index_map == nullptr ? row : group.data.index_map[row];
        if (dataRow >= rowCount)
            return OVSPATIALQUERY_API_INVALID_STATE;
        double bounds[6];
        std::memcpy(bounds, data + dataRow * sizeof(bounds), sizeof(bounds));
        bool missing = true;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
            missing = missing && bounds[axis] == emptyMin && bounds[axis + 3u] == emptyMax;
        if (missing)
            out[firstWord + row / 64u] |= uint64_t{1} << (row % 64u);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace

ovspatialquery_api_status_t readMembershipFingerprint(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                                      const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                      const std::vector<ovx_token_t>& definitionAttributes,
                                                      const std::vector<uint64_t>&    prototypePaths,
                                                      std::vector<uint64_t>&          outFingerprint,
                                                      ovstage_query_handle_t&         outQuery)
{
    outQuery                                 = OVSTAGE_INVALID_QUERY_HANDLE;
    uint64_t                    matchedCount = 0u;
    ovstage_query_handle_t      queryHandle  = OVSTAGE_INVALID_QUERY_HANDLE;
    ovstage_query_handle_t      schemaQuery  = OVSTAGE_INVALID_QUERY_HANDLE;
    ovspatialquery_api_status_t status       = OVSPATIALQUERY_API_SUCCESS;
    ovstage_read_handle_t       readHandle   = OVSTAGE_INVALID_READ_HANDLE;
    std::vector<uint64_t>       fingerprint;
    std::vector<ovx_primpath_t> paths;
    PrototypeQuery              prototypes(stage, dictionary);
    try
    {
        /* The same write-floor check and query the host path evaluates, without reading bounds. */
        status = checkWriteFloor(stage, filter, targetOrdinal);
        const bool exactPaths =
            filter.usdPathList != OVX_INVALID_PRIMPATH_LIST && stageVtable(stage)->query_from_path_list != nullptr;
        if (status == OVSPATIALQUERY_API_SUCCESS && exactPaths)
        {
            status = mapStageStatus(
                stageVtable(stage)->query_from_path_list(stageContext(stage), filter.usdPathList, &queryHandle));
            if (status == OVSPATIALQUERY_API_SUCCESS && queryHandle == OVSTAGE_INVALID_QUERY_HANDLE)
            {
                status = OVSPATIALQUERY_API_OPERATION_FAILED;
            }
        }
        else if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status = enqueueFilterQuery(stage, filter, queryHandle);
            ovstage_query_result_t queryResult{};
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = mapStageStatus(stageVtable(stage)->fetch_query_result(stageContext(stage), queryHandle,
                                                                               OVSTAGE_TIMEOUT_INFINITE, &queryResult));
            }
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                matchedCount = queryResult.total_prim_count;
                (void)stageVtable(stage)->release_query_result(stageContext(stage), &queryResult);
            }
        }
        fingerprint.push_back(matchedCount);
        /* Which prims the capture drops as PointInstancer prototypes: a retargeted instancer
         * changes them while every matched row stays the same (ADR-0050). */
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            std::vector<ovx_primpath_t> targets;
            status = readInstancerPrototypes(stage, dictionary, filter, targetOrdinal, targets);
            std::sort(targets.begin(), targets.end());
            targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
            fingerprint.push_back(targets.size());
            for (const ovx_primpath_t target : targets)
                fingerprint.push_back(handleValue(target));
        }
        std::vector<ovx_token_t> attributes{filter.worldExtent};
        const bool               readSchemas = std::find(definitionAttributes.begin(), definitionAttributes.end(),
                                                         filter.usdSchemas) != definitionAttributes.end();
        for (ovx_token_t attribute : definitionAttributes)
            if (attribute != filter.usdSchemas)
                attributes.push_back(attribute);
        if (!prototypePaths.empty())
        {
            attributes.push_back(filter.protoPath);
        }
        /* The filter's rows, then the prototype prims' definitions through their own query. */
        const ovstage_ordinal_range_t range{0u, static_cast<ovstage_ordinal_t>(targetOrdinal), false};
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            status =
                awaitHandle(stage,
                            stageVtable(stage)->read_attributes(stageContext(stage), queryHandle, attributes.data(),
                                                                attributes.size(), range, &readHandle),
                            readHandle, OVSTAGE_INVALID_READ_HANDLE);
        }
        bool                             prototypePass = false, schemaPass = false;
        std::vector<ovstage_predicate_t> schemaPredicates;
        const ovx_string_t               meshType{"Mesh", 4u};
        ReadCursor                       cursor(stage);
        while (status == OVSPATIALQUERY_API_SUCCESS)
        {
            ReadGroup                  group{};
            const ovstage_api_status_t fetchStatus = cursor.next(readHandle, OVSTAGE_TIMEOUT_INFINITE, group);
            if (fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION && !prototypePass && !prototypePaths.empty() &&
                !definitionAttributes.empty())
            {
                prototypePass = true;
                status        = releaseRead(stage, readHandle);
                fingerprint.push_back(prototypePaths.size());
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    std::vector<ovx_primpath_t> prototypeHandles;
                    prototypeHandles.reserve(prototypePaths.size());
                    for (const uint64_t path : prototypePaths)
                        prototypeHandles.push_back(makeHandle<ovx_primpath_t>(path));
                    status = prototypes.open(prototypeHandles);
                }
                if (status == OVSPATIALQUERY_API_SUCCESS)
                {
                    status = awaitHandle(stage,
                                         stageVtable(stage)->read_attributes(
                                             stageContext(stage), prototypes.handle(), definitionAttributes.data(),
                                             definitionAttributes.size(), range, &readHandle),
                                         readHandle, OVSTAGE_INVALID_READ_HANDLE);
                }
                continue;
            }
            if (fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION && readSchemas && !schemaPass)
            {
                schemaPass = true;
                status     = releaseRead(stage, readHandle);
                // Only authored Mesh schemas affect the family. Reading this
                // array on every primitive constructs an empty tensor per row.
                // Proxy definitions are read by the prototype pass above.
                schemaPredicates = filter.predicates;
                ovstage_predicate_t mesh{};
                mesh.attribute.token = filter.usdPrimType;
                mesh.op              = OVSTAGE_FILTER_OP_IN;
                mesh.values          = &meshType;
                mesh.value_count     = 1u;
                schemaPredicates.push_back(mesh);
                const ovstage_filter_t selected{schemaPredicates.data(),
                                                static_cast<uint32_t>(schemaPredicates.size())};
                if (status == OVSPATIALQUERY_API_SUCCESS)
                    status = awaitHandle(
                        stage,
                        stageVtable(stage)->query(stageContext(stage), &selected, &filter.usdSchemas, 1u, &schemaQuery),
                        schemaQuery, OVSTAGE_INVALID_QUERY_HANDLE);
                if (status == OVSPATIALQUERY_API_SUCCESS)
                    status =
                        awaitHandle(stage,
                                    stageVtable(stage)->read_attributes(stageContext(stage), schemaQuery,
                                                                        &filter.usdSchemas, 1u, range, &readHandle),
                                    readHandle, OVSTAGE_INVALID_READ_HANDLE);
                continue;
            }
            if (fetchStatus == OVSTAGE_ERROR_END_OF_ITERATION)
            {
                break;
            }
            if (fetchStatus != OVSTAGE_OK)
            {
                status = mapStageStatus(fetchStatus);
                break;
            }
            /* Which prims, bucket by bucket, carry each attribute; and what a definition
             * attribute holds, because a value written at creation is in no change range. */
            fingerprint.push_back(handleValue(group.attribute));
            fingerprint.push_back(group.is_delete ? 1u : 0u);
            fingerprint.push_back(group.prims.count);
            status = resolveGroupPaths(group, dictionary, paths);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                for (const ovx_primpath_t path : paths)
                    fingerprint.push_back(handleValue(path));
                if (!group.is_delete)
                {
                    status = group.attribute == filter.worldExtent ? appendExtentPresence(group, fingerprint)
                                                                   : appendGroupPayload(group, fingerprint);
                }
            }
            const ovstage_api_status_t releaseStatus = releaseReadGroup(stage, group);
            if (status == OVSPATIALQUERY_API_SUCCESS)
            {
                status = mapStageStatus(releaseStatus);
            }
        }
    }
    catch (const std::bad_alloc&)
    {
        status = OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        status = OVSPATIALQUERY_API_INTERNAL;
    }
    const ovspatialquery_api_status_t readReleaseStatus = releaseRead(stage, readHandle);
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = readReleaseStatus;
    }
    const ovspatialquery_api_status_t schemaReleaseStatus = releaseQuery(stage, schemaQuery);
    if (status == OVSPATIALQUERY_API_SUCCESS)
        status = schemaReleaseStatus;
    const ovspatialquery_api_status_t prototypeReleaseStatus = prototypes.close();
    if (status == OVSPATIALQUERY_API_SUCCESS)
    {
        status = prototypeReleaseStatus;
    }
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        (void)releaseQuery(stage, queryHandle);
        return status;
    }
    outFingerprint.swap(fingerprint);
    outQuery = queryHandle;
    return OVSPATIALQUERY_API_SUCCESS;
}

// The population's binding holds the validated stage, its dictionary and a non-null filter
// (create_ovstage_population and createEmptyPopulationForSource reject null ones, and
// createSourceView a null view filter); submission rejects a negative target ordinal.
// Fills `outSnapshot` only on success, and its refusal when a prim was refused.
ovspatialquery_api_status_t acquireTypedSnapshot(const void* sourceInstance, const void* sourcePathDictionary,
                                                 const std::shared_ptr<void>&          populationSourceState,
                                                 const std::vector<SourceViewRequest>& viewRequests,
                                                 ovspatialquery_ordinal_t              targetOrdinal,
                                                 const SourceAcquisitionBase* base, CopiedSourceSnapshot& outSnapshot)
{
    ovstage_instance_t* stage = const_cast<ovstage_instance_t*>(static_cast<const ovstage_instance_t*>(sourceInstance));
    path_dictionary_instance_t* dictionary =
        const_cast<path_dictionary_instance_t*>(static_cast<const path_dictionary_instance_t*>(sourcePathDictionary));
    PrimRefusal                           refusal;
    std::unordered_map<uint64_t, int32_t> excluded;
    ovspatialquery_api_status_t           status;
    for (;;)
    {
        refusal                      = {};
        const size_t priorExclusions = excluded.size();
        status = acquireTypedRows(stage, dictionary, std::static_pointer_cast<OwnedFilter>(populationSourceState),
                                  viewRequests, targetOrdinal, base, outSnapshot, refusal, excluded);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            break;
        const bool content = status == OVSPATIALQUERY_API_NOT_SUPPORTED ||
                             status == OVSPATIALQUERY_API_INVALID_ARGUMENT ||
                             status == OVSPATIALQUERY_API_INVALID_STATE;
        if (content && excluded.size() > priorExclusions)
            continue;
        if (!content || !refusal.noted || !refusal.prim || refusal.rule == kNoProxyRowsRule ||
            excluded.count(refusal.prim))
        {
            outSnapshot.refusal = describePrimRefusal(dictionary, refusal);
            break;
        }
        excluded.emplace(refusal.prim, refusal.rule == nullptr ? OVSPATIALQUERY_OVSTAGE_EXCLUSION_UNSUPPORTED_TYPE
                                                               : OVSPATIALQUERY_OVSTAGE_EXCLUSION_MALFORMED_GEOMETRY);
    }
    return status;
}

} // namespace detail
} // namespace ovspatialquery
