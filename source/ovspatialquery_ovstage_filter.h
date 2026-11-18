/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_FILTER_H
#define OVSPATIALQUERY_OVSTAGE_FILTER_H

#include "ovspatialquery_ovstage_abi.h"
#include "ovspatialquery_ovstage_scope.h"
#include "ovspatialquery_typed_geometry.h"
#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class PreparedTriangleMesh;
struct OvstageDeviceMembership;

/* The `usd-prim-type` values the adapter turns into objects, and the family each becomes; a
 * Mesh that carries a convexHull approximation becomes CONVEX instead. `instanceName` is the
 * type ovstage gives an instance proxy of that type (ADR-0050): the proxy is an object whose
 * definition comes from its `_protoPath` prim. An empty filter selects exactly these types,
 * both names, and the capture refuses every other type with NOT_SUPPORTED, so supporting
 * another primitive starts here, then teaches the capture to build its geometry. */
struct PublishedPrimType
{
    const char*         name;
    const char*         instanceName;
    ExactGeometryFamily family;
};
constexpr std::array<PublishedPrimType, 4u> kPublishedPrimTypes{{
    {"Sphere", "SphereInstance", ExactGeometryFamily::sphere},
    {"Cube", "CubeInstance", ExactGeometryFamily::box},
    {"Capsule", "CapsuleInstance", ExactGeometryFamily::capsule},
    {"Mesh", "MeshInstance", ExactGeometryFamily::triangleMesh},
}};

struct OwnedPredicate
{
    ovstage_predicate_t       value{};
    std::vector<std::string>  valueStorage;
    std::vector<uint8_t>      nullValues;
    std::vector<ovx_string_t> values;
};

struct PreparedOvstageMeshKey
{
    uint64_t              content = 0u;
    std::array<double, 9> metric{1, 0, 0, 0, 1, 0, 0, 0, 1};
    bool                  mirrored = false;

    bool operator<(const PreparedOvstageMeshKey& other) const noexcept
    {
        if (content != other.content)
            return content < other.content;
        const int order = std::memcmp(metric.data(), other.metric.data(), metric.size() * sizeof(double));
        return order != 0 ? order < 0 : mirrored < other.mirrored;
    }
};

struct OwnedFilter
{
    std::vector<OwnedPredicate>      storage;
    std::vector<ovstage_predicate_t> predicates;
    ovstage_filter_t                 filter{};
    ovx_token_t                      worldExtent             = OVX_INVALID_TOKEN;
    ovx_token_t                      points                  = OVX_INVALID_TOKEN;
    ovx_token_t                      faceVertexCounts        = OVX_INVALID_TOKEN;
    ovx_token_t                      faceVertexIndices       = OVX_INVALID_TOKEN;
    ovx_token_t                      doubleSided             = OVX_INVALID_TOKEN;
    ovx_token_t                      fabricWorldMatrix       = OVX_INVALID_TOKEN;
    ovx_token_t                      worldMatrix             = OVX_INVALID_TOKEN;
    ovx_token_t                      usdPrimType             = OVX_INVALID_TOKEN;
    ovx_token_t                      usdPath                 = OVX_INVALID_TOKEN;
    ovx_token_t                      radius                  = OVX_INVALID_TOKEN;
    ovx_token_t                      size                    = OVX_INVALID_TOKEN;
    ovx_token_t                      height                  = OVX_INVALID_TOKEN;
    ovx_token_t                      axis                    = OVX_INVALID_TOKEN;
    ovx_token_t                      orientation             = OVX_INVALID_TOKEN;
    ovx_token_t                      holeIndices             = OVX_INVALID_TOKEN;
    ovx_token_t                      rightHanded             = OVX_INVALID_TOKEN;
    ovx_token_t                      leftHanded              = OVX_INVALID_TOKEN;
    ovx_token_t                      usdSchemas              = OVX_INVALID_TOKEN;
    ovx_token_t                      physicsApproximation    = OVX_INVALID_TOKEN;
    ovx_token_t                      physicsMeshCollisionApi = OVX_INVALID_TOKEN;
    ovx_token_t                      convexHull              = OVX_INVALID_TOKEN;
    ovx_token_t                      hullVertexLimit         = OVX_INVALID_TOKEN;
    ovx_token_t                      protoPath               = OVX_INVALID_TOKEN;
    ovx_token_t                      protoPrimType           = OVX_INVALID_TOKEN;
    ovx_token_t                      instancerPath           = OVX_INVALID_TOKEN;
    /* A PointInstancer's prototype targets: `prototypes`, which ovstage 0.3's RENDERING rewrites
     * to its copies, `prototypes:usd-targets` (0.3, authored) and `_flattenedPrototypes` (0.4,
     * the copies). */
    std::array<ovx_token_t, 3u> instancerPrototypes{};
    std::array<ovx_token_t, 3u> principalAxes{};
    /* kPublishedPrimTypes' names and instance names as tokens, in its order. */
    std::array<ovx_token_t, kPublishedPrimTypes.size()> publishedPrimTypes{};
    std::array<ovx_token_t, kPublishedPrimTypes.size()> publishedInstanceTypes{};
    SourceGeometryPreparationContext                    preparation{};
    /* A filter of exactly one `usd-path IN` predicate names its prims, so its paths are
     * interned once, deduplicated, into a list that each publication queries through
     * Ovstage's path dictionary instead of a stage-side evaluation (Ovstage derives
     * `usd-path` per value at filter time: about 100 ms for 376 paths on the fleet scene,
     * 0.5 ms through the list). Invalid for every other filter, which Ovstage evaluates.
     * The list is released with the filter through the dictionary that interned it. */
    path_dictionary_instance_t* dictionary   = nullptr;
    ovx_primpath_list_t         usdPathList  = OVX_INVALID_PRIMPATH_LIST;
    size_t                      usdPathCount = 0u;
    /* The population's or view's path scope (ADR-0055); null selects everything the
     * filter selects. */
    std::shared_ptr<PathScope> scope;
    ~OwnedFilter();
    /* The family a published prim type becomes; false for every other type. */
    bool familyOf(ovx_token_t primType, ExactGeometryFamily& outFamily) const
    {
        for (size_t index = 0u; index < publishedPrimTypes.size(); ++index)
        {
            if (publishedPrimTypes[index] == primType)
            {
                outFamily = kPublishedPrimTypes[index].family;
                return true;
            }
        }
        return false;
    }
    /* The family of an instance proxy's type (`CubeInstance`); false for every other type. */
    bool proxyFamilyOf(ovx_token_t primType, ExactGeometryFamily& outFamily) const
    {
        for (size_t index = 0u; index < publishedInstanceTypes.size(); ++index)
        {
            if (publishedInstanceTypes[index] == primType)
            {
                outFamily = kPublishedPrimTypes[index].family;
                return true;
            }
        }
        return false;
    }
    /* Prepared authored meshes of this population's earlier acquisitions, by content
     * hash, for as long as a generation still holds them. Identical meshes (one asset
     * instanced under several placements, an unchanged mesh under a new pose) then
     * share one preparation and one cooked backend mesh instead of preparing and
     * cooking each prim of each publication again. */
    std::mutex                                                                       preparedMeshMutex;
    std::multimap<PreparedOvstageMeshKey, std::weak_ptr<const PreparedTriangleMesh>> preparedMeshes;
    /* Device publications (ADR-0047): the membership the newest admitted one published,
     * which the next one compares with; admissions hold deviceMutex. */
    std::mutex                               deviceMutex;
    std::shared_ptr<OvstageDeviceMembership> deviceMembership;
};

/* The first prim a publication refuses and the rule it broke, which the refused call's
 * diagnostic names. The first note wins; a refusal without one keeps the status's generic
 * text. `rule` is static text; a null `rule` on a noted refusal is the unpublished-type rule,
 * whose text lists kPublishedPrimTypes. */
struct PrimRefusal
{
    uint64_t    prim     = 0u; /* The prim's ovx_primpath_t; 0 when it could not be identified. */
    ovx_token_t primType = OVX_INVALID_TOKEN;
    const char* rule     = nullptr;
    bool        noted    = false;

    void note(uint64_t refusedPrim, ovx_token_t refusedType, const char* brokenRule) noexcept
    {
        if (!noted)
        {
            prim     = refusedPrim;
            primType = refusedType;
            rule     = brokenRule;
            noted    = true;
        }
    }
};

/* The prim's path, such as "/World/Mesh"; empty when the dictionary cannot name it. */
std::string primPathText(path_dictionary_instance_t* dictionary, uint64_t prim);

/* `"/World" is an Xform; ovstage populations publish Sphere, Cube, Capsule and Mesh prims`:
 * the refused prim's path, its type when known, then the rule. Empty when nothing was noted. */
std::string describePrimRefusal(path_dictionary_instance_t* dictionary, const PrimRefusal& refusal);

ovspatialquery_api_status_t mapStageStatus(ovstage_api_status_t status);

ovspatialquery_api_status_t mapDictionaryResult(path_dictionary_instance_t* dictionary, ovx_api_result_t result);

/* Calls `visit(index, elements, count)` with each path's elements, growing the token buffer
 * until the dictionary decomposes every path; "/" is one processed path with no elements. */
template <typename Visit>
ovspatialquery_api_status_t forEachPathElements(path_dictionary_instance_t* dictionary, const ovx_primpath_t* paths,
                                                size_t count, Visit visit)
{
    std::vector<ovx_token_t>  buffer(count * 16u + 16u);
    std::vector<ovx_token_t*> elements(count);
    std::vector<size_t>       elementCounts(count);
    size_t                    done = 0u;
    while (done < count)
    {
        size_t                            processed = 0u;
        const ovspatialquery_api_status_t status    = mapDictionaryResult(
            dictionary, dictionary->vtable->get_tokens_from_paths(dictionary->context, paths + done, count - done,
                                                                     buffer.data(), buffer.size(), elements.data() + done,
                                                                     elementCounts.data() + done, &processed));
        if (status != OVSPATIALQUERY_API_SUCCESS)
        {
            return status;
        }
        if (processed == 0u)
        {
            /* A path deeper than any stage holds means the dictionary is not decomposing it. */
            if (buffer.size() > (size_t{1} << 24u))
            {
                return OVSPATIALQUERY_API_NOT_SUPPORTED;
            }
            buffer.resize(buffer.size() * 2u);
            continue;
        }
        for (size_t index = done; index < done + processed; ++index)
        {
            visit(index, elements[index], elementCounts[index]);
        }
        done += processed;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t copyFilter(ovstage_instance_t* stage, const ovstage_filter_t& source,
                                       path_dictionary_instance_t* dictionary, std::shared_ptr<OwnedFilter>& outFilter,
                                       SourceGeometryPreparationContext preparation = {});

bool estimateFilterCharge(const OwnedFilter& filter, uint64_t& outCharge);

ovspatialquery_api_status_t waitAndRelease(ovstage_instance_t* stage, ovstage_op_id_t operation);

ovspatialquery_api_status_t releaseRead(ovstage_instance_t* stage, ovstage_read_handle_t& readHandle);

ovspatialquery_api_status_t releaseOrdinalQuery(ovstage_instance_t*             stage,
                                                ovstage_ordinal_query_handle_t& ordinalHandle);

ovspatialquery_api_status_t releaseQuery(ovstage_instance_t* stage, ovstage_query_handle_t& queryHandle);

/* What decides an Ovstage population's membership on the device path (ADR-0047), read on
 * the host at the sealed target ordinal through one evaluation of the filter: the matched
 * count, and per group of `_worldExtent` and of each definition attribute the attribute,
 * its prims' path values in row order and, for a definition attribute, each row's bytes.
 * With `prototypePaths`, the definition prims of the membership's instance proxies, it also
 * covers the proxies' `_protoPath` and those prims' definition attributes, so a prototype
 * edit is a membership change (ADR-0050). Two equal fingerprints name the same prims in the
 * same buckets with the same definitions. On success `outQuery` is that evaluation's
 * query, which the caller releases. */
ovspatialquery_api_status_t readMembershipFingerprint(ovstage_instance_t* stage, path_dictionary_instance_t* dictionary,
                                                      const OwnedFilter& filter, ovspatialquery_ordinal_t targetOrdinal,
                                                      const std::vector<ovx_token_t>& definitionAttributes,
                                                      const std::vector<uint64_t>&    prototypePaths,
                                                      std::vector<uint64_t>&          outFingerprint,
                                                      ovstage_query_handle_t&         outQuery);
/* Resolves the prim path of each of the group's logical rows, in row order. */
ovspatialquery_api_status_t resolveGroupPaths(const ReadGroup& group, path_dictionary_instance_t* dictionary,
                                              std::vector<ovx_primpath_t>& outPaths);

/* Rows a capture drops before any rule sees them, values included (ADR-0050): ovstage's
 * copies of native-instancing prototypes, flagged kDroppedPrototypeCopy, and a
 * PointInstancer's prototype prims and RENDERING's copies of them, flagged
 * kDroppedPointPrototype. `mark` flags each of `paths` nonzero to drop it. */
constexpr uint8_t kDroppedPrototypeCopy  = 1u;
constexpr uint8_t kDroppedPointPrototype = 2u;
class DroppedRows
{
  public:
    virtual ~DroppedRows()                                                           = default;
    virtual ovspatialquery_api_status_t mark(path_dictionary_instance_t*        dictionary,
                                             const std::vector<ovx_primpath_t>& paths,
                                             std::vector<uint8_t>&              outDropped) const = 0;
};

/* What the dropped rows of a capture were: how many, how many of them were native-instancing
 * prototype copies, and the first such copy's path. */
struct DroppedRowCount
{
    uint64_t count  = 0u;
    uint64_t copies = 0u;
    uint64_t first  = 0u;
};

/* Appends the group's rows as objects, flagging each 1 in outHasBounds when it is
 * bounded and 0 when it carries the package's missing-bounds sentinel. Rows `dropped`
 * marks are counted in `outDropped` and never validated. */
ovspatialquery_api_status_t appendGroup(const ReadGroup& group, ovx_token_t expectedAttribute,
                                        ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
                                        std::unordered_set<uint64_t>&  identities,
                                        std::vector<PopulationObject>& objects, std::vector<uint8_t>& outHasBounds,
                                        const DroppedRows* dropped = nullptr, DroppedRowCount* outDropped = nullptr);

ovspatialquery_api_status_t appendIdentityGroup(const ReadGroup& group, ovx_token_t expectedAttribute,
                                                ovspatialquery_ordinal_t       targetOrdinal,
                                                path_dictionary_instance_t*    dictionary,
                                                std::unordered_set<uint64_t>&  identities,
                                                std::vector<PopulationObject>& objects,
                                                const DroppedRows*             dropped    = nullptr,
                                                DroppedRowCount*               outDropped = nullptr);

/* The SourceSnapshotAcquirer registered for
 * PopulationPreparationStrategy::typedFiveFamily. Defined in
 * ovspatialquery_ovstage_typed_acquisition.cpp; referenced by the
 * create-population service-ABI body in ovspatialquery_ovstage_acquisition.cpp. */
ovspatialquery_api_status_t acquireTypedSnapshot(const void* sourceInstance, const void* sourcePathDictionary,
                                                 const std::shared_ptr<void>&          populationSourceState,
                                                 const std::vector<SourceViewRequest>& viewRequests,
                                                 ovspatialquery_ordinal_t              targetOrdinal,
                                                 const SourceAcquisitionBase* base, CopiedSourceSnapshot& outSnapshot);

} // namespace detail
} // namespace ovspatialquery

#endif
