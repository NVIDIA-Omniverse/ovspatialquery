/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_ovstage_filter.h"

#include "ovspatialquery_instance_host_ledger.h"

#include <ovspatialquery_instance_source.h>

#include <ovstage/ovstage_api/ovstage_api.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

ovspatialquery_api_status_t mapStageStatus(ovstage_api_status_t status)
{
    if (status == OVSTAGE_OK)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (status == OVSTAGE_ERROR_OUT_OF_MEMORY)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    if (status == OVSTAGE_ERROR_NOT_SUPPORTED)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    return OVSPATIALQUERY_API_OPERATION_FAILED;
}

ovstage_api_status_t ReadGroupHold::release() noexcept
{
    if (released)
    {
        return OVSTAGE_OK;
    }
    released = true;
    return stageVtable(stage)->release_groups(stageContext(stage), &group, 1u);
}

ReadGroupHold::~ReadGroupHold() { (void)release(); }

ovstage_api_status_t ReadCursor::next(ovstage_read_handle_t read, ovstage_timeout_ns_t timeout, ReadGroup& outGroup)
{
    outGroup                   = ReadGroup{};
    const OvstageLayout layout = ovstageLayout(mStage);
    if (layout == OvstageLayout::v03)
    {
        // The slot keeps its 0.3 signature on a 0.3 runtime; see ovspatialquery_ovstage_v03_layout.h.
        const auto fetch = reinterpret_cast<ovspatialquery_ovstage_v03_fetch_read_next_t>(
            reinterpret_cast<void (*)()>(stageVtable(mStage)->fetch_read_next));
        ovspatialquery_ovstage_v03_read_group_t& group = outGroup.v03;
        const ovstage_api_status_t status = fetch(stageContext(mStage), handleValue(read), timeout, &group);
        if (status != OVSTAGE_OK)
        {
            return status;
        }
        outGroup.layout    = OvstageLayout::v03;
        outGroup.attribute = makeHandle<ovx_token_t>(group.attribute);
        outGroup.ordinal   = group.ordinal;
        outGroup.is_delete = group.is_delete;
        outGroup.is_array  = group.is_array;
        outGroup.semantic  = group.semantic;
        outGroup.prims     = group.prims;
        outGroup.data      = group.data;
        outGroup.meta      = group.meta;
        return OVSTAGE_OK;
    }
    if (layout != OvstageLayout::v04)
    {
        return OVSTAGE_ERROR_NOT_SUPPORTED;
    }
    if (mPending == nullptr || mRead != read)
    {
        mPending.reset();
        auto                       hold      = std::make_shared<ReadGroupHold>();
        size_t                     delivered = 0u;
        const ovstage_api_status_t status =
            stageVtable(mStage)->fetch_read_next(stageContext(mStage), read, timeout, &hold->group, 1u, &delivered);
        if (status != OVSTAGE_OK || delivered != 1u)
        {
            hold->released = true;
            return status != OVSTAGE_OK ? status : OVSTAGE_ERROR_INTERNAL;
        }
        hold->stage = mStage;
        if (hold->group.column_count == 0u || hold->group.columns == nullptr)
        {
            return OVSTAGE_ERROR_NOT_SUPPORTED;
        }
        mPending    = std::move(hold);
        mRead       = read;
        mNextColumn = 0u;
    }
    const ovstage_read_column_t& column = mPending->group.columns[mNextColumn];
    outGroup.layout                     = OvstageLayout::v04;
    outGroup.attribute                  = column.attribute;
    outGroup.ordinal                    = column.ordinal;
    outGroup.is_delete                  = column.is_delete;
    outGroup.is_array                   = column.is_array;
    outGroup.semantic                   = column.semantic;
    outGroup.prims                      = mPending->group.prims;
    outGroup.data                       = column.data;
    outGroup.meta                       = column.meta;
    outGroup.v04                        = mPending;
    if (++mNextColumn == mPending->group.column_count)
    {
        // The views own the group from its last column on, so the last release reports.
        mPending.reset();
    }
    return OVSTAGE_OK;
}

ovstage_api_status_t releaseReadGroup(ovstage_instance_t* stage, ReadGroup& group)
{
    switch (group.layout)
    {
    case OvstageLayout::v03: {
        group.layout       = OvstageLayout::unsupported;
        const auto release = reinterpret_cast<ovspatialquery_ovstage_v03_release_group_t>(
            reinterpret_cast<void (*)()>(stageVtable(stage)->release_groups));
        return release(stageContext(stage), &group.v03);
    }
    case OvstageLayout::v04: {
        group.layout                        = OvstageLayout::unsupported;
        std::shared_ptr<ReadGroupHold> hold = std::move(group.v04);
        return hold != nullptr && hold.use_count() == 1 ? hold->release() : OVSTAGE_OK;
    }
    case OvstageLayout::unsupported: break;
    }
    return OVSTAGE_OK;
}

ovspatialquery_api_status_t mapDictionaryResult(path_dictionary_instance_t* dictionary, ovx_api_result_t result)
{
    if (result.status == OVX_API_SUCCESS)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    if (result.error.ptr != nullptr && dictionary->vtable->release_error != nullptr)
    {
        dictionary->vtable->release_error(dictionary->context, result.error);
    }
    return OVSPATIALQUERY_API_NOT_SUPPORTED;
}

std::string primPathText(path_dictionary_instance_t* dictionary, uint64_t prim)
{
    if (prim == 0u)
    {
        return {};
    }
    const auto tokenText = [&](ovx_token_t token, std::string& outText)
    {
        ovx_string_t text{nullptr, 0u};
        if (token == OVX_INVALID_TOKEN || dictionary->vtable->get_strings_from_tokens == nullptr ||
            mapDictionaryResult(dictionary,
                                dictionary->vtable->get_strings_from_tokens(dictionary->context, &token, 1u, &text)) !=
                OVSPATIALQUERY_API_SUCCESS ||
            text.ptr == nullptr)
        {
            return false;
        }
        outText.assign(text.ptr, text.length);
        return true;
    };
    /* A prim path decomposes into its element names; the pseudo-root into none. */
    std::string                   path;
    std::array<ovx_token_t, 256u> elements{};
    ovx_token_t*                  pathTokens = nullptr;
    size_t                        tokenCount = 0u;
    size_t                        processed  = 0u;
    const ovx_primpath_t          primPath   = makeHandle<ovx_primpath_t>(prim);
    if (mapDictionaryResult(dictionary, dictionary->vtable->get_tokens_from_paths(
                                            dictionary->context, &primPath, 1u, elements.data(), elements.size(),
                                            &pathTokens, &tokenCount, &processed)) == OVSPATIALQUERY_API_SUCCESS &&
        processed == 1u && (tokenCount == 0u || pathTokens != nullptr))
    {
        path = tokenCount == 0u ? "/" : "";
        std::string element;
        for (size_t index = 0u; index < tokenCount; ++index)
        {
            if (!tokenText(pathTokens[index], element))
            {
                return {};
            }
            path += "/" + element;
        }
    }
    return path;
}

std::string describePrimRefusal(path_dictionary_instance_t* dictionary, const PrimRefusal& refusal)
{
    if (!refusal.noted)
    {
        return {};
    }
    std::string rule;
    if (refusal.rule != nullptr)
    {
        rule = refusal.rule;
    }
    else
    {
        rule = "ovstage populations publish ";
        for (size_t index = 0u; index < kPublishedPrimTypes.size(); ++index)
        {
            rule += index == 0u ? "" : index + 1u == kPublishedPrimTypes.size() ? " and " : ", ";
            rule += kPublishedPrimTypes[index].name;
        }
        rule += " prims";
    }
    const auto tokenText = [&](ovx_token_t token, std::string& outText)
    {
        ovx_string_t text{nullptr, 0u};
        if (token == OVX_INVALID_TOKEN || dictionary->vtable->get_strings_from_tokens == nullptr ||
            mapDictionaryResult(dictionary,
                                dictionary->vtable->get_strings_from_tokens(dictionary->context, &token, 1u, &text)) !=
                OVSPATIALQUERY_API_SUCCESS ||
            text.ptr == nullptr)
        {
            return false;
        }
        outText.assign(text.ptr, text.length);
        return true;
    };
    const std::string path = primPathText(dictionary, refusal.prim);
    std::string       type;
    tokenText(refusal.primType, type);
    if (path.empty())
    {
        return type.empty() ? "a selected prim cannot be published: " + rule
                            : "a selected " + type + " prim cannot be published: " + rule;
    }
    if (type.empty())
    {
        return "\"" + path + "\": " + rule;
    }
    /* "an Xform", "an Imageable", "a Mesh". */
    const bool vowelSound = std::string("AEIOUX").find(type[0]) != std::string::npos;
    return "\"" + path + "\" is " + (vowelSound ? "an " : "a ") + type + "; " + rule;
}

OwnedFilter::~OwnedFilter()
{
    if (usdPathList != OVX_INVALID_PRIMPATH_LIST && dictionary != nullptr && dictionary->vtable != nullptr &&
        dictionary->vtable->release_path_list_reference != nullptr)
    {
        (void)mapDictionaryResult(dictionary,
                                  dictionary->vtable->release_path_list_reference(dictionary->context, usdPathList));
    }
}

namespace
{

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
std::atomic<bool> gFailNextGroupMaterialization{false};
#endif

/* Interns the paths of a filter that is one `usd-path IN` predicate into `filter.usdPathList`.
 * Equivalent paths intern to one handle, so the list holds each prim once however the
 * predicate repeated it. Anything the dictionary rejects leaves the list invalid and the
 * filter on Ovstage's evaluation, which then owns the diagnosis. */
void internExactPathList(OwnedFilter& filter, ovx_token_t usdPathToken, path_dictionary_instance_t* dictionary)
{
    if (filter.storage.size() != 1u || dictionary->vtable->create_paths_from_strings == nullptr ||
        dictionary->vtable->create_path_list_from_paths == nullptr ||
        dictionary->vtable->release_path_list_reference == nullptr)
    {
        return;
    }
    const OwnedPredicate& predicate = filter.storage[0];
    const bool            namesPath = predicate.value.attribute.token == usdPathToken;
    if (!namesPath || predicate.value.op != OVSTAGE_FILTER_OP_IN || predicate.values.empty() ||
        std::find(predicate.nullValues.begin(), predicate.nullValues.end(), uint8_t{1u}) != predicate.nullValues.end())
    {
        return;
    }
    std::vector<ovx_primpath_t> paths(predicate.values.size(), OVX_INVALID_PRIMPATH);
    if (mapDictionaryResult(dictionary, dictionary->vtable->create_paths_from_strings(
                                            dictionary->context, predicate.values.data(), predicate.values.size(),
                                            paths.data())) != OVSPATIALQUERY_API_SUCCESS ||
        std::find(paths.begin(), paths.end(), OVX_INVALID_PRIMPATH) != paths.end())
    {
        return;
    }
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    ovx_primpath_list_t list = OVX_INVALID_PRIMPATH_LIST;
    if (mapDictionaryResult(dictionary, dictionary->vtable->create_path_list_from_paths(
                                            dictionary->context, paths.data(), paths.size(), &list)) !=
            OVSPATIALQUERY_API_SUCCESS ||
        list == OVX_INVALID_PRIMPATH_LIST)
    {
        return;
    }
    filter.usdPathList  = list;
    filter.usdPathCount = paths.size();
}

/* Releases a read, ordinal-query or query handle and waits for the release operation. */
template <typename Handle>
ovspatialquery_api_status_t releaseHandle(ovstage_instance_t* stage, Handle& handle, Handle invalid,
                                          ovstage_enqueue_result_t (*release)(ovstage_data_plane_context_t*, Handle))
{
    if (handle == invalid)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    ovstage_enqueue_result_t result = release(stageContext(stage), handle);
    while (result.status == OVSTAGE_ERROR_QUEUE_FULL)
    {
        std::this_thread::yield();
        result = release(stageContext(stage), handle);
    }
    if (result.status != OVSTAGE_OK)
    {
        return mapStageStatus(result.status);
    }
    handle = invalid;
    return waitAndRelease(stage, result.op_index);
}

} // namespace

ovspatialquery_api_status_t copyFilter(ovstage_instance_t* stage, const ovstage_filter_t& source,
                                       path_dictionary_instance_t* dictionary, std::shared_ptr<OwnedFilter>& outFilter,
                                       SourceGeometryPreparationContext preparation)
{
    if ((source.count != 0u && source.predicates == nullptr) || dictionary == nullptr ||
        dictionary->vtable == nullptr || dictionary->vtable->create_tokens_from_strings == nullptr ||
        dictionary->vtable->get_strings_from_tokens == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    // A 0.3 selector names its attribute by token unless the token is zero; a 0.4 selector
    // by string whenever the string is non-null.
    const bool stringFirst = ovstageLayout(stage) == OvstageLayout::v04;
    /* The published types' names, then their instance names, in kPublishedPrimTypes' order. */
    constexpr size_t                          kTypeCount = kPublishedPrimTypes.size();
    std::array<ovx_string_t, 2u * kTypeCount> publishedTypes{};
    for (size_t index = 0u; index < kTypeCount; ++index)
    {
        const PublishedPrimType& type      = kPublishedPrimTypes[index];
        publishedTypes[index]              = ovx_string_t{type.name, std::strlen(type.name)};
        publishedTypes[kTypeCount + index] = ovx_string_t{type.instanceName, std::strlen(type.instanceName)};
    }
    /* ovstage's empty filter matches every prim, the pseudo-root, Xforms, Scopes and its own
     * bookkeeping prims included, and no population can publish those; the adapter's empty
     * filter selects the prim types it turns into objects instead, instance proxies included
     * (ADR-0050). */
    ovstage_predicate_t publishedTypePredicate{};
    publishedTypePredicate.attribute.string = ovx_string_t{"usd-prim-type", sizeof("usd-prim-type") - 1u};
    publishedTypePredicate.op               = OVSTAGE_FILTER_OP_IN;
    publishedTypePredicate.values           = publishedTypes.data();
    publishedTypePredicate.value_count      = publishedTypes.size();
    const ovstage_filter_t selected = source.count == 0u ? ovstage_filter_t{&publishedTypePredicate, 1u} : source;
    try
    {
        std::shared_ptr<OwnedFilter> result = std::make_shared<OwnedFilter>();
        result->preparation                 = preparation;
        result->storage.reserve(selected.count);
        result->predicates.reserve(selected.count);
        for (size_t predicateIndex = 0u; predicateIndex < selected.count; ++predicateIndex)
        {
            const ovstage_predicate_t& input = selected.predicates[predicateIndex];
            const uint64_t             token = handleValue(input.attribute.token);
            const bool                 useString =
                source.count == 0u || (stringFirst ? input.attribute.string.ptr != nullptr : token == 0u);
            // Token zero is invalid in 0.3 and the empty string in 0.4; UINT64_MAX is invalid in 0.4.
            if ((useString && (input.attribute.string.ptr == nullptr || input.attribute.string.length == 0u)) ||
                (!useString && (token == 0u || token == std::numeric_limits<uint64_t>::max())) ||
                (input.value_count != 0u && input.values == nullptr))
            {
                return OVSPATIALQUERY_API_INVALID_ARGUMENT;
            }
            OwnedPredicate owned;
            owned.value.op              = input.op;
            owned.value.attribute.token = input.attribute.token;
            if (useString)
            {
                // Send only an interned token to Ovstage, independent of which headers
                // built this companion and which selector convention the runtime uses.
                const auto internStatus = mapDictionaryResult(
                    dictionary, dictionary->vtable->create_tokens_from_strings(
                                    dictionary->context, &input.attribute.string, 1u, &owned.value.attribute.token));
                if (internStatus != OVSPATIALQUERY_API_SUCCESS)
                {
                    return internStatus;
                }
            }
            owned.valueStorage.reserve(input.value_count);
            owned.nullValues.reserve(input.value_count);
            owned.values.reserve(input.value_count);
            for (size_t valueIndex = 0u; valueIndex < input.value_count; ++valueIndex)
            {
                if (input.values[valueIndex].ptr == nullptr && input.values[valueIndex].length != 0u)
                {
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                if (input.values[valueIndex].ptr == nullptr)
                {
                    owned.valueStorage.emplace_back();
                    owned.nullValues.push_back(1u);
                }
                else
                {
                    owned.valueStorage.emplace_back(input.values[valueIndex].ptr, input.values[valueIndex].length);
                    owned.nullValues.push_back(0u);
                }
            }
            result->storage.push_back(std::move(owned));
        }
        for (OwnedPredicate& owned : result->storage)
        {
            for (size_t valueIndex = 0u; valueIndex < owned.valueStorage.size(); ++valueIndex)
            {
                const std::string& stringValue = owned.valueStorage[valueIndex];
                owned.values.push_back(owned.nullValues[valueIndex] != 0u
                                           ? ovx_string_t{nullptr, 0u}
                                           : ovx_string_t{stringValue.data(), stringValue.size()});
            }
            owned.value.values      = owned.values.empty() ? nullptr : owned.values.data();
            owned.value.value_count = owned.values.size();
            result->predicates.push_back(owned.value);
        }
        result->filter.predicates = result->predicates.empty() ? nullptr : result->predicates.data();
        result->filter.count      = result->predicates.size();

        /* Each token the acquisition reads, by name; the last six have no member pointer: the
           instancer's three prototype columns fill instancerPrototypes, and X, Y and Z fill
           principalAxes. usd-path also interns the path list. */
        const std::pair<const char*, ovx_token_t OwnedFilter::*> namedTokens[] = {
            {"_worldExtent", &OwnedFilter::worldExtent},
            {"points", &OwnedFilter::points},
            {"faceVertexCounts", &OwnedFilter::faceVertexCounts},
            {"faceVertexIndices", &OwnedFilter::faceVertexIndices},
            {"doubleSided", &OwnedFilter::doubleSided},
            {"omni:fabric:worldMatrix", &OwnedFilter::fabricWorldMatrix},
            {"worldMatrix", &OwnedFilter::worldMatrix},
            {"usd-prim-type", &OwnedFilter::usdPrimType},
            {"radius", &OwnedFilter::radius},
            {"orientation", &OwnedFilter::orientation},
            {"holeIndices", &OwnedFilter::holeIndices},
            {"rightHanded", &OwnedFilter::rightHanded},
            {"leftHanded", &OwnedFilter::leftHanded},
            {"size", &OwnedFilter::size},
            {"height", &OwnedFilter::height},
            {"axis", &OwnedFilter::axis},
            {"usd-schemas", &OwnedFilter::usdSchemas},
            {"physics:approximation", &OwnedFilter::physicsApproximation},
            {"PhysicsMeshCollisionAPI", &OwnedFilter::physicsMeshCollisionApi},
            {"convexHull", &OwnedFilter::convexHull},
            {"physxConvexHullCollision:hullVertexLimit", &OwnedFilter::hullVertexLimit},
            {"usd-path", &OwnedFilter::usdPath},
            {"_protoPath", &OwnedFilter::protoPath},
            {"_protoPrimType", &OwnedFilter::protoPrimType},
            {"_instancerPath", &OwnedFilter::instancerPath},
            {"prototypes", nullptr},
            {"prototypes:usd-targets", nullptr},
            {"_flattenedPrototypes", nullptr},
            {"X", nullptr},
            {"Y", nullptr},
            {"Z", nullptr},
        };
        constexpr size_t                      kTokenCount = sizeof(namedTokens) / sizeof(namedTokens[0]);
        std::array<ovx_string_t, kTokenCount> tokenStrings{};
        for (size_t index = 0u; index < kTokenCount; ++index)
        {
            tokenStrings[index] = ovx_string_t{namedTokens[index].first, std::strlen(namedTokens[index].first)};
        }
        std::array<ovx_token_t, kTokenCount> tokens{};
        const ovx_api_result_t               tokenResult = dictionary->vtable->create_tokens_from_strings(
            dictionary->context, tokenStrings.data(), tokenStrings.size(), tokens.data());
        const ovspatialquery_api_status_t tokenStatus = mapDictionaryResult(dictionary, tokenResult);
        if (tokenStatus != OVSPATIALQUERY_API_SUCCESS ||
            std::find(tokens.begin(), tokens.end(), OVX_INVALID_TOKEN) != tokens.end())
        {
            return tokenStatus == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_NOT_SUPPORTED : tokenStatus;
        }
        for (size_t index = 0u; index + 6u < kTokenCount; ++index)
        {
            (*result).*namedTokens[index].second = tokens[index];
        }
        result->instancerPrototypes = {tokens[kTokenCount - 6u], tokens[kTokenCount - 5u], tokens[kTokenCount - 4u]};
        result->principalAxes       = {tokens[kTokenCount - 3u], tokens[kTokenCount - 2u], tokens[kTokenCount - 1u]};
        std::array<ovx_token_t, 2u * kTypeCount> typeTokens{};
        const ovspatialquery_api_status_t        typeStatus = mapDictionaryResult(
            dictionary, dictionary->vtable->create_tokens_from_strings(dictionary->context, publishedTypes.data(),
                                                                              publishedTypes.size(), typeTokens.data()));
        if (typeStatus != OVSPATIALQUERY_API_SUCCESS ||
            std::find(typeTokens.begin(), typeTokens.end(), OVX_INVALID_TOKEN) != typeTokens.end())
        {
            return typeStatus == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_NOT_SUPPORTED : typeStatus;
        }
        std::copy(typeTokens.begin(), typeTokens.begin() + kTypeCount, result->publishedPrimTypes.begin());
        std::copy(typeTokens.begin() + kTypeCount, typeTokens.end(), result->publishedInstanceTypes.begin());
        result->dictionary = dictionary;
        internExactPathList(*result, result->usdPath, dictionary);
        outFilter = std::move(result);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

bool estimateFilterCharge(const OwnedFilter& filter, uint64_t& outCharge)
{
    uint64_t charge = static_cast<uint64_t>(sizeof(OwnedFilter)) + 128u;
    if (!addHostCharge(filter.storage.capacity(), sizeof(OwnedPredicate), charge) ||
        !addHostCharge(filter.predicates.capacity(), sizeof(ovstage_predicate_t), charge))
    {
        return false;
    }
    for (const OwnedPredicate& predicate : filter.storage)
    {
        if (!addHostCharge(predicate.valueStorage.capacity(), sizeof(std::string), charge) ||
            !addHostCharge(predicate.nullValues.capacity(), sizeof(uint8_t), charge) ||
            !addHostCharge(predicate.values.capacity(), sizeof(ovx_string_t), charge))
        {
            return false;
        }
        for (const std::string& value : predicate.valueStorage)
        {
            if (!addHostCharge(value.capacity(), sizeof(char), charge))
            {
                return false;
            }
        }
    }
    if (filter.scope != nullptr && !addHostCharge(1u, filter.scope->ruleCharge(), charge))
    {
        return false;
    }
    outCharge = charge;
    return true;
}

ovspatialquery_api_status_t waitAndRelease(ovstage_instance_t* stage, ovstage_op_id_t operation)
{
    if (operation == OVSTAGE_INVALID_OP_ID)
    {
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    }
    ovstage_op_wait_result_t waitResult{};
    ovstage_api_status_t     waitStatus = OVSTAGE_ERROR_TIMEOUT;
    while (waitStatus != OVSTAGE_OK && waitStatus != OVSTAGE_ERROR_OP_FAILED)
    {
        waitResult = {};
        waitStatus = stageVtable(stage)->wait_op(stageContext(stage), operation, OVSTAGE_TIMEOUT_INFINITE, &waitResult);
        if (waitStatus != OVSTAGE_OK && waitStatus != OVSTAGE_ERROR_OP_FAILED)
        {
            // A successfully enqueued operation remains owned until a terminal
            // wait result makes release_op valid.  The qualified ovstage 0.1
            // wait contract is terminal OK/OP_FAILED; retain ownership and
            // yield on every nonterminal result rather than leaking the op.
            std::this_thread::yield();
        }
    }
    const bool operationFailed =
        waitStatus == OVSTAGE_ERROR_OP_FAILED ||
        (waitResult.error_op_id_count != 0u &&
         (waitResult.error_op_ids == nullptr ||
          std::count(waitResult.error_op_ids, waitResult.error_op_ids + waitResult.error_op_id_count, operation) != 0));
    const ovstage_api_status_t releaseStatus = stageVtable(stage)->release_op(stageContext(stage), operation);
    if (operationFailed)
    {
        return OVSPATIALQUERY_API_OPERATION_FAILED;
    }
    return mapStageStatus(releaseStatus);
}

ovspatialquery_api_status_t releaseRead(ovstage_instance_t* stage, ovstage_read_handle_t& readHandle)
{
    return releaseHandle(stage, readHandle, OVSTAGE_INVALID_READ_HANDLE, stageVtable(stage)->release_read);
}

ovspatialquery_api_status_t releaseOrdinalQuery(ovstage_instance_t*             stage,
                                                ovstage_ordinal_query_handle_t& ordinalHandle)
{
    return releaseHandle(stage, ordinalHandle, OVSTAGE_INVALID_ORDINAL_QUERY_HANDLE,
                         stageVtable(stage)->release_ordinal_query);
}

ovspatialquery_api_status_t releaseQuery(ovstage_instance_t* stage, ovstage_query_handle_t& queryHandle)
{
    return releaseHandle(stage, queryHandle, OVSTAGE_INVALID_QUERY_HANDLE, stageVtable(stage)->release_query);
}

ovspatialquery_api_status_t resolveGroupPaths(const ReadGroup& group, path_dictionary_instance_t* dictionary,
                                              std::vector<ovx_primpath_t>& outPaths)
{
    if (group.prims.list == OVX_INVALID_PRIMPATH_LIST ||
        (group.prims.count != 0u && group.prims.index_map == nullptr &&
         static_cast<uint64_t>(group.prims.offset) + group.prims.count > static_cast<uint64_t>(UINT32_MAX)) ||
        (group.prims.index_map != nullptr && group.prims.offset != 0u))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    size_t                      listCount = 0u;
    ovspatialquery_api_status_t status =
        mapDictionaryResult(dictionary, dictionary->vtable->get_num_paths_from_path_list(dictionary->context,
                                                                                         group.prims.list, &listCount));
    if (status != OVSPATIALQUERY_API_SUCCESS)
    {
        return status;
    }
    /* A read's groups share one list, and Ovstage may split a column into a group per
     * prim, so read only the span the group's rows occupy: resolving the whole list for
     * each group cost a read of N single-row groups N^2 path copies (about 14 ms at 10,000). */
    const uint32_t* indexMap = group.prims.index_map;
    size_t          first    = group.prims.offset;
    size_t          end      = first + group.prims.count;
    if (indexMap != nullptr && group.prims.count != 0u)
    {
        const std::pair<const uint32_t*, const uint32_t*> span =
            std::minmax_element(indexMap, indexMap + group.prims.count);
        first = *span.first;
        end   = static_cast<size_t>(*span.second) + 1u;
    }
    std::vector<ovx_primpath_t> paths;
    if (group.prims.count != 0u)
    {
        if (end > listCount)
        {
            return OVSPATIALQUERY_API_INVALID_STATE;
        }
        paths.resize(end - first);
        size_t resolvedCount = 0u;
        status               = mapDictionaryResult(
            dictionary, dictionary->vtable->get_paths_from_path_list(dictionary->context, group.prims.list, first,
                                                                                   paths.size(), paths.data(), &resolvedCount));
        if (status != OVSPATIALQUERY_API_SUCCESS || resolvedCount != paths.size())
        {
            return status == OVSPATIALQUERY_API_SUCCESS ? OVSPATIALQUERY_API_NOT_SUPPORTED : status;
        }
    }
    if (indexMap != nullptr)
    {
        std::vector<ovx_primpath_t> gathered(group.prims.count);
        for (uint32_t row = 0u; row < group.prims.count; ++row)
        {
            gathered[row] = paths[indexMap[row] - first];
        }
        paths = std::move(gathered);
    }
    outPaths = std::move(paths);
    return OVSPATIALQUERY_API_SUCCESS;
}

namespace
{

/* Flags the group's rows `dropped` marks and counts them; no flags without `dropped`. */
ovspatialquery_api_status_t markDroppedRows(path_dictionary_instance_t*        dictionary,
                                            const std::vector<ovx_primpath_t>& paths, const DroppedRows* dropped,
                                            DroppedRowCount* outDropped, std::vector<uint8_t>& outFlags)
{
    outFlags.clear();
    if (dropped == nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    const ovspatialquery_api_status_t status = dropped->mark(dictionary, paths, outFlags);
    for (size_t row = 0u; status == OVSPATIALQUERY_API_SUCCESS && outDropped != nullptr && row < paths.size(); ++row)
    {
        outDropped->count += outFlags[row] != 0u ? 1u : 0u;
        if (outFlags[row] == kDroppedPrototypeCopy)
        {
            outDropped->first = outDropped->copies++ == 0u ? handleValue(paths[row]) : outDropped->first;
        }
    }
    return status;
}

} // namespace

ovspatialquery_api_status_t appendGroup(const ReadGroup& group, ovx_token_t expectedAttribute,
                                        ovspatialquery_ordinal_t targetOrdinal, path_dictionary_instance_t* dictionary,
                                        std::unordered_set<uint64_t>&  identities,
                                        std::vector<PopulationObject>& objects, std::vector<uint8_t>& outHasBounds,
                                        const DroppedRows* dropped, DroppedRowCount* outDropped)
{
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    if (gFailNextGroupMaterialization.exchange(false, std::memory_order_relaxed))
    {
        throw std::bad_alloc{};
    }
#endif
    if (group.attribute != expectedAttribute || group.is_delete || group.is_array ||
        group.semantic != OVSTAGE_SEMANTIC_NONE ||
        group.meta.attribute_write_floor_ordinal != static_cast<ovstage_ordinal_t>(targetOrdinal) ||
        group.ordinal > static_cast<ovstage_ordinal_t>(targetOrdinal) || group.data.tensor_count != 1u ||
        group.data.tensors == nullptr || group.data.mask != nullptr || group.data.cuda_sync.stream != 0u ||
        group.data.cuda_sync.wait_event != 0u)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const DLTensor& tensor = group.data.tensors[0];
    /* Ovstage reads the six-value tuple as a component axis, [N, 6] lanes=1, in compact rows. */
    const bool componentRows = tensor.dtype.lanes == 1u && tensor.ndim == 2 && tensor.shape != nullptr &&
                               tensor.shape[1] == 6 &&
                               (tensor.strides == nullptr || (tensor.strides[0] == 6 && tensor.strides[1] == 1));
    if (tensor.device.device_type != kDLCPU || tensor.device.device_id != 0 || tensor.dtype.code != kDLFloat ||
        tensor.dtype.bits != 64u || !componentRows || tensor.shape == nullptr || tensor.shape[0] < 0 ||
        tensor.byte_offset != 0u)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const uint64_t physicalRowCount64 = static_cast<uint64_t>(tensor.shape[0]);
    if (physicalRowCount64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const size_t     physicalRowCount = static_cast<size_t>(physicalRowCount64);
    constexpr size_t kBoundsBytes     = sizeof(double) * 6u;
    if (physicalRowCount > std::numeric_limits<size_t>::max() / kBoundsBytes ||
        (physicalRowCount != 0u && tensor.data == nullptr) ||
        (group.data.index_map != nullptr && group.data.count != group.prims.count) ||
        /* Locked ovstage 0.1 reports exact logical count here although the
         * generic fixed-row form documents zero. Qualify only both observed
         * representations; no other count is accepted without an index map. */
        (group.data.index_map == nullptr && group.data.count != 0u && group.data.count != group.prims.count) ||
        (group.data.index_map == nullptr && physicalRowCount != group.prims.count))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const uintptr_t dataAddress = reinterpret_cast<uintptr_t>(tensor.data);
    if (dataAddress > std::numeric_limits<uintptr_t>::max() - physicalRowCount * kBoundsBytes)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }

    std::vector<ovx_primpath_t> paths;
    ovspatialquery_api_status_t pathStatus = resolveGroupPaths(group, dictionary, paths);
    std::vector<uint8_t>        droppedRows;
    if (pathStatus == OVSPATIALQUERY_API_SUCCESS)
    {
        pathStatus = markDroppedRows(dictionary, paths, dropped, outDropped, droppedRows);
    }
    if (pathStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return pathStatus;
    }

    for (uint32_t logicalRow = 0u; logicalRow < group.prims.count; ++logicalRow)
    {
        if (!droppedRows.empty() && droppedRows[logicalRow] != 0u)
        {
            continue;
        }
        const size_t dataRow = group.data.index_map == nullptr ? logicalRow : group.data.index_map[logicalRow];
        if (dataRow >= physicalRowCount || paths[logicalRow] == OVX_INVALID_PRIMPATH ||
            !identities.insert(handleValue(paths[logicalRow])).second)
        {
            return OVSPATIALQUERY_API_INVALID_STATE;
        }
        double               bounds[6];
        const unsigned char* bytes = static_cast<const unsigned char*>(tensor.data) + dataRow * sizeof(bounds);
        std::memcpy(bounds, bytes, sizeof(bounds));
        /* USD's empty extent, (FLT_MAX, -FLT_MAX), marks a present _worldExtent row without a
         * value. Every other value only establishes that a spatial source exists. */
        const double emptyMinimum = static_cast<double>(std::numeric_limits<float>::max());
        const double emptyMaximum = -emptyMinimum;
        bool         missing      = true;
        for (uint32_t axis = 0u; axis < 3u; ++axis)
        {
            missing = missing && bounds[axis] == emptyMinimum && bounds[axis + 3u] == emptyMaximum;
        }
        PopulationObject object{};
        object.id = objectIdFromValue(handleValue(paths[logicalRow]));
        if (missing)
        {
            objects.push_back(object);
            outHasBounds.push_back(0u);
            continue;
        }
        objects.push_back(object);
        outHasBounds.push_back(1u);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t appendIdentityGroup(const ReadGroup& group, ovx_token_t expectedAttribute,
                                                ovspatialquery_ordinal_t       targetOrdinal,
                                                path_dictionary_instance_t*    dictionary,
                                                std::unordered_set<uint64_t>&  identities,
                                                std::vector<PopulationObject>& objects, const DroppedRows* dropped,
                                                DroppedRowCount* outDropped)
{
    if (group.attribute != expectedAttribute || group.is_delete ||
        group.meta.attribute_write_floor_ordinal != static_cast<ovstage_ordinal_t>(targetOrdinal) ||
        group.ordinal > static_cast<ovstage_ordinal_t>(targetOrdinal))
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    std::vector<ovx_primpath_t> paths;
    ovspatialquery_api_status_t pathStatus = resolveGroupPaths(group, dictionary, paths);
    std::vector<uint8_t>        droppedRows;
    if (pathStatus == OVSPATIALQUERY_API_SUCCESS)
    {
        pathStatus = markDroppedRows(dictionary, paths, dropped, outDropped, droppedRows);
    }
    if (pathStatus != OVSPATIALQUERY_API_SUCCESS)
    {
        return pathStatus;
    }
    for (uint32_t logicalRow = 0u; logicalRow < group.prims.count; ++logicalRow)
    {
        if (!droppedRows.empty() && droppedRows[logicalRow] != 0u)
        {
            continue;
        }
        if (paths[logicalRow] == OVX_INVALID_PRIMPATH || !identities.insert(handleValue(paths[logicalRow])).second)
        {
            return OVSPATIALQUERY_API_INVALID_STATE;
        }
        PopulationObject object{};
        object.id = objectIdFromValue(handleValue(paths[logicalRow]));
        objects.push_back(object);
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

} // namespace detail
} // namespace ovspatialquery

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
extern "C" void ovspatialquery_test_ovstage_fail_next_group_materialization()
{
    ovspatialquery::detail::gFailNextGroupMaterialization.store(true, std::memory_order_relaxed);
}
#endif
