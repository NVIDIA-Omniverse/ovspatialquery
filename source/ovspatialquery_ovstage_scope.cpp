/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_ovstage_scope.h"

#include "ovspatialquery_ovstage_filter.h"

#include <algorithm>
#include <map>
#include <new>
#include <stdexcept>

namespace ovspatialquery
{
namespace detail
{
namespace
{

constexpr size_t kBatch = 1024u;
/* One cached verdict: an unordered_map node and its bucket slot, rounded up. */
constexpr uint64_t kVerdictBytes = 48u;
constexpr uint64_t kRootKey      = 0x9e3779b97f4a7c15ull;

uint64_t extendKey(uint64_t key, ovx_token_t element) noexcept
{
    key ^= handleValue(element) + 0x9e3779b97f4a7c15ull + (key << 6) + (key >> 2);
    key *= 0xff51afd7ed558ccdull;
    return key ^ (key >> 33);
}

} // namespace

ovspatialquery_api_status_t PathScope::create(path_dictionary_instance_t*           dictionary,
                                              const ovspatialquery_ovstage_scope_t& scope,
                                              const ConvexHullPreparationBudget&    budget,
                                              std::shared_ptr<PathScope>&           outScope)
{
    outScope = nullptr;
    if ((scope.include_count == 0u) != (scope.include_paths == nullptr) ||
        (scope.exclude_count == 0u) != (scope.exclude_paths == nullptr))
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (scope.include_count == 0u && scope.exclude_count == 0u)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    try
    {
        /* One rule per distinct path; the exclude wins when a path is in both lists. */
        std::map<std::vector<ovx_token_t>, bool> rules;
        bool                                     malformed = false;
        const auto                               add = [&](const ovx_primpath_t* paths, uint32_t count, bool include)
        {
            for (uint32_t first = 0u; !malformed && first < count; first += static_cast<uint32_t>(kBatch))
            {
                const size_t batch = std::min<size_t>(kBatch, count - first);
                if (std::any_of(paths + first, paths + first + batch,
                                [](ovx_primpath_t path) { return path == OVX_INVALID_PRIMPATH; }))
                {
                    malformed = true;
                    return OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
                const ovspatialquery_api_status_t status =
                    forEachPathElements(dictionary, paths + first, batch,
                                        [&](size_t, const ovx_token_t* elements, size_t elementCount)
                                        {
                                            auto inserted = rules.emplace(
                                                std::vector<ovx_token_t>(elements, elements + elementCount), include);
                                            if (!include)
                                                inserted.first->second = false;
                                        });
                if (status != OVSPATIALQUERY_API_SUCCESS)
                {
                    malformed = true;
                    return status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? status : OVSPATIALQUERY_API_INVALID_ARGUMENT;
                }
            }
            return OVSPATIALQUERY_API_SUCCESS;
        };
        ovspatialquery_api_status_t status = add(scope.include_paths, scope.include_count, true);
        if (status == OVSPATIALQUERY_API_SUCCESS)
            status = add(scope.exclude_paths, scope.exclude_count, false);
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        /* An empty include list stands for "/", unless "/" itself is excluded. */
        if (scope.include_count == 0u)
            rules.emplace(std::vector<ovx_token_t>{}, true);

        auto result     = std::shared_ptr<PathScope>(new PathScope());
        result->mBudget = budget;
        result->mRules.reserve(rules.size());
        for (auto& entry : rules)
        {
            uint64_t key = kRootKey;
            for (ovx_token_t element : entry.first)
                key = extendKey(key, element);
            result->mRuleByKey.emplace(key, result->mRules.size());
            result->mRules.push_back(Rule{entry.first, entry.second});
        }
        outScope = std::move(result);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

uint64_t PathScope::ruleCharge() const noexcept
{
    uint64_t charge = sizeof(PathScope) + mRules.capacity() * sizeof(Rule) + mRuleByKey.size() * 32u +
                      mRuleByKey.bucket_count() * sizeof(void*);
    for (const Rule& rule : mRules)
        charge += rule.elements.capacity() * sizeof(ovx_token_t);
    return charge;
}

bool PathScope::decide(const ovx_token_t* elements, size_t count, std::vector<uint64_t>& prefixKeys) const
{
    prefixKeys.resize(count + 1u);
    prefixKeys[0] = kRootKey;
    for (size_t depth = 0u; depth < count; ++depth)
        prefixKeys[depth + 1u] = extendKey(prefixKeys[depth], elements[depth]);
    /* The nearest listed ancestor-or-self decides; each path has one rule. */
    for (size_t depth = count + 1u; depth-- > 0u;)
    {
        const auto range = mRuleByKey.equal_range(prefixKeys[depth]);
        for (auto it = range.first; it != range.second; ++it)
        {
            const Rule& rule = mRules[it->second];
            if (rule.elements.size() == depth && std::equal(rule.elements.begin(), rule.elements.end(), elements))
                return rule.include;
        }
    }
    return false;
}

ovspatialquery_api_status_t PathScope::reserveCache(size_t entries)
{
    if (entries <= mChargedEntries || mBudget.acquire == nullptr)
        return OVSPATIALQUERY_API_SUCCESS;
    const size_t          target = std::max<size_t>({entries, mChargedEntries * 2u, kBatch});
    std::shared_ptr<void> owner;
    switch (mBudget.acquire(mBudget.state, (target - mChargedEntries) * kVerdictBytes, owner))
    {
    case ConvexHullPreparationOutcome::success: break;
    case ConvexHullPreparationOutcome::resourceExhausted: return OVSPATIALQUERY_API_HOST_BUDGET;
    case ConvexHullPreparationOutcome::outOfMemory: return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    default: return OVSPATIALQUERY_API_INTERNAL;
    }
    mCacheCharges.push_back(std::move(owner));
    mChargedEntries = target;
    return OVSPATIALQUERY_API_SUCCESS;
}

ovspatialquery_api_status_t PathScope::contains(path_dictionary_instance_t* dictionary, const uint64_t* paths,
                                                size_t count, std::vector<uint8_t>& outInScope)
{
    try
    {
        outInScope.assign(count, 0u);
        const std::lock_guard<std::mutex> lock(mMutex);
        /* Prims that left keep their verdicts until the cache is far larger than what is
         * selected now; then it starts over, with its charge. */
        if (mVerdicts.size() > 2u * count + kBatch)
        {
            mVerdicts.clear();
            mVerdicts.rehash(0u);
            mCacheCharges.clear();
            mChargedEntries = 0u;
        }
        std::vector<size_t> unseen;
        for (size_t index = 0u; index < count; ++index)
        {
            const auto found = mVerdicts.find(paths[index]);
            if (found != mVerdicts.end())
                outInScope[index] = found->second;
            else
                unseen.push_back(index);
        }
        if (unseen.empty())
            return OVSPATIALQUERY_API_SUCCESS;
        ovspatialquery_api_status_t status = reserveCache(mVerdicts.size() + unseen.size());
        if (status != OVSPATIALQUERY_API_SUCCESS)
            return status;
        mVerdicts.reserve(mVerdicts.size() + unseen.size());
        std::vector<ovx_primpath_t> batch;
        std::vector<uint64_t>       prefixKeys;
        for (size_t first = 0u; status == OVSPATIALQUERY_API_SUCCESS && first < unseen.size(); first += kBatch)
        {
            const size_t size = std::min(kBatch, unseen.size() - first);
            batch.resize(size);
            for (size_t index = 0u; index < size; ++index)
                batch[index] = makeHandle<ovx_primpath_t>(paths[unseen[first + index]]);
            status = forEachPathElements(dictionary, batch.data(), size,
                                         [&](size_t index, const ovx_token_t* elements, size_t elementCount)
                                         {
                                             const uint8_t verdict =
                                                 decide(elements, elementCount, prefixKeys) ? 1u : 0u;
                                             outInScope[unseen[first + index]] = verdict;
                                             mVerdicts.emplace(handleValue(batch[index]), verdict);
                                         });
        }
        return status;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (const std::length_error&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
}

} // namespace detail
} // namespace ovspatialquery
