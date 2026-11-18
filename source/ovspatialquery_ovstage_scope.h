/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_SCOPE_H
#define OVSPATIALQUERY_OVSTAGE_SCOPE_H

#include "ovspatialquery_typed_geometry.h"

#include <ovspatialquery/ovspatialquery_ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

/* An Ovstage population's or view's include and exclude path scope (ADR-0055). Rules are
 * kept as path elements, so membership compares tokens and never builds a path string.
 * Each prim's verdict is cached by path value, because it depends on nothing but the
 * path; a publication evaluates only the prims it has not seen. The cache grows in
 * charged chunks through `budget` and is cleared when it holds far more entries than the
 * prims the last call asked about, so it stays proportional to the selected prims. */
class PathScope
{
  public:
    /* Validates and copies `scope`. A scope with no entries selects everything and yields
     * a null `outScope`, so an unscoped population pays nothing. INVALID_ARGUMENT for a
     * pointer whose nullness disagrees with its count, a zero entry, or an entry the
     * dictionary cannot decompose. */
    static ovspatialquery_api_status_t create(path_dictionary_instance_t*           dictionary,
                                              const ovspatialquery_ovstage_scope_t& scope,
                                              const ConvexHullPreparationBudget&    budget,
                                              std::shared_ptr<PathScope>&           outScope);

    /* Host bytes of the copied rules, charged with the owning filter. */
    uint64_t ruleCharge() const noexcept;

    /* Sets `outInScope[i]` to 1 when the scope contains the prim at `paths[i]`, 0 when not.
     * HOST_BUDGET when the cache cannot be charged for the prims it has not seen. */
    ovspatialquery_api_status_t contains(path_dictionary_instance_t* dictionary, const uint64_t* paths, size_t count,
                                         std::vector<uint8_t>& outInScope);

  private:
    struct Rule
    {
        std::vector<ovx_token_t> elements;
        bool                     include = false;
    };

    /* The verdict of the nearest listed ancestor-or-self of a path given as elements. */
    bool decide(const ovx_token_t* elements, size_t count, std::vector<uint64_t>& prefixKeys) const;
    ovspatialquery_api_status_t reserveCache(size_t entries);

    std::vector<Rule>                         mRules;
    std::unordered_multimap<uint64_t, size_t> mRuleByKey; // prefix key -> index into mRules
    ConvexHullPreparationBudget               mBudget{};
    std::mutex                                mMutex;
    std::unordered_map<uint64_t, uint8_t>     mVerdicts;
    std::vector<std::shared_ptr<void>>        mCacheCharges;
    size_t                                    mChargedEntries = 0u;
};

} // namespace detail
} // namespace ovspatialquery

#endif
