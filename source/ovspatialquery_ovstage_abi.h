/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OVSTAGE_ABI_H
#define OVSPATIALQUERY_OVSTAGE_ABI_H

/* The companion is built against the Ovstage 0.4 headers and serves 0.3 and 0.4 stages.
 * The two releases share every function-table slot and struct the companion reads except
 * the read slots; ovstageLayout picks the read path from the stage's runtime version, and
 * ReadGroup presents a group of either release as one attribute's column. */

#include "ovspatialquery_ovstage_v03_layout.h"

#include <ovstage/ovstage_api/ovstage_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace ovspatialquery::detail
{

inline const ovstage_data_plane_vtable_t* stageVtable(const ovstage_instance_t* stage) noexcept
{
    return stage->data_plane.vtable;
}

inline ovstage_data_plane_context_t* stageContext(const ovstage_instance_t* stage) noexcept
{
    return stage->data_plane.context;
}

/* The raw value of an Ovstage or ovx handle: a token, a path, a path list or a stage handle. */
template <typename Handle>
constexpr uint64_t handleValue(Handle handle) noexcept
{
    static_assert(sizeof(Handle) == sizeof(uint64_t) && std::is_trivially_copyable_v<Handle>);
    return handle.raw_id;
}

template <typename Handle>
constexpr Handle makeHandle(uint64_t value) noexcept
{
    static_assert(sizeof(Handle) == sizeof(uint64_t) && std::is_trivially_copyable_v<Handle>);
    return Handle{value};
}

enum class OvstageLayout : uint8_t
{
    unsupported,
    v03,
    v04,
};

/* The read layout of the stage's runtime: 0.3 or 0.4, any patch; anything else is unsupported. */
inline OvstageLayout ovstageLayout(const ovstage_instance_t* stage) noexcept
{
    uint32_t major = 0u, minor = 0u, patch = 0u;
    stageVtable(stage)->get_version(stageContext(stage), &major, &minor, &patch);
    if (major != 0u)
    {
        return OvstageLayout::unsupported;
    }
    return minor == 3u ? OvstageLayout::v03 : minor == 4u ? OvstageLayout::v04 : OvstageLayout::unsupported;
}

/* Releases one fetched 0.4 group, once, when the last column view of it lets go. */
struct ReadGroupHold
{
    ovstage_instance_t*  stage = nullptr;
    ovstage_read_group_t group{};
    bool                 released = false;

    ovstage_api_status_t release() noexcept;
    ~ReadGroupHold();
};

/* One attribute's column of a fetched read group, in either release's layout. A 0.3 group
 * holds one attribute; a 0.4 group holds one column per attribute the read requested, and
 * each becomes its own ReadGroup that shares the group's hold. */
struct ReadGroup
{
    ovx_token_t                  attribute{};
    ovstage_ordinal_t            ordinal{};
    bool                         is_delete{};
    bool                         is_array{};
    ovstage_attribute_semantic_t semantic{};
    ovstage_prim_group_t         prims{};
    ovstage_data_t               data{};
    ovstage_attribute_meta_t     meta{};

    OvstageLayout                           layout = OvstageLayout::unsupported;
    ovspatialquery_ovstage_v03_read_group_t v03{};
    std::shared_ptr<ReadGroupHold>          v04;
};

/* Fetches a read's groups one attribute column at a time, as 0.3's fetch_read_next does. */
class ReadCursor
{
  public:
    explicit ReadCursor(ovstage_instance_t* stage) noexcept : mStage(stage) {}

    /* The next column of `read`; END_OF_ITERATION and errors as fetch_read_next reports them. */
    ovstage_api_status_t next(ovstage_read_handle_t read, ovstage_timeout_ns_t timeout, ReadGroup& outGroup);

  private:
    ovstage_instance_t*            mStage;
    ovstage_read_handle_t          mRead{};
    std::shared_ptr<ReadGroupHold> mPending;
    size_t                         mNextColumn = 0u;
};

/* Releases a column view; a 0.4 group is released with its last view, whose status this
 * returns, and earlier views succeed. A view no cursor delivered succeeds unchanged. */
ovstage_api_status_t releaseReadGroup(ovstage_instance_t* stage, ReadGroup& group);

} // namespace ovspatialquery::detail

#endif
