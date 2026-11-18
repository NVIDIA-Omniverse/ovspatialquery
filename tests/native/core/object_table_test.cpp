/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_object_table.h"
#include "test_failure.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

// Object tables shared between versions (ADR-0019): a derived table reads as
// its base with the written rows, owns only the chunks those rows fall in,
// leaves its base unchanged, and keeps each batch's lease until the last table
// sharing it is gone.

namespace
{

using ovspatialquery::detail::ObjectTable;
using ovspatialquery::detail::PopulationObject;

PopulationObject object(uint64_t id, double x)
{
    PopulationObject value{};
    value.id.value = id;
    value.bounds   = {{x, 0.0, 0.0}, {x + 1.0, 1.0, 1.0}};
    return value;
}

// Each table's objects against the expected ids and x positions.
void requireReads(const ObjectTable& table, const std::vector<PopulationObject>& expected)
{
    REQUIRE(table.size() == expected.size());
    size_t index = 0u;
    for (const PopulationObject& value : table)
    {
        REQUIRE(value.id.value == expected[index].id.value &&
                value.bounds.minimum[0] == expected[index].bounds.minimum[0] && &table[index] == &value);
        ++index;
    }
    REQUIRE(index == expected.size());
}

} // namespace

int main()
{
    constexpr size_t chunk = ObjectTable::chunkSize;
    // 2.5 chunks, so the last chunk is partial.
    std::vector<PopulationObject> objects;
    for (uint64_t id = 1u; id <= chunk * 5u / 2u; ++id)
    {
        objects.push_back(object(id, static_cast<double>(id)));
    }
    std::unique_ptr<ObjectTable> base(new ObjectTable(objects));
    REQUIRE(base->contiguousData() != nullptr &&
            base->ownBatchBytes() == ObjectTable::contiguousBatchBytes(objects.size()));
    std::weak_ptr<int> baseLeaseAlive;
    {
        std::shared_ptr<int> lease = std::make_shared<int>(1);
        baseLeaseAlive             = lease;
        base->attachBatchLease(std::move(lease));
    }
    REQUIRE(base->ownBatchBytes() == 0u);
    requireReads(*base, objects);

    // Rows in the first chunk, the partial last chunk and appended after it.
    std::vector<PopulationObject> expected = objects;
    std::vector<ObjectTable::Row> rows;
    for (const size_t position : {size_t{3u}, objects.size() - 1u, objects.size(), objects.size() + 1u})
    {
        PopulationObject value = object(position < objects.size() ? objects[position].id.value : 1000u + position,
                                        -static_cast<double>(position));
        rows.push_back({static_cast<uint32_t>(position), value});
        if (position < expected.size())
        {
            expected[position] = value;
        }
        else
        {
            expected.push_back(value);
        }
    }
    ObjectTable derived = ObjectTable::derive(*base, rows);
    REQUIRE(derived.contiguousData() == nullptr);
    // Two chunks were copied: the first and the partial last, which the appended rows extend.
    REQUIRE(derived.ownBatchBytes() == ObjectTable::contiguousBatchBytes(2u * chunk));
    requireReads(derived, expected);
    requireReads(*base, objects);
    // An untouched chunk is shared, not copied.
    REQUIRE(&derived[chunk] == &(*base)[chunk]);
    REQUIRE(&derived[3u] != &(*base)[3u]);

    // A derived table of a derived table, appending a new chunk.
    std::vector<ObjectTable::Row> more;
    std::vector<PopulationObject> expectedMore = expected;
    for (size_t position = expected.size(); position < chunk * 4u; ++position)
    {
        more.push_back({static_cast<uint32_t>(position), object(5000u + position, static_cast<double>(position))});
        expectedMore.push_back(more.back().object);
    }
    ObjectTable further = ObjectTable::derive(derived, more);
    requireReads(further, expectedMore);
    requireReads(derived, expected);

    // Rows must be sorted, unique and contiguous past the end.
    bool rejected = false;
    try
    {
        (void)ObjectTable::derive(*base, {{5u, object(6u, 0.0)}, {5u, object(6u, 1.0)}});
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    REQUIRE(rejected);
    rejected = false;
    try
    {
        (void)ObjectTable::derive(*base, {{static_cast<uint32_t>(objects.size() + 1u), object(9000u, 0.0)}});
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    REQUIRE(rejected);

    // The base's lease lives while a derived table shares one of its chunks.
    derived.attachBatchLease(std::make_shared<int>(2));
    REQUIRE(derived.ownBatchBytes() == 0u);
    base.reset();
    REQUIRE(!baseLeaseAlive.expired());
    derived = ObjectTable();
    REQUIRE(!baseLeaseAlive.expired()); // `further` still shares base chunks
    further = ObjectTable();
    REQUIRE(baseLeaseAlive.expired());
    // A replaced chunk's objects let go of what they hold once no table reads
    // that chunk, while another chunk of the same batch is still shared; the
    // batch's lease is released after its objects.
    {
        std::vector<PopulationObject> owned;
        for (uint64_t id = 1u; id <= 2u * chunk; ++id)
        {
            owned.push_back(object(id, static_cast<double>(id)));
            owned.back().definitionOwner = std::make_shared<const int>(0);
        }
        const std::weak_ptr<const void> replaced(owned[0].definitionOwner);
        const std::weak_ptr<const void> kept(owned[chunk].definitionOwner);
        struct LeaseProbe
        {
            std::weak_ptr<const void> object;
            bool*                     releasedAfterObjects;
            ~LeaseProbe() { *releasedAfterObjects = object.expired(); }
        };
        bool        releasedAfterObjects = false;
        ObjectTable original(std::move(owned));
        original.attachBatchLease(std::make_shared<LeaseProbe>(LeaseProbe{kept, &releasedAfterObjects}));
        ObjectTable replacing = ObjectTable::derive(original, {{0u, object(1u, -1.0)}});
        original              = ObjectTable();
        REQUIRE(replaced.expired() && !kept.expired() && !releasedAfterObjects);
        REQUIRE(replacing[chunk].definitionOwner != nullptr && replacing[0].bounds.minimum[0] == -1.0);
        replacing = ObjectTable();
        REQUIRE(kept.expired() && releasedAfterObjects);
    }
    // A contiguous table that alone holds its batch frees it at once, still
    // releasing the lease after the objects.
    {
        std::vector<PopulationObject> owned(3u * chunk, object(1u, 0.0));
        owned.back().definitionOwner = std::make_shared<const int>(0);
        struct LeaseProbe
        {
            std::weak_ptr<const void> object;
            bool*                     releasedAfterObjects;
            ~LeaseProbe() { *releasedAfterObjects = object.expired(); }
        };
        bool                            releasedAfterObjects = false;
        const std::weak_ptr<const void> last(owned.back().definitionOwner);
        {
            ObjectTable alone(std::move(owned));
            alone.attachBatchLease(std::make_shared<LeaseProbe>(LeaseProbe{last, &releasedAfterObjects}));
            REQUIRE(!last.expired());
        }
        REQUIRE(last.expired() && releasedAfterObjects);
    }
    std::puts("object table test passed");
    return 0;
}
