/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_OBJECT_TABLE_H
#define OVSPATIALQUERY_OBJECT_TABLE_H

#include "ovspatialquery_internal.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

// A version's objects, by position (ADR-0019), in chunks of `chunkSize`
// objects that versions share. A table built from a complete object list is
// one contiguous batch; a table derived from a base shares the base's chunks
// and owns copies only of the chunks its changed and appended objects fall
// in. Chunks are counted individually: when no table references a chunk its
// objects are reset, releasing their geometry and definitions, even while
// other chunks of the batch live. Each batch may carry a lease that charges
// its storage to a budget; it is released after the storage, with the batch's
// last chunk.
class ObjectTable
{
  public:
    static constexpr size_t chunkShift = 5u;
    static constexpr size_t chunkSize  = size_t{1u} << chunkShift;

    // One changed or appended object and its position.
    struct Row
    {
        uint32_t         position = 0u;
        PopulationObject object;
    };

    class const_iterator
    {
      public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = PopulationObject;
        using difference_type   = std::ptrdiff_t;
        using pointer           = const PopulationObject*;
        using reference         = const PopulationObject&;

        const_iterator() = default;
        const_iterator(const ObjectTable* table, size_t index) : mTable(table), mIndex(index) {}

        reference       operator*() const { return (*mTable)[mIndex]; }
        pointer         operator->() const { return &(*mTable)[mIndex]; }
        reference       operator[](difference_type offset) const { return (*mTable)[mIndex + offset]; }
        const_iterator& operator++()
        {
            ++mIndex;
            return *this;
        }
        const_iterator operator++(int)
        {
            const_iterator previous = *this;
            ++mIndex;
            return previous;
        }
        const_iterator& operator--()
        {
            --mIndex;
            return *this;
        }
        const_iterator operator--(int)
        {
            const_iterator previous = *this;
            --mIndex;
            return previous;
        }
        const_iterator& operator+=(difference_type offset)
        {
            mIndex = static_cast<size_t>(static_cast<difference_type>(mIndex) + offset);
            return *this;
        }
        const_iterator&        operator-=(difference_type offset) { return *this += -offset; }
        friend const_iterator  operator+(const_iterator iterator, difference_type offset) { return iterator += offset; }
        friend const_iterator  operator+(difference_type offset, const_iterator iterator) { return iterator += offset; }
        friend const_iterator  operator-(const_iterator iterator, difference_type offset) { return iterator -= offset; }
        friend difference_type operator-(const const_iterator& left, const const_iterator& right)
        {
            return static_cast<difference_type>(left.mIndex) - static_cast<difference_type>(right.mIndex);
        }
        friend bool operator==(const const_iterator& left, const const_iterator& right)
        {
            return left.mIndex == right.mIndex;
        }
        friend bool operator!=(const const_iterator& left, const const_iterator& right) { return !(left == right); }
        friend bool operator<(const const_iterator& left, const const_iterator& right)
        {
            return left.mIndex < right.mIndex;
        }
        friend bool operator>(const const_iterator& left, const const_iterator& right) { return right < left; }
        friend bool operator<=(const const_iterator& left, const const_iterator& right) { return !(right < left); }
        friend bool operator>=(const const_iterator& left, const const_iterator& right) { return !(left < right); }

      private:
        const ObjectTable* mTable = nullptr;
        size_t             mIndex = 0u;
    };

    ObjectTable() = default;
    // One contiguous batch holding `objects`.
    ObjectTable(std::vector<PopulationObject> objects);
    ObjectTable(const ObjectTable& other) = default;
    ObjectTable(ObjectTable&& other) noexcept;
    ObjectTable& operator=(const ObjectTable& other) = default;
    ObjectTable& operator=(ObjectTable&& other) noexcept;
    ~ObjectTable();

    // `base` with `rows` written at their positions, which are sorted, unique
    // and either inside `base` or appended contiguously after it. The new
    // chunks form one batch, which the caller charges (ownBatchBytes) before
    // attaching the lease.
    static ObjectTable derive(const ObjectTable& base, std::vector<Row> rows);

    // Resident bytes a table built from `count` objects holds in its batch.
    static uint64_t contiguousBatchBytes(size_t count) noexcept;

    // Charges the storage this table added over its base: the lease is
    // released with the last table sharing the batch. Call once, before the
    // table is shared.
    void attachBatchLease(std::shared_ptr<void> lease);
    // Resident bytes of the batch this table created, until its lease is
    // attached, and of its own chunk entries.
    uint64_t ownBatchBytes() const noexcept;
    uint64_t chunkEntryBytes() const noexcept;

    size_t                  size() const noexcept { return mSize; }
    bool                    empty() const noexcept { return mSize == 0u; }
    const PopulationObject& operator[](size_t index) const noexcept
    {
        return mContiguous != nullptr ? mContiguous[index]
                                      : mChunks[index >> chunkShift].data()[index & (chunkSize - 1u)];
    }
    const PopulationObject& front() const noexcept { return (*this)[0u]; }
    const_iterator          begin() const noexcept { return const_iterator(this, 0u); }
    const_iterator          end() const noexcept { return const_iterator(this, mSize); }
    // The objects as one array when the table is a single contiguous batch.
    const PopulationObject*       contiguousData() const noexcept { return mContiguous; }
    std::vector<PopulationObject> toVector() const;

  private:
    struct Batch;

    // One reference to one chunk of a batch.
    class ChunkRef
    {
      public:
        ChunkRef() = default;
        ChunkRef(Batch* batch, size_t chunk) noexcept;
        ChunkRef(const ChunkRef& other) noexcept;
        ChunkRef(ChunkRef&& other) noexcept;
        ChunkRef& operator=(ChunkRef other) noexcept;
        ~ChunkRef();

        const PopulationObject* data() const noexcept { return mData; }
        Batch*                  batch() const noexcept { return mBatch; }
        // Drops the reference without counting it down; the caller frees the batch.
        void forget() noexcept { mBatch = nullptr; }

      private:
        void release() noexcept;

        const PopulationObject* mData  = nullptr;
        Batch*                  mBatch = nullptr;
        size_t                  mChunk = 0u;
    };

    size_t                  mSize       = 0u;
    const PopulationObject* mContiguous = nullptr;
    std::vector<ChunkRef>   mChunks;
    // The batch this table created, until its lease is attached; the table's
    // own chunk references keep it alive.
    Batch* mOwnBatch = nullptr;
};

// exactGeometryResidentByteCount of the table's objects.
uint64_t exactGeometryResidentByteCount(const ObjectTable& objects) noexcept;

} // namespace detail
} // namespace ovspatialquery

#endif
