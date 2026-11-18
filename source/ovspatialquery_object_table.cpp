/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_object_table.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>

namespace ovspatialquery
{
namespace detail
{

// Objects in chunks with one reference count each. The lease is declared
// first so that it is destroyed after the objects.
struct ObjectTable::Batch
{
    std::shared_ptr<void>                    lease;
    std::vector<PopulationObject>            objects;
    std::unique_ptr<std::atomic<uint32_t>[]> chunkReferences;
    std::atomic<size_t>                      liveChunks{0u};
};

namespace
{

size_t chunkCountFor(size_t count) noexcept { return (count + ObjectTable::chunkSize - 1u) >> ObjectTable::chunkShift; }

} // namespace

ObjectTable::ChunkRef::ChunkRef(Batch* batch, size_t chunk) noexcept
    : mData(batch->objects.data() + chunk * chunkSize), mBatch(batch), mChunk(chunk)
{
    mBatch->chunkReferences[mChunk].fetch_add(1u, std::memory_order_relaxed);
}

ObjectTable::ChunkRef::ChunkRef(const ChunkRef& other) noexcept
    : mData(other.mData), mBatch(other.mBatch), mChunk(other.mChunk)
{
    if (mBatch != nullptr)
    {
        mBatch->chunkReferences[mChunk].fetch_add(1u, std::memory_order_relaxed);
    }
}

ObjectTable::ChunkRef::ChunkRef(ChunkRef&& other) noexcept
    : mData(other.mData), mBatch(other.mBatch), mChunk(other.mChunk)
{
    other.mData  = nullptr;
    other.mBatch = nullptr;
}

ObjectTable::ChunkRef& ObjectTable::ChunkRef::operator=(ChunkRef other) noexcept
{
    std::swap(mData, other.mData);
    std::swap(mBatch, other.mBatch);
    std::swap(mChunk, other.mChunk);
    return *this;
}

ObjectTable::ChunkRef::~ChunkRef() { release(); }

void ObjectTable::ChunkRef::release() noexcept
{
    if (mBatch == nullptr || mBatch->chunkReferences[mChunk].fetch_sub(1u, std::memory_order_acq_rel) != 1u)
    {
        return;
    }
    // No table reads this chunk any more: its objects let go of what they hold.
    const size_t first = mChunk * chunkSize;
    const size_t last  = std::min(first + chunkSize, mBatch->objects.size());
    for (size_t index = first; index < last; ++index)
    {
        mBatch->objects[index] = PopulationObject{};
    }
    if (mBatch->liveChunks.fetch_sub(1u, std::memory_order_acq_rel) == 1u)
    {
        delete mBatch;
    }
}

ObjectTable::ObjectTable(std::vector<PopulationObject> objects)
{
    std::unique_ptr<Batch> batch(new Batch());
    const size_t           chunkCount = chunkCountFor(objects.size());
    batch->chunkReferences.reset(new std::atomic<uint32_t>[chunkCount]());
    batch->objects = std::move(objects);
    mSize          = batch->objects.size();
    mContiguous    = batch->objects.data();
    mChunks.reserve(chunkCount);
    if (chunkCount == 0u)
    {
        return;
    }
    batch->liveChunks.store(chunkCount, std::memory_order_relaxed);
    Batch* const owned = batch.release();
    for (size_t chunk = 0u; chunk < chunkCount; ++chunk)
    {
        mChunks.emplace_back(owned, chunk);
    }
    mOwnBatch = owned;
}

ObjectTable::~ObjectTable()
{
    // A contiguous table that alone references every chunk of its batch
    // frees the batch in one pass instead of resetting chunk by chunk.
    if (mContiguous == nullptr || mChunks.empty())
    {
        return;
    }
    Batch* const batch = mChunks.front().batch();
    if (batch == nullptr || batch->liveChunks.load(std::memory_order_acquire) != mChunks.size())
    {
        return;
    }
    for (size_t chunk = 0u; chunk < mChunks.size(); ++chunk)
    {
        if (mChunks[chunk].batch() != batch || batch->chunkReferences[chunk].load(std::memory_order_acquire) != 1u)
        {
            return;
        }
    }
    for (ChunkRef& reference : mChunks)
    {
        reference.forget();
    }
    delete batch;
}

ObjectTable::ObjectTable(ObjectTable&& other) noexcept
    : mSize(other.mSize), mContiguous(other.mContiguous), mChunks(std::move(other.mChunks)), mOwnBatch(other.mOwnBatch)
{
    other.mSize       = 0u;
    other.mContiguous = nullptr;
    other.mChunks.clear();
    other.mOwnBatch = nullptr;
}

ObjectTable& ObjectTable::operator=(ObjectTable&& other) noexcept
{
    if (this != &other)
    {
        mSize             = other.mSize;
        mContiguous       = other.mContiguous;
        mChunks           = std::move(other.mChunks);
        mOwnBatch         = other.mOwnBatch;
        other.mSize       = 0u;
        other.mContiguous = nullptr;
        other.mChunks.clear();
        other.mOwnBatch = nullptr;
    }
    return *this;
}

ObjectTable ObjectTable::derive(const ObjectTable& base, std::vector<Row> rows)
{
    ObjectTable table;
    table.mSize = base.mSize;
    for (size_t index = 0u; index < rows.size(); ++index)
    {
        const size_t position = rows[index].position;
        if ((index != 0u && position <= rows[index - 1u].position) || position > table.mSize)
        {
            throw std::invalid_argument("object table rows are not sorted, unique and contiguous");
        }
        table.mSize = std::max(table.mSize, position + 1u);
    }
    std::vector<size_t> touched;
    for (const Row& row : rows)
    {
        const size_t chunk = row.position >> chunkShift;
        if (touched.empty() || touched.back() != chunk)
        {
            touched.push_back(chunk);
        }
    }
    std::unique_ptr<Batch> batch(new Batch());
    batch->objects.resize(touched.size() * chunkSize);
    batch->chunkReferences.reset(new std::atomic<uint32_t>[touched.size()]());
    for (size_t index = 0u; index < touched.size(); ++index)
    {
        const size_t chunk = touched[index];
        if (chunk < base.mChunks.size())
        {
            const size_t count = std::min(chunkSize, base.mSize - chunk * chunkSize);
            std::copy(base.mChunks[chunk].data(), base.mChunks[chunk].data() + count,
                      batch->objects.data() + index * chunkSize);
        }
    }
    size_t cursor = 0u;
    for (Row& row : rows)
    {
        const size_t chunk = row.position >> chunkShift;
        while (touched[cursor] != chunk)
        {
            ++cursor;
        }
        batch->objects[cursor * chunkSize + (row.position & (chunkSize - 1u))] = std::move(row.object);
    }
    table.mChunks = base.mChunks;
    table.mChunks.resize(chunkCountFor(table.mSize));
    if (touched.empty())
    {
        return table;
    }
    batch->liveChunks.store(touched.size(), std::memory_order_relaxed);
    Batch* const owned = batch.release();
    for (size_t index = 0u; index < touched.size(); ++index)
    {
        table.mChunks[touched[index]] = ChunkRef(owned, index);
    }
    table.mOwnBatch = owned;
    return table;
}

uint64_t ObjectTable::contiguousBatchBytes(size_t count) noexcept
{
    return static_cast<uint64_t>(sizeof(Batch)) + static_cast<uint64_t>(count) * sizeof(PopulationObject) +
           static_cast<uint64_t>(chunkCountFor(count)) * sizeof(std::atomic<uint32_t>);
}

uint64_t ObjectTable::ownBatchBytes() const noexcept
{
    return mOwnBatch == nullptr ? 0u : contiguousBatchBytes(mOwnBatch->objects.capacity());
}

uint64_t ObjectTable::chunkEntryBytes() const noexcept
{
    return static_cast<uint64_t>(mChunks.capacity()) * sizeof(ChunkRef);
}

void ObjectTable::attachBatchLease(std::shared_ptr<void> lease)
{
    if (mOwnBatch != nullptr)
    {
        mOwnBatch->lease = std::move(lease);
        mOwnBatch        = nullptr;
    }
}

std::vector<PopulationObject> ObjectTable::toVector() const { return std::vector<PopulationObject>(begin(), end()); }

} // namespace detail
} // namespace ovspatialquery
