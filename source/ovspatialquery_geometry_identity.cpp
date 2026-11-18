/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_geometry_identity.h"
#include "ovspatialquery_internal.h"
#include "ovspatialquery_object_table.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace ovspatialquery
{
namespace detail
{
namespace
{

constexpr uint64_t kMapNodeAllowance = 128u;

double normalizedScalar(double value) noexcept { return value == 0.0 ? 0.0 : value; }

void normalizeZeros(std::vector<float>& values) noexcept
{
    for (float& value : values)
    {
        if (value == 0.0f)
        {
            value = 0.0f;
        }
    }
}

template <typename Value>
uint64_t addVectorCapacity(uint64_t total, const std::vector<Value>& values)
{
    return saturatingResidentByteAdd(total, static_cast<uint64_t>(values.capacity()),
                                     static_cast<uint64_t>(sizeof(Value)));
}

bool validFaces(size_t pointCount, const std::vector<uint32_t>& counts, const std::vector<uint32_t>& indices) noexcept
{
    if (pointCount == 0u || counts.empty())
    {
        return false;
    }
    size_t expectedIndexCount = 0u;
    for (uint32_t count : counts)
    {
        if (count < 3u || static_cast<size_t>(count) > std::numeric_limits<size_t>::max() - expectedIndexCount)
        {
            return false;
        }
        expectedIndexCount += static_cast<size_t>(count);
    }
    if (expectedIndexCount != indices.size())
    {
        return false;
    }
    return std::all_of(indices.begin(), indices.end(),
                       [pointCount](uint32_t index) { return static_cast<size_t>(index) < pointCount; });
}

bool normalizeDefinition(GeometryDefinitionValue& definition)
{
    return std::visit(
        [](auto& value)
        {
            using Value = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<Value, SphereDefinitionValue>)
            {
                value.radius = normalizedScalar(value.radius);
                return std::isfinite(value.radius) && value.radius > 0.0;
            }
            else if constexpr (std::is_same_v<Value, BoxDefinitionValue>)
            {
                value.size = normalizedScalar(value.size);
                return std::isfinite(value.size) && value.size > 0.0;
            }
            else if constexpr (std::is_same_v<Value, CapsuleDefinitionValue>)
            {
                value.radius = normalizedScalar(value.radius);
                value.height = normalizedScalar(value.height);
                return std::isfinite(value.radius) && value.radius > 0.0 && std::isfinite(value.height) &&
                       value.height > 0.0 && value.axis >= CapsuleAxis::x && value.axis <= CapsuleAxis::z;
            }
            else
            {
                normalizeZeros(value.points);
                for (size_t index = 0u; index < value.points.size(); ++index)
                {
                    const double point = value.points[index];
                    if (!std::isfinite(point))
                    {
                        return false;
                    }
                }
                if (value.points.empty() || value.points.size() % 3u != 0u)
                {
                    return false;
                }
                const bool hasNoSourceFaces = value.faceVertexCounts.empty() && value.faceVertexIndices.empty();
                if ((!std::is_same_v<Value, ConvexDefinitionValue> || !hasNoSourceFaces) &&
                    !validFaces(value.points.size() / 3u, value.faceVertexCounts, value.faceVertexIndices))
                {
                    return false;
                }
                if constexpr (std::is_same_v<Value, MeshDefinitionValue>)
                {
                    size_t               triangleCount = 0u;
                    std::vector<uint8_t> isHole;
                    if (!value.holeIndices.empty())
                    {
                        isHole.assign(value.faceVertexCounts.size(), 0u);
                        for (uint32_t hole : value.holeIndices)
                            if (hole < isHole.size())
                                isHole[hole] = 1u;
                    }
                    for (size_t face = 0; face < value.faceVertexCounts.size(); ++face)
                    {
                        if (!isHole.empty() && isHole[face] != 0u)
                            continue;
                        const uint32_t count        = value.faceVertexCounts[face];
                        const size_t   contribution = static_cast<size_t>(count - 2u);
                        if (contribution > std::numeric_limits<size_t>::max() - triangleCount)
                        {
                            return false;
                        }
                        triangleCount += contribution;
                    }
                    if (value.sourceFaces.size() != triangleCount ||
                        triangleCount > std::numeric_limits<size_t>::max() / 3u ||
                        value.sourceCorners.size() != triangleCount * 3u)
                    {
                        return false;
                    }
                    for (size_t triangle = 0u; triangle < triangleCount; ++triangle)
                    {
                        const uint64_t face = value.sourceFaces[triangle];
                        if (static_cast<size_t>(face) >= value.faceVertexCounts.size())
                        {
                            return false;
                        }
                        const uint32_t cornerCount = value.faceVertexCounts[face];
                        for (uint32_t corner = 0u; corner < 3u; ++corner)
                        {
                            if (value.sourceCorners[triangle * 3u + corner] >= cornerCount)
                            {
                                return false;
                            }
                        }
                    }
                }
                return true;
            }
        },
        definition);
}

struct DefinitionLess
{
    bool operator()(const GeometryDefinitionValue& first, const GeometryDefinitionValue& second) const noexcept
    {
        if (first.index() != second.index())
        {
            return first.index() < second.index();
        }
        return std::visit(
            [](const auto& left, const auto& right) noexcept
            {
                using Left  = std::decay_t<decltype(left)>;
                using Right = std::decay_t<decltype(right)>;
                if constexpr (!std::is_same_v<Left, Right>)
                {
                    return false;
                }
                else
                {
                    const int metricOrder = std::memcmp(left.shapeMetric.data(), right.shapeMetric.data(),
                                                        left.shapeMetric.size() * sizeof(double));
                    if (metricOrder != 0)
                        return metricOrder < 0;
                    if (left.mirrored != right.mirrored)
                        return left.mirrored < right.mirrored;
                    if constexpr (std::is_same_v<Left, SphereDefinitionValue>)
                    {
                        return left.radius < right.radius;
                    }
                    else if constexpr (std::is_same_v<Left, BoxDefinitionValue>)
                    {
                        return left.size < right.size;
                    }
                    else if constexpr (std::is_same_v<Left, CapsuleDefinitionValue>)
                    {
                        return std::tie(left.radius, left.height, left.axis) <
                               std::tie(right.radius, right.height, right.axis);
                    }
                    else if constexpr (std::is_same_v<Left, ConvexDefinitionValue>)
                    {
                        return std::tie(left.hullVertexLimit, left.points, left.faceVertexCounts,
                                        left.faceVertexIndices) < std::tie(right.hullVertexLimit, right.points,
                                                                           right.faceVertexCounts,
                                                                           right.faceVertexIndices);
                    }
                    else
                    {
                        return std::tie(left.leftHanded, left.holeIndices, left.points, left.faceVertexCounts,
                                        left.faceVertexIndices, left.sourceFaces, left.sourceCorners,
                                        left.doubleSided) < std::tie(right.leftHanded, right.holeIndices, right.points,
                                                                     right.faceVertexCounts, right.faceVertexIndices,
                                                                     right.sourceFaces, right.sourceCorners,
                                                                     right.doubleSided);
                    }
                }
            },
            first, second);
    }
};

struct RegistryEntry
{
    GeometryReference reference{};
    /* The prepared geometries that carry this definition; empty means the definition
     * stays for the registry's lifetime. */
    std::vector<std::weak_ptr<const void>> owners;
    /* Bytes charged for the owner slots; never below what `owners` has allocated, and
     * released with the entry. */
    uint64_t ownerBytes = 0u;

    bool expired() const noexcept
    {
        if (owners.empty())
        {
            return false;
        }
        for (const std::weak_ptr<const void>& owner : owners)
        {
            if (!owner.expired())
            {
                return false;
            }
        }
        return true;
    }

    bool records(const std::shared_ptr<const void>& owner) const noexcept
    {
        for (const std::weak_ptr<const void>& recorded : owners)
        {
            if (!recorded.expired() && !recorded.owner_before(owner) && !owner.owner_before(recorded))
            {
                return true;
            }
        }
        return false;
    }

    size_t liveOwnerCount() const noexcept
    {
        return static_cast<size_t>(std::count_if(
            owners.begin(), owners.end(), [](const std::weak_ptr<const void>& owner) { return !owner.expired(); }));
    }

    /* Charges the slots `owners` has allocated beyond what is already charged. */
    uint64_t chargeOwnerSlots() noexcept
    {
        const uint64_t allocated = saturatingResidentByteAdd(0u, static_cast<uint64_t>(owners.capacity()),
                                                             static_cast<uint64_t>(sizeof(std::weak_ptr<const void>)));
        const uint64_t added     = allocated > ownerBytes ? allocated - ownerBytes : 0u;
        ownerBytes               = std::max(ownerBytes, allocated);
        return added;
    }
};

/* Owners a staging adds to a committed entry. They join it at commit, with the slots
 * charged at staging, so a staging that is abandoned leaves the entry as it was. */
struct PendingOwners
{
    std::vector<std::shared_ptr<const void>> owners;
    uint64_t                                 ownerBytes = 0u;

    bool records(const std::shared_ptr<const void>& owner) const noexcept
    {
        for (const std::shared_ptr<const void>& pending : owners)
        {
            if (!pending.owner_before(owner) && !owner.owner_before(pending))
            {
                return true;
            }
        }
        return false;
    }
};

using RegistryMap = std::map<GeometryDefinitionValue, RegistryEntry, DefinitionLess>;

uint64_t nodeResidentByteCount(const GeometryDefinitionValue& definition) noexcept
{
    uint64_t total             = static_cast<uint64_t>(sizeof(RegistryMap::value_type));
    total                      = saturatingResidentByteAdd(total, 1u, kMapNodeAllowance);
    const uint64_t allocations = geometryDefinitionRetainedByteCount(definition);
    return allocations == std::numeric_limits<uint64_t>::max() ? allocations
                                                               : saturatingResidentByteAdd(total, allocations, 1u);
}

struct GeometryReferenceClaimRecord
{
    std::shared_ptr<void> memoryOwner;
    uint64_t              pendingToken = 0u;
};

constexpr uint64_t kClaimNodeAllowance = static_cast<uint64_t>(sizeof(GeometryReference)) +
                                         static_cast<uint64_t>(sizeof(GeometryReferenceClaimRecord)) +
                                         6u * static_cast<uint64_t>(sizeof(void*));

} // namespace

struct GeometryReferenceClaimDirectory::Impl
{
    mutable std::mutex                                                               mutex;
    std::map<GeometryReference, GeometryReferenceClaimRecord, GeometryReferenceLess> claims;
    std::map<GeometryReference, GeometryReferenceClaimRecord, GeometryReferenceLess> pendingClaims;
    uint64_t                                                                         nextPendingToken = 1u;
};

struct GeometryReferenceClaimDirectory::Batch::Impl
{
    explicit Impl(std::shared_ptr<GeometryReferenceClaimDirectory::Impl> ownerValue) : owner(std::move(ownerValue)) {}

    ~Impl()
    {
        if (owner == nullptr || committed || pendingToken == 0u)
        {
            return;
        }
        std::lock_guard<std::mutex> ownerLock(owner->mutex);
        for (GeometryReference reference : references)
        {
            const auto found = owner->pendingClaims.find(reference);
            if (found != owner->pendingClaims.end() && found->second.pendingToken == pendingToken)
            {
                owner->pendingClaims.erase(found);
            }
        }
    }

    std::shared_ptr<GeometryReferenceClaimDirectory::Impl> owner;
    std::vector<GeometryReference>                         references;
    uint64_t                                               pendingToken            = 0u;
    uint64_t                                               additionalResidentBytes = 0u;
    uint64_t                                               stagingResidentBytes    = 0u;
    bool                                                   committed               = false;
};

GeometryReferenceClaimDirectory::Batch::Batch()                                                             = default;
GeometryReferenceClaimDirectory::Batch::~Batch()                                                            = default;
GeometryReferenceClaimDirectory::Batch::Batch(Batch&&) noexcept                                             = default;
GeometryReferenceClaimDirectory::Batch& GeometryReferenceClaimDirectory::Batch::operator=(Batch&&) noexcept = default;

uint64_t GeometryReferenceClaimDirectory::Batch::additionalResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->additionalResidentBytes;
}

uint64_t GeometryReferenceClaimDirectory::Batch::stagingResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->stagingResidentBytes;
}

GeometryReferenceClaimDirectory::GeometryReferenceClaimDirectory() : mImpl(std::make_shared<Impl>()) {}

GeometryReferenceClaimDirectory::~GeometryReferenceClaimDirectory() = default;

uint64_t GeometryReferenceClaimDirectory::fixedResidentByteCount() noexcept
{
    // InstanceState owns the directory through make_shared, while the
    // directory owns its state through a second make_shared allocation.
    return static_cast<uint64_t>(sizeof(GeometryReferenceClaimDirectory)) + static_cast<uint64_t>(sizeof(Impl)) +
           6u * static_cast<uint64_t>(sizeof(void*));
}

GeometryIdentityOutcome GeometryReferenceClaimDirectory::stage(const std::vector<GeometryReference>& references,
                                                               Batch&                                outBatch) noexcept
{
    outBatch = Batch{};
    if (references.empty())
    {
        return GeometryIdentityOutcome::invalidArgument;
    }
    try
    {
        std::unique_ptr<Batch::Impl> candidate(new Batch::Impl(mImpl));
        candidate->references.reserve(references.size());
        std::set<GeometryReference, GeometryReferenceLess> uniqueReferences;
        for (GeometryReference reference : references)
        {
            if (reference.value == 0u || !uniqueReferences.insert(reference).second)
            {
                return GeometryIdentityOutcome::invalidArgument;
            }
            candidate->references.push_back(reference);
        }
        candidate->additionalResidentBytes =
            saturatingResidentByteAdd(0u, static_cast<uint64_t>(candidate->references.size()), kClaimNodeAllowance);
        candidate->stagingResidentBytes = saturatingResidentByteAdd(
            static_cast<uint64_t>(sizeof(Batch::Impl)), static_cast<uint64_t>(candidate->references.capacity()),
            static_cast<uint64_t>(sizeof(GeometryReference)));
        if (candidate->additionalResidentBytes == UINT64_MAX || candidate->stagingResidentBytes == UINT64_MAX)
        {
            return GeometryIdentityOutcome::resourceExhausted;
        }
        std::lock_guard<std::mutex> lock(mImpl->mutex);
        if (mImpl->nextPendingToken == 0u)
        {
            return GeometryIdentityOutcome::resourceExhausted;
        }
        for (GeometryReference reference : candidate->references)
        {
            if (mImpl->claims.find(reference) != mImpl->claims.end() ||
                mImpl->pendingClaims.find(reference) != mImpl->pendingClaims.end())
            {
                return GeometryIdentityOutcome::invalidArgument;
            }
        }
        candidate->pendingToken = mImpl->nextPendingToken++;
        try
        {
            for (GeometryReference reference : candidate->references)
            {
                mImpl->pendingClaims.emplace(reference, GeometryReferenceClaimRecord{{}, candidate->pendingToken});
            }
        }
        catch (...)
        {
            for (GeometryReference reference : candidate->references)
            {
                const auto found = mImpl->pendingClaims.find(reference);
                if (found != mImpl->pendingClaims.end() && found->second.pendingToken == candidate->pendingToken)
                {
                    mImpl->pendingClaims.erase(found);
                }
            }
            throw;
        }
        outBatch.mImpl = std::move(candidate);
        return GeometryIdentityOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return GeometryIdentityOutcome::outOfMemory;
    }
    catch (...)
    {
        return GeometryIdentityOutcome::internalFailure;
    }
}

void GeometryReferenceClaimDirectory::commit(Batch& batch, std::shared_ptr<void> memoryOwner) noexcept
{
    if (batch.mImpl == nullptr || batch.mImpl->owner.get() != mImpl.get() || batch.mImpl->committed)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (GeometryReference reference : batch.mImpl->references)
    {
        const auto found = mImpl->pendingClaims.find(reference);
        if (found == mImpl->pendingClaims.end() || found->second.pendingToken != batch.mImpl->pendingToken)
        {
            return;
        }
    }
    for (GeometryReference reference : batch.mImpl->references)
    {
        auto node                  = mImpl->pendingClaims.extract(reference);
        node.mapped().memoryOwner  = memoryOwner;
        node.mapped().pendingToken = 0u;
        mImpl->claims.insert(std::move(node));
    }
    batch.mImpl->committed = true;
}

uint64_t GeometryReferenceClaimDirectory::release(GeometryReference reference) noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    const auto                  found = mImpl->claims.find(reference);
    if (found == mImpl->claims.end())
    {
        return 0u;
    }
    mImpl->claims.erase(found);
    return kClaimNodeAllowance;
}

size_t GeometryReferenceClaimDirectory::claimCount() const noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    return mImpl->claims.size();
}

struct GeometryIdentityRegistry::Impl
{
    explicit Impl(std::shared_ptr<GeometryReferenceClaimDirectory> claimsValue) : claims(std::move(claimsValue)) {}

    mutable std::mutex                               mutex;
    std::shared_ptr<GeometryReferenceClaimDirectory> claims;
    RegistryMap                                      entries;
    uint64_t                                         nextValue     = kFirstAdapterGeometryValue;
    uint64_t                                         residentBytes = 0u;
};

struct GeometryIdentityRegistry::Batch::Impl
{
    explicit Impl(std::shared_ptr<GeometryIdentityRegistry::Impl> ownerValue, std::unique_lock<std::mutex> ownerLock)
        : owner(std::move(ownerValue)), lock(std::move(ownerLock))
    {}

    std::shared_ptr<GeometryIdentityRegistry::Impl> owner;
    std::unique_lock<std::mutex>                    lock;
    RegistryMap                                     newEntries;
    std::vector<GeometryReference>                  references;
    std::map<RegistryEntry*, PendingOwners>         ownerAdditions;
    uint64_t                                        nextValue               = kFirstAdapterGeometryValue;
    uint64_t                                        additionalResidentBytes = 0u;
    uint64_t                                        stagingResidentBytes    = 0u;
    GeometryReferenceClaimDirectory::Batch          claimBatch;
    bool                                            committed = false;
};

bool operator==(const GeometryReference& first, const GeometryReference& second) noexcept
{
    return first.family == second.family && first.value == second.value;
}

bool operator!=(const GeometryReference& first, const GeometryReference& second) noexcept { return !(first == second); }

GeometryIdentityRegistry::Batch::Batch()                                                            = default;
GeometryIdentityRegistry::Batch::~Batch()                                                           = default;
GeometryIdentityRegistry::Batch::Batch(Batch&& other) noexcept                                      = default;
GeometryIdentityRegistry::Batch& GeometryIdentityRegistry::Batch::operator=(Batch&& other) noexcept = default;

const std::vector<GeometryReference>& GeometryIdentityRegistry::Batch::references() const noexcept
{
    static const std::vector<GeometryReference> empty;
    return mImpl == nullptr ? empty : mImpl->references;
}

uint64_t GeometryIdentityRegistry::Batch::additionalResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->additionalResidentBytes;
}

uint64_t GeometryIdentityRegistry::Batch::stagingResidentByteCount() const noexcept
{
    return mImpl == nullptr ? 0u : mImpl->stagingResidentBytes;
}

uint64_t GeometryIdentityRegistry::Batch::releasedResidentByteCount() const noexcept { return mReleasedResidentBytes; }

uint64_t GeometryIdentityRegistry::Batch::releaseStaging() noexcept
{
    if (mImpl == nullptr)
    {
        return 0u;
    }
    const uint64_t                 released = mImpl->stagingResidentBytes;
    std::vector<GeometryReference> empty;
    mImpl->references.swap(empty);
    mImpl->stagingResidentBytes = 0u;
    return released;
}

GeometryIdentityRegistry::GeometryIdentityRegistry()
    : GeometryIdentityRegistry(std::make_shared<GeometryReferenceClaimDirectory>())
{}

GeometryIdentityRegistry::GeometryIdentityRegistry(std::shared_ptr<GeometryReferenceClaimDirectory> claimDirectory)
    : mImpl(std::make_shared<Impl>(std::move(claimDirectory)))
{}

GeometryIdentityRegistry::~GeometryIdentityRegistry() = default;

uint64_t GeometryIdentityRegistry::fixedResidentByteCount() noexcept
{
    // make_shared owns one combined Impl/control-block allocation. The
    // implementation payload is exact; three pointer widths conservatively
    // cover the implementation-defined control block.
    return static_cast<uint64_t>(sizeof(Impl)) + 3u * static_cast<uint64_t>(sizeof(void*));
}

GeometryIdentityOutcome GeometryIdentityRegistry::stage(std::vector<GeometryDefinitionValue>&           definitions,
                                                        Batch&                                          outBatch,
                                                        const std::vector<std::shared_ptr<const void>>* owners) noexcept
{
    outBatch = Batch{};
    if (definitions.empty() || (owners != nullptr && owners->size() != definitions.size()))
    {
        return GeometryIdentityOutcome::invalidArgument;
    }
    try
    {
        std::unique_ptr<Batch::Impl> candidate(new Batch::Impl(mImpl, std::unique_lock<std::mutex>(mImpl->mutex)));
        candidate->references.reserve(definitions.size());
        candidate->nextValue = mImpl->nextValue;
        /* Definitions nothing carries any more leave now, before the lookups, so a
         * source republishing new content each time keeps the registry at the size of
         * what its retained generations still hold. */
        for (RegistryMap::iterator entry = mImpl->entries.begin(); entry != mImpl->entries.end();)
        {
            if (!entry->second.expired())
            {
                ++entry;
                continue;
            }
            const uint64_t nodeBytes =
                saturatingResidentByteAdd(nodeResidentByteCount(entry->first), entry->second.ownerBytes, 1u);
            const uint64_t claimBytes = mImpl->claims->release(entry->second.reference);
            const uint64_t released   = saturatingResidentByteAdd(nodeBytes, claimBytes, 1u);
            mImpl->residentBytes      = mImpl->residentBytes > released ? mImpl->residentBytes - released : 0u;
            // Retirement is permanent even if a later definition cannot be staged.
            outBatch.mReleasedResidentBytes = saturatingResidentByteAdd(outBatch.mReleasedResidentBytes, released, 1u);
            entry                           = mImpl->entries.erase(entry);
        }
        for (size_t index = 0u; index < definitions.size(); ++index)
        {
            // Normalized in place and moved into the registry when it is new there: a
            // large mesh definition is not copied a second time.
            GeometryDefinitionValue&          definition = definitions[index];
            const std::shared_ptr<const void> owner =
                owners != nullptr ? (*owners)[index] : std::shared_ptr<const void>();
            if (!normalizeDefinition(definition))
            {
                return GeometryIdentityOutcome::invalidArgument;
            }
            const RegistryMap::iterator existing = mImpl->entries.find(definition);
            if (existing != mImpl->entries.end())
            {
                RegistryEntry& entry = existing->second;
                if (owner != nullptr && !entry.owners.empty() && !entry.records(owner))
                {
                    PendingOwners& pending = candidate->ownerAdditions[&entry];
                    if (!pending.records(owner))
                    {
                        pending.owners.push_back(owner);
                        // Commit drops the released owners, so the live ones and these fit.
                        const uint64_t needed = saturatingResidentByteAdd(
                            0u, static_cast<uint64_t>(entry.liveOwnerCount() + pending.owners.size()),
                            static_cast<uint64_t>(sizeof(std::weak_ptr<const void>)));
                        const uint64_t charged = std::max(entry.ownerBytes, pending.ownerBytes);
                        if (needed > charged)
                        {
                            candidate->additionalResidentBytes =
                                saturatingResidentByteAdd(candidate->additionalResidentBytes, needed - charged, 1u);
                            pending.ownerBytes = needed;
                        }
                    }
                }
                candidate->references.push_back(entry.reference);
                continue;
            }
            const RegistryMap::iterator staged = candidate->newEntries.find(definition);
            if (staged != candidate->newEntries.end())
            {
                RegistryEntry& entry = staged->second;
                if (owner != nullptr && !entry.owners.empty() && !entry.records(owner))
                {
                    entry.owners.push_back(owner);
                    candidate->additionalResidentBytes =
                        saturatingResidentByteAdd(candidate->additionalResidentBytes, entry.chargeOwnerSlots(), 1u);
                }
                candidate->references.push_back(entry.reference);
                continue;
            }
            if (candidate->nextValue == 0u)
            {
                return GeometryIdentityOutcome::resourceExhausted;
            }
            const GeometryReference reference{geometryDefinitionFamily(definition), candidate->nextValue};
            RegistryEntry           entry;
            entry.reference = reference;
            if (owner != nullptr)
            {
                entry.owners.push_back(owner);
            }
            const std::pair<RegistryMap::iterator, bool> inserted =
                candidate->newEntries.emplace(std::move(definition), std::move(entry));
            if (!inserted.second)
            {
                return GeometryIdentityOutcome::internalFailure;
            }
            const uint64_t nodeBytes = saturatingResidentByteAdd(nodeResidentByteCount(inserted.first->first),
                                                                 inserted.first->second.chargeOwnerSlots(), 1u);
            if (nodeBytes == std::numeric_limits<uint64_t>::max() ||
                candidate->additionalResidentBytes > std::numeric_limits<uint64_t>::max() - nodeBytes)
            {
                return GeometryIdentityOutcome::resourceExhausted;
            }
            candidate->additionalResidentBytes += nodeBytes;
            candidate->references.push_back(reference);
            if (candidate->nextValue == std::numeric_limits<uint64_t>::max())
            {
                candidate->nextValue = 0u;
            }
            else
            {
                ++candidate->nextValue;
            }
        }
        // Guards the registry's count against the definition bytes; commit adds
        // them together with the claim batch's bytes.
        if (candidate->additionalResidentBytes > std::numeric_limits<uint64_t>::max() - mImpl->residentBytes)
        {
            return GeometryIdentityOutcome::resourceExhausted;
        }
        std::vector<GeometryReference> newReferences;
        newReferences.reserve(candidate->newEntries.size());
        for (const RegistryMap::value_type& entry : candidate->newEntries)
        {
            newReferences.push_back(entry.second.reference);
        }
        if (!newReferences.empty())
        {
            const GeometryIdentityOutcome claimOutcome = mImpl->claims->stage(newReferences, candidate->claimBatch);
            if (claimOutcome != GeometryIdentityOutcome::success)
            {
                return claimOutcome;
            }
            candidate->additionalResidentBytes = saturatingResidentByteAdd(
                candidate->additionalResidentBytes, candidate->claimBatch.additionalResidentByteCount(), 1u);
        }
        candidate->stagingResidentBytes =
            saturatingResidentByteAdd(0u, static_cast<uint64_t>(candidate->references.capacity()),
                                      static_cast<uint64_t>(sizeof(GeometryReference)));
        outBatch.mImpl = std::move(candidate);
        return GeometryIdentityOutcome::success;
    }
    catch (const std::bad_alloc&)
    {
        return GeometryIdentityOutcome::outOfMemory;
    }
    catch (const std::length_error&)
    {
        return GeometryIdentityOutcome::outOfMemory;
    }
    catch (...)
    {
        return GeometryIdentityOutcome::internalFailure;
    }
}

void GeometryIdentityRegistry::findMeshDefinitions(
    std::unordered_map<uint64_t, const MeshDefinitionValue*>& inOutByReference) const noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    for (const RegistryMap::value_type& entry : mImpl->entries)
    {
        const MeshDefinitionValue* mesh = std::get_if<MeshDefinitionValue>(&entry.first);
        const std::unordered_map<uint64_t, const MeshDefinitionValue*>::iterator wanted =
            mesh == nullptr ? inOutByReference.end() : inOutByReference.find(entry.second.reference.value);
        if (wanted != inOutByReference.end())
        {
            wanted->second = mesh;
        }
    }
}

void GeometryIdentityRegistry::commit(Batch& batch) noexcept
{
    if (batch.mImpl == nullptr || batch.mImpl->owner.get() != mImpl.get() || batch.mImpl->committed ||
        !batch.mImpl->lock.owns_lock())
    {
        return;
    }
    for (std::pair<RegistryEntry* const, PendingOwners>& addition : batch.mImpl->ownerAdditions)
    {
        RegistryEntry& entry = *addition.first;
        entry.ownerBytes     = std::max(entry.ownerBytes, addition.second.ownerBytes);
        if (addition.second.owners.empty())
        {
            continue;
        }
        // The entry gains an owner, so dropping its released ones never leaves it unowned.
        entry.owners.erase(std::remove_if(entry.owners.begin(), entry.owners.end(),
                                          [](const std::weak_ptr<const void>& owner) { return owner.expired(); }),
                           entry.owners.end());
        try
        {
            entry.owners.reserve(entry.owners.size() + addition.second.owners.size());
        }
        catch (...)
        {
            /* Unrecorded owners only let the definition leave earlier; content staged
             * again then takes a new reference value, as the header allows. */
            if (entry.owners.empty())
            {
                entry.owners.emplace_back();
            }
            continue;
        }
        entry.owners.insert(entry.owners.end(), addition.second.owners.begin(), addition.second.owners.end());
    }
    // A committed batch the caller keeps must not keep the owners alive.
    batch.mImpl->ownerAdditions.clear();
    mImpl->entries.merge(batch.mImpl->newEntries);
    if (batch.mImpl->claimBatch.additionalResidentByteCount() != 0u)
    {
        mImpl->claims->commit(batch.mImpl->claimBatch);
    }
    mImpl->nextValue = batch.mImpl->nextValue;
    mImpl->residentBytes += batch.mImpl->additionalResidentBytes;
    batch.mImpl->committed = true;
    batch.mImpl->lock.unlock();
}

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
bool GeometryIdentityRegistry::inspect(GeometryIdentityRegistryInspection& outInspection) const noexcept
{
    std::lock_guard<std::mutex> lock(mImpl->mutex);
    outInspection = {
        static_cast<uint64_t>(mImpl->entries.size()),
        mImpl->nextValue,
        mImpl->residentBytes,
    };
    return true;
}
#endif

ExactGeometryFamily geometryDefinitionFamily(const GeometryDefinitionValue& definition) noexcept
{
    static constexpr ExactGeometryFamily families[] = {
        ExactGeometryFamily::sphere, ExactGeometryFamily::box,          ExactGeometryFamily::capsule,
        ExactGeometryFamily::convex, ExactGeometryFamily::triangleMesh,
    };
    return families[definition.index()];
}

uint64_t geometryDefinitionRetainedByteCount(const GeometryDefinitionValue& definition) noexcept
{
    return std::visit(
        [](const auto& value) noexcept
        {
            using Value    = std::decay_t<decltype(value)>;
            uint64_t total = 0u;
            if constexpr (std::is_same_v<Value, ConvexDefinitionValue> || std::is_same_v<Value, MeshDefinitionValue>)
            {
                total = addVectorCapacity(total, value.points);
                total = addVectorCapacity(total, value.faceVertexCounts);
                total = addVectorCapacity(total, value.faceVertexIndices);
            }
            if constexpr (std::is_same_v<Value, MeshDefinitionValue>)
            {
                total = addVectorCapacity(total, value.sourceFaces);
                total = addVectorCapacity(total, value.sourceCorners);
                total = addVectorCapacity(total, value.holeIndices);
            }
            return total;
        },
        definition);
}

uint64_t typedGeometryStagingResidentByteCount(
    const std::vector<PopulationObject>& objects, const std::vector<GeometryDefinitionValue>& definitions,
    const std::vector<std::shared_ptr<const PreparedConvexHull>>& preparedConvexHulls) noexcept
{
    uint64_t total = saturatingResidentByteAdd(0u, static_cast<uint64_t>(objects.capacity()),
                                               static_cast<uint64_t>(sizeof(PopulationObject)));
    total          = saturatingResidentByteAdd(total, static_cast<uint64_t>(definitions.capacity()),
                                               static_cast<uint64_t>(sizeof(GeometryDefinitionValue)));
    total          = saturatingResidentByteAdd(total, static_cast<uint64_t>(preparedConvexHulls.capacity()),
                                               static_cast<uint64_t>(sizeof(std::shared_ptr<const PreparedConvexHull>)));
    for (const GeometryDefinitionValue& definition : definitions)
    {
        const uint64_t definitionBytes = geometryDefinitionRetainedByteCount(definition);
        total                          = saturatingResidentByteAdd(total, definitionBytes, 1u);
    }
    const uint64_t kernelBytes = exactGeometryResidentByteCount(objects);
    return saturatingResidentByteAdd(total, kernelBytes, 1u);
}

std::vector<const PreparedTriangleMesh*> typedGeometryTriangleMeshes(
    const std::vector<PopulationObject>&                          objects,
    const std::vector<std::shared_ptr<const PreparedConvexHull>>& preparedConvexHulls)
{
    std::vector<const PreparedTriangleMesh*> meshes;
    for (const PopulationObject& object : objects)
    {
        collectTriangleMesh(object.triangleGeometry.get(), meshes);
        if (object.exactGeometry != nullptr)
            collectTriangleMesh(object.exactGeometry->triangleSurface(), meshes);
    }
    for (const std::shared_ptr<const PreparedConvexHull>& hull : preparedConvexHulls)
        if (hull != nullptr)
            collectTriangleMesh(hull->preparedSurface().get(), meshes);
    return meshes;
}

namespace
{

template <typename Objects>
uint64_t countExactGeometryResidentBytes(const Objects& objects) noexcept
{
    try
    {
        ExactGeometryResidentAccumulator accumulator;
        for (const PopulationObject& object : objects)
        {
            if (!accumulateExactGeometryResidentAllocations(accumulator, object))
            {
                return std::numeric_limits<uint64_t>::max();
            }
        }
        return accumulator.residentByteCount();
    }
    catch (...)
    {
        return std::numeric_limits<uint64_t>::max();
    }
}

struct ObjectSpan
{
    const PopulationObject* first;
    const PopulationObject* last;

    const PopulationObject* begin() const { return first; }
    const PopulationObject* end() const { return last; }
};

} // namespace

uint64_t exactGeometryResidentByteCount(const std::vector<PopulationObject>& objects) noexcept
{
    return countExactGeometryResidentBytes(objects);
}

uint64_t exactGeometryResidentByteCount(const ObjectTable& objects) noexcept
{
    const PopulationObject* contiguous = objects.contiguousData();
    return contiguous != nullptr ? countExactGeometryResidentBytes(ObjectSpan{contiguous, contiguous + objects.size()})
                                 : countExactGeometryResidentBytes(objects);
}

bool accumulateExactGeometryResidentAllocations(ExactGeometryResidentAccumulator& accumulator,
                                                const PopulationObject&           object) noexcept
{
    if (object.ovstageShape && !accumulator.add(object.ovstageShape.get(), sizeof(OvstageShapeTransform)))
        return false;
    // An Ovstage mesh owns its face material table: charge its capacity and a shared-owner
    // allowance, as an explicit definition charges the table its instances share.
    constexpr uint64_t kSharedOwnerAllowance = 128u;
    if (object.faceMaterialIds != nullptr && object.definitionOwner == nullptr &&
        !accumulator.add(object.faceMaterialIds.get(),
                         static_cast<uint64_t>(object.faceMaterialIds->capacity()) * sizeof(uint64_t) +
                             kSharedOwnerAllowance))
        return false;
    if (object.exactGeometry != nullptr)
    {
        object.exactGeometry->accumulateResidentAllocations(accumulator, object.geometryKey == 0u);
        return accumulator.residentByteCount() != std::numeric_limits<uint64_t>::max();
    }
    if (object.triangleGeometry == nullptr)
    {
        return true;
    }
    if (!accumulator.add(object.triangleGeometry.get(), sizeof(TriangleObjectGeometry)))
    {
        return false;
    }
    return object.geometryKey != 0u || object.triangleGeometry->mesh == nullptr ||
           accumulator.add(object.triangleGeometry->mesh.get(),
                           triangleGeometryResidentByteCount(*object.triangleGeometry));
}

} // namespace detail
} // namespace ovspatialquery
