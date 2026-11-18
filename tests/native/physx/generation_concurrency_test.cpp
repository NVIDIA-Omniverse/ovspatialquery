/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <PxPhysicsAPI.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace
{

class EntryGate
{
  public:
    explicit EntryGate(uint32_t expectedEntrants) : mExpectedEntrants(expectedEntrants) {}

    bool waitUntilAllEntered()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        return mCondition.wait_for(lock, std::chrono::seconds(10), [this]() { return mEntered == mExpectedEntrants; });
    }

    void enterAndWait()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        ++mEntered;
        mCondition.notify_all();
        mCondition.wait(lock, [this]() { return mReleased; });
    }

    void release()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mReleased = true;
        mCondition.notify_all();
    }

  private:
    const uint32_t          mExpectedEntrants;
    uint32_t                mEntered  = 0u;
    bool                    mReleased = false;
    std::mutex              mMutex;
    std::condition_variable mCondition;
};

class CoordinatedInsertionCallback final : public physx::PxInsertionCallback
{
  public:
    CoordinatedInsertionCallback(physx::PxInsertionCallback& delegate, EntryGate& entryGate)
        : mDelegate(delegate), mEntryGate(entryGate)
    {}

    physx::PxBase* buildObjectFromData(physx::PxConcreteType::Enum type, void* data) override
    {
        mEntryGate.enterAndWait();
        return mDelegate.buildObjectFromData(type, data);
    }

  private:
    physx::PxInsertionCallback& mDelegate;
    EntryGate&                  mEntryGate;
};

class BlockingRaycastCallback final : public physx::PxBVH::RaycastCallback
{
  public:
    explicit BlockingRaycastCallback(EntryGate& entryGate) : mEntryGate(entryGate) {}

    bool reportHit(physx::PxU32 boundsIndex, physx::PxReal& distance) override
    {
        (void)boundsIndex;
        (void)distance;
        ++mHitCount;
        mEntryGate.enterAndWait();
        return true;
    }

    uint32_t hitCount() const { return mHitCount; }

  private:
    EntryGate& mEntryGate;
    uint32_t   mHitCount = 0u;
};

physx::PxBVH* createBvh(const physx::PxBounds3& bounds, physx::PxInsertionCallback& insertionCallback)
{
    physx::PxBVHDesc descriptor;
    descriptor.enlargement   = 0.0f;
    descriptor.bounds.count  = 1u;
    descriptor.bounds.stride = sizeof(physx::PxBounds3);
    descriptor.bounds.data   = &bounds;
    return PxCreateBVH(descriptor, insertionCallback);
}

bool characterizeTwoIndependentBuilds(physx::PxPhysics& physics)
{
    const physx::PxBounds3       firstBounds(physx::PxVec3(2.0f, -1.0f, -1.0f), physx::PxVec3(3.0f, 1.0f, 1.0f));
    const physx::PxBounds3       secondBounds(physx::PxVec3(4.0f, -1.0f, -1.0f), physx::PxVec3(5.0f, 1.0f, 1.0f));
    EntryGate                    insertionGate(2u);
    CoordinatedInsertionCallback insertionCallback(physics.getPhysicsInsertionCallback(), insertionGate);
    physx::PxBVH*                firstBvh  = nullptr;
    physx::PxBVH*                secondBvh = nullptr;

    std::thread firstBuild([&]() { firstBvh = createBvh(firstBounds, insertionCallback); });
    std::thread secondBuild([&]() { secondBvh = createBvh(secondBounds, insertionCallback); });

    const bool bothBuildsReachedInsertion = insertionGate.waitUntilAllEntered();
    insertionGate.release();
    firstBuild.join();
    secondBuild.join();

    const bool succeeded = bothBuildsReachedInsertion && firstBvh != nullptr && secondBvh != nullptr;
    if (firstBvh != nullptr)
    {
        firstBvh->release();
    }
    if (secondBvh != nullptr)
    {
        secondBvh->release();
    }
    return succeeded;
}

bool characterizeOldTraversalDuringNewBuild(physx::PxPhysics& physics)
{
    const physx::PxBounds3 oldBounds(physx::PxVec3(2.0f, -1.0f, -1.0f), physx::PxVec3(3.0f, 1.0f, 1.0f));
    physx::PxBVH*          oldBvh = createBvh(oldBounds, physics.getPhysicsInsertionCallback());
    if (oldBvh == nullptr)
    {
        return false;
    }

    EntryGate               traversalGate(1u);
    BlockingRaycastCallback raycastCallback(traversalGate);
    bool                    traversalCompleted = false;
    std::thread             traversal(
        [&]()
        {
            traversalCompleted = oldBvh->raycast(physx::PxVec3(0.0f, 0.0f, 0.0f), physx::PxVec3(1.0f, 0.0f, 0.0f),
                                                             10.0f, raycastCallback);
        });

    const bool                   traversalReachedCallback = traversalGate.waitUntilAllEntered();
    const physx::PxBounds3       newBounds(physx::PxVec3(6.0f, -1.0f, -1.0f), physx::PxVec3(7.0f, 1.0f, 1.0f));
    EntryGate                    insertionGate(1u);
    CoordinatedInsertionCallback insertionCallback(physics.getPhysicsInsertionCallback(), insertionGate);
    physx::PxBVH*                newBvh = nullptr;
    std::thread                  build([&]() { newBvh = createBvh(newBounds, insertionCallback); });

    const bool buildReachedInsertionWhileTraversalBlocked = insertionGate.waitUntilAllEntered();
    insertionGate.release();
    traversalGate.release();
    build.join();
    traversal.join();

    const bool succeeded = traversalReachedCallback && buildReachedInsertionWhileTraversalBlocked &&
                           traversalCompleted && raycastCallback.hitCount() == 1u && newBvh != nullptr;
    if (newBvh != nullptr)
    {
        newBvh->release();
    }
    oldBvh->release();
    return succeeded;
}

} // namespace

int main()
{
    physx::PxDefaultAllocator     allocator;
    physx::PxDefaultErrorCallback errorCallback;
    physx::PxFoundation*          foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errorCallback);
    if (foundation == nullptr)
    {
        return 1;
    }
    physx::PxPhysics* physics =
        PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale(), false, nullptr);
    if (physics == nullptr)
    {
        foundation->release();
        return 2;
    }

    const bool independentBuildsOverlap = characterizeTwoIndependentBuilds(*physics);
    const bool traversalAndBuildOverlap = characterizeOldTraversalDuringNewBuild(*physics);

    physics->release();
    foundation->release();
    if (!independentBuildsOverlap)
    {
        return 3;
    }
    return traversalAndBuildOverlap ? 0 : 4;
}
