/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Direct pinned PhysX scene-query adapter: the raw baseline both installed paths are compared with.
#ifndef BENCH_PHYSX_ADAPTER_H
#define BENCH_PHYSX_ADAPTER_H
#include "fixture.h"

#include <vector>

namespace bench
{
struct Errors : physx::PxErrorCallback
{
    bool   failed              = false;
    size_t performanceWarnings = 0;
    void   reportError(physx::PxErrorCode::Enum code, const char* message, const char*, int) override;
};
struct Filter : physx::PxQueryFilterCallback
{
    bool                        even = false, multiple = false;
    uint64_t                    bits = 0;
    physx::PxQueryHitType::Enum preFilter(const physx::PxFilterData&, const physx::PxShape*,
                                          const physx::PxRigidActor* actor, physx::PxHitFlags&) override;
    physx::PxQueryHitType::Enum postFilter(const physx::PxFilterData&, const physx::PxQueryHit&, const physx::PxShape*,
                                           const physx::PxRigidActor*) override;
};
class Direct
{
    physx::PxDefaultAllocator           allocator;
    Errors                              errors;
    physx::PxFoundation*                foundation = nullptr;
    physx::PxPhysics*                   physics    = nullptr;
    physx::PxDefaultCpuDispatcher*      dispatcher = nullptr;
    physx::PxMaterial*                  material   = nullptr;
    physx::PxScene*                     scene      = nullptr;
    std::vector<physx::PxRigidStatic*>  actors;
    std::vector<physx::PxOverlapHit>    overlapHits;
    Filter                              filter;
    std::vector<physx::PxTriangleMesh*> meshes;
    std::vector<physx::PxConvexMesh*>   convexes;

  public:
    double geometrySetupMs = 0, publicationMs = 0;
    explicit Direct(const Fixture& f);
    ~Direct();
    uint64_t sourceFace(const Object& object, uint32_t cookedFace) const;
    size_t   warningCount() const { return errors.performanceWarnings; }
    void     update(const Fixture& f, size_t count);
    void     query(const Fixture& f, Results& results);
};
} // namespace bench
#endif
