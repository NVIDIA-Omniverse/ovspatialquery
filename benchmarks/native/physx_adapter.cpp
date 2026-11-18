/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Direct pinned PhysX scene-query adapter: the raw baseline both installed paths are compared with.
#include "physx_adapter.h"

#include <iostream>

namespace bench
{
void Errors::reportError(physx::PxErrorCode::Enum code, const char* message, const char*, int)
{
    if (code == physx::PxErrorCode::ePERF_WARNING)
    {
        ++performanceWarnings;
        return;
    }
    if (code != physx::PxErrorCode::eDEBUG_INFO)
    {
        failed = true;
        std::cerr << "PhysX " << int(code) << ": " << message << '\n';
    }
}
physx::PxQueryHitType::Enum Filter::preFilter(const physx::PxFilterData&, const physx::PxShape*,
                                              const physx::PxRigidActor* actor, physx::PxHitFlags&)
{
    const auto* object = static_cast<const Object*>(actor->userData);
    if ((even && object->id % 2 != 0) || (bits && !(bits & object->bits)))
        return physx::PxQueryHitType::eNONE;
    return multiple ? physx::PxQueryHitType::eTOUCH : physx::PxQueryHitType::eBLOCK;
}
physx::PxQueryHitType::Enum Filter::postFilter(const physx::PxFilterData&, const physx::PxQueryHit&,
                                               const physx::PxShape*, const physx::PxRigidActor*)
{
    return physx::PxQueryHitType::eNONE;
}
Direct::Direct(const Fixture& f) : overlapHits(f.objects.size())
{
    foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
    require(foundation != nullptr, "PxCreateFoundation failed");
    physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, physx::PxTolerancesScale(), false);
    require(physics != nullptr, "PxCreatePhysics failed");
    {
        dispatcher = physx::PxDefaultCpuDispatcherCreate(0);
        physx::PxSceneDesc desc(physics->getTolerancesScale());
        desc.cpuDispatcher   = dispatcher;
        desc.filterShader    = physx::PxDefaultSimulationFilterShader;
        desc.staticStructure = physx::PxPruningStructureType::eSTATIC_AABB_TREE;
        scene                = physics->createScene(desc);
        material             = physics->createMaterial(0.5f, 0.5f, 0);
        require(scene && material, "PhysX scene/material creation failed");
    }
    const auto geometryStart = Clock::now();
    for (const auto& mesh : f.meshes)
    {
        physx::PxTriangleMeshDesc meshDesc;
        meshDesc.points.count     = uint32_t(mesh.vertices.size());
        meshDesc.points.stride    = sizeof(mesh.vertices[0]);
        meshDesc.points.data      = mesh.vertices.data();
        meshDesc.triangles.count  = uint32_t(mesh.triangles.size());
        meshDesc.triangles.stride = sizeof(mesh.triangles[0]);
        meshDesc.triangles.data   = mesh.triangles.data();
        physx::PxCookingParams cooking(physics->getTolerancesScale());
        auto*                  cooked = PxCreateTriangleMesh(cooking, meshDesc, physics->getPhysicsInsertionCallback());
        require(cooked != nullptr, "PhysX mesh cooking failed");
        meshes.push_back(cooked);
    }
    for (const auto& vertices : f.convexes)
    {
        physx::PxConvexMeshDesc desc;
        desc.points.count  = uint32_t(vertices.size());
        desc.points.stride = sizeof(vertices[0]);
        desc.points.data   = vertices.data();
        desc.flags         = physx::PxConvexFlag::eCOMPUTE_CONVEX | physx::PxConvexFlag::ePLANE_SHIFTING;
        desc.vertexLimit   = 255;
        desc.polygonLimit  = 255;
        physx::PxCookingParams cooking(physics->getTolerancesScale());
        auto*                  cooked = PxCreateConvexMesh(cooking, desc, physics->getPhysicsInsertionCallback());
        require(cooked != nullptr, "PhysX convex cooking failed");
        convexes.push_back(cooked);
    }
    geometrySetupMs             = elapsed(geometryStart);
    const auto publicationStart = Clock::now();
    for (const auto& object : f.objects)
    {
        auto*           actor = physics->createRigidStatic(pose(object.p, object.q));
        physx::PxShape* shape = nullptr;
        if (object.capsule)
        {
            shape = physics->createShape(physx::PxCapsuleGeometry(.5f, .625f), *material, true);
            require(shape != nullptr, "PhysX capsule creation failed");
            // PhysX capsules lie on X; the installed API uses local Y.
            shape->setLocalPose(physx::PxTransform(physx::PxQuat(physx::PxHalfPi, physx::PxVec3(0, 0, 1))));
        }
        else if (object.convex)
            shape = physics->createShape(physx::PxConvexMeshGeometry(convexes.at(size_t(object.convex - 1))), *material,
                                         true);
        else if (object.mesh)
        {
            physx::PxMeshGeometryFlags flags;
            if (f.meshes.at(size_t(object.mesh - 1)).doubleSided)
                flags |= physx::PxMeshGeometryFlag::eDOUBLE_SIDED;
            shape = physics->createShape(
                physx::PxTriangleMeshGeometry(meshes.at(size_t(object.mesh - 1)), physx::PxMeshScale(), flags),
                *material, true);
        }
        else
            shape = object.sphere ? physics->createShape(physx::PxSphereGeometry(float(sphereRadius)), *material, true)
                                  : physics->createShape(physx::PxBoxGeometry(px(boxHalf)), *material, true);
        require(actor && shape, "PhysX actor/shape creation failed");
        actor->attachShape(*shape);
        shape->release();
        actor->userData = const_cast<Object*>(&object);
        scene->addActor(*actor);
        actors.push_back(actor);
    }
    scene->flushQueryUpdates();
    require(!errors.failed, "PhysX reported setup error");
    publicationMs = elapsed(publicationStart);
}
Direct::~Direct()
{
    if (scene)
        scene->release();
    for (auto* actor : actors)
        actor->release();
    for (auto* mesh : meshes)
        mesh->release();
    for (auto* convex : convexes)
        convex->release();
    if (material)
        material->release();
    if (dispatcher)
        dispatcher->release();
    if (physics)
        physics->release();
    if (foundation)
        foundation->release();
}
uint64_t Direct::sourceFace(const Object& object, uint32_t cookedFace) const
{
    if (!object.mesh || cookedFace == UINT32_MAX)
        return UINT64_MAX;
    const auto* mesh = meshes.at(size_t(object.mesh - 1));
    require(cookedFace < mesh->getNbTriangles(), "invalid cooked face index");
    const auto* remap = mesh->getTrianglesRemap();
    return remap ? remap[cookedFace] : cookedFace;
}
void Direct::update(const Fixture& f, size_t count)
{
    for (size_t i = 0; i < count; ++i)
        actors[i]->setGlobalPose(pose(f.objects[i].p, f.objects[i].q));
    scene->flushQueryUpdates();
    require(!errors.failed, "PhysX reported update error");
}
void Direct::query(const Fixture& f, Results& results)
{
    filter.even               = f.even;
    filter.multiple           = f.kind == QueryKind::zones || f.kind == QueryKind::sphereOverlap;
    physx::PxQueryFlags flags = physx::PxQueryFlag::eSTATIC;
    if (anyOverlap(f.kind))
        flags |= physx::PxQueryFlag::eANY_HIT;
    if (filter.multiple)
        flags |= physx::PxQueryFlag::eNO_BLOCK;
    physx::PxHitFlags hitFlags = physx::PxHitFlag::ePOSITION | physx::PxHitFlag::eNORMAL;
    if (f.pointCloud)
        hitFlags = physx::PxHitFlag::ePOSITION;
    else if (!f.meshes.empty())
        hitFlags |= physx::PxHitFlag::eFACE_INDEX;
    for (size_t i = 0; i < f.queries.size(); ++i)
    {
        const auto& q    = f.queries[i];
        auto&       hits = results[i].hits;
        hits.clear();
        filter.bits                             = q.bits;
        const bool                     filtered = f.even || q.bits != 0;
        const physx::PxQueryFilterData fd(filtered ? flags | physx::PxQueryFlag::ePREFILTER : flags);
        physx::PxQueryFilterCallback*  callback = filtered ? &filter : nullptr;
        if (f.kind == QueryKind::rays)
        {
            const PxVec3           direction(q.direction[0], q.direction[1], q.direction[2]);
            const PxVec3           origin = px(q.p) + direction * float(q.nearDistance);
            physx::PxRaycastBuffer buffer;
            scene->raycast(origin, direction, float(q.farDistance - q.nearDistance), buffer, hitFlags, fd, callback);
            if (buffer.hasBlock)
            {
                const auto& h = buffer.block;
                hits.push_back(
                    {static_cast<Object*>(h.actor->userData)->id,
                     double(h.distance) + q.nearDistance,
                     {h.position.x, h.position.y, h.position.z},
                     f.pointCloud ? std::array<float, 3>{} : std::array<float, 3>{h.normal.x, h.normal.y, h.normal.z},
                     static_cast<Object*>(h.actor->userData)->geometryType(),
                     static_cast<Object*>(h.actor->userData)->geometryValue(),
                     f.pointCloud ? UINT64_MAX : sourceFace(*static_cast<Object*>(h.actor->userData), h.faceIndex)});
            }
        }
        else if (f.kind == QueryKind::sweep || f.kind == QueryKind::boxSweep)
        {
            physx::PxSweepBuffer buffer;
            const PxVec3         direction(q.direction[0], q.direction[1], q.direction[2]);
            const PxVec3         start = px(q.p) + direction * float(q.nearDistance);
            if (f.kind == QueryKind::boxSweep)
                scene->sweep(physx::PxBoxGeometry(px(f.half)),
                             physx::PxTransform(start, physx::PxQuat(q.q[0], q.q[1], q.q[2], q.q[3])), direction,
                             float(q.farDistance - q.nearDistance), buffer, hitFlags, fd, callback);
            else
                scene->sweep(physx::PxSphereGeometry(float(f.rowRadii ? q.radius : f.radius)),
                             physx::PxTransform(start), direction, float(q.farDistance - q.nearDistance), buffer,
                             hitFlags, fd, callback);
            if (buffer.hasBlock)
            {
                const auto& h      = buffer.block;
                const auto& object = *static_cast<Object*>(h.actor->userData);
                hits.push_back({object.id,
                                double(h.distance) + q.nearDistance,
                                h.distance == 0 ? Vec{} : Vec{h.position.x, h.position.y, h.position.z},
                                {h.normal.x, h.normal.y, h.normal.z},
                                object.geometryType(),
                                object.geometryValue(),
                                sourceFace(object, h.faceIndex),
                                h.distance != 0});
            }
        }
        else
        {
            physx::PxOverlapBuffer buffer(overlapHits.data(), uint32_t(overlapHits.size()));
            if (sphereShape(f.kind))
                scene->overlap(physx::PxSphereGeometry(float(f.rowRadii ? q.radius : f.radius)), pose(q.p, identity),
                               buffer, fd, callback);
            else
                scene->overlap(physx::PxBoxGeometry(px(f.half)), pose(q.p, q.q), buffer, fd, callback);
            if (anyOverlap(f.kind))
            {
                if (buffer.hasBlock || buffer.nbTouches)
                    hits.push_back(Hit{1});
            }
            else
            {
                require(!buffer.hasBlock, "multiple overlap unexpectedly returned block");
                for (uint32_t j = 0; j < buffer.nbTouches; ++j)
                    hits.push_back(Hit{static_cast<Object*>(buffer.touches[j].actor->userData)->id});
            }
        }
    }
    require(!errors.failed, "PhysX reported query error");
}
} // namespace bench
