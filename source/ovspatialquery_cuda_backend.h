/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_BACKEND_H
#define OVSPATIALQUERY_CUDA_BACKEND_H

#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_population_table.h"
#include "ovspatialquery_prepared_array.h"
#include "ovspatialquery_private_api.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class ExactGeometry;

enum class CudaCompletionState : uint32_t
{
    pending,
    complete,
    error,
    deviceLost,
    invalid,
};

struct CudaPopulationDescription
{
    DLDevice                    device{kDLCUDA, 0};
    ovspatialquery_ordinal_t    sourceOrdinal = 0;
    ovspatialquery_cuda_event_t ready{};
    uint64_t                    objectCount    = 0u;
    const void*                 objectIds      = nullptr;
    const void*                 objectBits     = nullptr;
    const void*                 objectMinimums = nullptr;
    const void*                 objectMaximums = nullptr;
    const void*                 objectPoses    = nullptr;
    // Prepared by device kernels (ADR-0024): no host objects exist for it.
    // `objectCount` is then the slot capacity the tables are sized for and
    // `liveObjectCount` the objects present in this version; removed and
    // unassigned slots hold inverted bounds that no query hits.
    bool        devicePrepared  = false;
    uint64_t    liveObjectCount = 0u;
    const void* geometryTypes   = nullptr;
    const void* geometryValues  = nullptr;
    // Device uint32 that is nonzero once the preparation of this version, or of
    // a version it was built on, has failed (ADR-0021). Null when the version
    // cannot fail after admission. A plan launched against the version reads it
    // after `ready` and writes the safe empty result when it is set.
    const void* preparationFailed = nullptr;
    // Device `cuda_family::DeviceBoxGrid` of a device-prepared version, or null
    // when the version has no grid and plans scan its bounds.
    const void* grid = nullptr;
    // Device uint32[changedSlotCount]: the slots whose pose or bits this
    // version's rows rewrote relative to the version it was built on, listed by
    // its prepare kernel, when the host knows the list is complete (a POSES or
    // POSES_AND_BITS device update; ADR-0019 decision 3). Null otherwise, when
    // the version's record says everything may have changed.
    // Padded: the version's rows were every slot, so the list holds ~0u for each
    // slot the update left unmoved and its length bounds the moved slots (ADR-0052).
    const void* changedSlots       = nullptr;
    uint64_t    changedSlotCount   = 0u;
    bool        changedSlotsPadded = false;
};

// What an exact CUDA plan reads to derive a readable explicit device-prepared
// version's exact objects on the device, as its host copy would hold them,
// without waiting for the version or copying it (BH-065). `membership` names
// the version's membership state, which the versions of one membership share
// (a POSES update keeps it, a membership change replaces it): the definitions
// its objects bind. Slots [0, slots) of the population's slot history were
// assigned when the version was published, and their identity, binding and
// authored shape parameters never change. The device arrays hold, per slot, those
// parameters (double[3]) and the index of its hull or mesh record (uint32),
// and the records (cuda_family::DeviceShapeRecord): a hull's prepared vertices
// and a mesh's local bounds, from which the host derives an instance's bounds.
struct CudaExactMembership
{
    std::weak_ptr<const void> membership;
    uint64_t                  slots       = 0u;
    const void*               params      = nullptr;
    const void*               definitions = nullptr;
    const void*               records     = nullptr;
};

// One convex hull or triangle mesh the slots reference, placed at the
// identity: its exact geometry supplies the local arrays every instance shares.
struct CudaExactDefinition
{
    std::shared_ptr<const ExactGeometry>         local;
    std::shared_ptr<const std::vector<uint64_t>> faceMaterialIds;
};

// The slot history's host side at `slots` slots. Its census counts the slots
// of each primitive family and, per hull or mesh definition the membership
// binds (in the order a slot first names it), the slots that bind it, with
// the definition placed at the identity; a slot whose definition the
// membership no longer binds was removed, no version of that membership holds
// it, and it is absent. Asked per slot, it gives instead each slot's public
// geometry type and its index in that definition order, UINT32_MAX for a
// sphere, box or capsule and kAbsent for an absent slot: the host memory that
// grows with the slots, which the caller charges before asking.
struct CudaExactSlots
{
    static constexpr uint32_t kAbsent = UINT32_MAX - 1u;

    uint64_t                         spheres = 0u, boxes = 0u, capsules = 0u;
    std::vector<CudaExactDefinition> definitions;       // census
    std::vector<uint64_t>            definitionSlots;   // census, parallel to `definitions`
    std::vector<int32_t>             types;             // per slot
    std::vector<uint32_t>            definitionIndices; // per slot
};

class CudaPopulationVersion
{
  public:
    virtual ~CudaPopulationVersion()                             = default;
    virtual CudaPopulationDescription description() const        = 0;
    virtual CudaCompletionState       resultState() const        = 0;
    virtual void                      releaseReadyDependencies() = 0;
    // Whether a CPU reader may ask for this device-prepared version's objects
    // (ON_DEVICE_HOST_READABLE); `hostObjects` then downloads the completed
    // version's poses and prepares its objects as the host path would, in
    // ascending ID order. NOT_SUPPORTED for a version that is not readable.
    virtual bool                        hostReadable() const noexcept { return false; }
    virtual ovspatialquery_api_status_t hostObjects(std::vector<PopulationObject>& outObjects) const
    {
        (void)outObjects;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    // A readable explicit version's membership for exact plans (see
    // CudaExactMembership); false for every other version, whose exact plans
    // read its host copy. `exactSlots` is membership-rate host work.
    virtual bool exactMembership(CudaExactMembership& outMembership) const noexcept
    {
        (void)outMembership;
        return false;
    }
    virtual ovspatialquery_api_status_t exactSlots(uint64_t slots, bool perSlot, CudaExactSlots& outSlots) const
    {
        (void)slots;
        (void)perSlot;
        (void)outSlots;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    virtual uint64_t deviceByteCharge() const = 0;
};

class CudaPopulationCapture
{
  public:
    virtual ~CudaPopulationCapture()                                    = default;
    virtual DLDevice                    device() const                  = 0;
    virtual uint64_t                    deviceByteCharge() const        = 0;
    virtual uint64_t                    deviceStagingByteCharge() const = 0;
    virtual uint64_t                    hostStagingByteCharge() const   = 0;
    virtual uint64_t                    retainedOperationByteCharge() const { return 0u; }
    virtual ovspatialquery_api_status_t prepareAdmission(const GenerationLease& base) = 0;
    virtual ovspatialquery_api_status_t captureAdmissionInputs() { return OVSPATIALQUERY_API_SUCCESS; }
    virtual bool                        materializesHostObjects() const { return false; }
    virtual bool                        requiresCpuQueryBackend() const { return false; }
    // The captured rows as host objects; an update derives them from `base`
    // and says what changed, as a CPU explicit update does (ADR-0019).
    virtual ovspatialquery_api_status_t materializeHostObjects(const GenerationLease&, ObjectTable&,
                                                               std::optional<VersionChangeRecord>&,
                                                               std::optional<uint64_t>&)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    virtual std::vector<std::shared_ptr<const void>> releaseDefinitionOwners() noexcept { return {}; }
    // A source that excludes objects reports what its publication indexed and
    // excluded, as a host source publication does; false leaves the counts to
    // the published version.
    virtual bool objectCounts(uint64_t& outIndexed, uint64_t& outExcluded) const noexcept
    {
        (void)outIndexed;
        (void)outExcluded;
        return false;
    }
    // The completed publication's exclusions, copied into operation-owned host storage.
    // False means this source has no exclusion report.
    virtual uint64_t exclusionCapacity() const noexcept { return 0u; }
    virtual bool     exclusions(std::vector<uint64_t>& outObjects, std::vector<int32_t>& outReasons) const
    {
        (void)outObjects;
        (void)outReasons;
        return false;
    }
    // The version this capture prepared became the head.
    virtual void publicationCommitted() noexcept {}
    // The publication ended without becoming the head: failed or abandoned.
    virtual void publicationAbandoned() noexcept {}
    // A version whose device work was already enqueued at admission (ADR-0021),
    // so the generation can be built and selected before the publication
    // completes; null when the version is prepared by `execute`.
    virtual std::shared_ptr<CudaPopulationVersion> admittedVersion() const { return {}; }
    // Prepares or, for an admitted version, observes the version. NOT_READY
    // means the device has not finished it yet and nothing was consumed: the
    // worker requeues the operation and serves other work instead of blocking
    // on a producer's event.
    virtual ovspatialquery_api_status_t execute(ovspatialquery_ordinal_t sourceOrdinal, const GenerationLease& base,
                                                const ObjectTable&                      objects,
                                                std::shared_ptr<CudaPopulationVersion>& outVersion) = 0;
    // After `execute` returned NOT_READY: rings `signal` once the device has
    // finished, so a worker with nothing else to do waits for that instead of
    // polling. Best effort.
    virtual void ringWhenReady(const std::shared_ptr<DeviceSignal>& signal) noexcept { (void)signal; }
};

ovspatialquery_api_status_t createCudaPopulationVersionFromHostObjects(
    DLDevice device, ovspatialquery_ordinal_t sourceOrdinal, const ObjectTable& objects,
    std::shared_ptr<CudaPopulationVersion>& outVersion);

// The CUDA module (the bundled PTX loaded into a device's primary context) an
// instance keeps loaded on each device it asked cudaBackendUnavailableReason
// about. Its plans, pair reporters and device-prepared publications take the
// same module, so the PTX loads once per device while the instance lives, however
// often those come and go: with the driver's JIT cache unusable, every load
// compiles the PTX again, which takes seconds. The modules are held type-erased,
// so this header needs no CUDA types.
class CudaModulePins
{
  public:
    // Keeps `module` as device `deviceId`'s, in place of a module kept before.
    void keep(int32_t deviceId, std::shared_ptr<void> module) noexcept;

  private:
    std::mutex                                             mMutex;
    std::vector<std::pair<int32_t, std::shared_ptr<void>>> mModules;
};

// Null when the CUDA backend can run plans on device `deviceId`; otherwise why
// not, for the NOT_SUPPORTED diagnostic of a CUDA request. Each device is
// probed once per process; the probe loads the device's shared module. When the
// device can run plans, `pins` (the requesting instance's) keeps that module,
// loading it again when no holder kept it since. Null for a negative ID, which
// the request's own validation refuses.
const char* cudaBackendUnavailableReason(int32_t deviceId, CudaModulePins* pins);
// Whether `stream`, a caller's stream for CUDA device `deviceId`, is capturing
// into a CUDA graph, which every CUDA submission refuses with NOT_SUPPORTED
// before it queues anything (ADR-0060). It asks in the device's primary
// context, where the caller's stream belongs. False when it cannot tell; the
// submission then proceeds and reports its own CUDA failure.
bool cudaStreamIsCapturing(int32_t deviceId, uintptr_t stream) noexcept;
// The probe behind cudaBackendUnavailableReason; `outModule` receives the
// device's shared module when it is available.
const char* cudaDispatchBackendUnavailableReason(int32_t deviceId, std::shared_ptr<void>& outModule);
// Device `deviceId`'s shared module in `outModule`, loaded when no holder has it;
// null when it cannot be loaded.
void cudaDispatchAcquireModule(int32_t deviceId, std::shared_ptr<void>& outModule) noexcept;
// The reason of a build without the CUDA backend; it holds for every device.
extern const char* const kCudaNoBackend;

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
// Totals read by ovspatialquery_test_get_cuda_dispatch_evidence: bytes, and
// module loads (`change` 1) and unloads (-1).
void noteCudaDeviceToHost(uint64_t bytes);
void noteCudaDeviceToDevice(uint64_t bytes);
void noteCudaModuleLoad(int change);
#endif

} // namespace detail
} // namespace ovspatialquery

#endif
