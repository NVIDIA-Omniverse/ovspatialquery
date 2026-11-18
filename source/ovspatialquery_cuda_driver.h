/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_CUDA_DRIVER_H
#define OVSPATIALQUERY_CUDA_DRIVER_H

#include "ovspatialquery_cuda_backend.h"
#include "ovspatialquery_cuda_failure_points.h"
#include "ovspatialquery_device_signal.h"
#include "ovspatialquery_input_layout.h"

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND) && defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

// Shared CUDA driver-API vocabulary and the dlopen-style loader used by every
// split of the former ovspatialquery_cuda_dispatch.cpp translation unit.
// These were internal-linkage (anonymous namespace) members of that single
// file; giving them namespace-scope external linkage here is what lets the
// batch dispatch, population-version, and family-plan translation units
// share one `struct Driver` shape and one set of CUDA result constants.

using CuResult    = int;
using CuDevice    = int;
using CuContext   = void*;
using CuModule    = void*;
using CuFunction  = void*;
using CuStream    = void*;
using CuEvent     = void*;
using CuDevicePtr = unsigned long long;

constexpr CuResult     kCuSuccess              = 0;
constexpr CuResult     kCuOutOfMemory          = 2;
constexpr CuResult     kCuInvalidDevice        = 101;
constexpr CuResult     kCuInvalidContext       = 201;
constexpr CuResult     kCuInvalidHandle        = 400;
constexpr CuResult     kCuNotReady             = 600;
constexpr CuResult     kCuDeviceLost           = 719;
constexpr CuResult     kCuCaptureImplicit      = 906;
constexpr unsigned int kCuStreamNonBlocking    = 1u;
constexpr unsigned int kCuEventBlockingSync    = 1u;
constexpr unsigned int kCuEventDisableTiming   = 2u;
constexpr unsigned int kCuMemHostAllocPortable = 1u;
constexpr unsigned int kCuMemHostAllocMapped   = 2u;
constexpr int          kCuLimitStackSize       = 0;
constexpr int          kCuFuncLocalSizeBytes   = 3;
constexpr int          kCuMemoryTypeDevice     = 2;

// cuMemcpy2DAsync's descriptor (CUDA_MEMCPY2D).
struct CuMemcpy2D
{
    size_t      srcXInBytes   = 0u;
    size_t      srcY          = 0u;
    int         srcMemoryType = 0;
    const void* srcHost       = nullptr;
    CuDevicePtr srcDevice     = 0u;
    void*       srcArray      = nullptr;
    size_t      srcPitch      = 0u;
    size_t      dstXInBytes   = 0u;
    size_t      dstY          = 0u;
    int         dstMemoryType = 0;
    void*       dstHost       = nullptr;
    CuDevicePtr dstDevice     = 0u;
    void*       dstArray      = nullptr;
    size_t      dstPitch      = 0u;
    size_t      widthInBytes  = 0u;
    size_t      height        = 0u;
};

// The failure-point test hook (ovspatialquery_cuda_dispatch.cpp) arms this
// state and the catalogue family plan reads it through the three helpers
// below, so it needs one definition with external linkage
// (ovspatialquery_cuda_driver.cpp).  Without the test hooks the helpers are
// constants and the injected-failure branches fold out of the shipped library.
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
extern std::atomic<uint32_t> gCudaFailurePoint;
// While set, a DeviceSignal ring's host function waits on the driver's thread
// before ringing, as a host function the driver runs late does, for at most
// 30 s; one that gives up sets gCudaSignalRingHoldExpired.
extern std::atomic<bool> gCudaHoldSignalRings;
extern std::atomic<bool> gCudaSignalRingHoldExpired;

// Whether `point` is armed; leaves it armed.
inline bool cudaFailurePointArmed(uint32_t point) noexcept { return gCudaFailurePoint.load() == point; }

// Consumes whatever is armed and returns it (0 when nothing is armed).
inline uint32_t takeCudaFailurePoint() noexcept { return gCudaFailurePoint.exchange(0u); }

// Consumes `point` if it is the armed one; other points stay armed.
inline bool takeCudaFailurePoint(uint32_t point) noexcept
{
    uint32_t expected = point;
    return gCudaFailurePoint.compare_exchange_strong(expected, 0u);
}
#else
constexpr bool cudaFailurePointArmed(uint32_t) noexcept { return false; }

inline uint32_t takeCudaFailurePoint() noexcept { return 0u; }

constexpr bool takeCudaFailurePoint(uint32_t) noexcept { return false; }
#endif

// The CUDA driver library is loaded by name at run time (nvcuda.dll on
// Windows, libcuda.so.1 on Linux); the engine never links it.
#if defined(_WIN32)
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void*;
#endif

LibraryHandle openCudaDriverLibrary() noexcept;
void          closeLibrary(LibraryHandle module) noexcept;
void*         librarySymbol(LibraryHandle module, const char* name) noexcept;

template <typename Function>
Function loadFunction(LibraryHandle module, const char* name)
{
    return reinterpret_cast<Function>(reinterpret_cast<std::uintptr_t>(librarySymbol(module, name)));
}

struct Driver
{
    LibraryHandle module                                                           = nullptr;
    CuResult (*init)(unsigned int)                                                 = nullptr;
    CuResult (*deviceGet)(CuDevice*, int)                                          = nullptr;
    CuResult (*contextGetCurrent)(CuContext*)                                      = nullptr;
    CuResult (*contextSetCurrent)(CuContext)                                       = nullptr;
    CuResult (*contextGetDevice)(int*)                                             = nullptr;
    CuResult (*primaryRetain)(CuContext*, CuDevice)                                = nullptr;
    CuResult (*primaryRelease)(CuDevice)                                           = nullptr;
    CuResult (*moduleLoad)(CuModule*, const char*)                                 = nullptr;
    CuResult (*moduleUnload)(CuModule)                                             = nullptr;
    CuResult (*moduleGetFunction)(CuFunction*, CuModule, const char*)              = nullptr;
    CuResult (*functionGetAttribute)(int*, int, CuFunction)                        = nullptr;
    CuResult (*contextGetLimit)(size_t*, int)                                      = nullptr;
    CuResult (*contextSetLimit)(int, size_t)                                       = nullptr;
    CuResult (*streamCreate)(CuStream*, unsigned int)                              = nullptr;
    CuResult (*streamDestroy)(CuStream)                                            = nullptr;
    CuResult (*streamQuery)(CuStream)                                              = nullptr;
    CuResult (*streamSynchronize)(CuStream)                                        = nullptr;
    CuResult (*streamWaitEvent)(CuStream, CuEvent, unsigned int)                   = nullptr;
    CuResult (*eventCreate)(CuEvent*, unsigned int)                                = nullptr;
    CuResult (*eventRecord)(CuEvent, CuStream)                                     = nullptr;
    CuResult (*eventQuery)(CuEvent)                                                = nullptr;
    CuResult (*eventSynchronize)(CuEvent)                                          = nullptr;
    CuResult (*eventDestroy)(CuEvent)                                              = nullptr;
    CuResult (*memAlloc)(CuDevicePtr*, size_t)                                     = nullptr;
    CuResult (*memFree)(CuDevicePtr)                                               = nullptr;
    CuResult (*memHostAlloc)(void**, size_t, unsigned int)                         = nullptr;
    CuResult (*memFreeHost)(void*)                                                 = nullptr;
    CuResult (*hostDevicePointer)(CuDevicePtr*, void*, unsigned int)               = nullptr;
    CuResult (*copyHtoDAsync)(CuDevicePtr, const void*, size_t, CuStream)          = nullptr;
    CuResult (*copyDtoH)(void*, CuDevicePtr, size_t)                               = nullptr;
    CuResult (*copyDtoHAsync)(void*, CuDevicePtr, size_t, CuStream)                = nullptr;
    CuResult (*copyDtoDAsync)(CuDevicePtr, CuDevicePtr, size_t, CuStream)          = nullptr;
    CuResult (*copy2DAsync)(const CuMemcpy2D*, CuStream)                           = nullptr;
    CuResult (*memsetD8Async)(CuDevicePtr, unsigned char, size_t, CuStream)        = nullptr;
    CuResult (*memsetD32Async)(CuDevicePtr, unsigned int, size_t, CuStream)        = nullptr;
    CuResult (*launchKernel)(CuFunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, CuStream, void**, void**) = nullptr;
    // Optional: without it, waiters fall back to polling.
    CuResult (*launchHostFunc)(CuStream, void (*)(void*), void*) = nullptr;
    // Optional: without it, a capturing stream is not refused at admission.
    CuResult (*streamIsCapturing)(CuStream, int*) = nullptr;

    bool open();

    ~Driver();
};

std::filesystem::path ptxPath();

// Device buffers that copied input columns are staged in, kept for the next
// copy (ADR-0053): allocating or freeing device memory synchronizes with a busy
// GPU, which a publishing or launching call must not do (ADR-0021). The owner
// gives a buffer back only once its last reader has finished, so a kept buffer
// is idle. At most kKept buffers are kept; their bytes were charged while in
// use and, like released version tables, are not charged while kept.
class CudaStagingPool
{
  public:
    static constexpr size_t kKept = 8u;

    ~CudaStagingPool();

    // The smallest kept buffer of at least `bytes` in `context`, or a new one.
    CuResult take(Driver& driver, CuContext context, size_t bytes, CuDevicePtr& outBuffer, size_t& outBytes) noexcept;

    // Keeps an idle buffer, or frees it with `driver` when the pool is full.
    void give(Driver& driver, CuContext context, CuDevicePtr buffer, size_t bytes) noexcept;

    // Retains `device`'s primary context until clear(), once per device, so the
    // buffers a caller that held the last retain gives back stay valid; false
    // when it could not, and the caller frees its buffers instead.
    bool keepPrimary(CuDevice device) noexcept;

    // Frees every kept buffer, then releases the primary contexts kept; their
    // contexts must otherwise be current or live.
    void clear() noexcept;

  private:
    struct Buffer
    {
        CuContext   context = nullptr;
        CuDevicePtr pointer = 0u;
        size_t      bytes   = 0u;
    };
    std::mutex            mMutex;
    std::vector<Buffer>   mBuffers;
    std::vector<CuDevice> mPrimaries;
    Driver                mDriver;
    bool                  mDriverOpened = false;
};

// Copies a device column with the layout `layout` (ADR-0053) in row-major order
// into `layout.compactBytes()` bytes at `destination`, on `stream`.
CuResult copyInputColumnAsync(Driver& driver, CuDevicePtr source, const InputColumnLayout& layout,
                              CuDevicePtr destination, CuStream stream) noexcept;

DLTensor makeCudaTensor(CuDevicePtr pointer, int32_t deviceId, uint8_t code, uint8_t bits, int32_t rank,
                        const int64_t* shape, const int64_t* strides);

// Raises the current context's per-thread stack limit to what `function`
// needs. Otherwise its first launch grows the stack itself, and that waits for
// the whole device, behind any work a caller holds back on its own stream.
// `module` holds `function` and the empty kernel that applies a raised limit.
CuResult reserveStack(Driver& driver, CuModule module, CuFunction function) noexcept;

// The bundled PTX loaded once into one device's primary context. Every CUDA
// plan, pair reporter, device box population and Ovstage device publication of
// this library copy on that device uses it, and so does device storage one of
// them leaves to another (a plan's resident mesh definitions); the availability
// probe loads it, and an instance that asked about the device keeps it
// (CudaModulePins). It keeps its own driver open and the context retained until
// the last holder is gone, then unloads. The kernels keep no module-scope state,
// so sharing the module changes nothing a launch computes.
struct CudaContextModule
{
    Driver            driver;
    CuDevice          device   = 0;
    CuContext         context  = nullptr;
    CuModule          module   = nullptr;
    bool              retained = false;
    std::atomic<bool> lost{false};

    ~CudaContextModule();
};

// The module of CUDA device `deviceId`, loaded when no live holder has one.
// Concurrent callers for one device wait for a single load; other devices load
// independently. NOT_SUPPORTED when the driver, device, context or PTX is
// unavailable, OUT_OF_MEMORY when the host allocation fails; `outReason`, when
// given, then receives the availability probe's diagnostic. The caller's current
// context is unchanged.
ovspatialquery_api_status_t acquireCudaContextModule(int32_t deviceId, std::shared_ptr<CudaContextModule>& outModule,
                                                     const char** outReason = nullptr) noexcept;

bool multiplyWithoutOverflow(uint64_t first, uint64_t second, size_t& bytes);

// Reasons cudaKernelUnavailableReason returns when the driver has no device
// with the requested ID, when the driver library cannot be loaded, when it
// cannot be initialized, and when the host cannot allocate the module's
// bookkeeping. The middle two hold for every device; the last is transient.
extern const char* const kCudaNoSuchDevice;
extern const char* const kCudaNoDriver;
extern const char* const kCudaDriverInitFailed;
extern const char* const kCudaHostOutOfMemory;

// Null when device `deviceId`'s shared module (acquireCudaContextModule) loads
// and exposes `kernelName`, and then that module in `outModule`; otherwise a
// static diagnostic naming the first step that failed.
const char* cudaKernelUnavailableReason(int32_t deviceId, const char* kernelName,
                                        std::shared_ptr<CudaContextModule>& outModule) noexcept;

// Rings `signal` from a CUDA host function once the work queued before it on
// `stream` has completed. Only for library-owned streams: a host function
// holds back the work queued behind it. False when it could not be queued.
// The stream's context is current.
bool ringAfter(Driver& driver, CuStream stream, const std::shared_ptr<DeviceSignal>& signal) noexcept;

// Makes `context` current for one scope and restores the caller's context at
// restore() or the scope's end. Device loss reported by any of these calls, or
// by a result passed to observe(), is recorded in `lost`.
class CudaCurrentContext
{
  public:
    CudaCurrentContext(Driver& driver, CuContext context, std::atomic<bool>& lost) noexcept;
    ~CudaCurrentContext() { (void)restore(); }

    CudaCurrentContext(const CudaCurrentContext&)            = delete;
    CudaCurrentContext& operator=(const CudaCurrentContext&) = delete;

    // kCuSuccess once `context` is current.
    CuResult status() const noexcept { return mStatus; }
    CuResult observe(CuResult status) const noexcept
    {
        if (status == kCuDeviceLost)
        {
            mLost.store(true, std::memory_order_relaxed);
        }
        return status;
    }
    // Restores the caller's context once; kCuSuccess when there is none to restore.
    CuResult restore() noexcept;

  private:
    Driver&            mDriver;
    std::atomic<bool>& mLost;
    CuContext          mRestore  = nullptr;
    CuResult           mStatus   = kCuSuccess;
    bool               mCaptured = false;
};

// The stream a CUDA family plan or pair reporter queues the host functions
// that wake the reaper on (CudaResultSlotEvents::ringWhenConsumed), and how
// many queued there have not run yet. A consumer event the caller holds back
// holds back every later ring of the owner, so a retirement queues none once
// `limit` are outstanding: rings only wake the reaper early, and an enqueue
// that finds every slot taken reclaims retired slots itself.
struct CudaSignalRings
{
    CuStream                               stream = nullptr;
    std::shared_ptr<std::atomic<uint32_t>> outstanding;
    uint32_t                               limit = 0u;

    // Creates the stream and the count; `stream` is null unless it succeeds.
    CuResult create(Driver& driver, uint32_t ringLimit) noexcept;
};

// The streams and events of one result slot of a CUDA family plan or pair
// reporter. `ready` ends a launch's work and keeps the default flags because
// consumers receive it as `result_ready`; the worker sleeps on `wake`, its
// blocking-sync twin recorded right after it; `consumer` is recorded on
// `cleanupStream` behind the caller's consumer-done event.
struct CudaResultSlotEvents
{
    CuStream cleanupStream = nullptr;
    CuEvent  ready         = nullptr;
    CuEvent  wake          = nullptr;
    CuEvent  consumer      = nullptr;

    CuResult create(Driver& driver) noexcept;
    // Destroys what was created, stopping at the first result `observe` rejects.
    template <typename Observe>
    bool destroy(Driver& driver, Observe observe) noexcept
    {
        return (consumer == nullptr || observe(driver.eventDestroy(consumer))) &&
               (wake == nullptr || observe(driver.eventDestroy(wake))) &&
               (ready == nullptr || observe(driver.eventDestroy(ready))) &&
               (cleanupStream == nullptr || observe(driver.streamDestroy(cleanupStream)));
    }
    // Records `consumer` behind the caller's `event` for an operation that has
    // not lost its device or recorded a consumer yet. NOT_SUPPORTED means the
    // device is lost; a failed restore of the caller's context fails the
    // transfer.
    ovspatialquery_api_status_t transferConsumerDone(Driver& driver, CuContext context, std::atomic<bool>& lost,
                                                     ovspatialquery_cuda_event_t event, bool operationLost,
                                                     bool recorded) const noexcept;
    // Rings `signal` once the last recorded `consumer` has completed, from a
    // host function on the owner's `rings` stream: on `cleanupStream` the
    // function would hold back the slot's next `consumer` until the driver ran
    // it, and the slot would stay taken that long. The slot's context is
    // current. Best effort: false when nothing was queued.
    bool ringWhenConsumed(Driver& driver, const CudaSignalRings& rings,
                          const std::shared_ptr<DeviceSignal>& signal) const noexcept;
};

#endif // defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

} // namespace detail
} // namespace ovspatialquery

#endif
