/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_cuda_driver.h"

#if defined(OVSPATIALQUERY_HAS_CUDA_BACKEND)

#if !defined(_WIN32)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>

namespace ovspatialquery
{
namespace detail
{

#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
std::atomic<uint32_t> gCudaFailurePoint{0u};
std::atomic<bool>     gCudaHoldSignalRings{false};
std::atomic<bool>     gCudaSignalRingHoldExpired{false};

namespace
{
// Every Driver of a test runtime loads and unloads modules through these, so the
// tests count the PTX loads of every site, not only of the shared module.
std::atomic<CuResult (*)(CuModule*, const char*)> gDriverModuleLoad{nullptr};
std::atomic<CuResult (*)(CuModule)>               gDriverModuleUnload{nullptr};

CuResult countedModuleLoad(CuModule* module, const char* path)
{
    const CuResult status = gDriverModuleLoad.load()(module, path);
    if (status == kCuSuccess)
    {
        noteCudaModuleLoad(1);
    }
    return status;
}

CuResult countedModuleUnload(CuModule module)
{
    const CuResult status = gDriverModuleUnload.load()(module);
    if (status == kCuSuccess)
    {
        noteCudaModuleLoad(-1);
    }
    return status;
}
} // namespace
#endif

LibraryHandle openCudaDriverLibrary() noexcept
{
#if defined(_WIN32)
    return LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    return dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void closeLibrary(LibraryHandle module) noexcept
{
#if defined(_WIN32)
    FreeLibrary(module);
#else
    dlclose(module);
#endif
}

void* librarySymbol(LibraryHandle module, const char* name) noexcept
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(GetProcAddress(module, name)));
#else
    return dlsym(module, name);
#endif
}

bool Driver::open()
{
    module = openCudaDriverLibrary();
    if (module == nullptr)
    {
        return false;
    }
    init                 = loadFunction<decltype(init)>(module, "cuInit");
    deviceGet            = loadFunction<decltype(deviceGet)>(module, "cuDeviceGet");
    contextGetCurrent    = loadFunction<decltype(contextGetCurrent)>(module, "cuCtxGetCurrent");
    contextSetCurrent    = loadFunction<decltype(contextSetCurrent)>(module, "cuCtxSetCurrent");
    contextGetDevice     = loadFunction<decltype(contextGetDevice)>(module, "cuCtxGetDevice");
    primaryRetain        = loadFunction<decltype(primaryRetain)>(module, "cuDevicePrimaryCtxRetain");
    primaryRelease       = loadFunction<decltype(primaryRelease)>(module, "cuDevicePrimaryCtxRelease_v2");
    moduleLoad           = loadFunction<decltype(moduleLoad)>(module, "cuModuleLoad");
    moduleUnload         = loadFunction<decltype(moduleUnload)>(module, "cuModuleUnload");
    moduleGetFunction    = loadFunction<decltype(moduleGetFunction)>(module, "cuModuleGetFunction");
    functionGetAttribute = loadFunction<decltype(functionGetAttribute)>(module, "cuFuncGetAttribute");
    contextGetLimit      = loadFunction<decltype(contextGetLimit)>(module, "cuCtxGetLimit");
    contextSetLimit      = loadFunction<decltype(contextSetLimit)>(module, "cuCtxSetLimit");
    streamCreate         = loadFunction<decltype(streamCreate)>(module, "cuStreamCreate");
    streamDestroy        = loadFunction<decltype(streamDestroy)>(module, "cuStreamDestroy_v2");
    streamQuery          = loadFunction<decltype(streamQuery)>(module, "cuStreamQuery");
    streamSynchronize    = loadFunction<decltype(streamSynchronize)>(module, "cuStreamSynchronize");
    streamWaitEvent      = loadFunction<decltype(streamWaitEvent)>(module, "cuStreamWaitEvent");
    eventCreate          = loadFunction<decltype(eventCreate)>(module, "cuEventCreate");
    eventRecord          = loadFunction<decltype(eventRecord)>(module, "cuEventRecord");
    eventQuery           = loadFunction<decltype(eventQuery)>(module, "cuEventQuery");
    eventSynchronize     = loadFunction<decltype(eventSynchronize)>(module, "cuEventSynchronize");
    eventDestroy         = loadFunction<decltype(eventDestroy)>(module, "cuEventDestroy_v2");
    memAlloc             = loadFunction<decltype(memAlloc)>(module, "cuMemAlloc_v2");
    memFree              = loadFunction<decltype(memFree)>(module, "cuMemFree_v2");
    memHostAlloc         = loadFunction<decltype(memHostAlloc)>(module, "cuMemHostAlloc");
    memFreeHost          = loadFunction<decltype(memFreeHost)>(module, "cuMemFreeHost");
    copyHtoDAsync        = loadFunction<decltype(copyHtoDAsync)>(module, "cuMemcpyHtoDAsync_v2");
    copyDtoH             = loadFunction<decltype(copyDtoH)>(module, "cuMemcpyDtoH_v2");
    copyDtoHAsync        = loadFunction<decltype(copyDtoHAsync)>(module, "cuMemcpyDtoHAsync_v2");
    hostDevicePointer    = loadFunction<decltype(hostDevicePointer)>(module, "cuMemHostGetDevicePointer_v2");
    copyDtoDAsync        = loadFunction<decltype(copyDtoDAsync)>(module, "cuMemcpyDtoDAsync_v2");
    copy2DAsync          = loadFunction<decltype(copy2DAsync)>(module, "cuMemcpy2DAsync_v2");
    memsetD8Async        = loadFunction<decltype(memsetD8Async)>(module, "cuMemsetD8Async");
    memsetD32Async       = loadFunction<decltype(memsetD32Async)>(module, "cuMemsetD32Async");
    launchKernel         = loadFunction<decltype(launchKernel)>(module, "cuLaunchKernel");
    launchHostFunc       = loadFunction<decltype(launchHostFunc)>(module, "cuLaunchHostFunc");
    streamIsCapturing    = loadFunction<decltype(streamIsCapturing)>(module, "cuStreamIsCapturing");
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    if (moduleLoad != nullptr && moduleUnload != nullptr)
    {
        gDriverModuleLoad.store(moduleLoad);
        gDriverModuleUnload.store(moduleUnload);
        moduleLoad   = &countedModuleLoad;
        moduleUnload = &countedModuleUnload;
    }
#endif
    return init != nullptr && deviceGet != nullptr && contextGetCurrent != nullptr && contextSetCurrent != nullptr &&
           contextGetDevice != nullptr && primaryRetain != nullptr && primaryRelease != nullptr &&
           moduleLoad != nullptr && moduleUnload != nullptr && moduleGetFunction != nullptr &&
           functionGetAttribute != nullptr && contextGetLimit != nullptr && contextSetLimit != nullptr &&
           streamCreate != nullptr && streamDestroy != nullptr && streamQuery != nullptr &&
           streamSynchronize != nullptr && streamWaitEvent != nullptr && eventCreate != nullptr &&
           eventRecord != nullptr && eventQuery != nullptr && eventSynchronize != nullptr && eventDestroy != nullptr &&
           memAlloc != nullptr && memFree != nullptr && memHostAlloc != nullptr && memFreeHost != nullptr &&
           hostDevicePointer != nullptr && copyHtoDAsync != nullptr && copyDtoH != nullptr &&
           copyDtoHAsync != nullptr && copyDtoDAsync != nullptr && copy2DAsync != nullptr && memsetD8Async != nullptr &&
           memsetD32Async != nullptr && launchKernel != nullptr && init(0u) == kCuSuccess;
}

Driver::~Driver()
{
    if (module != nullptr)
    {
        closeLibrary(module);
    }
}

CudaStagingPool::~CudaStagingPool() { clear(); }

CuResult CudaStagingPool::take(Driver& driver, CuContext context, size_t bytes, CuDevicePtr& outBuffer,
                               size_t& outBytes) noexcept
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        size_t                      best = mBuffers.size();
        for (size_t index = 0u; index < mBuffers.size(); ++index)
        {
            const Buffer& buffer = mBuffers[index];
            if (buffer.context == context && buffer.bytes >= bytes &&
                (best == mBuffers.size() || buffer.bytes < mBuffers[best].bytes))
            {
                best = index;
            }
        }
        if (best != mBuffers.size())
        {
            outBuffer = mBuffers[best].pointer;
            outBytes  = mBuffers[best].bytes;
            mBuffers.erase(mBuffers.begin() + static_cast<std::ptrdiff_t>(best));
            return kCuSuccess;
        }
    }
    outBytes = bytes;
    return driver.memAlloc(&outBuffer, bytes);
}

void CudaStagingPool::give(Driver& driver, CuContext context, CuDevicePtr buffer, size_t bytes) noexcept
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (!mDriverOpened)
        {
            mDriverOpened = mDriver.open();
        }
        if (mDriverOpened && mBuffers.size() < kKept)
        {
            try
            {
                mBuffers.push_back({context, buffer, bytes});
                return;
            }
            catch (...)
            {}
        }
    }
    (void)driver.memFree(buffer);
}

bool CudaStagingPool::keepPrimary(CuDevice device) noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (!mDriverOpened)
    {
        mDriverOpened = mDriver.open();
    }
    if (!mDriverOpened)
    {
        return false;
    }
    if (std::find(mPrimaries.begin(), mPrimaries.end(), device) != mPrimaries.end())
    {
        return true;
    }
    try
    {
        mPrimaries.reserve(mPrimaries.size() + 1u);
    }
    catch (...)
    {
        return false;
    }
    CuContext context = nullptr;
    if (mDriver.primaryRetain(&context, device) != kCuSuccess)
    {
        return false;
    }
    mPrimaries.push_back(device);
    return true;
}

void CudaStagingPool::clear() noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    if (mBuffers.empty())
    {
        for (const CuDevice device : mPrimaries)
        {
            (void)mDriver.primaryRelease(device);
        }
        mPrimaries.clear();
        return;
    }
    CuContext previous = nullptr;
    (void)mDriver.contextGetCurrent(&previous);
    for (const Buffer& buffer : mBuffers)
    {
        if (mDriver.contextSetCurrent(buffer.context) == kCuSuccess)
        {
            (void)mDriver.memFree(buffer.pointer);
        }
    }
    (void)mDriver.contextSetCurrent(previous);
    mBuffers.clear();
    for (const CuDevice device : mPrimaries)
    {
        (void)mDriver.primaryRelease(device);
    }
    mPrimaries.clear();
}

CuResult copyInputColumnAsync(Driver& driver, CuDevicePtr source, const InputColumnLayout& layout,
                              CuDevicePtr destination, CuStream stream) noexcept
{
    const size_t elementBytes = static_cast<size_t>(layout.elementBytes);
    const size_t rowBytes     = static_cast<size_t>(layout.width) * elementBytes;
    if (layout.compact || layout.rows == 0u)
    {
        return layout.rows == 0u
                   ? kCuSuccess
                   : driver.copyDtoDAsync(destination, source, static_cast<size_t>(layout.compactBytes()), stream);
    }
    // Rows that all repeat the first are copied once, then doubled in place.
    const size_t copiedRows = layout.rowStride == 0u ? 1u : static_cast<size_t>(layout.rows);
    const size_t srcPitch   = static_cast<size_t>(layout.rowStride) * elementBytes;
    // A pitched copy takes at most a 32-bit signed pitch; wider strides copy
    // element by element.
    constexpr size_t kMaximumPitch = 0x7fffffffu;
    const bool       adjacent      = layout.width == 1u || layout.columnStride == 1u;
    CuResult         status        = kCuSuccess;
    if (srcPitch > kMaximumPitch || rowBytes > kMaximumPitch)
    {
        for (size_t row = 0u; status == kCuSuccess && row < copiedRows; ++row)
        {
            for (size_t element = 0u; status == kCuSuccess && element < layout.width; ++element)
            {
                status = driver.copyDtoDAsync(destination + row * rowBytes + element * elementBytes,
                                              source + row * srcPitch +
                                                  element * static_cast<size_t>(layout.columnStride) * elementBytes,
                                              elementBytes, stream);
            }
        }
    }
    else if (adjacent && (copiedRows == 1u || srcPitch >= rowBytes))
    {
        CuMemcpy2D copy{};
        copy.srcMemoryType = kCuMemoryTypeDevice;
        copy.srcDevice     = source;
        copy.srcPitch      = copiedRows == 1u ? rowBytes : srcPitch;
        copy.dstMemoryType = kCuMemoryTypeDevice;
        copy.dstDevice     = destination;
        copy.dstPitch      = rowBytes;
        copy.widthInBytes  = rowBytes;
        copy.height        = copiedRows;
        status             = driver.copy2DAsync(&copy, stream);
    }
    else
    {
        // One pitched copy per element of a row: Fortran order, an inner
        // stride other than one, or rows that overlap.
        for (size_t element = 0u; status == kCuSuccess && element < layout.width; ++element)
        {
            CuMemcpy2D copy{};
            copy.srcMemoryType = kCuMemoryTypeDevice;
            copy.srcDevice     = source + element * static_cast<size_t>(layout.columnStride) * elementBytes;
            copy.srcPitch      = copiedRows == 1u ? elementBytes : srcPitch;
            copy.dstMemoryType = kCuMemoryTypeDevice;
            copy.dstDevice     = destination;
            copy.dstXInBytes   = element * elementBytes;
            copy.dstPitch      = rowBytes;
            copy.widthInBytes  = elementBytes;
            copy.height        = copiedRows;
            status             = driver.copy2DAsync(&copy, stream);
        }
    }
    for (size_t filled = copiedRows; status == kCuSuccess && filled < layout.rows;)
    {
        const size_t count = std::min(filled, static_cast<size_t>(layout.rows) - filled);
        status = driver.copyDtoDAsync(destination + filled * rowBytes, destination, count * rowBytes, stream);
        filled += count;
    }
    return status;
}

CuResult reserveStack(Driver& driver, CuModule module, CuFunction function) noexcept
{
    // Plans created at once on one device share its primary context; without the
    // lock one could read the old limit and then lower another's larger
    // reservation. The limit is per context and setting it waits for that device,
    // so plans on other devices take other locks.
    static std::array<std::mutex, 64> reservations;
    int                               device = 0;
    if (const CuResult status = driver.contextGetDevice(&device); status != kCuSuccess)
    {
        return status;
    }
    const std::lock_guard<std::mutex> lock(reservations[size_t(device) % reservations.size()]);
    int                               needed = 0;
    size_t                            limit  = 0u;
    CuResult                          status = driver.functionGetAttribute(&needed, kCuFuncLocalSizeBytes, function);
    if (status == kCuSuccess)
    {
        status = driver.contextGetLimit(&limit, kCuLimitStackSize);
    }
    if (status != kCuSuccess || size_t(needed) <= limit)
    {
        return status;
    }
    status = driver.contextSetLimit(kCuLimitStackSize, size_t(needed));
    // The driver finishes applying the new limit at the next launch; until then a
    // legacy-stream synchronization, in this library or its caller, waits for every
    // stream on the device. An empty launch applies it while nothing is held.
    CuFunction apply  = nullptr;
    CuStream   stream = nullptr;
    if (status == kCuSuccess)
    {
        status = driver.moduleGetFunction(&apply, module, "ovspatialquery_cuda_apply_stack_limit");
    }
    if (status == kCuSuccess)
    {
        status = driver.streamCreate(&stream, kCuStreamNonBlocking);
    }
    if (status == kCuSuccess)
    {
        status                   = driver.launchKernel(apply, 1u, 1u, 1u, 1u, 1u, 1u, 0u, stream, nullptr, nullptr);
        const CuResult destroyed = driver.streamDestroy(stream);
        status                   = status == kCuSuccess ? destroyed : status;
    }
    return status;
}

// The PTX is installed next to the library that loads it.
std::filesystem::path ptxPath()
{
#if !defined(_WIN32)
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(&cudaBackendUnavailableReason)), &info) == 0 ||
        info.dli_fname == nullptr)
    {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path() / "ovspatialquery_cuda.ptx";
#else
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&cudaBackendUnavailableReason), &module) == 0)
    {
        return {};
    }
    std::wstring path(32768u, L'\0');
    const DWORD  length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0u || length == path.size())
    {
        return {};
    }
    path.resize(length);
    return std::filesystem::path(path).parent_path() / "ovspatialquery_cuda.ptx";
#endif
}

DLTensor makeCudaTensor(CuDevicePtr pointer, int32_t deviceId, uint8_t code, uint8_t bits, int32_t rank,
                        const int64_t* shape, const int64_t* strides)
{
    DLTensor tensor{};
    tensor.data    = reinterpret_cast<void*>(static_cast<uintptr_t>(pointer));
    tensor.device  = DLDevice{kDLCUDA, deviceId};
    tensor.ndim    = rank;
    tensor.dtype   = DLDataType{code, bits, 1u};
    tensor.shape   = const_cast<int64_t*>(shape);
    tensor.strides = const_cast<int64_t*>(strides);
    return tensor;
}

bool multiplyWithoutOverflow(uint64_t first, uint64_t second, size_t& bytes)
{
    if (first != 0u && second > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) / first)
    {
        return false;
    }
    bytes = static_cast<size_t>(first * second);
    return true;
}

namespace
{
const char* const kCudaCannotOpenDevice    = "CUDA is unavailable: the requested CUDA device could not be opened";
const char* const kCudaCannotChangeContext = "CUDA is unavailable: the current CUDA context could not be changed";
const char* const kCudaCannotRestoreContext =
    "CUDA is unavailable: the CUDA availability probe could not restore the caller's context";
const char* const kCudaPtxNotLoaded =
    "CUDA is unavailable: the NVIDIA driver could not load the bundled ovspatialquery_cuda.ptx";

// The bundled PTX is generated by NVRTC 12.8 (PTX ISA 8.7, sm_75), which the
// driver JIT-compiles; docs/development/releases.md states the same floor.
const char* moduleLoadFailure(CuResult status) noexcept
{
    switch (status)
    {
    case 222: // CUDA_ERROR_UNSUPPORTED_PTX_VERSION
        return "CUDA is unavailable: the NVIDIA driver is too old for the bundled ovspatialquery_cuda.ptx "
               "(PTX ISA 8.7); install a CUDA 12.8 (R570) or newer driver, 570.26+ on Linux or 570.65+ on Windows";
    case 209: // CUDA_ERROR_NO_BINARY_FOR_GPU
        return "CUDA is unavailable: the bundled ovspatialquery_cuda.ptx needs a GPU of compute capability 7.5 "
               "or newer";
    case 301: // CUDA_ERROR_FILE_NOT_FOUND
        return "CUDA is unavailable: ovspatialquery_cuda.ptx is missing from the directory of the ovspatialquery "
               "library";
    default: return kCudaPtxNotLoaded;
    }
}
} // namespace

const char* const kCudaNoSuchDevice = "CUDA is unavailable: the NVIDIA driver reports no CUDA device with the "
                                      "requested ID";
const char* const kCudaNoDriver =
    "CUDA is unavailable: the NVIDIA driver library (libcuda.so.1 or nvcuda.dll) could not be loaded";
const char* const kCudaDriverInitFailed =
    "CUDA is unavailable: the NVIDIA driver failed to initialize (cuInit) or lacks a required entry point";
const char* const kCudaHostOutOfMemory = "CUDA is unavailable: the host ran out of memory while loading the bundled "
                                         "ovspatialquery_cuda.ptx";

const char* cudaKernelUnavailableReason(int32_t deviceId, const char* kernelName,
                                        std::shared_ptr<CudaContextModule>& outModule) noexcept
{
    outModule.reset();
    const char*                        reason = nullptr;
    std::shared_ptr<CudaContextModule> module;
    if (acquireCudaContextModule(deviceId, module, &reason) != OVSPATIALQUERY_API_SUCCESS)
    {
        return reason;
    }
    {
        CudaCurrentContext scope(module->driver, module->context, module->lost);
        CuFunction         function = nullptr;
        if (scope.status() != kCuSuccess)
        {
            reason = kCudaCannotChangeContext;
        }
        else if (module->driver.moduleGetFunction(&function, module->module, kernelName) != kCuSuccess)
        {
            reason = "CUDA is unavailable: the bundled ovspatialquery_cuda.ptx lacks a required kernel";
        }
        if (scope.restore() != kCuSuccess && reason == nullptr)
        {
            reason = kCudaCannotRestoreContext;
        }
    }
    if (reason == nullptr)
    {
        outModule = std::move(module);
    }
    return reason;
}

CudaContextModule::~CudaContextModule()
{
    if (!retained || lost.load(std::memory_order_relaxed))
    {
        return;
    }
    if (module != nullptr)
    {
        CudaCurrentContext scope(driver, context, lost);
        if (scope.status() == kCuSuccess)
        {
            (void)scope.observe(driver.moduleUnload(module));
        }
    }
    if (!lost.load(std::memory_order_relaxed))
    {
        (void)driver.primaryRelease(device);
    }
}

ovspatialquery_api_status_t acquireCudaContextModule(int32_t deviceId, std::shared_ptr<CudaContextModule>& outModule,
                                                     const char** outReason) noexcept
{
    const char*  ignored = nullptr;
    const char*& reason  = outReason != nullptr ? *outReason : ignored;
    reason               = nullptr;
    // Per device, so a first load on one device does not hold back another's. An
    // entry stays for the process; entries are only made for devices the driver
    // has, so requests naming arbitrary IDs cannot grow the registry.
    struct Entry
    {
        std::mutex                       mutex;
        std::weak_ptr<CudaContextModule> module;
    };
    static std::mutex               registry;
    static std::map<int32_t, Entry> entries;
    Entry*                          entry = nullptr;
    {
        const std::lock_guard<std::mutex>        lock(registry);
        const std::map<int32_t, Entry>::iterator found = entries.find(deviceId);
        entry                                          = found != entries.end() ? &found->second : nullptr;
    }
    if (entry == nullptr)
    {
        Driver   check;
        CuDevice device = 0;
        if (!check.open())
        {
            reason = check.module == nullptr ? kCudaNoDriver : kCudaDriverInitFailed;
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        const CuResult found = check.deviceGet(&device, deviceId);
        if (found != kCuSuccess)
        {
            reason = found == kCuInvalidDevice ? kCudaNoSuchDevice : kCudaCannotOpenDevice;
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        try
        {
            const std::lock_guard<std::mutex> lock(registry);
            entry = &entries[deviceId];
        }
        catch (const std::bad_alloc&)
        {
            reason = kCudaHostOutOfMemory;
            return OVSPATIALQUERY_API_OUT_OF_MEMORY;
        }
    }
    const std::lock_guard<std::mutex> lock(entry->mutex);
    outModule = entry->module.lock();
    if (outModule != nullptr)
    {
        return OVSPATIALQUERY_API_SUCCESS;
    }
    std::shared_ptr<CudaContextModule> loaded;
    std::string                        path;
    try
    {
        loaded = std::make_shared<CudaContextModule>();
        path   = ptxPath().string();
    }
    catch (const std::bad_alloc&)
    {
        reason = kCudaHostOutOfMemory;
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        // A library path the narrow code page cannot represent (Windows).
        reason = kCudaPtxNotLoaded;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    Driver& driver = loaded->driver;
    if (!driver.open())
    {
        reason = driver.module == nullptr ? kCudaNoDriver : kCudaDriverInitFailed;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    const CuResult found = driver.deviceGet(&loaded->device, deviceId);
    if (found != kCuSuccess || driver.primaryRetain(&loaded->context, loaded->device) != kCuSuccess)
    {
        reason = found == kCuInvalidDevice ? kCudaNoSuchDevice : kCudaCannotOpenDevice;
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    loaded->retained = true;
    {
        CudaCurrentContext scope(driver, loaded->context, loaded->lost);
        if (scope.status() != kCuSuccess)
        {
            reason = kCudaCannotChangeContext;
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        const CuResult status = scope.observe(driver.moduleLoad(&loaded->module, path.c_str()));
        if (status != kCuSuccess)
        {
            loaded->module = nullptr;
            reason         = moduleLoadFailure(status);
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
        if (scope.restore() != kCuSuccess)
        {
            reason = kCudaCannotRestoreContext;
            return OVSPATIALQUERY_API_NOT_SUPPORTED;
        }
    }
    entry->module = loaded;
    outModule     = std::move(loaded);
    return OVSPATIALQUERY_API_SUCCESS;
}

namespace
{
// A queued ring: the signal, and the count of the owner's outstanding rings
// when it has one (CudaSignalRings).
struct QueuedRing
{
    std::shared_ptr<DeviceSignal>          signal;
    std::shared_ptr<std::atomic<uint32_t>> outstanding;
};

void ringSignal(void* data)
{
#if defined(OVSPATIALQUERY_ENABLE_TEST_HOOKS)
    const std::chrono::steady_clock::time_point giveUp = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (gCudaHoldSignalRings.load())
    {
        if (std::chrono::steady_clock::now() >= giveUp)
        {
            gCudaSignalRingHoldExpired.store(true);
            break;
        }
        std::this_thread::yield();
    }
#endif
    const std::unique_ptr<QueuedRing> ring(static_cast<QueuedRing*>(data));
    if (ring->outstanding != nullptr)
    {
        ring->outstanding->fetch_sub(1u);
    }
    ring->signal->ring();
}
} // namespace

bool ringAfter(Driver& driver, CuStream stream, const std::shared_ptr<DeviceSignal>& signal) noexcept
{
    if (driver.launchHostFunc == nullptr || signal == nullptr)
    {
        return false;
    }
    QueuedRing* ring = new (std::nothrow) QueuedRing{signal, nullptr};
    if (ring == nullptr)
    {
        return false;
    }
    if (driver.launchHostFunc(stream, &ringSignal, ring) != kCuSuccess)
    {
        delete ring;
        return false;
    }
    return true;
}

CudaCurrentContext::CudaCurrentContext(Driver& driver, CuContext context, std::atomic<bool>& lost) noexcept
    : mDriver(driver), mLost(lost)
{
    mStatus   = observe(driver.contextGetCurrent(&mRestore));
    mCaptured = mStatus == kCuSuccess;
    if (mCaptured)
    {
        mStatus = observe(driver.contextSetCurrent(context));
    }
}

CuResult CudaCurrentContext::restore() noexcept
{
    if (!mCaptured)
    {
        return kCuSuccess;
    }
    mCaptured = false;
    return observe(mDriver.contextSetCurrent(mRestore));
}

CuResult CudaResultSlotEvents::create(Driver& driver) noexcept
{
    CuResult status = driver.streamCreate(&cleanupStream, kCuStreamNonBlocking);
    if (status == kCuSuccess)
    {
        status = driver.eventCreate(&ready, kCuEventDisableTiming);
    }
    if (status == kCuSuccess)
    {
        status = driver.eventCreate(&wake, kCuEventBlockingSync | kCuEventDisableTiming);
    }
    if (status == kCuSuccess)
    {
        status = driver.eventCreate(&consumer, kCuEventDisableTiming);
    }
    return status;
}

ovspatialquery_api_status_t CudaResultSlotEvents::transferConsumerDone(Driver& driver, CuContext context,
                                                                       std::atomic<bool>&          lost,
                                                                       ovspatialquery_cuda_event_t event,
                                                                       bool operationLost, bool recorded) const noexcept
{
    const CuEvent supplied = reinterpret_cast<CuEvent>(event.value);
    if (supplied == nullptr)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    if (operationLost)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (recorded || supplied == ready || supplied == consumer)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    CudaCurrentContext scope(driver, context, lost);
    CuResult           status = scope.status();
    if (status == kCuSuccess)
    {
        status = scope.observe(driver.streamWaitEvent(cleanupStream, supplied, 0u));
    }
    if (status == kCuSuccess)
    {
        status = scope.observe(driver.eventRecord(consumer, cleanupStream));
    }
    CuResult restored = scope.restore();
    if (restored == kCuSuccess && takeCudaFailurePoint(kCudaFailureConsumerTransferRestore))
    {
        restored = kCuInvalidContext;
    }
    else if (restored == kCuSuccess && takeCudaFailurePoint(kCudaFailureConsumerTransferRestoreDeviceLost))
    {
        restored = scope.observe(kCuDeviceLost);
    }
    if (status == kCuDeviceLost || restored == kCuDeviceLost)
    {
        return OVSPATIALQUERY_API_NOT_SUPPORTED;
    }
    if (status != kCuSuccess || restored != kCuSuccess)
    {
        return status == kCuInvalidHandle ? OVSPATIALQUERY_API_INVALID_ARGUMENT : OVSPATIALQUERY_API_INTERNAL;
    }
    return OVSPATIALQUERY_API_SUCCESS;
}

CuResult CudaSignalRings::create(Driver& driver, uint32_t ringLimit) noexcept
{
    try
    {
        outstanding = std::make_shared<std::atomic<uint32_t>>(0u);
    }
    catch (const std::bad_alloc&)
    {
        return kCuOutOfMemory;
    }
    limit                 = ringLimit;
    const CuResult status = driver.streamCreate(&stream, kCuStreamNonBlocking);
    if (status != kCuSuccess)
    {
        stream = nullptr;
    }
    return status;
}

bool CudaResultSlotEvents::ringWhenConsumed(Driver& driver, const CudaSignalRings& rings,
                                            const std::shared_ptr<DeviceSignal>& signal) const noexcept
{
    if (rings.outstanding == nullptr || rings.outstanding->fetch_add(1u) >= rings.limit)
    {
        if (rings.outstanding != nullptr)
        {
            rings.outstanding->fetch_sub(1u);
        }
        return false;
    }
    QueuedRing* ring = nullptr;
    if (driver.launchHostFunc != nullptr && signal != nullptr)
    {
        ring = new (std::nothrow) QueuedRing{signal, rings.outstanding};
    }
    if (ring == nullptr || driver.streamWaitEvent(rings.stream, consumer, 0u) != kCuSuccess)
    {
        delete ring;
        rings.outstanding->fetch_sub(1u);
        return false;
    }
    // The wait stays queued even if the ring cannot follow it, so it keeps its
    // place in the count.
    if (driver.launchHostFunc(rings.stream, &ringSignal, ring) != kCuSuccess)
    {
        delete ring;
        return false;
    }
    return true;
}

} // namespace detail
} // namespace ovspatialquery

#endif
