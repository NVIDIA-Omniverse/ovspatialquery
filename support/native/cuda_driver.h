/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_SUPPORT_CUDA_DRIVER_H
#define OVSPATIALQUERY_SUPPORT_CUDA_DRIVER_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// Function-pointer-loads the CUDA driver API from nvcuda.dll (Windows) or libcuda.so.1 (Linux).
// Shared by the native test tree
// (stream/event/allocation lifetime tests, foreign-context ownership-transfer tests, host
// callbacks, and a PTX scheduling-delay kernel) and
// by lesson 11's native truth producer (examples/native/11_capability_showcase.cpp), which only
// needs the create/upload/download/wait/record/synchronize subset. Benchmark experiment drivers
// also time device work with its timing events.
class CudaDriver
{
  public:
    using DevicePointer = unsigned long long;
    using Context       = void*;
    using Stream        = void*;
    using Event         = void*;
#if defined(_WIN32)
    using LibraryHandle = HMODULE;
#else
    using LibraryHandle = void*;
#endif

    struct ForeignEvent
    {
        Context context = nullptr;
        Event   event   = nullptr;
    };

    // Who destroys the streams, events and allocations this driver creates.
    // manual: the caller does. tracked: the destructor does, and code that
    // hands an event to the library calls forgetTransferredEvent first.
    enum class Ownership
    {
        manual,
        tracked,
    };

    // Works in the primary context of CUDA device `ordinal`, current from construction to destruction.
    explicit CudaDriver(Ownership ownership = Ownership::manual, int ordinal = 0)
        : m_tracked(ownership == Ownership::tracked)
    {
        try
        {
#if defined(_WIN32)
            m_module = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
            m_module = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
            if (m_module == nullptr)
            {
                throw std::runtime_error("CUDA driver is unavailable");
            }
            load(m_init, "cuInit");
            load(m_device_get, "cuDeviceGet");
            load(m_context_get_current, "cuCtxGetCurrent");
            load(m_primary_retain, "cuDevicePrimaryCtxRetain");
            load(m_primary_release, "cuDevicePrimaryCtxRelease_v2");
            load(m_context_set_current, "cuCtxSetCurrent");
            load(m_context_create, "cuCtxCreate_v2");
            load(m_context_destroy, "cuCtxDestroy_v2");
            load(m_stream_create, "cuStreamCreate");
            load(m_stream_destroy, "cuStreamDestroy_v2");
            load(m_event_create, "cuEventCreate");
            load(m_event_record, "cuEventRecord");
            load(m_event_synchronize, "cuEventSynchronize");
            load(m_event_query, "cuEventQuery");
            load(m_event_elapsed_time, "cuEventElapsedTime");
            load(m_event_destroy, "cuEventDestroy_v2");
            load(m_stream_wait_event, "cuStreamWaitEvent");
            load(m_module_load, "cuModuleLoad");
            load(m_module_unload, "cuModuleUnload");
            load(m_module_get_function, "cuModuleGetFunction");
            load(m_launch_kernel, "cuLaunchKernel");
            load(m_launch_host_function, "cuLaunchHostFunc");
            load(m_mem_alloc, "cuMemAlloc_v2");
            load(m_mem_free, "cuMemFree_v2");
            load(m_mem_alloc_host, "cuMemAllocHost_v2");
            load(m_mem_free_host, "cuMemFreeHost");
            load(m_mem_host_get_flags, "cuMemHostGetFlags");
            load(m_copy_host_to_device, "cuMemcpyHtoDAsync_v2");
            load(m_copy_device_to_host, "cuMemcpyDtoHAsync_v2");
            load(m_stream_begin_capture, "cuStreamBeginCapture_v2");
            load(m_stream_end_capture, "cuStreamEndCapture");
            load(m_graph_get_nodes, "cuGraphGetNodes");
            load(m_graph_destroy, "cuGraphDestroy");

            if (m_init(0u) != 0 || m_device_get(&m_device, ordinal) != 0 ||
                m_context_get_current(&m_previous_context) != 0)
            {
                throw std::runtime_error("the CUDA device is unavailable");
            }
            check(m_primary_retain(&m_context, m_device));
            m_primary_retained = true;
            check(m_context_set_current(m_context));
            m_context_selected = true;

            m_streams.reserve(2u);
            m_events.reserve(3u);
            m_allocations.reserve(10u);
        }
        catch (...)
        {
            rollbackConstruction();
            throw;
        }
    }

    CudaDriver(const CudaDriver&)            = delete;
    CudaDriver& operator=(const CudaDriver&) = delete;

    ~CudaDriver()
    {
        if (m_context != nullptr)
        {
            m_context_set_current(m_context);
            if (m_gate_module != nullptr)
            {
                m_module_unload(m_gate_module);
            }
            for (Event event : m_events)
            {
                m_event_destroy(event);
            }
            for (DevicePointer pointer : m_allocations)
            {
                m_mem_free(pointer);
            }
            for (Stream stream : m_streams)
            {
                m_stream_destroy(stream);
            }
            m_context_set_current(m_previous_context);
            m_primary_release(m_device);
        }
        if (m_module != nullptr)
        {
#if defined(_WIN32)
            FreeLibrary(m_module);
#else
            dlclose(m_module);
#endif
        }
    }

    Stream createStream()
    {
        Stream stream = nullptr;
        check(m_stream_create(&stream, kNonBlockingStream));
        track(m_streams, stream);
        return stream;
    }

    Event createEvent()
    {
        Event event = nullptr;
        check(m_event_create(&event, kDisableTimingEvent));
        track(m_events, event);
        return event;
    }

    // An event cuEventElapsedTime can read; for benchmark drivers that time device work.
    Event createTimingEvent()
    {
        Event event = nullptr;
        check(m_event_create(&event, 0u));
        track(m_events, event);
        return event;
    }

    // Milliseconds between two completed timing events.
    float elapsedMilliseconds(Event start, Event end)
    {
        float milliseconds = 0.0f;
        check(m_event_elapsed_time(&milliseconds, start, end));
        return milliseconds;
    }

    // Releases tracked ownership of an event handed off to other code (e.g. a CUDA-consumer
    // process) without destroying it, so the destructor does not double-free it.
    bool forgetTransferredEvent(Event event) noexcept { return untrack(m_events, event); }

    ForeignEvent createForeignEvent()
    {
        ForeignEvent foreign{};
        check(m_context_create(&foreign.context, 0u, 0));
        check(m_event_create(&foreign.event, kDisableTimingEvent));
        check(m_event_record(foreign.event, nullptr));
        check(m_event_synchronize(foreign.event));
        check(m_context_set_current(m_context));
        return foreign;
    }

    void destroyForeignEvent(ForeignEvent foreign)
    {
        check(m_context_set_current(foreign.context));
        check(m_event_destroy(foreign.event));
        check(m_context_set_current(m_context));
        check(m_context_destroy(foreign.context));
    }

    void destroyForeignContext(ForeignEvent foreign)
    {
        check(m_context_set_current(m_context));
        check(m_context_destroy(foreign.context));
    }

    DevicePointer allocate(size_t bytes)
    {
        DevicePointer pointer = 0u;
        check(m_mem_alloc(&pointer, bytes));
        track(m_allocations, pointer);
        return pointer;
    }

    // Page-locked host memory, so a copy to it is queued without the host waiting;
    // the caller frees it with freeHost.
    void* allocateHost(size_t bytes)
    {
        void* pointer = nullptr;
        check(m_mem_alloc_host(&pointer, bytes));
        return pointer;
    }

    void freeHost(void* pointer) { check(m_mem_free_host(pointer)); }

    // Whether `pointer` lies in page-locked memory the driver still holds.
    bool isPageLocked(void* pointer)
    {
        unsigned int flags = 0u;
        return m_mem_host_get_flags(&flags, pointer) == 0;
    }

    void uploadAsync(DevicePointer destination, const void* source, size_t bytes, Stream stream)
    {
        check(m_copy_host_to_device(destination, source, bytes, stream));
    }

    void downloadAsync(void* destination, DevicePointer source, size_t bytes, Stream stream)
    {
        check(m_copy_device_to_host(destination, source, bytes, stream));
    }

    void record(Event event, Stream stream) { check(m_event_record(event, stream)); }

    void wait(Stream stream, Event event) { check(m_stream_wait_event(stream, event, 0u)); }

    void synchronize(Event event) { check(m_event_synchronize(event)); }

    int synchronizeRaw(Event event) { return m_event_synchronize(event); }

    bool isComplete(Event event)
    {
        const int result = m_event_query(event);
        if (result == 600)
        {
            return false;
        }
        check(result);
        return true;
    }

    // Launches the scheduling-delay gate kernel from ovspatialquery_cuda_test.ptx next to
    // the executable. Only the native tests call this; the module is loaded on first use so a
    // caller without the test PTX beside it (lesson 11) can still construct the driver.
    void enqueueDelay(Stream stream, uint64_t clocks)
    {
        loadGateKernel();
        void* parameters[] = {&clocks};
        check(m_launch_kernel(m_gate_function, 1u, 1u, 1u, 1u, 1u, 1u, 0u, stream, parameters, nullptr));
    }

    // Starts capturing `stream` into a CUDA graph, in relaxed mode so that no
    // other driver call on this thread disturbs the capture.
    void beginCapture(Stream stream) { check(m_stream_begin_capture(stream, kRelaxedCapture)); }

    // Ends the capture `beginCapture` started and returns how many nodes it
    // recorded; the graph is destroyed.
    size_t endCapture(Stream stream)
    {
        void* graph = nullptr;
        check(m_stream_end_capture(stream, &graph));
        size_t    count   = 0u;
        const int counted = m_graph_get_nodes(graph, nullptr, &count);
        check(m_graph_destroy(graph));
        check(counted);
        return count;
    }

    void enqueueHostFunction(Stream stream, void (*function)(void*), void* data)
    {
        check(m_launch_host_function(stream, function, data));
    }

    void free(DevicePointer pointer)
    {
        check(m_mem_free(pointer));
        untrack(m_allocations, pointer);
    }

    void destroyStream(Stream stream)
    {
        check(m_stream_destroy(stream));
        untrack(m_streams, stream);
    }

    void destroyEvent(Event event)
    {
        check(m_event_destroy(event));
        untrack(m_events, event);
    }

  private:
    static constexpr unsigned int kNonBlockingStream  = 1u;
    static constexpr unsigned int kDisableTimingEvent = 2u;
    static constexpr int          kRelaxedCapture     = 2; // CU_STREAM_CAPTURE_MODE_RELAXED

    void loadGateKernel()
    {
        if (m_gate_function != nullptr)
        {
            return;
        }
        // The test PTX is copied next to every CUDA test executable that gates a stream.
        // It is never installed; the shipped ovspatialquery_cuda.ptx has no gate kernel.
#if defined(_WIN32)
        std::wstring executable_path(32768u, L'\0');
        const DWORD  path_length =
            GetModuleFileNameW(nullptr, executable_path.data(), static_cast<DWORD>(executable_path.size()));
        if (path_length == 0u || path_length == executable_path.size())
        {
            throw std::runtime_error("CUDA test executable path is unavailable");
        }
        executable_path.resize(path_length);
        const std::filesystem::path executable(executable_path);
#else
        std::error_code             error;
        const std::filesystem::path executable = std::filesystem::read_symlink("/proc/self/exe", error);
        if (error)
        {
            throw std::runtime_error("CUDA test executable path is unavailable");
        }
#endif
        const std::filesystem::path ptx = executable.parent_path() / "ovspatialquery_cuda_test.ptx";
        if (m_module_load(&m_gate_module, ptx.string().c_str()) != 0 ||
            m_module_get_function(&m_gate_function, m_gate_module, "ovspatialquery_cuda_scheduling_delay") != 0)
        {
            throw std::runtime_error("CUDA scheduling gate kernel is unavailable "
                                     "(ovspatialquery_cuda_test.ptx)");
        }
    }

    void rollbackConstruction() noexcept
    {
        if (m_gate_module != nullptr && m_module_unload != nullptr)
        {
            m_module_unload(m_gate_module);
            m_gate_module = nullptr;
        }
        if (m_context_selected && m_context_set_current != nullptr)
        {
            m_context_set_current(m_previous_context);
            m_context_selected = false;
        }
        if (m_primary_retained && m_primary_release != nullptr)
        {
            m_primary_release(m_device);
            m_primary_retained = false;
        }
        if (m_module != nullptr)
        {
#if defined(_WIN32)
            FreeLibrary(m_module);
#else
            dlclose(m_module);
#endif
            m_module = nullptr;
        }
    }

    template <typename Function>
    void load(Function& function, const char* name)
    {
#if defined(_WIN32)
        function = reinterpret_cast<Function>(GetProcAddress(m_module, name));
#else
        function = reinterpret_cast<Function>(reinterpret_cast<std::uintptr_t>(dlsym(m_module, name)));
#endif
        if (function == nullptr)
        {
            throw std::runtime_error("CUDA driver export is unavailable");
        }
    }

    static void check(int result)
    {
        if (result != 0)
        {
            throw std::runtime_error("CUDA driver call failed");
        }
    }

    // Removes a handle the caller destroyed explicitly from RAII tracking, so the destructor
    // does not attempt to destroy it a second time.
    template <typename T>
    void track(std::vector<T>& tracked, T value)
    {
        if (m_tracked)
        {
            tracked.push_back(value);
        }
    }

    template <typename T>
    static bool untrack(std::vector<T>& tracked, T value) noexcept
    {
        const typename std::vector<T>::iterator position = std::find(tracked.begin(), tracked.end(), value);
        if (position == tracked.end())
        {
            return false;
        }
        tracked.erase(position);
        return true;
    }

    LibraryHandle              m_module           = nullptr;
    Context                    m_context          = nullptr;
    Context                    m_previous_context = nullptr;
    bool                       m_tracked          = false;
    bool                       m_primary_retained = false;
    int                        m_device           = 0;
    bool                       m_context_selected = false;
    void*                      m_gate_module      = nullptr;
    void*                      m_gate_function    = nullptr;
    std::vector<Stream>        m_streams;
    std::vector<Event>         m_events;
    std::vector<DevicePointer> m_allocations;
    int (*m_init)(unsigned int)                                              = nullptr;
    int (*m_device_get)(int*, int)                                           = nullptr;
    int (*m_context_get_current)(Context*)                                   = nullptr;
    int (*m_primary_retain)(Context*, int)                                   = nullptr;
    int (*m_primary_release)(int)                                            = nullptr;
    int (*m_context_set_current)(Context)                                    = nullptr;
    int (*m_context_create)(Context*, unsigned int, int)                     = nullptr;
    int (*m_context_destroy)(Context)                                        = nullptr;
    int (*m_stream_create)(Stream*, unsigned int)                            = nullptr;
    int (*m_stream_destroy)(Stream)                                          = nullptr;
    int (*m_event_create)(Event*, unsigned int)                              = nullptr;
    int (*m_event_record)(Event, Stream)                                     = nullptr;
    int (*m_event_synchronize)(Event)                                        = nullptr;
    int (*m_event_query)(Event)                                              = nullptr;
    int (*m_event_elapsed_time)(float*, Event, Event)                        = nullptr;
    int (*m_event_destroy)(Event)                                            = nullptr;
    int (*m_stream_wait_event)(Stream, Event, unsigned int)                  = nullptr;
    int (*m_module_load)(void**, const char*)                                = nullptr;
    int (*m_module_unload)(void*)                                            = nullptr;
    int (*m_module_get_function)(void**, void*, const char*)                 = nullptr;
    int (*m_launch_kernel)(void*, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, Stream, void**, void**)             = nullptr;
    int (*m_launch_host_function)(Stream, void (*)(void*), void*)            = nullptr;
    int (*m_mem_alloc)(DevicePointer*, size_t)                               = nullptr;
    int (*m_mem_free)(DevicePointer)                                         = nullptr;
    int (*m_mem_alloc_host)(void**, size_t)                                  = nullptr;
    int (*m_mem_free_host)(void*)                                            = nullptr;
    int (*m_mem_host_get_flags)(unsigned int*, void*)                        = nullptr;
    int (*m_copy_host_to_device)(DevicePointer, const void*, size_t, Stream) = nullptr;
    int (*m_copy_device_to_host)(void*, DevicePointer, size_t, Stream)       = nullptr;
    int (*m_stream_begin_capture)(Stream, int)                               = nullptr;
    int (*m_stream_end_capture)(Stream, void**)                              = nullptr;
    int (*m_graph_get_nodes)(void*, void**, size_t*)                         = nullptr;
    int (*m_graph_destroy)(void*)                                            = nullptr;
};

#endif
