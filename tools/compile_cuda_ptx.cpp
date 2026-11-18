/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Build-time NVRTC driver: loads the packaged NVRTC library by path and
// compiles one CUDA source (ovspatialquery_cuda_kernel.cu, or the test-only
// ovspatialquery_cuda_test_kernel.cu) to PTX. Windows and Linux.
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
using LibraryHandle = HMODULE;
#else
#include <dlfcn.h>
using LibraryHandle = void*;
#endif

#include <ovspatialquery/ovspatialquery.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

using NvrtcProgram                  = void*;
using NvrtcResult                   = int;
constexpr NvrtcResult kNvrtcSuccess = 0;

using CreateProgram     = NvrtcResult (*)(NvrtcProgram*, const char*, const char*, int, const char* const*,
                                      const char* const*);
using CompileProgram    = NvrtcResult (*)(NvrtcProgram, int, const char* const*);
using GetProgramLogSize = NvrtcResult (*)(NvrtcProgram, size_t*);
using GetProgramLog     = NvrtcResult (*)(NvrtcProgram, char*);
using GetPtxSize        = NvrtcResult (*)(NvrtcProgram, size_t*);
using GetPtx            = NvrtcResult (*)(NvrtcProgram, char*);
using DestroyProgram    = NvrtcResult (*)(NvrtcProgram*);

LibraryHandle openLibrary(const std::filesystem::path& path)
{
#if defined(_WIN32)
    if (SetDllDirectoryW(path.parent_path().c_str()) == 0)
    {
        return nullptr;
    }
    return LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_USER_DIRS);
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void closeLibrary(LibraryHandle module)
{
#if defined(_WIN32)
    FreeLibrary(module);
#else
    dlclose(module);
#endif
}

template <typename Function>
Function load(LibraryHandle module, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<Function>(GetProcAddress(module, name));
#else
    return reinterpret_cast<Function>(reinterpret_cast<std::uintptr_t>(dlsym(module, name)));
#endif
}

std::string readFile(const char* path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: compile_cuda_ptx <nvrtc library> <source.cu> <output.ptx>\n";
        return 2;
    }
    const std::filesystem::path nvrtcPath(argv[1]);
    LibraryHandle               module = openLibrary(nvrtcPath);
    if (module == nullptr)
    {
        std::cerr << "failed to load NVRTC from " << nvrtcPath << "\n";
        return 3;
    }
    const CreateProgram     createProgram  = load<CreateProgram>(module, "nvrtcCreateProgram");
    const CompileProgram    compileProgram = load<CompileProgram>(module, "nvrtcCompileProgram");
    const GetProgramLogSize getLogSize     = load<GetProgramLogSize>(module, "nvrtcGetProgramLogSize");
    const GetProgramLog     getLog         = load<GetProgramLog>(module, "nvrtcGetProgramLog");
    const GetPtxSize        getPtxSize     = load<GetPtxSize>(module, "nvrtcGetPTXSize");
    const GetPtx            getPtx         = load<GetPtx>(module, "nvrtcGetPTX");
    const DestroyProgram    destroyProgram = load<DestroyProgram>(module, "nvrtcDestroyProgram");
    if (createProgram == nullptr || compileProgram == nullptr || getLogSize == nullptr || getLog == nullptr ||
        getPtxSize == nullptr || getPtx == nullptr || destroyProgram == nullptr)
    {
        std::cerr << "NVRTC exports are incomplete\n";
        closeLibrary(module);
        return 4;
    }

    const std::string source = readFile(argv[2]);
    if (source.empty())
    {
        std::cerr << "CUDA source is empty\n";
        closeLibrary(module);
        return 5;
    }
    NvrtcProgram program = nullptr;
    NvrtcResult  result  = createProgram(&program, source.c_str(), argv[2], 0, nullptr, nullptr);
    // NVRTC does not need the host declarations or standard-library headers.
    // Compile its geometry tags from the same public enum as the native library.
    const std::string geometryOptions[] = {
        "-DOVSPATIALQUERY_GEOMETRY_SPHERE=" + std::to_string(OVSPATIALQUERY_GEOMETRY_SPHERE),
        "-DOVSPATIALQUERY_GEOMETRY_BOX=" + std::to_string(OVSPATIALQUERY_GEOMETRY_BOX),
        "-DOVSPATIALQUERY_GEOMETRY_CAPSULE=" + std::to_string(OVSPATIALQUERY_GEOMETRY_CAPSULE),
        "-DOVSPATIALQUERY_GEOMETRY_CONVEX=" + std::to_string(OVSPATIALQUERY_GEOMETRY_CONVEX),
        "-DOVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH=" + std::to_string(OVSPATIALQUERY_GEOMETRY_TRIANGLE_MESH),
    };
    std::vector<const char*> options = {
        "--std=c++17",
        "--gpu-architecture=compute_75",
        "--device-as-default-execution-space",
    };
    for (const auto& option : geometryOptions)
    {
        options.push_back(option.c_str());
    }
    if (result == kNvrtcSuccess)
    {
        result = compileProgram(program, static_cast<int>(options.size()), options.data());
    }
    if (result != kNvrtcSuccess)
    {
        size_t logSize = 0u;
        if (program != nullptr && getLogSize(program, &logSize) == kNvrtcSuccess)
        {
            std::vector<char> log(logSize);
            if (!log.empty() && getLog(program, log.data()) == kNvrtcSuccess)
            {
                std::cerr << log.data();
            }
        }
        if (program != nullptr)
        {
            destroyProgram(&program);
        }
        closeLibrary(module);
        return 6;
    }
    size_t ptxSize = 0u;
    result         = getPtxSize(program, &ptxSize);
    std::vector<char> ptx(ptxSize);
    if (result == kNvrtcSuccess)
    {
        result = getPtx(program, ptx.data());
    }
    if (result == kNvrtcSuccess)
    {
        // nvrtcGetPTXSize counts the terminating NUL; the file gets the text only.
        while (!ptx.empty() && ptx.back() == '\0')
        {
            ptx.pop_back();
        }
        std::ofstream output(argv[3], std::ios::binary);
        output.write(ptx.data(), static_cast<std::streamsize>(ptx.size()));
        if (!output)
        {
            result = -1;
        }
    }
    destroyProgram(&program);
    closeLibrary(module);
    return result == kNvrtcSuccess ? 0 : 7;
}
