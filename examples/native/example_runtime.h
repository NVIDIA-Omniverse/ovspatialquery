/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "example_evidence.h"

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovspatialquery/ovspatialquery.h>
#include <ovstage/ovstage.h>

#include "example_ovstage.h"
#include <ovx/path_dictionary/path_dictionary.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace example_runtime
{

// OvRTX 0.6 handles are integers; 0.7 handles are one-member structs. Zero is invalid in both.
inline uint64_t rawHandle(uint64_t handle) { return handle; }
template <typename Handle>
inline auto rawHandle(Handle handle) -> decltype(handle.raw_id)
{
    return handle.raw_id;
}

constexpr char kRenderProduct[] = "/Render/CameraProduct";
constexpr char kRenderVar[]     = "/Render/CameraProduct/LdrColor";
constexpr char kCameraPath[]    = "/Render/Camera";

inline std::string& latestScene()
{
    static std::string scene;
    return scene;
}

inline bool succeeded(ovrtx_result_t result) { return result.status == OVRTX_API_SUCCESS; }

inline void require(bool condition, const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

// Destroys a plan or pair reporter. A CUDA operation retired with ovspatialquery_destroy_op_after_cuda
// gives its result slot back on the library's reaper thread, and until then the destroy returns
// BUSY; wait up to 10 seconds for it, as the Python close() does.
template <typename Destroy>
bool destroyWhenRetired(Destroy destroy)
{
    const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    do
    {
        const ovspatialquery_api_status_t status = destroy().status;
        if (status != OVSPATIALQUERY_API_BUSY)
        {
            return status == OVSPATIALQUERY_API_SUCCESS;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

inline bool sameString(ovx_string_t value, const char* literal)
{
    const size_t length = std::char_traits<char>::length(literal);
    return value.ptr != nullptr && value.length == length &&
           std::char_traits<char>::compare(value.ptr, literal, length) == 0;
}

inline bool waitStageOperation(ovstage_instance_t* stage, ovstage_enqueue_result_t operation)
{
    if (operation.status != OVSTAGE_OK || operation.op_index == OVSTAGE_INVALID_OP_ID)
    {
        return false;
    }
    ovstage_op_wait_result_t   result{};
    const ovstage_api_status_t waited = example_ovstage::table(stage)->wait_op(
        example_ovstage::context(stage), operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result);
    const ovstage_api_status_t released =
        example_ovstage::table(stage)->release_op(example_ovstage::context(stage), operation.op_index);
    return waited == OVSTAGE_OK && result.error_op_id_count == 0u && released == OVSTAGE_OK;
}

inline void waitPopulationOpen(ovstage_instance_t* stage, ovstage_population_enqueue_result_t operation)
{
    require(operation.status == OVSTAGE_OK && operation.op_index != OVSTAGE_POPULATION_INVALID_OP_ID,
            "Ovstage rejected inline USDA population");
    ovstage_population_op_wait_result_t result{};
    require(ovstage_population_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &result) == OVSTAGE_OK &&
                result.error_op_id_count == 0u,
            "Ovstage inline USDA population failed");
}

inline void waitOvrtx(ovrtx_renderer_t* renderer, ovrtx_enqueue_result_t operation)
{
    require(operation.status == OVRTX_API_SUCCESS && rawHandle(operation.op_index) != 0u,
            "OvRTX rejected an attached-stage operation");
    ovrtx_op_wait_result_t waited{};
    require(succeeded(ovrtx_wait_op(renderer, operation.op_index, ovrtx_timeout_infinite, &waited)) &&
                waited.num_error_ops == 0u && rawHandle(waited.lowest_pending_op_id) == 0u,
            "OvRTX attached-stage operation failed");
}

inline std::vector<uint8_t> renderFrame(ovrtx_renderer_t* renderer, ovstage_ordinal_t ordinal)
{
    const ovx_string_t               renderProduct{kRenderProduct, sizeof(kRenderProduct) - 1u};
    const ovrtx_render_product_set_t products{&renderProduct, 1u};
    ovrtx_step_result_handle_t       handle{};
    const ovrtx_enqueue_result_t operation = ovrtx_step_with_stage(renderer, products, 1.0 / 60.0, ordinal, &handle);
    require(operation.status == OVRTX_API_SUCCESS && rawHandle(handle) != 0u, "OvRTX rejected the render step");
    ovrtx_render_product_set_outputs_t outputs{};
    require(succeeded(ovrtx_fetch_results(renderer, handle, ovrtx_timeout_infinite, &outputs)) &&
                outputs.status == OVRTX_EVENT_COMPLETED,
            "OvRTX render result failed");

    std::vector<uint8_t> pixels;
    for (size_t productIndex = 0u; productIndex < outputs.output_count && pixels.empty(); ++productIndex)
    {
        const ovrtx_render_product_output_t& product = outputs.outputs[productIndex];
        for (size_t frameIndex = 0u; frameIndex < product.output_frame_count && pixels.empty(); ++frameIndex)
        {
            const ovrtx_render_product_frame_output_t& frame = product.output_frames[frameIndex];
            for (size_t varIndex = 0u; varIndex < frame.render_var_count && pixels.empty(); ++varIndex)
            {
                const ovrtx_render_product_render_var_output_t& variable = frame.output_render_vars[varIndex];
                if (!sameString(variable.render_var_path, kRenderVar))
                {
                    continue;
                }
                const ovrtx_map_output_description_t mapDesc{OVRTX_MAP_DEVICE_TYPE_CPU, 0u};
                ovrtx_render_var_output_t            mapped{};
                require(succeeded(ovrtx_map_render_var_output(renderer, variable.output_handle, &mapDesc,
                                                              ovrtx_timeout_infinite, &mapped)) &&
                            mapped.status == OVRTX_EVENT_COMPLETED && mapped.num_tensors > 0u &&
                            mapped.tensors != nullptr && mapped.tensors[0].dl != nullptr,
                        "OvRTX LdrColor map failed");
                try
                {
                    pixels = example_evidence::copyRgba(*mapped.tensors[0].dl);
                }
                catch (...)
                {
                    if (rawHandle(mapped.map_handle) != 0u)
                    {
                        ovrtx_unmap_render_var_output(renderer, mapped.map_handle, {});
                    }
                    ovrtx_destroy_results(renderer, handle);
                    throw;
                }
                require(rawHandle(mapped.map_handle) == 0u ||
                            succeeded(ovrtx_unmap_render_var_output(renderer, mapped.map_handle, {})),
                        "OvRTX LdrColor unmap failed");
            }
        }
    }
    const bool destroyed = succeeded(ovrtx_destroy_results(renderer, handle));
    require(!pixels.empty(), "OvRTX returned no LdrColor output");
    require(destroyed, "OvRTX result destroy failed");
    return pixels;
}

class Runtime
{
  public:
    Runtime(std::string name, std::string scene) : m_name(std::move(name)), m_scene(std::move(scene))
    {
        latestScene()                        = m_scene;
        const std::string          ovrtxRoot = (std::filesystem::path(OVRTX_PACKAGE_ROOT) / "bin").string();
        const ovrtx_config_entry_t entries[] = {
            ovrtx_config_entry_binary_package_root_path(ovx_string_t{ovrtxRoot.data(), ovrtxRoot.size()}),
            ovrtx_config_entry_sync_mode(true),
            ovrtx_config_entry_active_cuda_gpus(ovx_string_t{"0", sizeof("0") - 1u}),
        };
        const ovrtx_config_t config{entries, sizeof(entries) / sizeof(entries[0])};
        require(succeeded(ovrtx_register_schema_paths(&config)), "ovrtx_register_schema_paths failed");
        require(succeeded(ovrtx_initialize(&config)), "ovrtx_initialize failed");
        m_ovrtxInitialized = true;
        try
        {
            require(succeeded(ovrtx_create_renderer(&config, &m_renderer)) && m_renderer != nullptr,
                    "ovrtx_create_renderer failed");
            const std::string            ovstageRoot = (std::filesystem::path(OVSTAGE_PACKAGE_ROOT) / "bin").string();
            const ovstage_config_entry_t stageEntry =
                ovstage_config_entry_binary_package_root_path(ovx_string_t{ovstageRoot.data(), ovstageRoot.size()});
            const ovstage_config_t stageConfig{&stageEntry, 1u};
            require(ovstage_initialize(&stageConfig) == OVSTAGE_OK, "ovstage_initialize failed");
            m_ovstageInitialized = true;
            ovstage_instance_desc_t stageDesc{};
            stageDesc.name = m_name.c_str();
            require(ovstage_create_instance(&stageDesc, &m_stage) == OVSTAGE_OK && m_stage != nullptr,
                    "ovstage_create_instance failed");
            require(succeeded(ovrtx_attach_ovstage(m_renderer, m_stage)), "ovrtx_attach_ovstage failed");
            m_attached = true;
        }
        catch (...)
        {
            std::vector<std::string> ignored;
            close(ignored);
            throw;
        }
    }

    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;

    ~Runtime()
    {
        std::vector<std::string> ignored;
        close(ignored);
    }

    ovstage_instance_t* stage() const { return m_stage; }

    void openAndCompute(ovstage_ordinal_t openOrdinal = 1u, ovstage_ordinal_t hierarchyOrdinal = 2u)
    {
        require(m_stage != nullptr, "presentation runtime is closed");
        waitPopulationOpen(m_stage, example_ovstage::openUsda(m_stage, ovx_string_t{m_scene.data(), m_scene.size()},
                                                              openOrdinal, 0.0));
        ovstage_write_floor_desc_t floor{};
        floor.ordinal = openOrdinal;
        floor.scope   = OVSTAGE_SCOPE_ALL;
        require(waitStageOperation(m_stage, ovstage_advance_write_floor(m_stage, &floor)),
                "Ovstage initial write floor failed");
        require(waitStageOperation(m_stage,
                                   ovstage_compute_hierarchy(m_stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU,
                                                             openOrdinal, hierarchyOrdinal)),
                "Ovstage hierarchy computation failed");
        floor.ordinal = hierarchyOrdinal;
        require(waitStageOperation(m_stage, ovstage_advance_write_floor(m_stage, &floor)),
                "Ovstage hierarchy write floor failed");
    }

    std::vector<uint8_t> render(int frames, ovstage_ordinal_t ordinal)
    {
        require(frames > 0, "frames must be positive");
        require(m_renderer != nullptr, "presentation runtime is closed");
        waitOvrtx(m_renderer, ovrtx_update_from_stage(m_renderer, ordinal));
        std::vector<uint8_t> pixels;
        for (int frame = 0; frame < frames; ++frame)
        {
            pixels = renderFrame(m_renderer, ordinal);
        }
        return pixels;
    }

    void prepareOrbit()
    {
        require(m_stage != nullptr, "presentation runtime is closed");
        path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(m_stage);
        require(dictionary != nullptr, "Ovstage path dictionary is unavailable");
        const ovx_string_t cameraPath{kCameraPath, sizeof(kCameraPath) - 1u};
        ovx_api_result_t   pathResult =
            dictionary->vtable->create_path_list_from_strings(dictionary->context, &cameraPath, 1u, &m_orbitPathList);
        const bool pathReady = pathResult.status == OVX_API_SUCCESS && m_orbitPathList != OVX_INVALID_PRIMPATH_LIST;
        if (pathResult.error.ptr != nullptr)
        {
            dictionary->vtable->release_error(dictionary->context, pathResult.error);
        }
        require(pathReady, "could not create camera path list");
        require(ovstage_query_from_path_list(m_stage, m_orbitPathList, &m_orbitQuery) == OVSTAGE_OK,
                "could not create camera query");
        constexpr char     attributeName[] = "omni:xform";
        const ovx_string_t attributeString{attributeName, sizeof(attributeName) - 1u};
        ovx_api_result_t   tokenResult = dictionary->vtable->create_tokens_from_strings(
            dictionary->context, &attributeString, 1u, &m_orbitAttribute);
        const bool tokenReady = tokenResult.status == OVX_API_SUCCESS && m_orbitAttribute != OVX_INVALID_TOKEN;
        if (tokenResult.error.ptr != nullptr)
        {
            dictionary->vtable->release_error(dictionary->context, tokenResult.error);
        }
        require(tokenReady, "could not intern camera transform attribute");
        m_orbitStarted = std::chrono::steady_clock::now();
    }

    std::vector<uint8_t> nextOrbitFrame()
    {
        require(m_orbitQuery != OVSTAGE_INVALID_QUERY_HANDLE, "interactive orbit is not prepared");
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - m_orbitStarted).count();
        const double angle   = 1.0636978224 + elapsed * 0.12;
        const std::array<double, 3> eye{
            20.5912602819 * std::cos(angle),
            6.0,
            20.5912602819 * std::sin(angle),
        };
        std::array<double, 3> forward{-eye[0], -eye[1], -eye[2]};
        const double          forwardLength =
            std::sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
        for (double& value : forward)
        {
            value /= forwardLength;
        }
        std::array<double, 3> right{-forward[2], 0.0, forward[0]};
        const double          rightLength = std::sqrt(right[0] * right[0] + right[2] * right[2]);
        for (double& value : right)
        {
            value /= rightLength;
        }
        const std::array<double, 3> upward{
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        };
        std::array<double, 16> matrix{
            right[0],    right[1],    right[2],    0.0, upward[0], upward[1], upward[2], 0.0,
            -forward[0], -forward[1], -forward[2], 0.0, eye[0],    eye[1],    eye[2],    1.0,
        };
        int64_t        shape = 1;
        const DLTensor tensor{
            matrix.data(), DLDevice{kDLCPU, 0}, 1, DLDataType{kDLFloat, 64u, 16u}, &shape, nullptr, 0u,
        };
        ovstage_write_data_t data{};
        data.tensors      = &tensor;
        data.tensor_count = 1u;
        data.semantic     = OVSTAGE_SEMANTIC_MATRIX;
        data.is_array     = false;
        const decltype(ovstage_predicate_t::attribute) attribute{m_orbitAttribute, {}};
        ++m_orbitOrdinal;
        require(waitStageOperation(m_stage, ovstage_write_attribute(m_stage, m_orbitQuery, attribute, m_orbitOrdinal,
                                                                    data, OVSTAGE_PRIM_MODE_UPSERT)),
                "camera-only orbit write failed");
        // OvRTX consumes the global stage ordinal. This isolated presentation
        // stage has only the camera write at this ordinal, so advancing ALL
        // exposes no spatial publication.
        const ovstage_write_floor_desc_t floor{
            m_orbitOrdinal,
            OVSTAGE_SCOPE_ALL,
            nullptr,
            0u,
        };
        require(waitStageOperation(m_stage, ovstage_advance_write_floor(m_stage, &floor)),
                "camera-only orbit floor failed");
        return render(1, m_orbitOrdinal);
    }

    ovx_primpath_t pathFromString(const std::string& value) const
    {
        require(m_stage != nullptr, "presentation runtime is closed");
        path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(m_stage);
        require(dictionary != nullptr, "Ovstage path dictionary is unavailable");
        const ovx_string_t     string{value.data(), value.size()};
        ovx_primpath_t         path = OVX_INVALID_PRIMPATH;
        const ovx_api_result_t result =
            dictionary->vtable->create_paths_from_strings(dictionary->context, &string, 1u, &path);
        const bool succeeded = result.status == OVX_API_SUCCESS && path != OVX_INVALID_PRIMPATH;
        if (result.error.ptr != nullptr)
        {
            dictionary->vtable->release_error(dictionary->context, result.error);
        }
        require(succeeded, "Ovstage path interning failed");
        return path;
    }

    void closeRenderer(std::vector<std::string>& errors)
    {
        if (m_renderer != nullptr && m_attached)
        {
            if (!succeeded(ovrtx_detach_ovstage(m_renderer)))
            {
                errors.emplace_back("detach renderer failed");
            }
            m_attached = false;
        }
        if (m_renderer != nullptr)
        {
            if (!succeeded(ovrtx_destroy_renderer(m_renderer)))
            {
                errors.emplace_back("destroy renderer failed");
            }
            m_renderer = nullptr;
        }
        if (m_ovrtxInitialized)
        {
            if (!succeeded(ovrtx_shutdown()))
            {
                errors.emplace_back("shutdown OvRTX failed");
            }
            m_ovrtxInitialized = false;
        }
    }

    void close(std::vector<std::string>& errors)
    {
        if (m_stage != nullptr && m_orbitQuery != OVSTAGE_INVALID_QUERY_HANDLE)
        {
            if (!waitStageOperation(m_stage, ovstage_release_query(m_stage, m_orbitQuery)))
            {
                errors.emplace_back("release camera query failed");
            }
            m_orbitQuery = OVSTAGE_INVALID_QUERY_HANDLE;
        }
        if (m_stage != nullptr && m_orbitPathList != OVX_INVALID_PRIMPATH_LIST)
        {
            path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(m_stage);
            if (dictionary == nullptr)
            {
                errors.emplace_back("release camera path list failed");
            }
            else
            {
                ovx_api_result_t result =
                    dictionary->vtable->release_path_list_reference(dictionary->context, m_orbitPathList);
                if (result.error.ptr != nullptr)
                {
                    dictionary->vtable->release_error(dictionary->context, result.error);
                }
                if (result.status != OVX_API_SUCCESS)
                {
                    errors.emplace_back("release camera path list failed");
                }
            }
            m_orbitPathList = OVX_INVALID_PRIMPATH_LIST;
        }
        closeRenderer(errors);
        if (m_stage != nullptr)
        {
            if (ovstage_destroy_instance(m_stage) != OVSTAGE_OK)
            {
                errors.emplace_back("destroy stage failed");
            }
            m_stage = nullptr;
        }
        if (m_ovstageInitialized)
        {
            if (ovstage_shutdown() != OVSTAGE_OK)
            {
                errors.emplace_back("shutdown Ovstage failed");
            }
            m_ovstageInitialized = false;
        }
    }

  private:
    std::string                           m_name;
    std::string                           m_scene;
    ovstage_instance_t*                   m_stage              = nullptr;
    ovrtx_renderer_t*                     m_renderer           = nullptr;
    bool                                  m_attached           = false;
    bool                                  m_ovrtxInitialized   = false;
    bool                                  m_ovstageInitialized = false;
    ovx_primpath_list_t                   m_orbitPathList      = OVX_INVALID_PRIMPATH_LIST;
    ovstage_query_handle_t                m_orbitQuery         = OVSTAGE_INVALID_QUERY_HANDLE;
    ovx_token_t                           m_orbitAttribute     = OVX_INVALID_TOKEN;
    ovstage_ordinal_t                     m_orbitOrdinal       = 2u;
    std::chrono::steady_clock::time_point m_orbitStarted{};
};

inline void presentInteractive(const std::string& title, const std::vector<uint8_t>& canonicalPixels)
{
    require(!latestScene().empty(), "no example scene is available for orbit");
    example_evidence::requireInteractiveViewportAvailable();
    Runtime orbit(title + ".orbit", latestScene());
    orbit.openAndCompute();
    orbit.prepareOrbit();
    example_evidence::presentInteractive(title, canonicalPixels, [&orbit]() { return orbit.nextOrbitFrame(); });
}

// The "native" member of semantic.json: installed library and header identities plus the
// ovspatialquery, Ovstage and OvRTX header and runtime versions. Call while the stage is alive.
inline std::string nativeIdentityJson(ovstage_instance_t* stage)
{
    uint32_t spatialMajor = 0u;
    uint32_t spatialMinor = 0u;
    uint32_t spatialPatch = 0u;
    require(ovspatialquery_get_version(&spatialMajor, &spatialMinor, &spatialPatch).status ==
                OVSPATIALQUERY_API_SUCCESS,
            "ovspatialquery_get_version failed");
    uint32_t stageMajor = 0u;
    uint32_t stageMinor = 0u;
    uint32_t stagePatch = 0u;
    example_ovstage::table(stage)->get_version(example_ovstage::context(stage), &stageMajor, &stageMinor, &stagePatch);
    uint32_t rtxMajor = 0u;
    uint32_t rtxMinor = 0u;
    uint32_t rtxPatch = 0u;
    ovrtx_get_version(&rtxMajor, &rtxMinor, &rtxPatch);
    const std::filesystem::path libraryPath = OVSPATIALQUERY_LIBRARY_PATH;
    require(example_evidence::sha256File(libraryPath) == OVSPATIALQUERY_LIBRARY_SHA256,
            "runtime library digest differs from configured verified install");
    std::ostringstream json;
    json << "{\"ovrtx\":{\"header\":[" << OVRTX_VERSION_MAJOR << "," << OVRTX_VERSION_MINOR << ","
         << OVRTX_VERSION_PATCH << "],\"package_root\":" << example_evidence::jsonString(OVRTX_PACKAGE_ROOT)
         << ",\"runtime\":[" << rtxMajor << "," << rtxMinor << "," << rtxPatch
         << "]},\"ovspatialquery\":{\"companion_header\":"
         << example_evidence::jsonString(OVSPATIALQUERY_COMPANION_HEADER_PATH)
         << ",\"companion_header_sha256\":" << example_evidence::jsonString(OVSPATIALQUERY_COMPANION_HEADER_SHA256)
         << ",\"core_header\":" << example_evidence::jsonString(OVSPATIALQUERY_CORE_HEADER_PATH)
         << ",\"core_header_sha256\":" << example_evidence::jsonString(OVSPATIALQUERY_CORE_HEADER_SHA256)
         << ",\"library\":" << example_evidence::jsonString(libraryPath.string())
         << ",\"library_sha256\":" << example_evidence::jsonString(OVSPATIALQUERY_LIBRARY_SHA256) << ",\"runtime\":["
         << spatialMajor << "," << spatialMinor << "," << spatialPatch << "]},\"ovstage\":{\"header\":["
         << OVSTAGE_VERSION_MAJOR << "," << OVSTAGE_VERSION_MINOR << "," << OVSTAGE_VERSION_PATCH
         << "],\"package_root\":" << example_evidence::jsonString(OVSTAGE_PACKAGE_ROOT) << ",\"runtime\":["
         << stageMajor << "," << stageMinor << "," << stagePatch << "]}}";
    return json.str();
}

} // namespace example_runtime
