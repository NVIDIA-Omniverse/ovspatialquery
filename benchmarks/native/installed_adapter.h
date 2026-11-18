/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Installed ovspatialquery adapter: explicit registration, scalar calls and reusable CPU plans.
#ifndef BENCH_INSTALLED_ADAPTER_H
#define BENCH_INSTALLED_ADAPTER_H
#include "fixture.h"
#include "physx_adapter.h"

#include <ovspatialquery/ovspatialquery_explicit.h>

#include <memory>
#include <string>
#include <vector>

namespace bench
{
// Input owns its shape so descriptors remain valid until submission completes.
struct Input
{
    std::array<int64_t, 2>       shape{}, strides{};
    ovspatialquery_input_array_t value{};
    // A CUDA column borrows caller-owned device memory until the operation is terminal,
    // the ownership the demo's Warp arrays use; CPU columns are copied on submission.
    Input(void* data, size_t rows, size_t columns, uint8_t code, uint8_t bits, DLDevice device = {kDLCPU, 0})
    {
        shape                = {int64_t(rows), int64_t(columns)};
        strides              = {int64_t(columns ? columns : 1), 1};
        value.tensor.data    = data;
        value.tensor.device  = device;
        value.tensor.ndim    = columns ? 2 : 1;
        value.tensor.dtype   = {code, bits, 1};
        value.tensor.shape   = shape.data();
        value.tensor.strides = strides.data();
        value.ownership      = device.device_type == kDLCUDA ? OVSPATIALQUERY_INPUT_BORROWED_UNTIL_TERMINAL
                                                             : OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION;
    }
    Input(const Input&) = delete;
};
struct InstanceRows
{
    std::vector<uint64_t>                    ids, definitions, bits;
    std::vector<int32_t>                     types;
    std::vector<Vec>                         positions;
    std::vector<Quat>                        rotations;
    std::unique_ptr<Input>                   idInput, defInput, bitInput, typeInput, posInput, rotInput;
    ovspatialquery_explicit_instance_table_t table{};
    InstanceRows(const Fixture& f, size_t count);
};

// DLTensor strides and byte offsets are part of the installed result contract.
template <class T>
T element(const DLTensor& t, size_t row, size_t column = 0)
{
    require(t.data && t.shape && t.device.device_type == kDLCPU, "missing CPU result tensor");
    const size_t stride = t.strides ? size_t(t.strides[0]) : (t.ndim == 2 ? size_t(t.shape[1]) : 1);
    const size_t inner  = t.strides && t.ndim == 2 ? size_t(t.strides[1]) : 1;
    const auto*  values = reinterpret_cast<const T*>(static_cast<const uint8_t*>(t.data) + t.byte_offset);
    return values[row * stride + column * inner];
}
class Spatial
{
  public:
    ovspatialquery_instance_handle_t   instance   = OVSPATIALQUERY_INVALID_INSTANCE;
    ovspatialquery_population_handle_t population = OVSPATIALQUERY_INVALID_POPULATION;
    ovspatialquery_view_handle_t       view       = OVSPATIALQUERY_INVALID_VIEW;
    ovspatialquery_ordinal_t           ordinal    = 0;

  private:
    std::vector<ovspatialquery_overlap_hit_t> overlapHits;
    bool   sphereRegistered = false, boxRegistered = false, capsuleRegistered = false;
    size_t meshDefinitions = 0, convexDefinitions = 0;

  public:
    static void checkRelease(ovspatialquery_result_t result, const char* action) noexcept;
    // Waits for one admitted operation; returns the published ordinal when `publication`.
    ovspatialquery_ordinal_t finish(ovspatialquery_enqueue_result_t admitted, bool publication = false,
                                    const char* action = "registration");
    // A further explicit population on this instance, published at ordinal 1 from `f`.
    ovspatialquery_population_handle_t publishPopulation(const Fixture& f, ovspatialquery_ordinal_t& outOrdinal);
    double                             geometrySetupMs = 0, publicationMs = 0;
    explicit Spatial(const Fixture& f);
    struct InstanceOnly
    {
    };
    // An instance with no registered geometry or population; a source adapter publishes into it.
    explicit Spatial(InstanceOnly);
    ~Spatial();
    ovspatialquery_target_ref_t   target(bool even) const;
    ovspatialquery_query_common_t common(const Fixture& f, const Query& q, ovspatialquery_ordinal_t exact = -1) const;
    ovspatialquery_overlap_desc_t overlap(const Fixture& f, const Query& q, ovspatialquery_ordinal_t exact = -1) const;
    void                          update(const Fixture& f, size_t count);
    void                          update(const Fixture& f, size_t count, ovspatialquery_population_handle_t target,
                                         ovspatialquery_ordinal_t& ioOrdinal);
    void                          scalar(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact = -1);
    // Host-side phases of one CUDA launch: admission call, wait for the library's
    // completion, typed fetch, and the caller's stream wait plus column readback.
    struct CudaTiming
    {
        double enqueue = 0, wait = 0, fetch = 0, readback = 0;
    };
    // One reusable plan on the CPU or on CUDA device 0. The CUDA variant keeps the packed
    // query columns in caller-owned device memory, orders the launch behind its own ready
    // event, mirrors the result columns to the host through the library's completion event,
    // and retires the operation with destroy_op_after_cuda, like the warehouse demo's Warp path.
    class BatchPlan
    {
        struct CudaState;
        Spatial&                          owner;
        ovspatialquery_instance_handle_t  instance;
        DLDevice                          device;
        std::unique_ptr<CudaState>        cuda;
        std::vector<Vec>                  positions;
        std::vector<std::array<float, 4>> rotations;
        std::vector<std::array<float, 3>> directions;
        std::vector<double>               minimums, maximums, radii;
        std::vector<uint64_t>             queryBits;
        std::unique_ptr<Input>            posInput, rotInput, dirInput, minInput, maxInput, bitsInput, radInput;
        void                              mirror(DLTensor& tensor, size_t index);

      public:
        ovspatialquery_batch_plan_handle_t plan = OVSPATIALQUERY_INVALID_BATCH_PLAN;
        size_t                             capacity;
        CudaTiming                         last;
        // BOUNDS_CANDIDATE plans read device-prepared versions, which serve only bounds (ADR-0024).
        ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT;
        BatchPlan(Spatial& source, const Fixture& f, size_t maximumQueries, DLDevice device = {kDLCPU, 0},
                  ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT);
        ~BatchPlan();
        BatchPlan(const BatchPlan&) = delete;
        void pack(const Fixture& f);
        void batch(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact = -1, bool* truncated = nullptr);
    };
    std::unique_ptr<BatchPlan> defaultPlan;
    // Throws when `cuda` is requested but the driver, device or installed CUDA backend is absent.
    void makePlan(const Fixture& f, bool cuda = false,
                  ovspatialquery_query_precision_t precision = OVSPATIALQUERY_QUERY_EXACT);
    void batch(const Fixture& f, Results& result, ovspatialquery_ordinal_t exact = -1);
};

// Whether a CUDA plan can be created here; the message names the failing step otherwise.
bool cudaPlansAvailable(std::string& message);

// Whether every mesh registration from here on carries source-face columns (--mesh-source-faces).
void registerMeshSourceFaces(bool enabled);

void compareAdapters(Fixture& f, Direct& direct, Spatial& spatial, bool ties = false);
} // namespace bench
#endif
