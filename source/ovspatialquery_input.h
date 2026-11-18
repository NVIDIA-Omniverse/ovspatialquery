/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_INPUT_H
#define OVSPATIALQUERY_INPUT_H

#include "ovspatialquery_internal.h"
#include "ovspatialquery_population_table.h"

#include <memory>
#include <optional>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

class PreparedInputArray;
class CudaPopulationCapture;

class PreparedPopulationInput
{
  public:
    PreparedPopulationInput();
    ~PreparedPopulationInput();
    PreparedPopulationInput(PreparedPopulationInput&& other) noexcept;
    PreparedPopulationInput& operator=(PreparedPopulationInput&& other) noexcept;

    PreparedPopulationInput(const PreparedPopulationInput&)            = delete;
    PreparedPopulationInput& operator=(const PreparedPopulationInput&) = delete;

    static ovspatialquery_api_status_t prepareCopiedSet(ovspatialquery_ordinal_t targetOrdinal, ObjectTable objects,
                                                        PreparedPopulationInput& outInput);
    static ovspatialquery_api_status_t prepareCopiedSet(ovspatialquery_ordinal_t      targetOrdinal,
                                                        std::vector<PopulationObject> objects,
                                                        PreparedPopulationInput&      outInput);
    // A known exact-geometry charge of `objects` spares retainedByteCount a
    // recount; known changes against the base spare publication a comparison.
    static ovspatialquery_api_status_t prepareCopiedCompleteUpdate(ovspatialquery_ordinal_t baseOrdinal,
                                                                   ovspatialquery_ordinal_t targetOrdinal,
                                                                   ObjectTable              objects,
                                                                   PreparedPopulationInput& outInput,
                                                                   std::optional<uint64_t>  exactGeometryCharge = {},
                                                                   std::optional<VersionChangeRecord> changes   = {});

    bool isSet() const;
    void restrictToUnreadyPopulation();
    bool requiresUnreadyPopulation() const;
    // The base may be the newest admitted device-prepared publication instead
    // of the head, within the population's pending capacity (ADR-0021).
    void                                   chainPendingPublications();
    bool                                   chainsPendingPublications() const;
    void                                   enforceObjectValueNoReuse();
    bool                                   enforcesObjectValueNoReuse() const;
    ovspatialquery_ordinal_t               baseOrdinal() const;
    ovspatialquery_ordinal_t               targetOrdinal() const;
    void                                   setTargetOrdinal(ovspatialquery_ordinal_t ordinal);
    void                                   adoptInputLease(PreparedInputArray lease);
    void                                   adoptCudaCapture(std::shared_ptr<CudaPopulationCapture> capture);
    std::shared_ptr<CudaPopulationCapture> cudaCapture() const;
    uint64_t                               retainedByteCount();
    uint64_t materializationPeakByteCount(uint64_t baseObjectCount, uint64_t retainedByteCount) const;
    // The objects a host-prepared input carries: those of its version.
    uint64_t                    objectCount() const;
    ovspatialquery_api_status_t materialize(const GenerationLease& base, ObjectTable& outObjects,
                                            std::optional<uint64_t>& outGeometryCharge);
    // The changes the input recorded against its base, if it knows them.
    std::optional<VersionChangeRecord> releaseChanges();
    void                               reset();

  private:
    struct Impl;
    explicit PreparedPopulationInput(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
