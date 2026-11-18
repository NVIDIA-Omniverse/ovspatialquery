/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef OVSPATIALQUERY_EXPLICIT_REGISTRATION_OPERATION_H
#define OVSPATIALQUERY_EXPLICIT_REGISTRATION_OPERATION_H

#include "ovspatialquery_explicit_preparation.h"
#include "ovspatialquery_instance_host_ledger.h"
#include "ovspatialquery_instance_operation_runtime.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct ExplicitRegistrationOperationId
{
    ovspatialquery_op_id_t value = OVSPATIALQUERY_INVALID_OP;
};

struct ExplicitRegistrationAdmission
{
    ovspatialquery_api_status_t     outcome = OVSPATIALQUERY_API_INTERNAL;
    ExplicitRegistrationOperationId operation;
};

struct ExplicitRegistrationRuntimeTask;

/**
 * Private explicit-registration runtime owner.
 *
 * This module owns immutable task capture, instance operation admission, and
 * terminal-observed destruction. It deliberately does not project a public
 * registration ABI or materialize geometry: the preparation seam owns those
 * meanings.
 */
class ExplicitRegistrationOperationOwner
{
  public:
    static uint64_t                    fixedResidentByteCount() noexcept;
    static ovspatialquery_api_status_t create(ovspatialquery_instance_handle_t                     instance,
                                              std::unique_ptr<ExplicitRegistrationOperationOwner>& outOwner) noexcept;

    ~ExplicitRegistrationOperationOwner();
    ExplicitRegistrationOperationOwner(const ExplicitRegistrationOperationOwner&)            = delete;
    ExplicitRegistrationOperationOwner& operator=(const ExplicitRegistrationOperationOwner&) = delete;

    ExplicitRegistrationAdmission enqueuePrepared(std::shared_ptr<ExplicitPreparationRegistry> registry,
                                                  ExplicitPreparedRegistration                 registration,
                                                  std::vector<std::shared_ptr<HostLease>>      definitionMemoryLeases,
                                                  std::shared_ptr<HostLease>                   claimMemoryLease,
                                                  std::shared_ptr<HostLease> transientMemoryLease) noexcept;
    ovspatialquery_api_status_t   destroyOperation(ExplicitRegistrationOperationId& operation) noexcept;

  private:
    friend struct ExplicitRegistrationRuntimeTask;
    struct Impl;
    explicit ExplicitRegistrationOperationOwner(std::shared_ptr<Impl> impl) noexcept;

    std::shared_ptr<Impl> mImpl;
};

} // namespace detail
} // namespace ovspatialquery

#endif
