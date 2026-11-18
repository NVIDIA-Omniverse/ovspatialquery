/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ovspatialquery_explicit_registration_operation.h"

#include "ovspatialquery_instance_host_ledger.h"

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

namespace ovspatialquery
{
namespace detail
{

struct ExplicitRegistrationOperationOwner::Impl
{
    struct CaptureState
    {
        std::shared_ptr<ExplicitPreparationRegistry> registry;
        ExplicitPreparedRegistration                 registration;
        std::vector<std::shared_ptr<HostLease>>      definitionMemoryLeases;
        std::shared_ptr<HostLease>                   claimMemoryLease;
        std::shared_ptr<HostLease>                   transientMemoryLease;
    };

    explicit Impl(ovspatialquery_instance_handle_t value) noexcept : instance(value) {}

    ovspatialquery_instance_handle_t instance = OVSPATIALQUERY_INVALID_INSTANCE;

    static uint64_t taskHostByteCount(const CaptureState& state) noexcept;

    ExplicitRegistrationAdmission admit(const std::shared_ptr<Impl>&         self,
                                        const std::shared_ptr<CaptureState>& state) noexcept;
};

struct ExplicitRegistrationRuntimeTask final : OperationTask
{
    InstanceOperationExecution execute() noexcept override;

    std::shared_ptr<ExplicitRegistrationOperationOwner::Impl>               owner;
    std::shared_ptr<ExplicitRegistrationOperationOwner::Impl::CaptureState> capture;
    ovspatialquery_op_id_t                                                  operation = OVSPATIALQUERY_INVALID_OP;
};

uint64_t ExplicitRegistrationOperationOwner::Impl::taskHostByteCount(const CaptureState& state) noexcept
{
    constexpr uint64_t allocationAndNodeAllowance = 3u * 256u;
    uint64_t           total = static_cast<uint64_t>(sizeof(ExplicitRegistrationOperationOwner::Impl::CaptureState)) +
                     static_cast<uint64_t>(sizeof(ExplicitRegistrationRuntimeTask)) + allocationAndNodeAllowance;
    if (state.definitionMemoryLeases.capacity() >
        static_cast<size_t>((UINT64_MAX - total) / sizeof(std::shared_ptr<HostLease>)))
    {
        return UINT64_MAX;
    }
    total += static_cast<uint64_t>(state.definitionMemoryLeases.capacity()) * sizeof(std::shared_ptr<HostLease>);
    return total;
}

ExplicitRegistrationAdmission ExplicitRegistrationOperationOwner::Impl::admit(
    const std::shared_ptr<Impl>& self, const std::shared_ptr<CaptureState>& state) noexcept
{
    ExplicitRegistrationAdmission admission{};
    InstanceOperationReservation  reservation{};
    bool                          reservationActive = false;
    try
    {
        const ovspatialquery_api_status_t reserve =
            reserveInstanceOperation(instance, InstanceOperationKind::explicitRegistration, self.get(), reservation);
        if (reserve != OVSPATIALQUERY_API_SUCCESS)
        {
            // SUCCESS, INVALID_HANDLE, OPERATION_CAPACITY or INTERNAL.
            admission.outcome = reserve;
            return admission;
        }
        reservationActive = true;
        HostLease taskLease;
        if (!reserveInstanceOperationHostTask(instance, taskHostByteCount(*state), taskLease))
        {
            abandonInstanceOperation(instance, reservation);
            return {OVSPATIALQUERY_API_HOST_BUDGET, {}};
        }
        std::shared_ptr<ExplicitRegistrationRuntimeTask> task = std::make_shared<ExplicitRegistrationRuntimeTask>();
        task->owner                                           = self;
        task->capture                                         = state;
        task->operation                                       = reservation.operation;
        if (bindInstanceOperation(instance, reservation, task, std::move(taskLease), true) !=
            OVSPATIALQUERY_API_SUCCESS)
        {
            abandonInstanceOperation(instance, reservation);
            return {OVSPATIALQUERY_API_ERROR, {}};
        }
        admission.outcome         = OVSPATIALQUERY_API_SUCCESS;
        admission.operation.value = reservation.operation;
        reservationActive         = false;
        return admission;
    }
    catch (const std::bad_alloc&)
    {
        if (reservationActive)
        {
            abandonInstanceOperation(instance, reservation);
        }
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, {}};
    }
    catch (...)
    {
        if (reservationActive)
        {
            abandonInstanceOperation(instance, reservation);
        }
        return {OVSPATIALQUERY_API_INTERNAL, {}};
    }
}

ExplicitRegistrationOperationOwner::ExplicitRegistrationOperationOwner(std::shared_ptr<Impl> impl) noexcept
    : mImpl(std::move(impl))
{}

uint64_t ExplicitRegistrationOperationOwner::fixedResidentByteCount() noexcept
{
    constexpr uint64_t sharedControlAndAllocationAllowance = 256u;
    return static_cast<uint64_t>(sizeof(ExplicitRegistrationOperationOwner)) + static_cast<uint64_t>(sizeof(Impl)) +
           sharedControlAndAllocationAllowance;
}

ExplicitRegistrationOperationOwner::~ExplicitRegistrationOperationOwner() = default;

ovspatialquery_api_status_t ExplicitRegistrationOperationOwner::create(
    ovspatialquery_instance_handle_t instance, std::unique_ptr<ExplicitRegistrationOperationOwner>& outOwner) noexcept
{
    if (instance == OVSPATIALQUERY_INVALID_INSTANCE)
    {
        return OVSPATIALQUERY_API_INVALID_ARGUMENT;
    }
    try
    {
        std::shared_ptr<Impl>                               impl = std::make_shared<Impl>(instance);
        std::unique_ptr<ExplicitRegistrationOperationOwner> owner(
            new ExplicitRegistrationOperationOwner(std::move(impl)));
        outOwner = std::move(owner);
        return OVSPATIALQUERY_API_SUCCESS;
    }
    catch (const std::bad_alloc&)
    {
        return OVSPATIALQUERY_API_OUT_OF_MEMORY;
    }
    catch (...)
    {
        return OVSPATIALQUERY_API_INTERNAL;
    }
}

ExplicitRegistrationAdmission ExplicitRegistrationOperationOwner::enqueuePrepared(
    std::shared_ptr<ExplicitPreparationRegistry> registry, ExplicitPreparedRegistration registration,
    std::vector<std::shared_ptr<HostLease>> definitionMemoryLeases, std::shared_ptr<HostLease> claimMemoryLease,
    std::shared_ptr<HostLease> transientMemoryLease) noexcept
{
    if (registry == nullptr || !registration.initialized() ||
        definitionMemoryLeases.size() != registration.definitionCount() ||
        std::any_of(definitionMemoryLeases.begin(), definitionMemoryLeases.end(),
                    [](const std::shared_ptr<HostLease>& lease) { return lease == nullptr || !*lease; }) ||
        claimMemoryLease == nullptr || !*claimMemoryLease || transientMemoryLease == nullptr || !*transientMemoryLease)
    {
        return {OVSPATIALQUERY_API_INVALID_ARGUMENT, {}};
    }
    try
    {
        const std::shared_ptr<Impl::CaptureState> state = std::make_shared<Impl::CaptureState>();
        state->registry                                 = std::move(registry);
        state->registration                             = std::move(registration);
        state->definitionMemoryLeases                   = std::move(definitionMemoryLeases);
        state->claimMemoryLease                         = std::move(claimMemoryLease);
        state->transientMemoryLease                     = std::move(transientMemoryLease);
        return mImpl->admit(mImpl, state);
    }
    catch (const std::bad_alloc&)
    {
        return {OVSPATIALQUERY_API_OUT_OF_MEMORY, {}};
    }
    catch (...)
    {
        return {OVSPATIALQUERY_API_INTERNAL, {}};
    }
}

ovspatialquery_api_status_t ExplicitRegistrationOperationOwner::destroyOperation(
    ExplicitRegistrationOperationId& operation) noexcept
{
    ovspatialquery_op_id_t            value = operation.value;
    const ovspatialquery_api_status_t outcome =
        destroyInstanceOperation(mImpl->instance, InstanceOperationKind::explicitRegistration, mImpl.get(), value);
    if (outcome == OVSPATIALQUERY_API_SUCCESS)
    {
        operation = {};
    }
    return outcome;
}

InstanceOperationExecution ExplicitRegistrationRuntimeTask::execute() noexcept
{
    if (owner == nullptr || capture == nullptr || operation == OVSPATIALQUERY_INVALID_OP)
    {
        return {OVSPATIALQUERY_API_INTERNAL, "explicit registration task ownership is invalid"};
    }
    if (capture->registration.initialized())
    {
        if (capture->registry == nullptr)
        {
            return {OVSPATIALQUERY_API_INTERNAL, "explicit registration registry is unavailable"};
        }
        const std::vector<uint64_t>& definitionBytes         = capture->registration.definitionResidentByteCounts();
        const std::vector<uint64_t>& retainedDefinitionBytes = capture->registration.definitionRetainedByteCounts();
        if (definitionBytes.size() != capture->definitionMemoryLeases.size() ||
            retainedDefinitionBytes.size() != definitionBytes.size())
        {
            return {OVSPATIALQUERY_API_INTERNAL, "explicit definition accounting cardinality changed"};
        }
        std::vector<std::shared_ptr<void>> definitionOwners;
        try
        {
            definitionOwners.assign(capture->definitionMemoryLeases.begin(), capture->definitionMemoryLeases.end());
        }
        catch (const std::bad_alloc&)
        {
            return {OVSPATIALQUERY_API_OUT_OF_MEMORY, "explicit definition owners exhausted memory"};
        }
        const ovspatialquery_api_status_t status = capture->registry->commitRegistration(
            capture->registration, std::move(definitionOwners), capture->claimMemoryLease);
        if (status == OVSPATIALQUERY_API_SUCCESS)
        {
            // Each definition keeps only its retained bytes once committed.
            for (size_t index = 0u; index < definitionBytes.size(); ++index)
            {
                if (!capture->definitionMemoryLeases[index]->shrink(retainedDefinitionBytes[index]))
                {
                    return {OVSPATIALQUERY_API_INTERNAL, "explicit definition staging release failed"};
                }
            }
            return {};
        }
        return {status == OVSPATIALQUERY_API_OUT_OF_MEMORY ? OVSPATIALQUERY_API_OUT_OF_MEMORY
                                                           : OVSPATIALQUERY_API_OPERATION_FAILED,
                status == OVSPATIALQUERY_API_INVALID_ARGUMENT
                    ? "explicit geometry registration conflicts with an existing definition"
                    : "explicit geometry registration commit failed"};
    }
    return {OVSPATIALQUERY_API_INTERNAL, "explicit registration is not prepared"};
}

} // namespace detail
} // namespace ovspatialquery
