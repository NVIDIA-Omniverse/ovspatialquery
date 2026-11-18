/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Linux procfs memory sampling and JSON printing of snapshots and deltas.
#include "measurement.h"

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace bench
{
static ProcessMemory processMemory()
{
    // This is an executable-wide footprint sample, not backend allocator ownership.
    ProcessMemory result;
#ifndef _WIN32
    std::ifstream rollup("/proc/self/smaps_rollup");
    std::string   line;
    uint64_t      privateBytes = 0;
    bool          havePrivate  = false;
    while (std::getline(rollup, line))
    {
        const auto parse = [&line](const char* name) -> std::optional<uint64_t>
        {
            if (line.rfind(name, 0) != 0)
                return std::nullopt;
            uint64_t           kib = 0;
            std::istringstream stream(line.substr(std::char_traits<char>::length(name)));
            if (!(stream >> kib))
                return std::nullopt;
            return kib * 1024;
        };
        if (const auto rssValue = parse("Rss:"))
            result.rssBytes = rssValue;
        else if (const auto pssValue = parse("Pss:"))
            result.pssBytes = pssValue;
        else if (const auto privateCleanValue = parse("Private_Clean:"))
        {
            privateBytes += *privateCleanValue;
            havePrivate = true;
        }
        else if (const auto privateDirtyValue = parse("Private_Dirty:"))
        {
            privateBytes += *privateDirtyValue;
            havePrivate = true;
        }
        else if (const auto privateHugeValue = parse("Private_Hugetlb:"))
        {
            privateBytes += *privateHugeValue;
            havePrivate = true;
        }
    }
    if (havePrivate)
        result.privateBytes = privateBytes;
#endif
    return result;
}

static void printMemorySnapshot(const MemorySnapshot& snapshot)
{
    std::cout << "{\"rss_bytes\":";
    printOptionalValue(snapshot.process.rssBytes);
    std::cout << ",\"private_bytes\":";
    printOptionalValue(snapshot.process.privateBytes);
    std::cout << ",\"pss_bytes\":";
    printOptionalValue(snapshot.process.pssBytes);
    std::cout << ",\"ovspatialquery_host_charged_bytes\":";
    printOptionalValue(snapshot.ovspatialqueryHostChargedBytes);
    std::cout << ",\"ovspatialquery_device_charged_bytes\":";
    printOptionalValue(snapshot.ovspatialqueryDeviceChargedBytes);
    std::cout << '}';
}

std::optional<int64_t> deltaBytes(const std::optional<uint64_t>& value, const std::optional<uint64_t>& baseline)
{
    if (!value || !baseline)
        return std::nullopt;
    const uint64_t magnitude = *value >= *baseline ? *value - *baseline : *baseline - *value;
    if (magnitude > uint64_t(std::numeric_limits<int64_t>::max()))
        return std::nullopt;
    return *value >= *baseline ? int64_t(magnitude) : -int64_t(magnitude);
}

MemoryDelta memoryDelta(const MemorySnapshot& snapshot, const MemorySnapshot& baseline)
{
    return {deltaBytes(snapshot.process.rssBytes, baseline.process.rssBytes),
            deltaBytes(snapshot.process.privateBytes, baseline.process.privateBytes),
            deltaBytes(snapshot.process.pssBytes, baseline.process.pssBytes),
            deltaBytes(snapshot.ovspatialqueryHostChargedBytes, baseline.ovspatialqueryHostChargedBytes)};
}

static void printMemoryDelta(const MemoryDelta& delta)
{
    std::cout << "{\"rss_bytes\":";
    printOptionalValue(delta.rssBytes);
    std::cout << ",\"private_bytes\":";
    printOptionalValue(delta.privateBytes);
    std::cout << ",\"pss_bytes\":";
    printOptionalValue(delta.pssBytes);
    std::cout << ",\"ovspatialquery_host_charged_bytes\":";
    printOptionalValue(delta.ovspatialqueryHostChargedBytes);
    std::cout << '}';
}

static uint64_t peakRssBytes()
{
    uint64_t peak = 0;
#ifndef _WIN32
    // ru_maxrss can retain the Python launcher's pre-exec high-water mark.
    // VmHWM belongs to this executable's address space and excludes that parent floor.
    std::ifstream processStatus("/proc/self/status");
    std::string   line;
    while (std::getline(processStatus, line))
    {
        if (line.rfind("VmHWM:", 0) == 0)
        {
            uint64_t kib = 0;
            std::istringstream(line.substr(6)) >> kib;
            peak = kib * 1024;
            break;
        }
    }
#endif
    return peak;
}

void printMemoryRecord(bool budgeted, const char* processNote, const MemorySnapshot& baseline,
                       std::initializer_list<MemoryPhase>        phases,
                       std::optional<std::optional<MemoryDelta>> fromBeforeFirstUse)
{
    std::cout << "{\"scope\":\"process_and_ovspatialquery_host_ledger\",\"process_measurement\":"
#ifndef _WIN32
              << "\"linux_procfs_smaps_rollup\""
#else
              << "null"
#endif
              << ",\"host_budget_bytes\":" << (budgeted ? std::to_string(hostBudget) : "null");
    if (processNote)
        std::cout << ",\"process_note\":\"" << processNote << '"';
    std::cout << ",\"phases\":{\"baseline\":";
    printMemorySnapshot(baseline);
    for (const auto& phase : phases)
    {
        std::cout << ",\"" << phase.name << "\":";
        if (phase.snapshot)
            printMemorySnapshot(*phase.snapshot);
        else
            std::cout << "null";
    }
    std::cout << "},\"deltas_from_baseline\":{";
    const char* separator = "";
    for (const auto& phase : phases)
        if (phase.fromBaseline)
        {
            std::cout << separator << '"' << phase.name << "\":";
            separator = ",";
            if (phase.snapshot)
                printMemoryDelta(memoryDelta(*phase.snapshot, baseline));
            else
                std::cout << "null";
        }
    std::cout << '}';
    if (fromBeforeFirstUse)
    {
        std::cout << ",\"deltas_from_before_first_use\":{\"first_use_ready\":";
        if (*fromBeforeFirstUse)
            printMemoryDelta(**fromBeforeFirstUse);
        else
            std::cout << "null";
        std::cout << '}';
    }
    const uint64_t peakRss = peakRssBytes();
    std::cout << "},\"peak_rss_bytes\":" << (peakRss ? std::to_string(peakRss) : "null");
}

MemorySnapshot memorySnapshot(ovspatialquery_instance_handle_t instance)
{
    MemorySnapshot result{processMemory(), std::nullopt, std::nullopt};
    if (instance != OVSPATIALQUERY_INVALID_INSTANCE)
    {
        uint64_t charged = 0;
        check(ovspatialquery_get_instance_memory_usage(instance, {kDLCPU, 0}, &charged),
              "get ovspatialquery host memory usage");
        result.ovspatialqueryHostChargedBytes = charged;
        uint64_t device                       = 0;
        if (ovspatialquery_get_instance_memory_usage(instance, {kDLCUDA, 0}, &device).status ==
            OVSPATIALQUERY_API_SUCCESS)
            result.ovspatialqueryDeviceChargedBytes = device;
    }
    return result;
}
} // namespace bench
