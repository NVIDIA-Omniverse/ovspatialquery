# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import ctypes
import unittest
from ovspatialquery import ApiStatus, HitTruth, OpState, raw
from ovspatialquery.instance import Instance, Operation
from ovspatialquery.types import InstanceConfig


class _Function:
    def __init__(self, implementation) -> None:
        self.implementation = implementation
        self.calls = []

    def __call__(self, *arguments):
        self.calls.append(arguments)
        return self.implementation(*arguments)


class _Library:
    def __init__(self) -> None:
        self.exports = {
            "ovspatialquery_create_instance": _Function(self._create),
            "ovspatialquery_wait_op": _Function(self._wait),
            "ovspatialquery_fetch_raycast_batch": _Function(self._fetch_raycast),
            "ovspatialquery_enqueue_raycast_plan": _Function(self._enqueue_raycast),
            "ovspatialquery_destroy_op": _Function(self._destroy_op),
            "ovspatialquery_destroy_op_after_cuda": _Function(self._destroy_cuda),
            "ovspatialquery_destroy_instance": _Function(lambda _instance: raw.Result(0)),
            "ovspatialquery_get_last_error": _Function(lambda: raw.String(None, 0)),
        }

    @staticmethod
    def _create(_descriptor, output):
        output._obj.value = 41
        return raw.Result(0)

    @staticmethod
    def _wait(_instance, _operation, _timeout, output):
        output._obj.state = OpState.COMPLETED
        return raw.Result(0)

    @staticmethod
    def _destroy_op(_instance, operation):
        operation._obj.value = 0
        return raw.Result(0)

    @staticmethod
    def _fetch_raycast(_instance, _operation, output):
        value = output._obj
        value.common.truth = HitTruth.BOUNDS_CANDIDATE
        value.common.execution_device = raw.DLDevice(raw.K_DL_CUDA, 0)
        value.common.result_ready = raw.CudaEvent(88)
        shape_q = (ctypes.c_int64 * 1)(0)
        shape_h = (ctypes.c_int64 * 1)(0)

        def tensor(shape, bits=64):
            return raw.DLTensor(
                data=ctypes.c_void_p(0x1234),
                device=raw.DLDevice(raw.K_DL_CUDA, 0),
                ndim=1,
                dtype=raw.DLDataType(raw.K_DL_UINT, bits, 1),
                shape=shape,
            )

        value.columns.hit_ranges.query_status = tensor(shape_q, 32)
        value.columns.hit_ranges.stored_hit_count = tensor(shape_q)
        value.columns.hit_ranges.hit_offsets = tensor(shape_q)
        value.columns.hit_ranges.hit_counts = tensor(shape_q)
        value.columns.hit_ranges.truncated = tensor(shape_q, 8)
        value.columns.object_ids = tensor(shape_h)
        return raw.Result(0)

    @classmethod
    def _enqueue_raycast(cls, instance, _plan, _batch, output):
        cls._fetch_raycast(instance, 79, output)
        return raw.EnqueueResult(ApiStatus.SUCCESS, 79)

    @staticmethod
    def _destroy_cuda(_instance, operation, event):
        operation._obj.value = 0
        event._obj.value = 0
        return raw.Result(0)


class InstanceLifetimeTests(unittest.TestCase):
    def test_core_instance_and_publication_operation_are_distinct(self) -> None:
        library = _Library()
        instance = Instance.create(library, InstanceConfig())
        operation = Operation(instance, 77, keepalive=("submission",))
        self.assertEqual(operation.wait(), OpState.COMPLETED)
        operation.close()
        self.assertEqual(operation.keepalive, ())
        instance.close()

    def test_cuda_batch_fetch_keeps_device_descriptors_until_consumer_done(self) -> None:
        library = _Library()
        instance = Instance.create(library, InstanceConfig())
        operation = Operation(instance, 78, keepalive=("device-input",))
        result = operation.fetch_raycast_batch()
        self.assertEqual(result.common.result_ready, 88)
        self.assertEqual(result.object_ids.data, 0x1234)
        self.assertEqual(operation.keepalive, ("device-input",))
        operation.close_after_cuda(99)
        self.assertEqual(operation.op_id, 0)
        self.assertEqual(operation.keepalive, ())
        instance.close()


if __name__ == "__main__":
    unittest.main()
