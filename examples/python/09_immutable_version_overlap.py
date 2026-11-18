# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin an async query to the prior version before publishing the next sample."""

from __future__ import annotations

import sys
from typing import Any

import ovstage
import warp as wp

from ovspatialquery import (
    BatchResultField,
    BatchRowStatus,
    Capability,
    Device,
    DeviceBudget,
    HitMode,
    HitTruth,
    InstanceConfig,
    Library,
    PopulationConfig,
    QueryPrecision,
    RaycastPlanConfig,
    RaycastQuery,
)
from ovspatialquery.types import RaycastBatchReservation, RaycastBatchResult

from _evidence import (
    EvidencePaths,
    HEIGHT,
    WIDTH,
    example_argument_parser,
    parse_example_arguments,
    present_interactive,
    require_python_source,
    validate_evidence,
    write_semantic_json,
)
from _runtime import Runtime, frame_summary

EXAMPLE = "09_immutable_version_overlap"
TARGET_PATH = "/World/Moving"
TIME_DOMAIN = 0xE11
# CUDA work needs a byte budget for the device; the CPU needs none.
CUDA_BUDGET = DeviceBudget(Device.cuda(0), 64 << 20)

SCENE_USDA = f"""#usda 1.0
# PRESENTATION:USDA:BEGIN
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
    startTimeCode = 0
    endTimeCode = 1
    timeCodesPerSecond = 1
)
# PRESENTATION:USDA:END
def Material "MovingLook"
{{
    token outputs:surface.connect = </MovingLook/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.08, 0.4, 0.95)
        float inputs:metallic = 0.18
        float inputs:roughness = 0.24
        token outputs:surface
    }}
}}
def Material "PinnedLook"
{{
    token outputs:surface.connect = </PinnedLook/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.42, 0.01)
        color3f inputs:emissiveColor = (0.3, 0.05, 0)
        token outputs:surface
    }}
}}
def Xform "World"
{{
    # PRESENTATION:USDA:BEGIN
    def Cube "Moving" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        double3 xformOp:translate.timeSamples = {{
            0: (1, 0, 0),
            1: (8, 0, 0),
        }}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </MovingLook>
    }}
    # PRESENTATION:USDA:END
    def Cube "PinnedRay" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        matrix4d xformOp:transform = ((2.5, 0, 0, 0), (0, 0.028, 0, 0), (0, 0, 0.028, 0), (-1.5, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </PinnedLook>
    }}
    def Sphere "PinnedContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double radius = 0.2
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PinnedLook>
    }}
}}
def Xform "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.92, 0, -0.39, 0), (-0.1, 0.97, -0.23, 0), (0.38, 0.25, 0.89, 0), (13, 6, 20, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}
    def RenderProduct "CameraProduct"
    {{
        int2 resolution = ({WIDTH}, {HEIGHT})
        rel camera = </Render/Camera>
        rel orderedVars = [<LdrColor>]
        def RenderVar "LdrColor"
        {{
            string sourceName = "LdrColor"
        }}
    }}
    def SphereLight "Key"
    {{
        float inputs:intensity = 28000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-4, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.12, 0.2, 0.36)
        float inputs:intensity = 260
    }}
}}
"""


def _copy_identity_columns(pinned: RaycastBatchResult, device: str) -> dict[str, wp.array]:
    """Copies the identity-only result columns with Warp, on the plan's device.

    `wp.from_dlpack` borrows a column; on CUDA it also makes Warp's stream wait on the
    launch's `result_ready` event. The copies outlive the operation, so it can be retired.
    """
    ranges = pinned.hit_ranges
    columns = {
        "query_status": ranges.query_status,
        "stored_hit_count": ranges.stored_hit_count,
        "hit_offsets": ranges.hit_offsets,
        "hit_counts": ranges.hit_counts,
        "truncated": ranges.truncated,
        "object_ids": pinned.object_ids,
    }
    with wp.ScopedDevice(device):
        return {name: wp.clone(wp.from_dlpack(column)) for name, column in columns.items()}


def _read_pinned_batch(
    pinned: RaycastBatchResult, columns: dict[str, wp.array], cuda: bool
) -> tuple[int, int]:
    if (pinned.common.result_ready != 0) != cuda:
        raise AssertionError("only a CUDA result publishes a ready event")
    if pinned.common.truth != HitTruth.BOUNDS_CANDIDATE:
        raise AssertionError("pinned batch result truth changed")
    if (
        pinned.geometry is not None
        or pinned.exact is not None
        or pinned.candidate is None
        or pinned.candidate.bounds_positions is not None
        or pinned.candidate.bounds_normals is not None
        or pinned.candidate.distances is not None
    ):
        raise AssertionError("zero result_fields must leave optional tensors absent")
    # Copy to the host only the values this lesson records.
    status = columns["query_status"].list()
    stored = columns["stored_hit_count"].list()[0]
    counts = columns["hit_counts"].list()
    offsets = columns["hit_offsets"].list()
    truncated = columns["truncated"].list()
    if (
        status != [int(BatchRowStatus.SUCCESS)]
        or counts != [stored]
        or offsets != [0]
        or truncated != [0]
    ):
        raise AssertionError("pinned Q-to-H range is unexpected")
    object_ids = columns["object_ids"].list()
    if stored != 1:
        raise AssertionError("pinned query did not retain exactly one hit")
    return stored, object_ids[0]


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    cleanup_errors: list[str] = []
    pixels: Any | None = None
    accepted: RaycastBatchReservation | None = None
    pinned: RaycastBatchResult | None = None
    pinned_count = 0
    next_publication: Any | None = None
    pinned_path = ""
    current_result: Any | None = None
    native: dict[str, Any] = {}
    cuda = False
    try:
        runtime.open_and_compute()
        public_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, (TARGET_PATH,))]
        )
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance(InstanceConfig(device_budgets=(CUDA_BUDGET,))) as instance,
        ):
            population = instance.ovstage(ovstage, runtime.stage).create_population(
                public_filter, PopulationConfig(retained_ordinal_capacity=2)
            )
            population.enqueue_set(2).finish()

            # CUDA first: ask before creating the plan. The CPU runs it only when Warp has no
            # CUDA device or this library cannot plan raycasts there; a CUDA plan never falls back.
            cuda = wp.get_cuda_device_count() > 0 and instance.supports(
                Capability.RAYCAST_PLAN, Device.cuda(0)
            )
            device, warp_device = (Device.cuda(0), "cuda:0") if cuda else (Device.cpu(), "cpu")
            stream = wp.get_stream(warp_device) if cuda else None
            plan = instance.create_raycast_plan(
                RaycastPlanConfig(
                    target=population.target.reference,
                    hit_mode=HitMode.CLOSEST,
                    precision=QueryPrecision.BOUNDS_CANDIDATE,
                    maximum_query_count=1,
                    maximum_hits_per_query=1,
                    hit_capacity_per_slot=1,
                    result_slot_count=1,
                    result_fields=BatchResultField(0),
                    contact_tolerance=0.0,
                    device=device,
                )
            )
            # PRESENTATION:API:BEGIN
            # The plan reads these Warp arrays until the query completes, so they stay alive.
            with wp.ScopedDevice(warp_device):
                origins = wp.array([[-4.0, 0.0, 0.0]], dtype=wp.float64)
                directions = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32)
                min_distances = wp.zeros(1, dtype=wp.float64)
                max_distances = wp.full(1, 6.0, dtype=wp.float64)
            query_operation, accepted = plan.enqueue(
                origins=origins,
                unit_directions=directions,
                min_distances=min_distances,
                max_distances=max_distances,
                cuda_stream=stream,
            )
            if accepted.common.source_ordinal != 2 or (accepted.common.result_ready != 0) != cuda:
                raise AssertionError("async query did not pin the prior version")
            # PRESENTATION:API:END

            # PRESENTATION:API:BEGIN
            ovstage.population.update_from_usd_time(runtime.stage, 3, 1.0)
            runtime.stage.advance_write_floor(ordinal=3).wait()
            with population.enqueue_update(2, 3) as update_operation:
                update_operation.wait()
                next_publication = update_operation.fetch_population_update()
            # PRESENTATION:API:END

            # PRESENTATION:API:BEGIN
            with query_operation:
                query_operation.wait()
                pinned = query_operation.fetch_raycast_batch()
                columns = _copy_identity_columns(pinned, warp_device)
                if cuda:
                    # Retire the operation after the copies; the library takes the event.
                    consumer_done = stream.record_event()
                    query_operation.close_after_cuda(consumer_done.cuda_event)
                    consumer_done.owner = False
            pinned_count, pinned_object = _read_pinned_batch(pinned, columns, cuda)
            if pinned.common.source_ordinal != 2 or pinned_count != 1:
                raise AssertionError("pinned query lost its immutable prior version")
            # PRESENTATION:API:END
            pinned_path = runtime.path_to_string(pinned_object)

            current_result = population.target.raycast_closest(
                RaycastQuery(
                    origin=(-4.0, 0.0, 0.0),
                    unit_direction=(1.0, 0.0, 0.0),
                    max_distance=6.0,
                    precision=QueryPrecision.BOUNDS_CANDIDATE,
                )
            )
            if (
                pinned_path != TARGET_PATH
                or next_publication.published_ordinal != 3
                or current_result.source_ordinal != 3
                or current_result.hit is not None
            ):
                raise AssertionError("prior-version pin or next-version publication is incorrect")
            pixels = runtime.render(options.frames, 3)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if (
        pixels is None
        or accepted is None
        or pinned is None
        or next_publication is None
        or current_result is None
    ):
        raise AssertionError("example produced incomplete evidence")
    image = frame_summary(pixels)
    payload = {
        "schema_version": 2,
        "example": EXAMPLE,
        "language": "python",
        "execution": {
            "headless": bool(options.headless),
            "frames": options.frames,
            "resolution": [WIDTH, HEIGHT],
        },
        "inputs": {
            "stage": "inline-usda",
            "target_paths": [TARGET_PATH],
            "device": "cuda:0" if cuda else "cpu:0",
            "ordering": [
                "admit asynchronous query",
                "submit and wait for next publication",
                "fetch pinned query",
                "query current publication",
            ],
        },
        "publication": {
            "next_publication": {
                "source_ordinal": next_publication.published_ordinal,
            }
        },
        "result": {
            "pinned_query": {
                "admission_source_ordinal": int(accepted.common.source_ordinal),
                "result_source_ordinal": int(pinned.common.source_ordinal),
                "stored_hit_count": pinned_count,
                "prim_path": pinned_path,
            },
            "current_query": {
                "source_ordinal": current_result.source_ordinal,
                "has_hit": current_result.hit is not None,
            },
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: immutable version overlap", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
