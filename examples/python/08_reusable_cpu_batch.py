# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse one batch plan for two visible raycast launches, on the GPU when there is one."""

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
    QueryPrecision,
    RaycastPlanConfig,
)
from ovspatialquery.types import RaycastBatchResult

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

EXAMPLE = "08_reusable_cpu_batch"
TARGET_PATHS = ("/World/Targets/Hit", "/World/Targets/OffRay")

SCENE_USDA = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)
def Material "Target"
{{
    token outputs:surface.connect = </Target/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.04, 0.7, 0.95)
        float inputs:roughness = 0.25
        token outputs:surface
    }}
}}
def Material "Ray"
{{
    token outputs:surface.connect = </Ray/P.outputs:surface>
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
    def Xform "Targets"
    {{
        def Cube "Hit" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Target>
        }}
        def Cube "OffRay" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (3, 0, -3)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Target>
        }}
    }}
    # PRESENTATION:USDA:END
    def Cube "RayA" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.025, 0, 0), (0, 0, 0.025, 0), (-2, 0, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Ray>
    }}
    def Cube "RayB" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        matrix4d xformOp:transform = ((3, 0, 0, 0), (0, 0.025, 0, 0), (0, 0, 0.025, 0), (-1, 0, -3, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Ray>
    }}
}}
def Xform "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.88, 0, -0.47, 0), (-0.14, 0.95, -0.27, 0), (0.45, 0.31, 0.84, 0), (8, 5.5, 16, 1))
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

LAUNCHES = (
    ([[-4.0, 0.0, 0.0], [-4.0, 0.0, 4.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    ([[-4.0, 0.0, -3.0], [-4.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
)
# CUDA work needs a byte budget for the device; the CPU needs none.
CUDA_BUDGET = DeviceBudget(Device.cuda(0), 64 << 20)


def _copy_identity_columns(fetched: RaycastBatchResult, device: str) -> dict[str, wp.array]:
    """Copies the identity-only result columns with Warp, on the plan's device.

    `wp.from_dlpack` borrows a column; on CUDA it also makes Warp's stream wait on the
    launch's `result_ready` event. The copies outlive the operation, so it can be retired.
    """
    ranges = fetched.hit_ranges
    columns = {
        "query_status": ranges.query_status,
        "stored_hit_count": ranges.stored_hit_count,
        "hit_offsets": ranges.hit_offsets,
        "hit_counts": ranges.hit_counts,
        "truncated": ranges.truncated,
        "object_ids": fetched.object_ids,
    }
    with wp.ScopedDevice(device):
        return {name: wp.clone(wp.from_dlpack(column)) for name, column in columns.items()}


def _read_identity_only_batch(
    fetched: RaycastBatchResult,
    columns: dict[str, wp.array],
    query_count: int,
    cuda: bool,
) -> tuple[list[int], list[int], list[int]]:
    if (fetched.common.result_ready != 0) != cuda:
        raise AssertionError("only a CUDA result publishes a ready event")
    if fetched.common.truth != HitTruth.BOUNDS_CANDIDATE:
        raise AssertionError("batch result truth changed")
    if (
        fetched.geometry is not None
        or fetched.exact is not None
        or fetched.candidate is None
        or fetched.candidate.bounds_positions is not None
        or fetched.candidate.bounds_normals is not None
        or fetched.candidate.distances is not None
    ):
        raise AssertionError("zero result_fields must leave optional tensors absent")
    # Copy to the host only the values this lesson records.
    status = columns["query_status"].list()
    stored = columns["stored_hit_count"].list()[0]
    offsets = columns["hit_offsets"].list()
    counts = columns["hit_counts"].list()
    truncated = columns["truncated"].list()
    if len(counts) != query_count:
        raise AssertionError(f"unexpected query count: {len(counts)}")
    if status != [int(BatchRowStatus.SUCCESS)] * query_count or any(truncated):
        raise AssertionError("batch row status or truncation is unexpected")
    cursor = 0
    for offset, count in sorted((offset, count) for offset, count in zip(offsets, counts) if count):
        if offset != cursor:
            raise AssertionError("Q-to-H ranges are not compact")
        cursor += count
    if cursor != stored:
        raise AssertionError("Q-to-H ranges do not cover stored hits")
    object_ids = columns["object_ids"].list()
    if len(object_ids) != stored:
        raise AssertionError(f"unexpected stored hit count: {len(object_ids)}")
    return counts, offsets, object_ids


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    cleanup_errors: list[str] = []
    pixels: Any | None = None
    publication: Any | None = None
    launches: list[dict[str, Any]] = []
    native: dict[str, Any] = {}
    cuda = False
    try:
        runtime.open_and_compute()
        public_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, TARGET_PATHS)]
        )
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance(InstanceConfig(device_budgets=(CUDA_BUDGET,))) as instance,
        ):
            population = instance.ovstage(ovstage, runtime.stage).create_population(public_filter)
            with population.enqueue_set(2) as publication_operation:
                publication_operation.wait()
                publication = publication_operation.fetch_population_update()

            # PRESENTATION:API:BEGIN
            # CUDA first: ask before creating the plan. The CPU runs it only when Warp has no
            # CUDA device or this library cannot plan raycasts there; a CUDA plan never falls back.
            cuda = wp.get_cuda_device_count() > 0 and instance.supports(
                Capability.RAYCAST_PLAN, Device.cuda(0)
            )
            device, warp_device = (Device.cuda(0), "cuda:0") if cuda else (Device.cpu(), "cpu")
            descriptor = RaycastPlanConfig(
                target=population.target.reference,
                hit_mode=HitMode.CLOSEST,
                precision=QueryPrecision.BOUNDS_CANDIDATE,
                maximum_query_count=2,
                maximum_hits_per_query=1,
                hit_capacity_per_slot=2,
                # A CUDA slot returns to the plan only after its consumers, so the second
                # launch takes the other slot instead of waiting for the first.
                result_slot_count=2,
                result_fields=BatchResultField(0),
                contact_tolerance=0.0,
                device=device,
            )
            plan = instance.create_raycast_plan(descriptor)
            # Warp's stream orders the input writes, the launch and the column copies.
            stream = wp.get_stream(warp_device) if cuda else None
            # PRESENTATION:API:END
            for launch_index, (origin_rows, direction_rows) in enumerate(LAUNCHES):
                # PRESENTATION:API:BEGIN
                with wp.ScopedDevice(warp_device):
                    origins = wp.array(origin_rows, dtype=wp.float64)
                    directions = wp.array(direction_rows, dtype=wp.float32)
                    min_distances = wp.zeros(2, dtype=wp.float64)
                    max_distances = wp.full(2, 20.0, dtype=wp.float64)
                # The library copies each column, so the arrays can change once enqueue returns.
                query_operation, accepted = plan.enqueue(
                    origins=origins,
                    unit_directions=directions,
                    min_distances=min_distances,
                    max_distances=max_distances,
                    copy=True,
                    cuda_stream=stream,
                )
                with query_operation:
                    if (
                        accepted.common.source_ordinal != 2
                        or (accepted.common.result_ready != 0) != cuda
                    ):
                        raise AssertionError("batch admission did not pin ordinal 2")
                    query_operation.wait()
                    fetched = query_operation.fetch_raycast_batch()
                    columns = _copy_identity_columns(fetched, warp_device)
                    if cuda:
                        # Retire the operation after the copies; the library takes the event.
                        consumer_done = stream.record_event()
                        query_operation.close_after_cuda(consumer_done.cuda_event)
                        consumer_done.owner = False
                # PRESENTATION:API:END
                counts, offsets, object_ids = _read_identity_only_batch(fetched, columns, 2, cuda)
                stored = len(object_ids)
                hit_paths = [runtime.path_to_string(object_id) for object_id in object_ids]
                launches.append(
                    {
                        "launch_index": launch_index,
                        "source_ordinal": int(fetched.common.source_ordinal),
                        "query_hit_counts": counts,
                        "hit_offsets": offsets,
                        "stored_hit_count": stored,
                        "object_ids": object_ids,
                        "prim_paths": hit_paths,
                    }
                )
            if (
                launches[0]["query_hit_counts"] != [1, 0]
                or launches[0]["prim_paths"] != [TARGET_PATHS[0]]
                or launches[1]["query_hit_counts"] != [1, 1]
                or launches[1]["prim_paths"] != [TARGET_PATHS[1], TARGET_PATHS[0]]
            ):
                raise AssertionError(f"unexpected reusable batch results: {launches}")
            pixels = runtime.render(options.frames, 2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or publication is None or len(launches) != 2:
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
            "target_paths": list(TARGET_PATHS),
            "batch_plan": {
                "family": "raycast",
                "hit_mode": "closest",
                "precision": "aabb_candidate",
                "device": "cuda:0" if cuda else "cpu:0",
                "maximum_query_count": 2,
                "result_slot_count": 2,
            },
        },
        "publication": {
            "source_ordinal": publication.published_ordinal,
            "object_count": publication.object_count,
        },
        "result": {"plan_create_count": 1, "launches": launches},
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: reusable batch plan", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
