# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advance an Ovstage-backed pair reporter from ENTERED to EXITED."""

from __future__ import annotations

import sys
from typing import Any

import ovstage
import warp as wp

from ovspatialquery import (
    Capability,
    Device,
    DeviceBudget,
    InstanceConfig,
    Library,
    PairReporterConfig,
    PairReporterOutcome,
)

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

EXAMPLE = "10_pair_reporter_deltas"
SET_A = ("/World/Sets/A/A0",)
SET_B = ("/World/Sets/B/B0",)
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
def Scope "Looks"
{{
    def Material "First"
    {{
        token outputs:surface.connect = </Looks/First/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.9, 0.28)
            float inputs:roughness = 0.24
            token outputs:surface
        }}
    }}
    def Material "Second"
    {{
        token outputs:surface.connect = </Looks/Second/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.9, 0.15, 0.82)
            float inputs:roughness = 0.24
            token outputs:surface
        }}
    }}
    def Material "Baseline"
    {{
        token outputs:surface.connect = </Looks/Baseline/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.25, 0.45, 1)
            float inputs:opacity = 0.24
            token outputs:surface
        }}
    }}
    def Material "Future"
    {{
        token outputs:surface.connect = </Looks/Future/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.9, 0.15, 0.82)
            float inputs:opacity = 0.24
            token outputs:surface
        }}
    }}
    def Material "Ground"
    {{
        token outputs:surface.connect = </Looks/Ground/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.04, 0.055, 0.09)
            float inputs:roughness = 0.65
            token outputs:surface
        }}
    }}
}}
def Xform "World"
{{
    # PRESENTATION:USDA:BEGIN
    def Xform "Sets"
    {{
        def Xform "A"
        {{
            def Cube "A0" (prepend apiSchemas = ["MaterialBindingAPI"])
            {{
                double size = 2
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                double3 xformOp:translate = (-1.2, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/First>
            }}
        }}
        def Xform "B"
        {{
            def Cube "B0" (prepend apiSchemas = ["MaterialBindingAPI"])
            {{
                double size = 2
                float3[] extent = [(-1, -1, -1), (1, 1, 1)]
                double3 xformOp:translate.timeSamples = {{
                    0: (-0.35, 0, 0),
                    1: (4.1, 0, 0)
                }}
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Second>
            }}
        }}
    }}
    # PRESENTATION:USDA:END
    def Cube "BaselineAtT0" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        double size = 2.08
        float3[] extent = [(-1.04, -1.04, -1.04), (1.04, 1.04, 1.04)]
        double3 xformOp:translate = (-0.35, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Looks/Baseline>
    }}
    def Cube "FutureAtT1" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        double size = 2.08
        float3[] extent = [(-1.04, -1.04, -1.04), (1.04, 1.04, 1.04)]
        double3 xformOp:translate = (4.1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </Looks/Future>
    }}
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((5.5, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 3.5, 0), (1.2, -1.25, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Looks/Ground>
    }}
}}
def Xform "Render"
{{
    def Camera "Camera"
    {{
        uniform token projection = "perspective"
        float focalLength = 21
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.94, 0, -0.342, 0), (-0.097, 0.959, -0.266, 0), (0.328, 0.284, 0.901, 0), (8.5, 5.2, 16.5, 1))
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
        color3f inputs:color = (1, 0.82, 0.66)
        float inputs:intensity = 30000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def RectLight "Fill"
    {{
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 900
        float inputs:width = 8
        float inputs:height = 8
        double3 xformOp:translate = (6, 4, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.13, 0.19, 0.34)
        float inputs:intensity = 210
    }}
}}
"""


def _filter(paths: tuple[str, ...]) -> Any:
    return ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])


def _pair_values(rows: wp.array) -> list[tuple[int, int]]:
    if rows.dtype != wp.uint64 or rows.ndim != 2 or rows.shape[1] != 2:
        raise AssertionError("pair reporter delta tensor descriptor is unexpected")
    # Copy to the host only the pairs this lesson records.
    values = rows.list()
    return list(zip(values[0::2], values[1::2], strict=True))


# PRESENTATION:API:BEGIN
def _advance(reporter: Any, warp_device: str) -> dict[str, Any]:
    cuda = warp_device != "cpu"
    operation, _reservation = reporter.advance()
    with operation:
        operation.wait()
        result = operation.fetch_pair_reporter()
        if (result.result_ready != 0) != cuda:
            raise AssertionError("only a CUDA reporter publishes a ready event")
        # wp.from_dlpack borrows a column; on CUDA it also makes Warp's stream wait on
        # result_ready. The copies outlive the operation, so it can be retired.
        with wp.ScopedDevice(warp_device):
            entered = wp.clone(wp.from_dlpack(result.entered_pairs))
            exited = wp.clone(wp.from_dlpack(result.exited_pairs))
            if cuda:
                # Retire the operation after the copies; the library takes the event.
                consumer_done = wp.get_stream().record_event()
                operation.close_after_cuda(consumer_done.cuda_event)
                consumer_done.owner = False
    # PRESENTATION:API:END
    payload = {
        "first_source_ordinal": result.first_source_ordinal,
        "second_source_ordinal": result.second_source_ordinal,
        "outcome": result.outcome,
        "entered": _pair_values(entered),
        "exited": _pair_values(exited),
        "active_pair_count": result.accepted_active_pair_count,
    }
    return payload


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    cleanup_errors: list[str] = []
    populations: list[Any] = []
    publications: list[Any] = []
    advances: list[dict[str, Any]] = []
    pixels: Any | None = None
    native: dict[str, Any] = {}
    cuda = False
    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    try:
        runtime.open_and_compute()
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance(InstanceConfig(device_budgets=(CUDA_BUDGET,))) as instance,
        ):
            adapter = instance.ovstage(ovstage, runtime.stage)
            for endpoint_paths in (SET_A, SET_B):
                population = adapter.create_population(_filter(endpoint_paths))
                populations.append(population)
                with population.enqueue_set(2) as operation:
                    operation.wait()
                    publications.append(operation.fetch_population_update())

            # PRESENTATION:API:BEGIN
            # CUDA first: ask before creating the reporter. The CPU runs it only when Warp has
            # no CUDA device or this library has no CUDA reporter; a CUDA one never falls back.
            cuda = wp.get_cuda_device_count() > 0 and instance.supports(
                Capability.PAIR_REPORTER, Device.cuda(0)
            )
            device, warp_device = (Device.cuda(0), "cuda:0") if cuda else (Device.cpu(), "cpu")
            reporter = instance.create_pair_reporter(
                PairReporterConfig(
                    first=populations[0].target.reference,
                    second=populations[1].target.reference,
                    active_pair_capacity=4,
                    delta_pair_capacity=4,
                    # A CUDA slot returns to the reporter only after its consumers, so the
                    # second advance takes the other slot instead of waiting for the first.
                    result_slot_count=2,
                    result_device=device,
                )
            )
            advances.append(_advance(reporter, warp_device))
            # PRESENTATION:API:END

            # PRESENTATION:API:BEGIN
            ovstage.population.update_from_usd_time(runtime.stage, 3, 1.0)
            runtime.stage.advance_write_floor(ordinal=3).wait()
            with populations[1].enqueue_update(2, 3) as operation:
                # PRESENTATION:API:END
                operation.wait()
                publications.append(operation.fetch_population_update())

            # PRESENTATION:API:BEGIN
            advances.append(_advance(reporter, warp_device))
            # PRESENTATION:API:END

            for index, advance in enumerate(advances):
                kind = "entered" if index == 0 else "exited"
                pairs = advance[kind]
                if len(pairs) != 1:
                    raise AssertionError(f"unexpected {kind} pair count: {advance}")
                first_path = runtime.path_to_string(pairs[0][0])
                second_path = runtime.path_to_string(pairs[0][1])
                advance["pairs"] = [
                    {
                        "kind": kind,
                        "first_path": first_path,
                        "second_path": second_path,
                    }
                ]
                if (
                    advance["outcome"] is not PairReporterOutcome.ADVANCED
                    or (first_path, second_path) != (SET_A[0], SET_B[0])
                    or advance["active_pair_count"] != (1 if index == 0 else 0)
                    or advance["entered" if index == 1 else "exited"]
                ):
                    raise AssertionError(f"unexpected pair-reporter advance: {advance}")

            # Presentation shows the t0 baseline and the authored t1 destination
            # together; the two reporter advances above remain semantic authority.
            pixels = runtime.render(options.frames, ordinal=2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or len(advances) != 2 or len(publications) != 3:
        raise AssertionError("example did not produce complete evidence")
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
            "first_paths": list(SET_A),
            "second_paths": list(SET_B),
            "times": [0.0, 1.0],
            "active_pair_capacity": 4,
            "delta_pair_capacity": 4,
            "result_slot_count": 2,
            "precision": "aabb_broadphase",
            "device": "cuda:0" if cuda else "cpu:0",
        },
        "publication": {
            "initial": [
                {
                    "name": name,
                    "source_ordinal": receipt.published_ordinal,
                    "object_count": receipt.object_count,
                    "excluded": receipt.excluded_object_count,
                }
                for name, receipt in zip(("A", "B"), publications[:2], strict=True)
            ],
            "updated_second": {
                "name": "B",
                "source_ordinal": publications[2].published_ordinal,
                "object_count": publications[2].object_count,
                "excluded": publications[2].excluded_object_count,
            },
        },
        "result": {
            "advances": [
                {
                    "first_source_ordinal": advance["first_source_ordinal"],
                    "second_source_ordinal": advance["second_source_ordinal"],
                    "outcome": advance["outcome"].name.lower(),
                    "active_pair_count": advance["active_pair_count"],
                    "pairs": advance["pairs"],
                    "entered_count": len(advance["entered"]),
                    "exited_count": len(advance["exited"]),
                }
                for advance in advances
            ]
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: pair-reporter deltas", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
