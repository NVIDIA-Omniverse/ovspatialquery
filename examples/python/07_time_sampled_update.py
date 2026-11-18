# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish one Ovstage time-sampled transform update and query both ordinals."""

from __future__ import annotations

import sys
from typing import Any

import ovstage

from ovspatialquery import Library, PopulationConfig, QueryPrecision, RaycastQuery

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

EXAMPLE = "07_time_sampled_update"
TARGET_PATH = "/World/Moving"
TIME_DOMAIN = 0xE09
RAY_ORIGIN = (-4.0, 0.0, 0.0)
RAY_DIRECTION = (1.0, 0.0, 0.0)

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
        color3f inputs:diffuseColor = (0.05, 0.72, 0.95)
        float inputs:metallic = 0.18
        float inputs:roughness = 0.24
        token outputs:surface
    }}
}}
def Material "PathLook"
{{
    token outputs:surface.connect = </PathLook/P.outputs:surface>
    def Shader "P"
    {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (1, 0.45, 0.01)
        color3f inputs:emissiveColor = (0.25, 0.04, 0)
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
            1: (4, 0, 0),
        }}
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </MovingLook>
    }}
    # PRESENTATION:USDA:END
    def Cube "MotionPath" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double size = 2
        matrix4d xformOp:transform = ((2, 0, 0, 0), (0, 0.028, 0, 0), (0, 0, 0.028, 0), (2.5, -1.25, 0, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </PathLook>
    }}
    def Sphere "BeforeContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double radius = 0.15
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PathLook>
    }}
    def Sphere "AfterContact" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        double radius = 0.22
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        rel material:binding = </PathLook>
    }}
}}
def Xform "Render"
{{
    def Camera "Camera"
    {{
        float focalLength = 18
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.91, 0, -0.41, 0), (-0.11, 0.96, -0.25, 0), (0.39, 0.27, 0.88, 0), (10, 5.8, 17, 1))
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


def _hit(value: Any, path: str) -> dict[str, Any]:
    if value.hit is None:
        raise AssertionError("time-sampled ray must hit")
    return {
        "source_ordinal": value.source_ordinal,
        "has_hit": True,
        "object_id": value.hit.object_id,
        "prim_path": path,
        "distance": value.hit.distance,
        "position": list(value.hit.position),
        "normal": list(value.hit.normal),
        "truth": value.hit.truth.name.lower(),
    }


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    cleanup_errors: list[str] = []
    pixels: Any | None = None
    publications: list[Any] = []
    results: list[Any] = []
    result_paths: list[str] = []
    native: dict[str, Any] = {}
    try:
        runtime.open_and_compute()
        public_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, (TARGET_PATH,))]
        )
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            population = instance.ovstage(ovstage, runtime.stage).create_population(
                public_filter, PopulationConfig(retained_ordinal_capacity=2)
            )
            with population.enqueue_set(2) as operation:
                operation.wait()
                publications.append(operation.fetch_population_update())

            query = RaycastQuery(
                origin=RAY_ORIGIN,
                unit_direction=RAY_DIRECTION,
                max_distance=20.0,
                precision=QueryPrecision.BOUNDS_CANDIDATE,
            )
            before = population.target.raycast_closest(query)
            if before.hit is None:
                raise AssertionError("time zero ray must hit")
            before_path = runtime.path_to_string(before.hit.object_id)
            results.append(before)
            result_paths.append(before_path)

            # PRESENTATION:API:BEGIN
            ovstage.population.update_from_usd_time(runtime.stage, 3, 1.0)
            runtime.stage.advance_write_floor(ordinal=3).wait()
            with population.enqueue_update(2, 3) as operation:
                # PRESENTATION:API:END
                operation.wait()
                publications.append(operation.fetch_population_update())

            after = population.target.raycast_closest(query)
            if after.hit is None:
                raise AssertionError("time one ray must hit")
            after_path = runtime.path_to_string(after.hit.object_id)
            results.append(after)
            result_paths.append(after_path)
            if (
                result_paths != [TARGET_PATH, TARGET_PATH]
                or before.source_ordinal != 2
                or after.source_ordinal != 3
                or abs(before.hit.distance - 4.0) > 1.0e-9
                or abs(after.hit.distance - 7.0) > 1.0e-9
            ):
                raise AssertionError(f"unexpected sampled hits: {before}, {after}")
            pixels = runtime.render(options.frames, 3)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or len(publications) != 2 or len(results) != 2:
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
            "samples": [
                {"source_ordinal": 2, "sample_time": 0.0},
                {"source_ordinal": 3, "sample_time": 1.0},
            ],
            "ray": {
                "origin": list(RAY_ORIGIN),
                "unit_direction": list(RAY_DIRECTION),
                "max_distance": 20.0,
                "precision": "aabb_candidate",
                "mode": "closest",
            },
        },
        "publication": {
            "before": {
                "source_ordinal": publications[0].published_ordinal,
                "object_count": publications[0].object_count,
            },
            "after": {
                "source_ordinal": publications[1].published_ordinal,
                "object_count": publications[1].object_count,
            },
        },
        "result": {
            "before": _hit(results[0], result_paths[0]),
            "after": _hit(results[1], result_paths[1]),
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: time-sampled update", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
