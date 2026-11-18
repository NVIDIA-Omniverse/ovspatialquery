# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query the current AABB-overlap pairs between two Ovstage populations."""

from __future__ import annotations

import sys
from typing import Any

import ovstage

from ovspatialquery import Library

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

EXAMPLE = "05_two_set_pair_snapshot"
SET_A = ("/World/Sets/A/A0", "/World/Sets/A/A1")
SET_B = ("/World/Sets/B/B0", "/World/Sets/B/B1")
EXPECTED = ((SET_A[0], SET_B[0]),)

SCENE_USDA = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Scope "Looks"
{{
    def Material "Pair"
    {{
        token outputs:surface.connect = </Looks/Pair/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.9, 0.28)
            color3f inputs:emissiveColor = (0.02, 0.18, 0.04)
            float inputs:roughness = 0.24
            token outputs:surface
        }}
    }}
    def Material "First"
    {{
        token outputs:surface.connect = </Looks/First/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0.38, 0.04)
            float inputs:roughness = 0.3
            token outputs:surface
        }}
    }}
    def Material "Second"
    {{
        token outputs:surface.connect = </Looks/Second/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.08, 0.3, 0.95)
            float inputs:roughness = 0.3
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
                rel material:binding = </Looks/Pair>
            }}
            def Cube "A1" (prepend apiSchemas = ["MaterialBindingAPI"])
            {{
                double size = 1.6
                float3[] extent = [(-0.8, -0.8, -0.8), (0.8, 0.8, 0.8)]
                double3 xformOp:translate = (3.2, 0, 1.3)
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
                double3 xformOp:translate = (-0.35, 0, 0)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Pair>
            }}
            def Cube "B1" (prepend apiSchemas = ["MaterialBindingAPI"])
            {{
                double size = 1.5
                float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
                double3 xformOp:translate = (5.4, 0, -1.7)
                uniform token[] xformOpOrder = ["xformOp:translate"]
                rel material:binding = </Looks/Second>
            }}
        }}
    }}
    # PRESENTATION:USDA:END
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((5.5, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 3.5, 0), (1.5, -1.25, 0, 1))
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


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    cleanup_errors: list[str] = []
    populations: list[Any] = []
    publications: list[Any] = []
    pixels: Any | None = None
    snapshot: Any | None = None
    pair_paths: list[tuple[str, str]] = []
    native: dict[str, Any] = {}
    runtime = Runtime(f"ovspatialquery.{EXAMPLE}.python", SCENE_USDA)
    try:
        runtime.open_and_compute()
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            adapter = instance.ovstage(ovstage, runtime.stage)
            for endpoint_paths in (SET_A, SET_B):
                population = adapter.create_population(_filter(endpoint_paths))
                populations.append(population)
                with population.enqueue_set(2) as operation:
                    operation.wait()
                    publications.append(operation.fetch_population_update())

            # PRESENTATION:API:BEGIN
            snapshot = instance.query_pairs(
                populations[0].target,
                populations[1].target,
                capacity=8,
                contact_tolerance=0.0,
            )
            # PRESENTATION:API:END
            pair_paths = sorted(
                (
                    runtime.path_to_string(first),
                    runtime.path_to_string(second),
                )
                for first, second in snapshot.rows
            )
            if tuple(pair_paths) != EXPECTED or snapshot.truncated:
                raise AssertionError(
                    f"unexpected pair snapshot: {pair_paths}, truncated={snapshot.truncated}"
                )
            pixels = runtime.render(options.frames, ordinal=2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if pixels is None or snapshot is None or len(publications) != 2:
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
            "capacity": 8,
            "contact_tolerance": 0.0,
            "precision": "aabb_broadphase",
        },
        "publication": {
            "sets": [
                {
                    "name": name,
                    "source_ordinal": receipt.published_ordinal,
                    "object_count": receipt.object_count,
                    "excluded": receipt.excluded_object_count,
                }
                for name, receipt in zip(("A", "B"), publications, strict=True)
            ]
        },
        "result": {
            "first_source_ordinal": snapshot.first_source_ordinal,
            "second_source_ordinal": snapshot.second_source_ordinal,
            "pair_count": len(pair_paths),
            "truncated": snapshot.truncated,
            "pairs": [{"first_path": first, "second_path": second} for first, second in pair_paths],
        },
        "native": native,
        "evidence": image,
        "cleanup": {"clean": not cleanup_errors, "errors": cleanup_errors},
        "errors": [],
    }
    write_semantic_json(paths.json, payload)
    validate_evidence(paths.json)
    if not options.headless:
        present_interactive("ovspatialquery: two-set pair snapshot", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
