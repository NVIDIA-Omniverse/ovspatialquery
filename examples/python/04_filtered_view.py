# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restrict a published Ovstage population with one immutable filtered view."""

from __future__ import annotations

import sys
from typing import Any

import ovstage

from ovspatialquery import Library, QueryPrecision, RaycastQuery

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

EXAMPLE = "04_filtered_view"
POPULATION_PATHS = ("/World/Targets/Excluded", "/World/Targets/Included")
VIEW_PATH = POPULATION_PATHS[1]
RAY_ORIGIN = (-5.0, 0.0, 0.0)
RAY_DIRECTION = (1.0, 0.0, 0.0)
RAY_DISTANCE = 12.0

SCENE_USDA = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 1
)

def Scope "Looks"
{{
    def Material "Excluded"
    {{
        token outputs:surface.connect = </Looks/Excluded/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.42, 0.12, 0.2)
            float inputs:roughness = 0.5
            token outputs:surface
        }}
    }}
    def Material "Included"
    {{
        token outputs:surface.connect = </Looks/Included/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.05, 0.9, 0.28)
            float inputs:roughness = 0.24
            token outputs:surface
        }}
    }}
    def Material "Query"
    {{
        token outputs:surface.connect = </Looks/Query/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (1, 0.58, 0.02)
            color3f inputs:emissiveColor = (0.35, 0.1, 0)
            float inputs:roughness = 0.35
            token outputs:surface
        }}
    }}
    def Material "Ground"
    {{
        token outputs:surface.connect = </Looks/Ground/Preview.outputs:surface>
        def Shader "Preview"
        {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0.045, 0.06, 0.1)
            float inputs:roughness = 0.6
            token outputs:surface
        }}
    }}
}}

def Xform "World"
{{
    # PRESENTATION:USDA:BEGIN
    def Xform "Targets"
    {{
        def Cube "Excluded" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double size = 1.5
            float3[] extent = [(-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)]
            double3 xformOp:translate = (-1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Excluded>
        }}
        def Cube "Included" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            double3 xformOp:translate = (2, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Included>
        }}
    }}
    # PRESENTATION:USDA:END
    def Xform "QueryVisual"
    {{
        def Cube "RayToExcluded" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            matrix4d xformOp:transform = ((1.625, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (-3.375, 0, 0, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            rel material:binding = </Looks/Query>
        }}
        def Cube "RayToIncluded" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double size = 2
            float3[] extent = [(-1, -1, -1), (1, 1, 1)]
            matrix4d xformOp:transform = ((0.625, 0, 0, 0), (0, 0.035, 0, 0), (0, 0, 0.035, 0), (0.375, 0, 0, 1))
            uniform token[] xformOpOrder = ["xformOp:transform"]
            rel material:binding = </Looks/Query>
        }}
        def Sphere "Origin" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double radius = 0.16
            double3 xformOp:translate = (-5, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Query>
        }}
        def Sphere "SelectedMarker" (prepend apiSchemas = ["MaterialBindingAPI"])
        {{
            double radius = 0.16
            double3 xformOp:translate = (2, 1.25, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            rel material:binding = </Looks/Included>
        }}
    }}
    def Cube "Ground" (prepend apiSchemas = ["MaterialBindingAPI"])
    {{
        double size = 2
        float3[] extent = [(-1, -1, -1), (1, 1, 1)]
        matrix4d xformOp:transform = ((6, 0, 0, 0), (0, 0.08, 0, 0), (0, 0, 4, 0), (0, -1.35, -0.8, 1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        rel material:binding = </Looks/Ground>
    }}
}}

def Xform "Render"
{{
    def Camera "Camera"
    {{
        uniform token projection = "perspective"
        float focalLength = 20
        float horizontalAperture = 27.183644
        float verticalAperture = 15.2908
        double2 clippingRange = (0.01, 1000)
        matrix4d xformOp:transform = ((0.88235294, 0, -0.47058824, 0), (-0.1448566, 0.95144452, -0.27160613, 0), (0.4477386, 0.30782029, 0.83950987, 0), (8, 5.5, 16, 1))
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
        float inputs:intensity = 32000
        float inputs:radius = 1.5
        double3 xformOp:translate = (-5, 7, 8)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def RectLight "Fill"
    {{
        color3f inputs:color = (0.48, 0.68, 1)
        float inputs:intensity = 1100
        float inputs:width = 8
        float inputs:height = 8
        double3 xformOp:translate = (6, 4, 6)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
    def DomeLight "Environment"
    {{
        color3f inputs:color = (0.16, 0.22, 0.38)
        float inputs:intensity = 220
    }}
}}
"""


def run(arguments: list[str] | None = None) -> tuple[dict[str, Any], Any]:
    options = parse_example_arguments(example_argument_parser(__doc__), arguments)
    require_python_source(options)
    paths = EvidencePaths.create(options.evidence, EXAMPLE, "python")

    runtime = Runtime(EXAMPLE, SCENE_USDA)
    publication: Any | None = None
    population_result: Any | None = None
    view_result: Any | None = None
    pixels: Any | None = None
    population_path = ""
    view_path = ""
    cleanup_errors: list[str] = []
    native: dict[str, Any] = {}
    try:
        runtime.open_and_compute()
        population_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, POPULATION_PATHS)]
        )
        # PRESENTATION:API:BEGIN
        view_filter = ovstage.Filter(
            [ovstage.Predicate("usd-path", ovstage.FilterOp.IN, [VIEW_PATH])]
        )
        with (
            Library.open(options.library, require_ovstage=True) as library,
            library.instance() as instance,
        ):
            population = instance.ovstage(ovstage, runtime.stage).create_population(
                population_filter
            )
            view = population.create_view(view_filter)
            # PRESENTATION:API:END
            with population.enqueue_set(2) as operation:
                operation.wait()
                publication = operation.fetch_population_update()

            query = RaycastQuery(
                origin=RAY_ORIGIN,
                unit_direction=RAY_DIRECTION,
                min_distance=0.0,
                max_distance=RAY_DISTANCE,
                precision=QueryPrecision.BOUNDS_CANDIDATE,
            )
            # PRESENTATION:API:BEGIN
            population_result = population.target.raycast_closest(query)
            view_result = view.target.raycast_closest(query)
            # PRESENTATION:API:END
            if population_result.hit is None or view_result.hit is None:
                raise AssertionError("population and filtered view must both hit")
            population_path = runtime.path_to_string(population_result.hit.object_id)
            view_path = runtime.path_to_string(view_result.hit.object_id)
            if (
                population_path != POPULATION_PATHS[0]
                or view_path != VIEW_PATH
                or abs(population_result.hit.distance - 3.25) > 1.0e-12
                or abs(view_result.hit.distance - 6.0) > 1.0e-12
                or population_result.source_ordinal != view_result.source_ordinal
            ):
                raise AssertionError("filtered view did not exclude the nearer population object")
            pixels = runtime.render(options.frames, ordinal=2)
            native = runtime.native_identity(library, options.sdk_root)
    finally:
        cleanup_errors.extend(runtime.close())

    if (
        pixels is None
        or publication is None
        or population_result is None
        or population_result.hit is None
        or view_result is None
        or view_result.hit is None
    ):
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
            "population_paths": list(POPULATION_PATHS),
            "view_filter": {"usd_path_in": [VIEW_PATH]},
            "ray": {
                "origin": list(RAY_ORIGIN),
                "unit_direction": list(RAY_DIRECTION),
                "min_distance": 0.0,
                "max_distance": RAY_DISTANCE,
                "precision": "aabb_candidate",
                "mode": "closest",
            },
        },
        "publication": {
            "source_ordinal": publication.published_ordinal,
            "object_count": publication.object_count,
            "excluded": publication.excluded_object_count,
        },
        "result": {
            "source_ordinal": view_result.source_ordinal,
            "has_hit": True,
            "object_id": view_result.hit.object_id,
            "truth": view_result.hit.truth.name.lower(),
            "population_closest": {
                "prim_path": population_path,
                "distance": population_result.hit.distance,
            },
            "view_closest": {
                "prim_path": view_path,
                "distance": view_result.hit.distance,
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
        present_interactive("ovspatialquery: filtered view", pixels)
    return payload, pixels


if __name__ == "__main__":
    try:
        run()
    except BaseException as failure:
        print(f"{EXAMPLE}: {type(failure).__name__}: {failure}", file=sys.stderr)
        raise
