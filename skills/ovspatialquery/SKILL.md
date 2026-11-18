---
name: ovspatialquery
description: >-
  Write programs that run spatial queries with ovspatialquery: raycasts, sphere and box
  sweeps, overlaps, batched ray scans (lidar, depth, range sensors), object-pair snapshots
  and enter/exit tracking, over a USD scene loaded in Ovstage or over shapes you register
  yourself, in Python or C/C++. Use for collision, visibility, clearance, placement,
  sensor-simulation or warehouse/robotics queries against a USD stage, for GPU (CUDA)
  plans and GPU producer loops that publish moving objects every step, for querying only
  part of a stage (everything under a path except some subtrees, categories), or when an
  ovspatialquery or Ovstage call fails with NOT_SUPPORTED, STALE_ORDINAL, NOT_READY,
  ORDINAL_MISMATCH, BUSY or BufferError, or a publication leaves prims out
  (excluded_objects, UNSUPPORTED_TYPE, UNSUPPORTED_SHAPE).
license: Apache-2.0
compatibility: >-
  ovspatialquery 0.1.x with Ovstage 0.4 (0.3 stages also work). Python: the ovspatialquery wheel
  plus ovstage and numpy for scene traversal. C/C++: the installed SDK
  (find_package(ovspatialquery)) plus the ovstage package's CMake config. CPU is enough;
  CUDA plans need an NVIDIA driver.
metadata:
  author: NVIDIA Omniverse
---

# ovspatialquery

ovspatialquery answers geometric questions about a scene: which object a ray hits first,
what a moving sphere or box touches, what overlaps a shape, which pairs of objects touch
and which pairs started or stopped touching. The scene is either an ovstage stage opened
from USD (an *ovstage population*) or shapes you register directly (an *explicit
population*). Queries run against a published version of a population.

## Start here

1. Python: copy [`scripts/ovsq.py`](scripts/ovsq.py) next to your program and import it.
   It covers what the package leaves to you: ovstage ordinals and time units, choosing
   prims, unit directions, and per-ray closest hits from a batch. It is short; read it.
2. Find the asset closest to your task in the table below and adapt it. Every asset is a
   complete program that runs as is and checks its own answers; CI runs all of them.
3. Read the matching reference only for the part you are changing.

| Task | Asset to start from | Reference |
| --- | --- | --- |
| Closest/any/all-hit raycast, sphere or box sweep, overlap | [`assets/python/queries.py`](assets/python/queries.py) | [queries](references/queries.md) |
| Subsets of one population: views, "everything under X except Y", categories | [`assets/python/queries.py`](assets/python/queries.py) | [subsets](references/subsets.md) |
| Animated USD (time samples), querying older versions | [`assets/python/time_versions.py`](assets/python/time_versions.py) | [time and pairs](references/time-and-pairs.md) |
| Which pairs overlap; entered/exited pairs over time | [`assets/python/pairs.py`](assets/python/pairs.py) | [time and pairs](references/time-and-pairs.md) |
| Many rays per frame (lidar, depth image, range scan) | [`assets/python/batch_raycast.py`](assets/python/batch_raycast.py) | [batches](references/batches.md) |
| Large natively instanced scenes (warehouse racks), placement checks | [`assets/python/warehouse.py`](assets/python/warehouse.py) | [batches](references/batches.md) |
| No USD: register spheres, boxes, capsules, convexes, triangle meshes | [`assets/python/explicit_shapes.py`](assets/python/explicit_shapes.py) | [explicit shapes](references/explicit-shapes.md) |
| C or C++ program, CMake against the SDK | [`assets/native/`](assets/native/) | [native](references/native.md) |
| GPU: CUDA plans, publishing moving objects and querying every step without host waits | [`assets/native/gpu_producer.cpp`](assets/native/gpu_producer.cpp) | [cuda](references/cuda.md) |
| An error code you do not understand | | [errors](references/errors.md) |

## Rules that prevent most failures

1. **Load a USD scene in this order** (or use `ovsq.Scene`): open into ordinal 1,
   `advance_write_floor(1).wait()`, `compute_hierarchy(1, 2, DEFAULT_CPU)`,
   `advance_write_floor(2).wait()`, then publish ordinal 2. Every later ovstage write uses
   a new, larger ordinal and is sealed with `advance_write_floor(n).wait()` before you
   publish `n`. Publishing any other ordinal fails with `STALE_ORDINAL`.
2. **ovstage times are seconds, not time codes.** `open_usd*` and `update_from_usd_time`
   divide by the layer's `timeCodesPerSecond` (USD default 24 when unauthored). Pass
   `time_code / timeCodesPerSecond`, and pass a time when opening: the default ignores
   attributes that only have time samples. Wrong times fail silently.
3. **Select only `Cube`, `Sphere`, `Capsule`, and `Mesh` prims, and check what was left
   out.** Pass an `ovstage.Filter` to `create_population`. An empty filter selects every
   such prim, floors and helper gprims too. Stage content never fails a publication: a
   matched prim it cannot publish exactly (an Xform under a `usd-path PREFIX`, a light, a
   Cylinder, a Sphere or Capsule under non-uniform scale) is silently left out and listed
   in the result's `excluded_objects` / `exclusion_reasons`. `ovsq.publish` raises naming
   each one (`allow_exclusions=True` accepts them). Select exact paths
   (`Predicate("usd-path", FilterOp.IN, paths)` with `paths` from `ovsq.gprim_paths`) or
   AND a prefix ending in `/` with `usd-prim-type IN ["Cube", ...]` (`ovsq.gprims_under(root)`
   does both); predicate values are lists. Unknown paths and wrong-case type names match
   nothing, silently: check `object_count` as well as `excluded_object_count`.
   A filter cannot say "everything except": pass `exclude_paths=[...]` (and `include_paths`)
   to `create_population` or `create_view` instead. The nearest listed path at or above a prim
   decides, by path element, and scoped-out prims are not listed as exclusions
   ([subsets](references/subsets.md)).
4. **Native instances publish as their proxies, under their own type names.** Each supported
   gprim inside an `instanceable = true` instance is an object at its own path and world pose
   (ovstage's hidden `/__Prototype_*` copies never are). Open the stage as authored, with the
   `ALL` or `RENDERING` domain (`ovsq.Scene` uses `ovsq.physics_desc`: `ALL` on 0.3,
   `RENDERING` plus a physics selector on 0.4, which has no `ALL`). Proxies are `CubeInstance`,
   `SphereInstance`, `CapsuleInstance`, `MeshInstance`: `usd-prim-type IN ["Cube"]` selects
   only authored cubes, so a prefix plus `Cube` publishes 0 objects on instanced racks.
   List both names.
5. **Single queries are exact; batch plans are not.** `RaycastQuery`, sweeps, and
   `OverlapQuery` default to `QueryPrecision.EXACT`. Plan configs default to
   `BOUNDS_CANDIDATE` (axis-aligned box hits): pass `precision=QueryPrecision.EXACT` and
   request `BatchResultField.DISTANCE` when you need surface distances.
6. **Directions must be unit length** (to about 1e-5) and quaternions are unit `(x, y, z, w)`.
   Normalize in double precision (`ovsq.unit`); the library rejects, it never normalizes.
   `max_distance` and every returned `distance` are in scene units along that unit
   direction: normalize the direction but never rescale a distance limit you were given.
   For a segment from `a` to `b`, use `unit(b - a)` and `max_distance = |b - a|`.
7. **Results name objects by id.** For ovstage populations, `object_id` is the prim's path
   id: `ovsq.Scene.path(id)` or `ovstage.PathDictionary(stage).path_to_string(id)`. For
   explicit populations it is the object value you registered (use nonzero values).
8. **Create views and participation maps before the publication** that should include
   them; a view created after it is `NOT_READY` until the next publication.
9. **Let `with` blocks own lifetimes, inside the stage's.** Open the library and instance
   with `with`; the instance closes its plans, reporters, views, and populations when the
   block ends. Use `with operation:` (it waits; closing an unfinished operation by hand is
   `BUSY`) and read results inside it with `.to_numpy()`: a result read after the block
   raises `BufferError`. `operation.finish()` waits, closes, and raises on failure. Close
   every ovspatialquery object before the ovstage stage is destroyed: a Python population
   that outlives its stage can crash the process at exit.

## Python skeleton

```python
import ovstage
import ovsq  # copied from this skill's scripts/
from ovspatialquery import Library, RaycastQuery

with ovsq.Scene("scene.usda") as scene:  # ordinal bookkeeping, seconds conversion
    paths = ovsq.gprim_paths("scene.usda", roots=["/World/Obstacles"])
    selected = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
    with Library.open(require_ovstage=True) as library, library.instance() as instance:
        population = instance.ovstage(ovstage, scene.stage).create_population(selected)
        assert ovsq.publish(population, scene.ordinal).object_count == len(paths)
        result = population.target.raycast_closest(
            RaycastQuery(
                origin=(0.0, 0.0, 1.0), unit_direction=ovsq.unit((1, 0, 0)), max_distance=100.0
            )
        )
        if result.hit is not None:
            print(scene.path(result.hit.object_id), result.hit.distance, result.hit.position)
```

`Library.open()` with no argument loads the shared library bundled in the wheel; pass a
path (for example `$OVSQ_SDK/lib/libovspatialquery.so`) to use an SDK install instead.
Several copies of the library can be loaded in one process (a program linking the SDK
that also imports the wheel, two bundling packages), and each works on its own. A handle,
target, or `TargetRef` from one copy passed to another is undefined behavior: it is not
detected and can silently act on an unrelated object. Pass each only to the instance that
created it; `library.path` names the file a `Library` loaded.

## Verify before you finish

Run the program and compare at least one answer with a hand calculation (a ray along an
axis against an axis-aligned box is easy). Most wrong programs fail silently: zero
objects published, bounds hits from a plan, or time samples read at the wrong time.
