# Explicit shapes (no USD, no ovstage)

Register geometry definitions, place instances of them in an *explicit population*,
publish, query. Complete program: [`assets/python/explicit_shapes.py`](../assets/python/explicit_shapes.py).
Needs only `ovspatialquery` and `numpy`; every input is a NumPy (or other DLPack) column
passed by keyword, named as in `ovspatialquery_explicit.h`.

## Pattern

```python
import numpy as np
from ovspatialquery import ExplicitUpdateKind, GeometryType, Library, RaycastQuery

u64 = lambda v: np.asarray(v, dtype=np.uint64)
with Library.open(require_explicit=True) as library, library.instance() as instance:
    adapter = instance.explicit()
    adapter.register_spheres(definition_values=u64([1]), radii=np.array([0.5])).finish()
    adapter.register_boxes(
        definition_values=u64([1]), half_extents=np.array([(1.0, 0.5, 0.25)])
    ).finish()
    with adapter.create_population() as population:
        population.enqueue_set(
            1,
            object_values=u64([101, 102]),  # nonzero, unique; hits report them
            geometry_types=np.array([GeometryType.SPHERE, GeometryType.BOX], dtype=np.int32),
            geometry_values=u64([1, 1]),  # definition values
            translations=np.array([(0.0, 0, 0), (5, 0, 0)]),
            rotations_xyzw=np.array([(0.0, 0, 0, 1), (0, 0, 0.7071068, 0.7071068)]),
            uniform_scales=np.array([1.0, 2.0]),  # optional, default 1
        ).finish()
        hit = population.target.raycast_closest(RaycastQuery((5, -9, 0), (0, 1, 0), 100.0)).hit
        population.enqueue_update(  # move 101; base must be the head, target > base
            1,
            2,
            kind=ExplicitUpdateKind.POSES,
            object_values=u64([101]),
            translations=np.array([(0.0, 0, 3)]),
            rotations_xyzw=np.array([(0.0, 0, 0, 1)]),
        ).finish()
# Leaving the instance closes populations, then destroys every registered definition.
```

## Signatures (`adapter = instance.explicit()`, after `Library.open(require_explicit=True)`)

Every call returns an `Operation`; `.finish()` waits, closes it and raises `ApiError` if it
failed. All definition and instance arguments are keyword-only DLPack columns.

- `register_spheres(definition_values, radii, copy=False)`
- `register_boxes(definition_values, half_extents, copy=False)`
- `register_capsules(definition_values, radii, half_segments, copy=False)`
- `register_convexes(definition_values, vertex_offsets, vertex_counts, vertices,
  face_offsets=None, face_counts=None, face_vertex_offsets=None, face_vertex_counts=None,
  face_vertex_indices=None, enclosure_tolerance=0.0, hull_vertex_limit=0, copy=False)`; the
  five face columns come all together or not at all. Without them the library takes the convex
  hull of the points, which may have at most 255 vertices and 254 faces (129 points always fit);
  a larger hull is refused, not reduced. Points too nearly flat to form an enclosing hull of their
  own are refused too, not replaced by a box: give a thin collider a real thickness (1e-3 of its
  extent is safe; dense point sets can need 1e-2), or register a box or a triangle mesh. The
  refusal says how far outside the farthest point lies; an `enclosure_tolerance` up to 2^-16
  above that registers it. `hull_vertex_limit` 8 to 64 registers instead the hull a PhysX
  simulation collides with for that collider, as ovstage publishes every `convexHull` mesh: at
  most that many vertices, not necessarily points, possibly leaving points outside, and never
  refused for size.
- `register_triangle_meshes(definition_values, vertex_offsets, vertex_counts,
  triangle_offsets, triangle_counts, vertices, triangle_indices, double_sided=None,
  source_face_material_ids=None, source_face_indices=None, source_face_corner_indices=None,
  copy=False)`.
  Several definitions pack their rows back to back; offsets index the packed rows and
  triangle indices are local to each mesh.
- `create_population(PopulationConfig())` -> `ExplicitPopulation` (a context manager).
- `population.enqueue_set(target_ordinal, *, object_values, geometry_types, geometry_values,
  translations, rotations_xyzw, uniform_scales=None, participation_bits=None, material_ids=None,
  copy=False)`.
- `population.enqueue_update(base_ordinal, target_ordinal, *, kind=ExplicitUpdateKind.COMPLETE_ROWS,
  object_values, translations, rotations_xyzw, geometry_types=None, geometry_values=None,
  uniform_scales=None, participation_bits=None, material_ids=None, removed_object_values=None,
  copy=False)`.
  `COMPLETE_ROWS` adds or replaces rows (geometry columns required) and removes
  `removed_object_values`; `POSES` moves existing objects (no geometry columns, no bits);
  `POSES_AND_BITS` also requires `participation_bits`; poses rows take no `material_ids` and
  keep each object's material.
- `population.create_view(u64_object_values)` -> a `View` over those objects.
- `population.info()` -> `PopulationInfo(ready, head_ordinal, object_count, ...)`.
  For the publication record instead, use `with op: op.wait(); op.fetch_population_update()`.
- `adapter.destroy_definition(GeometryDefinitionRef(GeometryType.SPHERE, 1))`: optional; the
  instance destroys what is left when it closes.

Column dtypes (checked in Python before the call; a wrong one raises `ValueError` naming
the column and dtype):

| Column | dtype, shape |
| --- | --- |
| definition/object/removed values, offsets, vertex/triangle counts | `uint64[N]` |
| sphere/capsule radii, capsule half segments | `float64[G]` |
| box half extents | `float64[G,3]` |
| mesh and convex vertices | `float32[V,3]` (2-D, not flattened) |
| triangle indices | `uint32[T,3]` |
| geometry types | `int32[N]` (`GeometryType` values) |
| translations / rotations xyzw / uniform scales | float32 or float64 `[N,3]` / `[N,4]` / `[N]` |
| `double_sided` | `uint8[G]` (optional) |
| `participation_bits` | `uint64[N,2]`, column 0 = bits 0-63, 1 = bits 64-127 (optional, default all 128 bits) |
| `material_ids` | `uint64[N]`, the material every hit on the object reports (optional, default 0 = none) |
| `source_face_material_ids` | `uint64[T]`, one per triangle; 0 lets it report the instance's material (optional) |
| `source_face_indices` / `source_face_corner_indices` | `uint64[T]` / `uint32[T,3]`, together: the polygon each triangle came from and its three corners in it (optional) |

## Semantics that bite

- World point = `translation + R(rotation_xyzw) * (uniform_scale * local)`.
  Quaternions are `(x, y, z, w)`; yaw about +Z is `(0, 0, sin(a/2), cos(a/2))`
  (`ovsq.yaw_xyzw`).
- Capsule segment lies on local **Y** (`+-half_segment`); box and sphere are centered.
- `uniform_scales` is optional and defaults to 1; forgetting it gives no error.
- A pose has no per-axis scale, shear, or mirror: bake those into a convex or mesh
  definition's points (an ovstage population does this for you, exactly). When the baked
  matrix has a negative determinant (a mirror), also reverse each triangle's winding, or a
  single-sided mesh faces the other way.
- Definition values are 1 to 2^63 - 1 and unique per geometry type (sphere 1 and box 1 coexist);
  ovstage populations in the same instance take values from 2^63 up.
- Mesh hits: by default `hit.source_face_index` is the triangle's row in that mesh's
  `triangle_indices`, `hit.barycentrics[i]` weights that triangle's i-th listed vertex,
  `hit.source_face_corner_indices == (0, 1, 2)`. Other shapes give `None` for these.
  With `source_face_indices`/`source_face_corner_indices` (a quad: `(0,1,2)`, `(0,2,3)`) they
  report your polygon and its corners; a triangle may not name one corner twice. With
  `source_face_material_ids` too, one polygon's triangles need one material and every
  polygon index must be below 2^32 - 1 (`INVALID_ARGUMENT` otherwise).
- Meshes are front-face only: a ray must see the triangle counterclockwise
  (normal `(v1-v0) x (v2-v0)` facing the ray origin). A sweep ignores a single-sided
  triangle whose normal points along its motion (for a box, also one it moves parallel
  to) and meets the others from either side. Pass `double_sided` otherwise, or
  `both_sides=True` on one query or plan launch (back-face ray hits keep the front normal).
- A ray starting inside a sphere reports it at distance 0.
- The first set may use any ordinal >= 0; each update needs `base == head` and
  `target > base` (gaps allowed). `PopulationConfig().retained_ordinal_capacity == 2`,
  so `OrdinalSelection.retained_exact(old)` still queries the previous version.
- `hit.object_id` is the object value you registered; `hit.geometry_value` the definition.
- Overlap: `target.overlap_any(q)` returns one hit; `target.overlap_multiple(q, capacity)`
  returns `MultipleResult(source_ordinal, hits, truncated)`. Check `truncated`.
- Without `copy=True`, an operation borrows its columns until it is closed: do not write to
  those arrays before `.finish()` (or the `with` block) returns.

## Pitfalls (symptom -> fix)

- `ValueError: triangle_indices must be uint32; got ... 64 bits` (or any `must be <dtype>`)
  -> NumPy defaults lists to int64/float64; cast with `np.asarray(v, dtype=...)` per the table.
- `TypeError: radii must support DLPack; got list` -> pass NumPy arrays, not lists.
- `enqueue_explicit_population_set: INVALID_ARGUMENT` -> an object value is 0 or repeated,
  a geometry value names an unregistered definition, or a quaternion is not unit length
  (to 1e-5). Offset index-based ids (`1000 + i`).
- `..._registration: INVALID_ARGUMENT` -> definition value 0 or >= 2^63, nonpositive radius or
  half extent, or a convex with fewer than 4 non-coplanar points.
- `enqueue_explicit_population_set: INVALID_STATE` -> the population already has a head;
  use `enqueue_update(head, new_ordinal, ...)`.
- `enqueue_explicit_population_update: ORDINAL_MISMATCH` -> `base` is not the head
  (`population.info().head_ordinal`).
- POSES update `INVALID_ARGUMENT` -> a row names an object the head does not have.
- `ValueError: a POSES update takes no geometry_types` / `a COMPLETE_ROWS update requires
  geometry_types` -> match the columns to `kind`.
- `destroy_geometry_definition: BUSY` -> a live population still uses it; close the
  population first, or leave it to the instance.
- `raycast_closest_sync: INVALID_ARGUMENT` -> direction not unit length; `ovsq.unit(d)`.
- Ray from below a mesh returns `None` -> back face; flip the winding or pass
  `double_sided=np.ones(G, np.uint8)`.
- `operation.wait()` returned `FAILURE` and nothing raised -> `wait()` does not raise; use
  `.finish()`.
