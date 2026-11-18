# Single queries on a population

Runnable example: [`assets/python/queries.py`](../assets/python/queries.py) (raycast
closest/any/multiple, sphere and box sweeps, overlap any/multiple, mesh face hits, view,
participation bits, path scope). Each call below runs one query on the caller thread and returns at
once. For many queries per frame, use a plan ([batches](batches.md)).

## Calls

Everything here, enums included, is imported from `ovspatialquery`. `target` is
`population.target` or `view.target`.

```python
target.raycast_closest(RaycastQuery(origin, unit_direction, max_distance, min_distance=0.0,
                       ordinal=OrdinalSelection.head(), precision=QueryPrecision.EXACT,
                       query_bits=0, contact_tolerance=0.0))      -> RaycastResult
target.raycast_any(RaycastQuery(...))                              -> RaycastResult
target.raycast_multiple(RaycastQuery(...), capacity,
                        mesh_hits=MeshHits.ALL)                    -> MultipleResult
target.sweep_sphere_closest(SphereSweepQuery(center, radius, unit_direction, max_distance,
                            min_distance=0.0, ...same keywords)) -> SphereSweepResult
target.sweep_sphere_any(SphereSweepQuery(...))                    -> SphereSweepResult
target.sweep_sphere_multiple(SphereSweepQuery(...), capacity)      -> MultipleResult
target.sweep_box_closest(BoxSweepQuery(center, half_extents, rotation_xyzw, unit_direction,
                         max_distance, min_distance=0.0, ...))   -> BoxSweepResult
target.sweep_box_any(BoxSweepQuery(...))                           -> BoxSweepResult
target.sweep_box_multiple(BoxSweepQuery(...), capacity)            -> MultipleResult
target.overlap_any(OverlapQuery(shape, ordinal=..., precision=EXACT, query_bits=0, ...))
                                                                   -> OverlapResult
target.overlap_multiple(OverlapQuery(...), capacity)               -> MultipleResult
```

- Overlap shapes: `Sphere(center, radius)`, `OrientedBox(center, rotation_xyzw, half_extents)`
  (rotation before half extents), and `Aabb(minimum, maximum)`.
- Single results have `.source_ordinal` (the published ordinal read) and `.hit`, a
  `SceneHit` or `None` on a miss.
- `MultipleResult(source_ordinal, hits, truncated)`: `hits` is a tuple of `SceneHit` in
  unspecified order, so sort. A raycast reports one hit per triangle a mesh crosses (both
  walls of a closed double-sided mesh; only front faces of a single-sided one) and one per
  other object; `mesh_hits=MeshHits.NEAREST` keeps one hit per object, its nearest. A sweep
  reports one hit per object reached, a mesh included, at its first contact (an object the shape
  starts in at `min_distance`). `capacity` is required; `truncated=True` means more hits
  existed, and the stored ones are the nearest.
- `SceneHit` fields: `object_id`, `geometry_type` (`GeometryType.SPHERE/BOX/CAPSULE/CONVEX/TRIANGLE_MESH`),
  `geometry_value`, `truth` (`HitTruth.EXACT_SURFACE` or `BOUNDS_CANDIDATE`), `position`,
  `normal`, `distance`, `query_position` (sweeps only), `material_id` (0 when none; on an ovstage
  scene the bound physics material's prim path id, like `object_id`), and for
  an exact mesh hit `source_face_index`, `source_face_corner_indices`, and `barycentrics`. An overlap hit has
  only identity and `truth`; `position`, `normal`, and `distance` are `None`.

## Defaults that bite

| Default | Effect | What to pass |
| --- | --- | --- |
| `capacity` of `*_multiple` | Too small drops hits; the call still succeeds | check `result.truncated`, retry larger |
| `query_bits=0` | Accepts every object, even when a participation map is set | the bits you want |
| No participation map, or `default_bits` | Unlisted objects have all 128 bits (no map) or exactly `default_bits`, so a nonzero query still accepts them | `default_bits=0` for objects with no category |
| `mesh_hits=MeshHits.ALL` | A multiple raycast lists every mesh crossing; with `contact_tolerance > 0` it raises `NOT_SUPPORTED` | `MeshHits.NEAREST` for one hit per object |
| `min_distance=0.0` | Stored solids are filled. A ray or sweep that starts inside one reports it at distance 0 | start outside, or raise `min_distance` |
| Mesh `doubleSided` unauthored (false) | An exact ray from the back side passes through the mesh. Bounds still reports it | author `uniform bool doubleSided = 1` if both sides must block, or `both_sides=True` for one query |

Single queries default to `EXACT`; batch plan configs still default to
`BOUNDS_CANDIDATE` ([batches](batches.md)). Pass `precision=QueryPrecision.BOUNDS_CANDIDATE`
for cheap AABB candidates. `EXACT_SINGLE` works only for raycasts; sweeps and overlaps
that use it fail with `NOT_SUPPORTED`.

## Pattern

```python
from ovspatialquery import Aabb, BoxSweepQuery, OverlapQuery, RaycastQuery, SphereSweepQuery

hit = target.raycast_closest(RaycastQuery((0, 0, 0), ovsq.unit((1, 0, 0)), 20.0)).hit
if hit is not None:
    print(scene.path(hit.object_id), hit.distance, hit.position, hit.normal)

# A segment from a to b: unit direction, and max_distance = length.
d, length = ovsq.unit([b[i] - a[i] for i in range(3)]), math.dist(a, b)
s = target.sweep_sphere_closest(SphereSweepQuery(a, 0.5, d, length)).hit
center_at_contact = s.query_position if s else None

box = BoxSweepQuery(center=a, half_extents=(0.5, 0.5, 0.5), rotation_xyzw=ovsq.yaw_xyzw(45),
                    unit_direction=d, max_distance=length)

result = target.overlap_multiple(OverlapQuery(Aabb((-5, -1, -1), (6, 1, 1))), 256)
assert not result.truncated
paths = sorted(scene.path(h.object_id) for h in result.hits)
```

### Querying a subset of one population

Views, path scopes ("everything under X except Y"), and participation bits: refer to
[subsets](subsets.md).

### What a publication stores

- Every world matrix applies exactly, non-uniform scale, shear, and mirrors included, to
  meshes, `convexHull` meshes, and Cubes. A sheared Cube is stored as a convex hull of its
  corners (`GeometryType.CONVEX`). A Sphere or Capsule under non-uniform scale is left out
  as `UNSUPPORTED_SHAPE`, never stretched; `ovsq.publish` names it.
- A `convexHull` mesh is the hull a PhysX simulation collides with, not the points' own
  hull: at most `physxConvexHullCollision:hullVertexLimit` vertices (clamped to 8 to 64;
  64 when the stage does not hold it or it is negative), so some points can lie outside
  it. For the points' own hull, register them as an explicit convex.
- Bounds hits and pairs use boxes computed from the geometry and world matrix, never the
  authored `extent`.

## Reading hits

- **Exact ray hit.** `position` is on the surface, and `normal` is the outward unit normal
  (float32, so compare with a tolerance of about 1e-6). `distance` is measured along the
  unit direction.
- **Sweep hit.** `position` is the contact point **on the stored object** (on its AABB for
  a bounds hit). `query_position` is the moving shape's center at contact,
  `center + distance * unit_direction`.
- **Box sweep contact point.** When an edge or face touches a face, the contact point is
  any point of the touching set. Compare `distance` or `query_position`, not `position`.
- **Mesh hit.** `source_face_index` is the authored USD face, counted in
  `faceVertexCounts` order. `source_face_corner_indices` are corner numbers within that
  face, not point indices. A quad is split into a fan, (0,1,2) and (0,2,3).
  `barycentrics` weight those three corners. A front face winds counterclockwise when
  seen from the side its normal points to; a `leftHanded` or mirrored mesh keeps its
  authored front faces. Faces in `holeIndices` are skipped. `subdivisionScheme` is ignored:
  every mesh is queried as its authored faces (the control cage, not the refined surface).
- **Object ids.** An `object_id` is a prim path id. Convert it with `scene.path(id)`.

## Pitfalls (symptom -> fix)

- Distances are 0.5 to 1 short, or an overlap reports an object that clearly does not
  touch -> you are getting bounds hits: you passed `BOUNDS_CANDIDATE`, or the numbers came
  from a plan. Check `hit.truth == HitTruth.EXACT_SURFACE`.
- `ApiError ... INVALID_ARGUMENT` from a query -> the direction or quaternion is not unit
  length (tolerance about 1e-5 on the squared length). Normalize with `ovsq.unit`; for a
  rotation about Z use `ovsq.yaw_xyzw`.
- `ValueError: ray interval must be finite, ordered, and nonnegative` ->
  `min_distance > max_distance`, or a negative or infinite distance. Use a large finite
  `max_distance`.
- `NOT_READY` from a query on `view.target` -> the view was created after the publication.
  Create it before `ovsq.publish`. Publishing the same ordinal again fails with
  `STALE_ORDINAL`; only a new ovstage ordinal can republish.
- `NOT_SUPPORTED` from a sweep or overlap -> you passed `EXACT_SINGLE`. Use `EXACT`.
- A prim never appears in any result -> first check whether the publication left it out:
  its path and reason are in `excluded_objects` / `exclusion_reasons` and in the
  `ovsq.publish` error ([errors](errors.md)). If it was published, the ray misses it, a
  view, scope, or bit filter rejects it, or the ray reaches a single-sided mesh's back face.
- A ray through a mesh misses it with exact precision but hits it with bounds -> the ray
  reaches the back face of a single-sided mesh. Author `doubleSided = 1`, reverse the
  winding, or pass `both_sides=True` on that query or plan launch (a back-face hit then keeps the
  front normal, pointing along the ray).
- A sweep passes through a single-sided mesh -> the triangle's front normal
  `(v1-v0) x (v2-v0)` points along `unit_direction` (n . d > 0 for a sphere, n . d >= 0
  for a box, so a box moving parallel to it ignores it too). Other triangles stop a sweep
  from either side, at an edge or vertex too. Author `doubleSided = 1`, or pass
  `both_sides=True` on the query, to count every side.
- `overlap_any` returns a different object on another run or machine -> "any" makes no
  choice promise. Use `overlap_multiple` and sort.
- A `*_multiple` result is missing objects -> `truncated` is `True`; raise `capacity`.
