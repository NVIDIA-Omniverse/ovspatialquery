# Batches: many rays per frame, instanced warehouses, placement checks

Assets: [`batch_raycast.py`](../assets/python/batch_raycast.py) (range scan reused across
frames), [`warehouse.py`](../assets/python/warehouse.py) (instanced racks, 3D scan,
placement with `overlap_multiple`). Use a **plan** for hundreds of rays per frame: create
it once (it reserves memory), then `enqueue` one launch per frame. For a few queries use
the sync calls ([queries](queries.md)).

## The helper: `ovsq.RaycastBatch`

```python
with ovsq.RaycastBatch(instance, population.target, max_rays=N) as scan:  # EXACT by default
    out = scan.run(origins, directions, max_distance, min_distance=0.0)  # every frame
# out["hit"] bool[n]; out["distance"] float64[n], inf on miss; out["object_id"] uint64[n], 0 on miss
```

`run` takes any n <= `max_rays`, normalizes directions, broadcasts scalar distances,
raises unless every row status is SUCCESS, and closes the launch. Closest hit only; keep
`precision` EXACT (`BOUNDS_CANDIDATE` raises `AttributeError`). Depth image: `H*W`
directions, `reshape(H, W)` the outputs. 3D scan: rotate one local direction table by
the vehicle yaw each frame.

## The plan API (what the helper wraps)

`RaycastPlanConfig(target, hit_mode, maximum_query_count, maximum_hits_per_query,
hit_capacity_per_slot, precision=BOUNDS_CANDIDATE, device=Device.cpu(),
result_slot_count=1, result_fields=BatchResultField(0), contact_tolerance=0.0)`

- `target` is `population.target.reference`; `hit_mode` is `HitMode.CLOSEST | ANY | MULTIPLE`.
- **Plans default to `BOUNDS_CANDIDATE`** (single queries default to EXACT): pass
  `precision=QueryPrecision.EXACT` for surface hits.
- `hit_capacity_per_slot` must be >= `maximum_query_count * maximum_hits_per_query`.
- `result_fields` (`BatchResultField`, OR them): `POSITION` float64[H,3], `NORMAL`
  float32[H,3], `DISTANCE` float64[H], `GEOMETRY_IDENTITY` (`result.geometry.types/values`),
  `SOURCE_FEATURE` (EXACT only), `MATERIAL` (`result.material_ids` uint64[H]). `QUERY_POSITION`
  is sweep-only (ValueError on raycast).
  Hit ranges and `object_ids` are always present; the default 0 gives **no distances**.

```python
with instance.create_raycast_plan(
    RaycastPlanConfig(
        target=population.target.reference,
        hit_mode=HitMode.CLOSEST,
        maximum_query_count=N,
        maximum_hits_per_query=1,
        hit_capacity_per_slot=N,
        precision=QueryPrecision.EXACT,
        result_fields=BatchResultField.DISTANCE,
    )
) as plan:
    for frame in frames:  # n <= N rows per launch
        operation, reservation = plan.enqueue(
            origins=origins_f64,  # float64 [n,3]
            unit_directions=dirs_f32,  # float32 [n,3], unit length
            max_distances=maxs_f64,  # float64 [n]
            min_distances=None,  # optional float64 [n]; None = 0
            copy=True,  # library copies the columns; without it, do not touch them until close
        )
        with operation:  # waits and closes, freeing the result slot
            operation.wait()
            result = operation.fetch_raycast_batch().to_numpy()  # numpy copies outlive the op
        r = result.hit_ranges
        assert (r.query_status == BatchRowStatus.SUCCESS).all()  # uint32[n]
        offsets, counts, truncated = r.hit_offsets, r.hit_counts, r.truncated  # [n]
        ids = result.object_ids  # uint64[H], H = r.stored_hit_count[0]
        dist = result.exact.distances  # float64[H]; result.candidate.* for BOUNDS_CANDIDATE
```

Other `enqueue` keywords: `query_bits` uint64[n,2] (column 0 = bits 0-63, column 1 =
bits 64-127; PhysX filter words `uint32[n,4]` become it as
`np.ascontiguousarray(w, "<u4").view("<u8")`), `maximum_hits` uint64[n] (MULTIPLE only),
`ordinal=OrdinalSelection.head()`, `cuda_stream`/`cuda_wait_event` (CUDA plans). Any
`__dlpack__` array works (NumPy, PyTorch, Warp); lists and scalars do not.

Reading results: query `i` owns flat rows `offsets[i] : offsets[i] + counts[i]` of every
payload column (not row `i`); `counts[i] == 0` is a miss. MULTIPLE hits of one query come
in unspecified order; sort by distance yourself. `result.common.truth` says which arm is
filled: `EXACT_SURFACE` -> `result.exact` (`positions`, `normals`, `distances`),
`BOUNDS_CANDIDATE` -> `result.candidate` (`bounds_positions`, `bounds_normals`,
`distances`); the other is `None`.

### Overlap and sweep plans (verified minimal calls)

Same shape as the raycast plan; each launch is `with operation:` + `.to_numpy()`.

```python
oplan = instance.create_overlap_plan(
    OverlapPlanConfig(
        target=population.target.reference,
        shape=OverlapBatchShape.sphere(0.5),  # or .box((hx, hy, hz)) / .aabb((hx, hy, hz))
        hit_mode=HitMode.MULTIPLE,  # ANY or MULTIPLE, never CLOSEST
        maximum_query_count=Q,
        maximum_hits_per_query=K,
        hit_capacity_per_slot=Q * K,
        precision=QueryPrecision.EXACT,
    )
)
operation, _ = oplan.enqueue(translations=centers_f64, copy=True)  # BOX: rotations_xyzw=
# operation.fetch_overlap_batch(): hit_ranges + object_ids only
splan = instance.create_sphere_sweep_plan(
    SphereSweepPlanConfig(
        target=population.target.reference,
        radius=0.25,
        maximum_query_count=Q,
        precision=QueryPrecision.EXACT,
        result_fields=BatchResultField.DISTANCE | BatchResultField.QUERY_POSITION,
    )
)
operation, _ = splan.enqueue(
    centers=centers_f64, unit_directions=dirs_f32, max_distances=maxs_f64, copy=True
)  # radii=float64[Q] per row
# operation.fetch_sphere_sweep_batch(): exact.distances, exact.query_positions (center);
# exact.positions (contact on the object) needs BatchResultField.POSITION
```

Sweep plans default to `HitMode.CLOSEST` (capacity is set for you). `HitMode.ANY` keeps one hit
of any object reached. `HitMode.MULTIPLE` keeps one hit per object reached (a mesh gives one
row, not one per triangle), up to `maximum_hits_per_query` per row and `hit_capacity_per_slot`
per launch, nearest first when truncated. Overlap BOX
rotations are float32 or float64 [Q,4] xyzw. `BoxSweepPlanConfig(half_extents=...)` with
`enqueue(centers=, rotations_xyzw=float32[Q,4], unit_directions=, max_distances=)` and
`fetch_box_sweep_batch()` follows the sphere pattern.

## Instanced warehouses

Open racks authored as `instanceable = true` references as they are: each gprim part
(Cube, Sphere, Capsule, Mesh) of each instance is an **instance proxy**, one object at its own path
(`/World/Racks/Row_00/Bay_03/Upright_L`) and pose. ovstage's `/__Prototype_*` copies are
never published. Proxies exist only under the `ALL` or `RENDERING` domain (`ovsq.Scene`
uses `ovsq.physics_desc`: `ALL` on 0.3, `RENDERING` plus a physics selector on 0.4). Their type is `CubeInstance` (`SphereInstance`, ...), and explicit filters
use ovstage's names: `usd-prim-type IN ["Cube"]` selects authored cubes only.
`ovsq.gprims_under("/World/Racks")` ANDs the prefix with all eight names; the prefix alone
also matches the row and bay groups (`Xform`, `XformInstance`), which are not objects and
are left out as `UNSUPPORTED_TYPE` (`ovsq.publish` raises naming them), and dropping it
pulls in the floor. Scale: 720 boxes load in about 20 ms; a 5,760-ray exact scan about 4 ms on CPU.

## Placement checks

`population.target.overlap_multiple(OverlapQuery(OrientedBox(center=...,
rotation_xyzw=ovsq.yaw_xyzw(deg), half_extents=...)), capacity=K)` returns a
`MultipleResult(source_ordinal, hits, truncated)`; blockers are `hit.object_id` of each
hit (`overlap_any` returns only one). Single queries are EXACT by default. Overlap is
closed: touching counts. Keep surfaces a box may rest on (floor, shelf) out of the
population, or lift the box if resting is allowed. If `truncated`, raise `capacity`.

## Pitfalls

- `ValueError: unit_directions must be float32` / `origins must be float64` -> build
  columns with explicit dtypes (an integer `np.array([[1, 0, 0]])` is int64).
  `TypeError: ... must support DLPack; got float` -> pass arrays, not scalars or lists.
- `enqueue` raises `INVALID_ARGUMENT` (no detail) -> `n > maximum_query_count`.
- Every row "misses", `stored_hit_count == 0` -> `query_status` rows are
  `INVALID_ARGUMENT` (1): directions are not unit length. Normalize in float64, then cast.
  Always check `query_status == BatchRowStatus.SUCCESS`.
- Fewer hits than expected, no error -> `hit_capacity_per_slot` too small; rows report
  `truncated != 0` with count 0. Size it `>= maximum_query_count * maximum_hits_per_query`.
- `result.exact.distances` is `None` -> plan created without `BatchResultField.DISTANCE`.
- `'NoneType' object has no attribute 'distances'` on `result.exact` -> plan precision is
  `BOUNDS_CANDIDATE` (the plan default); read `result.candidate` or create it `EXACT`.
- `RESULT_CAPACITY` on the next `enqueue` -> the previous launch is still open. Use
  `with operation:` per launch, or raise `result_slot_count` (launches in flight).
- `BUSY` from `operation.close()` -> it was still running; `with operation:` waits first.
- `BufferError: ... exported result column(s) are still alive` on close -> you kept an
  `np.from_dlpack` view of a column; read through `.to_numpy()` instead.
- Overlap plan `INVALID_ARGUMENT` -> `HitMode.CLOSEST` at create, or a BOX launch without
  `rotations_xyzw`; `NOT_SUPPORTED` at create -> `EXACT_SINGLE` (raycast only).
- Instanced racks: `object_count == 0`, no error -> the filter says `Cube`, proxies are
  `CubeInstance`; `XformInstance` left out as `UNSUPPORTED_TYPE` -> AND a type predicate.
- `Device.cuda()` plan -> `DEVICE_BUDGET`: a default instance has no device budget. Pass
  `InstanceConfig(device_budgets=(DeviceBudget(Device.cuda(0), 256 << 20),))`; GPU loops in
  [cuda](cuda.md).
