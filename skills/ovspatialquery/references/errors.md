# Errors: symptom -> cause -> fix

Python raises `ovspatialquery.ApiError` as `<C function>: <STATUS>: <detail>` (`err.status`
is an `ApiStatus`, `err.detail` the text). Column and argument mistakes raise `TypeError` or
`ValueError` naming the column before C is called. C returns the status in `.status`; the
detail comes from `ovspatialquery_get_last_error()`. ovstage raises `ovstage.OvstageError`
(C: check `error_op_id_count` after `ovstage_population_wait_op`). `operation.wait()` returns
`OpState.FAILURE` without raising: use `operation.finish()`, which raises. A failure while
a `with` block unwinds is added to the original exception as a note.

Silent failures are the common ones. After every program, check
`publish(...).object_count`, per-row `query_status`, `truncated`, and one answer by hand.

Stage content is never an error. An ovstage publication succeeds without each matched prim
it cannot publish exactly and lists it in the result: `excluded_object_count`, then
`excluded_objects` (object ids, sorted) and `exclusion_reasons` (`OvstageExclusionReason`),
one row per prim. `ovsq.publish` raises `RuntimeError: publishing ordinal N left out K
prim(s): <path> <REASON>, ...`; `ovsq.exclusions(population, result)` gives the same
pairs. In C, the receipt holds both as host DLPack tensors (`uint64`, `int32`), valid until
`destroy_op`. Errors from set and update now mean a caller or configuration mistake.

## Build and link (C/C++)

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `Could not find a package configuration file provided by "ovspatialquery"` | SDK root not on the prefix path | `-DCMAKE_PREFIX_PATH="<sdk>;<ovstage dir>"`, `find_package(ovspatialquery CONFIG REQUIRED COMPONENTS ovstage)` | [native](native.md) |
| `unknown type name 'ovstage_instance_desc_t'`, implicit `ovstage_create_instance` | `ovspatialquery_ovstage.h` does not declare the ovstage entry points | `#include <ovstage/ovstage.h>` as well | [native](native.md) |
| `ovstage/ovstage_api/ovstage_api.h: No such file` | only `ovspatialquery::ovspatialquery` linked; the companion header needs ovstage's include dir | also link `ovstage::ovstage` (`find_package(ovstage CONFIG)`) | [native](native.md) |
| `undefined reference to 'ovstage_create_instance'` (or `ovstage_population_*`) | ovstage include dir added by hand, library not linked | `target_link_libraries(... ovstage::ovstage)` | [native](native.md) |
| `undefined reference to symbol 'sqrt@@GLIBC_2.2.5'` ... `DSO missing from command line` | your C code calls `sqrt` without libm | link `m` | [native](native.md) |
| `Failed to initialize ovstage loader: failed to load '<exe dir>/libovstage.so'` | `ovstage::ovstage_static` looks for the runtime beside the executable | use shared `ovstage::ovstage`, or call `ovstage_initialize` with the binary package root first | [native](native.md) |
| `ovspatialquery_create_instance` returns INVALID_ARGUMENT (3) | a zero `operation_capacity`, `max_cpu_concurrency`, or `host_byte_budget` | set all three nonzero | [native](native.md) |

## Scene loading (ovstage)

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `OvstageError: OP_FAILED: failed to open USD layer: <path>` (C: open returns OK, the wait reports it) | relative path resolved against the cwd | resolve paths from `__file__` / argv; always check the wait | [queries](queries.md) |
| `OvstageError: OP_FAILED: failed to parse inline USDA into anonymous layer` | USDA syntax, for example prim metadata or two properties on one line | one statement per line; `Sdf.Layer.CreateAnonymous(".usda").ImportFromString(text)` raises the parser's line and column | |
| `TypeError: ... unexpected keyword argument 'time_code'` | ovstage 0.3 renamed it `time` | pass the time positionally: `open_usd_from_string(stage, text, 1, seconds)` | [time and pairs](time-and-pairs.md) |
| `OvstageError: OP_FAILED: structural mutation ordinal is at or below the effective global write floor` | `update_from_usd_time` into an ordinal already written and sealed | a new, larger ordinal each write (`ovsq.Scene.set_time`) | [time and pairs](time-and-pairs.md) |
| `OvstageError: INVALID_ARGUMENT: NULL ovstage instance` | the stage was destroyed (`Scene` block left) | keep every write inside the stage's lifetime | |
| No error; animated prims sit at their un-animated pose (a ray from the origin hits at distance 0) | `open_usd*` default time is NaN (USD default time): attributes with only `timeSamples` are ignored | open at `start_code / timeCodesPerSecond` | [time and pairs](time-and-pairs.md) |
| No error; every step after the first is wrong (often clamped to the last sample) | ovstage takes **seconds**; a time code was passed | `time_code / timeCodesPerSecond` (24 when unauthored); `ovsq.Scene.set_time` does it | [time and pairs](time-and-pairs.md) |

## Filters and publication

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `NOT_SUPPORTED` naming a `/__Prototype_*` prim and `RENDERING` | native instances on a stage opened with `PHYSICS` alone: no proxy rows exist | open with `ovsq.physics_desc(ovstage)` (`PopulationDomain.ALL` on 0.3, `RENDERING` plus physics on 0.4) | [batches](batches.md) |
| `NOT_SUPPORTED: operation is not supported` (no prim named) | unknown predicate attribute (`"prim-type"`, `"type"`) | only `usd-path` and `usd-prim-type` | [queries](queries.md) |
| `INVALID_STATE: "/": it has no usd-prim-type`, or a `PREFIX` filter selecting everything | predicate `values` given as a bare string: it is iterated per character (`"/"` matches every prim) | always a list: `Predicate("usd-path", FilterOp.IN, ["/A/Box"])` | [queries](queries.md) |
| `TypeError: expected ovstage.Filter; got ...` | `create_population`/`create_view` take an `ovstage.Filter` | pass the filter itself (no `to_c()`, no raw struct) | [queries](queries.md) |
| `TypeError: include_paths and exclude_paths are sequences of prim paths` | a scope list given as a bare string | a list: `exclude_paths=["/World/Props"]` | [subsets](subsets.md) |
| `TypeError: prim paths are prim path strings or int tokens; got ...` / `ValueError: prim paths must fit uint64` | a scope entry that is not a `str` or a Python `int` in [0, 2**64) (bytes, None, a NumPy scalar) | path strings, or `int()` path ids | [subsets](subsets.md) |
| `OvxError: ovx error 1: Path string is not a valid prim path` (or `must be non-empty`) from `create_population`/`create_view` | a scope path with a trailing `/`, a `//`, or empty | `"/World/Props"` | [subsets](subsets.md) |
| `AttributeError: 'list' object has no attribute 'to_raw'` from `create_population` | a path list passed as the second positional argument, which is the config | keywords: `exclude_paths=[...]` | [subsets](subsets.md) |
| `create_ovstage_population` / `create_ovstage_view: INVALID_ARGUMENT: a scope list's pointer and count disagree, or an entry is zero ...` | a path id of 0 or one the stage's path dictionary cannot decompose; in C also a null list with a nonzero count, or the reverse | Python: path strings or ids from the stage; C: each pointer null exactly when its count is 0 | [native](native.md) |
| No error; a `usd-path PREFIX ["/World/Env"]` filter also selects `/World/Environment`, and there is no way to write "except" | PREFIX compares bytes; filters are an AND with no NOT | a scope: `include_paths` / `exclude_paths`, compared by path element | [subsets](subsets.md) |
| No error; a scoped-out subtree is still hit, or `object_count == 0` with `include_paths` | a misspelled scope path matches nothing; or a nearer include of the same view re-includes it | population scope: check `object_count`; view scope: query the view (a covering `overlap_multiple`), since publication counts ignore views; put exclusions every view needs on the population | [subsets](subsets.md) |
| `STALE_ORDINAL: target ordinal does not advance` from `enqueue_*_set`/`_update` | target is not the newest sealed ovstage ordinal (older, unwritten, unsealed, or already published) | seal every write and publish exactly that ordinal (`ovsq.Scene.ordinal`) | [queries](queries.md) |
| `enqueue_*_population_update: ORDINAL_MISMATCH: base ordinal does not match` | `base` is not the population's current head | pass the last published ordinal as base (gaps in target are fine) | [time and pairs](time-and-pairs.md) |
| `fetch_population_update: NOT_READY` (or another `fetch_*`) | fetched before the operation finished | `with op: op.wait(); op.fetch_...()` | [queries](queries.md) |
| No error; `object_count == 0` or lower than the path list | wrong-case type (`"cube"`), missing paths (dropped silently), `Cube` for prims inside native instances (they are `CubeInstance`) | exact type names, both spellings for instances; assert `object_count == len(paths)` | [queries](queries.md) |
| No error; a floor, decor, or the referenced `/Proto` is hit | an empty filter or a type-only filter selects every supported gprim | restrict by path as well | [batches](batches.md) |

## Prims left out of a publication (exclusions)

The publication succeeded; the head moved to a version without these prims. Symptom is the
reason that `ovsq.publish` or `exclusion_reasons` reports for a prim.

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `UNSUPPORTED_TYPE` for `/A` (an Xform, Cylinder, Cone, light, Scope, Material) | the filter matched a prim of a type that is not published; `PREFIX ["/A"]` matches `/A` itself (and `/AB/...`) | `usd-path IN` exact gprim paths (`ovsq.gprim_paths`), or `PREFIX ["/A/"]` AND `usd-prim-type IN [...]` | [queries](queries.md) |
| `UNSUPPORTED_TYPE` for `/Row/I1` (an `XformInstance`) | a `usd-path PREFIX` matched a native instance's root (a group) | `ovsq.gprims_under(root)`: prefix AND every published type, `CubeInstance` included | [batches](batches.md) |
| `UNSUPPORTED_SHAPE` for a Sphere or Capsule | non-uniform scale or shear on it or a parent (Cubes, meshes, and hulls take any scale or shear exactly) | uniform scale, author a `Mesh`, or accept it (`allow_exclusions=True`) | [queries](queries.md) |
| `MALFORMED_GEOMETRY` | bad topology or points, a zero or nonfinite scale, a `convexHull` mesh without four non-coplanar points or that PhysX cooks no hull for, every face in `holeIndices` | fix the prim | [queries](queries.md) |
| `NO_SPATIAL_SOURCE` for a Cube, Sphere, Capsule, or Mesh | ovstage keeps no world bounds (`_worldExtent`) for the prim; a stale authored `extent` is never the cause (other types report `UNSUPPORTED_TYPE` instead) | check the prim's bounds in ovstage; the authored `extent` is not read | [queries](queries.md) |
| `RESHAPE_PENDING` (device-prepared versions only) | the prim's proportions changed; the host rebuilds it | it returns at the next publication; publish shapes that change every frame on the host | [time and pairs](time-and-pairs.md) |
| Exclusions you expected as errors (holes, `leftHanded`) | not exclusions: hole faces are skipped and winding follows `orientation` | nothing to fix | [queries](queries.md) |
| Prims outside a scope are missing from `excluded_objects` | not exclusions: a scope deselects a prim as the filter does | nothing to fix; outside a population's scope they are not in `object_count` either, and a view's scope never changes the publication's counts | [subsets](subsets.md) |

## Queries

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `raycast_closest_sync` / `sweep_*_sync` / `overlap_any_sync: INVALID_ARGUMENT: invalid argument` | direction not unit to about 1e-5, or quaternion not unit | `ovsq.unit` in double; segment = unit direction + `max_distance = length` | [queries](queries.md) |
| `ValueError: ray interval must be finite, ordered, and nonnegative` (or `... must contain 3 finite values`) | `min_distance > max_distance`, negative or NaN values | checked in Python before C | [queries](queries.md) |
| `TypeError: query bits must be an int; got uint64` (or `participation bits`) / `ValueError: ... must fit 128 bits` | scalar bits are a Python int in [0, 2**128) | `int(x)`; build masks with Python ints | [queries](queries.md) |
| `overlap_any_sync` / `sweep_*_sync: NOT_SUPPORTED` | `EXACT_SINGLE` is raycast-only | `EXACT` (the default) | [queries](queries.md) |
| `raycast_closest_sync: NOT_READY` | queried before the first publication, or a view created after it | publish first; create views before the publication | [queries](queries.md) |
| `ORDINAL_NOT_RETAINED` | `retained_exact(n)` beyond `PopulationConfig(retained_ordinal_capacity=...)` (default 2 prior heads) | raise the capacity before creating the population | [time and pairs](time-and-pairs.md) |
| No error; only one overlap returned | `overlap_any` returns one arbitrary hit | `target.overlap_multiple(query, capacity)`; check `.truncated` | [queries](queries.md) |
| No error; a yawed box behaves as unrotated | quaternion passed as wxyz (`(1,0,0,0)` is accepted, a 180-degree turn) | `(x, y, z, w)`; `ovsq.yaw_xyzw(deg)` | [queries](queries.md) |
| Sweep distance or center wrong | `position` is the contact on the object; `distance` is the center's travel | `hit.distance`; the center at contact is `hit.query_position` | [queries](queries.md) |

## Batches (plans)

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `ValueError: unit_directions must be float32; got ...` / `origins must be float64` / `... has 3 rows; origins has 2` | column dtype or rows: origins f64[N,3], unit_directions **f32**[N,3], distances f64[N] | cast each column; one row count | [batches](batches.md) |
| `ValueError: query_bits must have shape (Q, 2); got (n,)` (or `participation_bits ... (N, 2)`) | bits are 128 wide: two uint64 words per row | column 0 = bits 0-63, column 1 = bits 64-127 | [batches](batches.md) |
| `TypeError: max_distances must support DLPack; got list` (or `float64`) / `takes 1 positional argument` | columns are keyword-only arrays | `np.full(n, d)`, `plan.enqueue(origins=..., ...)` | [batches](batches.md) |
| `enqueue_raycast_plan: INVALID_ARGUMENT` (no detail) | more rows than `maximum_query_count` | size the plan for the largest batch | [batches](batches.md) |
| No error; answers off by bounds (short distances, hits beside spheres), `result.exact is None` | plans default to `BOUNDS_CANDIDATE` (single queries do not) | `precision=QueryPrecision.EXACT` | [batches](batches.md) |
| `exact.distances is None` (or `candidate.distances`) | `result_fields` lacks `BatchResultField.DISTANCE` | request it at plan creation | [batches](batches.md) |
| No error; all rows look like misses | non-unit directions fail per row: `query_status == BatchRowStatus.INVALID_ARGUMENT` | normalize; assert every `query_status == 0` | [batches](batches.md) |
| No error; only the first K rays hit | `hit_capacity_per_slot` < rays x hits per ray; rows have `truncated != 0` | `hit_capacity_per_slot >= maximum_query_count * maximum_hits_per_query` | [batches](batches.md) |
| `enqueue_raycast_plan: RESULT_CAPACITY: result capacity is full` | previous launch's operation still open with `result_slot_count=1` | copy results, close the operation (`with op:`), then enqueue again | [batches](batches.md) |
| Hits assigned to the wrong rays | indexed result columns by ray index | ray `i` owns rows `hit_offsets[i] .. + hit_counts[i]` | [batches](batches.md) |
| `ValueError: SOURCE_FEATURE requires exact query precision` | Python-side check | EXACT plan | [batches](batches.md) |
| `create_raycast_plan: DEVICE_BUDGET: device byte budget is full` | CUDA device chosen without a device budget | `InstanceConfig(device_budgets=(DeviceBudget(Device.cuda(0), 256 << 20),))` (C: `device_budgets` in the instance desc) | [cuda](cuda.md) |

## Pairs and reporters

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `enqueue_pair_reporter_advance: STALE_ORDINAL` | neither side published a newer ordinal since the last advance | republish (`enqueue_update`) at least one side every step, even if nothing moved | [time and pairs](time-and-pairs.md) |
| `BACKWARD_ORDINAL: reporter ordinal moves backward` | an endpoint is older than the reporter's baseline | never go back; recreate the reporter | [time and pairs](time-and-pairs.md) |
| No exception; `outcome == ACTIVE_CAPACITY` (or `DELTA_CAPACITY`) and empty deltas | `active_pair_capacity` / `delta_pair_capacity` too small | capacity >= first x second (deltas twice that); assert `outcome is PairReporterOutcome.ADVANCED` | [time and pairs](time-and-pairs.md) |
| `query_pairs(...).truncated` is True | snapshot `capacity` too small | raise it | [time and pairs](time-and-pairs.md) |
| Pairs that do not touch are reported | pairs use closed-interval world AABBs, not exact shapes | expect bounds semantics, or confirm with an EXACT overlap | [time and pairs](time-and-pairs.md) |

## Explicit shapes

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `ValueError: geometry_types must be int32; got ...` (likewise `definition_values must be uint64`, `triangle_indices must be uint32`, `radii must be float64`) | NumPy defaults are int64/float64 | cast every column; translations and rotations take f32 or f64 | [explicit shapes](explicit-shapes.md) |
| `ValueError: a POSES update takes no geometry_types` / `convex topology columns ... come together` / `source_face_indices and source_face_corner_indices come together` | update kind or column set mismatch | COMPLETE_ROWS carries geometry columns, POSES does not; pass both source-face columns or neither | [explicit shapes](explicit-shapes.md) |
| `enqueue_*_registration` / `enqueue_explicit_population_set: INVALID_ARGUMENT` | value 0, duplicate or already registered values, zero radius, unknown definition, non-unit quaternion | nonzero unique values (offset indices by 1); unit `(x, y, z, w)` | [explicit shapes](explicit-shapes.md) |
| `enqueue_convex_registration: INVALID_ARGUMENT: convex definition N: its hull has more than 255 vertices` (or `254 faces`) | the points' convex hull exceeds the vertex or face limit (the library reduces no hull unless asked), and one such definition fails the whole registration | pass `hull_vertex_limit=64` to register the hull a PhysX simulation collides with, simplify the hull, keep at most 129 points, split it into several convexes, or register a triangle mesh | [explicit shapes](explicit-shapes.md) |
| `enqueue_convex_registration: INVALID_ARGUMENT: convex definition N: no hull with its points as vertices encloses them all` | the points lie too nearly in one plane for a hull of their own (the library does not substitute a box), and one such definition fails the whole registration; the message says how far outside the farthest point lies | pass an `enclosure_tolerance` above that distance (at most 2^-16), give the collider a real thickness (1e-3 of its extent is safe; dense point sets can need 1e-2), keep only its outline corners, or register a box or a triangle mesh | [explicit shapes](explicit-shapes.md) |
| `enqueue_triangle_mesh_registration: INVALID_ARGUMENT: triangle mesh definition N: its triangle T names one source face corner twice` (or `its triangles A and B belong to source face F but carry different materials`) | a `source_face_corner_indices` row repeats a corner, or one polygon's triangles carry different `source_face_material_ids` | give each triangle three distinct corners of its polygon, and every triangle of a polygon one material | [explicit shapes](explicit-shapes.md) |
| `enqueue_explicit_population_set: INVALID_STATE` | `enqueue_set` on a population that already has a head | `enqueue_update(base, target, ...)` | [explicit shapes](explicit-shapes.md) |
| No error; scaled instances wrong | `uniform_scales` column omitted (defaults to 1) | pass it | [explicit shapes](explicit-shapes.md) |
| No error; a ray from behind or inside a mesh misses | triangles are front-face only (CCW seen from the ray) | `double_sided=np.ones(G, np.uint8)`, `both_sides=True` on the query or plan launch, or fix winding | [explicit shapes](explicit-shapes.md) |

## Teardown

| Symptom | Cause | Fix | Refer to |
| --- | --- | --- | --- |
| `BufferError: 1 exported result column(s) are still alive` on `op.close()`, `with op`, or instance exit | an `np.from_dlpack` view of a result column is still referenced | `fetch_*().to_numpy()` copies; or `del` the view first | [batches](batches.md) |
| `BufferError: the operation that owned this result column was destroyed` | column read after its operation closed | `to_numpy()` inside the `with op:` block | [batches](batches.md) |
| `destroy_population: BUSY` / `destroy_pair_reporter: BUSY` on a manual `close()` | a view or reporter still uses the population; an advance is still open | close dependents first, or let the instance's `with` close everything | [time and pairs](time-and-pairs.md) |
| Crash at interpreter exit after correct output | a population that outlives its ovstage stage can crash at exit (not every run) | nest `with ovsq.Scene(...)` outside the library and instance | |
