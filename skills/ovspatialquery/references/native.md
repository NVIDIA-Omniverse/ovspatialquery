# Native C and C++ consumers

[`assets/native/CMakeLists.txt`](../assets/native/CMakeLists.txt) builds
[`raycast.c`](../assets/native/raycast.c) (C11: USD file, `usd-path IN`, exact raycasts, id
to path) and [`batch_raycast.cpp`](../assets/native/batch_raycast.cpp) (C++17: embedded
USDA, `PREFIX` + type filter, reusable CPU raycast plan, timing); CTest runs both.

```sh
cmake -S assets/native -B build -G Ninja -DOVSQ_SDK=<sdk> -DOVSTAGE_DIR=<ovstage> && cmake --build build && ctest --test-dir build --output-on-failure
```

`OVSQ_SDK` is the SDK root; `OVSTAGE_DIR` the ovstage package root
(`<venv>/lib/python3.X/site-packages/ovstage`). Without `-D` both come from the same-named
environment variables, read on the first configure only (cached).

## CMake

Both roots hold `lib/cmake/<name>/<name>Config.cmake`; put them on `CMAKE_PREFIX_PATH`:

```cmake
list(APPEND CMAKE_PREFIX_PATH "${OVSQ_SDK}" "${OVSTAGE_DIR}")
find_package(ovspatialquery CONFIG REQUIRED COMPONENTS ovstage)
find_package(ovstage CONFIG REQUIRED)     # the ovspatialquery package never finds it for you
target_link_libraries(app PRIVATE ovspatialquery::ovspatialquery ovstage::ovstage m)
```

- `ovspatialquery::ovspatialquery` brings `<root>/include` (with `dlpack/dlpack.h`);
  `COMPONENTS ovstage` fails at configure on an SDK without `ovspatialquery_ovstage.h`.
  Both imported targets put their library dirs in the build RPATH.
- `ovstage::ovstage` is the shared loader `bin/libovstage-dynamic.so`, which loads
  `bin/libovstage.so` beside itself. Link it whenever you include `ovspatialquery_ovstage.h`
  (it includes ovstage headers). ovstage 0.3 and 0.4 runtimes, any patch, take the
  same C calls; another version fails population creation with NOT_SUPPORTED.
- Includes: `ovspatialquery/ovspatialquery.h`, `ovspatialquery/ovspatialquery_ovstage.h`,
  `ovstage/ovstage.h`, `ovx/path_dictionary/path_dictionary.h`. C needs no C++ compiler.

## C call sequence (ovstage population)

| Step | Call |
| --- | --- |
| Stage | `ovstage_create_instance(&(ovstage_instance_desc_t){.name = "x"}, &stage)` returns `ovstage_api_status_t` (`OVSTAGE_OK`). No `ovstage_initialize` with the shared loader. |
| Open | `ovstage_population_open_usd_from_file(stage, (ovx_string_t){path, len}, 1, 0.0, OVSTAGE_POPULATION_DOMAIN_ALL)` or `..._from_string(stage, usda, ...)`; returns `{status, op_index}`. Time is seconds. ovstage 0.4 has no `ALL`: call `..._with_desc` with `OVSTAGE_POPULATION_DOMAIN_RENDERING` plus a selector of every prim's `physics`, `physxConvexHullCollision`, and `material:binding` properties, as `assets/native/raycast.c` does under `#if OVSTAGE_VERSION_MINOR >= 4`; without `physxConvexHullCollision` every `convexHull` mesh uses the vertex limit 64. |
| Wait open | `ovstage_population_wait_op(stage, op_index, OVSTAGE_TIMEOUT_INFINITE, &w)`; require `w.error_op_id_count == 0`. Not `ovstage_wait_op`. |
| Seal 1 | `ovstage_advance_write_floor(stage, &(ovstage_write_floor_desc_t){.ordinal = 1, .scope = OVSTAGE_SCOPE_ALL})` |
| Hierarchy | `ovstage_compute_hierarchy(stage, OVSTAGE_HIERARCHY_COMPUTATION_MODEL_DEFAULT_CPU, 1, 2)` |
| Seal 2 | `ovstage_advance_write_floor(...ordinal = 2...)`. Each stage op: `ovstage_wait_op(stage, op.op_index, OVSTAGE_TIMEOUT_INFINITE, &w)`, check `w.error_op_id_count`, `ovstage_release_op(stage, op.op_index)`. |
| Instance | `ovspatialquery_create_instance(&desc, &instance)`; `operation_capacity`, `max_cpu_concurrency`, `host_byte_budget` must all be nonzero. |
| Filter | `ovstage_predicate_t p = {0}; p.attribute.string = (ovx_string_t){"usd-path", 8}; p.op = OVSTAGE_FILTER_OP_IN; p.values = paths; p.value_count = n;` then `(ovstage_filter_t){&p, 1}` |
| Population | `ovspatialquery_ovstage_population_desc_t d = {0}; d.stage = stage; d.filter = f;` then `ovspatialquery_create_ovstage_population(instance, &d, &population)` |
| Scope (optional) | `ovx_primpath_t drop; dict->vtable->create_paths_from_strings(dict->context, &(ovx_string_t){"/World/Props/Scaffolding", 24}, 1, &drop);` then `d.scope.exclude_paths = &drop; d.scope.exclude_count = 1;` before creating (`include_paths`/`include_count` alike; `dict` as in [Object id to path](#object-id-to-path)). Each pointer is null exactly when its count is 0 and each entry a nonzero path of the stage's dictionary, else INVALID_ARGUMENT. `ovspatialquery_ovstage_view_desc_t` takes the same `scope` beside its `filter`. Rule and cost: [subsets](subsets.md). |
| Publish | `op = ovspatialquery_enqueue_ovstage_population_set(instance, population, 2)` (the last sealed ordinal), `ovspatialquery_wait_op(instance, op.op_id, OVSPATIALQUERY_TIMEOUT_INFINITE, &w)` with `w.state == OVSPATIALQUERY_OP_COMPLETED`, `ovspatialquery_fetch_population_update(instance, op.op_id, &receipt)` (`receipt.object_count`; `receipt.excluded_object_count` prims were left out, listed in `receipt.excluded_objects` / `exclusion_reasons`, host `uint64`/`int32` DLTensors valid until destroy), `ovspatialquery_destroy_op(instance, &op.op_id)`. |
| Raycast | `ovspatialquery_raycast_closest_sync(instance, &desc, &result)` (also `_any_sync`). |
| Teardown | `ovspatialquery_destroy_batch_plan`, `ovspatialquery_destroy_population`, `ovspatialquery_destroy_instance`, then `ovstage_destroy_instance(stage)` last: the stage must outlive the population. |

Raycast descriptor (`ovspatialquery_raycast_desc_t`, zero it first):

```c
desc.common.target.target.kind              = OVSPATIALQUERY_TARGET_POPULATION;
desc.common.target.target.handle.population = population;
desc.common.target.ordinal.kind             = OVSPATIALQUERY_ORDINAL_HEAD;
desc.common.precision                       = OVSPATIALQUERY_QUERY_EXACT; /* 0 = BOUNDS_CANDIDATE */
desc.common.query_bits = (ovspatialquery_participation_bits_t){{lo, hi}}; /* bits 0-63, 64-127; 0 = all */
desc.ray.origin[i]         = /* double[3] */;
desc.ray.unit_direction[i] = (float)(d[i] / length);  /* float[3], normalize in double */
desc.ray.min_distance = 0.0; desc.ray.max_distance = range;
```

If `result.has_hit`: EXACT fills `result.hit.data.exact.{distance, position, normal,
identity.object_id.value}`, BOUNDS_CANDIDATE `.bounds_candidate.{distance, bounds_position,
bounds_normal}`; `result.hit.truth` says which. A `PxFilterData` is the same 128 bits, word0
lowest: `memcpy(&bits, &filterData, 16)` on a little-endian host.

## Object id to path

`identity.object_id.value` is an `ovx_primpath_t` of the stage's path dictionary. There
is no C `path_to_string`: decompose into tokens, then strings, and join with `/`.

```c
path_dictionary_instance_t* dict = ovstage_get_path_dictionary(stage);
ovx_primpath_t path = (ovx_primpath_t){id}; ovx_token_t buf[64]; ovx_token_t* toks; size_t n, done;
dict->vtable->get_tokens_from_paths(dict->context, &path, 1, buf, 64, &toks, &n, &done);
ovx_string_t names[64];
dict->vtable->get_strings_from_tokens(dict->context, toks, n, names);  /* "/" + names[i] ... */
```

Check `.status == OVX_API_SUCCESS`, `done == 1`. Strings are dictionary-owned; zero tokens is `/`.

## Batch raycast plan (C/C++)

- `ovspatialquery_create_raycast_plan(instance, &desc, &plan)` with
  `ovspatialquery_raycast_plan_desc_t`: `target.kind/handle.population`, `hit_mode =
  OVSPATIALQUERY_HIT_MODE_CLOSEST`, `precision = OVSPATIALQUERY_QUERY_EXACT`, `device = {kDLCPU, 0}`,
  `maximum_query_count = N`, `maximum_hits_per_query = 1`, `hit_capacity_per_slot = N`,
  `result_slot_count = 1`, `result_fields = OVSPATIALQUERY_BATCH_RESULT_DISTANCE | ..._POSITION`.
- Inputs are `ovspatialquery_input_array_t{DLTensor tensor, ownership}` with
  `ownership = OVSPATIALQUERY_INPUT_COPIED_ON_SUBMISSION` (buffer reusable after enqueue),
  `strides = NULL` (compact): `origins` float64[N,3], `unit_directions` **float32**[N,3],
  `min_distances` float64[N] (optional), `max_distances` float64[N], `query_bits`
  uint64[N,2] (optional; ndim 2, a 1-D column is INVALID_ARGUMENT).
- Launch: `ovspatialquery_raycast_batch_t b = {}` (`ordinal.kind = ..._ORDINAL_HEAD`,
  `query_count`, `origins = &o`, ...), `ovspatialquery_enqueue_raycast_plan(instance, plan, &b,
  &reservation)`, `wait_op`, `ovspatialquery_fetch_raycast_batch(...)`, read, `destroy_op`.
- Read `result.columns.hit_ranges.{query_status (uint32), stored_hit_count (uint64[1]),
  hit_offsets, hit_counts (uint64), truncated (uint8)}`, `result.columns.object_ids`
  (uint64), and `result.columns.contact.exact.{distances, positions, normals}` for an EXACT
  plan (`contact.candidate.{distances, bounds_positions, bounds_normals}` otherwise).
  Ray `i` hit rows start at `hit_offsets[i]`, count `hit_counts[i]`; index columns by the
  offset, never by `i`. Data pointer is `(uint8_t*)t.data + t.byte_offset`.

## Errors

- `.status`: 0 SUCCESS, 3 INVALID_ARGUMENT, 6 NOT_SUPPORTED, 7 NOT_READY, 8 BUSY,
  9 STALE_ORDINAL, 14 RESULT_CAPACITY, 19 INTERNAL. Text: `ovspatialquery_get_last_error()`
  (this thread), or `ovspatialquery_get_last_op_error(instance, op_id)` before `destroy_op`;
  `{ptr, length}`, not NUL-terminated (`%.*s`). ovstage logs nothing: use
  `ovstage_population_get_last_op_error(op_index)` / `ovstage_get_last_op_error(stage, op_index)`.

## Pitfalls (symptom -> fix)

- `invalid operands to binary expression`, `incompatible type 'unsigned long'` on a handle or
  `op_id` -> handles and operation IDs are one-member structs `{uint64_t raw_id}`, not
  integers. In C, test `ovspatialquery_op_id_is_valid(op)` and compare with
  `ovspatialquery_view_handle_equal(a, b)` (one pair per handle kind); in C++, `op !=
  OVSPATIALQUERY_INVALID_OP` and `a == b` work, and handles key `std::unordered_map`. Never
  assign an integer to a handle; print `raw_id`.
- `undefined reference to symbol 'sqrt@@GLIBC_2.2.5'`, `DSO missing from command line` -> link `m`.
- Run time: `Failed to initialize ovstage loader: failed to load './build/libovstage.so'`
  -> you linked `ovstage::ovstage_static`. Use `ovstage::ovstage`, or call
  `ovstage_initialize(&(ovstage_config_t){&entry, 1})` first with `entry =
  ovstage_config_entry_binary_package_root_path((ovx_string_t){"<OVSTAGE_DIR>/bin", len})`.
- `Could not find a package configuration file provided by "ovspatialquery"` (or
  `"ovstage"`) -> `OVSQ_SDK` / `OVSTAGE_DIR` is not the root holding `lib/cmake/<name>`.
- `Target "app" links to target "ovstage::ovstage" but the target was not found` ->
  add `find_package(ovstage CONFIG REQUIRED)`; the ovspatialquery package does not import it.
- `error while loading shared libraries: libovspatialquery.so` (binary copied elsewhere)
  -> `INSTALL_RPATH` `${OVSQ_SDK}/lib;${OVSTAGE_DIR}/bin`, or `LD_LIBRARY_PATH`.
- Population wait: `failed to open USD layer: scene.usda` after open returned `OVSTAGE_OK`
  -> relative paths resolve against the cwd (CTest: the build dir); pass an absolute path.
- Publish succeeds with `excluded_object_count > 0` -> content is never an error; read
  `excluded_objects` (`uint64`, `data` + `byte_offset`; paths through `ovstage_get_path_dictionary`,
  as `pathString` in `batch_raycast.cpp`) and `exclusion_reasons`. `..._EXCLUSION_UNSUPPORTED_TYPE`
  -> the filter matched a non-gprim (`usd-path PREFIX` alone matches the Xform root): use
  `usd-path IN` gprim paths or AND `usd-prim-type IN ["Cube","Sphere","Capsule","Mesh"]`.
  `UNSUPPORTED_SHAPE` -> a Sphere or Capsule under non-uniform scale. Meshes skip
  `holeIndices` faces, honor `leftHanded`, and take any world matrix exactly.
- Publish: STALE_ORDINAL `target ordinal does not advance` -> publish the ordinal last
  passed to `ovstage_advance_write_floor` (2 after the sequence above).
- `destroy_instance`/`destroy_population` BUSY `resource is busy` -> an op id is still
  live; `destroy_op` after every fetch, and check every destroy status.
- Sync raycast INVALID_ARGUMENT, or batch rows with `query_status ==
  OVSPATIALQUERY_BATCH_ROW_INVALID_ARGUMENT` -> direction not unit length; normalize in double.
- Hits at AABB distances (rotated box, mesh, sphere off-center) -> precision left at 0
  (BOUNDS_CANDIDATE), in sync descriptors too (only Python defaults single queries to EXACT).
- Fewer batch hits than expected, all rows SUCCESS, `truncated[i] != 0` ->
  `hit_capacity_per_slot` smaller than the query count.
- Result column `shape == NULL` (crash) -> that field was not in the plan's `result_fields`.
- Instanced scene publishes 0 objects, no error -> prims inside native instances are
  `CubeInstance` (and so on): list `"CubeInstance"` beside `"Cube"` in the `usd-prim-type`
  predicate. Open with `OVSTAGE_POPULATION_DOMAIN_ALL` (0.4: RENDERING plus the physics
  selector above); never edit the instancing.
- Second launch enqueue returns status 14 RESULT_CAPACITY (empty error text) -> destroy
  the previous launch's op first (`result_slot_count = 1`), or raise `result_slot_count`.
