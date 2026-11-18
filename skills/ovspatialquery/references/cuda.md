# CUDA: plans on the GPU and a GPU producer loop

Start from [`assets/native/gpu_producer.cpp`](../assets/native/gpu_producer.cpp): a complete
C++ loop that publishes moving boxes and runs an overlap batch every step without the host
waiting on the GPU. It loads the CUDA driver at run time, so it needs no CUDA toolkit.

## Give the instance a device budget

Nothing runs on a GPU until the instance has a byte budget for that device; a CUDA plan on
a default instance fails with `DEVICE_BUDGET`. Check the device before choosing the CUDA path:

```python
GPU = Device.cuda(0)
config = InstanceConfig(device_budgets=(DeviceBudget(GPU, 256 << 20),))
with Library.open() as library, library.instance(config) as instance:
    use_gpu = instance.supports(Capability.PREPARE_ON_DEVICE, GPU) and instance.supports(
        Capability.OVERLAP_PLAN, GPU)
```

C: `ovspatialquery_device_budget_t budget{{kDLCUDA, 0}, 256 << 20};` in
`ovspatialquery_instance_desc_t{..., &budget, 1}`, then
`ovspatialquery_query_capability(instance, {kDLCUDA, 0}, OVSPATIALQUERY_CAPABILITY_...)`.
CUDA work never falls back to the CPU.

**Make the device's primary context current on the thread that enqueues** CUDA work: C
`cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`; Warp `with wp.ScopedDevice("cuda:0"):`.
Otherwise a CUDA publication returns `NOT_SUPPORTED: operation is not supported`.

## An honest GPU producer loop

A GPU engine writes poses into device memory every step and queries them. Measuring that
with host-resident inputs, a wait per step, and a copy back per step measures the host.

1. **Inputs live on the device, uploaded once.** Pass CUDA DLPack columns (C: `DLTensor`
   with `device = {kDLCUDA, 0}`; Python: any CUDA DLPack producer, Warp, CuPy, PyTorch) as
   borrowed inputs: no copy. Reuse the same query columns every launch. Stand in for a GPU
   producer by uploading every step's poses once and publishing step `s` from its slice.
2. **Publish prepared on the device.** Create the population with
   `pending_publication_capacity` of 3-4 (C `ovspatialquery_population_desc_t{retained,
   pending}`, Python `PopulationConfig(pending_publication_capacity=4)`). The first
   complete set uses `OVSPATIALQUERY_PREPARE_ON_DEVICE` (`Preparation.ON_DEVICE`) too and
   must complete before any update (the one wait). Each step then enqueues a `POSES` update
   from `s - 1` to `s` with `dynamic_execution = {device, {1, stream, 0}, PREPARE_ON_DEVICE}`
   (Python `preparation=Preparation.ON_DEVICE, cuda_stream=stream`). The base is the newest
   *admitted* ordinal, so never wait for it.
3. **Launch on the pending publication.** Select it with
   `OVSPATIALQUERY_ORDINAL_ADMITTED_EXACT, s` (`OrdinalSelection.admitted_exact(s)`) and pass
   your stream (`cuda_sync = {1, stream, 0}`, `cuda_stream=stream`): the launch is ordered
   after the publication on the device. `HEAD` without a wait reads an older version.
   Create the plan with `result_slot_count` 2-4 so launches stay in flight.
4. **Consume on the device.** The reservation's result columns exist at enqueue. Queue the
   consumer (your kernel, or `cuMemcpyDtoDAsync` into a history buffer) on the same stream
   right after the launch; stream order needs no wait on `result_ready`. Another stream
   waits on `common.result_ready` with `cuStreamWaitEvent`. Copy to the host once, at the end.
5. **Retire by polling.** Record an event after the consumer. Each step, poll the oldest
   operations with `ovspatialquery_query_op_status` (`op.status()`): destroy a finished
   publication with `destroy_op` (`close()`), and a finished launch with
   `ovspatialquery_destroy_op_after_cuda(op, consumer_event)` (`close_after_cuda(event)`),
   which takes ownership of the event. A refused enqueue is back-pressure, not an error:
   `BUSY` (pending window full) or `RESULT_CAPACITY` (every slot busy) means retire and retry.
6. **Time it as a GPU benchmark.** Record CUDA events on the stream before the first and
   after the last step for device time, and divide the steps by the wall time of the whole
   loop, draining included, for throughput. A host clock around `wait_op` measures waits.

## Python sketch (Warp)

```python
with wp.ScopedDevice("cuda:0"):  # context current; arrays below created on a wp.Stream
    population.enqueue_set(0, object_values=ids, geometry_types=types, geometry_values=values,
        translations=poses[0], rotations_xyzw=rotations,
        preparation=Preparation.ON_DEVICE, cuda_stream=stream).finish()
    for s in range(1, steps + 1):
        publication = submit(lambda: population.enqueue_update(s - 1, s,
            kind=ExplicitUpdateKind.POSES, object_values=ids, translations=poses[s],
            rotations_xyzw=rotations, preparation=Preparation.ON_DEVICE, cuda_stream=stream))
        launch, reservation = submit(lambda: plan.enqueue(translations=centers,
            half_extents=half, ordinal=OrdinalSelection.admitted_exact(s), cuda_stream=stream))
        with wp.ScopedStream(stream):  # consumer on the GPU, behind the launch
            wp.copy(history[s - 1], wp.from_dlpack(reservation.hit_ranges.hit_counts))
        in_flight.append([publication, launch, stream.record_event()])
        retire()  # poll; publication.close(); launch.close_after_cuda(event.cuda_event)
```

`submit` retries on `ApiError` whose `.status` is `ApiStatus.BUSY` or `RESULT_CAPACITY`
after calling `retire()`. After `close_after_cuda(event.cuda_event)` set
`event.owner = False`: the library destroys it. Drop every `from_dlpack` view first.

## Results the host reads

When the host reads every launch's hits (a tool, a sensor processed on the CPU), add
`OVSPATIALQUERY_BATCH_RESULT_HOST_MAPPED` (`BatchResultField.HOST_MAPPED`) to the CUDA plan's
`result_fields`. The columns are then pinned host memory the GPU writes (DLPack `kDLCUDAHost`):

- Read them in place after a wait, status poll, or fetch reports a terminal state other than
  `DEVICE_LOST` (which promises nothing about them); no stream sync, no copy. Python: `np.from_dlpack(column)`, `wp.from_dlpack(column)` (a CPU array), or
  `result.to_numpy()`; PyTorch and CuPy refuse `kDLCUDAHost`.
- Retire with `destroy_op` (`close()`, or the operation's `with` block): no consumer event,
  unless you also queued GPU work on the columns.
- It costs pinned host bytes, charged to `host_byte_budget` at plan creation (`HOST_BUDGET`),
  and one copy pass per launch: leave it off plans whose results stay on the GPU.

Without the bit, wait, read H from `stored_hit_count`, and copy the per-query columns and only
the first H rows of each per-hit column into pinned buffers allocated once, then sync once.

## Precision: what a device-prepared version serves

| Publication | CUDA plan, BOUNDS | CUDA plan, EXACT | CPU plan or single query | Views |
| --- | --- | --- | --- | --- |
| `PREPARE_ON_HOST` (default) | yes | yes | yes | yes |
| `PREPARE_ON_DEVICE` | yes, and CUDA pair reporters | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |
| `PREPARE_ON_DEVICE_HOST_READABLE` | yes | yes | yes, first read downloads | `NOT_SUPPORTED` |

Plans default to `BOUNDS_CANDIDATE`: AABB hits. For axis-aligned boxes against an AABB
overlap that is exact; for anything else filter the candidates yourself, or publish
`HOST_READABLE` and accept the download its first exact or CPU reader pays.

## Pitfalls

| Symptom | Cause | Fix |
| --- | --- | --- |
| `DEVICE_BUDGET: device byte budget is full` at plan creation | default `InstanceConfig` has no device budget | `device_budgets=(DeviceBudget(Device.cuda(0), 256 << 20),)` |
| `NOT_SUPPORTED: operation is not supported` on a CUDA set or update | no CUDA context current on this thread | `cuCtxSetCurrent(primary)`, `wp.ScopedDevice("cuda:0")` |
| `NOT_SUPPORTED` on an `ON_DEVICE` update | the head was published `ON_HOST`; a version never moves | publish the first set `ON_DEVICE` too |
| `NOT_SUPPORTED` launching an EXACT CUDA plan, or from a CPU plan or query | the version is plain `ON_DEVICE` | BOUNDS plan, or `PREPARE_ON_DEVICE_HOST_READABLE` |
| `ORDINAL_MISMATCH` on an update | base is not the newest admitted ordinal | base `s - 1` when you published `s - 1` |
| `BUSY` on an update | more pending publications than `pending_publication_capacity` | raise it; retire finished ones and retry |
| `RESULT_CAPACITY` on a launch | every result slot in flight | `result_slot_count` 2-4; retire and retry |
| `INVALID_ARGUMENT` on a POSES update | geometry columns given, or removals not empty | ids, translations, rotations only |
| `destroy_op` refuses a CUDA launch; Python `close()` raises, or `BufferError` | CUDA launches retire after their consumers | `destroy_op_after_cuda` / `close_after_cuda`, after dropping `from_dlpack` views; a `HOST_MAPPED` plan's launch takes `destroy_op` |
| `torch.from_dlpack` / `cupy.from_dlpack`: unsupported device type 3 | a `HOST_MAPPED` column is pinned host memory | `np.from_dlpack`, `wp.from_dlpack`, or `result.to_numpy()` |
| Counts from the previous step, no error | `HEAD` or `RETAINED_EXACT` without waiting | `ADMITTED_EXACT, s` |
| Inputs read before the upload landed | pageable `cuMemcpyHtoD` racing a non-blocking stream | upload on your stream, or synchronize once before the loop |
