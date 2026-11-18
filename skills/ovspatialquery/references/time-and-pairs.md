# Time samples, versions, and object pairs

Runnable programs: [`assets/python/time_versions.py`](../assets/python/time_versions.py)
(animated USD, head compared to older versions) and
[`assets/python/pairs.py`](../assets/python/pairs.py) (pair snapshot, entered/exited).

## Calls

```python
# Ovstage (seconds, not time codes; ovsq.Scene does the division). Pass the time
# positionally; Ovstage 0.3 names it time.
ovstage.population.update_from_usd_time(stage, n, tc / timeCodesPerSecond)
stage.advance_write_floor(ordinal=n).wait()
ovsq.Scene(usda_or_path, time_code=0)   # opens at ordinal 1, hierarchy into 2; scene.ordinal == 2
scene.set_time(tc) -> int               # writes + seals scene.ordinal + 1, returns it

# Publication
adapter.create_population(ovstage.Filter([...]), PopulationConfig(retained_ordinal_capacity=2))
population.enqueue_set(ordinal) / population.enqueue_update(base, ordinal) -> Operation
ovsq.publish(population, ordinal, base=None, *, allow_exclusions=False) -> PublicationResult(
    published_ordinal, object_count, excluded_object_count, excluded_objects, exclusion_reasons)
population.info() -> PopulationInfo(ready, head_ordinal, object_count, retained_ordinal_count, update_pending)

# Choosing a version (every query and query_pairs take one; default head)
OrdinalSelection.head()
OrdinalSelection.retained_exact(source_ordinal)
RaycastQuery(origin, unit_direction, max_distance, ordinal=OrdinalSelection.retained_exact(2))
result.source_ordinal                                 # version that answered

# Pairs (enums are top level: from ovspatialquery import PairReporterOutcome, ApiStatus)
instance.query_pairs(first_target, second_target, *, capacity, first_ordinal=head, second_ordinal=head,
                    contact_tolerance=0.0) -> PairSnapshotResult(first_source_ordinal, second_source_ordinal, rows, truncated)
instance.create_pair_reporter(PairReporterConfig(first=pop_a.target.reference, second=pop_b.target.reference,
                    active_pair_capacity=N, delta_pair_capacity=M, contact_tolerance=0.0)) -> PairReporter
reporter.advance(first=head, second=head) -> (Operation, PairReporterReservation)
operation.fetch_pair_reporter().to_numpy() -> PairReporterResult(result_slot, outcome,
    first_source_ordinal, second_source_ordinal, accepted_active_pair_count, result_ready,
    entered_pairs, exited_pairs)      # entered/exited: uint64 ndarray [K, 2] of (first_id, second_id)
PairReporterOutcome.ADVANCED | ACTIVE_CAPACITY | DELTA_CAPACITY | FAILED
```

`query_pairs` takes `population.target`; `PairReporterConfig` takes
`population.target.reference`. Rows are `(id from first, id from second)`; convert with
`scene.path(id)`. Rows are not sorted. Snapshot `rows` is a tuple of int pairs; reporter
deltas are NumPy arrays after `to_numpy()` (shape `(0, 2)` when empty).

## Defaults that bite

- `open_usd*` and `update_from_usd_time` take **seconds**. Unauthored
  `timeCodesPerSecond` means 24. Passing a time code as seconds clamps to the last sample:
  no error, wrong positions.
- Opening with the default time (NaN, USD default time) ignores attributes that have only
  `timeSamples`; animated prims sit at their untransformed place. Open at a real time.
- Between samples ovstage interpolates linearly; outside the range it holds the end sample.
- `update_from_usd_time` refreshes world transforms; no extra `compute_hierarchy` per step.
- `retained_ordinal_capacity` counts prior versions *besides* the head. Default 2.
- Pair tests compare world axis-aligned bounds, closed intervals: touching boxes are a pair,
  a sphere pairs with a box its bounds reach even if the sphere surface does not, and a
  rotated box pairs through its enlarged bounds. `contact_tolerance` widens each gap test.
- The first `advance` reports every pair alive now as entered.
- A capacity that is too small is a result, not an exception: snapshot `truncated=True`
  with `capacity` rows; reporter `outcome` `ACTIVE_CAPACITY`/`DELTA_CAPACITY` with empty deltas.

## Pattern: one version per time step

```python
with (
    ovsq.Scene(usda, time_code=times[0]) as scene,  # outermost: the stage outlives populations
    Library.open(require_ovstage=True) as library,
    library.instance() as instance,  # closes populations on exit
):
    selected = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
    population = instance.ovstage(ovstage, scene.stage).create_population(
        selected, PopulationConfig(retained_ordinal_capacity=2)
    )
    published, base = {}, None
    for tc in times:
        if tc != times[0]:
            scene.set_time(tc)
        assert ovsq.publish(population, scene.ordinal, base).object_count == len(paths)
        base = published[tc] = scene.ordinal
    head = population.target.raycast_closest(RaycastQuery(...))  # EXACT by default
    old = population.target.raycast_closest(
        RaycastQuery(..., ordinal=OrdinalSelection.retained_exact(published[times[0]]))
    )
    assert old.source_ordinal == published[times[0]]
```

## Pattern: entered/exited per step

```python
reporter = instance.create_pair_reporter(
    PairReporterConfig(
        first=movers.target.reference,
        second=static.target.reference,
        active_pair_capacity=len(mover_paths) * len(static_paths),
        delta_pair_capacity=len(mover_paths) * len(static_paths),
    )
)
base = scene.ordinal  # both populations already published here
for tc in times:
    if tc != times[0]:
        scene.set_time(tc)
        ovsq.publish(movers, scene.ordinal, base)  # every step, even if nothing moved
        base = scene.ordinal
    operation, _ = reporter.advance()
    with operation:
        operation.wait()
        result = operation.fetch_pair_reporter().to_numpy()  # copy before the block closes it
    assert result.outcome is PairReporterOutcome.ADVANCED
    for first_id, second_id in result.entered_pairs.tolist():
        ...
    for first_id, second_id in result.exited_pairs.tolist():
        ...
```

Only populations that change need republishing; the static one stays at its first ordinal.
Several populations on one stage all publish the same `scene.ordinal`.

## Pitfalls

- Animated prims never move, or all steps after the first are wrong, no error ->
  time passed in time codes; divide by `timeCodesPerSecond` (or use `ovsq.Scene`).
- Animated prims at the origin at the first step -> stage opened at default time; pass
  a time when opening (`ovsq.Scene(..., time_code=...)`).
- `TypeError: ... unexpected keyword argument 'time_code'` from `update_from_usd_time` or
  `open_usd*` -> ovstage 0.3 renamed it `time`; pass the time positionally.
- `STALE_ORDINAL: target ordinal does not advance` on `enqueue_set`/`enqueue_update` ->
  the ordinal is not the newest sealed one; call `advance_write_floor(ordinal=n).wait()`
  after the write and publish exactly `n`.
- `OvstageError: OP_FAILED: ... at or below the effective global write floor` from
  `update_from_usd_time` -> reused an ordinal; use a new, larger one each step.
- `ORDINAL_MISMATCH: base ordinal does not match` -> `enqueue_update` base must be that
  population's last `published_ordinal`.
- `ORDINAL_NOT_RETAINED` on a `retained_exact` query -> the version fell out; raise
  `retained_ordinal_capacity` to at least the number of newer publications since it.
- Retained query reports the head's ordinal -> `ordinal=` not passed to the query;
  check `result.source_ordinal`.
- `STALE_ORDINAL` from `reporter.advance()` -> neither population published a new ordinal
  since the last advance; republish the moving one before every advance.
- `NOT_READY` from `reporter.advance()` -> a population has not been published yet.
- `BufferError: the operation that owned this result column was destroyed` -> read the
  deltas after the operation closed; call `.to_numpy()` inside the `with operation:` block.
- Empty `entered`/`exited` although pairs changed -> `outcome` is a capacity outcome;
  raise `active_pair_capacity` (pairs alive at once) or `delta_pair_capacity` (pairs
  entering or exiting in one step).
- Snapshot missing pairs -> `truncated` is true; raise `capacity`.
- A prim exits and re-enters its pairs, no error -> that version left it out (a Sphere or
  Capsule whose animated scale turned non-uniform); `exclusion_reasons` names it.
- `BUSY` from `population.close()` -> a reporter on it is still open; close the reporter
  first, or let the instance close both.
- Crash at process exit -> a population outlived its ovstage stage; open the scene in an
  outer `with` (or destroy the stage only after the instance closed).
