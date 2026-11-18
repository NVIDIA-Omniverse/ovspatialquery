# Subsets of one population

Runnable example: [`assets/python/queries.py`](../assets/python/queries.py) (a filter view,
participation bits, a scoped population and view). Views and participation bits reuse the
population's acceleration data, so a subset never needs a second population.

## Pattern

Do these after `create_population` and **before** `ovsq.publish` (a view created later is
`NOT_READY` until the next publication):

```python
paths = ["/World/Props/Crate", "/World/Props/Ball"]
crate_and_ball = ovstage.Filter([ovstage.Predicate("usd-path", ovstage.FilterOp.IN, paths)])
view = population.create_view(crate_and_ball)  # query view.target
props = population.create_view(ovstage.Filter([]), include_paths=["/World/Props"],
                               exclude_paths=["/World/Props/Scaffolding"])  # a path scope
population.set_participation_map(default_bits=0b01, bits_by_path={"/World/Props/Panel": 1 << 100})
# a query with query_bits=b matches objects whose (b & bits) != 0
```

| To select by | Use | Changes |
| --- | --- | --- |
| Prim type, or an exact list of paths | the filter (`usd-prim-type IN`, `usd-path IN`) | never |
| Subtree: "everything under X except Y", carve-outs | a scope: `include_paths`, `exclude_paths` | never |
| A category each query or batch row picks | participation bits and `query_bits` | each publication |

- A view is a fixed subset: its filter and scope are set at creation. The instance closes it
  (before its population) on exit.
- `bits_by_path` keys are prim path strings (or path ids). Bits are a Python `int` in
  [0, 2**128), built with Python ints (`np.uint64(w) << 64` is 0; `int()` a NumPy scalar).
  A PhysX `PxFilterData` is `w0 | w1 << 32 | w2 << 64 | w3 << 96`.
- Bits pass an object that shares any bit, so they cannot say "not under X". Use a scope
  for that, and `query_bits` on the scoped view for the category.

## Path scopes

`create_population(filter, config, *, include_paths=(), exclude_paths=())` and
`create_view(filter, *, include_paths=(), exclude_paths=())` take lists of prim path
strings or int path ids, by keyword. A prim is selected when the filter matches it and the
scope contains it:

- Walk from the prim up to `/`: the first path found in either list decides, include in,
  exclude out. A path in both lists excludes; an include below an exclude brings that
  subtree back. No includes means `/`, so excludes alone mean "everything except".
- Paths compare by element: `/World/Env` covers `/World/Env/Lamp`, not `/World/Environment`
  (a `usd-path PREFIX` compares bytes). Write no trailing `/`.
- A view's scope narrows its population's objects and cannot bring back what the
  population left out. Put a subtree no view may see on the population: in a view's
  excludes, a deeper include of that view would win.
- Scoped-out prims are simply not selected, and never in `excluded_objects`. A misspelled
  path matches nothing, silently, so check what each scope holds. A population's scope shows
  in the publication's `object_count`. A view's scope does not: publication counts describe
  the population. Query the view instead, for example
  `view.target.overlap_multiple(OverlapQuery(Aabb(lo, hi)), capacity)` with a box covering
  the scene, and count or list its hits.
- A scope is fixed at creation. For another one, create another view.
- The first publication that scopes a prim may pay once for ovstage to split its path: up
  to about twice an unscoped publication when every prim is new, and often nothing
  measurable. Later publications, and another scoped population or view over the same
  prims, cost about the same as unscoped ones.

C takes the same lists as `ovspatialquery_ovstage_scope_t` ([native](native.md)).

## Pitfalls (symptom -> fix)

- `NOT_READY` from a query on `view.target` -> the view was created after the publication.
  Create it before `ovsq.publish`, or publish a new ordinal.
- Participation has no effect -> the query passes `query_bits=0`, or the map was set after
  the publication.
- A prim you excluded is still hit -> the path is misspelled, or an include nearer the prim
  (below the exclude) re-includes it. Put exclusions every view needs on the population.
- `/World/Environment` is selected (or dropped) along with `/World/Env` -> a `usd-path
  PREFIX` filter; use a scope.
- Scope errors (`TypeError`, `OvxError`, `INVALID_ARGUMENT`) -> [errors](errors.md).
