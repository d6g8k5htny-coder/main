# Finite Canvas model contract — 9 October 2026

Status: contract and external tests prepared; execution **NOT_RUN**. This is
Task 3's pure-model increment, not the Canvas/DOM, browser or deployment task.
Scientific effect is `NONE`; scientific-status authority is boolean `false`.

## Scope, ownership and execution sequence

Architecture coordinator OpenAI/Codex root01a11d37 owns workflow, publication,
hosted execution and integration. [Native scope 6086148751](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6086148751)
was created 2026-10-09T17:43:52Z. Actual author of this contract and the two tests
is OpenAI/Codex root01a0bbb5, delegated for Dylan Roy, source-exposed,
organizational independence 0. This is not Dylan's personal human review.
R17 claim `c2230001-89da-4a2a-af29-610091748001` reserves these three new files;
timestamp correction `c2230002-89da-4a2a-af29-610091746002` retains the original
entry and records observed-at 17:46:13Z; expiry remains 18:30Z.

The six-path scope is this contract, `interface/tests/{cubic,elder}.test.mjs`,
`interface/src/scripts/{cubic,elder}-model.mjs`, and coordinator-owned
`.github/workflows/architecture-canvas-tests.yml`. A distinct later author
receives the two product modules only after these tests freeze. Existing
site, release, package, lockfile, Pages and required-check DAG paths are excluded.

The coordinator's explicit sequencing amendment governs the earlier plan's
unachievable “behavioral RED before any product authoring” requirement:

1. Freeze and source-review contract/tests first. Top-level imports use only
   Node built-ins; candidate dynamic import errors are captured by preflight.
2. Missing/syntax-invalid/import-failing/wrong-export candidates are
   `SETUP_BLOCKED`, behavioral result `NOT_RUN`. Zero discovery or runner failure
   is `DISCOVERY_FAILED`. Neither is genuine behavioral RED.
3. A distinct author may supply a faithful predecessor-behavior port under the
   new named APIs, preserving real functionality and honestly identifying absent
   teaching-coordinate/trace extensions. Preserve the actual first candidate.
4. A hosted assertion mismatch after real candidate import and function execution
   can establish behavioral RED for that assertion. No intentionally broken
   stub or synthetic sensitivity control counts as candidate RED. If the first
   real candidate is complete and passes, record GREEN honestly instead.
5. Fix actual findings; run all applicable final controls once on stable inputs.
   GREEN requires all declared controls passing, no skipped/blocked cases and
   complete runtime/custody/disposal evidence. It does not finish Task 3 or the
   full project. Source review is not execution.

No project, Node, Python or YAML execution/installation occurs on the laptop.
Host the explicit two test files with Node built-ins only, network denied in a
disposable sandbox. No npm/Astro installation, private capture or credentials
enter that sandbox. The coordinator must freshly identify the actual bundled
Node binary/version; prior runtime preparation executed an extracted binary
and does not establish bundled-binary execution. Hosted logs bind repository,
checked commit, source/test/contract/workflow bytes, runtime, run/attempt,
discovery, preflight, assertion results and confirmed disposal. Retain original
failures and exact native artifact bytes. No green claim from inner digests.

## Frozen scientific and engineering sources

All source paths below belong to `d6g8k5htny-coder/main` commit
`032267c259162728229e3ddd14d08fe1c3f9a392`. Native content/blob reads were compared
with exact local Git objects; these are source pins, not claims of current HEAD.

| Path | Git blob | SHA-256 |
| --- | --- | --- |
| `docs/site/explore-models.mjs` | `edb0709b8efde17aed61235452930515ec35884b` | `fdbf01dd7235f0b35e1f7dad30fbd70f07d30a690b83ff9a87b3506390a7159b` |
| `tests/test_explore_models.mjs` | `8004df102215d454fef1e8c49e8f9426d98e6d4c` | `9ff140148fbeb088c664f482a754661819c4f609f8180b2ad34df5f9a696a2d7` |
| `experiments/periodic_h0/EXACT_H0.md` | `026d61fefcc422aca80a12d7b2501a5c2d080ed2` | `a788b46b1ea5640661e2692b84816bb3acfd1bae5a1fd3083dd18d5d8ba19fff` |
| `experiments/periodic_h0/exact_h0.py` | `ffbfe6a3cd5ed39465e804ecfec0d9f0c6855da2` | `48ee16f2316cb1a233594ed10466340aa321d553c2b888198fe86b4bd9052970` |
| `experiments/periodic_h0/test_exact_h0.py` | `d5ab63cad6e3a0e082342c77eb29bb73c375209a` | `c22930ef4fd0007ab8fd78079f4560cdad3043b210f3b074f7b0220d99039610` |

Parent plan `docs/plans/2026-10-08-architecture-interface.md` at PR327 head
`765b6a473b0652858dcd9724bd31c944b3f5ca04`: blob
`aad7b7d2f562be0b05c65d0cabe35b42306f5a1f`, SHA-256
`0b754e6fda9c32e0a67cd865f8f3759590937d8faf81cc1e90f5a8f3fa3a9c97`.
Its Task 3 supplies the API names and teaching intent. The choices below are
explicit new JavaScript/teaching definitions where the original sources do not
define a trace, normalization schema or teaching resource cap.

## Cubic API and new teaching coordinates

`cubicModel({r, k, b})` is synchronous and pure, with no defaults or coercion.
Input is a non-null non-array object with own `r`, `k`, `b` fields. Extra fields
are ignored. Each field must be a primitive finite Number, `r > 0`, `k > 0`.
Malformed record/field types throw `TypeError`; nonfinite values, invalid ranges
and failed representability checks throw `RangeError`. Exact error text is not
an API. Accessor/proxy objects are not a security boundary; hosted isolation is.

The exact output key set is:

```text
{r, k, b, rCubed, pins, gap, heightWindow, normalized}
normalized = {xDivisor, yOrigin, yDivisor, pins, heightWindow}
```

Define Number operations `rCubed = r ** 3`, `gap = k * rCubed`,
`pins = [-r/2, r/2]`, `heightWindow = [b-gap, b]`. Reject unless `rCubed` and
`gap` are finite and positive, pins are finite and distinct, and `b-gap` is
finite and strictly less than `b`. This deliberately rejects underflow,
overflow and subtraction rounded to the upper endpoint. Do not import the old
module's unrelated annulus/remote geometry or its release-query dependency.

The **new teaching normalization** is `X=x/r`, `Y=(y-b)/gap`, retaining all
normalizers: `xDivisor=r`, `yOrigin=b`, `yDivisor=gap`. Its nominal pins are
`[-0.5,0.5]`, nominal height window `[-1,0]`. These fixed coordinates describe
the symbolic affine definition; they do not assert that subtracting rounded
Number endpoints and dividing reproduces them exactly. The open source height
window and this illustrative coordinate range are not persistence intervals.
The cubic model uses floating-point teaching arithmetic, not exact rationals or
a proved continuum sampling law. Nonbinary controls use a stated numerical
tolerance; exact golden values use powers of two.

The input must remain unchanged. Calls return fresh arrays/objects; mutating
one result must not affect another. Arrays are tuples in the declared order.

## Exact finite elder API

`finiteElderTrace({values, n, scale})` is synchronous and pure. The argument is
a non-null non-array object with own `values`, `n`, `scale`; extra fields are
ignored. `n` is a primitive safe integer Number with `3 <= n <= 32`. This upper
limit is a new teaching cap, not the Python theorem's domain limitation.
`values` is an Array of length `n*n`, with an own element at every index and
only primitive BigInt numerators. No sparse/inherited slots, typed arrays,
numbers, booleans, strings or nested arrays. `scale` is required positive
primitive BigInt. Invalid record/types/dense-element conditions throw
`TypeError`; invalid dimension/range/shape or nonpositive scale throw
`RangeError`. A noninteger Number dimension is a range error. No message-text
requirements. No mutation of inputs; returned `values` is a copy.

The exact output keys are `{n, scale, values, events, levels, barcode}`. Integer
sample values, levels, births and deaths remain BigInt. IDs, offsets and counts
are safe integer Numbers. Never convert sample/level/birth/death BigInts to
Number or divide them in the exact engine; safe Number arithmetic for IDs and
indexes (including row/column division) is allowed. A future JSON adapter must separately specify decimal strings; it is
not part of this unit. Approximate drawing coordinates belong outside it.

The graph has row-major ID `x*n+y`, four axis neighbors with periodic seams,
no diagonal edges and no duplicated boundary vertices. Sweep descending value,
then increasing vertex ID. For each newly active vertex, inspect active
neighbors in order negative first coordinate, positive first coordinate,
negative second coordinate, positive second coordinate. Every undirected edge
is inspected once, when its later endpoint is activated. The edge tuple stores
`[newlyActiveVertex, alreadyActiveNeighbor]` in that order.

Mathematical elder priority is larger birth, then smaller vertex ID. Storage
roots, union size and path compression cannot change that priority.

`barcode` has exactly the source's five keys:

```text
{intervals: [[b,d],...], birth_vertices: [id,...],
 essential: [b], essential_vertices: [id], zero_count: Number}
```

Finite records satisfy `b>d` and sort ascending `(b,d,birthVertex)`, with arrays
aligned. Physical superlevel interval is **`(d/scale, b/scale]`**, lifetime
`(b-d)/scale`; the class is dead at the death threshold. There is one separate
essential class `(-infinity,b/scale]`, represented by the smallest maximum ID.
Never remove the longest finite bar. Successful equal-level merges count as
zero pairs, omitted from `intervals`. Redundant edges are not zero pairs.
`intervals.length + zero_count + 1 = n*n`.

## New explanatory trace, exact schema

The original Python source does not expose this stream or use “neutral” as an
event name. Each record below has exactly the shown keys:

```text
{kind:'activate', vertex, level}
{kind:'merge', edge:[u,v], level, survivor, dying, birth, death, zero}
{kind:'neutral', edge:[u,v], level, elder}
```

Activation precedes that vertex's edges. `merge` joins two distinct current
components; `survivor`/`dying` are their mathematical elder IDs before joining,
`birth=values[dying]`, `death=level`, `zero=(birth===death)` is boolean.
`neutral` means the endpoints were already in the same current component;
`elder` is its elder. A merge is never called neutral just because its lifetime
is zero. The exact stream contains `N` activations, `N-1` merges, `N+1` neutral
edges and `3N` records for `N=n*n`. This follows from the connected torus graph's
`2N` unique undirected edges; it is not a fresh execution result.

`levels` is descending, one record after each **complete equal-value block**:

```text
{level, eventEnd, activeCount, componentCount, positiveCount, zeroCount}
```

`eventEnd` is the exclusive stream offset, including every edge processed at
that level. `activeCount` is the current number of active vertices and
`componentCount` the current number of connected components; only
`positiveCount` and `zeroCount` are cumulative merge tallies. Only these full blocks represent the actual
superlevel sets `{v:values[v]>=level}`. Intermediate event playback illustrates
the chosen insertion order and is not a distinct filtration threshold. Do not
store a full grid snapshot in every event; the specified stream is linear-size.

## External controls and completion limits

The test files classify candidate preflight separately and skip behavioral
controls explicitly if preflight fails. These skips block GREEN. Oracle
sensitivity controls use literal good/corrupt barcodes; they validate the
checker, not a mocked model or genuine candidate RED. No product helper is
imported into expected-value calculations.

Cubic controls cover literal outputs, halving radius/eighth gap, both retained
normalizers, nonbinary arithmetic, invalid type/range/representability and
input/result independence. Elder controls cover source literal unequal-peak,
seam, no-diagonal, tie and negative fixtures; `2^96` translation, scaling and
fractional physical scale; strict input rejection; and maximum teaching size.

An independently constructed undirected graph supplies two stronger checks:
(1) complete regional-maximum birth labels plus BFS elder reachability at death
and death+1, and (2) all high/low threshold ranks on all 512 binary 3x3 fields
and deterministic explicitly generated tied fields. A separate partial-edge
BFS replays trace connectivity without union-find; it verifies exact schemas,
event order/coverage, mathematical elder identities, zero-versus-neutral,
barcode joins and full-block state counts. The six-column golden fixture also
has hand-derived component counts `[1,2,3,2,1,1]` at levels `[5,4,3,2,1,0]`.

Finite exact tests, model agreement and source review do not prove the missing
bridge from weighted critical pairs to actual ordinary persistence bars. They
do not equate branch adjacency with elder pairing, contact integrals with elder
events, or global factorial bounds with regional shrinking multiple-witness
collisions. No field-first sampling, all-small-bars converse, once-counted
continuum-bar law, source-node mapping or formal certificate is supplied here.
Canvas, equivalent DOM tables, keyboard/reduced-motion controls, real browser
checks, SSG, privacy admission and exact-artifact deployment remain later work.
