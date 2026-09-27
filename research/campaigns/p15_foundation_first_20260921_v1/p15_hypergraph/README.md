# P15 fixed-rank composition candidate

This packet addresses the higher-rank macro-template composition requested in
`P15_NEXT_WORK.md` item 5. Read `PROOF.md` first. It proves an author-side
implication for disjoint scalar-sandwich blocks coupled by a complete-transversal
macro hypergraph of any fixed finite rank. It does not discover such a block
decomposition for arbitrary families or solve an original prize problem.

The general palette is `ceil(408*kappa) * H_r(2)`. The rank-three refinement
uses `ceil(408*kappa) * 737280` colors, with a singleton-free low-failure
hazard coefficient `8369/9216`. The preceding graph-only source retains its
better palette. The proof keeps the original product law and explicitly prices
occupancy covers before lifting them. It includes original singletons, empty
blocks, empty generators, empty families, and both global and intermediate
zero-probability events.

## Foundations before downstream use

`SOURCES.json` records nine exact selected source bodies, their Drive IDs,
current admission metadata and machine-readable foundation edges. The bodies
are copied in `sources/`. The principal dependencies are P14-A's local
half-budget, the phase-qualified P11-D finite-rank half-budget, and the
phase-qualified P11-B/P11-A rank-three arguments. The zero-Q addendum is an
explicit dependency. P10-A is needed only for the optional disjoint-hierarchy
reuse stated in the proof.

`review/FOUNDATION_REVIEW.md` records a nonauthor mathematical review of the
proof and these arguments. It found no substantive gap and requested the
included intermediate zero-Q clarification. This review did not review the
finite companion and receives **zero organizational independence credit**.
The root's live Drive comparison receipt is separately copied in `review/`;
its identities authenticate source bytes and do not establish mathematics.

A missing, incorrect or incomplete foundation blocks downstream acceptance.
Finite tests do not replace that decision. This packet changes no original
prize result, q0 claim, canonical scientific status or external-review grade.

## Exact companion

Python 3.11 standard library suffices; there is nothing to install. From this
directory, point `P15_REPO` to the authorized research checkout:

```sh
P15_REPO=/absolute/path/to/research-main python3 -B -m unittest -v test_finite.py
P15_REPO=/absolute/path/to/research-main python3 -B -O -m unittest -v test_finite.py
python3 -B check.py --repo /absolute/path/to/research-main --output /fresh/output/normal.json
python3 -B -O check.py --repo /absolute/path/to/research-main --output /fresh/output/optimized.json
```

When the unpacked companion lives inside that checkout, tests can instead
discover its nearest ancestor containing both the Drive inventory and exclusion
metadata. There is no fallback to another checkout or a personal absolute path.

Use distinct fresh output paths outside the research checkout. The replay
guards the pinned source manifest, all nine actual source copies and mirror
bodies, exact inventory rows, and current exclusion metadata before and after
the run. Source or admission changes cause refusal pending review. No archived
program is imported or executed. The guard checks selected source identity;
it does not claim to audit all Drive content or all mathematical projects.

The 13 exact-rational scenarios check finite setwise composition, occupancy
price identities, dependence directions, singleton restrictions, recurrence
arithmetic and endpoint regressions. Their deliberately small palettes are
supplied conditional interfaces, **not** instances of the analytic half-budget
existence construction. They do not establish a theorem over every finite
ground set, implement the expensive P11-D proof algorithm, or turn its enormous
palette into an efficient solver. The 22 tests include tamper and malformed
interface rejection, and run under normal Python and `-O`.

`VALIDATION.md` records the observed results. `MANIFEST.json` binds the artifact
files. Publishing this packet for review is separate from accepting its theorem
or using it to advance downstream scientific claims.
