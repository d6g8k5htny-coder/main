# `research/cover/` — the spatial cover ledger and adaptive driver

Lane A5. `docs/OPEN_PROBLEMS.md` records **`D3-LEMMA-RN-UNIF` Piece 1 OPEN and
Piece 2 OPEN, with the annulus Riemann-sum driver *unwritten*** — "Schedule it
explicitly; do not hide it under a T4 push" — and quotes RN5's next exact
action:

> build a complete non-overlapping spatial cover of `0.1 ≤ |y| ≤ 5`, retaining
> boundary-area bounds and every rejected cell; sum `area × corrected cell
> supremum`; verify no cell remains pending; then reassemble the remote budget.
> Treat the near-axis refinement cost explicitly — the present ten boxes are
> **not** a coverage certificate.

This package is that driver, region-generic, with the accept / refine / reject
ledger as its **first-class output**.

---

## What this is NOT

Read this section first; it is the load-bearing one.

* **It does not close Piece 2 of `D3-LEMMA-RN-UNIF`.** It does not close
  Piece 1. Both remain **OPEN** exactly as `docs/OPEN_PROBLEMS.md` records
  them, and the lane receipts (`RNU_T4_PUSH_RECEIPT.json`,
  `RNU_EXECUTE_RECEIPT.json`) still carry `status=PROPOSED` and
  `lemma_closed: false`. Writing the driver the lane records as unwritten
  removes one named engineering gap. It supplies none of the mathematics.
* **It does not reassemble the remote budget.** That is the step *after* the
  sum in RN5's recipe and nothing here performs it.
* **It certifies no cell of the program's actual cover.** The integrands
  shipped here are REFERENCE functions, chosen because every step of their
  enclosure can be certified. None of them is the program's `kappa_far`; none
  is the corrected RN5 envelope; none appears in any claim in this repository.
* **It discharges, reduces, promotes and reclassifies nothing.**
  `OBL-H5-JETMOD`, `OBL-H5-ZBAND` (hi side), `OBL-H5-REMOTE-THRESHOLD` and
  `OBL-D1-PROMOTE` (chart side) stand exactly as recorded. A cover engine is
  not a cover certificate.
* **It composes no tracks.** The 2D upper, 2D lower and 3D lifetime tracks are
  related here in no way whatsoever.
* **It solves no prize problem** and bears on none.
* **A run is a run.** A green receipt is a record of a computation, not
  evidence. Passing tests say the code does what its docstrings say; they are
  not a mathematical review of the enclosure arguments, which are written out
  in the docstrings for a human to check.

The two cover regions instantiated here are **different regions serving
different purposes and are never merged or summed**: the RN5 annulus
`0.1 ≤ |y| ≤ 5` from `docs/OPEN_PROBLEMS.md` A5, and the T4 polar cover
`d ∈ [5, 17]` with theta-halving from `LANE_RN_UNIF.md`.

---

## What it does

| File | What it is |
|---|---|
| `ledger.py` | cells, dispositions, the exact partition invariant, `total()`, the receipt |
| `regions.py` | the region protocol and the two instantiations, plus the Cartesian bracket alternative |
| `driver.py` | the adaptive accept/refine/reject loop and the REFERENCE integrands |

```python
from fractions import Fraction
from research.cover import (DriverConfig, RadialGaussianReference,
                            radial_gaussian_closed_form, rn5_annulus_polar, run)

region = rn5_annulus_polar(split="radius")
ledger = run(region, RadialGaussianReference(),
             DriverConfig(tol=Fraction(1, 10), max_depth=20, prec=40))
total = ledger.total()          # raises while any cell is PENDING
print(ledger.receipt_json())
```

What that run actually produces, as the receipt records it:

| field | value |
|---|---|
| region | `RN5-annulus 0.1<=|y|<=5 (polar)`, `coords = polar(radius, turn)` |
| cells | 2,036 total — 1,020 ACCEPTED, 1,016 REFINED, 0 REJECTED, **0 PENDING** |
| refine depth | 10 |
| max cell width | 10 — **saturated**; see the warning below, this is not the achieved resolution |
| min cell width | `17/80 = 0.2125` — the finest leaf, and the value that actually moves |
| area rejected | **0** — the polar boundary is exact |
| area accounted | `π·(5² − (1/10)²)`, enclosed to ~1e-11 |
| total | `[6.2286534, 6.2784424]`, width `0.0498`, `certified: true` |
| closed form | `2π(e^{−1/200} − e^{−25/2}) = 6.251824374471…`, inside the total |

Both numbers enclose the same integral of a REFERENCE function. Neither bears
on any claim.

> **`max_cell_width` is the COARSEST leaf, not the achieved resolution, and it
> does not track the tolerance.** Two reasons, both real here. (1) Adaptive
> refinement leaves flat parts of the domain coarse *on purpose*, so the widest
> leaf can stop moving while the rest of the cover is refined. (2)
> `PolarRegion.diameter_bound` caps at `2·r_max`, which is correct and is the
> right cap for a full-turn ring — but under `split="radius"` theta is never
> subdivided, so **every leaf is a full-turn ring and every leaf reports the
> cap**. On this region that value is exactly `10`, the annulus's full outer
> diameter, at every depth and every tolerance: runs at `tol = 1, 1/4, 1/16`
> give `(10, 10, 10)` while the total width falls from `0.398` to `0.0335`.
> Reported next to a certified total of width `0.0498`, the `10` says nothing
> about the resolution achieved. The receipt now also carries `min_cell_width`
> and a `cell_width_note` saying this, and the quantities that do move with the
> tolerance are `refine_depth` (6 → 8 → 10 across those three runs) and
> `min_cell_width` (`0.3` → `0.25` → `0.2125`).

`run` returns the **ledger**, never a number. Getting a number means calling
`total()`, and `total()` refuses — by raising `PendingCellsError` — while any
cell is PENDING. There is no flag to bypass it. That ordering is the whole
point: "the present ten boxes are **not** a coverage certificate" is a
description of a partial cover reported as a total, and this is the object that
makes that impossible rather than merely discouraged.

### The exactness invariant is structural, not an area comparison

`sum(area(cell)) == area(domain)` can hold **with a gap and an overlap at the
same time** — the two errors cancel in the sum. `check_exact_partition`
therefore decides the question structurally over `Fraction`, by coordinate
compression: compress the `u` endpoints into strips; inside each strip the
covering cells' `v` intervals must tile the domain's `v` range with the first
`lo` at the bottom, each successive `lo` *equal* to the previous `hi`, and the
last `hi` at the top. A successive `lo` below the previous `hi` is an overlap;
above it is a gap. Both are reported with coordinates. Every comparison is
between exact rationals.

`tests/test_cover.py` builds the equal-area gap-and-overlap cover explicitly
and asserts it fails. That is the sharpest control in this package: it is the
one a driver that checks areas would pass.

The ledger checks a second, independent reading of the same invariant —
every `REFINED` cell's children tile it exactly, and the roots tile the domain,
so cover-wide tiling follows by induction. The two checks corroborate each
other instead of sharing a single point of failure.

### The sum is an enclosure, not just an upper bound

For a cell `C` of area `|C|`, `∫_C f = |C| · mean_C(f)` and the mean lies
between `inf_C f` and `sup_C f`. So for any certified enclosure `rng` of the
**range** of `f` on `C`,

```
∫_C f  ∈  |C| · rng          (interval product)
```

Summing over a cover with disjoint interiors gives a certified two-sided
enclosure of the region integral. RN5's "certified cell supremum" is exactly
`rng.hi`; keeping the lower endpoint costs nothing and makes the result
falsifiable from both sides. Areas come from `research/interval` (`π` for polar
cells, exactly rational for Cartesian ones) and every arithmetic step is
interval arithmetic over exact rational endpoints.

For a cell that only partly meets the region, `|S| ∈ [0, |C|]` gives
`∫_{C∩region} f ∈ Interval(0, |C|) · rng`, which is what a rejected boundary
cell carries as its `residual`. That is how a rejected cell is **bounded**
rather than ignored.

### Rejected cells are retained, with reasons and boundary-area bounds

`reject()` takes the reason and the boundary-area bound as *required*
arguments, so neither can be forgotten, and both reach the receipt. Three
kinds, and the distinction matters:

| kind | meaning | effect on the total |
|---|---|---|
| `OUTSIDE` | proved disjoint from the region by an exact rational test | contributes nothing; the total stays an enclosure |
| `UNRESOLVED_BOUNDARY` | straddles the boundary, unresolved at the depth limit | contributes its `residual`; without one, `covers_region=False` and `certified=False` |
| `EXCLUDED` | excluded by a stated predicate (unused here) | as above |

> **`boundary_area_bound` is the right name for two of those three kinds.** An
> `OUTSIDE` cell is *proved disjoint* from the region: it holds no boundary and
> contributes exactly zero. Its area is still retained — the recipe says retain
> every rejected cell — but summing it into one receipt field named after the
> boundary **overstates the unresolved boundary**. On the showcased bracket run
> the single figure `area_rejected_bound` is `28.90625`, of which `16.40625` is
> `OUTSIDE` (60 cells) and only `12.5` is genuinely `UNRESOLVED_BOUNDARY` (128
> cells). The direction is conservative so no bound is unsound, but the number
> does not mean what its name says. `Total.area_rejected_by_kind`,
> `Total.area_unresolved_boundary_bound` and the receipt's
> `area_rejected_by_kind` give the split without walking the cell list.

> **`Total.enclosure` is an enclosure of the region integral only when
> `covers_region` is `True`.** Otherwise it is a number about the *accounted*
> part, whatever the field is called. `Ledger.total()` raises on a PENDING cell
> but *returns* on an `UNRESOLVED_BOUNDARY` cell with no residual, with
> `covers_region=False`, `certified=False` and the caveat as free text — and a
> consumer that publishes the field under a `provenance="certified_interval"`
> stamp would publish a number about a strict subset. `Total.certified_enclosure()`
> makes the two failure modes symmetric: it raises `UncertifiedTotalError`
> unless both flags are `True`. **Any lane that stamps the word "certified" on a
> number from this package should call that accessor, not read the field.**

### The boundary, handled honestly

The annulus boundary `|y| = 1/10` and `|y| = 5` is not a rational polygon.
Both honest options in the task are implemented, on the **same** region, and
their totals are never added together:

* **`rn5_annulus_polar` (primary).** Polar cells with exact rational radii, so
  the boundary is represented *exactly* and **the rejected boundary area is
  exactly zero**. The difficulty does not vanish — it moves, and the module
  says where: the cell area `π·(t₁−t₀)·(r₁²−r₀²)` now carries `π` as a
  certified enclosure rather than an exact rational, and a cell's Cartesian
  extent needs certified `sin`/`cos` of `2πt` from `research/interval`, widened
  by the dependency problem. Parameter coordinates are `(radius, turn)` with
  `θ = 2π·turn`, so the parameter rectangle stays exactly rational. The map is
  injective except that `turn = 0` and `turn = 1` name the same ray; cells
  sharing that seam share a boundary segment of area zero, exactly as any two
  adjacent cells do. "Non-overlapping" means disjoint interiors throughout,
  which is what a Riemann sum needs. `r_lo > 0`, so there is no origin
  degeneracy.
* **`rn5_annulus_bracket` (alternative).** A rational Cartesian bracket.
  Classification is exact rational arithmetic on squared radii — no square
  root, no trigonometry, no float comparison — and straddling cells are refined
  while the depth budget allows and then REJECTED as `UNRESOLVED_BOUNDARY`
  with their areas retained as an explicit boundary-area bound and their
  possible contribution carried as a residual. It exists so the sliver
  accounting is exercised and tested.

### The near-axis refinement cost, faced

Stated plainly: the sources name this cost without defining "near-axis" in
terms this repository can resolve, so the reading used here is **stated as a
reading, not quoted as a source's**: the expensive part of `0.1 ≤ |y| ≤ 5` is
the inner edge, where the region approaches the excluded disc `|y| < 0.1` that
`reviews/records/REV-RN3-FARZONE-20260918.json` separately observes is
unaccounted for in RN3 §9's sum.

What the package does about it, concretely:

1. **`PolarRegion.uniform_cost(h)` is a count, not an estimate** — exact
   rationals, no fit, no sampling. It reports how many cells a *uniform* cover
   needs to reach a target Cartesian cell diameter, and the anisotropy factor
   `r_hi / r_lo`, which for the RN5 annulus is **50**: at a fixed angular step
   the arc extent of a cell is 50× larger at the outer edge than at the inner
   edge, so a uniform angular grid fine enough for one is badly wrong for the
   other. At a target diameter of `1/10` it returns 98 radial × 629 angular =
   **61,642 cells**, with an inner arc extent of about `9.99e-4` against an
   outer one of about `4.99e-2` — the 50× waste, as a number.
2. **The default split policy for the annulus is `"aspect"`** — split whichever
   direction currently dominates the cell's diameter bound — so refinement is
   spent where the geometry needs it. `"theta"`, `"radius"` and `"both"` are
   available and the receipt records which was used.
3. **The cost of a zeroth-order sup cover is quadratic in the tolerance and is
   not hidden.** The acceptance test `width(|C|·rng) ≤ tol·|C|/|domain|`
   reduces to `width(rng) ≤ tol/|domain|`: the cell area cancels, so the
   requirement is a *uniform* bound on the integrand's oscillation per cell.
   With a Lipschitz integrand that forces cell diameter `∼ tol/(L·|domain|)`
   and cell count `∼ |domain|³L²/tol²`. For the RN5 annulus `|domain| ≈ 78.5`,
   so a total width of 1 costs order 10⁵ cells of a *reference* integrand. Any
   real integrand will cost more. This is exactly why ten boxes are not a
   coverage certificate, and it is the argument for a higher-order cell bound
   (a certified Taylor or DS enclosure per cell) rather than a constant one —
   which is the same route `LANE_RN_UNIF.md`'s T4 item 1 already names as
   acceptable: "interval DS on each cell".

   **How this relates to RN5's own ten cells.** `docs/RESEARCH_MAP.md` §3 records
   that the RN5 repair's ten spatial certificates are **boxes**, certified by
   centered Taylor jets of order `N+2` with `N = 6`, separate marginal whitening,
   an L² remainder bound and interval Cholesky pivots — and that "ten boxes **do
   not** cover the annulus". Two consequences worth stating plainly. First, their
   geometry is Cartesian, so `rn5_annulus_bracket` is the closer analogue of it
   and `rn5_annulus_polar` is the boundary-exact alternative; **neither is their
   cover**, and no cell certified here is one of theirs. Second, their per-cell
   bound is already a *jet* bound, not a constant one — which is the higher-order
   route point 3 above argues for, and the reason the constant-per-cell cost model
   above is an upper bound on what a real cover would have to pay per unit of
   tolerance, not a prediction of it.
4. **The receipt carries `refine_depth`, `min_cell_width` and
   `max_cell_width`** next to any total. Read the first two for the resolution
   actually achieved: `max_cell_width` is the coarsest leaf and can saturate at
   a region's diameter cap, which it does in this package's own showcase
   configuration (see the warning above). An earlier version of this line said
   the receipt made "the achieved resolution always visible"; with a saturating
   cap that was satisfied only nominally, which is why the other two fields and
   the `cell_width_note` were added.

### `T4`: what the factory does and does not do

`t4_polar_cover()` is the cover geometry `LANE_RN_UNIF.md` names (T4 freeze
item 2, EXECUTE item 4) and nothing else. It does **not** supply T4:
`T4_form`, `T4_kap` and `C_comp` do not appear in the frozen engine at all
(`docs/ENGINE_RECOVERY.md`), and `rnu_t4.py` is listed CANNOT_VERIFY in
`docs/OPEN_PROBLEMS.md` §E. It also does not touch `env_tau`, which the lane
records as **fail-closing at `d = 5`** (λ₀ floor collapsed); do not evaluate
anything on that zone boundary through this region and call it certified.

With `split="theta"` — the policy the lane names — radial resolution is fixed
by the shell list, because theta-halving never shrinks a cell's radial extent.
A cell that stays over tolerance on radial width alone ends the run PENDING and
`total()` refuses. That is intended, visible in the receipt, and not papered
over.

### NON-CERTIFYING paths are labelled

An integrand declares `certifying`. One non-certifying integrand makes the
whole run non-certifying: the ledger is flagged at construction, the receipt
says so in `certifying` and `arithmetic`, and `Total.certified` is `False`.
`FloatProbeReference` exists only so a test proves that label propagates, so
that an `mpmath` integrand wired in later cannot be mistaken for a certified
one. It is not a bound.

---

## What must be bound for any of this to bear on Piece 2

Three things, none of which this package supplies:

1. **The corrected envelope, post-RN5-erratum.** `research/rn/moment_envelope.py`
   records why the pinned `d3_perc.py` helper `envelope_v` is not an upper
   bound at all — it used `E C⁴` where Hölder(4,4,2) requires `E C²`, and RN5's
   exact typed Gaussian counterexample separates them. Any cell supremum has to
   come from the **corrected** envelope. Note also the scope hold
   `Q-RN5-MOMENT-001` carried in `drive/source_map/Payloads.csv`, and that the
   recovery of `d3_perc.py` does not lift it.
2. **A certified cell supremum for `kappa_far`.** Today there is none. The
   frozen engine is mpmath at `mp.dps = 100` throughout with no exact-rational
   and no interval arithmetic anywhere (`docs/ENGINE_RECOVERY.md`), its
   `chi2_grad_bound` is reported ~1.57e14 against a true `|∇χ²|` ~1.563e-5 — a
   **~1e19 slack** — and its `mean_grad_exact` is missing chain-rule terms, so
   the "certified bound" downstream of it is not merely loose but void. A
   certified per-cell enclosure would have to be built, and the image-lattice
   tail would have to be bounded **uniformly over the cell** rather than at a
   point separation — which is verbatim the complaint `OBL-H5-JETMOD` makes
   about LAT's current tail bound.
3. **The remote-budget reassembly rule.** RN5's recipe ends "then reassemble
   the remote budget"; the rule for doing so, and for composing it with
   `B_remote = 19.55` carried symbolically at D3's grade, is not implemented
   here and is not stated here.

Until all three are bound, a run of this driver is a run of a REFERENCE
integrand over a geometry. It is infrastructure, not a result.

---

## Tests

`tests/test_cover.py`. Negative controls are the deliverable:

* partition exactness on a hand-verifiable region;
* a cover with a **gap** must fail — in each of the three distinct ways a gap
  can appear: an empty strip, an interior interruption, and a run that stops
  short of the top;
* a cover with an **overlap** must fail **even though its areas sum exactly to
  the domain area** — built deliberately, the sharpest control here;
* `total()` must **raise** while any cell is PENDING, including on a real
  driver run that exhausts its depth budget;
* a rejected cell must survive into the receipt with its reason and its
  boundary-area bound;
* the bracket run's **accounted area plus retained rejected area must equal the
  bracket area exactly**, and its certified total must **contain the true region
  integral** — computed for the constant reference integrand `f ≡ 1` from the
  region's own exact radii and the certified `π`, independently of any cover
  run. These are the two controls the reject path previously lacked;
* refinement must reduce total width monotonically, and the refined enclosure
  must be **contained in** the coarse one (the subdivision theorem, asserted
  directly);
* the reference integrand's certified total must **contain** the value from a
  dense direct float evaluation (labelled NON-CERTIFYING), and must intersect
  the independently derived closed form;
* a non-certifying integrand must reach `certified=False` in the receipt.

**The mutations enumerated in the test module's docstring were run against a
deliberately broken copy of the package and confirmed to fail there.** That is
the coverage claim, and it is the whole of it.

An earlier version of this section said "Every control in the file was run
against a deliberately broken copy of the package and confirmed to fail there;
each names its mutation in its docstring." Both halves were false. A mechanical
scan of the test docstrings shows that **17 of 29 named no mutation at all**
(`test_1`, `test_1b`, `test_2c`, `test_3c`, `test_4`, `test_4b`, `test_6b`,
`test_7b`, `test_8`, `test_11`, `test_15`, `test_16`, `test_17`, `test_18`,
`test_19`, `test_20`, `test_21`), and the mutation-to-control table maps its
mutations onto controls 2b, 3, 3b, 4, 5, 6, 6b, 7, 7b, 9, 10, 11, 12, 13, 14,
15 and 17 only — so ten controls were never confirmed to fail against any broken
copy. Those are ordinary behaviour tests, which is fine; calling them all
confirmed controls generalised past the evidence, and it is the sentence a
reader would rely on when deciding how much the suite is worth.

The honest version is the one the test module's own docstring makes: a specific,
enumerated list of mutations was run and caught. That claim is checkable.

### The systematic sweep of 2026-09-22, and what it measured

The paragraph above was written from a docstring scan. It has now been replaced
by a measurement. Every comparison and boolean operator in `ledger.py`,
`regions.py` and `driver.py` was flipped one at a time -- 93 single-operator
mutants -- and the suite was run against each.

| | before | after |
|---|---|---|
| mutants run | 93 | 93 |
| caught | 71 | **84** |
| survived | 22 | **9** |
| tests firing on at least one mutant | 36 of 37 | **45 of 46** |

So the "ten controls were never confirmed to fail" of the previous section was
both stale and, in the direction that matters, pessimistic: most of this file's
tests do catch something. What the sweep found instead was **22 specific holes**,
and controls 30-38 close the thirteen of them that were real rather than
equivalent. Each names the mutant it exists to catch in its own docstring, and
each was confirmed by hand-applying that mutation to a copy of the package
outside the repository -- not by trusting the sweep, which is the right order
given the caveat below.

**The nine survivors, classified.** None is left silent:

| site | mutation | why it survives |
|---|---|---|
| `ledger.py:394` | `or` -> `and` | inside the *text* of an error message (`" \| ".join(caveats) or "(none recorded)"`). Cosmetic. |
| `ledger.py:667` | `<=` -> `<` | the "(+N more)" threshold in a message, at exactly 8 pending cells. Cosmetic. |
| `ledger.py:684` | `or` -> `and` | a defensive branch already marked `# pragma: no cover`. |
| `regions.py:297`, `:299` | `>` -> `>=` | the theta-halving policy, differing only when arc and radial extent are *exactly* equal. |
| `regions.py:399`, `:412` (x2) | `<=` -> `<`, `<` -> `<=` | disjointness tests, differing only at exact tangency. |
| `driver.py:217` | `<=` -> `<` | accept-at-budget, differing only when the width is *exactly* the budget. |

Six of the nine are exact-equality boundaries that the reference geometry never
lands on, and three are message text or a `pragma`-marked branch. That is a
statement about what was measured, not a claim that they are all provably
equivalent; a cover whose cell widths happened to hit one of those equalities
would separate them.

**A caveat on the harness, stated because it bit.** The sweep identifies a
mutation site by index and reports the source line of the node it mutated. On
one site (`ledger.py:805`, the receipt's `min_cell_width`) its verdict
disagreed with hand-applying the same edit, which the suite caught at once. So
the per-site attributions above are the sweep's, spot-checked by hand where
they drove a decision, and every control added from them was verified by hand
mutation rather than by the harness. The headline counts are the sweep's own
and have not been reproduced by a second method.

### Mutations that survived, recorded rather than fixed quietly

**Three surviving mutations are on record**, two of them found by an adversarial
audit on 2026-09-18 and closed by controls 22–24:

* **MX3** — `CartesianBracketRegion.area_rational_upper` returning
  `param_area()/4`, understating every retained boundary bound fourfold *and*
  shrinking the driver's residual by the same factor.
* **MX4** — the driver's `UNRESOLVED_BOUNDARY` residual shrunk 1000×, leaving
  `boundary_area_bound` intact so the old control 9's `> 0` still passed.

**Both left all 29 tests green**, and both then produced a `Total` with
`certified=True`, `covers_region=True`, `caveats=()` whose `enclosure` **did not
contain the true region integral**. With the constant REFERENCE integrand
`f ≡ 1`, whose integral over the annulus is exactly `π(25 − 1/100) =
78.50840041…`, the bracket at `tol=30, max_depth=5` gives baseline
`[71.09375, 83.59375]` (contains it), MX3 `[71.09375, 74.21875]` and MX4
`[71.09375, 71.10625]` (both exclude it).

The shipped code was correct in both cases. **The controls were the hole**, and
it was precisely the hole the "negative controls are the deliverable" framing
claimed not to have: control 9 asserted only that the retained bound and the
residual were positive, control 10 supplied its residual by hand, and every
containment control ran on `rn5_annulus_polar`, which by construction rejects no
cells. The entire reject path — the only path where a boundary sliver is
*bounded* rather than represented exactly — had no containment control at all.

Controls 22 (exact area bookkeeping on the bracket: accounted + retained
rejected `== 100`, no slack), 23 (the bracket total must contain the true
annulus area, computed from the region's own exact radii and the certified `π`)
and 24 (the degenerate all-OUTSIDE classification fault, which balances the
bookkeeping and is caught only by containment) close them. Nothing here
establishes that no fourth mutation exists.

Re-run on scratch copies on 2026-09-18, with the current test file:

| broken copy | mutation | caught by |
|---|---|---|
| MX3 | `CartesianBracketRegion.area_rational_upper` returns `param_area()/4` | 3 tests (22, 23, 27) |
| MX4 | driver's `UNRESOLVED_BOUNDARY` residual shrunk 1000× | 1 test (23) |
| MX6 | `Total.certified_enclosure()` never refuses | 1 test (26) |
| MX7 | `area_rejected_by_kind` lumps every kind under `OUTSIDE` | 1 test (27) |
| MX8 | `provisional_leaf_counts` never reports an omitted leaf | 1 test (28) |
| MX9 | `min_cell_width` returns the maximum | 1 test (29) |

Note that **MX4 is caught by control 23 alone**: the area bookkeeping of
control 22 is untouched by it, and MX5 (the all-OUTSIDE fault) balances control
22 exactly. Containment against independently computed geometry is the control
that does the work; the bookkeeping identity is the one that localises the
fault when it fires.

**And one mutation survived the first version of that file**, recorded there
rather than quietly fixed. Disabling the interior-gap branch
(`lo > cur`) of `check_exact_partition` left the whole suite green, because the
gap controls written first both had gaps running to the *top* of the domain and
so were caught by the trailing check instead. Two controls were added for the
branches nothing reached. A suite that stays green under a mutation is not
testing that line, which is what negative controls are for.
