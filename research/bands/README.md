# `research/bands/` — band-certification machinery (lane A1, `OBL-H5-JETMOD`)

> **Status.** `OBL-H5-JETMOD` is **OPEN (display only)** and nothing in this
> directory changes that. Neither does anything here touch `OBL-H5-ZBAND` (hi
> side), `OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE` (chart side) or either
> Piece of `D3-LEMMA-RN-UNIF`. All stay exactly as `docs/OPEN_PROBLEMS.md`
> records them. **Original prize problems solved: 0.**

## The obligation this lane serves

`docs/OPEN_PROBLEMS.md` A1, carrying `CHART_SIDE_JETMOD_PLAN.md` quoting
`H5_PROMOTE.md` §3:

> **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set (not just
> the displayed c₂) over the r-bands `[r_{k+1}, r_k]`, with lattice-tail
> constants re-certified uniformly in the band (LAT's tail bound currently
> certifies at point separations). Content: for each jet `J` and band `B`,
> `J(B)/r^{p_J}` ∈ certified interval; falsifier: a band enclosure whose width
> exceeds the claimed modulus.

and the proof step it names (`H5_PROMOTE.md` §3(iii)):

> *Band certification (the proof step).* … evaluating those sums with `r` as an
> interval over the band yields G12-band enclosures, hence `Î(r)/r³ ≤
> F(G12-band)` for the whole band — a FINITE computation per band, never a
> fitted exponent.

## What is here

| File | What it is |
|---|---|
| `ladder.py` | the three published `I_hi/r³` **point** certifications as exact `Fraction` data with their packages and totals digests, plus the display-vs-enclosure gap made numerical |
| `lattice.py` | a certified periodized lattice-sum evaluator: interval-`r` truncated sum **plus a proved tail bound uniform over the band**, with a pluggable kernel protocol |
| `falsifier.py` | the obligation's own falsifier as an executable check, with `INSUFFICIENT_DATA` as a first-class outcome |
| `../../tests/test_bands.py` | 72 tests (61 functions, 20 of them negative controls) plus two closed-form faithfulness harnesses |

Standard library only, plus `research/interval/`. Python 3.11. No `mpmath`, no
`numpy`, no `float` anywhere a bound is claimed.

### `lattice.py` — the mathematical core

A periodized kernel on a torus of period `L` is `S(d) = Σ_{n∈Z²} kplane(d+Ln)`,
which is infinite; every implementation truncates it. The frozen engine
`d3_rn_unif.py` truncates at the **first image shell only** (`_IMG` = the eight
points of `{-1,0,1}² \ {0}`, `_LT = 24`), so `kdcov` sums nine plane-kernel
values and **returns that nine-point sum**. Its `tail_bound(order)` is asserted
at import to be `< 1e-60`, but nothing adds it: no enclosure widens by it. That
tail bound is also computed at a *point* separation (`rho = m*_LT - 17`
hard-coded), is itself truncated (`for m in range(2, 8)`, every `m ≥ 8` dropped
with no remainder term), and is `mpmath` floating point.

This module supplies the other version:

* the truncated sum is evaluated with the displacement as an **interval box**,
  through `research/interval/`, so one enclosure holds for **every**
  displacement in the box at once;
* the omitted lattice points are covered by a **closed-form, proved** tail bound
  with **no second truncation**, and the bound depends on the box only through a
  single radius `R` — that is the uniformity `OBL-H5-JETMOD` asks for;
* the kernel is **pluggable**: a callable plus a `DecayEnvelope` carrying its
  certified decay constants *and a mandatory written justification*. The
  constructor refuses an empty justification.

The two envelope branches and their proofs are written out in full in
`tail_bound`'s docstring. In outline, with `M = n_trunc+1` and
`a = L·M − R > 0` (checked, and refused otherwise):

* **shell geometry** `|d + Ln| ≥ L|n|_∞ − R = Lm − R` for every `d` in the box —
  the reverse triangle inequality, and the step that makes the bound uniform;
* **shell counts** `#{n : |n|_∞ = m} = (2m+1)² − (2m−1)² = 8m`;
* **Gaussian branch**, from `(a+Lk)² ≥ a² + (2aL+L²)k` for integer `k ≥ 0` and
  the two geometric series `Σq^k` and `Σk q^k`, with `q = exp(−B(2aL+L²)) < 1`:

  ```
  Σ_{|n|_∞>N} |k(d+Ln)|  ≤  8A·exp(−B a²)·[ M/(1−q) + q/(1−q)² ]
  ```

* **power branch**, from `Lm − R ≥ cLm` with `c = 1 − R/(LM) ∈ (0,1]` and the
  integral comparison `Σ_{m≥M} m^{1−p} ≤ M^{1−p} + M^{2−p}/(p−2)` (finite
  because `p > 2`):

  ```
  Σ_{|n|_∞>N} |k(d+Ln)|  ≤  8A·(cL)^{−p}·[ M^{1−p} + M^{2−p}/(p−2) ]
  ```

Every quantity is evaluated in certified interval arithmetic and the **upper**
endpoint is returned as an exact `Fraction`, so rounding only ever loosens the
bound.

`band_enclosure(kernel, r_band, ...)` returns a `BandEnclosure` carrying the
breakdown the obligation needs to be checkable: the truncated-sum enclosure, the
tail bound, the total, and the total width.

```python
from fractions import Fraction as F
from research.interval import Interval
from research.bands import band_enclosure, gaussian_reference

enc = band_enclosure(gaussian_reference(), Interval(F("0.025"), F("0.035355")))
enc.total                # certified enclosure over EVERY DISPLACEMENT IN THE BOX
enc.tail                 # the proved omitted-lattice bound, ~1e-498 at L = 24
enc.width()              # the quantity the falsifier tests
enc.certified            # evaluator_certified AND envelope_certified, both default False
enc.envelope_certified   # the decay envelope is a PREMISE; this says whether it was checked
```

**Read `enc.total` as "valid for every displacement in the box", and stop
there.** Carrying it to "hence for every `r` in the band" runs entirely through
the `displacement` map, and the maps shipped here — `axial_displacement`,
`diagonal_displacement` — are documented PLACEHOLDERS. With a stand-in map the
second half is a statement about the stand-in geometry, not about the program's
six-pin configuration. `enc.caveats` says so on every record.

**Two independent certification flags, both defaulting to `False`.**
`PlaneKernel.certified` covers the *evaluator* only. `DecayEnvelope.certified`
covers the *envelope*, which is a premise nothing here can check: hand
`tail_bound` an envelope claiming `B = 7` for a kernel that decays at `B = 1/2`
and it returns, in exact arithmetic, a "bound" 2.4e24 times too small.
`band_enclosure` writes `certified: true` only when both flags are set, and
attaches a NON-CERTIFYING caveat naming whichever is missing.

### `ladder.py` — the published numbers, and the gap

| package | r | `I_hi/r³` | totals sha256 |
|---|---|---|---|
| `H5_RUNG2_2026-09-15.md` | 0.025 | 664.3979 | `f7697bcfa0fe32b5…` |
| `H5_RUNG3_2026-09-15.md` | 0.035355 | 661.4712 | `808d6901e3254a73…` |
| cited in both rung docs | 0.05 | frozen v1 **731.4311** / live v3 clean **647.8048** | none quoted, none invented |

Two computed observations, both reproduced here from the `Fraction`
transcriptions rather than copied from prose:

1. **Same-`r` version spread.** At `r = 0.05` the two engine lines differ by
   exactly `83.6263`, which is `836263/6478048 = 12.9091819…%` of the live value.
   A point certification is therefore not by itself stable across engine
   versions at fixed `r`.
2. **The *minimal* implied modulus constant differs between the two bands.**
   (Not "there is no single constant" — there is one, and
   `common_admissible_constant` exhibits it; see below.) Under
   `|Δ(I_hi/r³)| ≤ C·δ^κ` at the displayed `κ = 1/8`:

   | band | δ | \|Δ\| | forced `C ≥` |
   |---|---|---|---|
   | [0.035355, 0.050000] | 0.014645 | 13.6664 | **23.1709028…** |
   | [0.025000, 0.035355] | 0.010355 | 2.9267 | **5.1818453…** |

   a ratio of **4.4715543…** between the two forced **minima**.
   `implied_modulus_constant(kappa)` takes `κ` as an argument — nothing is
   hardcoded to `1/8` — and `ratio_table()` reports the ratio across a sweep,
   where it is strictly decreasing in `κ` and crosses 1 between `κ = 4` and
   `κ = 5` (both bracketing evaluations certified). The crossing is where the
   two minima coincide. It is **not** the point at which a common constant
   first becomes possible: see the paragraph below.

> **THE ASSUMPTION, which travels with every one of those numbers.** This is the
> *natural reading* of the shipped display: a modulus of continuity on the
> quantity `I_hi/r³`, absolute form, with `κ` taken from the displayed `κ = 1/8`
> fit — itself a dense certified **sampling** of `c₂` plus an explicit fit,
> labelled DISPLAY by its own source. It is **not** a quotation of any source's
> own modulus statement. If the intended modulus is on a different quantity, or
> is relative rather than absolute, or carries a different exponent, these
> numbers change. `IMPLIED_MODULUS_ASSUMPTION` is attached to every returned
> object and printed by every formatter, so the number cannot travel without it.

**This is not a refutation and must never be presented as one.** It does not
show the displayed modulus is wrong. What it shows is why three point
certifications cannot stand in for a band enclosure: the supremum over a band is
unconstrained by the values at its endpoints absent an independently certified
modulus. That is the display-vs-enclosure distinction the sources already draw,
made numerical.

**And it is not an inconsistency — say the true version.** An earlier version of
this section said the published points "do not themselves exhibit a single
constant". *That is false, and the package now computes the correction.* A
modulus is an **inequality** `|Δ| ≤ C·δ^κ`; two bands forcing different
*minimal* constants are entirely compatible with one admissible constant, namely
the larger. `common_admissible_constant(κ)` exhibits it and re-checks both bands
exactly: at `κ = 1/8` a single `C ≈ 23.1709028` satisfies both, and
`max(C₁, C₂)` satisfies both at every `κ` in `ratio_table`'s default sweep
(`0, 1/8, 1/2, 1, 2, 4, 5`). What the two bands differ in is the **minimum each
forces**, by 4.4716×. That is a statement about **tightness** — how much slack a
common constant has to carry — and not about consistency. Likewise the ratio
crossing 1 between `κ = 4` and `κ = 5` is where the two *minima* coincide, not
where reconciliation first becomes possible; reconciliation is possible at every
`κ`. A package whose stated purpose is to refuse false inferences has to state
this the right way round.

### `falsifier.py` — the obligation's own test

```python
falsifies(enclosure_width, claimed_modulus)  ->  width > modulus
```

exactly the source's sentence, *"a band enclosure whose width exceeds the
claimed modulus"*. Strict: equality does not falsify. Both arguments are exact
`Fraction`s; `float` and `None` are refused rather than silently answered.

`format_report` prints a decimal column *and* the exact `Fraction`s for every
row, and labels the decimal column **NON-CERTIFYING** in the output itself: two
rows whose widths differ in the sixteenth digit print the same `%.6g` string and
can carry opposite verdicts. The verdict is decided on the exact values; the
column is for reading. Quote the `exact:` line.

`band_report` returns one row per band with `PASS` / `FALSIFIED` /
`INSUFFICIENT_DATA`. **`INSUFFICIENT_DATA` is a first-class outcome**: a band
whose enclosure or modulus is missing is *not* a pass, and `report_is_clean` is
`False` whenever any row is in that state. The claimed modulus is an **input**;
this module neither derives nor endorses one, and the constants in `ladder.py`
are emphatically not a source's claimed modulus.

---

## What this directory does **NOT** establish

This section is the point of the README, not an afterthought.

**It does not discharge `OBL-H5-JETMOD`, and no run of this code can.** The
obligation is over *the program's own 24-jet set*, with *the program's kernel*,
over *the program's r-bands*. What ships here is machinery plus reference
kernels. Machinery is not a result.

**Three bindings are missing, and none of the three exists in this repository.**

1. **The program's `kplane`, with a certified decay envelope for it.** The
   engine's kernel is
   `kplane(a1,a2,dx,dy) = (−1)^(a1+a2)·He_{a1}(dx)·He_{a2}(dy)·exp(−|z|²/2)`
   with `C_KERN = 1` and `He` the probabilists' Hermite polynomials. The
   reference kernel here is that expression at derivative multi-index `(0,0)`
   and nothing more. Binding the real one needs the Hermite factors implemented
   **and** a certified `A`, `B` (or `A`, `p`) dominating
   `|He_{a1}(dx)·He_{a2}(dy)|·exp(−|z|²/2)` at every order the 24-jet set
   reaches — including the orders 5…10 the engine's own `tail_bound` touches.
   Nothing in this repository establishes such constants. Note also that the
   engine's polynomial envelope helper `_he_abs_poly` is a one-axis majorant
   while the true tail term is two-axis; the engine's docstring asserts that
   reduction and its body does not prove it. A certified envelope would have to.
2. **The 24-jet definitions with their powers `p_J`.** The obligation's content
   line is `J(B)/r^{p_J} ∈` a certified interval, for each of twenty-four jets.
   This repository defines no jet and no `p_J`.
   `normalized_band_enclosure(kernel, r_band, power)` is the *shape* of that
   normalisation with `power` supplied by the caller; it is not that statement.
3. **The actual band endpoints `r_k`.** `ladder.adjacent_bands()` returns the
   intervals between the published points, which is not the same thing as the
   program's `[r_{k+1}, r_k]`. Those endpoints are not bound here.

Also missing, and worth naming separately: **the map from `r` to the
displacement box.** `axial_displacement` is a stand-in. The program's six-pin
configuration determines which covariance displacements a separation `r`
induces, and that geometry is not in this repository.

**Further non-claims.**

* A `PASS` row from the falsifier is not evidence for the obligation. It means
  one width was compared with one claimed modulus. A whole table of `PASS` rows
  would leave `OBL-H5-JETMOD` exactly as OPEN as it is now.
* A `FALSIFIED` row is not a refutation of the program either. It says the pair
  handed to the function failed the source's own test; which of the two is at
  fault, this code cannot say.
* The point certifications in `ladder.py` remain point certifications. No
  arithmetic performed on them turns them into a band enclosure.
* The rung ladder's engineering status is untouched: `r = 0.0177` stands at
  **42/70** cells and `r = 0.0125` at **21/70**, two shards resuming each. No
  line count and no computation here promotes that.
* The frozen engine is **not** patched, read or executed by this directory. Its
  defects are cited from `docs/ENGINE_RECOVERY.md` to explain what the machinery
  is for; citing a defect is not repairing one.
* Nothing here composes the 2D upper track, the 2D lower track or the 3D
  lifetime track. `ERRATA_AND_CLARIFICATIONS_2026-09-13` withdraws exactly that
  composition and this directory respects it.
* Nothing here bears on any prize problem.
* **A `DecayEnvelope` is a premise this package cannot check, and its
  `certified` flag is an assertion the caller makes.** Nothing here inspects
  the kernel's global behaviour, so nothing here can verify that `|kplane(z)|`
  really obeys the `A`, `B` (or `A`, `p`) handed to `tail_bound`. Feed it a
  false envelope and it returns, in exact certified arithmetic, a bound on a
  kernel that is not yours — the package's own control exhibits one **2.45e24×
  too small**. Every arithmetic step downstream of a false premise is still
  exact and still wrong. `envelope_certified` is where a human records that the
  written justification has been read; the constructor's refusal of an empty
  justification is a discipline, not a proof.
* A green `tests/test_bands.py` is not a mathematical review. The tail-bound
  argument is ordinary mathematics written out in `tail_bound`'s docstring for a
  human to check; the tests check that the code behaves as that argument says on
  the inputs exercised.

## If you want to make this bear on the obligation

In order, and none of them is small:

1. Establish and **write down the proof of** a certified decay envelope for the
   program's `kplane` at every jet order in the 24-jet set — two-axis, not the
   one-axis majorant the engine asserts. Put it in a `DecayEnvelope`
   `justification`; the constructor will not accept it empty.
2. Implement `kplane` as a certified `PlaneKernel.evaluate` over
   `research/interval/`. The Hermite recurrence is exact in `Fraction`
   arithmetic; only the `exp` factor needs the transcendental layer.
3. Bind the jet set: twenty-four `(J, p_J)` pairs, and the map from `r` to each
   jet's displacement box.
4. Bind the real band endpoints `r_k`.
5. Run `band_enclosure` per jet per band, feed the widths to
   `falsifier.band_report` against a modulus **whose provenance is stated**, and
   expect `INSUFFICIENT_DATA` for every band not yet covered — that is the
   report doing its job, not failing.

Even then, the result would be a certified band enclosure, which is a step the
obligation names. Whether the obligation is thereby discharged is a judgement
for the register and its operator, not for this code and not for whoever runs
it.

## Tests

```bash
python3 -m pytest -q tests/test_bands.py
```

72 tests, 20 of them negative controls, and the controls are the point of the
file: dropping the tail
bound breaks containment; a weakened tail breaks containment; a false decay
envelope (`B = 1` for a kernel that decays at `B = 1/2`) loses domination;
flipping the reverse triangle inequality loses domination, on both envelope
branches; undercounting a lattice shell loses domination; an inward-rounded band
enclosure excludes a true value on its boundary; a mistranscribed published digit
moves the forced constant outside its asserted window; a band with no data comes
back `INSUFFICIENT_DATA` and never `PASS`; and shrinking a claimed modulus below
a real certified width flips the falsifier.

Eight controls were added after an adversarial audit on 2026-09-18, each closing
a defect the audit demonstrated on the shipped code: a false decay envelope can
no longer produce a record reading `certified: true`; neither certification flag
alone suffices; both default to `False`; the normalised ratio `J(B)/r^{p_J}`
cannot be separated from its caveats; the falsifier's printed report labels its
lossy decimal columns and prints the exact `Fraction`s beneath them; and a
single admissible modulus constant is exhibited and checked at every `κ` in the
sweep. The uniformity test now sweeps the whole 2-D box including its
off-diagonal corners rather than the diagonal alone, and three parametrised
near-edge tests pin domination down to `a = L(N+1) − R = 1e-6`, where the
geometric ratio is nearest its floor. The `q.hi ≥ 1` guard is exercised
directly; the `c.lo ≤ 0` guard is documented as **defensive and unreachable
while `a > 0`** — with a test pinning that implication — rather than covered by
a control that could not fire.

Every control was additionally run against a deliberately broken copy of this
package in a scratch directory. **Thirteen mutations were tried and all
thirteen were caught.** The counts below were re-measured against the current
test file on 2026-09-18; they move when tests are added, which is why they are
dated rather than presented as constants.

| broken copy | mutation | caught by |
|---|---|---|
| 1 | `band_enclosure` returns the truncated sum, tail dropped — the frozen engine's shape | 4 tests |
| 2 | `a = L*M + R`, the reverse triangle inequality reversed | 14 tests |
| 3 | `4 * envelope.A` — a 4m shell count instead of 8m | 4 tests |
| 4 | `w >= m` in `falsifies` — equality now falsifies | 4 tests |
| 5 | `band_verdict` returns `PASS` for a missing input | 3 tests |
| 6 | `664.3979` → `664.3978`, one published digit | 5 tests |
| 7 | `_box_radius` uses `mig()` instead of `mag()` — an inward box radius | 4 tests |
| 8 | `DecayEnvelope.certified` defaults to `True` again | 3 tests |
| 9 | `PlaneKernel.certified` defaults to `True` again | 2 tests |
| 10 | `BandEnclosure.certified = kernel.certified` — the evaluator flag alone | 1 test |
| 11 | `normalized_band_enclosure` returns a bare `(Interval, BandEnclosure)` | 1 test |
| 12 | `falsifier.format_report` drops the `exact:` line under each row | 1 test |
| 13 | `common_admissible_constant` takes the `min` forced constant, not the `max` | 1 test |

**An earlier version of this table recorded row 2 as caught by 8 tests.** Rerun
on a scratch copy at the time it was **9** — the row was simply miscounted — and
it is **14** against the current file, because the new whole-box uniformity
sweep and the three near-edge parametrisations all catch it too. A table offered
as a reproducible receipt has to reproduce, so the miscount is corrected here
rather than left standing. Row 7 likewise moved from 3 to 4 for the same reason.

Two of those runs found real gaps in the first draft of this file and the tests
were strengthened rather than the runs reported as clean: mutations 1 and 3 were
each caught by only one or two assertions, and marginally. A point-band
containment test now catches 1 directly through `band_enclosure` (the wide-box
tests do not, because band variation swamps a 5.6e-3 tail), and an independent
closed-form reimplementation of both branches now catches 3 exactly (the
domination margin on the power branch is only about 2.0x, too small to catch a
4x undercount reliably).

One control is documented as *unable* to fire in one configuration rather than
quietly dropped: a 4× shell undercount is **not** caught on the Gaussian
reference kernel, because the proved bound there is about 13× the actual omitted
mass (every point of a shell is charged at the shell's nearest distance). The
control therefore runs on the inverse-power kernel, where the bound is about
2.0× the truth. A control that cannot fire is worse than no control, so the
limitation is stated in the test's own docstring.
