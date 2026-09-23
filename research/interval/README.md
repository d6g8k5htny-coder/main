# `research/interval/` — certified interval arithmetic over exact rationals

**The contract.** Every `Interval` this package returns **provably contains**
the true value. Containment is unconditional: it does not depend on the `prec`
hint, on operand size, or on the order of operations. Tightness is best effort
and does depend on `prec`.

Standard library only (`fractions`, `decimal`, `typing`). Python 3.11. No
`mpmath`, no `numpy`, no `math`, no float anywhere in the library.

## Why it is here

`docs/OPEN_PROBLEMS.md` states three blocking items in terms of **certified
interval bounds** and **certified enclosures**:

| Item | What it asks for |
|---|---|
| `OBL-H5-JETMOD` (A1) | `J(B)/r^{p_J} ∈` a certified interval, for each jet `J` and band `B`; lattice sums evaluated with `r` as an interval over the band |
| `OBL-H5-ZBAND` hi side (A3) | the **band** version of the LPW bracket |
| `D3-LEMMA-RN-UNIF` (A5) | a complete non-overlapping spatial cover with retained boundary-area bounds |

The program's actual computational carriers evaluate the corresponding
quantities in `mpmath` at high precision (RN3 at 384 bits, the RN5 repair at
384 bits). **High precision is not certification.** A 384-bit floating-point
evaluation with no interval discipline produces a number, not an enclosure.
This package is the arithmetic layer a certified carrier would need. It is
infrastructure, not a result.

## What this package does NOT establish

* It **discharges, reduces, closes, promotes and reclassifies nothing**.
  `OBL-H5-JETMOD`, `OBL-H5-ZBAND`, `OBL-H5-REMOTE-THRESHOLD`,
  `OBL-D1-PROMOTE`, `D3-LEMMA-RN-UNIF`, `PERC-DECAY`, `OBL-B1-BRANCH` and the
  `B4.loc` wrap/remote reconciliation stand exactly as `docs/OPEN_PROBLEMS.md`
  records them. Having a certified arithmetic is not having a certified result.
* It **re-certifies no existing number**. Quantities in the corpus produced in
  `mpmath` or `numpy` remain as certified — or as uncertified — as their own
  sources say. Nothing here relabels them.
* It supplies **no** lattice sum, **no** jet, **no** band enclosure, **no**
  rung, **no** moment, **no** coverage certificate. `OBL-H5-JETMOD` asks for a
  finite per-band computation; none of it is here.
* It relates the 2D upper track, the 2D lower track and the 3D lifetime track
  in **no way**, and composes none of them.
* It bears on **no** prize problem. None is solved.
* Green tests are not a mathematical review. The containment arguments are
  written out as short proofs in the function docstrings and a human must read
  them.

## Public API

```python
from research.interval import (Interval, to_fraction, sqrt, exp, log, sin,
                               cos, pi, erf, erfc, Phi, normal_sf, normal_pdf,
                               ERF_CROSSOVER, EXP_BIT_LIMIT, MILLS_MIN)
```

All sixteen names in `__init__.__all__` are shown. `to_fraction` converts a
scalar exactly; `ERF_CROSSOVER`, `EXP_BIT_LIMIT` and `MILLS_MIN` are the three
regime boundaries the tables below refer to by name.

`core.Interval` — endpoints `lo, hi : Fraction`, invariant `lo <= hi`.

| | |
|---|---|
| `Interval(lo, hi=None)` | `hi=None` gives a point interval |
| `Interval.exact(x)` | point interval from `int` / `Fraction` / `str` / `Decimal` |
| `+  -  *  /  -x  ** n  in` | outward-rounded; `**` takes a plain `int` |
| `width() mid() mag() mig()` | exact `Fraction` |
| `hull(other)` / `intersect(other)` | `intersect` returns `None` when disjoint |
| `contains_zero()` / `is_point()` | |
| `round_out(sig_bits)` | widen outward to bounded endpoint size; never narrows |
| `__repr__` | exact fractions **and** an outward-rounded decimal display |

`transcendental` — each `(x: Interval, prec: int) -> Interval`, except
`pi(prec) -> Interval`:
`sqrt` (needs `x.lo >= 0`), `exp`, `log` (needs `x.lo > 0`), `sin`, `cos`,
`pi`, `erf`, `erfc`, `Phi`, `normal_sf`, `normal_pdf`.

`erfc` and `normal_sf` (`normal_sf(x) = P(X > x) = 1 - Phi(x)`) carry the tail
mass as the primary quantity. **Use them for a tail bound; do not write
`1 - Phi(x)`.** The subtraction itself is exact, but the enclosure it subtracts
is rounded to a fixed number of significant bits, so beyond about `x = 26` the
upper endpoint of `Phi` rounds up to exactly 1 and the difference is `[0, …]` —
sound, and useless.

Deliberate refusals, each part of the contract:

* `float` endpoints raise `TypeError`. `Fraction(0.1)` is not `1/10`.
* `some_float in interval` raises `TypeError` too; it does **not** return a
  bool. Neither bool is honest — `False` is a wrong answer and `True` admits a
  binary double as an exact probe — so membership refuses. Convert the probe.
* A `Decimal` is accepted and converted **exactly**, including
  `Decimal(some_float)`, which *is* the binary double: `Interval(Decimal(0.1)).lo`
  is `3602879701896397/36028797018963968`, not `1/10`. The endpoint is an honest
  rational so containment is untouched, but the float refusal does not reach
  through a `Decimal` and a `Decimal` does not carry its provenance. Write
  `Decimal("0.1")`. This is the one hole in the refusal and it is tested so it
  stays visible.
* Division by an interval containing zero raises `ZeroDivisionError`. There is
  no extended-interval fallback.
* `x ** 0 == [1, 1]` for every `x`, including intervals containing zero.
* `exp` of an argument past `EXP_BIT_LIMIT` on the **positive** side raises
  `OverflowError` rather than allocating without bound (see below).
* `repr` never raises, whatever the endpoint size.

## How each enclosure is certified

Each is proved in the function's own docstring; in one line each:

| Function | Certificate |
|---|---|
| `sqrt` | integer Newton `floor(sqrt(p*q*N^2))`, endpoints rounded **outward**, then `lo^2 <= a` and `hi^2 >= a` **re-verified exactly in `Fraction` on the endpoints actually returned**. That check is the certificate; it is kept in the code on purpose. |
| `exp` | reduce `exp(t) = exp(t/2^k)^(2^k)` to `|u| <= 1/2`; Taylor with `\|R_n\| <= \|u\|^(n+1)/(n+1)! * 1/(1 - \|u\|/(n+2))`, derived from `(n+1+i)! >= (n+1)!(n+2)^i`; `k` squarings, monotone on non-negatives. |
| `log` | `a = m*2^k`, `m ∈ [1,2)`; `log m = 2 atanh((m-1)/(m+1))`, positive terms, tail `<= z^(2j+3)/((2j+3)(1-z^2))`; `log 2` gets its own certified enclosure. |
| `pi` | Machin `pi/4 = 4 atan(1/5) - atan(1/239)`, alternating-series remainder on each `atan`. |
| `sin`, `cos` | reduction into `[-pi/4, pi/4]` **against the certified `pi` enclosure**, interval-valued so the uncertainty in `pi` becomes width; alternating series, remainder = first omitted term (justified: ratio `<= 1/6` for `\|s\| <= 1`); interior extrema at multiples of `pi/2` added whenever their enclosure meets the input; `[-1, 1]` when the input is a full period wide. |
| `erf`, `erfc` | Maclaurin alternating series for `\|z\| <= 6`, truncated only at an index `n >= floor(z^2)+1` where the terms are provably decreasing, **intersected** for `\|z\| >= 1` with the two-sided Mills bracket below; above 6, the Mills bracket alone. `erf = 1 - erfc` exactly in `Fraction`. |
| Mills bracket | For `E(z) = int_z^inf e^(-t^2)dt` and `c_b(z) = z e^(-z^2)/(2z^2+b)`, differentiation gives `(E - c_b)' = -e^(-z^2)[2z^2(b-1) + b(b+1)]/(2z^2+b)^2`, and `E - c_b -> 0` at infinity. `b = 1` makes the bracket `+2 > 0`, so `E > c_1`: a **lower** bound valid for all `z > 0`. `b = 1 - 3/(2z^2)` makes it `d^2-3d-1 <= 0`, so `E <= c_b`: an **upper** bound valid for `z >= 1`. Hence `2z e^(-z^2)/(sqrt(pi)(2z^2+1)) < erfc(z) <= 4z^3 e^(-z^2)/(sqrt(pi)(4z^4+2z^2-3))`, relative width about `3/(4z^4)`. Full derivation in `_erfc_mills`. |
| `Phi`, `normal_sf` | `Phi(x) = (1 + erf(x/sqrt 2))/2`, `normal_sf(x) = erfc(x/sqrt 2)/2`. `Phi(0)` comes out exactly `[1/2, 1/2]`. |
| `normal_pdf` | `exp(-x^2/2)/sqrt(2 pi)`; the interior maximum at `x = 0` is carried by `Interval.__pow__`, which returns `[0, mag^2]` for an interval straddling zero. |

## What is coarse, and where — stated, not hidden

| Where | What happens | Why |
|---|---|---|
| `exp`, `normal_pdf`, `erfc`, `normal_sf` | `prec` is a **relative** width hint, not absolute. `exp(700, prec=30)` has width `3.1e+272`. | The remainder target is absolute on `exp(u)`, `\|u\| <= 1/2`; the `k` squarings restore the scale. |
| `erf`, `erfc`, `Phi`, `normal_sf` beyond `\|z\| = 6` (i.e. `\|x\| = 8.4853` for `Phi`) | width comes from the Mills bracket, relative `~3/(4z^4)`, and **`prec` has no effect at all** | the series is exact there but costs `~z^2` terms carrying `e^(z^2)`-sized rationals |
| `exp` past `EXP_BIT_LIMIT = 2**21` bits (`\|t\| > 1453635`; `\|z\| > 1205.6`; `\|x\| > 1705.1`) | underflow degrades to `[0, 2**-EXP_BIT_LIMIT]`; the tail **lower** bound becomes exactly 0; overflow raises `OverflowError` | an exact rational enclosure of `exp(t)` needs `1.4427\|t\|` bits of scale, which `round_out` cannot remove. `exp(-5e9)` would need 900 MB per endpoint and `exp(-5e11)` about 90 GB |
| `sin`, `cos` of an input a full period wide | `[-1, 1]` | correct and honest |
| `round_out(sig_bits)` | bounds the **significand**, not the **scale** | `round_out(64)` on an enclosure of `exp(-10**6)` still returns 1.44-million-bit endpoints |

Endpoint size in the Gaussian API is therefore bounded by
`EXP_BIT_LIMIT + O(prec)` bits — about 262 KB at the worst — and not by
`round_out`. A consumer accumulating many such endpoints in a lattice sum
should watch the magnitudes as well as calling `round_out`.

## Tests and negative controls

`tests/test_interval.py` — 75 tests. Run:

```bash
python3 -m pytest -q tests/test_interval.py
```

Eight **negative controls** each build a deliberately weakened variant inside
the test file (never by editing the library) and assert that containment then
FAILS: `sqrt` rounded inward; `exp` truncated one term early without widening;
`log` with the series tail dropped; the Machin remainder taken from the wrong
index; `sin`/`cos` reduced against a point value of `pi`; `x ** 2` without the
interior extremum; `sin`/`cos` without interior extrema; `erf` with a one-sided
alternating bracket.

The controls were verified by **mutating the library itself** — nine broken
copies built outside the repository, one weakening each — and confirming the
suite fails on every one. One honest limitation surfaced by that exercise: a
point value of `pi` taken at the *adaptive* precision `_pi_for` selects is
unjustified but its error stays below the returned width at the magnitudes
tested, so **no output test can detect it**. It is caught structurally, by
asserting the reduced argument is a non-degenerate interval.

### The 2026-09-18 adversarial audits

Four independent audits attacked this package: a containment audit with its own
exact-`Fraction` and `decimal` oracles, a rounding-direction audit, a
non-monotone-extrema audit, and a proof audit that refereed every remainder
bound as a paper. **None found a containment violation.** What they found, and
what was done, is recorded in group 6 of the test file, one test per defect,
each reproduced on the unfixed code before anything was changed:

| Reported | Status |
|---|---|
| unbounded memory in `exp`, reachable through `erf`/`Phi`/`normal_pdf`/`erfc`; `Phi(Interval(-10**5), 1)` died with `MemoryError` | fixed by `EXP_BIT_LIMIT`; `erf(Interval(10**4), 1)` went from 15.8 s to 0.07 s, `erf(10**5)` and `Phi(-10**6)` from impossible to 0.07 s |
| `repr` raised `ValueError` above 4300 digits, masking every guard and certificate alarm in the package | fixed; displays never raise, and the contracted `ZeroDivisionError` / `empty interval` messages now arrive intact |
| `_pi_for` used `len(str(v))`, so `sin`/`cos` raised `ValueError` at `10**4300` — in a bound path | fixed with a bit-length digit count |
| the tail bracket was one-sided: `Phi(x).lo == 0` for every `x <= -8.4853` at every `prec`, and `log(Phi(Interval(-9), 30), 30)` raised | fixed by the two-sided Mills bracket; `Phi(-9)` is now enclosed to a relative width of `4.6e-4` and the log-tail bound exists |
| `erf` tail endpoints were not size-bounded (144,269,749 bits measured) | bounded by the cap, and the tail branch now rounds its bound quantities outward |
| `prec` documented as absolute, relative for `exp` | documented per function |
| float membership, and `Decimal(float)`, undocumented | documented as refusals and pinned by tests |
| three containment-breaking mutants survived the whole suite (`__abs__` interior minimum; `sin` extremum sign at negative odd `j`; `erf`/`log` collapsed to the `x.lo` enclosure) | negative controls added for all three, plus the decreasing direction of `erfc`/`normal_sf` and the Mills lower bound with its `+1` dropped |
| the `j_hi - j_lo > 32` branch in `_sin_cos` is unreachable | kept (returning `[-1, 1]` is always sound) and documented as such; a test now measures the reduced-index span directly, so a regression shows up there rather than being masked |
| the `sin`/`cos` reduction docstring asserted `\|s\| <= pi/4` and `< 1` without the supporting inequality | the inequality is written out in `_sin_cos_point`, with `\|j\|` rather than `j`, and the margin (`1e-14` against `0.2146`) is stated. Containment does not rest on it: `_sin_cos_reduced` re-tests `mag(s) <= 1` and raises |

**What the fixes do not establish.** Nothing here promotes, closes, discharges
or reclassifies any claim, premise or obligation. A faster, better-bounded
arithmetic layer is not a certified result. `OBL-H5-JETMOD`, `OBL-H5-ZBAND`,
`OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE`, `D3-LEMMA-RN-UNIF`, `PERC-DECAY`,
`OBL-B1-BRANCH` and the `B4.loc` wrap/remote reconciliation stand exactly as
`docs/OPEN_PROBLEMS.md` records them. A certified enclosure of a Gaussian tail
is an enclosure of a Gaussian tail and of nothing else; in particular it is not
a bound on `1 - q(r)`, not the band version of the LPW bracket that
`OBL-H5-ZBAND` (A3) asks for, and not a lattice sum. The timings above are
measurements on one machine and are **NON-CERTIFYING**.

The float cross-check sweep (seeded, fixed literal `20260918`) is labelled
**NON-CERTIFYING** in its name and its docstring. Agreement with `math.sin` is
corroboration against gross implementation error. It is not a bound, it is not
a certificate, and no result in this program may cite it as evidence.
