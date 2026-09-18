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
from research.interval import Interval, sqrt, exp, log, sin, cos, pi, erf, Phi, normal_pdf
```

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
`pi`, `erf`, `Phi`, `normal_pdf`.

Deliberate refusals, each part of the contract:

* `float` endpoints raise `TypeError`. `Fraction(0.1)` is not `1/10`.
* Division by an interval containing zero raises `ZeroDivisionError`. There is
  no extended-interval fallback.
* `x ** 0 == [1, 1]` for every `x`, including intervals containing zero.

## How each enclosure is certified

Each is proved in the function's own docstring; in one line each:

| Function | Certificate |
|---|---|
| `sqrt` | integer Newton `floor(sqrt(p*q*N^2))`, endpoints rounded **outward**, then `lo^2 <= a` and `hi^2 >= a` **re-verified exactly in `Fraction` on the endpoints actually returned**. That check is the certificate; it is kept in the code on purpose. |
| `exp` | reduce `exp(t) = exp(t/2^k)^(2^k)` to `|u| <= 1/2`; Taylor with `\|R_n\| <= \|u\|^(n+1)/(n+1)! * 1/(1 - \|u\|/(n+2))`, derived from `(n+1+i)! >= (n+1)!(n+2)^i`; `k` squarings, monotone on non-negatives. |
| `log` | `a = m*2^k`, `m ∈ [1,2)`; `log m = 2 atanh((m-1)/(m+1))`, positive terms, tail `<= z^(2j+3)/((2j+3)(1-z^2))`; `log 2` gets its own certified enclosure. |
| `pi` | Machin `pi/4 = 4 atan(1/5) - atan(1/239)`, alternating-series remainder on each `atan`. |
| `sin`, `cos` | reduction into `[-pi/4, pi/4]` **against the certified `pi` enclosure**, interval-valued so the uncertainty in `pi` becomes width; alternating series, remainder = first omitted term (justified: ratio `<= 1/6` for `\|s\| <= 1`); interior extrema at multiples of `pi/2` added whenever their enclosure meets the input; `[-1, 1]` when the input is a full period wide. |
| `erf` | Maclaurin alternating series for `\|z\| <= 6`, truncated only at an index `n >= floor(z^2)+1` where the terms are provably decreasing; above that, the monotone bracket `[1 - e^(-z^2)/(z sqrt pi), 1]` from `t/z >= 1` on `t >= z`. |
| `Phi` | `(1 + erf(x/sqrt 2))/2`. `Phi(0)` comes out exactly `[1/2, 1/2]`. |
| `normal_pdf` | `exp(-x^2/2)/sqrt(2 pi)`; the interior maximum at `x = 0` is carried by `Interval.__pow__`, which returns `[0, mag^2]` for an interval straddling zero. |

## Tests and negative controls

`tests/test_interval.py` — 52 tests. Run:

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

The float cross-check sweep (seeded, fixed literal `20260918`) is labelled
**NON-CERTIFYING** in its name and its docstring. Agreement with `math.sin` is
corroboration against gross implementation error. It is not a bound, it is not
a certificate, and no result in this program may cite it as evidence.
