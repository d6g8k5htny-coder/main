# Research

Working code and per-lane indexes. Content mirrored from the Drive keeps its
source identity and status; anything added here is clearly marked as new.

| Path | What it is |
|---|---|
| `rn/moment_envelope.py` | exact-rational reimplementation of the correct Hölder(4,4,2) determinant-moment envelope, the defective `envelope_v` expression, and the RN5 typed Gaussian counterexample that separates them |
| `interval/` | certified interval arithmetic on exact rational endpoints: containment unconditional, tightness best-effort; four-lens adversarial audit and mutation-tested |
| `lpw/headline.py` | the LPW headline constant as an exact fraction and the admissibility of each rounded decimal by asserted direction |
| `identities/gaussian_moments.py` | Gaussian absolute-moment identities exact in the basis `{1, 1/π}`; refutes `E(\|ξ\|+\|η\|)⁴ = 12 + 16/π` (true value `12 + 32/π`) |
| `bands/` | lane A1 machinery for `OBL-H5-JETMOD`: the published rung ladder as exact data with the implied-modulus analysis, a certified periodized lattice-sum evaluator with a proved tail bound uniform over an interval box, and the obligation's own falsifier with `INSUFFICIENT_DATA` as a first-class outcome. Reference kernels only — no band bound for any jet of the program is computed |
| `cover/` | lane A5's spatial cover driver, the one `LANE_RN_UNIF.md` records as unwritten: partition exactness decided structurally over `Fraction` (never by area — an equal-area cover can carry both a gap and an overlap), `total()` raises while any cell is `PENDING` with no bypass, every rejected cell retained with its boundary bound, and the RN5 annulus `0.1 ≤ \|y\| ≤ 5` and the T4 polar region `d ∈ [5, 17]` instantiated separately and never merged. Reference integrands only — no cell of the program's actual cover is certified |
| `slack/` | the bound-slack registry: seven sourced records with claimed and attained values as certified intervals, slack ratios computed exactly, and a severity ladder anchored to the program's own constants. Utility and correctness are separate axes from disjoint data — `known_unsound` is reachable only through a witness the module re-computes — and the registry's own data proves why: its tightest record is unsound, its loosest is not known to be. Measures; repairs and refutes nothing |

See `docs/RESEARCH_MAP.md` for the lane-by-lane map and
`docs/OPEN_PROBLEMS.md` for the exact next actions.

## The RN5 defect, and why it is in CI

On 2026-09-17 the near-region certification route was found to rest on a wrong
determinant moment: `envelope_v` returned

```
(E A⁴ · E B⁴)^(1/4) · (E C⁴)^(1/2)
```

where Hölder with exponents (4, 4, 2) gives

```
E[ |ABC| · 1{M max} · 1{S saddle} · 1{y saddle} ]  ≤  (E A⁴ · E B⁴)^(1/4) · (E C²)^(1/2)
```

For small determinants the substitution *lowers* the value, so the defective
expression is not an upper bound at all. `research/rn/moment_envelope.py`
reproduces the counterexample from the moment formulas, independently of RN5's
reported numbers, and gets the same values:

| Quantity | Value |
|---|---|
| `P(event)` | ≥ 91/100 |
| typed expectation | ≥ 0.207675035568 |
| defective `envelope_v` | ≈ 0.062504062490 |
| correct Hölder(4,4,2) | ≈ 0.250004625009 |

`tests/test_rn_moment_envelope.py` asserts the separation with exact rational
comparisons of fourth powers, so the wrong-power bound cannot be reintroduced
without a red build. This verifies the counterexample only. It does not close
Piece 2 of `D3-LEMMA-RN-UNIF`.

**RN3 and the affected scope, precisely.** RN3's *far-region proof* is outside
the affected scope — it uses the correct second moment. That holds for the
far-region proof and for nothing else in the object. §9's conditional arithmetic
imports the near target `17.6804 r³`, which the RN5 repair places inside or
immediately adjacent to the affected scope (corrected diagnostic ≈ `2.34195 r³`
against wrong-power ≈ `17.67237 r³`, a factor of about 7.5), so the displayed sum
`I_near + I_far < 20.51352 r³` rests on a number the erratum touches. Read the
scope sentence as covering the whole object and you carry `20.51352 r³` out of
scope by mistake. Recorded in
[`reviews/records/REV-RN3-FARZONE-20260918.json`](../reviews/records/REV-RN3-FARZONE-20260918.json).
