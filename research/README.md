# Research

Working code and per-lane indexes. Content mirrored from the Drive keeps its
source identity and status; anything added here is clearly marked as new.

| Path | What it is |
|---|---|
| `rn/moment_envelope.py` | exact-rational reimplementation of the correct Hölder(4,4,2) determinant-moment envelope, the defective `envelope_v` expression, and the RN5 typed Gaussian counterexample that separates them |

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
Piece 2 of `D3-LEMMA-RN-UNIF`, and RN3's far-region proof is outside the
affected scope — it uses the correct second moment.
