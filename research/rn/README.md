# `research/rn/` — the RN-UNIF lane's exact-rational modules

Lane **A5**, `D3-LEMMA-RN-UNIF`. Three modules, all standard library only,
exact in `fractions.Fraction` wherever a bound is claimed, with certified
enclosures taken from [`research/interval/`](../interval/README.md).

| module | what it is |
|---|---|
| `moment_envelope.py` | the correct Hölder(4, 4, 2) determinant-moment envelope and the RN5 counterexample, so the pinned `envelope_v` defect stays falsifiable in CI |
| `hermite_envelope.py` | the two quantities the frozen `env_form` multiplies: the Hermite envelope `he_abs` in exact rationals, and `kern` as a certified enclosure |
| `env_form_reference.py` | the `env_form` shape assembled at orders 0–4 over certified intervals, on labelled **reference** moment data |

Tests: `tests/test_rn_moment_envelope.py`, `tests/test_hermite_envelope.py`,
`tests/test_env_form_reference.py`.

## Why an exact treatment of `env_form` is possible at all

`env_form(k, gamma, d, qord)` in the frozen body
(`engine/rn_engine/frozen/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py`
line 381) returns three additive parts, and each is **one exact rational
coefficient times one Gaussian factor**:

```
tot = C_tot(moments, gamma, d, q) * exp(-d²/2)
rem = C_rem(forms,   gamma, ρ, q) * exp(-ρ²/2)      ρ    = d − R/2
img = C_img(forms,   q, r_img)    * exp(-r_img²/2)  r_img = 24 − d − R/2
```

Every `C` is built from `he_abs`, factorials, binomials and the moment or
coefficient magnitudes. And `he_abs` is an **integer-coefficient polynomial**:
the frozen body tabulates it for `n ≤ 10` and generates `11 ≤ n ≤ 19` from an
all-plus recurrence, and the two agree. So the only transcendental anywhere in
the envelope is `exp`, which `research/interval/` certifies.

The consequence is that each part is enclosed with **no dependency widening at
all** — one rational, one exponential — and the measured relative width of the
total at `d = 5`, `prec = 60`, is about `5.6e-61`.

### The envelope identity, and its proof

`he_abs(n, ·)` is exactly the coefficient-wise absolute value of `He_n`, not
merely some upper bound. Writing `He_n = Σ_k c_k x^k`,

```
    Σ_k |c_k| t^k  =  (−i)^n He_n(i t)
```

because `He_n` carries only degrees `n, n−2, n−4, …`, the coefficient of
`x^(n−2j)` has sign `(−1)^j`, and substituting `x = it` turns every sign into
`i^n`. Nothing cancels, which is also why the two contributions to any one
coefficient in the all-plus recurrence never cancel. Containment is then the
triangle inequality:

```
    |He_n(x)|  ≤  Σ_k |c_k| |x|^k  =  he_abs(n, |x|)
```

The bound is tight where the terms share a sign and loose near a zero of
`He_n`. The frozen body's own docstring calls it "the rigorous envelope".

`tests/test_hermite_envelope.py` pins the identity three ways — against the
frozen explicit table, against `abs` of the exact coefficients, and against
`(−i)^n He_n(it)` computed in exact Gaussian rationals — and five negative
controls assert that containment FAILS when the envelope is weakened.

## Two observations about the frozen body, recorded and not acted on

Neither is a defect report against any claim, and no status turns on either.
A frozen body is never edited in place.

1. **`mx2` index sum.** The comment on line 408 reads `|beta|=9`; the loop on
   lines 409–410 runs `for d1 in range(12): d2 = 11 - d1`, so its index pairs
   sum to **11**. `env_form_reference.py` reproduces the code, because the code
   is what produced the frozen numbers, and records the disagreement in
   `MX2_INDEX_SUM`. A control asserts the two choices give different answers,
   so the fork is known to be real rather than harmless.
2. **`R` is a binary double.** The frozen `R = mpf('0.05')` is the double
   nearest `0.05`, not exactly `1/20`. This module carries `REFERENCE_R = 1/20`
   exactly and says so; a control asserts an inexact `R` changes the
   coefficients.

## Why there is no whitened `env_form` here

`docs/OPEN_PROBLEMS.md` §A5 carries, from RN5: "Build the whitened `env_form`
orders 2–4 runnable smoke (currently missing); order-1 chi-squared white is a
nearest neighbour only."

Whitening itself is unambiguous — a `y`-independent frame change `W` with
`W S₀ Wᵀ = I`, built in the order-1 carrier as `W[k, j] = V0[j, k] / √LAM0[k]`.
What a *whitened `env_form`* is, is not. The sources give two incompatible
readings:

* the only build that ever realised it computed `env_form(k, γ, d, q)/√LAM0[k]`
  for `q ∈ {2, 3, 4}` at `d = 5`, marked its own result **"PROXY only"**, and
  wrote "This is **not** the missing whitened residual-covariance construction
  for log q(A, m′)";
* and the author-side jet theorem refuses exactly that reading: "Norms of all
  covariance derivatives refer to their actual ordered Schur/inverse
  construction, not to a residual form divided by √λ."

Both are transcribed verbatim, with their paths and line numbers, in
`engine/lanes/A5.json` under `additional_source_directives`. Choosing between a
reading one source calls a proxy and a reading a later source refuses is a
mathematical decision, and this repository does not make those. So
`env_form_reference.py` builds the shape and stops at the whitening, and
`research/rn/env_form_smoke.py` does not exist. A control
(`test_the_module_refuses_to_whiten`) fails if a future edit adds a
`1/√λ` scaling, so such an edit has to argue for itself.

Both quoted documents sit in the Drive lane whose name is its own status word —
`2026-09-16 — HOLD_NOT_FOR_SUBMISSION` — and are cited as evidence of what the
sources say, never as authority for what is true.

## What this directory does NOT establish

* It **closes, discharges, promotes and reclassifies nothing.**
  `D3-LEMMA-RN-UNIF` Piece 1 and Piece 2 are **OPEN** exactly as
  `docs/OPEN_PROBLEMS.md` §A5 records them, and the lane receipts still carry
  `lemma_closed: false`. `OBL-H5-JETMOD`, `OBL-H5-ZBAND`,
  `OBL-H5-REMOTE-THRESHOLD`, `OBL-D1-PROMOTE`, `PERC-DECAY`, `OBL-B1-BRANCH`
  and the `B4.loc` wrap/remote reconciliation stand as recorded.
* It **is not the missing whitened smoke**, and having an exact arithmetic for
  the shape is not having the artifact.
* It **re-certifies nothing**. The frozen engine's numbers are `mpmath` at
  `mp.dps = 100` and remain as certified — or as uncertified — as their own
  sources say. That a certified arithmetic for the same shape now exists here
  relabels none of them.
* It holds **no moment table and no form of the program**. `MOMS[k]` and
  `FORMS[k]` are frozen-engine six-pin geometry. Everything exercised here is
  labelled REFERENCE data, and **no envelope of the program's actual residual
  forms is computed here, at any order.**
* It supplies no cell supremum, no band enclosure, no rung, no coverage
  certificate and no remote budget; it composes no two of the three tracks; it
  bears on no prize problem.
* Green tests establish that the enumerated mutations are caught. The
  containment arguments are ordinary mathematics written out in the docstrings
  for a human to check, and passing tests are not that check.
