# Session replay ledger — 2026-09-26 (xAI / Grok lane)

> **Supersession note (added 2026-09-27 at intake packaging; scientific effect NONE).**
> Files in this packet were written across several rounds on 2026-09-26. Rounds that
> predate 21:46:22Z say Math- PR64 is an unmerged draft whose erratum "404s on Math-
> default": `ERRATUM_CONGRUENCE.md` (header), `D1_INTERFACE_TABLE.md` (item 2),
> `A3_TYPE_CONVERGENCE.md` (requirement 1) and the "still open" D1 row below. Those
> statements were true when written and are now superseded by the record: Math- PR64
> was squash-merged at `d8f55054270532f2f95ae7cc7f5c613643d47e13` on
> 2026-09-26T21:46:22Z, and `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md`
> (1782 B, SHA-256 `bad7ef60…2028`) is on Math- default; `PROOF_INDEX.md` at
> `5ed3b455` already cites `d8f5505`, so `INVENTORY.md` line 46's "PROOF_INDEX still
> says … d573b99" is also stale. The later-round files (`INVENTORY.md`,
> `STATUS_PIN_NOTE.md`, `A3_UI_MAJORANT.md`, `A_M_TO_A0.md`) are the ones to read
> for custody. Intake is immutable after landing, so the earlier files are left as
> written rather than rewritten. Separately, `A3_UI_MAJORANT.md`'s "ACCEPT the
> structure" rests on parent §4 (A4), which `D1_INTERFACE_TABLE.md` rates as
> scaffolding, not closed; read it as conditional on A4. This note does not merge,
> accept, or promote anything.

**Scientific effect:** NONE.
**scientific_acceptance:** false (unchanged).
**lemma_closed:** false (unchanged).
**This packet does not edit** `STATUS.md` or `LANDING_CLAIMS.json`.

Owner asked for (1) an honest ranking of the mathematical work on the account and (2) an attempt to close what could be closed with live replay. This file records what was actually checked and what stayed open.

## Ranking recorded this session

1. **Flagship idea:** fold-cancellation short-lifetime law. Local max / index-(d-1) saddle contact measure collapses to `r dr` in every ambient dimension checked (`d = 1..7`). Cubic Morse gap `ell ~ r^3` pushes that measure to `nu(ell) ~ c ell^{-1/3}`.
2. **Most complete calculation:** SIDE24 coefficient enclosure on the periodized Bargmann–Fock field of side 24.
3. **Most self-contained theorem:** P15 Theorem F on the realized `d_i >= 2` family, with the demand-one extension kept dead.
4. **Process, not a theorem:** fail-closed museum / STATUS / source-identity discipline.

Older RUFL / RH-from-action / killed `gamma ~ D2` TDA bridge are **not** the strongest objects on the account.

## Engineering checks that passed

### SIDE24 (`Math-/coefficients/side24_v1`)

- Public tests: **30/30** in 1.073s.
- Cone moments re-derived:
  - `D_1 = 4/3` (half of `Var(A) = 8/3` by Gaussian symmetry).
  - `D_2 = 29/6 - sqrt(6)` via `8 sqrt(3/8) = 2 sqrt(6)`.
- Reference formula used:

```
c_{d,ref} = Gamma(7/6) * (3/2)^{1/3} * D_{d-1} / (2 * sqrt(3) * pi^{d-1} * sqrt(pi))
```

- mpmath (`dps = 80`) reference values sit inside the public intervals:
  - `c2 = 0.07340691930603427103013...` in `(0.07340691930603427103, 0.07340691930603427104)`
  - `c3 = 0.04177593184059834334293...` in `(0.04177593184059834334, 0.04177593184059834335)`
- IEEE float can sit ~`10^{-17}` outside those 20-decimal walls. That is why the package uses outward rational arithmetic. Do not treat a binary float print as a counterexample to the enclosure.
- Persistence reading of the coefficient remains **conditional on open D1 / main#63**.

### Lemma R3.2 (filtered-determinant / quadratic off-diagonal cancellation)

For

```
K_t = [[alpha, sqrt(t) beta^T], [sqrt(t) beta, A]]
```

the determinant identity

```
det K_t = alpha det A - t beta^T adj(A) beta
```

holds including singular `A` (checked symbolically for block size 1, 2, 3). Random numerical checks of the filtered-determinant bound on `n = 2,3,4` matrices: **0 failures**, max `lhs/bound = 1.000`. The linear-algebra core is real and sharp.

This does **not** close Theorem R. Theorem R still imports marked Kac–Rice, the full normalizer `Z`, and the elder convention from D1.

### P15 Theorem F and demand-one

- Public tests: **36/36** in 3.53s.
- Displayed constant `rho_* ~ 0.84547981724898672067` is correct for

```
h_* = -log(3 e^{-2} - 2 e^{-3}) = 3 - log(3e - 2)
rho_* = 1 / h_*
```

- `rho_* < 6/7` holds.
- Demand-one counterexample inequalities hold: `phi(1/2) = log 2 > 2/3` and cover cost `4/9 > log(4/3)`.

#### ASCII disambiguation (additive erratum, not a kill)

`PROOF.md` writes `3 - log(3e-2)`. That identity is **true** if `3e-2` is read as `3*e - 2`. It is **false** if `3e-2` is read as `3 e^{-2}` (that reading produces `h ~ 3.901`, `rho ~ 0.256`, which is not the displayed window).

Action: disambiguate the ASCII in the P15 source (write `3e-2` as `3*e - 2`, the intended reading). Do **not** change the numerical window and do **not** flip Theorem F.

## Still open — do not invent closures

| Object | Blocker |
|---|---|
| D1 Theorem A §2–7 | Parent selection / congruence / residual jet / full normalizer / elder interfaces still reopened on main#63. Congruence erratum file was not consumed. |
| D5 pin / microdisk | Inner-disk bound, collar to reviewed annulus, shrinking-collision factorial moments missing (Math-#58). Draft PR82 already says no `O(r^3)` lemma. |
| SARD-G A1/A6 | A1 still AMEND: chart membership must use relative-interior first hits and exclude endpoint contact (main#122). |
| D0 historical carriers | `rnu_env.py`, `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, `allcell_fdz_enclosures.json` remain ABSENT. |
| Lifetime reading of SIDE24 | Still HOLD_WITH_DOMAIN on parent Eq. 15.2. |

## Authorship

Landed proof files for D2/D3/D6 still name Author: OpenAI / ChatGPT. This packet is a nonauthor engineering/analytic replay by the xAI/Grok lane under owner instruction. Multi-model agreement is not independent journal refereeing.

## Next closure that would raise status

A source-bound independent review of D1 §2–7 that either accepts the parent selection chain at a named scope or replaces it. Until then the flagship object stays the most interesting unfinished object on the account.
