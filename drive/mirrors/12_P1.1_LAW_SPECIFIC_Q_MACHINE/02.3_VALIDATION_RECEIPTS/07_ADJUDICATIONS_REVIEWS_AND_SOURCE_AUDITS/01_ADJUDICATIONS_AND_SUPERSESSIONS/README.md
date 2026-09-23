# `…/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/01_ADJUDICATIONS_AND_SUPERSESSIONS`

Drive folder id `1OZ1Fc0IpVAjh2aHfJgoIsUvubEXc8jXP`. The 2026-09-17 inventory gives this
folder **5 items** — 3 `application/json` and 2 `text/markdown` — and all 5 are held
byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_UBG_RESIDUAL_ADJUDICATION.json` | byte-exact | 1,755 |
| `C095_UBG_RESIDUAL_ADJUDICATION.md` | byte-exact | 1,730 |
| `C096_BR_MARK_LIMIT_ADJUDICATION.json` | byte-exact | 6,434 |
| `C096_BR_MARK_LIMIT_ADJUDICATION.md` | byte-exact | 1,688 |
| `C098_GAMMA_MAXCOUNT_SUPERSESSION.json` | byte-exact | 7,488 |

## The status banners, verbatim

`C095_UBG_RESIDUAL_ADJUDICATION.md` records an amendment of type CORRECTION and states its
finding:

> **Amendment type:** CORRECTION

> **Frozen C094 bytes:** untouched

> Those residuals were not inputs to the C094 arithmetic expression. Gate 14 therefore fails by construction.

Its status block carries three entries. The one this directory turns on reads

> uniform decimal 4.35:

> BLOCKED on UB_G_RESIDUAL_UNIFORM

The other two, not reproduced here, record the structural UB-G decomposition as derived
and the q0 limit from the cubic UB-G rate as proven modulo the same residual.

and `C095_UBG_RESIDUAL_ADJUDICATION.json` records the affected claims and the unblock condition:

> A hashed full-domain certificate for Gamma(r)/r^3 and Collar(r)/r^3, or a new conservatively assembled coefficient containing explicit residual budgets.

`C096_BR_MARK_LIMIT_ADJUDICATION.md` records

> **Jet reduction:** `DERIVED-EXACT`

> **Scaling evidence:** `MEASURED-DIAGNOSTIC`

> **BR-MARK:** `OPEN-RESHAPED`

and its adjudication paragraph says exactly what was killed and what was not:

> This does not kill Bonferroni. It kills the factorization `spatial repulsion constant × globally bounded mark-density constant` as a uniform architecture.

`C096_BR_MARK_LIMIT_ADJUDICATION.json` records the same finding as

> "naive_uniform_Gaussian_mark_density": "FALSIFIED ON TESTED PAIR-SCALED POINTS",

with growth

> "generic_density_growth": "approximately r^-4",

`C098_GAMMA_MAXCOUNT_SUPERSESSION.json` records its grades as

> "deterministic_event_inclusion": "PROVEN",

> "pair_Palm_Kac_Rice_identity": "DERIVED-EXACT",

> "station_atlas": "MEASURED-DIAGNOSTIC",

> "uniform_cubic_count": "PROVEN-MODULO three count obligations"

## What this does not establish

These five files record adjudications: decisions by one line about what its own earlier
objects do and do not prove. Every grade above is transcribed. **`MEASURED-DIAGNOSTIC` is
the sources' own word for a number that is not a proof**, and the falsification recorded in
the BR-MARK adjudication is falsification on *tested* pair-scaled points — a finite family
of evaluations, which the file names as such. Nothing was recomputed here, no obligation was
discharged, and no grade moved. A supersession recorded in a file supersedes nothing in this
repository, whose claim graph is governed by `claims/graph.json` and by no document in this
lane.
