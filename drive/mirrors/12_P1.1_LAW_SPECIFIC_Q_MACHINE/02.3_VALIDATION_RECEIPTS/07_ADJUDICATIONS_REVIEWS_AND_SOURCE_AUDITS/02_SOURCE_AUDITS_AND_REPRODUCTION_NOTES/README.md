# `…/07_ADJUDICATIONS_REVIEWS_AND_SOURCE_AUDITS/02_SOURCE_AUDITS_AND_REPRODUCTION_NOTES`

Drive folder id `11YnKlKNv-a_rCJQ7_WVeWllSZME8lFc_`. The 2026-09-17 inventory gives this
folder **2 items** — one JSON and its markdown companion — and both are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C096_UBG_RESIDUAL_SOURCE_AUDIT.json` | byte-exact | 3,116 |
| `C096_UBG_RESIDUAL_SOURCE_AUDIT.md` | byte-exact | 3,680 |

## The status banners, verbatim

The markdown audit's header reads

> **Grade:** `DERIVED-EXACT-IMPLICATION-AUDIT`

> **Source status affected:** fixed-volume upper-rate display and the q₀ corollary

> **Frozen source files:** unchanged

Its central finding is that a positive constant cannot be absorbed into a cubic:

> A positive constant does not vanish when \(r\to0\).

and it records the exact crossover below which the absolute ceiling cannot be absorbed:

> Thus even ignoring the collar, the absolute station ceiling cannot be absorbed uniformly on a neighborhood punctured at zero.

Its final disposition block carries six entries. Two of them read

> uniform Gamma O(r^3):

> OPEN — GAMMA-DENSITY

and, after an entry for the collar that is not reproduced here,

> uniform decimal 4.35:

> BLOCKED

The four not quoted are the absolute Gamma station ceiling, the collar entry that sits
between the two above, the q(r) limit from UB-G, and the block's closing line. Read the
stored file for the block in full; nothing here is the whole of it.

and it closes

> The q₀ conclusion may still be true. The printed \(e^{-92}\) ceiling is not the theorem that proves it.

`C096_UBG_RESIDUAL_SOURCE_AUDIT.json` records the same adjudication in fields, including

> "q0_from_source_as_printed": false,

with the reason

> "No positive constant exp(-92) can be absorbed into C r^3 on the full punctured interval 0<r<=r0."

and the terminal status

> "absolute_e_minus_92": "station/rung-scale diagnostic ceiling, not a uniform r^3 coefficient on a punctured neighborhood of zero"

## What this does not establish

This is a negative result about an argument, recorded by the line that made the argument.
It says that a printed inequality does not imply what it was read as implying; it does not
say the conclusion is false, and the source says so itself in the line quoted above. Nothing
here was recomputed. The exact decimals in both files are the source's arithmetic, not this
repository's: no digit was checked, and `research/interval/` — the only route in this
repository through which a bound may be called certified — is not involved in any way.
