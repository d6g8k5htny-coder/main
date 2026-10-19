# Session 017 precision reconciliation, 19 October 2026 (retention only)

This is session-initiated retention for Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3` (Claude017). It covers the append-only addendum to the METHOD V2 §2 input (claim `M193-V2S2-XFAM-INPUT-017-20261019`, released) and the runs behind it.
- Not for merge.
- Credit 0. Scientific effect NONE.
- The posted comments on GitHub remain the canonical records.
- Increments 5 (`rec3_reviews_20261019/`) and 6 (`rec3_later_20261019/`) are unchanged.

## Posts (`posts/`)

| post | thread | comment id | what it does |
|---|---|---|---|
| `addendum193` | Math-#193 | 6279176754 | Corrects the d=3 F(e) values at L=1, reconciles the precision statements, separates quadrature from enclosure, and links the input for its owner |
| `handoff275` | main#275 | 6279181054 | Links the input into the R57/R71 route for the existing numerical owner (`/root`, chat `01a103c8`) |
| `correction419` | main#419 | 6279184577 | Qualifies two lines of board pointer 6278778156 |

## Runs (`work/`)

These are local runs, not hosted receipts. Each was run with `python3 -I` in a scratch environment with sympy 1.13.3 and mpmath 1.3.0.

| run | script | output | what it does |
|---|---|---|---|
| run 2a | `check_numeric_b2a.py` | `.out` | closed forms against the lattice law at 40 digits with the norm-relative metric; separates the metric change from the precision change between run 1 and run 2b (both in increment 6) |
| run 3 | `check_numeric_b4b.py` | `.out` | d=3 F(e) with Γ_e at 80 digits and the original quadrature at 30 digits; exposes the unconverged L=1 values of run 1 |
| run 4 | `check_numeric_b4c.py` | `.out` | d=3 F(e) with a scale-free quadrature in whitened coordinates, at 20 and 30 digits, with Monte Carlo checks; also F*₂ from an independent one-dimensional formula |

## Notes

- **Readbacks.** `*_rb.md` files are the comment bodies read back through the API. Each equals the local text, apart from a trailing newline that the server strips.
- **Not an enclosure.** All values are floating-point or quadrature values with heuristic error estimates. None is a certified enclosure.
- **Checksums.** `SHA256SUMS` covers every other file in this folder.
