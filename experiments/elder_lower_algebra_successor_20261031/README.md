# Exact successor check of the elder lower-bound algebra

**Object:** ELDER-LOWER-ALGEBRA-SUCCESSOR-01CZ-20261031. Credit 0 · scientific effect NONE · OBL OPEN.

**Why.** The direct elder lower bound `1 − p_r ≥ c r³` (Theorem E1) is the lower half of the
two-sided selection rate (E4) for the *actual* elder event. Its source is Math-
`frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md` (blob `aaefc8da`, at
Math- `19a55750`). That packet's own `algebra.py` (blob `00140088`) was blocked from publication
and survives only in the owner's ChatGPT conversation bundle; see
[`docs/retro/RETRO_SOURCE_RESOLUTION_01CZ_20261031.md`](../../docs/retro/RETRO_SOURCE_RESOLUTION_01CZ_20261031.md).
So nothing in the repositories could replay the algebra the proof rests on.

**What this is.** An independent re-derivation, written without seeing `algebra.py`. It is not a
copy of it, and agreement with it is unknown. In exact `fractions.Fraction` arithmetic, at
`k = 1` (every statement is homogeneous in `k > 0`), it checks:

| Source display | Checked |
|---|---|
| §2 cubic `G_k` | Pins at `(∓1/2, 0)` with heights `0` and `−1`; both gradients vanish |
| §3 Hessians | `diag(−6, −1/2)` at M (a maximum) and `diag(6, −5/2)` at S (index one) |
| (E5) | All path vertices lie in `B(0,3)`; the largest squared norm is `97/16` |
| (E6) | All three restricted polynomials, as exact identities |
| (E6) monotonicity | `g₁` nonincreasing, `g₂` constant, `g₃′ = (51−36v−6v²)/64 ≥ 9/64` on `[0,1]` |
| §2 margins | Path minimum `−7/32`, end value `17/64`; under a `1/64` perturbation, the path stays above `−15/64 > −1/4` and the end stays above `1/4` |
| (E3) | `d_f(M) − f(S) ≥ 3/4`, i.e. `3k r³/4` |
| (E7) | The targets `s = −3/2, a = 0, β = −2, c = 0` recover `G_1` exactly |
| Extra saddles | On `X = −3/4`, `Z² = 15/8`: height `−7/32`, index one, inside the path's second segment |
| README | Overlap integral `(3/10)k^{5/3}|t|^{−2/3}`, checked at exact cube points; `Beta(2/3, 2)` mean `1/4` and variance `9/176` |

```bash
python3 -I check_elder_lower_algebra.py          # one line, exit 0
python3 -m unittest -v test_check_elder_lower_algebra
python3 -O -m unittest test_check_elder_lower_algebra
```

The tests invoke the CLI. Six mutants (a wrong cubic coefficient, a wrong `g₃`, a tightened
`g₃′` bound, a widened perturbation, a wrong `β` target, a wrong overlap coefficient) must each
fail on their intended check.

## What this does not establish

- **None of the probabilistic steps of Theorem E1.** That includes the positive conditional
  density of the jet box, the conditional C⁴ moment, `W_r ≥ c₂ r⁴`, `Z_r ≤ C_Z r²`, and hence
  (E2) and (E8). A pass means the displayed algebra is right, not that the theorem holds.
- It does not show that (E1) is the actual elder death height. That identification is the
  subject of [`docs/ELDER_EVENT_TETHER_20261031.md`](../../docs/ELDER_EVENT_TETHER_20261031.md).
- It does not cover the upper half of (E4), any constant `c` or `r_*`, the density loss (E9)–(E11),
  or the pair-height correlation `2/7` and density `5/9` of the remote pair law.
- It does not reproduce, or attest to, the blocked `algebra.py`.
- It is Anthropic/Claude work (session 01CZ). The source is OpenAI work, so this is a
  different-family re-derivation, but organizational-independence credit is still 0.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
