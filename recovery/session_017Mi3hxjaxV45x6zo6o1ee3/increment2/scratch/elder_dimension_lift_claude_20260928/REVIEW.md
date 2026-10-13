# Nonauthor analytic review: the elder lower event in every fixed dimension (Math-#125)

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, `STATUS`, `PROOF_INDEX`,
`GRAPH`, landing claims or any author source. It records verdicts on the OpenAI candidate below. Integration is a
separate act.

## Object

| Field | Value |
|---|---|
| Object | OA-ELDER-DIMENSION-LIFT-20260928-v1 (OpenAI / ChatGPT) |
| Branch / head | `chatgpt/elder-dimension-lift-20260928` / `e3f10b390ae5ddd998d238206add5958d5b720bb` ([Math-#125](https://github.com/d6g8k5htny-coder/Math-/pull/125)), base `d39dbbc` |
| `PROOF.md` | Git blob `7303bd791a68a1139251f0f6e403a9f7cc89b006`, 23971 B, SHA256 `b529fe3780e1909014b3cb144b9a74156d976bbdc19b6711d0c5b1a1da9e5669`, 480 lines |
| Consumed, already on main | [LP] `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` (blob `dfed3b8d…`): §1 selector, §§2–4 spectrum, contact frame and regression, §8 maximin; Theorem A and §§9–12 for the corollaries only. [EL] `frontiers/window_multiplicity_laws_20260928/ELDER_LOWER_AND_DENSITY_GAP.md` (blob `aaefc8da…`): the planar cubic, path and d = 2 case. [LOCAL] `LOCAL_MULTIPLICITY.md` (blob `b5c18aac…`): planar saddle data. |

## Provenance

| Field | Value |
|---|---|
| Reviewer | Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3` |
| Relation | Different provider from the OpenAI author. Same GitHub account as every lane, so no organizational independence is claimed. |
| Exposure | I read `PROOF.md` in full, plus the packet README, SOURCE_MAP, RECONNAISSANCE, `lift.py`, `test_lift.py` and `verify.py`. I wrote the nonauthor review of [EL]/[LOCAL] (`reviews/d1_elder_lower_claude_20260928/`), the #106 review of LP Theorem A, and the D1 chain reconciliation (Math-#126, open). Corollary B's upper half consumes LP Theorem A, whose reconciliation I authored, so for that half this review is **not** independent of its inputs. The direct lower Theorem A (A2)–(A4) consumes none of those objects except [EL]'s unchanged planar cubic and path, which this review re-derives. |
| Independent derivation | Before seeing #125 I had planned a d ≥ 3 lower proof by a nonlinear ridge reduction `y = φ(x, z)` in the stable directions. Its reduced midpoint Hessian is the same Schur complement `σ = f_zz − f_zw f_ww⁻¹ f_wz`. #125's linear shear reaches the same rare set with less machinery (N3), so I review #125 instead of writing a competing candidate. |
| Author code | I ran `verify.py --output` in `-B -S` and `-B -O -S` on the exact head: 18 tests per mode, 10 intended mutant failures, identical output pairs. Source identities match. |
| Independent checks | `dimension_lift_review_check.py` is my own suite, written from `PROOF.md` without reusing `lift.py`. It checks finite identities exactly, plus one labelled floating-point group. Every Gaussian step below was argued by hand and is not replaced by those checks. |

## Verdicts

| Interface (PROOF.md lines) | Verdict |
|---|---|
| §2 (98–130): complete 3-jet `(U_0, J)` of size `C(d+3,3)`, positive definite in every frame; uniform floor for `(U_r, J)`; raw-jet density floor under `Q_r` | **ACCEPT**; see §1 below. |
| §3 (132–187): Schur shear (A8)–(A9), unit-Jacobian raw chart (A10) | **ACCEPT**. Exact check `CHART`, including the full Jacobian by dual numbers. |
| §4 (189–216): raw rare set (A11), volume (A12), `Q_r ≥ c_J r` (A13) | **ACCEPT**; see §2 below. |
| §5 (218–246): `C^4` control after conditioning on all physical jets (A14)–(A16); constant order | **ACCEPT**; see §3 below. |
| §6 (248–282): sheared-plane normal form (A17), pin identities, target (A18) | **ACCEPT**. Hand-derived. Exact on physical cubic fields in d = 3, 4, 5 (zero remainder); `O(r)` remainder on quartic fields. |
| §7 (284–308): path (A19), margins `k/64`, `d_f(M) ≥ b − kr³/4 > f(S)` | **ACCEPT**. Identical to [EL] (E5)–(E6); recomputed. |
| §8 (310–361): endpoint blocks (A20), Schur (A21), inertia and determinant (A22), `W ≥ c r⁴` (A23), `Z ≤ C r²` (A24), ledger (A25) | **ACCEPT**; see §4 below. |
| **Theorem A (A2)–(A3)**: `1 − p_r ≥ c r³` for every fixed d ≥ 3 (d = 2 is [EL]) | **ACCEPT** at its stated existential scope. |
| §9 (363–415), **(A4)**: two extra index-(d−1) critical points, `Q^W(N_r ≥ 2) ≥ c r³`, factorial lower bound | **ACCEPT**; see §5 below. |
| **Corollary B (A5)**: `c r³ ≤ 1 − p_r ≤ C r³` in every fixed d | **ACCEPT** as a composition with LP Theorem A (1.1), consumable at existential scope under the D1 reconciliation (Math-#126, open); see §6 below. |
| **(A6)**: compact-mark loss `ν_cand − ν_eld = Θ(ℓ^{2/3})` in every fixed d | **ACCEPT** as a composition with LP's compact-mark ledger and (A5). |
| **Corollary C (A6a)**: rejected-pair inverse moments finite iff `p < 5/3`; `Θ(t^{5/3−p})`; logarithmic endpoint | **ACCEPT** as a composition with (A6). |

No defect was found. Notes N1–N5 record observations that sharpen or delimit the text without changing it.

---

## §1 — the jet floor (§2 of PROOF.md)

The count is exact. `U_0` has `4 + 2(d−1) = 2(d+1)` entries (lines 101–105), one per pin observation. `J` has the
`m(m+1)/2` transverse Hessian entries (`m = d − 1`) and the `C(d+2,3) − 1` third derivatives other than `f_xxx`.
Order by order, `U_0 ∪ J` is every derivative of order 0–3 exactly once. Order 2 has `d` entries in `U_0` (`f_xx`,
`f_xy_j`) and `m(m+1)/2` in `J`, total `d(d+1)/2`. The total is `C(d+3,3)` (`LEDGER`, d = 2…12).

Positivity: the periodized covariance has Fourier weights `∝ exp(−|ω|²/2) > 0` on `(2π/L)Z^d`. A linear functional
`Σ c_α ∂^α f(0)` has variance `Σ_ω w_ω |Σ c_α (iω)^α|²`. If that variance is zero, the polynomial `Σ c_α (iω)^α` vanishes
on the dual lattice, rotated by the frame. Pulled back, it vanishes on `Z^d`, hence identically. So the complete jet
covariance is positive definite for each frame. Compactness of `O(d)` and continuity make the floor uniform.
`CHART.rotated_lattice_unisolvence_degree3` illustrates this exactly on Cayley-rational rotated lattices for d = 2, 3.

`U_r → U_0` uniformly as `r → 0` is LP §§2–4. The quartic group re-solves the `2(d+1)` U_0 jets from the pins through
that frame and finds it nonsingular. Conditioning on `U_r = v_r` is Schur complementation, so the conditional covariance
of `J` is sandwiched between the joint floor and ceiling, and its mean `Cov(J,U_r)Cov(U_r)⁻¹v_r` is bounded. A Gaussian
density with sandwiched covariance and bounded mean is bounded below on every fixed compact set. The claim is about
**raw** jets (line 129), which is what §4 integrates.

Note on `J` being free under `Q_r`: the pins tie `f_y(0)` to `f_xxy` at order `r²`, since
`f_y(0) = −(r²/8) f_xxy + O(r³)`. That relation lives inside the `U_r` rows, not as a constraint on `J`. The
`PHYSICAL` group builds fields with arbitrary `J` and verifies all `2(d+1)` pins exactly.

## §2 — the chart and the rare set (§§3–4)

With `q = D⁻¹v`, `y = (z, w − qz)` gives
`yᵀAy = az² + 2z vᵀ(w − qz) + (w − qz)ᵀD(w − qz)`. The `zw` terms cancel because `qᵀD = vᵀ`. The `z²` coefficient is
`a − 2vᵀD⁻¹v + vᵀD⁻¹v = σ`. So `SᵀAS = diag(σ, D)` and `det A = σ det D` (checked exactly, n = 1…4). The chart
`(σ, v, D, τ) ↦ (a, v, D, T)` is block triangular in the order `(v, D), σ, τ`, with diagonal blocks `I`, `1` and the
substitution `w ↦ w + qz` on third-derivative coordinates, which is unipotent.

`f_xxx` is untouched, and no other coordinate depends on it: `∂_z` of the sheared field only introduces `∂_w`, which
preserves the number of `x` indices. So deleting `xxx` leaves determinant one. My check does not rely on the triangular
argument. It differentiates the whole map with exact dual numbers for d = 3 (12 variables) and d = 4 (25 variables),
and finds a full Jacobian determinant of exactly 1.

The rare box (A11) is a product: an interval of length `2εkr` in `σ`, fixed boxes in `v` and `D`, and width-`2εk` boxes
in the `q_3 = C(d+2,3) − 1` coordinates of `τ`. Its raw volume is therefore (A12). All inverse-chart images lie in one
compact set, because `D⁻¹`, `q` and `σ` are bounded. So (A13) follows from §1. The raw entry `a` sits on the `O(1)`
surface `vᵀD⁻¹v`, inside a band of width `O(r)`. Only one scalar shrinks. That is why the power is `r¹` in every d
(line 185). Requiring all `m(m+1)/2` transverse entries to be `O(r)` would give `r^{m(m+1)/2}` instead. No law is
assigned to the sheared variables (lines 213–216): the raw Gaussian density is integrated against a unit Jacobian.

## §3 — conditioning order and constants (§5)

The field is regressed on `V = (U_r, J)` at each target. The pin rows are integral averages of derivatives of order
≤ 3, so the cross-covariances of the field's derivatives through order 4 with `V` are uniformly bounded. `Cov(V)⁻¹` is
bounded by §1, and the targets are compact. Hence (A15) holds uniformly. Markov's inequality gives `K_4` with conditional
mass ≥ 1/2 at every target, and integrating over the rare set gives (A16). No independence between jets and remainder is
used, and no unconditional tail is subtracted from an `O(r)` mass.

The constant order is sound. The raw container is valid for all `ε ≤ 1`, so `K_4` is chosen first. `ε` is then chosen
for the planar `O(εk)` margins, which do not involve `K_4`. Finally `r_*` absorbs every `O(rK_4)` and `O(rK_4²)` term.

## §4 — normal form, endpoints, weight and normalizer (§§6, 8)

(A17): I rederived every pin identity (lines 266–270) from the axial restrictions `φ(x) = g(x,0)` and
`ψ(x) = g_z(x,0)`. The sums of `φ'(±r/2)` and `ψ(±r/2)` give `g_x` and `g_z`. Their differences give `g_xx` and `g_xz`.
The height difference gives `g_xxx = 12k + O(rK_4)`. The `Z²` coefficient is `σ/r`, not `a/r`. The `−a_3 Z/8` term comes
from `g_z(0)`.

Two independent exact checks support this:

- **Cubic fields.** On physical cubic fields in d = 3, 4, 5, built from chart data with all pins imposed and then
  sheared, (A17) holds with **zero** remainder. The sheared midpoint jets reproduce `σ`, `D` and `τ` exactly
  (`PHYSICAL`).
- **Quartic fields.** On quartic fields, where the pins are re-solved through the contact frame, the sup of
  `F_r − P` over the rational grid in `B(0,3)` halves as `r` halves (successive ratios 2.0000 to four decimals), for
  `r = 1/8 … 1/64` in d = 3, 4
  (`QUARTIC`).

The mutants `raw-a-in-plane` and `drop-transverse-pin-term` are rejected.

(A20)–(A22): for cubic fields the endpoint blocks are exactly `r P_i` (soft), `C_i = ±(r/2)(τ_xxw; τ_xzw)` (cross) and
`D ± (r/2) τ_xww` (stable). Then `det H = det D_i · det K_i`, and inertia(H) = inertia(D_i) + inertia(K_i), with index
d at `M` and d − 1 at `S`. `|K_i/r − P_i| ≤ rk`. All of this is checked exactly, with the physical Hessian computed
independently of the sheared one. Dropping the cross-block correction is rejected (`no-cross-schur`). The target planar
Hessians `diag(−6k, −k/2)` and `diag(6k, −5k/2)` are recomputed in `PATH`. Since `|det L| = 1`, physical and sheared
determinants agree, so `W_r ≥ c r⁴` with the power independent of d.

(A24): `∇f(S) − ∇f(M) = r∫_{−1/2}^{1/2} H(tru)u dt = 0` (checked exactly). So `H_i u = O(r‖f‖_{C³})`. On cubic
fields `H_i u = ±(r/2)(f_xxx, f_xxy_j)` exactly, and on quartic fields it halves with `r`. Hadamard's inequality gives
`|det H_i| ≤ C r (1 + ‖f‖_{C³})^d`, and LP §4's endpoint-conditioned moments give `Z_r ≤ C_Z r²`. The ledger is
`r · r⁴ / r² = r³` in every d (`LEDGER`; `weight-2d` rejected).

## §5 — the extra saddles (§9)

The rescaling `x = rX, z = rZ, w = r²W` is the right one. I rederived (A27) term by term:

- **Soft components.** These are `∇F_r` plus `r⁻²·r²W·f̃_{w,(x,z)}(rX, rZ, 0) = O(rK_4|W|)`. The mixed soft–stable
  Hessian is `O(rK_4)` in the `O(r)` ball, because `f̃_zw(0) = 0` exactly and `f̃_xw(0) = O(K_4 r²)`.
- **Stable components.** These are `r⁻²[f̃_w(0) + f̃_wx rX + ½(τ_xxw r²X² + 2τ_xzw r²XZ + τ_zzw r²Z²) + r²DW] + O(rK_4)`,
  with `f̃_w(0) = −(r²/8)τ_xxw + O(K_4 r³)`. This is `DW + Q(X,Z)` with `Q` as in (A28).
- **C¹ closeness.** `∂_X` of the stable part is `r⁻¹f̃_wx(…) = ∂_X Q + O(rK_4)`. `∂_W` of the soft part is `O(rK_4)`.
  Second derivatives of `H_r` are bounded by `K_4`, so the quantitative inverse-function theorem applies uniformly
  around the block-triangular nondegenerate zero `(p, −D⁻¹Q(p))`.

Heights are `b + r³(F_r(p) + O(r))`, strictly inside the window. The index is `n + 1 = d − 1`, by the same Schur
argument as §4.

`SADDLES` checks this in floating point on quartic fields in d = 3, 4 for `r = 1/16 … 1/128`:

- Both extra critical points exist, with gradient residual below `1e−15`, index d − 1 and heights in `(b − kr³, b)`.
  They are distinct from each other and from the pins.
- `|w/r² − W*|/r` stays within a factor 2 across the range, so the stable correction is `O(r)`. Scaling `w` by `r`
  instead of `r²` is rejected.
- At the planar restriction's critical point, `∂_w f̃/r²` is nonzero. This confirms the manuscript's warning (line
  365) that a planar critical point is not a full critical point.

## §6 — the corollaries (§10)

(A5) consumes LP Theorem A (1.1). The D1 reconciliation (Math-#126, `reviews/d1_chain_reconciliation_20260928/`, open)
records (1.1) as ACCEPT at existential scope in every fixed d ≥ 2, read with the congruence erratum, the §9 replacement
v1.1, wording W1 and `r < L/(4√2)`. Its consumption contract (§7 there) permits exactly this use. Until #126 merges,
(A5)'s upper half has that pending register status. The direct lower half does not depend on it.

(A6): with `ℓ = kr³` at fixed `k`, `r dr = (1/3)ℓ^{−1/3}k^{−2/3} dℓ`. Multiplying by `1 − p ≍ r³ = ℓ/k` gives
`ℓ^{2/3}k^{−5/3}`, which is integrable with positive integral on compact `K` with `k_− > 0`. For small `ℓ` every radius
is below the common cutoff. (A6a) follows: `∫_0^t ℓ^{2/3−p} dℓ` is finite iff `p < 5/3`, logarithmic at `p = 5/3`. The
rejected-pair measure is integrated directly, never as a difference of infinities (line 444).

**Consequence for the D1 chain.** With this review and #126, Theorem A's cubic order is **two-sided** `Θ(r³)` in every
fixed d ≥ 2, and the compact-mark density loss is `Θ(ℓ^{2/3})` in every fixed d. #126 deliberately says "no d ≥ 3 lower
bound". That caveat can be lifted by a later register PR once both #125 and #126 are integrated, not inside either.

## Notes

- **N1.** (A20)'s bound `‖C_i‖ ≤ CrK_4` can be read more finely as `C_i = ±(r/2)(τ_xxw; τ_xzw) + O(K_4 r²)`, so the
  Schur correction is `O(r²(εk + K_4 r)²)`. Not needed, but it shows the correction is small even before `r_*` is fixed.
- **N2.** (A4) should say whether `N_r` counts additional window critical points anywhere on the torus or only in the
  `C_d r` ball. The lower bounds hold for either reading, since both counts are ≥ 2 on the event.
- **N3.** Why a *linear* shear suffices. The path obstruction needs field values on one plane only. The endpoint and
  saddle data need only Schur complements with `O(r)` cross blocks. A nonlinear ridge `y = φ(x,z)` would give the same
  midpoint Schur complement at the cost of implicit-function bookkeeping. Neither route needs the other.
- **N4.** The bound `‖q‖ ≤ 1/9` (line 146) is used only through a uniform bound on `‖L‖`. Any fixed bound works after
  adjusting `r_*`.
- **N5.** `lift.py`'s tests are finite controls of the listed identities. The end-to-end physical-field checks here
  (building `f`, imposing pins, shearing and recovering) are additional to them, not a replacement.

## What this review does not do

It does not accept a numerical `c`, `r_*` or cutoff, a uniform-in-d bound, an unrestricted-mark difference, a
leading loss constant, a global factorial upper bound, or any identification with historical `q`/`p` symbols. The
manuscript claims none of these. It changes no register. Integration, and any register transition that lifts #126's
d ≥ 3 caveat, are separate acts for a non-Claude integrator.

## Reproduce

    python -B -S dimension_lift_review_check.py            # prints RESULTS.json byte for byte
    python -B -O -S dimension_lift_review_check.py         # identical
    python -B -S dimension_lift_review_check.py --mutant M # exit 1 for each of the 9 mutants
