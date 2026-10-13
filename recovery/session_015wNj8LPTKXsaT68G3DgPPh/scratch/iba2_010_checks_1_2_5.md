## IBA2-010 nonauthor delivery — checks 1, 2 and 5 of the ledger: **PASS** (1, 2) at statement scope; (5) traced

Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh`. Dylan Roy — delegated AI work. Answers my claim 6003114849. Scientific effect **NONE**; credit **0**; **IBA2-010 stays OPEN** (checks 3 and 4 need a non-Claude owner; nothing here is a discharge). Read-only against Math- `main` `dcd2a886…`, the ledger's pin: [SC] `frontiers/spectral_cluster_closure_20260929/PROOF.md` blob `16c56821`, [RC] `frontiers/remote_collision_20260928/PROOF.md` `7b48a88e`, [RM] `frontiers/remote_window_20260924/PROOF.md` `b383bfcc`, [D5] `frontiers/d5_dimension_lift_20260929/PROOF.md` `9d82c707`, [EDL] `frontiers/elder_dimension_lift_20260928/PROOF.md` `7303bd79`. Line numbers are from `git show <pin>:<path>`.

**Independence.** I am not an author of [SC], [RC] (Claude, other session), [RM], [EDL]. I authored [D5]; for check 2 I therefore only match [SC]'s *use* of (1) to what [D5] *states*, and do not re-accept [D5] (its reviews are Math-#141 OpenAI 5357858391 / 5357882570 and the planar xAI/Claude records in the ledger).

### Shared model (the precondition for every check)
[SC] §1 (lines 12–21), [RC] "Setting" (10–29), [RM] §1 (14–35) and [D5] §1.1 (16–27) state the same object: variance-one periodized Gaussian field on `R^d/(LZ^d)`; pins `M = -(r/2)u`, `S = (r/2)u`, `f(M)=b`, `f(S)=b−kr³`, `∇f = 0`; `Q_r` = regression on the `2(d+1)` observations; `W_r = F_d(H_M)F_{d−1}(H_S)`, `Z_r = E_Q W_r`, `dQ_r^W = (W_r/Z_r)dQ_r`; window `I_r = (b−kr³, b)`; `N_{r,j}(E)` = index-`j` critical points in `E` with height in `I_r`, `N = Σ_j N_{r,j}` over all `d+1` indices; `D_ρ = {dist(x,0) ≥ ρ}` with `0` the midpoint; marks in compact `B`, `K`, `0<k_-`. Constants depend on `d, L, B, K` and the fixed radii, never on `r, b, k`, the frame, the index or the Borel set. ✔

### Check 1 — (2) for [SC]: **PASS**
[SC] (2), lines 37–41: `E_W N(D_ρ) = k r³ ∫_{D_ρ} Λ + O_ρ(r⁴)` and `E_W (N(D_ρ))₂ = O_ρ(r⁵)`, "with Λ the SUM of the index-specific normalized contact kernels", fixed `ρ > 0`.
- *First part.* [RM] Theorem A (40–44): `|E_{Q_r^W} N_{r,j}(E)/(kr³) − ∫_E Λ_j| ≤ C r|E|` for `r ≤ r_*`, **all** `b, k, R, j` and Borel `E ⊂ D_ρ`, `C = C(d,L,ρ,B,K)`; line 51: "Sum over the finitely many indices to count all critical types." With `E = D_ρ`, `|E| ≤ L^d`: `E N(D_ρ) = kr³ ∫ Λ + O_ρ(r⁴)`, `Λ = Σ_j Λ_j`. [RC] Corollary E (51–55) states the summed form explicitly: "The same holds for `N(E)` with `Λ = Σ_j Λ_j`." ✔
- *Second part.* [RC] Corollary D (42–47): `E_{Q_r^W} N(E)(N(E)−1) ≤ C r⁵|E|` for every Borel `E ⊂ D_ρ`, "with the same constants" as Theorem C (31–40: `C, r_*` depend on `d, L, ρ, η₀, B, K`; uniform over all marks, frames and index pairs). Proof (229–231) is the sum of Theorem C over all index pairs, so the bound is **all-index**, and the cross term `r⁶|E|²` is absorbed explicitly. With `E = D_ρ`: `(N(D_ρ))₂ = O_ρ(r⁵)`. ✔
- *Use in [SC].* §7 (385–387): `r⁻³Q_W(N_far^ρ = 1) → k∫_{D_ρ}Λ` and `Q_W(N_far^ρ ≥ 2) = O_ρ(r⁵)`. This is `Q(N=1) = E N − E[N·1{N≥2}]` with `E[N·1{N≥2}] ≤ E N(N−1)`, i.e. [RC]'s own Corollary E argument (233–237). §8 (419–428) uses (2) for `N_far ≥ 2` only. ✔
- *Hypotheses.* [RC]/[RM] need `0 < ρ < L/4` fixed and `r ≤ r_*(ρ)`; [SC] takes `r → 0` first at fixed `ρ` and claims only `O_ρ`, consistent with [RC] §8 ("not uniform as ρ → 0", 248). Same window `I_r`, same `Q_r^W`, same `D_ρ`. **No mismatch found.**

### Check 2 — (1) for [SC] and the choice `R_r = ρ′/r`: **PASS**
[SC] (1), lines 32–35: `E_W N_r ≤ C r³` and `E_W N({Rr < |x| < ρ}) ≤ C r³(R⁻² + ρ²)`, `R ≥ 4`. [SC] §7 (372–379): for `0 < ρ′ < ρ ≤ s₀`, `k∫_{ρ′≤|x|<ρ} Λ ≤ Cρ²` (21), "choose `R_r = ρ′/r`, which is permitted because the shell constant is uniform in `R_r ≥ 4`".
- *Source statement.* [D5] Theorem I_d (53–60): (1.5) `E N_{r,j}({s ≤ |X| ≤ 2s}) ≤ C r³[(r/s)² + s²]` for `0 < r ≤ s/4`, `0 < s ≤ s₀`; hence (1.6) "**uniformly for `A₀ ≥ 4` and `A₀ r < ρ ≤ s₀`**, `E N_{r,j}({A₀ r ≤ |X| ≤ ρ}) ≤ C r³(A₀⁻² + ρ²)`". Summing the `d+1` indices gives [SC]'s (1) second line with `R = A₀` (constant absorbs `d+1`). First line of (1) is Theorem G_d (1.7), 62–66. ✔
- *The scaling choice.* `R_r = ρ′/r` satisfies `A₀ = R_r ≥ 4 ⇔ r ≤ ρ′/4` (true as `r → 0`) and `A₀ r = ρ′ < ρ ≤ s₀`, exactly (1.6)'s range; since `C` in (1.6) does not depend on `A₀` or `ρ`, the bound reads `E N({ρ′ ≤ |x| ≤ ρ}) ≤ C r³((r/ρ′)² + ρ²)`. Divide by `kr³` and let `r → 0`: [RM] Theorem A on the fixed annulus `E = {ρ′ ≤ |x| ≤ ρ} ⊂ D_{ρ′}` (`ρ′ < L/4` fixed) gives `E N(E)/(kr³) → ∫_E Λ`, so `k∫_E Λ ≤ Cρ²` — (21). ✔ Then `β_far = k∫_{X∖{0}} Λ` (22) is finite by monotone convergence with (21) uniform in `ρ′`. ✔
- *§8 (25).* "with an absolute constant `C` independent of `R, ρ`" for the `N_mid > 0` term is Markov on (1.6), whose `C` is independent of `A₀` and `ρ` by [D5]'s statement. ✔
- *Distances.* [D5] §7 (568–575) measures `|X|` from the midpoint; [SC]'s `|x|` is from the origin = midpoint. Same `Q_r^W`, same `I_r`. **No mismatch found.**

### Check 5 — [EDL] (A4) and whether the lower bound is in scope: **traced; in scope; review on file**
- *Scope.* main#259's IBA2-010 row reads "**Sharp** factorial moments and the support-{1,2} cluster law". Sharpness is the two-sided `E_{Q_r^W}[N(N−1)] = Θ(r³)` of [CP] Corollary Θ (1.2), whose lower half is [EDL] (A4) ([CP] 81–87, 655–656). So the lower bound **is** in IBA2-010's scope and check 5 applies.
- *Statement.* [EDL] (A4), lines 55–63: `Q_r^W(N_r ≥ 2) ≥ c r³`, `E_{Q_r^W}[N_r(N_r−1)] ≥ c r³`, every fixed `d`; "a torus-wide `O(r⁵)` factorial bound is excluded". Consumed as the lower bound by [CP] (Corollary Θ, Corollary P) and [D5] (Corollary M_d, 76–79; §7 582).
- *Review trail.* `reviews/elder_dimension_lift_claude_20260928/REVIEW.md` (blob `e2b9fcdc`, Math-#125): Anthropic Claude, session `017Mi3hx…`, different provider from the OpenAI author; row 39: "§9 (363–415), **(A4)**: two extra index-(d−1) critical points, `Q^W(N_r ≥ 2) ≥ c r³`, factorial lower bound — **ACCEPT**". Integrated via Math-#129 (merged 2026-09-29, `e7f8aca4`). [EDL] blob on `main` at the pin equals the reviewed blob `7303bd79` (as [CP] `SOURCE_MAP.json` also records). Same account; no organizational independence, as every record says.
- *One wording note carried forward (not a finding).* That review's N2 asks (A4) to say whether `N_r` counts additional window critical points anywhere on the torus; [EDL] 63 ("torus-wide") and [CP]'s global `N` read it as torus-wide, consistently. No change requested.

### Optional observations (no AMEND, no BLOCK)
- **O1 (literature pointer).** [RC] §9 (256–266) already cites Ladgham–Lachièze-Rey (SPA 2023, exact second factorial moment in small balls), Beliaev–Cammarota–Wigman (ECP 2020), Gass–Stecconi (PTRF 2024) and Muirhead (SPL 2020); [C6L] 291 cites Gass–Stecconi. [SC] and [CP] cite only the Armentano–Azaïs–León framework. A one-line pointer to [RC] §9 in [SC]/[CP] would make the small-ball factorial-moment literature reachable from the cluster-law packets. A fresh Consensus pass today returned the same works and nothing treating the pinned determinant-weighted law `Q_r^W`.
- **O2.** [SC] line 428 and [RC] Theorem C are the two places where "all indices" is load-bearing; both are explicit.

### Still open for IBA2-010 (not mine)
- **Check 3** ([CP] §6 joint-floor rows R1–R4 vs the cited [DL] equations, under `Q_r` with the extra witness conditioning): needs a non-Claude owner — I authored [CP].
- **Check 4** (`d ≥ 3` resting on Math-#141's source-exposed review): needs a non-Claude co-sign — I authored [D5].
Lease 6003114849 **released**. CoS may route a Grok reader on this note; it binds the five blobs above.

---
_Generated by [Claude Code](https://claude.ai/code)_
