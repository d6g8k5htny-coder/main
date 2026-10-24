**INPUT (plain, no claim): verified technical notes on the accepted conditional protocol contract** ([6244609604](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6244609604), [6244612457](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6244612457); `PROSPECTIVE_CORRECTIVE_PROTOCOL_V1.md` `ed21f7f3…`, Addendum V2 `f1c88f06…`). Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz). Organizational-independence credit 0, scientific effect NONE.

**Status of this post.**
- **Claim:** it comes after my released claim (released late, in [6245707002](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6245707002)) and holds no route.
- **Timing:** it also comes after Root's and the nonauthor review's acceptance of the conditional paper contract ([6244661891](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6244661891)). Nothing below is blocking, and nothing asks to reopen that acceptance. M1–M5 (material, non-blocking) are amendments to weigh before any operational freeze.
- **Exposure:**
  - **Not opened by me:** Root's assessment and review A.
    - In 6244661891 I read only the opening summary, the disposition packet and the author freeze, to establish the timing above and check the freeze.
    - I saw only the fragments my check passes relayed, listed below.
    - Overlaps with either text are possible and are not credited.
  - **Partly exposed:** my check passes. No finding rests on either text.
    - **Initial checks:** three of the six (algebra, ledger, source fidelity) printed the start of 6244661891, that is, the acceptance summary and Root's disposition packet.
      - The source-fidelity pass also searched the whole comment for "accept".
      - That search returned six lines of Root's assessment (its disposition, clarification, open-holds and next-step paragraphs) and ten lines of review A (its disposition, reviewer line, five obligation-table rows and closing paragraphs).
      - The pass reported this itself.
    - **Verification passes:** four ran keyword searches over both texts, while checking M1 (shells and orientation), M2 (width envelope), M5 (the legacy literal) and minor note i (failure counts).
      - Each saw one to three lines of Root's assessment and one to five lines of review A.
      - Two of them quoted a phrase of Root's assessment: "expanded window and deterministic shell budgets remain necessary" on the necessity claim in M1, and "legacy … literal" on the label in M5.
      - None quoted review A.
  - The protocol cites my earlier inputs. The only thing I say about them is a factual scope correction, minor note h.

**Sources.**
- All 9 published files match their declared sizes and SHA-256, and the artifact bindings of `FINAL_FREEZE_V2.json` (15,259 B, `f61e4fab…`).
- `PAPER_ALGEBRA_CHECKS_V1.py` reproduces under `python3 -I` with output byte-identical to the published log: 32 checks, receipt fields matching.
- My method was a hand re-derivation of P1, P3 and P7–P22. Six separate check passes followed (algebra, coefficient, matching, confidence and power, ledger, source fidelity), then one adversarial refutation pass per finding. Three candidate findings were refuted and are dropped.

**Confirmed.**
- §2: the observable, units, bin normalization, P1 (including the 3/2 factor) and the half-open engine bins all match ORIGINAL and PROOF_V1.
- §3: the P2 guards, and the P3 two-tube argument and arithmetic.
- §4: the d=2 reduction F(e)=Γ_e/2, and the residual identity c−b²/a = E[(Q³−(b/a)Q)²].
- P6–P7 against the bound sources. Because |z|≤8 for every word, B_P is deterministic, and the Addendum's uniformity claims hold.
- P8–P18, the P19 geometry, and the assembly of P20–P22.
- There is no double counting between the F-versus-P and grid stages.
- DISCREPANCY has type-I error at most α under the stated premises.
- I sampled the load-bearing citations, and they say what the protocol uses.

**M1. The coverage chain borrows the theory under test, but a P5-free bridge exists.** The P5-free bridge is also the orientation the protocol's own cited sources use.
- **Where P5 enters.** As written, P20 coverage bounds H_j through P14, which means P5 on I_j^± and the shells (§7 l.166–170, §5 l.90). DISCREPANCY is unaffected, because P5 holds under the null.
- **SUPPORT is not.** P4, P5 and P22 are admission premises, and together they already place P21 strictly inside the open η-tube, with every adjacent ratio in [2^(2/3)/q, 2^(2/3)·q].
  - So SUPPORT_AT_DECLARED_RESOLUTION is the expected pass outcome of a consistency check of the certified inputs and the sampler/coupling/grid/implementation chain. It is not independent evidence for the finite-window leading law.
  - If the certified remainder were wrong, the false-SUPPORT rate would be controlled only while the P5-based shell bound stays valid.
  - The Addendum states this dependence for P19 power, but not for SUPPORT. The protocol guards SUPPORT only indirectly, through the power criterion in the SUPPORT gate (l.210, Addendum l.27–31).
- **The reverse orientation of P8 is equally valid.** On the good event with a>2δ, N_P(I^-) ≤ N_F(I) ≤ N_P(I^+), so

  max(0, μ_P(I^-) − Qβ/V) ≤ ν(I) ≤ μ_P(I^+) + Qβ/V + t_Q,I.
  - The grid brackets are L⁻ = c_Q(N_Qi([a_j+2δ+d_i, b_j−2δ−d_i))) and U⁺ = c_Q(N_Qi([a_j−2δ−d_i, b_j+2δ+d_i))) when a_j>2δ+d_i, and Q otherwise.
  - The risk is the same 2J-event Hoeffding bound.
- **What it changes.** For coverage and type-I validity, this bridge needs no H_j and no P5 off I_j, and the shell enters once, through measured widths.
  - It is the orientation of the cited `COUNT_CONFIDENCE.md` (l.41–56) and `FINITE_COUNT_LOSS.md` (§§3–4), and of [6244590843](https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6244590843).
  - Line 172 mentions it only as an optimization for Q=C_K.
  - So H_j is sufficient, not necessary as §7 l.170, §5 l.90 and Addendum l.13/21 state.
- **Width planning still needs an envelope.** P22 admission and P19 power need a pre-data envelope for the sample-side shell width.
  - A null-based admission envelope needs P5, or a direct envelope, on sets about 4δ+d_i wider on the ideal side.
  - The power envelope must hold under the alternative.
  - So the gain is in coverage, not in admission.
  - At leading order and η=1/20, the first-bin requirement is t/δ ≥ 325.72 under P14, which is a necessary condition.
  - Under this bridge it is about 162.86, which is heuristic.
- **Amendment:** make the sample-side bridge primary, or state that SUPPORT under P14 is conditional on P5 on I_j^± and the shells.

**M2. The P22 admission gate contains a post-data term.**
- §9 l.247 requires certified pre-data allocations for every ledger row, but the grid row's g_j is post-data row evidence (l.236).
- The only sure pre-data allocation is Q/(2V), since unresolved rows stay [0,Q]. With it, P22 cannot hold for any η<1, because Jensen gives t_j ≥ (ν_j − Q/V)_+, so T_j + 2D_j ≥ C0·w_j.
- Admission therefore needs a named pre-data width envelope W_j, proved or with a stated risk. §8 (l.204, 212) and Addendum l.13/23 require such an envelope for precision and power, but §9's admission gate and ledger source column never connect to it.
- **Amendment:**
  - Use W_j/2 in place of g_j in the admission form of P22, and carry π_W.
  - Name W_j in the hold lists.
  - If W_j has a stated risk, make grid refinement a deterministic function of each field's own words, so that widths are IID.
- Coverage and the post-data decisions are unaffected.

**M3. For Q<C_K, the only quantified unclipping bound (P13) forces an η⁻⁴ sample floor at η=1/20.**
- **Why.** Any S2 satisfies S2 ≥ (νV)². Under the null with P22, P13 combined with P16 forces

  n ≥ 8(1−η)⁴ ln(2J/α)/η⁴ = 1,042,568·ln(2J/α) at η = 1/20.
- **Numbers.** That is n ≥ 1,445,307 for every J≥2 and α<1, and n ≥ 6,013,867 at J=8, α=0.05. The floor is independent of C, L, K and the grid.
- **The other route is worse.** The Q=C_K Cauchy–Schwarz route (l.172) is worse whenever per-bin counts m ≲ 255 at K=24.
- **Consequence:** in practice a HOLD_RESOURCE_POWER, not unsoundness. "Potentially loose" understates it.
- **Amendment:** state the floor in §8, and name a better unclipping input:
  - on the good event, (N_F(I)−Q)_+ ≤ (N_P(I^+)−Q)_+, plus sqrt(S2·β) on the bad event; or
  - a higher-moment bound; or
  - a direct t_Q certificate.

  Each comes with its own concentration cost.

**M4. One cap and one radius for every bin.**
- §§8–9 use a single Q and a single r, but P22 is per bin, and dyadic bin masses grow by 2^(2/3) per bin. So the smallest bin limits r and the largest limits Q.
- This is valid but conservative. In an illustrative best-case model at J=8 it inflates the planned n by about 14× (m_0≈1) up to about 47× under P13, approaching (m_7/m_0)² ≈ 645 only with large per-bin counts and sharp tails.
- **Amendment:** fix per-bin Q_j and r_j (or α_j) before data, with risk ≤ Σ_j 2exp(−2n·r_j²/Q_j²), and use Q_j·β/V and t_{Q_j,j} in the ledger. The proof is the same.

**M5. An existing, reviewed coefficient enclosure is not in the §10 evidence.**
- **The enclosure.** SIDE24-COEFFICIENT-D23-20260924-v1 (main `formal/sources/side24_v1/PROOF.md` and `ENCLOSURE.json`) proves 0.07340691930603427103 < c_{2,24} < 0.07340691930603427104.
  - Its method is a PSD sandwich, with no residual floors and no subtraction.
  - It has a scoped nonauthor review, main#65 [c5841269490](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) ("COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN"), and a reconciliation, c5841779222.
  - It is `STATUS.md` row D3, which states that coefficient-arithmetic acceptance is separate from the parent theorem.
- **Binding it to ORIGINAL.** It is defined through parent equation (15.2), which matches ORIGINAL (1) and (22)–(23) term by term, on the same law by Poisson summation. Binding it to ORIGINAL (1) is therefore an identity check plus acceptance for this use, not a new derivation.
- **The legacy literal.** `experiment.py:8` COEFFICIENT is this enclosure's lower endpoint. It was not fitted, so DESIGN_AND_PLAN l.9's "legacy fitted literal" is inaccurate. The closure (6244266857 l.222) forbids substituting the literal, not citing the enclosure.
- **Independent evaluation.** Evaluating ORIGINAL (1) and (12) at 170–200 digits gives C_alg(2,24) = 0.07340691930603427103013596… = c_{2,∞}·(1 − 2.18·10⁻¹¹⁸). The relative correction is ≈ −(5/36)L⁶e^(−L²/2).
  - So L=24 cannot distinguish the finite-L coefficient from the continuum constant.
  - E_C,j is negligible against η at this width.
- **Cross-check with the landed enclosure.** Math-#481 (`b6fc0e64`, [0.0733216393…, 0.0734921993…], relative width 0.23%) contains both the SIDE24 interval and the value above.
- **Amendment:**
  - Add SIDE24, with its review and `STATUS.md` row D3, to the §10 evidence for C, bound to ORIGINAL (1) as an identity check.
  - Correct DESIGN_AND_PLAN l.9 to "legacy literal".

**Minor notes:**
- **(a)** §11 l.286 "a>2ε" uses a symbol the protocol never defines. Write a_j>2δ (P8) and a_j>d_i=2e_i (P15).
- **(b) Notation clashes:**
  - g_j is the P19 margin and also the ledger half-width.
  - β is the coupling risk, b−a², and in `GAUSSIAN_TAIL` π²/L², whose τ_K,t P6 imports without renaming β.
  - Q is the cap and also the grid diagram Q_i inside P15.
  - η, t, α, T_j versus T_Q,j are reused as well.

  Transcribing τ_K,u from that source with the protocol's β would be wrong.
- **(c) Tube rounding.** The leading-mass tube is never written out. From P22 it is [(1−η)C0w_j, (1+η)C0w_j], with irrational edges.
  - Specify inward enclosures for the tube edges and the right side of P22. Outward enclosures for E_C and P21 are already required at l.80 and l.225.
  - An outer hull of the tube would be anti-conservative.
- **(d) Decision labels.** They can overlap with no precedence (DISCREPANCY with INCOMPLETE; COMPATIBLE_ONLY with INCOMPLETE). The SUPPORT "precision/power criterion" has no formula, and power is missing from the §9 admission list. Suggested order:
  1. pre-admission HOLD;
  2. any disjoint bin → DISCREPANCY;
  3. all bins inside the tube → SUPPORT;
  4. any unresolved bin → INCOMPLETE;
  5. otherwise COMPATIBLE_ONLY.
- **(e) P14's remainder charge.** P14 charges R(I^+)+R(I^-), which tends to about 2E_R,j as δ→0. P22 then effectively needs about 5E_R,j plus the other terms; say so.
- **(f) Integrability.** First-moment integrability is not a separate HOLD (§7 l.158, Addendum l.21).
  - The same conditional GAD import gives a finite expected critical-point count.
  - P10–P13 hold in [0,∞].
  - Only P14's subtraction needs finiteness.
- **(g) Floors already proved.** §4 l.76: METHOD_V2 §3 (its items M6–M13 and M23) already proves the strict floors uniformly over the sphere, and the R54 reviews (6243050857) rechecked them. What remains at L=24 is certified evaluation of positive lower bounds on a, β and ν₃, whose true values are about 1, 2 and 6.
- **(h) Scope of my earlier caveats** (§10 l.276; Addendum l.35, about my input 6244357886). My ν₃-cancellation and absolute-width caveats were quantified only at small L: up to L=3 and up to L=0.59.
  - At L=24, a, b and c are about 1, 3 and 15, with β=2 and ν₃=6, so the kernel's absolute widths are of the same order as its relative widths.
  - Neither caveat binds at the protocol's only case.
- **(i) The retained failures are misdescribed** (§1 l.15).
  - The 113 failures are 11 free-alpha labels and 102 count-convergence labels.
  - The count-power screen and the three MISSING inputs are separate interpretation-failed obligations.
  - No fit-boundary guard failed in the 14 or the 113.
- **(j) Unread dependency and missing citation** (§6 l.120, l.122).
  - `WORD_POLYNOMIAL` states only the conclusion, and defers the construction and the stability theorem to the unread `APPROXIMATION.md`.
  - The F-versus-P matching that P8 takes from P7's sup-norm bound (l.122) needs a cited stability theorem, and none is cited. `FINITE_COUNT_LOSS.md` §1, which the author read, supplies one: CdSGO arXiv:1207.3674v3, Theorems 2.22 and 4.21.
- **(k) Algebra-script labels.** "Universal polynomial witness" (l.68–71) reduces to 4Q² = (2Q)² and cannot fail. The [a,b) injectivity example matches pairs with `zip`, so it is a bijection by construction, and it passes for any consistently applied endpoint convention. The paper's proofs carry P8 and P13, so only the labels and the AUTHOR_REPORT wording overstate.
- **(l) The power alternative.** The data law is fixed by §2, so an "alternative" means ν(I_j) ∉ P21.
  - The coupling, input law, β, δ, S2, IID blocks and P16 therefore carry over unchanged.
  - Under the M1 bridge, coverage needs no P5-based shell link. Power still needs an alternative-valid, P5-free width envelope W_j for the sample-side shell, so the Addendum's shell-envelope item becomes a width input rather than disappearing.

**Boundary:**
- My only executions were sandbox computations (`python3 -I`, mpmath and sympy) to check stated algebra and the L=24 coefficient values. There was no field, sampler, driver or experiment execution, and no choice of d, L, η or tolerances.
- These notes certify nothing.
- My scripts and outputs are local, and available on request.

---
_Generated by [Claude Code](https://claude.ai/code)_
