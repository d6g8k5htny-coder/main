# SIDE24 — P9 Audit: Part G Chain & the RP-A/RP-L Closure Claim
**Date:** 2026-08-01 · **Auditor:** Claude (independent) · **Input:** `OKComputer_Project_Gap_Closure.zip`, **P9 pinned `89d379fa9f7b0fa917d4480979abd21d235afa7b4455d98ffc554edc4455b5e1`**, 134 files — true successor of P3 (16-item diff: v5 tree dated 2026-08-02, two master registers, three new scripts, verifier v8, runs 9–10).

## 1. Reproduction — all clean
`verify_collar_factorization` · `verify_corrected_pin_floor` · `verify_palm_soft_factors`: each **exit 0 in normal and `-O` modes, byte-identical across modes, byte-identical to committed transcripts, `ALL_ASSERTIONS_PASS`** (ck-based; the Q1/Q2 quarantine discipline holds). Scope classification honored: collar script is **NUMERIC-only** (the ρ⁶ collar law is explicitly *not proved* — register Q10); pin-floor is numeric corroboration of analytic Lemma G.1.3; palm-soft-factors is exact algebra (S1/S2, K1/K2).

## 2. Independent verification — `verify_g_chain_v1.py` (17/17, fail-closed, `-O` identical)
Written from scratch against the v5 supplement's Part G: both **G.2.1 kernel identities** as unconstrained integration-by-parts identities (zero-mean case = the lemma); kernel affine in [0,1]; **G.2.2 soft factorization det H = r[α det Q − r βᵀadj(Q)β] exact**; the **G.3.1 Schur criterion** det H = det Q·(rα − r²βᵀQ⁻¹β); the **G.2.3 trapezoidal value-window identity** ∫(w²/2−r²/8)g‴ = [g] − (r/2)(g′(S)+g′(M)) — the Q8 fix — with kernel range [−r²/8,0] and **∫kernel = −r³/12 exactly** (Q9's canonical sign); **adj(Q) spectrum {−λ_max, −λ\*} for Q≺0** (Q12) and |det Q| = λ\*λ_max; the G.4.1 absorption inequality; **G.6 exact scaling τ_r = λ\*/(κr)**; the **G.7.1 power count r⁴·r/r² = r³**; Borell–TIS branch e^{−c/r²} = o(r³) as an exact limit; tube-cost bracket 3/2048; and η_R = 1/(8192R³) strictly inside the A.4 cone width 1/(512R³).

**[N1]** One readback flag: the G.2.1 application line renders as h = ∂ᵢf where the bookkeeping requires h = ∂_{ti}f (its antiderivative ∂ᵢf carries the pin difference). Almost certainly an HTML-flattening artifact; author confirmation requested.

## 3. Claim adjudication — the headline
The master register asserts **"CLOSED (proved): … Palm transfer RP-A/RP-L (Thm G.7.1)."** My calibrated verdict:

- **Checkable algebra of the entire G-chain: VERIFIED** — independently (17/17 here) and by the author suite (both modes).
- **Soft-analysis steps: REVIEW-CONCURRED** as standard-form arguments correctly deployed — G.3.1's lower bound (weak convergence + boundary-null events + uniform integrability + compactness), G.4.2's shallow-mass estimate (Weyl 1-Lipschitz λ_min, ε-tube of the rectifiable {det = 0} hypersurface, floored-covariance bounded density), G.5's Borell–TIS with polynomial-constant dominance, and the G.7.1 assembly dichotomy (adjacency/self-attachment failure ⟹ hypothesis failure; the outward above-max escape of A.4.3 separates components).
- **Conditional dependency:** G.7.1 consumes Part F.4's deterministic capture/escape. Its core inequalities (A.4.1–A.4.3) are machine-verified (my 35/35, 2026-08-01); **A.4.4–A.4.6 and the flow/invariance arguments remain prose** — next machine-audit slot.

**Register entry:** RP-A/RP-L — *written closure, independently audited: algebra machine-verified, analytic steps review-concurred; A.4.4–A.4.6 machine-audit pending.* This matches the RP-F precedent standard.

## 4. Updated interface register
F2 closed (×2) · **RP-F CLOSED** (G.8 — former imports now internal via G.3.1/G.2.2) · **RP-A/RP-L written-closed, audited per §3** · **RP-C/RP-S OPEN** — the single residual analytic step G.9 (uniform two-sided covariance control on compactified strata; collar ρ⁶ law NUMERIC only) · recombination conditional solely on G.9 · **candidate theorem: HOLD** (the register itself keeps HOLD conditional on exactly these two premises — concur).

Quarantine log Q1–Q13: exemplary practice, acknowledged. The register's external-sources note (Drive unreachable from its sandbox) independently cross-confirms my blocker B1. **Still pending on upload:** `SIDE24_PRE_REVIEW_2026-07-31.zip` (SHA `14eaf7d4…`) for the 14-pair annex rerun.

## 5. Run record
Python 3.12.3, sympy 1.14.0. New artifacts: `verify_g_chain_v1.py` + transcripts (normal/`-O` byte-identical); P9 extraction at `gap3/`; author-script fresh transcripts byte-matched. Delivered in chat outputs; mirrored to Drive `08-01-2026`.
