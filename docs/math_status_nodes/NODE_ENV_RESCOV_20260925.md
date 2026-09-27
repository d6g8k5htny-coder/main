# NODE ENV-RESCOV (LEMMA-RNU-ENV-RESCOV) — walk-down classification

**Date:** 2026-09-25 CT (America/Chicago). **Base:** `chatgpt/drive-github-hardening-20260919` @ `077464ef5e2859ce98cbb9307799d5867a820eaf`.
**Disposition:** dependency classification only. D3-LEMMA-RN-UNIF stays OPEN. `lemma_closed` stays false.
`certified_C_H` stays false. `prizes_solved` stays false. `discharges_OBL_H5_JETMOD` stays false. No FREEZE.
Nothing here is a certificate. It proves no bound for the SIDE24 field on any cell.

**Outcome: (b) replaced by a newer route, with an explicit implication map.** ENV-RESCOV is *not* proved.
On this route its SIDE24 content follows from (sufficient, not necessary) the input-jet node RN_WHITENED_JET_THEOREM §6 item 1 plus two positivity
side conditions (listed below). §6 item 1 is a separate node with no file carrying it (a missing-carrier node, still open;
see the per-file ledger, Drive `1NuKMa4cF4-pl8SLhpD61uchh9LVBSMlf`). It is not re-classified here.
No part of ENV-RESCOV holds on any SIDE24 sub-domain or cell subset without an input-jet carrier.
What holds without `rnu_env` is the conditional reduction map. It is not a field bound.

## 1. Dependent claim (walk-down start)

- **Dependent:** LEMMA-RNU-CH-LIFT, `C_H : ||Hess κ_pair||_box ≤ C_H · RSS(env_form_white q=2)`, as consumed by the
  C_H compose gate. It rests on the parent paragraph CL-RNU-001 §5 item 1 (below).
- **Parent paragraph (source identity):** CL-RNU-001-v1.0, Drive `16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6`. Tip mirror is
  `drive/mirrors/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15/RN_UNIF_2026-09-16/CL-RNU-001_RN-UNIF_ENGINE_STATUS_AND_CLOSURE_PLAN_2026-09-16.md`, §5 item 1, lines 126–134 (quote at 127–129):
  > "Remaining: over-cell bounds at orders 2–4 from the whitened closed form log q(A, m′), A = I − Σ′, with `env_form` (q ≤ 4) on the whitened residual covariances — loose but valid, and O(1)–O(100) in size where the old bound was 1e14."
- **Review surface that owns it:** `docs/math_status/STATUS_RN_UNIF.md` "Ordered blockers" (first entry of
  ENV-RESCOV → ALLCELL-FDZ-Q4 → LOGQ-TAIL → SYM-Fw-jet → CH-LIFT). STATUS.md line 76 mirrors that line.

## 2. Lower object: exact statement and hypotheses

**ENV-RESCOV (as the lane names it).** On each certification cell Ω ⊂ {|y| ≥ d} of the far zone, at fixed r = 0.05,
there are valid uniform bounds `sup_Ω ||∂^q A|| ≤ a_q` and `sup_Ω ||∂^q m|| ≤ c_q` for q ≤ 4. Here A = I − Σ′,
Σ′ = W Σ_pair Wᵀ, m = W(μ_pair − μ₆), and W = Λ₀^{-1/2}V₀ᵀ is the **y-free** whitening of Σ₆ = SPAIR0. The lane names
this symbol `env_form_white_rescov(k,γ,d,q)` on Σ′ blocks (skeleton receipt `RN_UNIF_LOGQ_HESS_SKELETON_RECEIPT.json`,
`missing_lemmas_symbols[0]`; compose gate `RN_UNIF_LOGQ_CH_COMPOSE_CERT_GATE_V1_RECEIPT.json`, `gates_blocking_certificate[0]`).
Those receipts are local workspace receipts. They are not vendored here.

Hypotheses carried by the parent: fixed r; y-free W; Cartesian or straight-line cell paths; far zone only (d ∈ [5,17]
per CL-RNU-001 §5 item 3); bounds must be valid, not sampled.

**Different node that shares words (do not merge).** The residual-**FORM** envelope `env_form(k,γ,d,q)/√λ_k` over
18 (k,γ) pairs is a different object. CL-RNU-001 §3 (mirror lines 66–70) attributes exactly this form-level output to
historical `rnu_env.py`. The tip's form-level q=2 box envelope (STATUS.md lines 14, 39: d_min=4.999, RSS≈3.099) is
also this object. It does not bound ||∂^q A|| or ||∂^q m|| (it has different hypotheses: form rows, not ordered
Schur/inverse construction). It is not a smaller-scope ENV-RESCOV. FORM/√λ proxy not promoted.

## 3. Replacement route and implication map

**Route:** MATH-20260917-b9c2 `RN_WHITENED_JET_THEOREM.md`, SHA-256
`1324a006a284c5d600a102781d783b5e2a63565daa217df632d4a10cc45d2511`, 15717 B, Drive `1O4hMhhUhVtmCvvkRNqaqevByyTwnWe3b`.
Registered on this tip as `registers/csv/frozen_objects.csv` line 160 (`MATH-20260917-b9c2-RN-WHITENED-JET-THEOREM`,
FROZEN AUTHOR-SIDE CANDIDATE). It is author-side and has no independent review. It is not an RN_INNER_WEDGE / H3 package.

Implication map (equation numbers are the theorem's):

1. **[input-jet node = b9c2 §6 item 1, restricted to Ω; separate missing-carrier node, not re-classified here]** valid interval enclosures of the jets Y_j (of YC) and V_j (of unconditioned TY_pair), j ≤ 4,
   over Ω, including periodization tails and fixed-data conditioning/whitening errors (theorem §6 item 1)
   ⇒ by (16), enclosures of F_j, D_j, z_j, j ≤ 4.
2. **[D > 0 on Ω and S_pair > 0 on Ω]** (theorem §6 item 2; interval LDL test, §5)
   ⇒ by (2) and (14)–(15), A = F D^{-1} Fᵀ and m = F D^{-1} z have enclosed jets A_n and m_n, n ≤ 4.
   **This is ENV-RESCOV on Ω** (the a_q, c_q of §2), constructed without `rnu_env` and without the FORM/√λ proxy.
3. [a₀ < 1] ⇒ by (10)–(11), |ℓ_n| ≤ L_n for n ≤ 4, with ℓ = log q from (1). By (12), (13) and (7), X̄_j ≥ sup|∂^jχ²|.
4. **[χ² ≥ δ > 0 on Ω]** (theorem §6 item 3) ⇒ by (8), √χ² jet envelopes follow. The κ_pair envelopes then follow
   after multiplying by the y-independent prefactor hL2/Z_LO.

Consequences for the dependent:
- ENV-RESCOV ⇐ (on this route) [b9c2 §6 item 1 on Ω] + [D>0, S_pair>0 on Ω]. The lane's label ALLCELL-FDZ-Q4 points at
  that same input-jet requirement, so on this route the first two ordered entries depend on one live input node.
- The CH-LIFT **shape** `C_H · RSS(env_form_white q=2)` is not produced by this route. It gives direct envelopes
  from (a_q, c_q). It also **adds** hypothesis 4 (a χ² floor δ on Ω), which the RSS shape never stated.
  CH-LIFT stays OPEN. Reading CH-LIFT through this route is conditional on items 1, 2 and 4.
- `rnu_env.py` is recorded ABSENT on the tip. Its per-file absent-carrier record is in the ledger at Drive
  `1NuKMa4cF4-pl8SLhpD61uchh9LVBSMlf` (six absent nodes). That ledger is cited here, not re-classified. Only the filename
  `allcell_fdz_enclosures.json` was coined by our own hunt receipts. The object behind it, the all-cell F,D,z
  enclosures of b9c2 §6 item 1, still has no file carrying it. It stays a missing-carrier node, now named by that
  section reference, and it is not retired. Its dependent sentences, including this node's route, stay conditional.
  By CL-RNU-001 §3,
  even recovered `rnu_env.py` bytes would carry the FORM envelope, not item 2. Recovery is therefore not the route
  to ENV-RESCOV.

**What was checked here (evidence, not proof of the field):** `env_rescov_b9c2_identity_spotcheck.py` (this directory)
independently re-implements (1), (2), (4), (7), (10)–(11), (15) on random non-commuting polynomial matrix paths (d=4,
order 4, mpmath dps 60). Results: eq (4) rel. err ≤ 2.6e-61; eq (1) against a direct 1-D Gaussian integral 6.7e-61;
eq (7) X₄ 9.7e-62; eq (11) holds with |ℓ_n|/L_n ∈ [0.15, 0.32]; eq (15) A_n max abs err ≤ 9.7e-63. The theorem's own
`VALIDATION_REPORT.json` (in the b9c2 bundle, not vendored) reports 123/123 PASS. These identities are standard calculus.
A spot-check is not a review and not a certificate.

## 4. Smaller-scope question

Is any part of ENV-RESCOV proved on a SIDE24 sub-domain or cell subset without a carrier? **No.** No cell Ω has
enclosed (Y_j, V_j), j ≤ 4. Point values at y = (5,0) (skeleton: ||A||_F ≈ 2.35e-5, ||m|| ≈ 1.39e-3, mpmath dps 60)
are point evaluations, not enclosures. They prove nothing on a cell. The polynomial fixture in the b9c2 bundle is a
proof for that fixture only, not SIDE24. Complement = every cell of the far zone. No domain rescope is licensed.
ENV-RESCOV does not rely on the frozen Piece-2 `envelope_v` or on the wedge domain W. Any rescope to the INNER / C11
wedge domain depends on the wedge/H3 classification, owned elsewhere (crosswalk at Drive `1Mrg6PT8vHAjpeAKBwDh_jiHIbm4MKM9x`;
its class (b) scope is W = {1/10 ≤ ρ ≤ 11/100, |t| ≤ 1/1024 turns} only). This record does not use those packages and does
not widen that scope.

## 5. Pinned successor (smallest calculation the parent explicitly deferred)

Deferral quoted and checked against the tip mirror (CL-RNU-001, lines 126–129, verbatim): "Remaining: over-cell bounds at
orders 2–4 from the whitened closed form log q(A, m′), A = I − Σ′, with `env_form` (q ≤ 4) on the whitened residual
covariances — loose but valid, and O(1)–O(100) in size where the old bound was 1e14." The quote matches. This is the
pinned successor for ENV-RESCOV. Its smallest instance is stated below. Its input is b9c2 §6 item 1 restricted to one cell. That is a separate missing-carrier node and is not re-classified here.

**SUCC-ENV-RESCOV-1CELL-(5,0):** one cell only, not all cells. Take the square cell centred at the frozen probe point y = (5,0),
half-width hw = 7e-4 (the CL-RNU-001 §4 table value at cap 0.68). Use fixed r = 0.05 and Cartesian straight-line paths. Produce valid interval enclosures of Y_j and V_j, j ≤ 4, including
periodization tails. Then run (16) → (14)–(15) with the interval LDL test for D>0 and S_pair>0, and report a_q, c_q
and whether a₀ < 1. Pass/fail is binary. An inconclusive pivot is a FAIL, not a bound. It is the one-cell instance of the CL-RNU-001 §5 item 1
deferral. It opens no new program. Status: **UNCLAIMED, NOT STARTED**. The χ² floor on that cell (item 4) is a
second, separate deferred check. It is not bundled into this successor.

## 6. Importing sentences touched

- `docs/math_status/STATUS_RN_UNIF.md` "Ordered blockers" section: one conditional paragraph appended (ENV-RESCOV is
  implied on the b9c2 route by the separate §6 item 1 input-jet node plus positivity, and is not cleared). PACKET.json pin updated for that file only.
- Full dependent-sentence audit: done elsewhere. See Drive `1iIB4tI2FSSBl066hDIlrTidM2NeuT2za` (155 importing
  sentences, 13 still conditional). It is cited here and not redone.
- Not edited here (covered by that audit): `STATUS.md` line 51 ("`rnu_env.py` /
  `rnu_white.py` still ABSENT — need whitened log-q(A,m′) Hess composition to certify C_H"). Read through this node,
  that sentence is conditional. `rnu_env.py` recovery is not the ENV-RESCOV route (§3).
- JETMOD: the absent-carrier classification of `explicit_interval_map_F_G12box_to_Rplus` is recorded by another agent.
  This record adds nothing to it and does not duplicate it.

*D3-LEMMA-RN-UNIF remains OPEN. lemma_closed: false. certified_C_H=false.*
