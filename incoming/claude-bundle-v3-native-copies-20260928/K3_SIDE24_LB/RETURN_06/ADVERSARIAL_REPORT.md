# ADVERSARIAL_REPORT — RETURN_06 adversarial review record

Sources (all hash-verified, SOURCE_CAPSULE.sha256): the frozen Stage-E corpus STAGE_E/REVIEW_{proof,
numerics, provenance, scope, topology}.md; D1 v2.1 §4 CHANGES (round-1 dispositions); D1 v2.2 §0/§4
(round-2 dispositions); the repair carriers R1–R4; lane FAILED_APPROACHES records.

## 1. Stage-E protocol

Independent reviewers, each with NONE prior exposure beyond the mission prompt (declared in every review);
unpooled findings; certificates RERUN, not read; custody hashes recomputed from bytes per each carrier's own
FREEZE convention; CANNOT-VERIFY recorded strictly separately from FAIL. The frozen corpus carries FIVE
review files; the lead consolidated them into FAIL-1/FAIL-2; finding IDs G1–G6 do not exist anywhere in the
frozen corpus (flagged by D1 v2.1, unresolved, no claim depends on them).

## 2. Reviewer dispositions (round 1, target: D1 v2.0)

| Reviewer | Disposition | Findings |
|---|---|---|
| PROOF | **FAIL** (three independent grounds, each sufficient) | F1 fatal: H4-JC pointwise disjoint-carrier premise FALSE (level conflation; proof + explicit counterexample: ens-0 pair (875,951), window saddles [915], qual_D2 ∧ N_loop-member; true overlap 176/1435) — the frozen carriers prove only the twice-paid 2·I_hi form. F2: rung normalizer floor uncertified (H3 r₀ existential; "0.05 inside H3's radius" used, never named; D1 §8 "unconditionally of every open item" false). F3: remote bracket drops D3's (1 + κ_far) ≈ 1.63 factor; ε_QMC is MC evidence; zone uniformity is the OPEN D3-LEMMA-RN-UNIF |
| NUMERICS | **FAIL-grade finding F-1** (+ minors) | F-1: certification-chain gap on the consumed remote constant (19.55·r³ vs certified κ_far = 0 value 19.5483·r³ with the source stating (1+κ_far), κ_far ≤ 0.63 + ε_QMC). F-2: H5 A≡B two-path claim vacuous (modes parameter-identical; digests valid). F-3: stale records (state-file hash quote; code drift; H4 probes-ladder headline). F-4: freeze-time exact totals unrecoverable — CANNOT-VERIFY, ≤ 4.6e-12, immaterial. All certificates re-executed PASS (H3/H4/H5 digests reproduced; I_hi = 8.097558925e-02 independently reproduced to 40 digits) |
| PROVENANCE | **FAIL** (six findings, two material; three CANNOT-VERIFY) | F2 (material): the consumed bracket drops (1+κ_far); a QMC estimate rides inside a "certified" bracket. F3: H5-AXIS v3 absent from the frozen consumed carrier; S-disk radius records disagree. F4: D3-LEMMA-RN-UNIF dropped between D3 and Theorem (1)/§8. F1/F5/F6: H3 c₀ prose +1.4e-40 (E-H3-1); C_RN ladder display rounds the 0.025 rung DOWN; C1 "a₆ = 15" flat (a₆ − 15 = −3.1e-117, E-C1-1). CV1–CV3: CANNOT-VERIFY (H5 first-merge full-precision totals; stitch part-level drift; "45 cks" count, cosmetic). Hunt list: 3D firewall PASS; planar-limit PASS except F6; superseded-carrier hygiene PASS |
| SCOPE | **FAIL** (three exact scope violations) | F1: D3-LEMMA-RN-UNIF dropped at Theorem (1)/§8. F2: (1+κ_far) dropped from the consumed 19.55·r³ bracket (κ instantiated at its QMC-EVIDENCE value inside a certified display). F3: PERC-DECAY missing from Theorem (2)'s hypotheses while B4.rem/B2-far consume it. CANNOT-VERIFY: trapezoid-as-certified-sup question; κ_far = 0 exactness for d ≥ 5; unstated O(r³) B4.rem pricing |
| TOPOLOGY | **T6 FAIL** (H4-JC); T1–T5, T7 PASS | T6 = the same H4-JC level conflation (counterexample reproduced inline; reproduction recipe frozen in the review). T5 (D2-COMPLEMENT 4-channel) PASS; taxonomy/complement/LPW-placement targets PASS |

Convergent consolidation: FAIL-1 = proof F1 = topology T6; FAIL-2 = the "unconditional" over-claim;
proof F3 = numerics F-1 = scope F2 = prov F2 (the κ_far break); scope F1 = prov F4; scope F3 standalone.

## 3. The three convergent breaks and their repairs R1–R4

1. **H4-JC level conflation** → **R1**: H4_JC_EVENT_LEVEL.md (H4JC-R1, body 42ee88da…) — event-level
   joint carrier proved from taxonomy coverage + subset counting; display preserved verbatim, ZERO constant
   changes; mirrors replaced (OLD 132/175 violations pinned fail-closed as a certified datum; NEW 0/439);
   downstream audit (D2 unaffected; E[N_loop] ≥ P(A) side consequence; loop factor re-pointed to
   OBL-B1-BRANCH(loop|B1)); the retracted form cited nowhere in v2.1/v2.2 (gate ck; retraction-string
   injection kills the gate).
2. **Z-rung out-of-scope** (rung floor uncertified) → **R2**: H3_RUNG_FLOOR.md (whole 6347275d…) —
   certified interval Z_{0.05} ∈ [7.7592917375327855e-3, 1.1468646473404396e-2], margin +92.12% over
   c_Z(0.05)² = 4.0387231691087558e-3; stretch rungs 0.045/0.055 same margins.
3. **κ_far under-pricing / dropped factor** → **R3**: D3_REMOTE_AMENDMENT.md (v1, κ_far ≤ 0.66 certified,
   QMC demoted to consistency display, B_remote = 21.2153·r³; discriminating falsifier window rejects the
   κ = 0 sum 19.5465) — then round-2 V1 exposed the floor breach → the **v2 amendment** below.
4. **H5-AXIS v3 unfrozen / totals discipline** → **R4**: tightening amendment frozen (7fefa17b…, H5-AXIS v3
   in full; radius resolved at 0.062); rung-scoped merger + version-pinned v3 totals (8d7028e4…) +
   freeze/errata documents; I_lo rung-pollution repaired by the rung-scoped merger.

## 4. Round-2 findings and repairs (target: D1 v2.1)

- **Scope V1**: κ_far assembles 0.6773 > 0.66 at the certified floor; the G.7 consumption dropped from the
  register; the "zero MC" claim false → **REPAIRED AT SOURCE** by D3_REMOTE_AMENDMENT_v2.md (6796deea…):
  κ pieces and I_ann re-evaluated at R2's certified floor Z_lo (κ_cross 0.64990178 + κ_pair 0.0025176 +
  κ_y 0.0243696 = 0.677284905 → round-UP κ_far ≤ 0.68, cap covers BOTH the amendment's display and the
  assembly recomputation 0.67678902 — nit N-D3V2-1); B_remote = 17.6804 + 2.5282637·(1+κ_far) =
  21.9279·r³ (exact bracket 21.92788306…), genuinely Monte-Carlo-free; the v2.1 false claim recorded as
  FALSE CLAIM REPAIRED; 19.5465/20.9/21.2153/21.2658 all SUPERSEDED; G.7-at-the-rung recorded CLOSED with
  its mechanism; gate v3 rejects anything below 21.92788306.
- **Scope V2**: B4.loc's dam line must be a VALIDITY premise (no raw counted-class envelope exists;
  grep-verified across C2/D2/D3/B1/H4) → **REPAIRED** in D1 v2.2 §1: moved into Theorem (2)'s premise
  list with exact content, including the asserted-not-established cut-net ≡ D2 9-pin-tube (ii)
  identification; gate v3 cks its presence (removal kills the gate).
- **Round-2 focused re-reviews** (dispositions consumed by D1 v2.2; standalone re-review files are not in
  the tree — CANNOT-VERIFY as files, dispositions verified via the hash-verified D1 v2.2 body):
  **H4JC-R1 counterexample instance: PASS-WITHSTOOD** (event-level carrier attacked, no change);
  **numerics v2.1: PASS** for the v2.1 gate/chain with two BRANCH_dir receipts defects (owner's errata
  landed 2026-09-15: 7547e76a… — receipts grade only, "NOT soundness"; the certificate passes both modes,
  rung independently reproduced q = 0.999855 / counts 3112/888, falsifier catches all 4 mutations) and the
  κ-probe sensitivity flag (repaired at source by V1); the h3 CR5f mislabel recorded (E-H3-2);
  **scope v2.1: FAIL V1/V2** → both repaired in v2.2 as above.

## 5. D1's gate self-test catches (d1_falsify lineage)

- v2.1-era: the gate's body-only citation ck exposed by D1's OWN mutation test → whole-file ck (process
  break repaired; recorded in D1 v2.2's failed-attack ledger).
- v3 mutation self-tests (all killed; restored state PASS byte-identical): retracted-string injection →
  FAIL; upward-drift h5_totals_v4.json injection → FAIL; QMC-denominator (21.2153) substitution → FAIL
  F-remote-display; B4.loc-premise removal → FAIL F-B4loc-premise-bullet. The bracket ck was STRENGTHENED
  from the v2.1 form to consume ≥ 21.92788306 (the initially-loose ck tightened; extends D3's operative
  edge [21.27, 22.5]-class window into the assembly gate). 92 fail-closed cks, exit 0 both modes,
  transcripts byte-identical; re-executed live by this build agent (digest d800849e…).

## 6. Failed-approach preservation register (every failed approach, failure type, why it cannot be resurrected)

| Failed approach | Type | Exact reason it cannot be resurrected (carrier) |
|---|---|---|
| Dropped-square-root Cauchy–Schwarz (WP "missing-sqrt") | mathematical | Factor p_grad·√(E[det²H])·min(P_type,P_window) as an upper bound is invalid (needs √(min)); confirmed 175.3× at the witness; mechanical root-insertion too weak (~1e-2 zone class); quarantined (FAILED_APPROACHES #1); W3's harness mutates it in and CHECK2 kills it |
| Finite-rung tables + fitted exponents → all-small-r | mathematical | Extrapolation without a scaled asymptotic law (FAILED_APPROACHES #2); O(r^1.6) mislabeled O(r³) (#3) |
| T5 certificate defects; float64 9-pin Gram inversion; complex-arithmetic spectral covariance; double-hex-line encodings | numerical/custody | KeyError crash, silent garbage at det ~1.3e-42, NaN propagation, stale-hash-line defect (FAILED_APPROACHES #4–7); repaired by mpmath-100dps + Neumaier residual certification, real parity decomposition, single-hex-line discipline |
| Interval-box evaluation of the Λ-grid functional (DER-027b) | numerical | Structural cancellation junk ~7500× box width (FAILED_APPROACHES #8); abandoned for exact J2/J3 jets + exact midpoints + named premise H-B3 |
| Global sup-bound arithmetic for the Λ-grid | numerical | Magnitude compounding through frame/Neumann stages (#9) |
| Zone-wide value-kill as a uniform bound (C030 (i)) | mathematical | Rim band (92.8% of the upper-bound integral) and ridge arcs refute uniformity; the printed rigidity figure was a single-station value × full disk area; G-F2a gate FAILED-AS-WRITTEN and was waived (#10) |
| Direct tensor Gauss–Hermite on the tail-dominated 4D expectation | numerical | Unstable across meshes (1.1e-21 → 4.1e-21); replaced by exact 1D u-slice reduction (#11) |
| Cantelli as the in-slice saddle-probability bound near the window | mathematical | Too loose where Edet(u) is several SDs out (#12) |
| **Global-CS modulus 4.2·r^{3/2}** (W6 WP-min) | mathematical | Genuine singularity ρ_CS ~ ℓ^{1/2}d^{−4} near the pin cluster: modulus fails for r ≲ 0.003 (W12 B1, self-caught FAILED_APPROACHES #13); the truth is dead there (ρ ≤ 1e-171) — a bound artifact; repair = zone split, identified not executed; W12's B3 independently mis-quantifies premise F1 (4.236 > 4.2) |
| Missing-sqrt-restoration mutation (W13 F6) | custody (accepted gap) | No landed certificate implements the restoration mutation BY DESIGN: the defective factor is quarantined upstream and the WP-min closure's exact-integrand chain contains no such factor; all existing mutation classes verified fail-closed |
| **Isserlis mean-drop** (LEAD_SYMBOLIC_WP §3(A)) | mathematical | E[det²H] display false for nonzero means (missing −2ν_iν_jν_kν_l; overcount exactly 2(det ννᵀ)²; counterexample 7 vs 5; witness absolutes inflated √1.34488 — conservative direction, 22+ orders of margin absorb it where inherited; record correction owed) (W12 B2) |
| **LPW_CONSTANT v1** | mathematical | D1: published decimal above the exact fraction (rounded denominator); D2: false identity E[(|X|+|Y|)⁴] = 12 + 16/π (true 12 + 32/π). Frozen historical; v2's amplitude majorant is the only valid form; v1 caps survive conservatively, its decimal/derivation do not |
| **H4-JC false pointwise form** (H4_CLOSURE §2.3) | mathematical (level conflation) | Every α-qualifying saddle is an N_loop member (proved + counterexample); the once-payment at that grade is FALSE; only the event-level form H4JC-R1 survives; the retracted §2.3 is preserved as historical and gate-banned from citation |
| **κ = 0 bracket (19.5465·r³ class)** | mathematical | D3-THM carries (1 + κ_far) with κ_far > 0; the falsifier's named gate rejects 19.5465 (window [20.54646, 22.5]); the assembly gate rejects below 21.92788306; no polarity argument making κ_far = 0 exact exists in any carrier (scope CANNOT-VERIFY) |
| **C⁰ single-slice dam for B1.dir** (H4-SD) | mathematical (certified impossibility) | Theorem H4-SD (slice-dam vacuity): no single-slice C⁰ dam prices B1.dir at o(r³) — the smallness is irreducibly the interaction of channels; certified, not merely observed |
| **C¹ tube route for B1.dir** | mathematical (certified-dead) | The entropy/RN budget blows up (4 ln C_RN + 2 ln 2 term), killing the o(r³) tube route; displayed certified-dead per H4-SD's record; forced the NearSwap/FarRoute split |
| **W8 v2 rung-stability track** (DER-027b draft) | custody + incomplete | Six placeholders/unfinished sections itemized (expected-verdict language, unattached hashes/transcripts, v1 script without receipts); formal nonclosure successor receipted at the certificate standard with the exact missing piece named |
| **W8 scalar-SV Phase-2 track** | numerical | Scalar Leibniz arithmetic is cancellation-blind to the W-congruence (~400× invisible; certifies 3536 vs TRUE 0.0182 — 2e5× inflation): an all-scalar sweep would FALSELY ATTRIBUTE soundly-certifiable cells as failures |
| **The refused 31h all-ledger sweep** | process (refusal receipted) | Exactly the falsely-attributed all-ledger outcome above; REFUSED and never launched (PHASE2_SCOPE.md lead-mandated ledger entry); replaced by the J3 route with depth-capped subdivision |
| W8 depth-0 σ⁴-box freeze (Phase-2 mid-course) | numerical (bug, resolved) | Froze sups at coarse depth-0 values; the X9 Neumann tower (s4 ~ 1e15 vs TRUE 1.3e3) then never shrank → unclosable type gate; REVERTED to current-sub-box sups; the 6 ledgered cells declared INVALID under the fixed build; banks cleaned before relaunch |

## 7. B1 adversarial batteries (successful defenses)

B2a outside-cover, B2b boundary (NO-BREAK-WITH-CONCERNS: three Lemma R proof-scope gaps G4/G1/G5 named —
closed by H2 reg_lemmas Lemma R countersignature), B2c ties (B3 ⊂ N, P_r = 0), B2d torus/global
(NO-BREAK-WITH-CONCERNS): the frozen taxonomy survived all four; coverage gate satisfied.
