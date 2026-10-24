# Top verified recoveries

Session 01B6 historical recovery. Gap snapshot 2026-09-28; verification 2026-10-13. **Scientific effect: NONE.** These are recovered or located historical objects ranked by expected mathematical impact if they are reintegrated. None of them changes a status by itself. Agreement between models is not evidence of correctness. Every object below is author-side or carries only AI-lane review, unless stated otherwise, and all reviewer lanes run on one shared owner account, so no lane earns organizational-independence credit.

Locator conventions are those of RECOVERY_MATRIX.md (`cust` = custody DG-MIGRATION-20260919 on branch `chatgpt/drive-github-hardening-20260919`, "hard"; `blob` = git blob; `main#N cID` = GitHub comment). Quotes are copied from verifier-CONFIRMED fragments (line breaks normalized). Full sha256 values are given where the verifier recomputed them.

---

## 1. RN-FIELD-001 and its certificate: far-zone input jets and logQ bounds (G04, G06; bears on G05, G07, G09, G11)

- **Found:** the author-side proof RN-FIELD-001 and RN_FIELD_CERTIFICATE.json. They give Arb-384 uniform bounds, orders 0-4, on A, m and logQ for all 13,604 cells covering 5≤∣y∣≤17 at r=1/20, b=6/5. This is the content that the ENV-RESCOV and LOGQ-TAIL predicates ask for, in the far zone. The hardening NODE_ENV_RESCOV note and the Math- D0 audit were written without citing it.
- **Locator:** cust a19382f9 = hard blob 03c5d20c (RN_FIELD_PROOF, §4 l.154, §5 l.158-166); certificate member closure_round2/RN_FIELD_CERTIFICATE.json of hard blob c4065c67 (m18) = custody zip eb5fc2c2.
- **sha256:** proof `a19382f9ccf8a0a5936e44c6d015ca6ea47a60a0f54f9331b27d88675f847507`; certificate member `945079bf7a009fcd9524826bedcd2da975f684fe959823c650ee73b534dc2249`; zip `eb5fc2c2279787df628ac763c5f48df6cddf9cd8763b103074dc4f6cae0d3a6b`.
- **Verdict:** CONFIRMED, EXACT CARRIER (a2-d0math-F01, F02, F03). No rejection or quarantine was found. Quote: "These are actual field inputs to the previously derived RN calculus, closing the prior conditional-input gap for this sector and fixed r."
- **Caveats:** OpenAI-family author line; provider_independence 0; never replayed (the REV-RN3 reviewer did not obtain rn_field.py). It uses Cholesky whitening, where the historical route uses eigen whitening. It encloses B, F, z and D=E−BᵀB, not the node's literal (Y_j, V_j). It gives no 4th derivative of √χ² without a floor.
- **Reintegration:** a nonauthor replay of rn_field.py (d9167ae8) and its audit script; a written check that the two whitenings give equal norms; a successor note amending NODE_ENV_RESCOV §4 and naming RN-FIELD in Math- hist.LOGQ-TAIL. Optionally test it as the far-zone carrier for ALLCELL-FDZ-Q4 (G05).

## 2. Normalizer floor Z_r ≥ c r²: accepted r→0 carriers already in GitHub (G98, G41, G123; bears on G31, G97)

- **Found:** the PREMISE-Z-LOWER blocker has GitHub-resident accepted carriers for its r→0 half. (a) Parent (5.4)-(5.5) (existential floor 0<z_*≤Z_r/r² on (0,r_*]), accepted pre-snapshot in main#112 c5841352633 (N6). (b) D4 RN-FIXED-REMOTE-WINDOW (10)-(11), accepted in main#76 c5841783172 item 3. (c) The separated range [δ,0.025], derived from C102 in main#121 c5842152602 itself. The same theorem is stated as P02-LM-006 / LCR-DER-019.
- **Locator:** Math- blob dfed3b8d l.138-183; Math- blob b383bfcc l.102-146; comments above; LCR-DER-019 body in main blob e8d2e822 l.126-451 (custody docx 9dab5a5c).
- **sha256:** parent `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`; D4 `a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7`; LCR-DER-019 body `239ed094f70aa4352021572ad8bdc0c6ac6404edab4e1d7e61addec0c6659150`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A3D1-F01, F02, F15, F17, F34; A8-F07). DOWNGRADED: H3_CLOSURE.md (A3D1-F13; Drive-only, custody sha256 `387e4bae4881a036486f17f317ae5e26b7d94bd5e1e829cc76946d090fb81637`) is self-graded and existential. The H3_BAND_FLOOR certificate (A3D1-F14) is not established. Quote: "Joined to the separated-range bound, any positive `r_0` yields one constant on `(0, 0.025]`. No numerical value of `c` is required."
- **Caveats:** existential constants only. D4 was accepted at proof-sketch depth. The reviewers are self-identified Grok via Cursor on the shared account. C103 leaves the direction t free. The live-Drive lane's verifier would keep G98 at C: no single carrier states the floor on all of (0,0.025]. Post-snapshot main#229 c6002603423 passes LM006 conditionally but does not accept the lower-bound application.
- **Reintegration:** write a composition record for the C103 law with t fixed; bind the PREMISE_Z_LOWER gate to (5.5) + D4 (11) + the C102 range; mirror H3_CLOSURE.md; get one review.

## 3. RN3 far-zone joint proof: interval core and sources for its unsourced imports (G27, G02, G10; bears on G28)

- **Found:** the RN3 RN-JOINT-001 text; its unreplayed 384-bit interval core (rn_joint.py, audit_rn_joint.py, RN_MODE_REPLAY.json); and the sources REV-RN3 said were missing. The Kac-Rice integrand is eq. (L1) of LEAD_INTENSITY_DERIVATION.md and D3_PERCOLATION §3. The 0.6798 budget matches KAP_BUDGET 0.68 − CELL_MARGIN 0.0002 in d3_rn_unif.py; this identification is the agent's.
- **Locator:** hard blob d7a737e9 (l.5, 184-188); cust 1a4df0e9 m257/m63/m62; LEAD_INTENSITY_DERIVATION.md in cust a2136bc0 m736 (Drive-only in the owner's private archive); D3_PERCOLATION blob c4065c67 m132; D3_REMOTE_AMENDMENT_v2 m134; review main@a01c72f1:reviews/records/REV-RN3-FARZONE-20260918.md.
- **sha256:** RN3 `0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373`; rn_joint.py `6cad452dca32294e47add365f25f7e9c64df858615cead4f7ab3a50968ac7de5`; bundle `1a4df0e9fd264004d4ab876a8034fe61aea11d05e4d3c24873d0c3b741ff6766`; LEAD member `2a75ad7e47386878b899ffb70e4f7cc9ef34a552dfa7a6cd0e9739d7f0a739b3`; D3_PERCOLATION member `45f58a76749025aa21bfce53daa1c1f06092e677a67a9765fb9cd1d90434c12d`.
- **Verdict:** CONFIRMED (a2-d0math-F16, F17, F19, F20, F21, F22); d3_rn_unif.py DOWNGRADED nominally (mirror copy). Quote (RN3): "At the fixed rung **r = 1/20**, the complete far-region marked saddle intensity has relative upper correction **less than 0.12058**, below the requested **0.6798** budget."
- **Caveats:** author-side (OpenAI family). REV-RN3 is AMEND with independence 0, although it found no mathematical error. The (L1) ↔ RN3 ρ(y,v) mapping and the 0.6798 identification are the search agent's, not statements in RN3. The quarantined 17.6804 r³ import must be dropped.
- **Reintegration:** an RN3 v1.1 text amendment citing these sources; an out-of-tree nonauthor replay of the interval core; LEAD_INTENSITY_DERIVATION.md imported into GitHub with its hash.

## 4. Theorem B gate verdicts: a review map that already exists (G48, G49, G50, G51, G53, G55)

- **Found:** organizationally distinct verdicts for most Theorem B repair gates. CL-AUD-LSDER031 reviews the TB-G5 off-fold complement (R13-R18, R22-R24) and the C_* normalization (R12). CL-AUD-TBG3CHAIN covers LS-DER-021/022/023. CL-AUD-S2DER010 and Moonshot KIMI-AUD-016 cover TB-G1, with the M1 Borel amendment. CL-AUD-LSDER035 handles the LS-DER-024 counterexample and the corrected trichotomy; CL-AUD-LSDER036 approves LS-DER-036 v1.0; CL-AUD-LSDER030 covers the 576 area normalization.
- **Locator:** blobs 24b513db, 03a35080, 016cfa4a, 85fb1cf8 (main history, commit e5fc08ef); cust 595ea17c m4; cust 01b3f61c m242; cust ab07e7d7 m4; M1 amendment raw blob 5ea626ca.
- **sha256:** LSDER031 `4ae51f3eb3fe446220ea16dc224224f6c93a2577458c8fbcbec382b398c908ce`; TBG3CHAIN `67094cf06a86070b8426752142aae39426d5187d8b3dc810632f1cece42c3708`; S2DER010 docx `595ea17c1e88905c3d1f25c06ee8d3f8fa8db5680803a78926941323dc0f5930`; KIMI-AUD-016 text `1dbbbe62e5962b6fff6b1fe586f68590011ebac6a7cbb7e1a0ab5eff0be5c6f3`; M1 raw `e88536402a093ba7c08d3a5852b013023abd5b27717b44bf7574d5d4e31e11c1`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A7-F05..F12, F14). Quote: "R13. Reconstruct the off-diagonal Fourier strict-positivity argument imported from LS-DER-027. PASS, WITH TWO STEPS COMPLETED BY THIS REVIEWER."
- **Caveats:** mostly one Anthropic lineage reviewing OpenAI authors, and two TB-G5 steps were written by the reviewer. KIMI-AUD-016 is a distinct family but is not byte-bound to the reviewed object. A later register asks for a third provider on DQ-023. Theorem B stays RETRACTED. LS-DER-026, 032, 036-v1.1, 038 and 039-042 still lack exact-object reviews, and TB-G6/G7 are open.
- **Reintegration:** a gate-by-gate map binding these verdicts to the DQ rows; dispatch only the residual objects.

## 5. SIDE24 V3.3 and V3.4 archives located byte-exact (G138, G139, G141, G134, G135, G145)

- **Found:** the V3.3 adjudication checkpoint that every 2026-09-25/26 census called unrecovered is in custody. The V3.4 replay archive is on the hardening branch; it pins V3.3 and holds the facewise RP-C/RP-S closure originals.
- **Locator:** cust 4040d654 (12,388,862 B; internal ledger 161/161); hard blob b2901c1d = SIDE24_V3_4_AUDIT_REPLAY_FULL_ARCHIVE.zip (12,465,983 B; PR #126); members drafts/rp_c_rp_s_facewise_closure.md, 13_RP_C_RP_S_CLOSURE_DISPOSITION.md, drafts/palm_transfer_repair_v3_3.md.
- **sha256:** V3.3 `4040d654bddb1fc3b22eafa87bea7d0875166c03150d12602d6c1a0cb6385cfc`; V3.4 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`; facewise closure `3f84a4ad0136fceb463be2f2fde5dfd98d9593fb4c9297bf1fb249f55b4eaeeb`; disposition `a614fbd824be65fb1f67973e882491d0197515d6850d2b5c96080783f8057fd4`; palm repair `787fb6348e83b988ee1f6165c05726a0c15731008321152d6ec6057061f37343`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A1-F01, A1-F08, R-S24-135-01, R-S24-138-01/03, A8-F24, A8-F30). DOWNGRADED: A1-F02/F03. The archive holds no §C(a) r⁻⁷/s⁻⁷ table; a nested numeric pin-floor table (8be3f094…) is §C(c) material. Quote (V3.3 read-me): "**Project status:** **HOLD — candidate theorem not proved; do not submit**".
- **Caveats:** sandbox author line, model not stated. V3.3 itself is HOLD. The ratification AO48-OPR-045 is operator acceptance of an AI audit chain, not an independent review. The V3.3-tables slot may never have existed separately.
- **Reintegration:** a successor record binding 4040d654 to G138/G139 and to the eigenfloor README; a Math- binding to blob b2901c1d; a package manifest rebuilt from these bytes; an external review.

## 6. RN5 near-moment repair: the corrected method for the open Piece-2 cover (G29, G03, G22)

- **Found:** RN5_NEAR_MOMENT_REPAIR.md identifies the wrong-power envelope_v in the crude spine. It supplies the corrected Hölder(4,4,2) integrand, a centered Taylor L² remainder, a covariance error and marginal whitening, with code and certificates in the RN5 bundle.
- **Locator:** cust ac89f60b (l.18, l.129, l.144); hard mirror drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/; bundle cust 28c1c385 (round5/near_boxes.py, near_moments.py, verify_round5.py).
- **sha256:** `ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383`.
- **Verdict:** CONFIRMED, EXACT CARRIER (a2-d0math-F14, F15, F38). Quote: "The pinned `d3_perc.py` function `envelope_v` instead returns `(EA^4 EB^4)^(1/4) sqrt(EC^4)` and describes it as Cauchy–Schwarz twice."
- **Caveats:** author-side (OpenAI/Codex), zero independence. It covers only 10 boxes + 65 point certificates and is not a coverage certificate. RV-RN5 is READY / UNASSIGNED.
- **Reintegration:** a nonauthor replay of verify_round5.py. The full cover of 0.1≤∣y∣≤5 is new mathematics (G03 stays A).

## 7. Theorem A: pre-snapshot second-provider reviews of §§3, 4, 5 and §9 (G39, G43, G44, G45; bears on G40, G41)

- **Found:** xAI full-depth ACCEPTs of the frozen parent's Section 3 (C1-C5), Section 4 (R1-R5) and Section 5 (N1-N6, incl. the floor), and an ACCEPT of the §9 Borel repair v1.1 (C1-C6). None is cited in the Math- reconciliation LEDGER.
- **Locator:** main#112 c5841317882, c5841289285, c5841352633; Math-#37 c5841032512. The reviewed object is Math- blob dfed3b8d.
- **sha256:** reviewed parent `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`; n/a for the comments (re-fetched via the GitHub API by the verifier).
- **Verdict:** CONFIRMED, EXACT CARRIER (A3D1-F02, F03, F26, F27). Quote: "All five interfaces are **ACCEPT** on the small-r regime where the Section 3 imports below are in force. None is a counterexample or a block."
- **Caveats:** PR #112 is a never-main documentation PR, closed unmerged. The reviewer self-identifies as Grok 4.7 via Cursor; the provider is not externally attested and there is no organizational independence. Scope is §§3-5 only. Post-snapshot, C1 (Anthropic) ACCEPTs A1-A7 at existential scope, so these reviews add a second provider rather than closing anything.
- **Reintegration:** add the four comment ids to reviews/d1_chain_reconciliation_20260928/LEDGER.json as second-provider reviews of A2/A3/A4 and §9.

## 8. SIDE24 RP-A/RP-L/RP-F repair layer (G136, G134)

- **Found:** the V3.3 adjudication of the Kimi supplement. It accepts RP-A/RP-L only with seven printed-step repairs (palm_transfer_repair_v3_3.md), rejects the G.8 global Palm penalty (the ratio tends to 1/36), and keeps RP-F on compact positive marks. The Kimi successor supplement absorbed only repair 3 (erratum N1).
- **Locator:** 11_V5_FORWARD_BRANCH_ADJUDICATION.md in cust 4040d654 m31 and blob b2901c1d; successor supplement in cust 50bef1fc.
- **sha256:** adjudication `e7fbef9eaffc330c9ae1a7b6fca605ef3aa0d442d7138cc0e2152902024ad9e2`; successor supplement `33609b70d8c4d970d3594ebdc2cb59200508bfb2e53002ae7c6b10b3bc19d41f`.
- **Verdict:** the adjudication is DOWNGRADED (R-S24-136-01/02; still EXACT CARRIER). It is a critique by the adjudicator's own line, and its proofread is still open in the V3.3/V3.4 registers. The successor supplement is CONFIRMED (R-S24-136-03).
- **Caveats:** KIMI-AUD-006/006b and OPR-045 rely on the unrepaired G.7.1 and never answer V3.3. Post-snapshot main#389 (2026-10-13, author-side) corroborates V3.3 §3 but supersedes nothing.
- **Reintegration:** bind the V3.3 adjudication and its repairs into the package; get one nonauthor proofread of the repair together with G.7.1/G.8.

## 9. d=3 side-24 coefficient closed at reviewed scope (G60, G61, G62)

- **Found:** the Anthropic verdict chain CL-AUD-LSDER071/072/073V13/075/076, and LS-CLS-077, which identifies c_(3,24) = 0.0417759318… as the H0 short-lifetime first-moment coefficient. CL-AUD-LSDER062 reproduces the LS-DER-062 rational certificate.
- **Locator:** blob 8a6c2281 (071), blob dab4c821, cust 30d6acbc m4 (LS-CLS-077; blob fa19bcf9, hardening HEAD only), blob c67778ab (062).
- **sha256:** 073V13 `e04c48b0f47a288193621aef47c5168f612ba1f5f6abed67396df963de92222b`; LS-CLS-077 export `403bac366522d371a3d0a133b6e9b54659c769379a6d354d07f25aad15fc5910`; 062 `f4d1bbbb0dc4b8bca8a50dc98d62bddfe58fc59f7ba2cbad4bd648eaa842657e`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A7-F26, F27, F28). Quote: "ENERGY REPAIR PASS — SELECTED SIDE-24 COEFFICIENT CLOSED AT REVIEWED SCOPE."
- **Caveats:** one Anthropic lineage reviewing OpenAI authors. LS-CLS-077 is an author-side ledger. Its tie to the Math- Eq.15.2 constant is a 20-digit numerical match, not a source crosswalk. CL-AUD-LSDER062 credits only six objects. R-CERT (full-digit directed rounding) and import (33) remain open. Post-snapshot Math- 9d27fb4 reached REVIEWED_SCOPED by another route.
- **Reintegration:** cite the chain in the SIDE24 landing record for d=3; answer D3-F3; dispatch R-CERT.

## 10. H3 band ceiling/floor with the recorded tail-conversion defect (G19, G31, G21)

- **Found:** Kimi's all-r bracket 2.3066 ≤ E[G_r] ≤ 3.7477 on (0,0.05] (H3_BAND_FLOOR / H3_BAND_CEIL with scripts), and the later q0_twelve replay showing that iv_Phi converts Gordon-Mills tail bounds to binary64, so at x=−40 the enclosure becomes [0,0]. The floor script's iv_Phi lines 436-470 are byte-identical to the ceiling's.
- **Locator:** main blob 9b015fbd (q0_twelve zip) m60 (ceiling sources) and m52 (defect record, l.158-184); cust a2136bc0 m1111 (floor; on main since d41e57fc).
- **sha256:** H3_BAND_CEIL.md `18e109bc2e7a186c307f689f940ad476d69fc543e828081e0e706de2c578800d`; h3_band_ceil.py `b97c5428a20e5dd33f3e53653d20447f1e27515dcc7dc7a7c2004ebe30029452`; defect record `1c1f3c1881c45b31ab4916aded37e118f1fd7b88b8500e8c300ed733e40cac6e`; H3_BAND_FLOOR.md `7a40e2675232244bc63fcab2304b4b9cd3cd59a5c4965a3be5e65db7436fcfc2`; h3_band_floor.py `a907eeedd767b0b97461df8e237cdca130a607b5dede50ad5ba4b05c957d22a5`.
- **Verdict:** ceiling and defect record CONFIRMED (a2-d0math-F24, F25). The floor is DOWNGRADED (F26, A3D1-F14): it is not established as a certificate. Quote: "This finding does **not** show that the final historical ceiling is false: other slack might absorb the lost terms."
- **Caveats:** Kimi self-grades it Grade A. The defect sits on the load-bearing side of the floor, and the three smallest-r cells show exactly-zero rare-event pieces. Whether the lost terms change either constant has not been assessed.
- **Reintegration:** patch iv_Phi to outward rational endpoints; replay all seven cells for both scripts; review.

## 11. UPPER2D frozen carriers behind D1 v2.3 and the Stage-E targets (G23, G24, G25, G26, G36; G14)

- **Found:** B4LOC_DAMLINE.md (B4LOC-R1), PERC_DECAY.md, D2_BRANCH_CONTROL.md, H4_CLOSURE.md, C2_B_CLASSES.md and H4_JC_EVENT_LEVEL.md inside the 09152026 OKComputer gap-closure zip, plus the D1 v2.3 draft whose receipt digest reproduces.
- **Locator:** cust a2136bc0 (30,148,285 B) m1293, m1207, m1225, m1091, m1252; D1 v2.3 draft hard blob 249971e8. Post-snapshot, PR #199/#200 landed 46 natives on origin/main (D2_BRANCH_CONTROL.md excepted).
- **sha256:** container `a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`; D2_BRANCH_CONTROL text `aa35cad756856d0413ffc9dc6717e137e6e5bd7af6a983f28c26bf46b5b15fc7`; H4_CLOSURE `859696e254b3fac2d50437135963d9f3dfe00a33573e309a01ce8df326ac652c`; PERC_DECAY `f971b81a54dd2b78924c77e412e0c19b798e5356795057fcdadec3235e960745`; D1 v2.3 draft `15053f8bf3ad103d87c4f3bed82c0d5897d86533dbbe2cb365b24bdc0cce7d69`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A7-F17, F18, F19, F22; A3D1-F30). Quote: "with no raw counted-class envelope for B4.loc anywhere in the carriers, the dam line is the only route to ANY O(r^3) bound on that summand."
- **Caveats:** Kimi author-side. CL-AUD-001 is only a re-execution and custody check. D1 v2.3 is PROPOSED. Theorem H4-JC as frozen is false as stated; it is repaired by H4_JC_EVENT_LEVEL. The PD-CONN (i)/(ii) pieces and the sharp LOOP constant are open.
- **Reintegration:** replay D1_V2_3_RECEIPTS against the landed carriers; an operator issuance decision; add the H4/C2/D2 texts to the Math- STAGE_E packet.

## 12. D5 historical region carriers: pin collar, intermediate scale, nested disk (G67, G73, G74, G69, G70, G72, G76)

- **Found:** the 2D carriers from July 2026 that assert the all-height O(r³) region bounds which the D5 lane later re-derived without citing them: C100 Collar Cubic Closure, LS-DER-024 §5/§7, LS-DER-033 (exact endpoint determinants, envelope ≤ C r⁴ t⁻⁶), and the FlatSaddle Tail Lemma (nested-disk O(r⁵) under (U1)-(U4)).
- **Locator:** blob 04d6e1c5 (main history) = cust 24eb9d55; cust ce1eafee m4; cust e373caf4 m4; cust 3875016e.
- **sha256:** C100 `24eb9d55bcda133513392478d8aecb593db74b188b3f1c68629240e856241ce6`; LS-DER-024 docx `ce1eafee53c06bbe518c9bb22d29c3915e5e5ddb1350c969b6b187b90ba9724f`; LS-DER-033 docx `e373caf4a1f9ca81394770b26bd1219c58f68d444a0b68236f5112fb11d4c789`; FlatSaddle `3875016ea8663d0f9117089d2918e456191b28f1fde909c25a149b7df6e4999d`.
- **Verdict:** CONFIRMED (A4D5-F01 BYTE-IDENTICAL COPY; F02, F04, F05 EXACT CARRIER). DOWNGRADED: CL-AUD-TBG3B (F03) accepts only the station algebra and power counts, and its R2-R4 were later withdrawn. Lemma I Package (F07): several of its rows were contradicted in-program. Quote (C100): "Therefore the \(\rho^{-2}\) density and \(\rho^2\) determinant zero cancel, and \(F_r\) has an integrable pin-collision limit."
- **Caveats:** same-family and author-side. The density derivations were never reviewed, and the (U1)-(U4) discharge is author-side. The carriers are windowed and fixed at L=24. MODZONE limits LS-DER-033 but does not refute it. Post-snapshot Math-#138 supersedes these at planar existential scope, so their value is now an independent consistency check.
- **Reintegration:** cross-link them as predecessors in the D5 PROOF_INDEX; compare their ledgers with the P2 punctured-pin proof d8acf0bc.

## 13. P0.2 external conversions: KIMI-AUD-021 and KIMI-AUD-020 (G118)

- **Found:** third-family APPROVE verdicts: E2 closes (LCR-DER-061-v1.1 with the LCR-DER-057-v1.1 selector) and E0 closes (GP-DER-197). The register still lists Carrier B conversion as OPEN.
- **Locator:** main blob b3cb0e88 (KIMI-AUD-021 corrected build); cust 9daf84d0 (KIMI-AUD-020); LS-CTL-003-v1.3 in blob 0047bad1 m4 l.456; register hard easy_closure_queue.json blob d721510c.
- **sha256:** KIMI-AUD-021 `d9ab4d6cfce07677eddcf9cd47d3e63a7ec589037fee0ed3cd261f9525e1b532`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A3D1-F36, F37). Quote: "VERDICT: APPROVE - at its stated scope."
- **Caveats:** a Moonshot-family review; whether it earns independence credit is the operator's call. A stale build exists. The registers are unchanged since 2026-09-27.
- **Reintegration:** refresh easy_closure_queue P02-LM-013 and the claims-graph RV-LM013; get an operator ruling.

## 14. LPW 2D lower-bound package (G104, G105, G106)

- **Found:** the local-path lower-bound theorem LPW-CAND; Kimi Addendum 4 (which resolves the body-rule separator); the raw QC package behind the 1e-1235 fallback; and the LPW_CONSTANT v2/v4 successors.
- **Locator:** cust cf58f72e = hard blob adc4bd82 (l.52-58); cust a2136bc0 m381 (Addendum 4); hard blob 511717a4 (KIMI_QC_NUMERICAL_PACKAGE.zip) and blob 4ccff182; cust a2136bc0 m1409 (v4).
- **sha256:** LPW-CAND `cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5`; Addendum 4 member `09c0ada7e1db0e85fec7451877a8eead162e8133a7b537d0681c59b6b4ce15c1`; QC candidate `18a6e27cb945c7fc80975e20f775724989de0d368ce7a6f2ee29f11bea165d70`; v4 report `b06ee40c43f6e8aef266de2a7f4231fd8321676024f33c9132966b89b654b81c`.
- **Verdict:** CONFIRMED, EXACT CARRIER (A6-LB-01, 02, 07, 10). Quote: "It does **not** provide evaluated numerical constants. Its scope is fixed b=6/5, fixed side 24, dimension two, and the exact typed pair-Palm law above."
- **Caveats:** a single external family (Kimi) plus operator acceptance at review scope. The event identity LPW → GP-DER-118 adjacency is not supplied. v4 is on HOLD and consumes an unpromoted H3 ceiling (entry 10). Post-snapshot, the qualitative lower bound for every fixed d is accepted by another route (Math-#125/#127/#129).
- **Reintegration:** cross-link LPW as the d=2 explicit-constant special case; get one independent review of v2/v4.

## 15. H5-REPAIR-002: repaired kernel and consumer list (G30, G18, G21)

- **Found:** the H5 kernel defect analysis (odd absolute moments, missing normalizer tail), the repaired kernel 60c63257, one dominant-box replay, and machine-readable lists of the affected objects.
- **Locator:** H5_IMPACT_AND_REPAIR.md, member m8 of hard blob c4065c67 (l.15); H5_AFFECTED_OBJECTS.json and H5_DOWNSTREAM_WITNESSES.json in the same zip.
- **sha256:** `9fcdc965b8d4267a056473cebca3ebbc82cf7c2f146fd23b4880a6b316e9a389`.
- **Verdict:** CONFIRMED, EXACT CARRIER (a2-d0math-F31). Quote: "This is a scoped replay. It does not revalidate the complete cover, the remaining legacy arithmetic, H5 totals, D1, or all-small-r interpolation."
- **Caveats:** author-side; one box only.
- **Reintegration:** use H5_AFFECTED_OBJECTS.json as the checklist for a full nonauthor replay; rerun the rung ladder.

## 16. SARD-G: A2-A5 review and the d06b2eff defect (G101)

- **Found:** the standalone RI1-RI4 lemma, the in-source rho/tau clearance version, the A2-A5 ACCEPT review, and the post-snapshot audit showing that condition 2 of d06b2eff is unsatisfiable.
- **Locator:** Math- blob 0ee24f95; hard 4d9784f7 (blob d06b2eff, l.24-57); main#122 c5841682597; main#122 c5877292759.
- **sha256:** RI lemma `37020f4d44084faf50ed3665a18593e9554b8bce38096a2c43d8ea5d1bf2da03`; d06b2eff `c0460ff394ef338c1090cfaff2b9ef7d24ce17d547ae0b87448c077d426654b6`; reviewed object of c5841682597 `2f02495e272bc26054311cea03b66ee1de0ca6079dd097deb3084794a9514032`.
- **Verdict:** CONFIRMED (A9-F01, F02, F04; F03 STRONG HISTORICAL REFERENCE). The Clarification A/C citation was DOWNGRADED (F05): it is a paraphrase, and the originals are in KIMI-AUD-022 (blob e968385a). Quote: "A1 and A6 need a local predicate fix. A2–A5 hold."
- **Caveats:** the reviews ran on the shared account and are nonauthor by provider family only. Lemma OPEN is a sketch. Post-snapshot, the robust-charts successor (Math- 7e223706) is scoped ACCEPT, so the original A1 stays AMEND as written.
- **Reintegration:** re-status G101 against the 2026-09-29 records; add d06b2eff and both comments to the row's sources.

## 17. P14-E disjoint-substitution identity (G90)

- **Found:** the identity that REV-P15-D records as absent from its archive: P14 package proofs/04_LIMITATIONS_AND_HIERARCHY.md §E, (E1)-(E2).
- **Locator:** cust 1e11b246 m81 = main blob f55d426e (DG-MIGRATION pack 55e00c96).
- **sha256:** `a90833637b7180094a63b03740424dcf10551182b83da82411a7480cc5b2863c` (member text).
- **Verdict:** CONFIRMED, EXACT CARRIER (A7-F32).
- **Caveats:** author-side; naming it "P14-E" is an inference from the registry label "P14-D/E".
- **Reintegration:** bind the bytes to REV-P15-D and re-review the P15-D composition.

## 18. SIDE24 custody items: cross-link only (G142, G143, G144, G145, G146, G147)

- **Found:** arithmetic_ledgers.py in two identities with its receipt; the raw KIMI-AUD-006/006b audits and their identity reconciliation; the intact audit-evidence members behind a corrupt zip; the 18,701-B eigenfloor verifier; the three V3.4 diagnostics; and the executed Task-6 comparison AO48-AUD-047.
- **Locator:** blobs 68da0f8c, cf1434d6, d72a3f9d; ca975de0, 6908cf6d; b2901c1d members; 2c7fdde7; 81444dd4, 2884f63a, 80160588; ec3e74aa.
- **sha256:** arithmetic_ledgers.py `40ad76a1973248d9d1a48ab093b9a2b6bcebf4a6cd8c36c0a09067ea64109f4c` / `ffbc6ccb48c08f78f4ab169aa98fba1c3eb0d96ed36c5fbf451d4a45e1b8bea4`; KIMI-AUD-006 `17c8eba9fcc4a5d1475d7f9e6f3b51ff060aa5ecca78ee879617a20ea0c127fd`; 006b `2a38f2d412b4a7b51f008fc5593889b2598fbe01197ad1d873646bb7444877ce`; audit findings `e9df4b6f2db277199843aebdf206b61fb361ec0537aa67afc17cf106e5dd274a`; verifier `17209cb60d74c4caa4bc7988e1f1c0b8615c98835d5c9ee2ddf812e534ce2d4d`.
- **Verdict:** CONFIRMED (A1-F04..F08, R-S24-141..147). Quote (receipt line): "ASSEMBLED_EPSILON_24 < 1e-180 (dominated by 5e-185 quadratic + 1e-210 linear-tail)".
- **Caveats:** the 1e-180 line is a literal print over declared bounds. It is not a derivation of |ε_24| < 1e-180 and must not be cited as one.
- **Reintegration:** cross-links and Math- bindings only.
