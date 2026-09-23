# SIDE24 — Lemma A.4.1–A.4.3 Audit (3.4b deterministic core) & Session Blockers
**Date:** 2026-08-01 · **Auditor:** Claude (independent) · **Source:** `SIDE24_v4_2026-08-01/build/supp_body.html`, Gap item 5 ("Consolidated capture/escape proof"), held locally inside the pinned P3 archive (`bd01de23…c0f`) — byte-clean provenance, no transfer needed.

---

## 1. Blockers recorded (fail-closed disclosure)

**[B1] Annex rerun blocked.** Direct download of the canonical R2 package (`SIDE24_PRE_REVIEW_2026-07-31.zip`, SHA `14eaf7d4…`) failed: DNS resolution to Drive is denied at the container egress proxy. **[B2] Byte-faithful Drive→container transfer unreliable** via chat context (base64 clipping/transcription, incident E7). **Unblock:** re-upload the R2 zip to the chat — uploads mount byte-clean at `/mnt/user-data/uploads` and hash-verify against `14eaf7d4…`. One upload simultaneously unblocks work-order items 1–2 (full R2 Lemma 3.4a/3.4b text) and 5 (14-pair reproduction annex).

**Pivot taken:** the v4 supplement's Section A.4 — this lineage's written consolidated capture/escape argument, i.e. the deterministic core of the Lemma-3.4b / RP-A–RP-L interface — is fully present locally and was audited now.

## 2. Independent verification — `verify_34b_capture_escape_v1.py`

**35 checks, fail-closed, `ALL_ASSERTIONS_PASS`, normal and `-O` byte-identical.** Machine-checked from scratch, exact Fractions/sympy:

- **Setup (A.10).** p₀(−½)=0 (max level anchor), p₀(½)=−1/6, p₀(5/4)=49/192, p₀′=X²−¼.
- **A.4.1 (endpoint energy slope).** The Young step is an exact perfect square; the weight chain (τ≥2R ⇒ (3R/4)/τ≤3/8; 1−¼−⅜=⅜) yields E(v_Y)≤(8/3)(R²/τ)|v_X|² ≤ the stated 4(R²/τ)|v_X|².
- **A.4.2 (outward energy cone).** q=2Rξ and |Y|≤ξ/(KR⁴) exact at the constraint floor; F_X≥13ξ/16≥3ξ/4; all four adverse brackets re-derived — (1+δ)/4≤257/1024 (δ=1/512<1/256), 257/(1024K), 1/(2048K²), 2ε₀ — **exact sum 8625610753/34359738368 ≈ 0.25104 < 1/3** ⇒ flux margin −(2/3)q².
- **A.4.3 (outward strip + level).** q=3R/4 exact; strip max X²−¼=21/16 gives ratio 7/8 exactly; ratios 5/(8K), 3/(32K²), (4/3)ε₀ re-derived; **exact sum 1409532425/1610612736 ≈ 0.87515 < 9/10**; entry 4δ²<9/16; **level ledger independently reconstructed over 2²⁴: 2688+1+12+6144+1024 = 9869**, and 9869/2²⁴ < 1/768; arrival level 49/192−1/768 = **65/256 > 1/4 > 0 = P(M)** — the outward branch exits strictly above the maximum level. The energy adaptation (½E* ≤ 9/(64KR³) < 1/(4K)) confirms **no λ_max(T) bound is used** — the point of the v1.3 energy formulation.

**[REVIEWED, not machine-checked]** A.4.4 (inward capture), A.4.5 (exact-field C¹ robustness), Theorem A.4.6 (assembly), the ODE invariance/flow arguments, and the Jacobian block reduction (A≥3/4, |B|≤R, ‖E_D‖≤3R/4 "after one fixed small-r reduction"). These are next in the A.4 audit queue.

## 3. Register impact

| Item | Status |
|---|---|
| A.4.1–A.4.3 exact algebraic content | **VERIFIED** (independent, fail-closed, 35/35) |
| Deterministic capture/escape (Gap item 5 / Module J replacement) | WRITTEN; core inequalities now independently machine-verified; flow/assembly steps pending |
| RP-A/RP-L probabilistic transfer (O(r³) failure probabilities; r⁵ weighted shallow-eigenvalue/coarea; items F.4 (i)–(v)) | **OPEN** — unchanged, per GAP_REGISTER_V4 §C.1 |
| RP-C / RP-S joint blow-up compactifications | OPEN |
| Candidate theorem | **HOLD** |

**Next:** (1) A.4.4–A.4.6 checkable content; (2) on unblock (R2 zip re-upload): full 3.4a/3.4b text audit + 14-pair annex rerun.

## 4. Run record
Python 3.12.3, sympy 1.14.0. Artifacts: `verify_34b_capture_escape_v1.py` + `.out.txt` + `-O` transcript (byte-identical), delivered in chat outputs and mirrored to the Drive `08-01-2026` folder.
