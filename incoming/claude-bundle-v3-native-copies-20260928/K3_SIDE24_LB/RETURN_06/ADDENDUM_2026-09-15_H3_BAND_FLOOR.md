# ADDENDUM 2026-09-15 — H3 BAND FLOOR frozen: normalizer sub-part of OBL-D1-PROMOTE DISCHARGED (uniform modulus on (0, 0.05])

Append-delta to RETURN_06 (assembled 2026-09-15T00:05:36Z). No frozen file is modified; every hash below was
recomputed FROM BYTES by the addendum agent on 2026-09-15 before being cited. Root: /mnt/agents/output/K3_SIDE24_LB/.

## 1. The new carrier (frozen in UPPER2D/H3_closure/, record H3-BAND-FLOOR-2026-09-15)

| artifact | rule | recomputed sha256 | bytes | verdict |
|---|---|---|---|---|
| UPPER2D/H3_closure/H3_BAND_FLOOR.md | body-excl-hash-line (whole file minus the final `Body hash of this document: …` line; LF) | 281477c39412840bc9ca56a256884f3ddd5f0ae4283f90a017162676771382a7 | 11133 (body 11029) | MATCH vs lane MANIFEST.sha256 + RECEIPT_h3.json |
| UPPER2D/H3_closure/h3_band_floor.py | whole file | a907eeedd767b0b97461df8e237cdca130a607b5dede50ad5ba4b05c957d22a5 | 37315 | MATCH |
| UPPER2D/H3_closure/band_normal.txt | whole file | dcb3c8e6152876021c8b116903c9a7c1ac6be7417a9b4042b89c00b911fe565d | 2835 | MATCH; ≡ band_O.txt BYTE-IDENTICAL |
| UPPER2D/H3_closure/band_O.txt | whole file | dcb3c8e6152876021c8b116903c9a7c1ac6be7417a9b4042b89c00b911fe565d | 2835 | MATCH (both interpreter modes) |
| UPPER2D/H3_closure/band_out.txt | whole file | 617b2d07fcaef23313427e68020182d0b27cad78032b399336a1b925db489f54 | 2743 | MATCH; equals the in-transcript self-hash line `sha256(transcript-above)` |
| UPPER2D/H3_closure/band_mut_planar.txt | whole file | 4fc8105d1050dc2a284750ada417f959725711b8e19865ced682192da59e4774 | 305 | MATCH; CK_FAIL at CB1a (exit 1, as designed) |
| UPPER2D/H3_closure/band_mut_window.txt | whole file | ff1765d4ace7fb5c61402e227c5896017bd1995cad0e2f8475761a3850a01c51 | 833 | MATCH; CK_FAIL at CB3 window identity (exit 1, as designed) |
| UPPER2D/H3_closure/RECEIPT_h3.json | whole file | ff788042c6bce3bb89bccee17ee88478b81b72426b5b474929a009b0fe8ee209 | 4482 | MATCH; records all of the above |

Extraction-rule note (verified from bytes, recorded for the capsule): the capsule's generic `sepbody` rule
(bytes before the first `"\n---\n\n"`) does NOT apply to this file — the document carries a separator after
its header block, and sepbody recomputes to 94089e9f696e520cd660b37f53d5e472b217134adfafe4c1c06939217107d19d,
NOT the registered value. The lane-FREEZE-recorded rule is `body-excl-hash-line` (annotated verbatim in the
lane's MANIFEST.sha256 as `H3_BAND_FLOOR.md(body-excl-hash-line)` and keyed identically in RECEIPT_h3.json);
under that rule the recomputed body hash is 281477c39412840bc9ca56a256884f3ddd5f0ae4283f90a017162676771382a7
— an exact MATCH from bytes.

Custody supersession (no frozen file touched): SOURCE_CAPSULE.sha256 Section B line for
`UPPER2D/H3_closure/h3_band_floor.py` (sha256 9eb0f426…, 36181 B, custody-only) and TREE_MANIFEST.sha256's
lines for `h3_band_floor.py` (5506e648…, 37254 B) and `band_normal.txt` (empty, e3b0c442…) are pre-freeze
snapshots of a lane both files' headers explicitly declare VOLATILE; the addendum values above supersede them
for custody going forward. The frozen document carriers (H3_CLOSURE 387e4bae…, H3_RUNG_FLOOR 6347275d…) are
unchanged and re-verified by the lane MANIFEST.

## 2. Certified statement (quoted from the frozen body 281477c3…)

For EVERY r ∈ (0, 0.05]:

  E[G_r] ≥ 2.30659559567154 > c_Z = 1.615489267643502474048123284509081575382

hence the normalizer satisfies Z_r = r²·E[G_r] ≥ c_Z·r² UNIFORMLY on (0, 0.05]; worst certified margin
+42.78% (attained on the rung-adjacent cell (0.04, 0.05]). Per-cell certified floors (interval lower
endpoints, 15 digits):

| sub-interval      | certified E[G_r] ≥ | margin vs c_Z |
|-------------------|--------------------|---------------|
| (0, 0.0025]       | 2.58288222715895   | +59.88%       |
| (0.0025, 0.005]   | 2.58174158877406   | +59.81%       |
| (0.005, 0.01]     | 2.54243388604045   | +57.38%       |
| (0.01, 0.02]      | 2.45593199230367   | +52.02%       |
| (0.02, 0.03]      | 2.44022641168325   | +51.05%       |
| (0.03, 0.04]      | 2.41792545239159   | +49.67%       |
| (0.04, 0.05]      | 2.30659559567154   | +42.78%       |

Method (from the frozen doc): exact Laurent-series Gram algebra at r = 0 (corrected-basis poles cancel
exactly; genuine soft poles z⁻¹, z⁻² isolated and annihilated by the conditioning), exact series division to
J = 60, Neumann-certified correction on cells 2–7 (R-majorant series tails on the first cell only), interval
Horner evaluation, certified series tails, conditional-Wick truncated integrals + exact-Gaussian-tail
Chernoff/Cauchy–Schwarz rare-event pieces (t₀ = 3.6–9.7σ → ∞ as r → 0), certified Φ machinery (series
|x| ≤ 6 with geometric-tail certificate, Gordon–Mills outside). No whitening, no Monte Carlo in the
certificate (MC labeled diagnostic-only, consistent). Discipline: fail-closed ck → SystemExit; both modes
(`python3` / `python3 -O`) byte-identical transcripts; mutation suite caught at the designed guards
(MUTATION=planar_moments → CB1a; MUTATION=wrong_window → CB3).

## 3. Scope note (uniform vs point)

The uniform band minimum 2.30659559567154 (⇒ Z_r ≥ 2.30659559567154·r² everywhere on (0, 0.05]) is
necessarily weaker than the sharper rung-point certificate at r = 0.05: R2's certified
Z_{0.05} ∈ [7.7592917375327855e-3, 1.1468646473404396e-2], i.e. Z_{0.05} ≥ 3.1037·r² (H3_RUNG_FLOOR.md
6347275d…, unchanged). The band floor's dip at the top cell is enclosure loss, not physics (true E[G_r]
≈ 3.2, MC-labeled). The band certificate covers (0, 0.05] only; it says nothing about r > 0.05 and does not
replace the frozen point certificates.

## 4. Discharge statement

The NORMALIZER sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus, by the following exact
mechanism: the frozen certificate H3_BAND_FLOOR.md (body 281477c3…) certifies Z_r ≥ c_Z·r² for EVERY
r ∈ (0, 0.05] with c_Z = 1.615489267643502474048123284509081575382 — i.e. H3's existential theorem-grade
bound is now certified with an explicit uniform constant on the whole rung band, extending R2's pointwise
floors to the continuum. Exact mechanism per the certificate: interval-r carried through the no-whitening
Wick pipeline as exact Laurent series at r = 0 with pole-cancelled corrected basis, exact series division
(J = 60), Neumann/Cauchy-certified truncation corrections, and literal-bound certified tails; both modes
byte-identical; mutations fail-closed; receipt H3-BAND-FLOOR-2026-09-15.

The remaining promotion content of OBL-D1-PROMOTE is the CHART side — H5's r-scaled rung family + band
interpolation (OBL-H5-JETMOD / OBL-H5-REMOTE-THRESHOLD) — which remains OPEN / RUNNING (rungs r = 0.025,
0.035355 executing per RUNNING_TASKS.md; unchanged by this addendum). Theorem (2)'s r₀ stays EXISTENTIAL:
the chart envelope, not the normalizer, is now the blocking premise on that side.

## 5. Obligation-ledger delta (OBLIGATION_LEDGER.md, not edited — this delta governs)

- §1 OBL-D1-PROMOTE: **normalizer sub-part CLOSED** (discharge mechanism in §4 above); chart side OPEN
  (OBL-H5-JETMOD, OBL-H5-REMOTE-THRESHOLD unchanged; H5 rungs executing).
- §1 sub-obligation OBL-H5-ZBAND: the lo side now rides a FROZEN uniform certificate
  (Z_r ≥ c_Z·r² for ALL r ∈ (0, 0.05], body 281477c3…) — no longer an unbanked run; the hi side (band LPW
  bracket) remains OPEN.
- §8 "H3 band-floor assessment": **COMPLETE and FROZEN** — H3_BAND_FLOOR.md body 281477c3…, script
  a907eeed…, both-mode byte-identical transcripts dcb3c8e6…, transcript body 617b2d07…, mutation transcripts
  4fc8105d… / ff1765d4… (caught at the designed guards), RECEIPT_h3.json ff788042…. Supersedes the ledger's
  "State: RUNNING (script modified 07:58:13 CST; band_normal.txt created empty …)". One bookkeeping item
  remains with the D1 lane: optional gate consumption of the band floor (the rung consumption via R2 already
  stands); this is bookkeeping, not an evidence gap — the discharge rests on the frozen certificate alone.

## 6. Running-tasks delta (RUNNING_TASKS.md, not edited — this delta governs)

- §5 "H3 band-floor assessment — RUN COMPLETED, NOT YET FROZEN" → **COMPLETE, FROZEN** 2026-09-15: the
  missing items listed at assembly (FREEZE record, -O companion transcript, mutation harness run) all exist
  and are hash-verified above; band_normal.txt (empty at the TREE_MANIFEST snapshot, declared volatile) is
  now the frozen transcript dcb3c8e6… ≡ band_O.txt. The lane-summary row "H3 band | RUN COMPLETE, unfrozen"
  → "H3 band | COMPLETE, FROZEN | H3_BAND_FLOOR.md 281477c3… + RECEIPT_h3.json ff788042…".
- All other lanes (H5 promotion, W3, W8, BRANCH) unchanged by this addendum.

## 7. Executive-state one-line delta (00_EXECUTIVE_STATE.md, not edited — this delta governs)

H3 band-floor assessment: RUNNING → **COMPLETE, FROZEN 2026-09-15 — CERTIFIED uniform normalizer floor
Z_r ≥ c_Z·r² for EVERY r ∈ (0, 0.05] (band min E[G_r] ≥ 2.30659559567154 > c_Z, worst margin +42.78%);
the normalizer sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus; the chart side (H5 rung
family + interpolation) remains OPEN** (carrier H3_BAND_FLOOR.md body 281477c3…; CANNOT-VERIFY item 7's
"continuum band remains unexecuted" is superseded on the rung band (0, 0.05]).
