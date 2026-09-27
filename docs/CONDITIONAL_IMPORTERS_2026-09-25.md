# Conditional importers register — 2026-09-25

Status: NONAUTHORITATIVE sidecar. Adds no source of truth, re-proves nothing, and flips no flag.
Branch base: `chatgpt/drive-github-hardening-20260919` @ `077464ef5e2859ce98cbb9307799d5867a820eaf`.
Source audit: agent 13, `AGENT13_SOT_CARRIER_CLASSIFICATION_2026-09-25` (Drive `1iIB4tI2FSSBl066hDIlrTidM2NeuT2za`),
which lists 155 importing sentences: 142 AGREES, 13 STILL_CONDITIONAL.

This register closes out those 13 STILL_CONDITIONAL sentences. Each one is marked here as
conditional on its named absent carrier, or rescoped to the domain its lower result actually covers.

## Why this is a sidecar and not an in-place edit

All 12 repo-side sentences sit in files whose bytes are pinned elsewhere in the tree
(`drive/source_map/Files.csv`, `drive/source_map/Payloads.csv`, `drive/inventory.jsonl`, folder
`_MANIFEST.jsonl` files, `governance/PROVENANCE.json`, and for `H5_PROMOTE.md` also
`research/parallel/c2/source_binding.json` and `research/parallel/c2/c2_band.py`). Ten of them are
Drive mirrors, whose bytes must keep matching their Drive originals. Editing any of them in place
would break a pin or falsify a mirror, and re-pinning a mirror would stop it being a mirror. So each
sentence is marked here, at its exact `path:line`, and the pinned bytes are left untouched.

The one Drive-only sentence is listed at the end for the Drive Lane Librarian.

## Flags (unchanged)

`lemma_closed=false`, `prizes_solved=0`, `discharges_OBL_H5_JETMOD=false`, `certified_C_H=false`,
`freeze=false`, `inventable_attempt_accepted=false`. OBL-H5-JETMOD stays OPEN.
Engineering is not discharge.

## A. `rnu_env` (historical `rnu_env.py`, CL-RNU-001) — 2 sentences

Carrier status: CARRIER_ABSENT_STOP_REPLAY (per-file record: agent 15; absence scans: agent 11).
Only smaller-domain result on hand: certified whitened residual-form q=2 envelopes over a box
(`docs/math_status/STATUS.md:14`). Complement not covered: q=3, q=4, the log q(A, m′) composition, and C_H.

Path prefix `K` = `drive/mirrors/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY/05_ANTHROPIC_AUDIT_STATE_REGEN_2026-09-15`.

| # | path:line | sentence (prefix) | marking |
|---|---|---|---|
| A1 | `K/RN_UNIF_2026-09-16/CL-RNU-001_RN-UNIF_ENGINE_STATUS_AND_CLOSURE_PLAN_2026-09-16.md:6` | RECEIPTS: `rnu_run1.txt` …, `rnu_env.py` … | CONDITIONAL on `rnu_env.py`. The "in this folder" claim is not met: the mirror folder and the Drive hub `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG` do not carry it. |
| A2 | same file `:66` | The whitened residual-form envelopes are honest in magnitude (`rnu_env.py`, d = 5): values 0.002–0.07, … | CONDITIONAL on `rnu_env.py`. The quoted magnitudes are author-reported and cannot be reproduced from any carrier in the repo or on Drive. |

Successor (owner: ENV-RESCOV walk-down, agent 12, PR #99): CL-RNU-001 §5 item 1, the over-cell bounds at orders 2–4.

## B. `CL_ANTHROPIC_BUNDLE_2026-09-16_v3` — 4 sentences

Carrier status: CARRIER_ABSENT. This is a separate node from the `2026-09-17_v5` bundle; the two
share a name stem but not contents or hypotheses. v3 is described as "delivered in chat" and has no
Drive or archive carrier.

| # | path:line | sentence (prefix) | marking |
|---|---|---|---|
| B1 | `K/00_LANDING_NOTE.md:22` | Local bundle zip (complete, 87+ files): CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip, delivered in chat. | CONDITIONAL on v3. "Complete, 87+ files" is author-reported only. |
| B2 | `K/D1_v2_3_DRAFT/D1_V2_3_RECEIPTS.txt:20` | CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip; the Drive copies in this folder … must be re-hashed by the gate before consumption. | CONDITIONAL on v3. The sentence already declares itself conditional. The canonical bytes of `D1_ASSEMBLY_v2_3_DRAFT.md` and `d1_falsify_v4.py` are unverified against v3. |
| B3 | `drive/mirrors/2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY/06_UNMIRRORED_FROZEN_CARRIERS_BYTE_NATIVES/CL-MIRROR-001_MANIFEST.sha256.txt:5` | # CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip (delivered in chat, …) | CONDITIONAL on v3. The manifest names v3 as its byte source, and that source is absent. |
| B4 | `drive/mirrors/2026-09-16 — PEER_REVIEW_SUBMISSION_PACKAGES/PKG-01 — U2D_CONDITIONAL_UPPER_D1_v2_2/13c_LANDING_NOTE.md:22` | Local bundle zip (complete, 87+ files): CL_ANTHROPIC_BUNDLE_2026-09-16_v3.zip, delivered in chat. | CONDITIONAL on v3 (PKG-01 copy of B1, identical bytes). |

Successor: none. No parent text defers a calculation for the v3 bundle identity.

## C. F-map, H5_PROMOTE §3(i) and §3(iii) — 6 sentences, rescoped

Carrier status: `explicit_interval_map_F_G12box_to_Rplus` is CARRIER_ABSENT_STOP_REPLAY (agent 11).
H5_PROMOTE §3(i) defines F only as "the cover's explicit interval computation", meaning the cover
run itself. No separate map from G12 boxes to the positive reals exists.

**Rescoped domain.** These sentences hold only at the point rungs where H5_PROMOTE §3(ii)
(`research/parallel/c2/sources/H5_PROMOTE.md:53-59`) displays interval-enclosed values:

r₀ = 0.05, r₁ = 0.0354 (0.035355), r₂ = 0.025, r₃ = 0.0177.

**Complement (not covered).** Every open band between consecutive rungs,
(r₁, r₀) = (0.0354, 0.05), (r₂, r₁) = (0.025, 0.0354), (r₃, r₂) = (0.0177, 0.025),
and every r < r₃ = 0.0177, which the displayed ladder does not reach. The band inequality
"Î(r)/r³ ≤ F(G12-band) for the whole band" is a proof step that has not been executed on any of these bands.

| # | path:line | sentence (prefix) | rescoped reading |
|---|---|---|---|
| C1 | `research/parallel/c2/sources/H5_PROMOTE.md:50` | Î(r)/r³ = F(G12(r)) | Per-rung point identity only, at r ∈ {r₀, r₁, r₂, r₃}. It does not assert an explicit map on G12 boxes. The band version is OPEN on the complement above. |
| C2 | `research/parallel/c2/sources/H5_PROMOTE.md:67` | G12-band enclosures, hence Î(r)/r³ ≤ F(G12-band) for the whole band — a | CONDITIONAL. It holds at the rungs only. On each open band in the complement it is OPEN and requires the explicit interval map plus uniform band tails (OBL-H5-JETMOD). |
| C3 | `docs/FULL_DOCS_MATH_READ.md:21` | > *Band certification (the proof step).* … hence Î(r)/r³ ≤ F(G12-band) for the whole band … | Verbatim quote of C2. Same rescoping: rungs only, complement OPEN. |
| C4 | `drive/mirrors/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/CHART_SIDE_JETMOD_PLAN.md:36` | (same quote) | Mirror of C2. Same rescoping. |
| C5 | `drive/mirrors/2026-09-16 — HOLD_NOT_FOR_SUBMISSION/FULL_DOCS_MATH_READ.md:21` | (same quote) | Mirror of C3. Same rescoping. |
| C6 | `drive/mirrors/2026-09-16 — PEER_REVIEW_SUBMISSION_PACKAGES/PKG-01 — U2D_CONDITIONAL_UPPER_D1_v2_2/17_CHART_SIDE_JETMOD_PLAN.md:36` | (same quote) | PKG-01 mirror of C4. Same rescoping. |

Successor (owner: OBL-H5-JETMOD lane), quoted from H5_PROMOTE §3 "Named sub-obligations":
lattice-tail constants re-certified uniformly in the band ("LAT's tail bound currently certifies at
point separations"). The refinement in `research/bands/README.md:224-236` (binding 1) asks for certified
A, B (or A, p) dominating |He_{a1}(dx)·He_{a2}(dy)|·exp(−|z|²/2) at every order the 24-jet set reaches,
including orders 5 through 10. Nothing in this repository establishes those constants.

Duplicate-node note: the point identity in §3(i) and the band inequality in §3(iii) have different
hypotheses (point r against interval r), so they are different nodes. `explicit_interval_map_F_G12_band_to_Rplus_AND_Lip_F_G12`
is a third node (band map plus a Lipschitz bound) and is not the G12-box map.

## D. Drive-only sentence — for the Drive Lane Librarian

| # | Drive location | sentence (prefix) | marking |
|---|---|---|---|
| D1 | Drive `1aCa-QG9CSrNUB9SUFKISifghSf-41fRy`, `CL-RNU-003_PIECE1_RUN_PIECE2_CHARACTERISED_2026-09-17.md` line 5 (sha256 prefix `59b8f002`) | RECEIPTS (bundle CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip, folder RN_UNIF_2026-09-16/): `rnu_t4.py` … | CONDITIONAL on the v5 bundle (CARRIER_ABSENT, per-file record by agent 15). The Piece-1 coverage (1,170 closed and 561 pending cells) and "Piece-2 integrand certified" rest on these receipts and are author-reported at report time only. The repo already records this as CANNOT_VERIFY in `drive/mirrors/2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM/RN5_NEAR_MOMENT_REPAIR.md:138`. Suggested Drive action: add a conditional banner or sidecar next to the file; do not edit its bytes. |

## Tally

12 repo-side sentences marked (A 2, B 4, C 6) and 1 Drive-only sentence listed (D 1), for 13 in total,
matching agent 13's STILL_CONDITIONAL count. No pinned bytes changed. No sentence was upgraded to AGREES by this register.
