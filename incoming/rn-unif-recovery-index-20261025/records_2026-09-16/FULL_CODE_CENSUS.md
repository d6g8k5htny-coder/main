# FULL_CODE_CENSUS

**As of:** 2026-09-16 ~22:20 CT (America/Chicago)
**Base:** `/workspace/drive_peer_review_triage/`
**Index:** `FULL_CODE_INDEX.csv` (1939 rows)
**Rule:** Fail-closed. **D3-LEMMA-RN-UNIF stays OPEN.** Prize originals solved stay **0**. No promotions.

---

## 0. Standing verdict (do not upgrade)

| Claim | Status |
|---|---|
| D3-LEMMA-RN-UNIF Piece 1 | **OPEN** |
| D3-LEMMA-RN-UNIF Piece 2 | **OPEN** |
| `lemma_closed` (EXECUTE + T4 receipts) | **false** / status **PROPOSED** |
| D1 v2.3 draft | **PROPOSED** (not promoted) |
| H5_ZBAND consumption | **PROPOSED**; remote-threshold still rides RN-UNIF |
| Prize `original_prizes_solved` | **0** (Phase10 CLAIM_REGISTRY / METRICS) |
| PKG readiness upgrades this pass | **none** |

---

## 1. Totals

| Metric | Count |
|---|---:|
| Code index rows | 1939 |
| Local path rows | 1839 |
| Drive-id / mirror rows retained | 100 |
| Local code files scanned | 1839 |
| Docs index rows (`FULL_DOCS_INDEX.csv`) | 448 |

### By authority_lane

| Lane | Rows |
|---|---:|
| side24 | 1534 |
| prize | 157 |
| pkg | 122 |
| other | 85 |
| governance | 24 |
| rn_unif | 17 |

### By extension

| Ext | Rows |
|---|---:|
| py | 1846 |
| sh | 67 |
| lean | 26 |
| ipynb/js/ts | 0 |

Heavy duplication is expected: gap-closure / SepOK / OKComputer(4) / SIDE24 V3.* checkpoints / `_new` re-extracts share near-identical trees.

---

## 2. By lane — entrypoints + role

### 2.1 rn_unif (lemma **OPEN**)

**Local hub:** `extracts/RN_UNIF_2026-09-16/`
**Frozen engine (dependency):** `extracts/09152026OKComputer_Project_Gap_Closure/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py` (103166 B, sha `85d7725fab42…`)

| Script | Size | sha256 (12) | Role (skim) |
|---|---:|---|---|
| `rnu_ds3.py` | 9704 | `bd3074fd900f` | **PREFERRED** DS3 lift of kappa_far third derivatives (CL-RNU-002 infra) |
| `rnu_ds3_7p2kb_DUPLICATE.py` | 7201 | `c3d5adc8b887` | SUPERSEDED scalar DS3 |
| `rnu_execute.py` | 12567 | `89b36db4bc97` | EXECUTE driver; writes receipts; **does not claim lemma closed** |
| `rnu_t4_push.py` | 7924 | `7b7cc46ba560` | T4 candidate / Lipschitz probe; **PROPOSED** |
| `rnu_meanfix.py` | 2800 | `3693655f45d0` | E-RNU-1 mean_grad chain-rule fix validation |
| `rnu_chi2_white_v2.py` | 5005 | `6b61af7b549c` | Whitened χ² exact gradient helper |
| `d3_rn_unif.py` | 103166 | `85d7725fab42` | Frozen Piece-1/2 engine (imported by rnu_*) |
| `h5_zband_consume.py` | 6488 | `f5965c43f3fb` | H5 Z-band consume cert (PROPOSED; gated on lemma) |

**Receipts (authoritative flags):**
- `RNU_EXECUTE_RECEIPT.json` / `.md`: `lemma_closed: false`, `status: PROPOSED`
- `RNU_T4_PUSH_RECEIPT.json`: `lemma_closed: false`, `status: PROPOSED`
- Remaining-to-freeze (from EXECUTE): whitened env_form bounds orders 2–4; DS3 wired through kappa_far (not only FD Hessian); valid T4 from env_form not measured-scale
- CL-GROK-CLOSE-001 closes **session portfolio**, not the lemma

### 2.2 side24

Largest mass: pre-review ledgers, V3.1–V3.4 audit/recon checkpoints, gap-fill verifiers, KIMI packages, LPW returns, UPPER2D H2/H3/H5/B4LOC/D1/D3 trees inside gap-closure.

| Entrypoint | Path hint | Role |
|---|---|---|
| `run_all.py` | `…/06_REPRODUCTION_FILES/` | SIDE24 pre-review ledger runner (assert + ALL_ASSERTIONS_PASS) |
| `d3_perc.py` / `d3_perc_decay.py` / `d3_falsifier.py` / `d3_amend*.py` | UPPER2D/D3_percolation | Percolation / amend / falsify suite |
| `b4loc_driver.py` | UPPER2D/B4LOC_damline | B4.loc driver |
| `d1_falsify_v3.py` | UPPER2D/D1_assembly | Frozen v2.2 falsify gate |
| `h3_band_floor.py` / `h3_band_ceil.py` | UPPER2D/H3_closure | H3 band carriers consumed by H5 |
| `lpw_constant_v4.py` | LPW_CONSTANT/v4 | LPW v4 constant brick (HOLD / NOT READY upstream) |
| many `verify_*.py` | SIDE24_gap_fill + V3.* scripts | Checkpoint / collar / disposition verifiers |

### 2.3 prize

Phases 01–10 under `extracts/Prize_Research_PhaseXX_…` (+ `prize_research/` mirrors). Entrypoints: `verify_phase0N.py`, topic verifiers (`verify_laminar`, `verify_crossing`, …), `verify_manifest.py`.

**Receipt:** Phase10 `CLAIM_REGISTRY.json` / `receipts/METRICS.json` → `original_prizes_solved: 0`. No invented closures.

### 2.4 pkg

Local `packages/pkg01`…`pkg04` (+ hold / LPW v4 hold). Includes H5 run/promote/stitch scripts and LPW review numerics. Drive-native PKG code ids retained in index (lb*_certificate.py etc.).

| Notable | Note |
|---|---|
| `d1_falsify_v4.py` | D1 v2.3 **DRAFT** gate — PROPOSED; strict superset of v3 |
| pkg01 `h5_*.py` / `promote_*.py` / `*.sh` | H5 packaging / promote tooling |
| pkg04 `RB_R2_*` / `rc_r3_numeric.py` | LPW review brick numerics |

### 2.5 governance

DQ060 `verify_obligations.py`; OP-RECON `verify_installation*.py`; GP-FOR-189/192 Lean/research-formal-core; Peer_Review_Packets (mostly docs); FS2-related not heavily coded locally.

### 2.6 other

Residual scripts not clearly lane-tagged (misc extracts, small helpers).

---

## 3. Receipts summary (machine + human)

| Artifact | Path | Flag |
|---|---|---|
| EXECUTE JSON | `extracts/RN_UNIF_2026-09-16/RNU_EXECUTE_RECEIPT.json` | lemma_closed=**false** |
| EXECUTE MD | `…/RNU_EXECUTE_RECEIPT.md` | STATUS PROPOSED; lemma NOT closed |
| T4 JSON | `…/RNU_T4_PUSH_RECEIPT.json` | lemma_closed=**false**, status PROPOSED |
| Prize Phase10 | `…/CLAIM_REGISTRY.json`, `…/receipts/METRICS.json` | original_prizes_solved=**0** |
| D1 v2.3 | `extracts/_new/anthropic_cl_hub/D1_v2_3_DRAFT/` | DRAFT / PROPOSED |
| H5_ZBAND | `extracts/_new/anthropic_cl_hub/H5_ZBAND/` | consume cert PROPOSED |

Empty stubs `extracts/H5_ZBAND/` and `extracts/D1_v2_3_DRAFT/` exist; real bytes live under `extracts/_new/anthropic_cl_hub/…`.

---

## 4. Method / coverage notes

- Prioritized **local extracts** (find + sha256sum + skim headers of rnu_/d3_/verify_/run_all).
- Drive MCP **not required** for this local-complete pass; 100 Drive-id rows retained from prior successful pulls.
- One historically corrupt zip (`SIDE24_AUDIT_EVIDENCE_2026-08-01`) remains partial — noted in `ZIP_MANIFEST.csv` / prior census.
- Index includes duplicate trees; content-familiarization treats gap-closure + RN_UNIF + packages as primary authority mirrors.

---

## 5. Paths written

- `/workspace/drive_peer_review_triage/FULL_CODE_INDEX.csv`
- `/workspace/drive_peer_review_triage/FULL_CODE_CENSUS.md`
- `/workspace/drive_peer_review_triage/FULL_DOCS_INDEX.csv`
- `/workspace/drive_peer_review_triage/FULL_CONTENT_FAMILIARIZATION_2026-09-16.md`

---

*End code census. Fail-closed. Lemma OPEN. Prize solved = 0. No promotions.*
