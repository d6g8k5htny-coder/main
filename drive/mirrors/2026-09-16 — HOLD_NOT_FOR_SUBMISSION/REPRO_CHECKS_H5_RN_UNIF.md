# REPRO CHECKS — H5 + RN-UNIF (agent 5, fail-closed)

**Mapped:** 2026-09-16 ~15:22 CT (America/Chicago)  
**Role:** fail-closed repro only. **No premise status promotion. No theorem closure claimed.**

Sources: local Sep-15 extract under `extract_sep15/K3_SIDE24_LB/UPPER2D/` + Drive IDs from `PREMISE_BLOCKERS_D1_PROMOTE_AND_RN_UNIF.md`.

---

## Explicit non-claims

- Does **not** promote OBL-D1-PROMOTE / chart-side status.
- Does **not** claim D3-LEMMA-RN-UNIF closed, FREEZE, or Theorem (2) closure.
- H5_RUNG2/3 package certification language is quoted as artifact content only (rung packages ≠ premise discharge).

---

## 1) H5 lane

### Local extract

| Artifact | Path | Size | sha256 |
|---|---|---:|---|
| H5_RUNG2_2026-09-15.md | `…/H5_closure/H5_RUNG2_2026-09-15.md` | 5311 | `5f40b65287fbf2279e58d1f2c16e3f595af369fdad67e71e10ec274a036d2c28` |
| H5_RUNG3_2026-09-15.md | `…/H5_closure/H5_RUNG3_2026-09-15.md` | 3157 | `9cfa79bbb8202e7e41e291f4ab7f73339abce33d766f83f5e12279ac9d19d4f2` |
| h5_totals_r0.025_v2.json | same folder | 2385 | `f7697bcfa0fe32b5c87c8ef8adef5d4384ab1bda7cd4a7c4782fef8c99103fcb` |
| h5_totals_r0.035355_v2.json | same folder | 2399 | `808d6901e3254a73181aa696904ed04567d401b6e58300454d489046c36f4f64` |

**Body-sha256 (doc-claimed, verified):**

- RUNG2 claims `91a34d83688b5110b886ab4fc168a55030343cc0da4a98e38347d3b0e6e48102` — **PASS** (body = bytes before `body-sha256:` line, with trailing newline).
- RUNG3 claims `f4c3414fe8b072008f5f5c8c5676ec9e20944e115535e0f3dd9c13c1bb89c0d8` — **PASS** (same convention).

**Cited totals hashes in docs vs local files:** both **PASS** (exact match to table above).

Also present: `CODE_HASHES.txt`, `H5_RECEIPTS.txt`, `BODY_HASH.txt`, `BODY_HASH_PROMOTE.txt` (r=0.05-era / promote-era; not re-executed here).

### Drive verify

| Drive ID | Title | Drive size (metadata) | Local size | Byte match |
|---|---|---:|---:|---|
| `13eLq6qnVIxyGI-GE1L0T5TL5FKcj1qyZ` | H5_RUNG2_2026-09-15.md | 5311 | 5311 | **PASS** (sha256 identical) |
| `1IdE6t7TRBZqWLGRKNXBTi-pQfr2hVTcI` | H5_RUNG3_2026-09-15.md | 3157 | 3157 | **PASS** (sha256 identical) |

**Commands:**

```bash
# metadata via MCP user-Google-drive get_file_metadata
# content via: curl -fsL -o FILE "https://drive.google.com/uc?export=download&id=ID"
sha256sum H5_RUNG2_2026-09-15.md H5_RUNG3_2026-09-15.md
# cmp / byte-equal vs extract_sep15/.../H5_closure/
```

Drive fetch sha256 (both): same as local rows above. MCP metadata `fileSize` matched. Parent folder id on Drive: `1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y`.

### H5 checks summary

| Check | Result |
|---|---|
| Local RUNG2/3 present | **PASS** |
| Local body-sha256 self-consistent | **PASS** |
| Local totals json hashes match docs | **PASS** |
| Drive IDs present + sizes match | **PASS** |
| Drive vs local byte/hash match | **PASS** |

---

## 2) RN-UNIF lane

### Drive folder `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG`

Listed (MCP `search_files`):

| Title | Drive ID | Size |
|---|---|---:|
| RNU_EXECUTE_RECEIPT.md | `1G8XM7uRYuT8mtK3JtJQuV1EH5MhZo6Hn` | 497 |
| RNU_EXECUTE_RECEIPT.json | `1OXS646pxXoS_mOSSvjpPG3vDoZokMHLx` | 4815 |
| rnu_execute.py | `18frBWrX92vgIy2gxp-ukl1uUGoESQTAI` | 12567 |
| rnu_ds3.py | `1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21` | 7201 |
| rnu_chi2_white_v2.py | `1v5Pt4UJGoX18U9ZQHqir6bXz7GcQZtam` | 5005 |
| rnu_meanfix.py | `1NKK9aXwXEswzem-5YunoqqeEjqFpilfU` | 2800 |
| CL-RNU-001_…md | `16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6` | 12291 |

### Receipt status (Drive, authoritative for “still say NOT closed”)

**RNU_EXECUTE_RECEIPT.md** (sha256 `bb1786884882a913ad20a744b8149d361ea4ca03c16ce336bf9b6522a6f42c3a`):

> STATUS: PROPOSED. D3-LEMMA-RN-UNIF is NOT closed.

**RNU_EXECUTE_RECEIPT.json** (sha256 `ed02743d33ee8bbee9cedcdb28bc8bf2f869b2897bbe60aa318305b7aa3f0f5f`):

- `"status": "PROPOSED"`
- `"lemma_closed": false`
- `steps.A_ds3.pass: false` (`max_abs_err: "2.1746255"`)
- `next_required_for_freeze` still lists whitened env_form bounds, DS3 through kappa_far, valid T4 from env_form, full polar cover, both-mode transcript + MUT-RN-1..5 + FREEZE rule-id

| Check | Result |
|---|---|
| Drive receipts present | **PASS** |
| md says lemma NOT closed | **PASS** (confirmed) |
| json `lemma_closed: false` | **PASS** (confirmed) |

### Local engine `d3_rn_unif.py`

- **Found:** `extract_sep15/K3_SIDE24_LB/UPPER2D/D3_percolation/d3_rn_unif.py`
- Size 103166; sha256 `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`
- Hash pins at import (all **PASS** locally): `d3_perc.py`, `d3_amend.py`, `cov_exact.py`, `pin_transform.py`, `reg_lemmas.py`, `H3_RUNG_FLOOR.md`

### Smoke / self-check (as receipt describes)

Receipt is produced by Drive `rnu_execute.py` (imports `d3_rn_unif`, runs DS3 selftest + mean_grad / whitened chi2 / T3 / cell probes / landscape, writes receipts with `lemma_closed: false`).

**Environment:** system Python lacked `sympy`; used venv `/tmp/repro_h5_rn/venv` (`numpy 2.5.3`, `sympy 1.14.0`, `mpmath 1.3.0`).

#### A) `python3 d3_rn_unif.py` (module-level pins + probes)

```bash
cd …/D3_percolation
/tmp/repro_h5_rn/venv/bin/python d3_rn_unif.py
# EXIT: 0   ELAPSED_SEC: 22
```

Key output (abbrev): hash pins verified; Z_lo drift ck armed; cov_exact max gap 2.29e-100; fast assembly kap_far = 0.6772849052 matches frozen v2; four probe points under KAP_BUDGET 0.68. **PASS** (exit 0).

#### B) `rnu_execute.py` (receipt-described suite) — sandbox

```bash
# scripts fetched from Drive (confirm=t); engine symlinked from extract
cd /tmp/repro_h5_rn/rnu_sandbox/D3_percolation
/tmp/repro_h5_rn/venv/bin/python rnu_execute.py
# EXIT: 0   ELAPSED_SEC: 26
# printed: LEMMA_CLOSED=NO
```

Regen receipt again has `"lemma_closed": false`, `"status": "PROPOSED"`.

| Step | Drive receipt | This re-run |
|---|---|---|
| A DS3 selftest | max err 2.1746255, pass **false** | max err **0.0**, pass true |
| B mean_grad_fixed | 3.4e-34 | 3.4e-34 |
| C whitened chi2 rel | 1.61e-84 | 1.61e-84 |
| D \|T3\| | 18.534 | 18.534 |
| lemma_closed | **false** | **false** |

| Check | Result |
|---|---|
| `d3_rn_unif.py` present | **PASS** |
| Module smoke exit 0 | **PASS** |
| `rnu_execute` smoke exit 0 + LEMMA_CLOSED=NO | **PASS** |
| Drive receipts still NOT closed | **PASS** |
| Byte-identical regen vs Drive receipt | **FAIL / PARTIAL** (DS3 max_abs_err differs: Drive 2.1746255 vs regen 0.0; both still `lemma_closed: false`) |

---

## 3) Overall

| Lane | Overall |
|---|---|
| H5 local+Drive match | **PASS** |
| RN-UNIF Drive NOT-closed confirmation | **PASS** |
| RN-UNIF local engine smoke | **PASS** (exit 0; lemma still not closed) |
| RN-UNIF full receipt byte-repro | **PARTIAL** (DS3 numeric disagreement vs frozen Drive receipt) |

**No premise status promotion. No theorem closure.**

---

## Caveats / unverified

1. Did **not** re-run H5 cell/stitch/merge workloads or mutation suites; only verified doc/totals hashes and Drive mirrors.
2. `CODE_HASHES.txt` / `H5_RECEIPTS.txt` are historical multi-epoch logs; not all lines re-hashed against current tree files beyond RUNG2/3 totals cites.
3. Full polar certification / MUT-RN-1..5 / FREEZE path **not** run (explicitly still in `next_required_for_freeze`).
4. Drive `.py` fetch via plain `uc?export=download` hits virus-scan HTML; used `drive.usercontent.google.com/...&confirm=t` (and MCP download) for executable scripts.
5. Smoke required a local venv (`sympy`/`numpy`); not the original Kimi execution host. DS3 selftest discrepancy vs Drive receipt is **unexplained here** — fail-closed: do not treat Drive `A_ds3.pass:false` as overturned for ledger purposes; only note local regen got 0.0 while still refusing lemma close.
6. Accidental HTML stubs briefly touched extract `D3_percolation/rnu_*.py`; removed before final state. Sandbox writes only under `/tmp/repro_h5_rn/`.
7. MCP `read_file_content` returns normalized/escaped text — **not** used for byte hashes; hashes from curl/Drive binary download + local files.

