# RN-UNIF next smoke — whitened env_form / CL-RNU-001 step-1

**Mapped:** 2026-09-16 ~16:10 CT (America/Chicago)  
**Role:** fail-closed smoke / entrypoint inventory. **No lemma close. No premise promotion.**

Drive RN-UNIF folder: `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG`  
Extract root: `/workspace/drive_peer_review_triage/extract_sep15/K3_SIDE24_LB/UPPER2D/`

---

## What CL-RNU-001 / receipts name as step 1

From Drive `CL-RNU-001_…md` (`16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6`) §5 item 1 and `RNU_EXECUTE_RECEIPT.json` `next_required_for_freeze[0]`:

> whitened `env_form` bounds on residual covariances at orders 2–4

CL-RNU-001 also states: exact **first** derivative of whitened χ² is **DONE** (`rnu_chi2_white_v2.py`, FD-validated); remaining construction is orders **2–4** via whitened residual covariances + `env_form`.

---

## Search results — runnable code inventory

| Item | Present? | Location |
|---|---|---|
| Whitened χ² **exact gradient** smoke (order 1) | **YES** | Drive `rnu_chi2_white_v2.py` (`1v5Pt4UJ…`) |
| Mean-grad fix helper | **YES** | Drive `rnu_meanfix.py` |
| Aggregate execute suite (A–F) | **YES** | Drive `rnu_execute.py` (marks T4 as INTERNAL-NOT-ENV_FORM) |
| Native (unwhitened) `env_form(k,γ,d,qord)` | **YES** | extract `D3_percolation/d3_rn_unif.py` ~L381 |
| **Whitened** `env_form` residual-cov bounds orders 2–4 as runnable smoke | **NO** | Not in extract; not in Drive RN-UNIF folder listing |
| `rnu_env.py` / `rnu_white.py` (named in CL-RNU-001 RECEIPTS header) | **NOT in folder now** | Drive search for `rnu_env` / `env_form` titles under recent RN work returned only `rnu_chi2_white_v2.py` |

**Conclusion:** CL-RNU-001 step-1 *remainder* (whitened env_form orders 2–4) **does not exist as runnable code** in extract or the current Drive RN-UNIF folder. Nearest existing entrypoints are documented and smoked below.

---

## Smoke A — whitened χ² order-1 (`rnu_chi2_white_v2.py`) — nearest step-1

**Cwd:** `/tmp/repro_h5_rn/rnu_sandbox/D3_percolation`  
(scripts copied from fresh Drive download; `d3_rn_unif.py` + deps symlinked from extract; `../H2_foundations` present)

**Command:**

```bash
cd /tmp/repro_h5_rn/rnu_sandbox/D3_percolation
/tmp/repro_h5_rn/venv/bin/python rnu_chi2_white_v2.py
```

**Exit code:** `0`  
**Elapsed:** ~29 s (dominated by `d3_rn_unif` import)

**Key stdout (abbrev):**

```text
y=(5.0,0.0): chi2=1.944285687e-6  |grad|exact=1.5630931e-5  |grad|FD=1.5630931e-5  rel.diff=9.53e-40
      engine chi2=1.944285687e-6  => exact |grad kap_pair| = 0.0101202   (engine crude bound: 1.02e+17)
y=(4.8,1.6): … rel.diff=7.42e-40 …
y=(0.5,6.2): … rel.diff=2.2e-39 …
y=(5.1,-0.3): … rel.diff=1.02e-39 …
```

**Result:** **PASS** as order-1 whitened identity/FD check. Does **not** implement orders 2–4 env_form bounds. Does **not** close the lemma.

Full capture: `/tmp/repro_h5_rn/chi2_white_smoke.out`

---

## Smoke B — native `env_form` at qord 0–4 (unwhitened; probe only)

**Purpose:** confirm the engine entrypoint callable at orders including 2–4; **not** the whitened residual-cov construction named for freeze.

**Command:**

```bash
cd /tmp/repro_h5_rn/rnu_sandbox/D3_percolation
/tmp/repro_h5_rn/venv/bin/python - <<'PY'
# import d3_rn_unif; call env_form(k, gamma, d=5, qord) for qord in 0..4
PY
```

**Exit code:** `0` (~22 s import)

**Key lines:**

```text
env_form entrypoint: d3_rn_unif.env_form(k, gamma, d, qord)
NOTE: this is UNWHITENED engine env_form; whitened residual cov orders 2-4 smoke DOES NOT EXIST as runnable code
qord=0 sample_max≈5.84e-06
qord=1 sample_max≈6.32e-05
qord=2 sample_max≈6.95e-04
qord=3 sample_max≈7.76e-03
qord=4 sample_max≈8.79e-02
DONE
```

**Result:** entrypoint **live**; magnitudes O(1e-6)→O(1e-1) on the tiny sample — **not** a certification of whitened residual bounds.

---

## Smoke C — prior `rnu_execute.py` suite (context only)

Already documented in `REPRO_CHECKS_H5_RN_UNIF.md`: exit 0, `LEMMA_CLOSED=NO`, Drive receipt `lemma_closed: false`. T4 step explicitly `INTERNAL-NOT-ENV_FORM`. DS3 A-step drifts vs Drive (see `DS3_RECEIPT_DRIFT.md`); lemma still not closed.

---

## Explicit non-claims

- No claim that whitened env_form orders 2–4 are implemented or smoked.
- No FREEZE / MUT-RN / D3-LEMMA-RN-UNIF close.
- No promotion of OBL-D1-PROMOTE / chart-side status from the H5 subsection below.

---

## Subsection — H5 cell progress harness (r = 0.0177 / 0.0125)

### Executable harness: **YES** (under `H5_closure/`; STAGE_E has mut/snap copies of kernels, not the primary rung drivers)

| Script | Role |
|---|---|
| `H5_closure/h5_run_r0p0177.py` | Generated rung driver for r=0.0177 |
| `H5_closure/h5_run_r0p0125.py` | Generated rung driver for r=0.0125 |
| `H5_closure/promote_run.py` | Regenerates/runs scaled driver: `--r R --part PART --shard S --nshards N [--one]` |
| `H5_closure/supervisor_rung.sh` | Loop: cells shard0/1 + stitch with `--one` until exit 42 |
| `H5_closure/keeper_promote.sh` | Keeper (paths hard-coded to `/mnt/agents/output/...`; currently stitch-focused for 0.0177) |
| `H5_closure/queue_runner.sh` | Queues cheap parts for `0.0177` then `0.0125` |

`H5_RUNG2_2026-09-15.md` still records ladder language: `r=0.0177: cells 42/70 …`; `r=0.0125: cells 21/70 …` (package snapshot — not re-certified here).

### How to run (extract tree; adjust `DIR` if not on original `/mnt/agents/...` host)

```bash
cd /workspace/drive_peer_review_triage/extract_sep15/K3_SIDE24_LB/UPPER2D/H5_closure

# one-shot cell unit (preferred via promote_run)
python3 promote_run.py --r 0.0177 --part cells --shard 0 --nshards 2 --one
python3 promote_run.py --r 0.0177 --part cells --shard 1 --nshards 2 --one

python3 promote_run.py --r 0.0125 --part cells --shard 0 --nshards 2 --one
python3 promote_run.py --r 0.0125 --part cells --shard 1 --nshards 2 --one

# or supervisor loop (original host path inside script is /mnt/agents/output/...)
# sh supervisor_rung.sh 0.0177
# sh supervisor_rung.sh 0.0125

# direct generated drivers (same argv shape)
# python3 h5_run_r0p0177.py --r 0.0177 --part cells --shard 0 --nshards 2
```

Exit convention (from supervisor comments): `0` = unit banked, `42` = no unbanked work, other = fail/retry.

### Observational local jsonl counts (not a premise update)

Under extract `H5_closure/`:

| r | `h5_results_r…_cells.jsonl` R1 unique cells (this tree) | RUNG2 text |
|---|'---|---|
| 0.0177 | 70 R1 rows / 70 unique cells | 42/70 |
| 0.0125 | 42 R1 rows / 42 unique cells | 21/70 |

Treat as **raw log inventory** that may post-date or disagree with the RUNG2 package sentence; **do not** treat as OBL-D1-PROMOTE discharge. Merge/`h5_merge.py` still requires 70/70 with fail-closed `ck`. No full cell/stitch workload re-executed in this smoke.

### STAGE_E

`STAGE_E/mut_H5`, `mut_D1/H5_closure`, `h5snap/` hold kernel/merge copies and mutation reviews — **not** a separate turnkey r=0.0177/0.0125 cell supervisor. Use `H5_closure/` drivers above.
