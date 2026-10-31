# MATH PUSH — whitened env_form orders 2–4 (stub smoke)

**Written:** 2026-09-16 ~23:05 CT (America/Chicago).  
**Agent:** 5 eng-lead executor.  
**Discipline:** fail-closed. **`lemma_closed: false`.** No FREEZE / MUT-RN / premise promotion.  
**Plan followed:** `RN_UNIF_WHITENED_ENV_FORM_STUB_PLAN.md`.

---

## 1. Pins

| Artifact | Bytes | sha256 |
|---|---:|---|
| **PREFERRED** `extracts/RN_UNIF_2026-09-16/rnu_ds3.py` | **9704** | `bd3074fd900fc80bebdd430e34ae953f0a78685fff672758e7b927cc1e6d9421` |
| Duplicate `rnu_ds3_7p2kb_DUPLICATE.py` | 7201 | (not wired; sha distinct) |

Missing modules (gap explicit): **`rnu_env.py`**, **`rnu_white.py`** — still ABSENT on Drive hub and local extract.

---

## 2. What was built

**Code:** `/workspace/drive_peer_review_triage/code_prototypes/rnu_whitened_env_form_stub.py`  
**Receipt:** `/workspace/drive_peer_review_triage/code_prototypes/RN_UNIF_ENV_FORM_SMOKE_RECEIPT.json`

| Step | Action | Claim level |
|---|---|---|
| D3 | Pin preferred DS3; install; DS3 vs engine DS at (5,0) | numerics gate only |
| W1 | Build W=Λ₀^{−1/2}V₀ᵀ; whitened χ² vs engine | order-1 frame only |
| E2–E4 | `env_form(k,γ,d,q)/√LAM0[k]` for q∈{2,3,4} at d=5 | **PROXY only** |

**Provenance string on E2–E4:**  
`proxy:env_form/sqrt(LAM0[k]); rnu_env ABSENT`  
Native `env_form` already acts on V0-basis residual FORMS; the stub applies the CL-RNU-001-cited 1/√λ scaling. This is **not** the missing whitened residual-covariance construction for log q(A, m′).

---

## 3. Smoke commands + exit codes

```bash
ENG=/workspace/drive_peer_review_triage/extract_sep15/K3_SIDE24_LB/UPPER2D/D3_percolation
cd "$ENG"
PYTHONPATH="$ENG:$ENG/../H2_foundations" \
  /tmp/rnvenv/bin/python \
  /workspace/drive_peer_review_triage/code_prototypes/rnu_whitened_env_form_stub.py
# EXIT 0  (~24s)
# Log: /tmp/rnu_whitened_env_form_smoke.out
```

Companion preferred-only smokes (unchanged; still lemma open):

```bash
PYTHONPATH="$ENG:$ENG/../H2_foundations" /tmp/rnvenv/bin/python \
  /workspace/drive_peer_review_triage/extracts/RN_UNIF_2026-09-16/rnu_ds3.py
# EXIT 0

PYTHONPATH="$ENG:$ENG/../H2_foundations" /tmp/rnvenv/bin/python \
  /workspace/drive_peer_review_triage/extracts/RN_UNIF_2026-09-16/rnu_chi2_white_v2.py
# EXIT 0 — order-1 only
```

**Stub observed (2026-09-16 CT):**
- PIN PASS (9704B / sha match).
- D3: DS3 vs DS max abs err ≈ **1.98e-87** at y=(5,0) — PASS (<1e-60 gate).
- W1: whitened χ² rel.diff ≈ **1.61e-84** vs engine — PASS.
- E2–E4 proxy at d=5 (18 non-rigid (k,γ) pairs):

| q | max | RSS |
|--:|---:|---:|
| 2 | ~1.29 | ~3.09 |
| 3 | ~14.5 | ~34.7 |
| 4 | ~165 | ~396 |

- Receipt: `status: PROPOSED`, **`lemma_closed: false`**, gaps listed.

---

## 4. Missing pieces (still OPEN)

1. Historical **`rnu_env.py` / `rnu_white.py`** bytes (or a true reconstruction of whitened residual-cov blocks for log q(A, m′), A = I − Σ′).
2. **Valid T₄ from env_form** (not measured-scale INTERNAL).
3. Over-cell Lipschitz / FREEZE / MUT-RN path — not started.
4. Piece-2 annulus driver — still UNWRITTEN (`_piece2_driver_outline_8.md` only).
5. Do **not** rewrite `RNU_EXECUTE_RECEIPT` A_ds3 historical ledger from this smoke (`DS3_RECEIPT_DRIFT.md`).

---

## 5. Explicit non-claims

- **`lemma_closed: false`** — D3-LEMMA-RN-UNIF stays OPEN.
- Proxy E2–E4 bounds ≠ FREEZE / ≠ lemma close / ≠ valid T4.
- Preferred DS3 PASS ≠ receipt rewrite / ≠ premise flip.
- Regenerating chi2_white / DS3 PASS ≠ Piece 1 or 2 close.

*End. Lemma remains OPEN.*
