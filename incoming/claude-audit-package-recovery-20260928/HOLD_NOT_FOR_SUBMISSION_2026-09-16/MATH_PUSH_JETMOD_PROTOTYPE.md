# MATH PUSH — JETMOD first-band interval-r enclosure PROTOTYPE

**Written:** 2026-09-16 ~23:10 CT (America/Chicago).  
**Agent:** 5 eng-lead executor.  
**Discipline:** fail-closed. **No premise flips / no promotions.** `lemma_closed: false`.

---

## 1. Status (as written)

| Item | Status |
|---|---|
| **OBL-H5-JETMOD** | **OPEN** (display only) — **NOT discharged** |
| OBL-D1-PROMOTE chart side | OPEN |
| OBL-H5-ZBAND hi / REMOTE-THRESHOLD | OPEN (untouched) |
| This artifact | engineering prototype only (c2 / 1 of 24 jets) |

---

## 2. Design

**Primary code:** `code_prototypes/jetmod_first_band_interval_r.py` (@5)  
**Sibling:** `code_prototypes/jetmod_first_band_proto.py` (bot 7; same method, different band)

| | @5 `interval_r` | bot 7 `proto` |
|---|---|---|
| Band | B0 = `[0.025/√2, 0.025]` ≈ `[0.01768, 0.025]` (below RUNG2) | B0 = `[0.035355, 0.05]` (RUNG3↔0.05) |
| Method | interval-r `LAT.k1` + **32-slab hull** for `c2=(G00−G0S)/r²` | same (standalone Lattice twin) |
| Kernel | `h5_kernel.LAT` via `_h5_local_run` + H2 | embedded Lattice twin |

Naive whole-band IA suffers dependency inflation (width ≈ 0.75). Subdivision controls that (width ≈ 0.0257) but still ≫ display residual / struct κ=1/8 halfwidth — falsifier content exercised; **no discharge**.

---

## 3. Smoke commands + outcomes

```bash
cd /workspace/drive_peer_review_triage
# @5 — band below RUNG2
_venv_h5_smoke/bin/python code_prototypes/jetmod_first_band_interval_r.py
# EXIT 0
# receipt: code_prototypes/jetmod_first_band_smoke_receipt.json
# log: /tmp/jetmod_first_band_smoke.out

# sibling bot 7 — coarsest band
_venv_h5_smoke/bin/python code_prototypes/jetmod_first_band_proto.py
# EXIT 0
# receipt: code_prototypes/jetmod_first_band_receipt.json
```

**Observed (@5, 2026-09-16 CT):**
- naive width ≈ **0.749863**; subdivided width ≈ **0.025722**
- point-r display `c2(r_lo)≈0.49996`, `c2(r_hi)≈0.49992` contained in hull
- width > struct_model_halfwidth (≈1.95e-5) → True (diagnostic)
- `discharges_OBL_H5_JETMOD: false`, `jets_done: 1/24`, `lemma_closed: false`

**Observed (bot 7 sibling):** subdivided width ≈ **0.025714**; same OPEN conclusion.

---

## 4. What it DOES

- Proves interval-r `k1` + slab hull yields a finite **c2** band enclosure on a ladder band.
- Exercises the OBL falsifier *shape* (width vs display residual / κ model) without closing it.
- Documents fail-closed scope: c2 only.

## 5. What it does NOT certify

- Full **24-jet** set; `Î(r)/r³ ≤ F(G12-band)`; uniform-in-band lattice-tail beyond `LAT.tail(n)`.
- Modulus falsifier **discharge** (width still exceeds display residual / struct halfwidth).
- Cell/stitch/merge at 0.0177/0.0125; RUNG2/3 premise discharge; ZBAND-hi; REMOTE-THRESHOLD.
- Any promotion of OBL-H5-JETMOD off **OPEN (display only)**.

## 6. Limitations / next eng (not done)

- More slabs / derivative bounds to shrink IA width toward structural scale.
- Extend to remaining 23 jets + scaled `J(B)/r^{p_J}`.
- Interval-r `PinFrame` conditioning (`T`,`A`, Neumann).
- Operator/ledger path only after a real modulus comparison passes — not claimed here.

*End. OBL-H5-JETMOD still OPEN. No promotions.*
