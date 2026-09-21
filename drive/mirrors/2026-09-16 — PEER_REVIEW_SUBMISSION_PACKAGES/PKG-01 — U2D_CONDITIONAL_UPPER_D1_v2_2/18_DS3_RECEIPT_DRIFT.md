# DS3 receipt drift — Drive vs current `rnu_ds3.py` / extract engines

**Mapped:** 2026-09-16 ~16:10 CT (America/Chicago)  
**Role:** fail-closed isolation only. **Hypothesis, not closure.**  
**Explicit:** Does **NOT** claim D3-LEMMA-RN-UNIF closed. Does **NOT** promote premise status. Regen DS3 pass does **not** close the lemma.

Sources: Drive folder `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG` + local Sep-15 extract `extract_sep15/K3_SIDE24_LB/UPPER2D/`.

---

## Artifacts located

| Artifact | Where | Drive ID (if any) | Size | sha256 |
|---|---|---|---:|---|
| `rnu_ds3.py` | Drive only (not in Sep-15 extract) | `1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21` | 7201 | `c3d5adc8b88788b7eed43bd68efe9381f7076775a7ec60817a20a8267d9116da` |
| `rnu_execute.py` | Drive only | `18frBWrX92vgIy2gxp-ukl1uUGoESQTAI` | 12567 | `89b36db4bc972e2c64a180e544288674eff70e722215d9ee3ec3e57ab7944960` |
| `RNU_EXECUTE_RECEIPT.json` | Drive | `1OXS646pxXoS_mOSSvjpPG3vDoZokMHLx` | 4815 | `ed02743d33ee8bbee9cedcdb28bc8bf2f869b2897bbe60aa318305b7aa3f0f5f` |
| `RNU_EXECUTE_RECEIPT.md` | Drive | `1G8XM7uRYuT8mtK3JtJQuV1EH5MhZo6Hn` | 497 | `bb1786884882a913ad20a744b8149d361ea4ca03c16ce336bf9b6522a6f42c3a` |
| `d3_rn_unif.py` (rung engine) | local extract `…/D3_percolation/` | — | 103166 | `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930` |
| `rnu_chi2_white_v2.py` | Drive | `1v5Pt4UJGoX18U9ZQHqir6bXz7GcQZtam` | 5005 | `6b61af7b549c46b098fe3239803553fe9b555446cf9b36c1e0609d9a68963749` |
| `rnu_meanfix.py` | Drive | `1NKK9aXwXEswzem-5YunoqqeEjqFpilfU` | 2800 | `3693655f45d03d326e9ceaa896acbec884e7acd2aae8abe3884c7b5f5248f5c5` |

Drive folder created/modified (UTC→CT): `rnu_ds3.py` 2026-09-16 12:36:32 CT; receipts / `rnu_execute.py` 12:36:34 CT (same batch upload window).

**Local extract has no `rnu_*.py`.** DS3 arithmetic lives only on Drive; extract engine is `d3_rn_unif.py` (second-order `DS`/`DM`, not DS3).

Hashes from binary `curl`/`drive.usercontent…&confirm=t` downloads (not MCP text normalize).

---

## Numeric disagreement (the PARTIAL)

| Field | Drive receipt (frozen) | Regen with current Drive `rnu_ds3.py` + extract `d3_rn_unif` |
|---|---|---|
| `steps.A_ds3.max_abs_err` | `"2.1746255"` | `"0.0"` |
| `steps.A_ds3.pass` | **false** | **true** |
| `lemma_closed` | **false** | **false** |
| `status` | `PROPOSED` | `PROPOSED` |
| Other steps B/C/D (prior repro) | match | match |

Receipt JSON still lists `next_required_for_freeze` including whitened `env_form` orders 2–4, DS3 through `kappa_far`, valid T4 from `env_form`, polar cover, MUT-RN + FREEZE.

---

## Evidence tying `2.1746255` to a specific magnitude

`rnu_execute.py` records:

```python
mx, errs = ds3_selftest()
out["steps"]["A_ds3"] = {"max_abs_err": ns(mx, 8), "pass": float(mx) < 1e-20}
```

with `ns = lambda x, n=8: mpmath.nstr(x, n)`.

On current host (`mpmath` 1.3.0, `mp.dps` as engine sets):

```text
nstr(0.8 * exp(1), 8) == "2.1746255"   # exact string match to Drive receipt
```

Current Drive `rnu_ds3.ds3_selftest()` at the documented point `(5,0)` for `f=(x²+y³)exp(x/5)` returns **all ten jet errs = 0.0**, `max=0.0`.

So Drive `A_ds3.max_abs_err` is **string-identical** to an absolute error of **exactly `0.8·e`** on some selftest component — a magnitude that **does not appear** when re-running the selftest against the **byte-identical** Drive `rnu_ds3.py` now on disk.

---

## Cause hypothesis (evidence-backed; not proven)

**Primary hypothesis — receipt/script vintage mismatch (stale DS3 fail frozen beside a later-clean `rnu_ds3.py`):**

1. Drive receipt encodes a real selftest failure of size `0.8·e` (formatting proves it is not a random float).
2. The `rnu_ds3.py` bytes currently on Drive selftest clean to 0.0 under the same `ds3_selftest` closed form.
3. Upload timestamps are within ~2 s (batch folder push), which is consistent with uploading a **pre-built local folder** that already mixed (a) a receipt from an earlier failing run with (b) a corrected `rnu_ds3.py` — **or** with a different `rnu_ds3` on `sys.path` at receipt generation than the file that was uploaded.
4. `rnu_ds3.py` is absent from the Sep-15 extract; there is no second local pin to cross-check an older revision.

**Secondary (weaker) hypotheses considered:**

- **Different float/mpmath path on original host:** unlikely to produce *exactly* `0.8·e` then vanish to 0.0 on identical source.
- **Wrong closed-form arm in an older selftest** (unused `fxxx` draft line in current file differs from `fxxx_cf` by `0.4·e`, not `0.8·e`) — does not alone explain the receipt magnitude under *current* source.
- **Import shadowing:** `rnu_execute` does `from rnu_ds3 import DS3, ds3_selftest` after `chdir` to script dir — a neighboring older `rnu_ds3.py` on the original machine could explain receipt≠current-file without any change to extract engines.

**Not claimed:** which third-jet component was off; that current regen “fixes” the ledger; that DS3 is wired through `kappa_far` (receipt still lists that as remaining).

---

## Explicit non-claims

- D3-LEMMA-RN-UNIF remains **NOT closed** (`lemma_closed: false` on Drive and regen).
- Regen `A_ds3.pass: true` does **not** overturn Drive receipt for fail-closed ledger purposes; it only shows **current bytes ≠ receipt numeric A_ds3**.
- No FREEZE / MUT-RN / premise promotion.

---

## Commands used (cwd / exit)

```bash
# fresh Drive binary fetch
mkdir -p /tmp/repro_h5_rn/drive_fresh && cd /tmp/repro_h5_rn/drive_fresh
curl -fsL -o rnu_ds3.py "https://drive.usercontent.google.com/download?id=1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21&export=download&confirm=t"
# … same pattern for execute + receipts …
sha256sum rnu_ds3.py RNU_EXECUTE_RECEIPT.json

# selftest
/tmp/repro_h5_rn/venv/bin/python -c "from mpmath import mp; import sys; sys.path.insert(0,'.'); from rnu_ds3 import ds3_selftest; print(ds3_selftest()[0])"
# => 0.0
```

Sandbox regen receipt (prior agent-5 repro): `/tmp/repro_h5_rn/rnu_sandbox/D3_percolation/RNU_EXECUTE_RECEIPT.json` sha256 `9df3c61a…` with `A_ds3: {max_abs_err: 0.0, pass: true}`, `lemma_closed: false`.
