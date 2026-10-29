# CL-AUD-001-v1.0 — THIRD-FAMILY RE-EXECUTION + TREE CUSTODY AUDIT, 2026-09-15

AUTHOR: Claude (Anthropic), foreign environment (different machine, path root, Python install; no network,
  no Drive read during execution) · CREATED: 2026-09-15 · CLASS: AUD — independent re-execution audit
STATUS: PROPOSED · AUTHORITY: none · CANONICAL IMPACT: NONE (confirms; asserts no new theorem content)
SOURCE: `09152026OKComputer_Project_Gap_Closure.zip` (a2136bc0…), tree `K3_SIDE24_LB/`
FALSIFICATION: any receipt below failing to reproduce on re-run of the named script against the same bytes.

## 1. Gate and certificate re-executions

| artifact (frozen) | result here | frozen record |
|---|---|---|
| `UPPER2D/D1_assembly/d1_falsify_v3.py` (D1 v2.2 operative gate) | exit 0; 91 `PASS` cks + summary (= "92" in the record); **normal vs `-O` byte-identical**; digest `d800849ed5966b4d583a10cf72525e89350d9a53780b45a807a21ccbadf91085` | RETURN_06 digest **MATCH** |
| `UPPER2D/H4_JC_repair/h4jc_falsifier.py` (H4JC-R1) | exit 0; mirror both modes byte-identical; 427 pairs / 1612 saddles; OLD 132/175, NEW 0; digest `72fdc49bd8b5d994b7935de4aa559dc748e50b59fdb00fc0a8a456cb080d8215` | MATCH |
| `UPPER2D/B4LOC_damline/b4loc_driver.py` (B4LOC-R1) | exit 0; **stdout byte-identical to frozen `t_normal.txt`** (11baaccc…); digest `43ff9c1ebfa9cf8f27c06e27e844030e8378e439e89bdd374949166a1a490a1b` | FREEZE.txt **MATCH** |
| `UPPER2D/D3_percolation/d3_perc_decay.py` (PERC engine) | reached the frozen VERDICT text verbatim; timed out in this sandbox's 230 s window inside the PD6 mutation suite (partial; not a failure) | — |
| `cl_reexec_20260915.py` (this pass; exact-arithmetic audit of the headline) | exit 0 both modes, byte-identical, digest `631564ef88b92629d7cf1c6dcc0448e3bfa7236b6c6820241a2330377226df2e` | new |
| `UPPER2D/D3_percolation/d3_rn_unif.py` (RN-UNIF engine, unfrozen; 2026-09-16) | runs 45 s to its current end (probe stage); reproduces the frozen v2 κ values; see CL-RNU-001 | — |

Transcripts shipped alongside: `d1_falsify_v3_reexec_transcript.txt`, `b4loc_driver_reexec_transcript.txt`,
`perc_engine_reexec_partial_transcript.txt`, `cl_reexec_t_normal.txt` ≡ `cl_reexec_t_O.txt`.

## 2. Tree custody (script `cl_custody_check.py`, transcript `cl_custody_check_transcript.txt`)

- **TREE_MANIFEST.sha256 (RETURN_06, snapshot 00:05:36Z): 985 MATCH · 46 DRIFTED · 0 MISSING of 1031.**
- All 46 drifts reconcile: W3 lbox/live/watch (declared volatile); H5 banks, transcripts, promote/keeper logs
  (declared volatile); `d3_falsifier.py` + `tf_*` (F7 landing — matches FREEZE_PERC_DECAY abbb40d2…);
  `h3_band_floor.py` / `RECEIPT_h3.json` / H3 MANIFEST (frozen version a907eeed…, documented in ADDENDUM-1 with
  the pre-freeze hashes labelled); `D1_V2_2_RECEIPTS.txt`; **H5 code files** (`h5_merge.py`, `h5_promote.py`,
  `promote_run.py` — pinned in CODE_HASHES.txt; `h5_run.py`, `h5_run_r0p025.py`, `h5_run_r0p035355.py`,
  `keeper_promote.sh` — **not pinned at current bytes**, see CL-PIN-001 / CL-ERR-001 E4).
- **289 post-snapshot births** (excl. `__pycache__`): H5_closure 140, LPW v4 47, LPW v3 47, H3_closure 17,
  RETURN_06 14, B4LOC 8, D3 7, LPW 5, W3 3, D1 1 — i.e. the whole 2026-09-15 campaign is outside the manifest.
  (The RN-UNIF engine `d3_rn_unif.py` / `_dbg.py` are among the D3 births.)
- **SOURCE_CAPSULE.sha256: 62/70 rows verify** under a declared rule. The 8 misses are superseded rows
  (`d3_falsifier.py` pre-F7, `h3_band_floor.py` pre-freeze, `RECEIPT_h3.json` pre-ceiling, five H5 code
  files) whose replacements the addenda record. The capsule retains stale rows next to fresh ones; a naive
  verifier fails them.

## 3. Carrier hash chain (recomputed from bytes; rule per CL-REG-001)

MATCH: `h3_band_ceil.py` b97c5428…; `ceil_normal.txt` ≡ `ceil_O.txt` 26d08534…; `H3_BAND_CEIL.md` 18e109bc…
(whole) / cfe8a3a4… (excl_bodyhash_line); `B4LOC_DAMLINE.md` ae4c9c82… (whole) / 0d5c1b32… (marker);
`b4loc_driver.py` bf3b0225…; `b4loc_falsifier.py` d59ad790…; `t_normal.txt` ≡ `t_opt.txt` 11baaccc…;
`d3_perc_decay.py` ee68ac75…; `PERC_DECAY.md` body 5137a811… (marker_raw); `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md`
body c1d5e95d… (marker_strip_LF); `D1_ASSEMBLY_v2_2.md` body 490ad6b2… (marker_strip_LF); H3 MANIFEST
contains the ceiling script, document and transcripts; `H5_RUNG2/3` bodies 91a34d83… / f4c3414f…
(before_hashline).

## 4. Exact-arithmetic reproductions (all from stated recipes, none from prose)

- LPW v3: 4·(597/128)·(2596849/204800)·(9/10000)·(14587/2621440)⁴ / 2082000000000 = the stated fraction
  exactly = 9.8040863135804911886149013570…e-23 ✓; ≥ 9.80e-23 ≥ 1e-23 ✓.
- LPW v4: 16·(same)/U, U = 366282864761194/10¹⁴ = the stated fraction exactly = 2.2291086664054236617851686509…e-10
  ✓; ≥ 2.22e-10 ≥ 1e-10 ✓; v4/v3 = 4B₃/U = 2273652633308.309… ✓; v3/v2 = 8.632e20 ✓; 2¹⁸·8690 = 2278031360 ✓.
- LPW v2 exact expansion 1.13577628463478070056633120196926…e-43 — confirms H1 F-C29 (v2's 29-digit display
  2.1e-60 ABOVE exact).
- B4LOC: 3.47·√(e^{−43.536} + e^{−107.93}) = 1.2207e-9 ✓ (nearest-rounded display 1.22e-9 → E-B4LOC-1);
  c_eff = −r² ln P → 0.0513/0.0645/0.0709 ✓; o(r³) threshold √(12 ln 20 + 4 ln 3.47 + 2 ln 2) = 6.5047 ✓;
  κ₁r, κ₂r inside the stated windows ✓.
- H5: I_hi/r³ = 647.8047(14) / 661.4712 / 664.3979 ✓; steps +2.11% / +0.44% ✓; C_unif margin 10.1% ✓;
  C1_TOTAL(0.025) = 2.006643750e-5 ✓; C1_TOTAL(0.035355) = 5.67548e-5 at the literal r ✓.
- H3: floor margins vs c_Z 59.88% / 42.78% ✓; 2.3066 < 3.6628 < 4 ✓.
- D1 headline: I_hi/r³ − 19.55 + 21.9279 = 650.182614016 = "exact 650.1826140…" ✓; 650.1827 round-UP ✓.
- D3 bracket: 2.5282637 × 1.68 = 4.247483016 ≠ stated 4.247483056 → CL-ERR-001 E1.
- LPW v1 D2: E[(|X|+|Y|)⁴] = 12 + 32/π from the absolute moments ✓ (12 + 16/π false); E[R⁴] = 8, E[R] ≤ √2 ✓.
- RN-UNIF (2026-09-16): whitened χ² = engine χ² to 1.6e-84; exact whitened ∇χ² = FD to 1e-39 (with the
  corrected mean gradient); `kappa_far_ds` ∇/∇² = FD to 1e-26; engine's `mean_grad_exact` ≠ FD (E-RNU-1).

## 5. Scope statement

Nothing here bears on the mathematical validity of the frozen theorem bodies beyond what their own gates
check; those bodies were not re-derived line by line. The 3D firewall was checked only inside the zip (no
`AO48-OPR-045` citation anywhere in the tree). Drive-side claims are outside this audit.
