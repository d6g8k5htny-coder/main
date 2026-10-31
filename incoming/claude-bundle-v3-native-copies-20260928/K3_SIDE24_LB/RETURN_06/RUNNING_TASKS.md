# RUNNING_TASKS — exact states read fresh at assembly time: 2026-09-15 ~08:15 CST

DISCIPLINE: nothing on this page is evidence for any theorem. Every lane is RUNNING (or newly complete,
not yet frozen); each is labeled with its LAST VERIFIED ARTIFACT. The consumed theorem content comes only
from the frozen carriers in SOURCE_CAPSULE.sha256.

## 1. H5 promotion rungs (OBL-D1-PROMOTE) — EXECUTING

- Supervision: keeper_promote.sh A/B (locks .keeperp_A/B.lock, 04:57 CST); refine-3 stitch lane PAUSED
  (~14:05 2026-09-15, documented; banked records stand; v3 totals shipped).
- Rung r = 0.05: COMPLETE — frozen v1 731.43·r³; live clean v3 647.8048·r³ (h5_totals_v3.json 8d7028e4…,
  version-pinned; tightening amendment + errata 2026-09-15).
- Rung r = 0.025: patches COMPLETE (lo = 7.1329e-10); probes COMPLETE (10 rim columns, 6 scone, 12 m-back,
  5 sdisk + 6 sring; one site re-certified at η = 6e-5 on retry, receipt banked); cells EXECUTING — banked
  jsonl lines: s0p2_cells 83 + s1p2_cells 19; stitch EXECUTING — s0p1_stitch 6 lines (M-ring sec 0 = 0 via
  the S-disk skip, as at r = 0.05). Last writes 04:34–05:08 CST.
- Rung r = 0.035355: cells EXECUTING — banked s0p2_cells 42 + s1p2_cells 20; stitch 3 lines;
  patches/rimprobes/probes banked earlier (03:19–03:50 CST). Last write 04:57 CST.
- Merger fail-closed spec per rung: 70 cells, 10 rim cols, 6+12+5+6 probe sites, 8 patches, 22 stitch
  sectors, rung-scaled coverage grid, C1_TOTAL(r) = 1.605315e-4·s³ ck; merge not yet runnable (cells/stitch
  open at both rungs).
- r-law displays (certified quantities, H5_PROMOTE_UPDATE): patches lo ratio 0.1222 vs r³ = 0.125;
  sdisk/sring edge-probe max ratios ≈ 0.5 = r¹; rim probes ≤ r¹ (conservative direction); falsifier watches
  for slower-than-ledger parts — none seen. κ = 1/8 modulus band DISPLAYED (12 certified points; certified
  band enclosure remains OBL-H5-JETMOD).
- LAST VERIFIED ARTIFACTS: h5_totals_v3.json (frozen, consumed); banked rung jsonl (custody only).
- Next milestone: close cells+stitch at r = 0.025 → rung merge → r = 0.035355 merge.

## 2. W3 LOWER driver (WP rigorous enclosure) — RUNNING, STALE ~90 MIN AT ASSEMBLY

- Drivers: w3_lbox.py two processes over x ∈ [−0.08, 0.08], y ∈ [0.50, 0.68], 0.0025 boxes (4608 total):
  band A = y ∈ [0.50, 0.575], 1920 boxes; band B = y ∈ [0.575, 0.68], 2688 boxes. Checkpoint/resume
  append-only.
- Fresh reads (lbox_A.txt / lbox_B.txt + .live, last writes 06:45:19 / 06:45:49 CST):
  band A: box 39/1920 completed, acc (certified lower) = 2.303858e-10, acc_up = 3.911173e-3,
  acc_ex = 9.168770e-9; band B: box 38/2688 completed, acc = 8.564328e-10, acc_up = 4.596954e-3,
  acc_ex = 1.516962e-8. (Supersedes the register's earlier 33/1920 + 30/2688 snapshot.)
- Per-box cost ~100 s (A) / ~71 s (B) at edge columns; W3_REPORT ETA ~53 h (both bands). NOTE: no new
  banked box in ~90 min before assembly (contention with the W8 relaunch 07:54 + H3 band run is possible;
  driver processes are not visible to this build agent) — state reported as RUNNING-STALE, not failed.
- W3's own calibration: certified LOWER expected ~1–1.5e-6, BELOW the 3.328125e-6 = 0.213·r³ refutation
  threshold — the formal refutation is EXPECTED INCONCLUSIVE (OPEN), neither refutation nor confirmation.
- LAST VERIFIED ARTIFACTS: harness receipts (CHECK0–CHECK5 exit 0 both modes, BYTEIDENT PASS; mutations
  incl. missing-sqrt/pin fail-closed); lbox_A/B.txt accumulators.
- Next milestone: band completion → G2 comparison vs W4 (4.7569e-6 ± 0.4%) → WP disposition.

## 3. W8 Phase-2 (Λ-side cell certification) — RUNNING (post-fix relaunch)

- Workspace: /mnt/agents/output/19fcef2e-c1c2-8c6c-8000-0f5a243156d9/work/ (outside the frozen tree;
  reported as an auxiliary lane).
- Scope (PHASE2_SCOPE.md): J3 certification on the 837 Phase-1-certified cells (623 interior + 214 clip-1)
  of the 1707-cell grid of record; the 864 Phase-1-ledgered cells (783 clip-mixed branch-ambiguous, 81
  denominator-straddle) contribute NO sups and any rung conclusion EXCLUDES their boxes.
- Diagnostic RESOLVED this session: three stacked cancellation-blindness layers, ALL arithmetic (constant-G_k
  sup blindness; red σ-sups now Bernstein-tight; ROOT: the depth-0 σ⁴-box freeze — reverted to
  current-sub-box sups). VERDICT: bug, not a genuine obstruction. The 6 ledgered Phase-2 cells (M|0,13,
  M|1,13, O|0,9, O|1,9, F|5,5,*) are INVALID under the fixed build → banks cleaned before relaunch.
- FRESH STATE: 4 shard drivers relaunched 07:54:43–07:54:51 CST (phase2_cert.py {C,M,O,F}, flock
  singletons); all phase2_cells_*.jsonl and phase2_ledger_*.jsonl = 0 lines at 08:15 CST (0 certified
  cells banked, 0 ledger entries — supersedes the register's "0 certified / 98 ledger entries" snapshot,
  which predates the bank cleaning); keepers A/B alive (30 s heartbeats, last 1789431275 = 08:14:35 CST);
  no phase2_*.live files yet; first cell lands ~10–20 min per the status file; ~2 min per depth-0 cell,
  hours for depth-6 cells.
- LAST VERIFIED ARTIFACTS: Phase-1 banks (pre-existing); KIMI-DER-027c_NONCLOSURE.md (formal nonclosure,
  certificate standard); verify_lambda_grid_v2/v3 transcripts.
- Next milestone: 4 shard receipts → verify_lambda_grid_v4.py full per-cell re-computation (completeness
  1707 = certified + phase2-ledger + phase1-ledger) + mutation suite → final report with hashes.

## 4. BRANCH rungs — COMPLETE

- rung05 production: COMPLETE pre-errata and independently reproduced by the Stage-E numerics reviewer
  (N = 4000; q = 0.999855, 1−q = 1.453e-4 = 1.162·r³, se 5.1e-5; counts 3112/888).
- rung025 production: LANDED — completed 2026-09-14T21:47:19Z (keeper_rung025.receipt; the freeze-time
  empty-log defect repaired; N = 1500; q = 0.999989, 1−q = 1.075e-5 = 0.688·r³, se 6.1e-6; counts 1183/317;
  validate worst_mu = 1.29e-10, worst_var_rel = 1.57e-9).
- Post-errata products: certificate both modes BYTE-IDENTICAL ALL CKS PASS; falsifier both modes
  BYTE-IDENTICAL, all 4 mutations caught; wall-clock/runner labels moved to sidecar receipts
  (determinism-charter repair, BRANCH_DIR_RECEIPTS_ERRATA_2026-09-15.md 7547e76a…).
- All BRANCH q-displays are EVIDENCE/diagnostic, never premises. No open BRANCH task.

## 5. H3 band-floor assessment (normalizer promotion) — RUN COMPLETED, NOT YET FROZEN

- h3_band_floor.py: CERTIFIED uniform normalizer floor Z_r ≥ c_Z r² for ALL r ∈ (0, 0.05] (exact Taylor
  series at r = 0, exact series division, interval evaluation, Wick/conditional-Wick band bound; no
  whitening, no MC in the certificate).
- FRESH STATE: transcript band_normal.txt (2883 B, completed 08:12:31 CST after 88.0 s runtime):
  ALL_CHECKS_PASS — CB6 PASS: uniform certified floor on (0, 0.05]: min lo = 2.30659559567154 >
  c_Z = 1.6154892676…, margins 59.88% / 59.81% / 57.38% / 52.02% / 51.05% / 49.67% / 42.78% across the
  seven sub-intervals (0,0.0025]…(0.04,0.05]; CB7 Neumann containment PASS + MC DIAGNOSTIC (labeled NOT
  load-bearing); transcript sha256 61e7bc49d8146c7669a8580cb52e2c7111480cb00777b86652256a7e7f78ee84
  (self-recorded).
- LABEL: transcript-grade ONLY at assembly time — no FREEZE record, no -O companion transcript, no mutation
  harness run, and no D1-gate consumption exists yet. NOT evidence; feeds OBL-H5-ZBAND / OBL-D1-PROMOTE
  when the lane freezes it. (If later frozen and consumed, this is the continuum-band piece whose absence
  keeps Theorem (2)'s r₀ existential.)
- Next milestone: both-mode + mutation runs → freeze record → D1 gate consumption.

## 6. Lane summary

| Lane | State | Last verified artifact | Next milestone |
|---|---|---|---|
| H5 promotion | EXECUTING | h5_totals_v3.json (frozen); rung jsonl banks | r = 0.025 cells+stitch close → merge |
| W3 LOWER | RUNNING (stale ~90 min) | lbox_A/B.txt 06:45 CST; harness receipts | band completion → G2 vs W4 |
| W8 Phase-2 | RUNNING (relaunched 07:54) | Phase-1 banks; nonclosure report | 4 shard receipts → v4 verification |
| BRANCH | COMPLETE | rung025.log + post-errata byte-identical products | none open |
| H3 band | RUN COMPLETE, unfrozen | band_normal.txt ALL_CHECKS_PASS (transcript-grade) | freeze + consumption |
