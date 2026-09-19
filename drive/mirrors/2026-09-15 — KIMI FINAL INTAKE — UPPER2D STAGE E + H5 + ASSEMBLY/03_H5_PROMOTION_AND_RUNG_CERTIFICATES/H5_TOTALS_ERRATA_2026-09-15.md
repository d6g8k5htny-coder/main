# H5 totals — ERRATA (2026-09-15): v2 I_lo contamination root-caused and fixed

Correction to H5_CLOSURE_TIGHTENING_2026-09-15.md (frozen, body 7fefa17b…)
and to the v2 totals freeze. New package per the corrections rule; the frozen
documents are not altered.

## Root cause (exact)

The pre-fix h5_merge.py loaded records with the RUNG-UNSCOPED glob
`h5_results_*.jsonl`. During the OBL-D1-PROMOTE rung-driver shakedown
(2026-09-14 ~19:55) the r = 0.025 driver banked 8 patch records
(h5_results_r0.025_s0p1_patches.jsonl, part='patch'). The unscoped merger
summed them into patches_lo, drifting the live r = 0.05 totals:
    I_lo: 1.166721190e-08  ->  1.2328893552e-08   (rung patches added)
The drifted bytes were frozen into h5_totals_v2_2026-09-15.json and cited by
the tightening amendment (its "I_lo upward: more patch mass banked" line is
thereby superseded — the increase was rung pollution, not new r = 0.05 mass).

## v2 I_hi is CLEAN (verified two ways)

1. Rung R1 cells were banked only AFTER the v2 freeze (driver launches
   post-date it); a name-collision audit found exactly 2 colliding cells
   (th160_dl0.033, th165_dl0.033), both banked post-freeze.
2. The fixed (rung-scoped) merger reproduces v2's I_hi to all printed digits:
   8.0975589252e-02 = 647.8047 r^3.

## Correct clean totals (h5_totals_v3.json, sha256
8d7028e4d0d49d54d3a7c4589a898289ec29eee6c2de9c774e99df0da46596bb)

    I_lo = 1.166721190e-08   (== the v1/D1-frozen pin; Stage-E gate green)
    I_hi = 8.0975589252e-02  (= 647.8047 r^3; refine-3 genuine, downward-only
                              vs the frozen v1 9.142887704e-02)
    containment ck PASS; coverage self-test PASS; mutation suite 6/6.

## Fix shipped

- h5_merge.py: record glob rung-scoped to `h5_results_r<r>_*.jsonl`;
  versioned 'x'-mode writes only (h5_totals.json never rewritten).
- h5_run.py: driver resume-skip glob rung-scoped (2 sites). Running rung
  drivers restarted on the fixed generator (keeper_promote.sh A/B).
- Collision hazard note: rung cell labels can coincide with r = 0.05 labels
  (0.066·0.5 = 0.033); the rung scoping makes this harmless, and the rung
  mergers (h5_merge.py --r 0.025 / --r 0.035355) read only their own rung.
