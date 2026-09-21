# H5 state (2026-09-15 ~14:40) — OBL-D1-PROMOTE rung family executing

## Version discipline (Stage-E R4)
h5_totals.json = v1 consumed artifact (never rewritten; bytes currently the
drifted pre-fix state, lineage in H5_TOTALS_FREEZE_2026-09-15.md).
Versions: v2_2026-09-15 (I_lo contaminated — see ERRATA), v3 = CLEAN live
(I_lo 1.166721190e-08 == v1 pin; I_hi 8.0975589252e-02 = 647.8047 r^3).
Merges write h5_totals_v{N}.json ('x'-mode, fail-closed). Merger is
rung-scoped + rung-scaled (s = r/0.05); bremote parameter (default 19.55 =
frozen v1 consumption; D3 kappa-inclusive value when R3 lands).

## Frozen documents (do not alter)
H5_CLOSURE.md body 465c97c0553ecc0a8f461a86c9315ce1c078a326f07b81498d7a74da8603d301
H5_CLOSURE_TIGHTENING_2026-09-15.md body 7fefa17b… (I_lo line superseded by errata)
H5_TOTALS_ERRATA_2026-09-15.md sha a7ce5299bdaa…
H5_PROMOTE.md body 8d8b353815ec… ; H5_PROMOTE_UPDATE_2026-09-15.md sha 535a098768da…

## Rung fleet (keeper_promote.sh A/B supervises: relaunch = setsid nohup sh keeper_promote.sh {A,B})
python3 promote_run.py --r {0.025,0.035355} --part cells --shard {0,1} --nshards 2
python3 promote_run.py --r {0.025,0.035355} --part stitch --shard 0 --nshards 1
done: r=0.025 patches/probes; r=0.035355 patches/rimprobes/probes
rung merge (when cells+stitch close): python3 h5_merge.py --r 0.025
  (fail-closed: 70 cells, 10 rim cols, 6+12+5+6 probe sites, 8 patches,
   22 stitch sectors, rung-scaled coverage grid, C1_TOTAL(r)=1.605315e-4*s^3 ck)
refine-3 (SECONDARY, paused 14:05): relaunch = setsid nohup sh keeper.sh {A,B}
