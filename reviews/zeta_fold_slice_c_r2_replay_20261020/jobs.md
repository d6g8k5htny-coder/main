# Jobs

59 jobs from `reviews/zeta_fold_slice_c_rerun_20261019/scripts/driver.py` at source commit `5cab7663820898aaee2466cbc8662eb04984cde0`, both lanes concurrent (`ZF_LANE=dbn` and `ZF_LANE=stats`).
`$PY` is `/tmp/r2-venv/bin/python`. Working directory is the packet unless the argv path is `experiments/universality/fold_controls.py`, which the driver runs from the repository root.
Every exit code is 0. `test_numeric` writes its unittest stream to stderr (2024 bytes); its stdout file is empty.

Summed wall time: 2492.9 s. Earliest start 2026-10-20T02:36:32Z. Span from earliest start to latest finish: 728.8 s.

| chunk | job | rc | wall s | command |
|---|---|---:|---:|---|
| C1 | `test_numeric` | 0 | 1.0 | `$PY -B -m unittest -v test_exact test_numeric` |
| C1 | `validate_H0` | 0 | 3.8 | `$PY -B dbn/validate_H0.py` |
| C1 | `count_1_103_-1.87` | 0 | 2.7 | `$PY -B dbn/count_check.py 1 103 -1.87` |
| C1 | `count_1_103_-1.89` | 0 | 2.7 | `$PY -B dbn/count_check.py 1 103 -1.89` |
| C1 | `count_1_103_0` | 0 | 2.9 | `$PY -B dbn/count_check.py 1 103 0` |
| C1 | `count_1_270_-0.38` | 0 | 7.6 | `$PY -B dbn/count_check.py 1 270 -0.38` |
| C1 | `count_1_270_-0.39` | 0 | 7.2 | `$PY -B dbn/count_check.py 1 270 -0.39` |
| C1 | `count_1_270_0` | 0 | 8.6 | `$PY -B dbn/count_check.py 1 270 0` |
| C1 | `indep_collision` | 0 | 3.4 | `$PY -B dbn/indep_collision.py` |
| C1 | `kappa_check` | 0 | 0.3 | `$PY -B dbn/kappa_check.py` |
| C1 | `kappa_indep_degree6` | 0 | 0.4 | `$PY -B dbn/kappa_indep_degree6.py` |
| C1 | `kappa_indep_random` | 0 | 0.8 | `$PY -B dbn/kappa_indep_random.py` |
| C1 | `zero_velocity` | 0 | 8.0 | `$PY -B dbn/zero_velocity.py` |
| C1 | `smoke_hflow` | 0 | 0.6 | `$PY -B dbn/smoke_hflow.py` |
| C1 | `indep_ht_xi` | 0 | 12.8 | `$PY -B dbn/indep_ht.py 60 xi` |
| C1 | `check_theta` | 0 | 1.2 | `$PY -B theta/check_theta.py` |
| C1 | `check_gue_transfer` | 0 | 0.9 | `$PY -B stats/check_gue_transfer.py` |
| C1 | `fold_controls` | 0 | 0.0 | `$PY -B experiments/universality/fold_controls.py` |
| C2 | `track_z270` | 0 | 152.7 | `$PY -B dbn/track.py 270` |
| C2 | `track_low10` | 0 | 43.1 | `$PY -B dbn/track.py 103 -40 0.02 low10` |
| C2 | `collision_table` | 0 | 0.0 | `$PY -B dbn/collision_table.py` |
| C2 | `fold_pair34_35` | 0 | 8.0 | `$PY -B dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35` |
| C2 | `fold_pair9_10` | 0 | 2.3 | `$PY -B dbn/fold_check.py -1.8779090730 99.3284693056 pair9_10` |
| C2 | `fold_analyze` | 0 | 0.1 | `$PY -B dbn/fold_analyze.py pair34_35 pair9_10` |
| C2 | `refine_check` | 0 | 27.8 | `$PY -B dbn/refine_check.py` |
| C3 | `track_dt0.01` | 0 | 28.4 | `$PY -B dbn/track.py 103 -40 0.01 dt0.01` |
| C3 | `track_dt0.02` | 0 | 40.4 | `$PY -B dbn/track.py 103 -40 0.02 dt0.02` |
| C3 | `track_dt0.025` | 0 | 40.6 | `$PY -B dbn/track.py 103 -40 0.025 dt0.025` |
| C3 | `track_dt0.04` | 0 | 13.5 | `$PY -B dbn/track.py 103 -40 0.04 dt0.04` |
| C3 | `track_dt0.05` | 0 | 26.9 | `$PY -B dbn/track.py 103 -40 0.05 dt0.05` |
| C4 | `check_dbn_basic` | 0 | 82.0 | `$PY -B dbn/check_dbn_basic.py` |
| C4 | `check_collision` | 0 | 127.1 | `$PY -B dbn/check_collision.py` |
| C4 | `check_collision_persistence` | 0 | 31.3 | `$PY -B dbn/check_collision_persistence.py` |
| C4 | `indep_ht_c910` | 0 | 142.3 | `$PY -B dbn/indep_ht.py 60 c910` |
| C4 | `indep_ht_c3435` | 0 | 410.6 | `$PY -B dbn/indep_ht.py 60 c3435` |
| C4 | `lobe_identity` | 0 | 335.7 | `$PY -B dbn/lobe_identity.py` |
| C4 | `project_theta_checks` | 0 | 223.8 | `$PY -B theta/project_theta_checks.py` |
| C5 | `rs_validate` | 0 | 10.8 | `$PY -B stats/rs_zeta.py validate` |
| C5 | `rs_A` | 0 | 6.3 | `$PY -B stats/rs_zeta.py A 1e4 1e5` |
| C5 | `rs_B` | 0 | 173.2 | `$PY -B stats/rs_zeta.py B 1e5 1e6` |
| C5 | `gauss` | 0 | 14.1 | `$PY -B stats/gauss_proc.py 500 4` |
| C5 | `cue12` | 0 | 5.9 | `$PY -B stats/cue.py 12 86016 4` |
| C5 | `cue128` | 0 | 68.0 | `$PY -B stats/cue.py 128 8064 4` |
| C6 | `analyze` | 0 | 23.2 | `$PY -B stats/analyze.py` |
| C6 | `matched_tail` | 0 | 5.1 | `$PY -B stats/matched_tail.py` |
| C6 | `make_plots` | 0 | 0.4 | `$PY -B stats/make_plots.py` |
| C6 | `check_gauss` | 0 | 0.8 | `$PY -B stats/check_gauss.py` |
| C6 | `check_zeta_quad` | 0 | 13.5 | `$PY -B stats/check_zeta_quad.py` |
| C6 | `barcode_bruteforce` | 0 | 0.4 | `$PY -B stats/barcode_bruteforce.py` |
| C6 | `gauss_hidden` | 0 | 18.5 | `$PY -B stats/gauss_hidden.py` |
| C6 | `localamp_baseline` | 0 | 8.6 | `$PY -B stats/localamp_baseline.py` |
| C6 | `cue_gap_shape` | 0 | 15.1 | `$PY -B stats/cue_gap_shape.py` |
| C6 | `rs_spot_check` | 0 | 0.9 | `$PY -B stats/rs_spot_check.py` |
| C6 | `cue_endpoint_check` | 0 | 0.4 | `$PY -B stats/cue_endpoint_check.py` |
| C6 | `block_sensitivity` | 0 | 4.6 | `$PY -B stats/block_sensitivity.py` |
| C7 | `sample_600` | 0 | 119.1 | `$PY -B stats/sample_Z.py 600 900 0.02` |
| C7 | `sample_5000` | 0 | 202.1 | `$PY -B stats/sample_Z.py 5000 5200 0.015` |
| C7 | `checkZ_600` | 0 | 0.2 | `$PY -B stats/check_Z_persistence.py Z_grid_600_900.npz` |
| C7 | `checkZ_5000` | 0 | 0.2 | `$PY -B stats/check_Z_persistence.py Z_grid_5000_5200.npz` |

Per-chunk summed wall seconds: C1 64.9, C2 234.0, C3 149.8, C4 1352.8, C5 278.3, C6 91.5, C7 321.6
