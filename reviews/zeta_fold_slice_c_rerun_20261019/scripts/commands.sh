#!/usr/bin/env bash
# Exact commands of the rerun, from comparison/ledger.json. $PY = pinned interpreter
# (Python 3.13.16 venv with numpy==2.5.3 mpmath==1.4.1); $WT = clean checkout of main 0a7e5145b55f020e02e7782fed318266d13c2447.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

# [dbn] test_numeric  start 2026-10-19T19:52:02Z  wall 1.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B -m unittest -v test_exact test_numeric)
# [dbn] count_1_103_-1.87  start 2026-10-19T19:52:02Z  wall 3.9s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 103 -1.87)  # stdout -> results/dbn/count_1_103_-1.87.out
# [dbn] validate_H0  start 2026-10-19T19:52:02Z  wall 5.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/validate_H0.py)  # stdout -> results/dbn/validate_H0.out
# [dbn] count_1_103_-1.89  start 2026-10-19T19:52:04Z  wall 3.8s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 103 -1.89)  # stdout -> results/dbn/count_1_103_-1.89.out
# [dbn] count_1_103_0  start 2026-10-19T19:52:06Z  wall 3.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 103 0)  # stdout -> results/dbn/count_1_103_0.out
# [dbn] count_1_270_-0.39  start 2026-10-19T19:52:08Z  wall 10.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 270 -0.39)  # stdout -> results/dbn/count_1_270_-0.39.out
# [dbn] count_1_270_-0.38  start 2026-10-19T19:52:08Z  wall 11.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 270 -0.38)  # stdout -> results/dbn/count_1_270_-0.38.out
# [dbn] count_1_270_0  start 2026-10-19T19:52:10Z  wall 12.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/count_check.py 1 270 0)  # stdout -> results/dbn/count_1_270_0.out
# [dbn] check_dbn_basic  start 2026-10-19T19:52:19Z  wall 112.8s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/check_dbn_basic.py)  # stdout -> results/dbn/check_dbn_basic.out
# [dbn] check_collision  start 2026-10-19T19:52:19Z  wall 142.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/check_collision.py)  # stdout -> results/dbn/check_collision.out
# [dbn] refine_check  start 2026-10-19T19:52:22Z  wall 42.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/refine_check.py)  # stdout -> results/dbn/refine_check.out
# [dbn] indep_collision  start 2026-10-19T19:53:04Z  wall 4.4s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/indep_collision.py)  # stdout -> results/dbn/indep_collision.out
# [dbn] kappa_check  start 2026-10-19T19:53:09Z  wall 0.3s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/kappa_check.py)  # stdout -> results/dbn/kappa_check.out
# [dbn] kappa_indep_degree6  start 2026-10-19T19:53:09Z  wall 0.4s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/kappa_indep_degree6.py)  # stdout -> results/dbn/kappa_indep_degree6.out
# [dbn] kappa_indep_random  start 2026-10-19T19:53:10Z  wall 1.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/kappa_indep_random.py)  # stdout -> results/dbn/kappa_indep_random.out
# [dbn] zero_velocity  start 2026-10-19T19:53:11Z  wall 9.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/zero_velocity.py)  # stdout -> results/dbn/zero_velocity.out
# [dbn] smoke_hflow  start 2026-10-19T19:53:20Z  wall 0.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/smoke_hflow.py)  # stdout -> results/dbn/smoke_hflow.out
# [dbn] indep_ht_xi  start 2026-10-19T19:53:20Z  wall 14.3s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/indep_ht.py 60 xi)  # stdout -> results/dbn/indep_ht_xi.out
# [dbn] indep_ht_c910  start 2026-10-19T19:53:35Z  wall 172.8s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/indep_ht.py 60 c910)  # stdout -> results/dbn/indep_ht_c910.out
# [dbn] indep_ht_c3435  start 2026-10-19T19:54:11Z  wall 604.4s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/indep_ht.py 60 c3435)  # stdout -> results/dbn/indep_ht_c3435.out
# [dbn] lobe_identity  start 2026-10-19T19:54:41Z  wall 500.5s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/lobe_identity.py)  # stdout -> results/dbn/lobe_identity.out
# [dbn] track_z270  start 2026-10-19T19:56:28Z  wall 229.4s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 270)  # stdout -> results/dbn/track_z270.out
# [dbn] track_low10  start 2026-10-19T20:00:17Z  wall 60.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.02 low10)  # stdout -> results/dbn/track_low10.out
# [dbn] track_dt0.01  start 2026-10-19T20:01:17Z  wall 43.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.01 dt0.01)
# [dbn] track_dt0.02  start 2026-10-19T20:02:00Z  wall 60.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.02 dt0.02)
# [dbn] track_dt0.04  start 2026-10-19T20:03:01Z  wall 21.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.04 dt0.04)
# [dbn] track_dt0.025  start 2026-10-19T20:03:01Z  wall 61.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.025 dt0.025)
# [dbn] track_dt0.05  start 2026-10-19T20:03:23Z  wall 40.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/track.py 103 -40 0.05 dt0.05)
# [dbn] check_theta  start 2026-10-19T20:04:02Z  wall 2.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B theta/check_theta.py)  # stdout -> results/theta/check_theta.out
# [dbn] project_theta_checks  start 2026-10-19T20:04:03Z  wall 352.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B theta/project_theta_checks.py)  # stdout -> results/theta/project_theta_checks.out
# [dbn] check_gue_transfer  start 2026-10-19T20:04:04Z  wall 1.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/check_gue_transfer.py)  # stdout -> results/stats/check_gue_transfer.out
# [dbn] fold_controls  start 2026-10-19T20:04:05Z  wall 0.0s  rc 0
(cd $WT && $PY -B experiments/universality/fold_controls.py)  # stdout -> results/theta/fold_controls_rerun.out
# [dbn] check_collision_persistence  start 2026-10-19T20:04:05Z  wall 45.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/check_collision_persistence.py)  # stdout -> results/dbn/check_collision_persistence.out
# [dbn] collision_table  start 2026-10-19T20:04:16Z  wall 0.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/collision_table.py)
# [dbn] fold_pair34_35  start 2026-10-19T20:04:16Z  wall 11.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35)  # stdout -> results/dbn/fold_pair34_35.out
# [dbn] fold_pair9_10  start 2026-10-19T20:04:28Z  wall 3.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/fold_check.py -1.8779090730 99.3284693056 pair9_10)  # stdout -> results/dbn/fold_pair9_10.out
# [dbn] fold_analyze  start 2026-10-19T20:04:31Z  wall 0.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B dbn/fold_analyze.py pair34_35 pair9_10)  # stdout -> results/dbn/fold_analyze.out
# [stats] rs_validate  start 2026-10-19T19:52:11Z  wall 16.9s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/rs_zeta.py validate)
# [stats] rs_A  start 2026-10-19T19:52:28Z  wall 13.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/rs_zeta.py A 1e4 1e5)
# [stats] rs_B  start 2026-10-19T19:52:42Z  wall 266.4s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/rs_zeta.py B 1e5 1e6)
# [stats] gauss  start 2026-10-19T19:57:08Z  wall 19.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/gauss_proc.py 500 4)
# [stats] cue12  start 2026-10-19T19:57:28Z  wall 7.0s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/cue.py 12 86016 4)
# [stats] cue128  start 2026-10-19T19:57:35Z  wall 96.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/cue.py 128 8064 4)
# [stats] analyze  start 2026-10-19T19:59:11Z  wall 42.8s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/analyze.py)
# [stats] matched_tail  start 2026-10-19T19:59:54Z  wall 9.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/matched_tail.py)
# [stats] make_plots  start 2026-10-19T20:00:03Z  wall 0.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/make_plots.py)
# [stats] sample_600  start 2026-10-19T20:00:03Z  wall 190.3s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/sample_Z.py 600 900 0.02)  # stdout -> results/stats/sample_600.log
# [stats] sample_5000  start 2026-10-19T20:03:14Z  wall 319.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/sample_Z.py 5000 5200 0.015)  # stdout -> results/stats/sample_5000.log
# [stats] checkZ_600  start 2026-10-19T20:08:33Z  wall 0.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/check_Z_persistence.py Z_grid_600_900.npz)  # stdout -> results/stats/Z_grid_600_900_persistence.out
# [stats] checkZ_5000  start 2026-10-19T20:08:33Z  wall 0.3s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/check_Z_persistence.py Z_grid_5000_5200.npz)  # stdout -> results/stats/Z_grid_5000_5200_persistence.out
# [stats] barcode_bruteforce  start 2026-10-19T20:08:33Z  wall 0.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/barcode_bruteforce.py)  # stdout -> results/stats/barcode_bruteforce.out
# [stats] check_gauss  start 2026-10-19T20:08:33Z  wall 1.1s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/check_gauss.py)
# [stats] check_zeta_quad  start 2026-10-19T20:08:33Z  wall 20.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/check_zeta_quad.py)
# [stats] localamp_baseline  start 2026-10-19T20:08:34Z  wall 22.5s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/localamp_baseline.py)  # stdout -> results/stats/localamp_baseline.out
# [stats] gauss_hidden  start 2026-10-19T20:08:34Z  wall 25.6s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/gauss_hidden.py)  # stdout -> results/stats/gauss_hidden.out
# [stats] cue_gap_shape  start 2026-10-19T20:08:54Z  wall 22.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/cue_gap_shape.py)  # stdout -> results/stats/cue_gap_shape.out
# [stats] rs_spot_check  start 2026-10-19T20:08:57Z  wall 1.2s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/rs_spot_check.py)  # stdout -> results/stats/rs_spot_check.out
# [stats] cue_endpoint_check  start 2026-10-19T20:08:58Z  wall 0.5s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/cue_endpoint_check.py)  # stdout -> results/stats/cue_endpoint_check.out
# [stats] block_sensitivity  start 2026-10-19T20:08:59Z  wall 4.7s  rc 0
(cd $WT/experiments/zeta_fold_exploration && $PY -B stats/block_sensitivity.py)
