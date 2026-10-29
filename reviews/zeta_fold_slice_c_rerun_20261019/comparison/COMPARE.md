# Slice C rerun vs committed outputs

Jobs run: 59; nonzero rc: none; summed wall time: 3670.6 s.
Committed output files under results/: 66; byte-unchanged after rerun: 50.
dt-sensitivity file: rebuilt.

| changed file | class | old bytes | new bytes | diff lines |
|---|---|---|---|---|
| `results/dbn/check_dbn_basic.out` | TIMING-ONLY | 2021 | 2022 |  |
| `results/dbn/indep_collision.out` | TIMING-ONLY | 440 | 440 |  |
| `results/dbn/indep_ht_c3435.out` | DIFFERENT | 591 | 1143 | 13 |
| `results/dbn/indep_ht_c910.out` | DIFFERENT | 579 | 580 | 4 |
| `results/dbn/indep_ht_xi.out` | TIMING-ONLY | 305 | 306 |  |
| `results/dbn/smoke_hflow.out` | TIMING-ONLY | 1139 | 1139 |  |
| `results/dbn/validate_H0.out` | TIMING-ONLY | 2955 | 2956 |  |
| `results/stats/cue128_info.json` | TIMING-ONLY | 180 | 180 |  |
| `results/stats/cue12_info.json` | TIMING-ONLY | 177 | 177 |  |
| `results/stats/gauss_info.json` | TIMING-ONLY | 231 | 231 |  |
| `results/stats/rs_validation.json` | TIMING-ONLY | 1127 | 1127 |  |
| `results/stats/sample_5000.log` | TIMING-ONLY | 77 | 77 |  |
| `results/stats/sample_600.log` | TIMING-ONLY | 75 | 75 |  |
| `results/stats/summary.json` | TIMING-ONLY | 28761 | 28760 |  |
| `results/stats/zeta_A_info.json` | TIMING-ONLY | 310 | 311 |  |
| `results/stats/zeta_B_info.json` | TIMING-ONLY | 319 | 320 |  |

| raw sample | manifest sha256 | rerun sha256 | match |
|---|---|---|---|
| `results/raw/zeta_A.npz` | `f365833e32edbf35…` | `f365833e32edbf35…` | True |
| `results/raw/zeta_B.npz` | `6104917e71cfb2f3…` | `6104917e71cfb2f3…` | True |
| `results/raw/rs_cheb_coeffs.npz` | `7b86066629cb2be0…` | `7b86066629cb2be0…` | True |
| `results/raw/gauss.npz` | `1a705fbed3fe41d9…` | `1a705fbed3fe41d9…` | True |
| `results/raw/cue12.npz` | `a7cbd4fd4db065d1…` | `a7cbd4fd4db065d1…` | True |
| `results/raw/cue128.npz` | `a96192115efc6a1e…` | `a96192115efc6a1e…` | True |
| `results/raw/ecdf_samples.npz` | `e368290b558a3b59…` | `e368290b558a3b59…` | True |
| `results/raw/Z_grid_600_900.npz` | `30fb1b7b579fcf95…` | `30fb1b7b579fcf95…` | True |
| `results/raw/Z_grid_5000_5200.npz` | `a30f88b7844e84ae…` | `a30f88b7844e84ae…` | True |

Untracked files left by the rerun: `experiments/zeta_fold_exploration/results/dbn/track_log_dt0.01.txt`, `experiments/zeta_fold_exploration/results/dbn/track_log_dt0.02.txt`, `experiments/zeta_fold_exploration/results/dbn/track_log_dt0.025.txt`, `experiments/zeta_fold_exploration/results/dbn/track_log_dt0.04.txt`, `experiments/zeta_fold_exploration/results/dbn/track_log_dt0.05.txt`

## diff excerpt: `results/dbn/indep_ht_c3435.out`
```
--- committed
+++ rerun
@@ -4,2 +4,9 @@
-3 -0.384561116627981414864807288345 223.275780185615519913542719805 8.2634e-32
-4 -0.384561116627981414864807288345 223.275780185615519913542719805 1.2375e-63
+3 -0.384561116627981414864807288345 223.275780185615519913542719805 8.2336e-32
+4 -0.384561116627981414864807288345 223.275780185615519913542719805 1.7025e-33
+5 -0.384561116627981414864807288345 223.275780185615519913542719805 2.4785e-34
+6 -0.384561116627981414864807288345 223.275780185615519913542719805 1.2692e-33
+7 -0.384561116627981414864807288345 223.275780185615519913542719805 1.2365e-33
+8 -0.384561116627981414864807288345 223.275780185615519913542719805 7.0004e-34
+9 -0.384561116627981414864807288345 223.275780185615519913542719805 1.0982e-33
+10 -0.384561116627981414864807288345 223.275780185615519913542719805 5.585e-34
+11 -0.384561116627981414864807288345 223.275780185615519913542719805 6.2944e-34
@@ -8 +15 @@
-time 413.21186900138855
+time 604.147679567337
```

## diff excerpt: `results/dbn/indep_ht_c910.out`
```
--- committed
+++ rerun
@@ -5 +5 @@
-4 -1.87790907297253526476683971813 99.328469305575750948022768886 4.128e-50
+4 -1.87790907297253526476683971813 99.328469305575750948022768886 8.3279e-51
@@ -8 +8 @@
-time 112.64180207252502
+time 172.69612407684326
```
