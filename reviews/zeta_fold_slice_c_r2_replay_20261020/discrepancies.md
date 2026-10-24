# Discrepancies

Nothing was edited, fixed, or rerun after the first pass. Both driver lanes exited 0. All 59 jobs exited 0.

## Counts against the committed packet

- 66 tracked files under `results/`: 50 IDENTICAL, 14 TIMING-ONLY, 2 DIFFERENT.
- `results/dbn/track_low10_dt_sensitivity.out`, rebuilt from the five `track_dt*` stdout logs by the header rule, is IDENTICAL to the committed file.
- 9 of 9 raw samples match `MANIFEST.json` in SHA-256 and byte length.
- The five untracked `results/dbn/track_log_dt*.txt` files were left in the execution clone and are not in this record. Their SHA-256 values:

| file | sha256 |
|---|---|
| `track_log_dt0.01.txt` | `38ef62f02aad4c45bcfde7fa23a140353fb6f3dd1e85fc5b2f9a0ee05d84d612` |
| `track_log_dt0.02.txt` | `756ed7b7bd759c83613a8a24986a15562d955b73abb6bdc4fb923b0c706b6266` |
| `track_log_dt0.025.txt` | `e2973730468b0c0eacaf97ee70ba83be89b95812afb523fa118b7d3075297f84` |
| `track_log_dt0.04.txt` | `edbd12eb54668f2d22b26fe27877292a8f6027509174d3597d1b2c9880521bcf` |
| `track_log_dt0.05.txt` | `a4d38f87fdd71148132f7e60357f234db45ef6bd838405f67086956e9b785341` |

## The two DIFFERENT files

`results/dbn/indep_ht_c910.out` and `results/dbn/indep_ht_c3435.out`. Unified diffs are in `diffs/`. Line quotes are in `residuals.md`.

Converged lines are the same as the committed file in both cases: `t*`, `z0`, `A`, `B/A`, `C/A`, `kappa_r`, `kappa_l`.

Post-convergence residual steps differ from the committed file and match the same-provider copy at `reviews/zeta_fold_slice_c_rerun_20261019/reproduced/results/dbn/`, except the `time` line.

`indep_ht_c910.out`, iteration 4, the only residual-step line that differs:

```
committed:      4 -1.87790907297253526476683971813 99.328469305575750948022768886 4.128e-50
this replay:    4 -1.87790907297253526476683971813 99.328469305575750948022768886 8.3279e-51
same-provider:  4 -1.87790907297253526476683971813 99.328469305575750948022768886 8.3279e-51
```

`indep_ht_c3435.out`, from iteration 3. The committed file stops at iteration 4. This replay and the same-provider copy both continue through iteration 11:

```
committed:      3 ... 8.2634e-32
committed:      4 ... 1.2375e-63
this replay:    3 ... 8.2336e-32
this replay:    4 ... 1.7025e-33
this replay:    5 ... 2.4785e-34
this replay:    6 ... 1.2692e-33
this replay:    7 ... 1.2365e-33
this replay:    8 ... 7.0004e-34
this replay:    9 ... 1.0982e-33
this replay:   10 ... 5.585e-34
this replay:   11 ... 6.2944e-34
```

The same-provider copy has those same eight replay lines (iterations 3 through 11). `t*` and `z0` on every one of those lines are the committed values. The `time` lines differ across all three copies and are timing, not the residual trail.

This replay does not explain the trail. It reproduces the same-provider trail rather than the committed trail.

## Class membership versus the same-provider rerun

The same-provider `comparison/COMPARE.md` has 14 TIMING-ONLY files including `results/dbn/indep_collision.out` and not including `results/stats/rs_spot_check.out`. This replay swaps those two, and the count stays 14.

`indep_collision.out` is byte-identical to the committed file here (440 bytes). The same-provider copy differs only in two timing tokens, which the mask would call TIMING-ONLY:

```
committed / this replay:   (1 s)    and   (2 s)
same-provider copy:        (2 s)    and   (3 s)
```

The other lines of that file match.

`rs_spot_check.out` is TIMING-ONLY here. One token differs:

```
committed:    window [999700,1e+06]: ours 573  nzeros 573  (0.1s)
this replay:  window [999700,1e+06]: ours 573  nzeros 573  (0.0s)
```

The counts on that line are 573 and 573 in both copies. The mask `\([0-9.]+ ?s\)` treats `(0.1s)` and `(0.0s)` as the same timing token.

## Environment deviations

Recorded in `environment.md`: Clang-built uv CPython 3.13.16 instead of a distro `/usr/bin/python3.13` venv; kernel `6.12.94+`; CPU at 2400 MHz. Numpy, mpmath, the mpmath backend, and the OpenBLAS version string match the pinned run.
