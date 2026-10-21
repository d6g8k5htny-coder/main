# Zeta fold residual-trail diagnostics: retained outputs (retention only)

**Retention only. Not for merge. No review, no packet change, no status change. Scientific effect NONE.**

This directory keeps the raw outputs behind three diagnostic data notes on main#458, so they no longer exist only in one session's ephemeral container:
- [6303179629](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6303179629): `c910` against integer dps, and a warm-cache run;
- [6304057120](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304057120): `c3435` against integer dps, and `c910` extended;
- [6321367411](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6321367411): both trails against every working precision in bits within a window.

It also keeps two later checks, `versions_dps60/` and `old_script_c561d95b/`, which this README describes and #458 links to.

The notes are the interpretation; this directory is the evidence. **Update, 21 October:** `old_script_c561d95b/` reproduces both committed residual-trail files exactly apart from `time`, using the packet's earlier review-math script at dps 55 (`c910`) and dps 90 (`c3435`). The earlier sweeps below used the current script and could not find it. That section has the details and limits. In the earlier sweeps, every output line other than the iteration-4 step and `time` matches the committed `results/dbn/indep_ht_c910.out` and `indep_ht_c3435.out`, so reported values (`t*`, `z0`, A, B/A, C/A, κ) are unaffected.

Producer: Anthropic Claude Opus 5.5 (`claude-opus-5-5`), Claude Code session `session_01DKKe26v3CSgdYch8kiJLt9`, the same provider as the packet author. Data, not a review. Credit 0.

## Source and environment

`experiments/zeta_fold_exploration/dbn/indep_ht.py`, git blob `736760f2`, from a worktree at `main` `157b6a97`. The same blob is at the replay source `5cab7663` and at `main` `1e16211d`. The environment is in [`environment.txt`](environment.txt): CPython 3.13.16, numpy 2.5.3, mpmath 1.4.1 (python backend), single-threaded BLAS.

## `integer_dps/`

Run from `experiments/zeta_fold_exploration/`:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B dbn/indep_ht.py <dps> c910     # dps 50–61, 65, 70, 80, 85, 90, 95, 100
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B dbn/indep_ht.py <dps> c3435    # dps 55–57, 60, 65, 70, 75, 80, 85, 90–95, 100, 105, 110, 120
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B warm.py dbn/indep_ht.py 60:xi 60:c910   # both modes in one process
```

The output for each is `<collision>_dps<dps>.out`, and the warm-cache output is `warm_xi_then_c910.out`. [`warm.py`](integer_dps/warm.py) runs the two modes in one interpreter so that mpmath's quadrature caches are warm for the second. Every run exited 0 with empty stderr ([`run_log.txt`](integer_dps/run_log.txt); stderr files not kept). [`SHA256SUMS`](integer_dps/SHA256SUMS) is the file inlined in the first two notes, unchanged, and covers all 39 outputs and `warm.py`.

## `bit_prec/`

[`run_prec.py`](bit_prec/run_prec.py) loads `indep_ht.py` unchanged, sets `mp.mp.prec = <bits>` directly, and calls the same `collide(...)` with the documented start values:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B run_prec.py <dir containing indep_ht.py> <bits> c910    # 168–207 bits, skipping bit counts an integer dps already sets
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B run_prec.py <dir containing indep_ht.py> <bits> c3435   # 287–311 bits, same rule
```

The first output line, `#prec <bits> dps <dps>`, records the precision actually set. `c910_p203.out` is the sanity run: 203 bits is what dps 60 sets, and the output matches `integer_dps/c910_dps60.out` byte for byte apart from the `#prec` and `time` lines. [`src/indep_ht.py`](bit_prec/src/indep_ht.py) is a byte copy of blob `736760f2`, kept so that [`SHA256SUMS`](bit_prec/SHA256SUMS) (SHA-256 `dd4c09b5…d8f6ea`, cited in the third note) checks complete. All 51 runs exited 0 with empty stderr. [`run_log.txt`](bit_prec/run_log.txt) lists the 50 sweep runs; the sanity run was launched separately.

## `versions_dps60/`

`c3435` at the documented `60` under four other environments, all with the python backend:
- mpmath 1.3.0 and 1.4.0, on Python 3.13.16;
- mpmath 1.4.1, on Python 3.11.17 and on Python 3.12.3.

The environments are listed in [`environments.txt`](versions_dps60/environments.txt). The command is the same as in `integer_dps/`, and every run exited 0 with empty stderr ([`run_log.txt`](versions_dps60/run_log.txt)). **All four outputs equal `integer_dps/c3435_dps60.out` apart from `time`.** That is the same non-converging trail, with iteration 3 at 8.2336e-32 rather than the committed 8.2634e-32. R1 already found the same for the gmpy backend (`reviews/zeta_fold_slice_c_rerun_20261019/experiments/gmpy_indep_ht_c3435.out`). So in none of these environments does the documented invocation produce the committed `c3435` file. [`SHA256SUMS`](versions_dps60/SHA256SUMS) covers the four outputs.

## `old_script_c561d95b/`: the exact reproductions

The committed `indep_ht_c910.out` and `indep_ht_c3435.out` are the same git blobs as `zeta_raw/review-math/c910.out` and `c3435.out` in the packet author's recovery checkpoint (`recovery/session_01S16677s1SGeAJmN3QMz4ip/`, commit `5d298b9e`). That folder also holds the script that wrote them, `review-math/indep_ht.py`, blob `c561d95b`, copied here as [`src/indep_ht.py`](old_script_c561d95b/src/indep_ht.py).

The one functional difference from the packet's `736760f2` is where the requested dps is applied:
- **`c561d95b`** sets `mp.mp.dps` from `argv` at import, before `U = mp.mpf('1.3')`.
- **`736760f2`** sets dps 60 at import and applies `argv` only in `__main__`, so `U` is always the 203-bit rounding of 1.3.

Run from this directory:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B src/indep_ht.py <dps> c910     # dps 50–70
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B src/indep_ht.py <dps> c3435    # dps 85–100
```

**Results:**
- `c910_dps55.out` equals the committed `indep_ht_c910.out` apart from `time`. No other dps from 50 to 70 does.
- `c3435_dps90.out` equals the committed `indep_ht_c3435.out` apart from `time`. No other dps from 85 to 100 does.
- All 37 runs exited 0 with empty stderr ([`run_log.txt`](old_script_c561d95b/run_log.txt)).
- Runs at c3435 dps 61–84 were not completed and are not kept. At those precisions the iteration-4 step is many orders of magnitude above 1e-63 (`integer_dps/`).

[`SHA256SUMS`](old_script_c561d95b/SHA256SUMS) covers the script copy and the 37 outputs.

## Check

```
cd integer_dps && sha256sum -c SHA256SUMS
cd bit_prec && sha256sum -c SHA256SUMS
cd versions_dps60 && sha256sum -c SHA256SUMS
cd old_script_c561d95b && sha256sum -c SHA256SUMS
```

To compare with the committed files, ignore `time` (and `#prec`) lines. Iteration `4`'s fourth column is the step size `|d[0]|+|d[1]|` that the notes call the iteration-4 residual.

## Limits

- **What is established:** `old_script_c561d95b/` gives an exact generating invocation for both committed files, consistent with their recorded provenance.
- **What is not established:** the original command lines, which were never recorded. This shows a sufficient invocation, not a log of what was run.
- **The earlier sections** (`integer_dps/`, `bit_prec/`, `versions_dps60/`) use the current script `736760f2`. They show what that script does, and that it does not produce the committed files at any tested setting.
- **Not tested:** other machines.
- This retention changes no packet file or status.
