# Zeta fold residual-trail diagnostics: retained outputs (retention only)

**Retention only. Not for merge. No review, no packet change, no status change. Scientific effect NONE.**

This directory keeps the raw outputs behind three diagnostic data notes on main#458, so they no longer exist only in one session's ephemeral container:
- [6303179629](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6303179629): `c910` against integer dps, and a warm-cache run;
- [6304057120](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6304057120): `c3435` against integer dps, and `c910` extended;
- [6321367411](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6321367411): both trails against every working precision in bits within a window.

The notes are the interpretation; this directory is the evidence. The notes' conclusion is unchanged: **both residual-trail discrepancies remain unexplained.** Every output line other than the iteration-4 step and `time` matches the committed `results/dbn/indep_ht_c910.out` and `indep_ht_c3435.out`, so reported values (`t*`, `z0`, A, B/A, C/A, κ) are unaffected.

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

## Check

```
cd integer_dps && sha256sum -c SHA256SUMS
cd bit_prec && sha256sum -c SHA256SUMS
```

To compare with the committed files, ignore `time` (and `#prec`) lines. Iteration `4`'s fourth column is the step size `|d[0]|+|d[1]|` that the notes call the iteration-4 residual.

## Limits

- **Not tested:**
  - other mpmath versions for `c3435` (R1 tested versions only for `c910`, at dps 60);
  - other machines;
  - a precision that changes mid-run;
  - possible unrecorded edits to the script.
- **No cause is established.** This retention closes no part of slice C.
