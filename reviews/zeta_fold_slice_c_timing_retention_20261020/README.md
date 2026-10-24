# Slice C timing-only retention, 20 October 2026

Raw bytes of a fresh re-run of the 14 outputs classed TIMING-ONLY in the replay record on main#458 (comment 6282286422). This directory is the retention. It does not replace packet `results/`, and it does not change README §9.

Source commit: `157b6a9753fce110bd0769418b10e201720d24ad`. Pickup: main#458 comment 6304156331. Performer: xAI Grok 4.7, Cursor cloud agent `bc-8084f8ef-3895-401f-833b-33fbb172458c`. Credit 0. Scientific effect NONE.

## What is here

- `bytes/` holds the 14 output streams, exact bytes.
- `ledger.json` holds each command, return code, start time and wall time.
- `mask_result.json` holds SHA-256, byte counts, the class, and the unmasked unified diff. Every `masked_unified_diff` is empty.
- `environment.md` holds the interpreter and the thread pins.

Stdout-copied files (`validate_H0.out`, `smoke_hflow.out`, `indep_ht_xi.out`, `sample_600.log`, `sample_5000.log`, `rs_spot_check.out`) are the captured stdout. The other eight are the files the scripts wrote. `analyze.py` ran only after `rs_A`, `rs_B`, `gauss`, `cue12` and `cue128`. `rs_spot_check.py` ran only after `rs_B`, and it read this run's `results/raw/zeta_B.npz`.

## Commands

Working directory: the extracted packet. Each return code was 0 and each stderr was empty. Summed wall time 775.346 s. Span 293.971 s, from 2026-10-20T23:35:55Z.

```
python -B dbn/validate_H0.py
python -B dbn/smoke_hflow.py
python -B dbn/indep_ht.py 60 xi
python -B dbn/check_dbn_basic.py
python -B stats/rs_zeta.py validate
python -B stats/rs_zeta.py A 1e4 1e5
python -B stats/rs_zeta.py B 1e5 1e6
python -B stats/gauss_proc.py 500 4
python -B stats/cue.py 12 86016 4
python -B stats/cue.py 128 8064 4
python -B stats/sample_Z.py 600 900 0.02
python -B stats/sample_Z.py 5000 5200 0.015
python -B stats/analyze.py
python -B stats/rs_spot_check.py
```

`indep_ht.py 60 c910` and `indep_ht.py 60 c3435` were not run.

## Mask

Rules: `reviews/zeta_fold_slice_c_rerun_20261019/comparison/timing_mask_rules.txt`, applied as in `reviews/zeta_fold_slice_c_rerun_20261019/scripts/compare.py` (the three anchored patterns use `re.M`). Committed bytes are `git show 157b6a9753fce110bd0769418b10e201720d24ad:experiments/zeta_fold_exploration/<path>`.

All 14 are TIMING-ONLY. Masked text is identical. There is no residual difference after the mask. The unmasked diffs are only the timing tokens, and they are quoted in `mask_result.json`.
