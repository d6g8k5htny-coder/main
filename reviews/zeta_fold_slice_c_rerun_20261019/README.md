# Slice C rerun of `experiments/zeta_fold_exploration` — same-provider numerical reproduction record

**For review. Scientific effect: NONE.** This record re-executes the numerics of the zeta fold
exploration packet and compares every output byte-for-byte with what is committed. It is not a
theorem, an acceptance, or a read of the packet's mathematics. It does **not** satisfy the
packet's open slice C as stated there, because slice C asks for a reproduction by a *different
provider* and the performer here is the same provider as the packet's author (Anthropic Claude);
organizational-independence credit is 0. The different-provider requirement in the packet's
README §9 remains open.

Object ID `ZETA-FOLD-SLICE-C-RERUN-20261019`. Every file in this directory is listed with its
byte count and SHA-256 in [`MANIFEST.json`](MANIFEST.json).

## 1. Source identity

- Repository `d6g8k5htny-coder/main`, default-branch head on 2026-10-19:
  `0a7e5145b55f020e02e7782fed318266d13c2447`. The packet directory was last changed by `4675c5d3`
  (2026-10-12) and landed through main#373 (merge `3e589ae0`).
- Packet files at that commit (blob · bytes · SHA-256):
  - `README.md` `1fcb74c3…` · 99,120 · `d9416dc35e198de3a1c2799b6c5fd23f5e93a16d054b35c403604a3a2728913c`
  - `MANIFEST.json` `06e6e7ff…` · 19,750 · `99441cc43dae0ace1ea5a080880e93e2495c61665ca74bfdd42dfd452812700e`
  - `requirements.txt` `1f9f4d21…` · 27 · `f5eed4fa0ff9d50171b9857728e41177edf2e0f8195a51f285746c54ade5ecda`
  - `dbn/indep_ht.py` `736760f2…`; `results/dbn/indep_ht_c910.out` `ad0f7739…`;
    `results/dbn/indep_ht_c3435.out` `4d685efe…`
- `python3 -B make_manifest.py --check` → `ok` on the fresh detached worktree before any run.
- The open item being addressed: README §9 "C. The numerics, re-run independently, with zero
  counts checked against `nzeros` and a different seed or an independent Riemann–Siegel check.
  **NOT_PERFORMED; still open**" (offered to GROK-H01 in main#373 comment 6118393130 and main#307
  comment 6123501853; not picked up). `docs/REPRODUCE.md` and `docs/RESEARCH_INDEX.md` item 7
  record the same.

## 2. Performer, exposure, authority

Anthropic Claude — Claude Code cloud session `session_01DKKe26v3CSgdYch8kiJLt9`, model
`claude-fable-5-1` — under the owner's standing delegation (OP-AUTONOMY-20260923-v2.1) and the
owner's explicit instruction in that session to prepare this package. Nonauthor at the session
level (this session did not write the packet), **same provider as the author**, source-exposed
and not blind: the packet README, scripts and committed outputs were read before and during the
run. The run itself preceded any board pickup because the session had no GitHub API credential
at the time; the coordination steps taken once a credential route became available are in
[`HANDOFF.md`](HANDOFF.md). No change was made to any packet file; work was done in a detached
`git worktree` of `0a7e5145`.

## 3. Environment

Exactly the pinned versions ([`environment.json`](environment.json)): Python 3.13.16 (fresh venv
from `/usr/bin/python3.13`), `numpy==2.5.3`, `mpmath==1.4.1` (pure-Python integer backend; no
`gmpy2`), numpy BLAS `scipy-openblas 0.3.34.106.0`; Linux 6.18.44 x86_64, glibc 2.39, 4 vCPU
(Intel Xeon 2.10 GHz), 15 GiB. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for
every job, as README §8 requires for byte-identical samples. Pillow not installed, so no PNG
previews (the committed SVGs are produced). The packet README states the author's runs used
Python 3.13.16.

## 4. Method

**4.1 Rerun.** Every command in README §8 was executed verbatim; the exact argv, working
directory, start time, wall time, exit code and stdout SHA-256 of all 59 jobs are in
[`comparison/ledger.json`](comparison/ledger.json) and, as a shell transcript,
[`scripts/commands.sh`](scripts/commands.sh). Orchestration: [`scripts/driver.py`](scripts/driver.py)
(dbn/theta jobs three at a time; stats sample generators one at a time, each with its own
4-process pool as the scripts specify). Each job's stdout/stderr is in [`logs/`](logs/). Where a
script's docstring says "stdout saved as X", the captured stdout was copied to X unless the
script rewrote X itself (mtime check; mode recorded per job). `track_low10_dt_sensitivity.out`
was rebuilt from the five `dt<DT>` runs as its own header prescribes (blocks in DT order, EVENT
lines cut to 160 characters).

**4.2 Comparison rules.** [`scripts/compare.py`](scripts/compare.py) runs `git status`/`git diff`
of the worktree against `0a7e5145` and classifies each committed output as IDENTICAL (same
bytes), TIMING-ONLY (same after the regex substitutions listed in
[`comparison/timing_mask_rules.txt`](comparison/timing_mask_rules.txt): `setup Ns`, `zeta zeros in
Ns`, `time N`, `scan time N`, `"seconds*": N`, `"sec_per_1e5_evals_at_9e5": N`, `(Ns)`, `samples
in Ns`, `init N N …`), or DIFFERENT (unified diff kept). Nothing else is masked. Regenerated raw
`.npz` samples are hashed against `MANIFEST.json` `raw_files`. The committed and rerun SHA-256 of
all 66 output files are in [`comparison/hashes.json`](comparison/hashes.json); for the 16 files
whose bytes differ, both byte streams are kept side by side in [`original/`](original/) and
[`reproduced/`](reproduced/).

**4.3 Reviewer's independent check** ([`scripts/indep_check.py`](scripts/indep_check.py), not a
packet script). RNG seed **8675309** (the packet's validation uses 20261012): (a) refit the
Riemann–Siegel Chebyshev coefficients into a private directory and hash them; (b) compare
`stats/rs_zeta.py:Z` with `mpmath.siegelz` at 30 digits at 200 uniform random points on each of
[1e4, 1e5] and [1e5, 1e6], with 1, 2 and 3 correction terms; (c) count sign changes of `Z` on
fine grids over four windows, each at two grid steps, against the exact
`mpmath.nzeros(T1) − nzeros(T0)`; (d) for the `validate_H0.py` window [1, 250], compute
`nzeros(125)` and the bracketing zeros 2γ₄₁, 2γ₄₂. Output: [`reproduced/indep_check.json`](reproduced/indep_check.json).

**4.4 Experiments on the two non-reproduced files** ([`experiments/`](experiments/), index in
`experiments/INDEX.json`): the committed `dbn/indep_ht.py` rerun under `gmpy2` 2.3.2
(`BACKEND gmpy`), under mpmath 1.3.0 and 1.4.0, under Python 3.12.3 and 3.11.17 (mpmath 1.4.1),
and the two earlier versions of the script preserved in branch `claude/recovery-S-20261013` @
`5d298b9e` (`review-math/indep_ht.py`, blob `c561d95b`; `apply-baseline/…/indep_ht.py`, blob
`c152559b`) run with the pinned environment. Each run is compared, modulo the `time` line, with
the committed file and with the pinned rerun.

## 5. Numerical reproduction — results

**5.1 Run.** 59 jobs (37 dbn/theta, 22 stats), every exit code 0, no stderr except the unittest
report. 17.9 min wall-clock (19:52:02Z–20:09:55Z, both lanes concurrent on 4 cores), 61.2
CPU-minutes. `python3 -B -m unittest -v test_exact test_numeric`: 20 tests OK (14 exact, 6 numeric).

**5.2 Committed outputs (66 files under `results/`).**

| class | count | files |
|---|---|---|
| byte-identical | **50** | all collision/fold CSV and JSON files, `fold_analyze.out`, the six `count_*.out`, `kappa_*.out`, `refine_check.out`, `zero_velocity.out`, `track_z270.out`, `track_low10.out`, both committed `track_log_*.txt`, `collision_events.csv`, `lobe_identity.out`, `check_collision.out`, `check_collision_persistence.out`, `check_theta.out`, `project_theta_checks.out`, `fold_controls_rerun.out`, `check_gue_transfer.out`, both `Z_grid_*_persistence.out`, `matched_tail.json`, `block_sensitivity.json`, `zeta_quad_check.json`, the ECDF and slope CSVs, both SVGs, the rebuilt `track_low10_dt_sensitivity.out` |
| timing-only | **14** | `check_dbn_basic.out`, `indep_collision.out`, `indep_ht_xi.out`, `smoke_hflow.out`, `validate_H0.out`, `cue12_info.json`, `cue128_info.json`, `gauss_info.json`, `rs_validation.json`, `sample_600.log`, `sample_5000.log`, `summary.json`, `zeta_A_info.json`, `zeta_B_info.json` |
| different (residual trail) | **2** | `indep_ht_c910.out`, `indep_ht_c3435.out` — §5.5 |

**5.3 Regenerated raw samples.** **9 of 9** byte-identical to the `MANIFEST.json` hashes:
`zeta_A.npz`, `zeta_B.npz`, `rs_cheb_coeffs.npz`, `gauss.npz`, `cue12.npz`, `cue128.npz`,
`ecdf_samples.npz`, `Z_grid_600_900.npz`, `Z_grid_5000_5200.npz`. README §8 records the last two
as not regenerated by the author; this is their first independent regeneration.

**5.4 Independent check (seed 8675309).**

| check | result |
|---|---|
| RS Chebyshev coefficient refit | 2,194 bytes, SHA-256 `7b86066629cb2be07d0c1f98cb8ed74357b62ddd2480ad7d9d063c1be9ed10fc` = manifest entry |
| `Z` vs `mpmath.siegelz`, [1e4, 1e5], 200 points, max abs error (1/2/3 corrections) | 7.33e-05 / 4.70e-07 / 5.52e-10 (packet, 60 points, seed 20261012: 5.99e-05 / 3.49e-07 / 3.45e-10) |
| `Z` vs `mpmath.siegelz`, [1e5, 1e6], 200 points | 1.57e-05 / 2.54e-08 / 1.48e-08 (packet: 6.79e-06 / 2.66e-08 / 6.48e-09); max |Z| 29.4 |
| zero count [10000, 12000], steps 0.01 and 0.005 | 2377 = 2377; `nzeros` difference 2377 |
| zero count [99000, 101000], steps 0.005 and 0.0025 | 3080 = 3080; 3080 |
| zero count [500000, 500500], steps 0.005 and 0.0025 | 897 = 897; 897 |
| zero count [990000, 991000], steps 0.004 and 0.002 | 1906 = 1906; 1906 |
| H₀ window [1, 250] | `nzeros(125)` = 41 = the scan's count; 2γ₄₁ = 248.514 ≤ 250 < 2γ₄₂ = 255.033 |

**5.5 The two residual-trail differences.** `dbn/indep_ht.py` (the third, tanh-sinh H_t
implementation) prints the Newton step |Δt|+|Δz| at each iteration and stops once it is below
10⁻³⁵. Compared with the committed files:

- (9/10): iterations 0–3 byte-identical (step 2.0282e-25 at iteration 3); iteration 4 shows
  8.3279e-51 here vs 4.128e-50 committed; both stop there. 580 vs 579 bytes.
- (34/35): iterations 0–2 byte-identical; iteration 3 shows 8.2336e-32 here vs 8.2634e-32
  committed; the committed run then reached 1.2375e-63 at iteration 4 and stopped, while here the
  step stayed between 2.5e-34 and 1.7e-33 from iteration 4 through 11 and all 12 iterations ran.
  1,143 vs 591 bytes; `time` 604 s vs 413 s.
- In both files the lines `t*`, `z0`, `A`, `B/A`, `C/A`, `kappa_r`, `kappa_l` are byte-identical
  to the committed ones (30 printed digits for `t*`/`z0`, 12 for `A`, `B/A`, `C/A`, 10 for κ).

Experiments (§4.4): every variant — gmpy backend, mpmath 1.3.0, mpmath 1.4.0, Python 3.12.3,
Python 3.11.17, and both earlier script versions — reproduces the **pinned rerun's** trail
modulo the `time` line, never the committed trail (`experiments/INDEX.json`). At dps 60 the three
script versions differ only in the module docstring and in where `dps` is assigned.

## 6. Limitations — read separately from §5

1. **Provider.** Same provider as the author; organizational-independence credit 0. This record
   does not resolve slice C's different-provider requirement, and the packet README's sentence
   "no read from a different provider has reproduced the numerics" remains true.
2. **What byte-identity does and does not cover.** 50 outputs and 9 raw samples reproduced
   byte-for-byte; 14 outputs differ only in timing tokens under the stated mask; 2 outputs were
   **not** reproduced byte-for-byte. Agreement of the collision points and κ coefficients to the
   printed 30/12/10 digits in those 2 files is evidence about the converged values only; it does
   not establish byte reproduction of those files.
3. **The residual-trail discrepancy is unexplained.** Tested and excluded as causes: mpmath integer
   backend, mpmath version (1.3.0, 1.4.0, 1.4.1), Python version (3.11, 3.12, 3.13), and script
   version (the two preserved earlier copies). The committed command reproduces one stable trail
   in all of these and never the committed one. That the committed trails were produced with a
   different `dps` argument, or by a code variant that was not preserved, are **hypotheses, not
   established causes**; neither was tested. Investigation stopped here because no claimed result
   or acceptance criterion depends on the post-convergence Newton steps: the packet README §5.1
   claims only that the tanh-sinh route "confirms both collisions to the 30 printed digits", which
   is reproduced, and README §8's own list of outputs re-run byte-identically after the scripts
   were moved does not include these two files (every recovered copy of them is the same blob).
   A packet maintainer may wish to record the environment of those two outputs or regenerate
   them; that is not claimed here.
4. **Scope of review.** Not blind; source-exposed. No part of slices A, B or D (mathematics,
   citations, wording) was re-read. The packet is exploratory, not interval-certified, scientific
   effect NONE, and this record changes none of that.
5. **Comparison mechanics.** Byte-identity of regenerated samples depends on single-threaded
   BLAS (README §8); timing fields differ by construction and are the only masked tokens. The
   five `track_log_dt<DT>.txt` event logs produced by the dt-sensitivity runs are not committed
   in the packet (its header says to delete them); they are preserved here under
   `reproduced/uncommitted/` for completeness only.

## 7. Package contents

```
README.md                 this record            HANDOFF.md            coordination record and next steps
MANIFEST.json             bytes + SHA-256 of every file here; bound source blobs
environment.json          pinned run environment, host, extra test environments
comparison/               COMPARE.md, compare.json, hashes.json (all 66 outputs), ledger.json, timing_mask_rules.txt
original/                 committed bytes of the 16 outputs whose rerun bytes differ
reproduced/               rerun bytes of those 16; uncommitted/ dt event logs; indep_check.json
experiments/              indep_ht runs under other backends/versions/script copies; INDEX.json; *_env.txt
logs/                     stdout/stderr of every job, lane logs, ledgers (JSONL)
scripts/                  driver.py, compare.py, indep_check.py, make_identity.py, build_package.py, commands.sh
```

To replay: clone `main` at `0a7e5145`, create a Python 3.13 venv with `numpy==2.5.3
mpmath==1.4.1`, export `ZF_PKT`, `ZF_PY`, `ZF_LOG`, run `ZF_LANE=dbn scripts/driver.py` and
`ZF_LANE=stats scripts/driver.py` (or the lines of `scripts/commands.sh`), then
`scripts/compare.py`; `scripts/indep_check.py` needs `ZF_INDEP_RAW` for its private coefficient
directory. A different seed, window set or grid in `indep_check.py` is a stronger check than
repeating these values.

## 8. Follow-ups exposed by this record (not claimed)

- A **different-provider** numerical rerun is still the open slice C; this package gives such a
  reviewer the exact environment, commands and hashes to replay or to vary.
- Packet maintainers: record or regenerate the environment behind `indep_ht_c910.out` and
  `indep_ht_c3435.out`; optionally add this record to README §9 as "same-provider rerun,
  2026-10-19, credit 0" while keeping slice C open.
