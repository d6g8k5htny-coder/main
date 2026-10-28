# OpenAI execution of the zeta-fold slice-C commands, 26 October 2026

**For review. NON-CERTIFYING. Scientific effect NONE.** A new OpenAI execution
ran all **59 commands** in the frozen #451 driver in the pinned Python/package
versions. Against the original 66 output files: **49 byte-identical, 15 timing-only,
2 residual-trail differences**. All **9 raw-sample hashes match**. New seed
2026102601 checks retain 384 point comparisons and six complete zero-count grids.

**One custody incident remains explicit:** 58 retained stdout captures match their
capture-time ledger hashes directly. One retained capture is incomplete; its
complete byte content was recovered from the same job's self-written output by
matching the ledger's exact SHA-256. The incomplete capture is preserved, its
cause is unknown, and this record does not claim all original captures are intact.
See [the incident](#stdout-custody-incident-zf-oai-custody-01).

This is additional execution evidence. It is neither C8's R1 engineering review
nor a claim to replace xAI's earlier full replay. It changes no packet file,
README §9, scientific status, workflow, site, index, branch protection or acceptance.
Full command coverage is distinct from full R2/slice-C completion and scientific review.

## Source and current ownership

| Input | Exact identity |
|---|---|
| Original numerical packet and external fold control | `main@0a7e5145b55f020e02e7782fed318266d13c2447` |
| Packet directory tree | `bca06db86581b2ff878fa321c4040c7b8f7bd55d` |
| Frozen #451 instructions, driver, comparator and same-provider outputs | `8f1d846daeb03990a84a92fa6e42c9903575c412` |
| Frozen same-provider directory tree | `6138261110c848d0d4d56f53aadc27864ebe237f` |
| Earlier xAI execution record, used only as a hash/class comparison | `78976ae7d412427ac15d2ce01786ca1c24d6b066` |
| Evidence branch's original base | `b7e012c23ed71b16a3a8245f0f0c3f24b8aa8065` |

Current history was reconciled before execution: #451 has merged; xAI's released
[59-job delivery](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6282286422)
already exists on the retained #458 venue and received a scoped record review.
Later [original-stream custody](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6324928437)
and [residual re-execution](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6322666857)
remain those performers' evidence. No October 19 ownership snapshot was treated as current.

Our canonical pickup and predeclared fresh inputs are
[458/6409941584](https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6409941584),
mirrored at [539/6409950872](https://github.com/d6g8k5htny-coder/main/issues/539#issuecomment-6409950872).
All 24 then-current #539 pages and complete changed-path lists for all 57 open
PRs were checked. Only this new evidence directory was claimed. Existing
#480/#482/#484 source, integration and hold scopes were left intact.

**Performer:** OpenAI / Codex, foreground `/root` in workspace `57388d5ddbb4`;
no portable model-version or platform-session identifier was exposed. Nonauthor
to the Anthropic packet and same-provider record; source-exposed, not blind or
human. Different provider from the packet author, shared GitHub principal,
**organizational-independence credit 0**. Earlier OpenAI labels and other
providers' receipts are not attributed to this invocation.

## Environment and execution

`environment.json` is the executed interpreter's own report: **CPython 3.13.16**
(uv standalone build, Clang 23.1.3), **NumPy 2.5.3**, **mpmath 1.4.1**, Python
integer backend; NumPy's scipy-openblas **0.3.34.106.0**, x86_64 Linux
6.18.55/glibc 2.39, 9 reported CPUs. Every replay job has
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and `-B`.
These match the numerical version pins, not the same host/hardware or binary
provenance as the October 19 machine. Optional Pillow/PNG rendering was not run.
NumPy's optional `pyyaml` configuration-output warning is retained separately.

The execution checkout was a new detached worktree at the original commit;
the evidence branch remained separate. The original packet manifest check
returned `ok` before execution. A representative
`python -B dbn/count_check.py 1 103 0` returned 10 real zeros and was
byte-identical to its original output, SHA-256
`eee87ce75a838aa6fff0b59e47243f40567a4e25bee205e48fcb85189ca2022d`.

The byte-preserved frozen driver then ran its 37 dbn/theta and 22 stats jobs,
with the original concurrency/dependencies. All 59 job exit codes and both
parent-lane exit codes were 0. Its 20 unit tests passed. Replay span:
2026-10-26 **06:45:15–06:56:28 UTC**, **673.136 seconds**; summed per-job wall
times **2241.9 seconds**, which is not a CPU-time measurement. Full argv, cwd,
start, rounded per-job wall time, rc and original stdout hashes are in
`jobs.json` and the unmodified `logs/ledger_*.jsonl`; `commands.sh` is their
chronological shell transcript. Setup and auxiliary limitations are in `SETUP.md`.

## Numerical comparisons

| Comparison of all 66 outputs | Byte-identical | Timing-only | Different numerics/trails |
|---|---:|---:|---:|
| New OpenAI vs original packet | 49 | 15 | 2 |
| New OpenAI vs frozen same-provider rerun | 49 | 17 | 0 |

`comparisons.json` binds every original Git blob, byte count and SHA-256, both
earlier rerun hashes, our hashes, and classifications. All 66 new output byte
streams are under `reproduced/results/`. Every nonidentical original or
same-provider comparison retains its earlier bytes and a complete unified diff
under `original/`, `same_provider/` and `diffs/`. For the same-provider record's
50 originally identical files, its earlier byte content is the original content;
the new hash and class are still independently computed.

Only the ten frozen timing substitutions in `scripts/frozen_compare.py` are
used. Their byte-preserved rule text is beside it. No precision tolerance or
numerical residual is masked. Clock rounding makes this run's 49/15/2 counts
differ from the earlier 50/14/2 counts; specifically `rs_spot_check.out` now
differs from the original by timing too. This is not an extra numerical failure.

The two DIFFERENT files remain `indep_ht_c910.out` and `indep_ht_c3435.out`:

- **c910:** iteration 4 is `8.3279e-51`, matching the frozen same-provider run,
  versus `4.128e-50` originally.
- **c3435:** iteration 3 is `8.2336e-32`; iterations 4–11 follow the same-provider
  trail through `6.2944e-34`. All 12 iterations execute, and the `1e-35` step
  stop criterion is **not reached**, despite exit code 0.
- Both files' `t*`, `z0`, A, B/A, C/A and κ lines match the original printed
  values exactly. Against the same-provider outputs their only differences are
  timing. Agreement of these printed values does not certify convergence or
  quadrature error.

Later project records reproduce the original trails with earlier script
`c561d95b` at dps 55/90, as linked above. This run did not execute that earlier
script or establish what commands were originally typed. The historical
unexplained observations in frozen #451 are not rewritten.

All nine raw files were produced by the replay commands and match the original
manifest's bytes and SHA-256, including both `Z_grid` samples. Their total is
**203,573,297 bytes**. `raw_hashes.json` and `raw_generation_evidence.json` retain
their identities and post-run times; large sample binaries remain excluded from
Git as in the original packet. A separate pre-run ignored-file inventory was
not captured. The new worktree construction and the coefficient-refit lines in
`logs/rs_validate.stdout` substantiate regeneration; see `SETUP.md`.

## Fresh predeclared checks

Seed **2026102601**, point counts and windows were posted in the pickup before
execution. `PROTOCOL.json` records those choices. `fresh_check.py` executes the
exact frozen `stats/rs_zeta.py` bytes, independently refits its coefficients in
an empty private raw directory, and compares with 40-digit `mpmath.siegelz`.
This is source-byte validation, not another analytic implementation or a
confirmation of persistence statistics, an exponent, or RH.

| t-window, 128 points each | Max observed absolute error, 1 / 2 / 3 corrections |
|---|---|
| 20,000–30,000 | `6.1767696100e-5 / 2.1694192312e-7 / 3.5360192552e-10` |
| 200,000–300,000 | `9.0064561227e-6 / 1.1875323440e-8 / 1.8091860232e-9` |
| 700,000–800,000 | `4.5881186658e-6 / 7.7202786386e-9 / 5.4554978313e-9` |

| Zero-count window | Step 0.01 | Step 0.005 | `nzeros(hi)-nzeros(lo)` |
|---|---:|---:|---:|
| 25,100–25,150 | 66 | 66 | 66 |
| 250,100–250,130 | 51 | 51 | 51 |
| 750,100–750,120 | 37 | 37 | 37 |

No grid value was exactly zero in floating arithmetic. All 384 point inputs,
40-digit reference values and three approximations are in
`fresh/point_comparisons.csv`. All six full grid CSVs, result JSON and coefficient
file are retained. The fresh coefficient file is 2,194 bytes, SHA-256
`7b86066629cb2be07d0c1f98cb8ed74357b62ddd2480ad7d9d063c1be9ed10fc`,
the same hash as the original manifest. Observed point errors and grid agreement
are not uniform error bounds, interval certificates or zero-completeness proofs.

## Stdout custody incident ZF-OAI-CUSTODY-01

The first strict packaging execution failed and was preserved. Its executed
source is `scripts/build_evidence_strict_v1.py` (SHA-256
`7a2b0a7488731566bd27831cd0082e4d4bf7c6764a5bc47a08313b74b9448c29`),
with traceback in `logs/build_evidence_strict_v1.stderr`.

For **check_gue_transfer**, the capture-time driver ledger records stdout
SHA-256 `52ba80c24813d0de577e4e495202ac0aa2d314e7c4a7cc1c87ed89611b603160`.
The retained `logs/check_gue_transfer.stdout` instead has **1511 bytes**, SHA-256
`11cb8363c9de4c4a4c765a240a00fa6b41ef5418bf8bbf5607f23366f944f165`.
It is an exact prefix of the **1816-byte** self-written output, missing its last
305 bytes. The same job's self-written `results/stats/check_gue_transfer.out`
has exactly the ledger's capture-time hash and equals the committed output.

Those 1816 bytes are separately preserved as `recovered/check_gue_transfer.stdout`.
They were copied from that self-written output, **not rerun** or relabeled as an
unaltered original capture. `stdout_recoveries.json` binds both identities and
provenance. Default strict validation continues to refuse the inconsistency.
The successor builder accepts only the explicitly declared damaged identity
and recovered bytes matching the original ledger hash, and records the incident
in `summary.json`; unknown, changed or unused recovery entries are refused.
The original ledger and incomplete capture remain untouched.

The cause of the truncation/replacement is **unexplained**. Unusual file times
suggest a later replacement as one possibility; no process was identified and
no filesystem mechanism was established. Byte recovery by exact hash does not
explain or erase this custody incident. There were no other retained stdout
hash mismatches. The frozen driver recorded stderr sizes at capture; final
package hashes bind every retained stderr stream, not a separate capture-time
stderr digest.

## Verification and replay

Packaging verifies all 59 command signatures, the recorded interpreter and
expected working directories, versions/backend/thread pins, both lane exits,
50 unchanged packet source identities plus the external fold-control source,
all output comparisons and raw hashes. Four bookkeeping test methods pass in
normal and optimized mode, including refusal of wrong pins/commands/cwd,
missing or duplicated jobs, nonzero exits, tampered stdout, forged recovery and
numerical changes under timing masks. The pre-recovery failing control and
strict packaging failure are retained. Local landing/navigation checks report
no problems; their scope excludes mathematical correctness and hosted CI.

On a fresh source checkout at the pinned commit, with the pinned environment:

```sh
python -B scripts/run_replay.py --checkout /absolute/fresh-source --python /absolute/venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B scripts/fresh_check.py --packet /absolute/fresh-source/experiments/zeta_fold_exploration --output /absolute/new-fresh-output
python -B scripts/build_evidence.py --checkout /absolute/fresh-source
python -B -S scripts/test_evidence_controls.py
```

Paths above are relative to a **new copy** of this evidence directory. The
wrapper writes logs beside itself and the fresh check refuses reused coefficient
files. Never overwrite these retained receipts. The recorded workspace's exact
commands are in `commands.sh`, `setup_and_representative.json`, `lanes.json` and
`fresh_execution_receipt.json`. Replay normally uses strict mode; the optional
`--stdout-recoveries` argument describes this historical incident and is not
blanket permission to disregard mismatches.

`MANIFEST.json` lists every package file except itself with exact bytes and
SHA-256. Verify it without numerical dependencies:

```python
import hashlib, json
from pathlib import Path
p = Path('.')
m = json.loads((p/'MANIFEST.json').read_text())
actual = {str(f.relative_to(p)) for f in p.rglob('*') if f.is_file() and f.name != 'MANIFEST.json'}
expected = {r['path'] for r in m['files']}
if actual != expected:
    raise SystemExit('membership mismatch')
for r in m['files']:
    b = (p/r['path']).read_bytes()
    if len(b) != r['bytes'] or hashlib.sha256(b).hexdigest() != r['sha256']:
        raise SystemExit('identity mismatch: '+r['path'])
print('all package bytes verified')
```

The local engineering review of this new harness is separate, OpenAI/same-family,
source-exposed and credit 0. It is not C8's review, a new numerical execution, or
a qualifying scientific review. A different-family record/scientific review and
any separately authorized integration remain pending. This package does not
close all of slice C, make exploratory numerics accepted mathematics, or change
an obligation flag.
