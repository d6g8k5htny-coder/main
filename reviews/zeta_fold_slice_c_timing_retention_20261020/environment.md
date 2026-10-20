# Environment

Retained re-run of the 14 timing-only outputs. Source tree: `git archive` of `main` `157b6a9753fce110bd0769418b10e201720d24ad`, path `experiments/zeta_fold_exploration`. The eleven scripts that were executed have the same git blobs at that commit as at `5cab7663820898aaee2466cbc8662eb04984cde0`.

| Item | Value |
| --- | --- |
| Performer | xAI Grok 4.7 (`grok-4.7-high-fast`), Cursor cloud agent `bc-8084f8ef-3895-401f-833b-33fbb172458c` |
| Interpreter | CPython 3.13.16 (main, Oct 14 2026, 01:04:01) [Clang 23.1.3], `/tmp/r2-venv/bin/python` (uv standalone; not distro `/usr/bin/python3.13`) |
| numpy | 2.5.3 |
| mpmath | 1.4.1, backend `python` (no gmpy2) |
| BLAS | scipy-openblas 0.3.34.106.0; OpenBLAS 0.3.34.106.0 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64 |
| Pins | `OMP_NUM_THREADS=1` `OPENBLAS_NUM_THREADS=1` `MKL_NUM_THREADS=1` `PYTHONDONTWRITEBYTECODE=1`; invocation `python -B` |
| Host | Linux 6.12.94+ SMP PREEMPT_DYNAMIC, glibc 2.39, 4 vCPU Intel Xeon (family 6, model 207), KVM |
| `ZETA_FOLD_RAW` | unset; raw `.npz` files were written under the extracted packet `results/raw/` and are not retained |

Deviation from a distro CPython build: this interpreter is the uv standalone build named above. Kernel and CPU string are this VM's.
