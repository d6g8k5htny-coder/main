# Environment

Performer: Cursor cloud agent `bc-8084f8ef-3895-401f-833b-33fbb172458c`.
Provider: xAI. Model: Grok 4.7 (run model `grok-4.7-high-fast`). Not an Anthropic model.
Pickup: https://github.com/d6g8k5htny-coder/main/pull/458#issuecomment-6282014067

## Source

- Repository commit: `5cab7663820898aaee2466cbc8662eb04984cde0` (this PR's base).
- `experiments/zeta_fold_exploration` tree: `bca06db86581b2ff878fa321c4040c7b8f7bd55d`.
- Separate clone: `/tmp/r2`, detached at that commit. The packet scripts overwrote result files only in that clone.
- Runner: `reviews/zeta_fold_slice_c_rerun_20261019/scripts/driver.py` from that commit, both lanes at once (`ZF_LANE=dbn` and `ZF_LANE=stats`). `ZF_PKT=/tmp/r2/experiments/zeta_fold_exploration`, `ZF_PY=/tmp/r2-venv/bin/python`, `ZF_LOG=/tmp/r2-logs`.

## Interpreter

- Python 3.13.16, `main, Oct 14 2026, 01:04:01) [Clang 23.1.3 ]`.
- Venv `/tmp/r2-venv`, created from the uv standalone `cpython-3.13.16-linux-x86_64-gnu`.
- `numpy==2.5.3`.
- `mpmath==1.4.1`, backend `python`. `gmpy2` is not installed. Pillow is not installed.
- NumPy BLAS and LAPACK: `scipy-openblas` `0.3.34.106.0`.
- OpenBLAS configuration string: `OpenBLAS 0.3.34.106.0  USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`.

## Host

- Linux `6.12.94+` x86_64, glibc 2.39 (Ubuntu GLIBC 2.39-0ubuntu8.9).
- 4 vCPU, `Intel(R) Xeon(R) Processor`, `cpu MHz` 2400.000.
- 15 GiB RAM.

## Thread pins

Set by `driver.py` for every job, and present in the process environment:

`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`

## Deviations from the pinned run in `reviews/zeta_fold_slice_c_rerun_20261019/environment.json`

1. This image has no `/usr/bin/python3.13`. The venv is the uv standalone CPython 3.13.16 built with Clang 23.1.3, not a venv of a distro `/usr/bin/python3.13`. The patch version is 3.13.16.
2. Kernel `6.12.94+`. The pinned host kernel is `6.18.44`.
3. The CPU model string here is `Intel(R) Xeon(R) Processor` at 2400 MHz. The pinned record says `Intel(R) Xeon(R) Processor @ 2.10GHz`.

Package pins, the mpmath backend, and the OpenBLAS version string match the pinned run. The jobs were run with these deviations recorded.
