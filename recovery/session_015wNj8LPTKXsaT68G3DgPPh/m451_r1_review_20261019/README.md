# Evidence for C8's R1 review of main#451

This folder holds the checks behind `C8-M451-R1-ENGINEERING-REVIEW-20261019`, the nonauthor engineering review of main#451 at head `8f1d846d` (the same-provider slice C rerun record of `experiments/zeta_fold_exploration`). The performer is Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8): Dylan Roy — delegated AI review. The pickup is [main#451/6278955485](https://github.com/d6g8k5htny-coder/main/pull/451#issuecomment-6278955485), and the verdict is posted in the same thread. Credit 0. Scientific effect NONE.

| File | What it is |
|---|---|
| `classify.py` → `classify_log.txt` | C8's own recomputation of the 50/14/2 classification for check (d). It applies the ten substitutions transcribed from `comparison/timing_mask_rules.txt`. It also checks every committed SHA-256 and git blob in `comparison/hashes.json` against `main` `0a7e5145`, the `original/` copies against the committed bytes, and the `reproduced/` copies against the rerun hashes. Standard library only. Usage: `python3 -I -B classify.py <record dir> <packet dir at 0a7e5145>`. |
| `count_1_103_0.replay.out` | stdout of `dbn/count_check.py 1 103 0`, replayed for check (c) in a clean `git archive` of the packet at `0a7e5145`. The environment was Python 3.13.12 (the record pins 3.13.16), mpmath 1.4.1 with the python backend, numpy 2.5.3, and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. SHA-256 `eee87ce75a838aa6fff0b59e47243f40567a4e25bee205e48fcb85189ca2022d`, byte-identical to the committed `results/dbn/count_1_103_0.out` and to the ledger's hash. Wall time 3.8 s. |
| `unittest_log.txt` | `python -B -m unittest test_exact test_numeric` in the same environment: 20 tests OK. |
