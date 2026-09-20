# Reproduction guide — PKG-01

## Environment
Python 3.x; no network required for the assembly gate.

## Gate (both modes must match)
From the package copies of the D1 assembly materials:
```
python d1_falsify_v3.py
python -O d1_falsify_v3.py
```
Compare transcripts to `v3_t_normal.txt` / `v3_t_opt.txt` (byte-identical expected).

## Hash checks
- Zip intake SHA-256: a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b
- D1_ASSEMBLY_v2_2.md whole-file: 7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174
- Frozen body (between BEGIN_FROZEN_BODY / END_FROZEN_BODY): 490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6

## Scope
Reproduction validates the assembly gate and hash pin. It does not discharge the five open validity premises of Theorem (2).
