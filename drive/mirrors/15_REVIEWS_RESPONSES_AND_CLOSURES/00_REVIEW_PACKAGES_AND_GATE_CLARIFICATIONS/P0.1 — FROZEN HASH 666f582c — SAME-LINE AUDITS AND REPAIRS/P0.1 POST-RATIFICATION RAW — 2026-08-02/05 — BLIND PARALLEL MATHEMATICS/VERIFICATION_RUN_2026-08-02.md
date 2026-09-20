# Blind axial checks — run record

**Date:** 2026-08-02  
**Environment:** Python 3.12.13, SymPy 1.14.0, mpmath 1.3.0  
**Source boundary:** local non-Kimi project carriers only; no Drive query,
web query, or Kimi artifact was used

## Results

| Verifier | Checks | Normal | `python -O` | Transcript |
|---|---:|---:|---:|---:|
| `verify_axial_d7_mechanism.py` | 24 | PASS | PASS | byte-identical |
| `verify_axial_suppression_cone.py` | 18 | PASS | PASS | byte-identical |

Both verifiers use explicit `ck()` failures and no Python `assert`
statements.  Each was also run with its forced-failure environment variable:

- `D7_FORCE_FAILURE=1`: exit status 1;
- `SUPPRESSION_FORCE_FAILURE=1`: exit status 1.

The committed transcript for each verifier is its normal/optimized common
output.  Neither script asserts the complete RP-C/RP-S integration.

## Replay

```text
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python verify_axial_d7_mechanism.py
.venv/bin/python -O verify_axial_d7_mechanism.py
.venv/bin/python verify_axial_suppression_cone.py
.venv/bin/python -O verify_axial_suppression_cone.py
```

