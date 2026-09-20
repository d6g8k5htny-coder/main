# SIDE24 V3.4 diagnostic handoff

**Artifact ID:** GP-SIDE24-DATA-001-v1.0  
**Date:** 2026-08-02  
**Purpose:** byte-verified Drive carriers for the three V3.4 diagnostics absent
from Kimi's received corpus.

Each connected-Drive carrier below was fetched and compared byte-for-byte with
the local replay source whose SHA-256 is shown.

| Script | Drive ID | Bytes | SHA-256 | Declared result/scope |
|---|---:|---:|---|---|
| `verify_hybrid_mixed_curvature_and_axis_absorption.py` | `1VPYlprjcWgNmoIMiVaTL5sa6wUirseVr` | 31,533 | `0ba99085cf5b1655deb95859a63f02478057491fe1d394124d853e93bea8c481` | 202/202 PASS; fail-closed; normal/optimized transcript identity |
| `verify_facewise_collar_axis_integration.py` | `1a67DJHnF8NKPxDdhdKvIDpOHi0FWCeQR` | 19,538 | `bb4fdb739636e504d2c2eeafd8d8582dbf1dba12225bca88788927b049d58e5f` | 62/62 PASS in normal and optimized modes |
| `diagnose_side24_generic_collar.py` | `1iiyLpIDkYR0YDiwv9vGRVQ-Rel8uh0sD` | 6,551 | `abc5b8cd3f03c251a8bf47115a9d0097ef03ae62dd65debcb23082ba973176c0` | `ALL_CHECKS_PASS`; numerical diagnostic on its declared grid |

The exact byte comparisons succeeded for all three carriers. These programs
corroborate identities, regression guards, finite grids, and integration
ledgers within their own scope statements. They do not replace the analytic
Fourier-independence, compact-atlas, Gaussian-regression, conditional-moment,
or Kac--Rice arguments in the controlling proof notes.

Replay environment recorded by the V3.4 receipt: Python 3.12.13, SymPy 1.14.0,
mpmath 1.3.0, NumPy 2.4.4, and SciPy 1.17.1.

This handoff closes the missing-carrier logistics note. It does not change the
operator's mathematical reopening conditions in AO48-OPR-045.
