# `02.3_VALIDATION_RECEIPTS/04_VALIDATION_SCRIPTS_AND_BATTERIES`

Drive folder id `1sEAwQDf8tsvphtS0mVEQxHzabGYxa2XK`. The 2026-09-17 inventory gives this
folder **9 items**, all `text/x-python-script`, and all 9 are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C047R_verification_battery_01.py` | byte-exact | 3,370 |
| `audit_q0_c093_release.py` | byte-exact | 10,121 |
| `build_q0_c093_release.py` | byte-exact | 4,999 |
| `q0_c091_contract_checker.py` | byte-exact | 2,445 |
| `validate_gate_kernel_v2_0.py` | byte-exact | 39,814 |
| `validate_q0_llm_verifier_v2.py` | byte-exact | 4,586 |
| `validate_q0_llm_verifier_v4.py` | byte-exact | 4,429 |
| `validate_q0_llm_verifier_v5.py` | byte-exact | 25,938 |
| `verify_q0_c093_release.py` | byte-exact | 2,894 |

## The status banners, verbatim

These are source files, so their banners are docstrings and headers.

`C047R_verification_battery_01.py` opens with its own identity line:

> # U-ID: U009-appendix | Claude / C047R | 2026-07-20

> # Re-runnable verification battery for DEEP_DIVE_01 (C099/C100). Requires sympy.

`audit_q0_c093_release.py`:

> """Static and dynamic audit for the Q0-C093 closeout release."""

`build_q0_c093_release.py`:

> """Build, deep-audit, fresh-extract, and attest the Q0-C093 release."""

`q0_c091_contract_checker.py`:

> """Machine checks for the C091 proof-gate reduction."""

`validate_gate_kernel_v2_0.py`:

> """Independent validation battery for Gate Kernel 2.0."""

`validate_q0_llm_verifier_v2.py`:

> """Independent stress tests for q0_llm_verifier_v2.py."""

`validate_q0_llm_verifier_v4.py`:

> """Independent validation for q0_llm_verifier_v4.py."""

`validate_q0_llm_verifier_v5.py`:

> """Independent regression and discrimination battery for verifier v5."""

`verify_q0_c093_release.py`:

> """Verify the Q0-C093 manifest in ZIP or extracted-directory mode."""

## What this does not establish

Nine executable files, **none of them run here**. The word *independent* appears in four of
these docstrings; in every case it means independent of the instrument being tested, inside
one author's line — not organisationally independent, and not a second provider.
`C047R-VAL-002` in a sibling folder records that several of these scripts cannot run
standalone at all without co-located modules or data that are not in this lane, and that
`build_q0_c093_release.py` was refused by that session's safety policy for shelling out.
Those are facts about reproducibility, recorded by the source.

`q0_c091_contract_checker.py` reads six JSON reports from a hard-coded `/mnt/data` path that
does not exist here; several of the reports it wants are in the sibling folder
`05_CONTRACT_REPORTS_AND_VALIDATION_OUTPUTS`, under different paths. Do not read the
adjacency of the two folders as a reproduction: nothing in this repository wires them
together, and nothing was executed.
