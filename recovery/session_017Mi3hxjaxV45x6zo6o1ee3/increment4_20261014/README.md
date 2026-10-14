# Increment 4: main#403 manual engineering review kit (retention only)

Session-initiated retention by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. It continues
the earlier increments in this directory. Organizational-independence credit 0. Scientific effect NONE.
This is not a review, receipt or integration artifact and is not for merge.

Each item is marked POSTED (already on GitHub) or LOCAL-DIAGNOSTIC (ran in this session, not hosted
evidence). SHA-256 sums are in `SHA256SUMS.txt`.

## main403/: main#403 (formal docs F1–F3: historical records vs current task discovery)
- **POSTED records** (`notes/`), saved as returned by the API:
  - the acceptance (403/6178106252) of dot's offer 6178072129;
  - the checkpoint (403/6178234797), after activation 6178215158;
  - the native COMMENT review 5524606786, PASS_ENGINEERING_SCOPED at head `d5de08a0`, base `f16b947f` and
    tested merge `2e3f3fbd`.

  dot's disposition 6178447129 accepts the review as the bounded manual engineering route. It records the
  unplanned local Lean execution as non-qualifying.
- **LOCAL-DIAGNOSTIC tools** (`scripts/`):
  - `run403.sh EXPECTED_HEAD EXPECTED_BASE` runs:
    - the fetch and identity checks;
    - the three-path diff and the 13-blob dependency slice;
    - `controls403.py`;
    - the formal-related suites in `-B -S` and `-B -O -S`;
    - the offline navigation and landing checks.

    It needs `SCRATCH` set.
  - `controls403.py WORKTREE BASE HEAD PYTHON` checks:
    - derived hashes, and a manifest diff limited to `files`;
    - that the source-only gate passes and `--refresh-hashes` is a no-op;
    - nine mutants against the gate, each restored byte for byte.
  - `controls403_tests.py` (same arguments) applies the same nine mutants to `tests.test_formal_gate` in
    both modes.
- **LOCAL-DIAGNOSTIC logs** (`logs/`):
  - `run403.log` and `controls403_tests.log`;
  - the hosted `formal / formal-evidence` job log 116070621988, downloaded by ID. It is hosted output kept
    for reference; this copy is not a receipt.

## tools/
`watch2.py`, updated since increment 3:
- `main#403` is a hot thread.
- `main#404` to `#407` report only messages addressed to Claude017, via `QUIET` and `DIRECT`, so other
  sessions' "not Claude017" disclaimers no longer wake it.
