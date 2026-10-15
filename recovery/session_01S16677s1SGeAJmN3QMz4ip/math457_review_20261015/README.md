# Math-#457 engineering review evidence (15 October 2026)

Session `session_01S16677s1SGeAJmN3QMz4ip` (Anthropic Claude, "Claude S"). Scientific effect NONE; organizational-independence credit 0. The GitHub comment is canonical: Math-#457/6196423607 (verdict PASS_SCOPED + release), under activation 6196132595.

- `mutants.py`: single-edit mutant harness (14 mutants x 3 workflows) run against a minimal copy (`.github/`, `tests/`, `tools/`) of head `21b2ea5b2e0bbe20425fe0c88011166ee025e851` with CPython 3.13.16 `-B -S -m unittest tests.test_full_suite_routing tests.test_c6_factorial_job_budget`.
- `all.sorted.jsonl`: per-run results (return code, summary, failing tests); all 42 runs fail, i.e. every mutant is killed.
- `orig.sh` / `cand.sh`: the exact manual `run:` blocks of `c6-palm-route.yml` at the parent `6e502c1f` and the head, used for the native empty/pass/fail control under CPython 3.11.17 and 3.13.16 (exit codes recorded in the verdict).
