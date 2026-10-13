# main#350 read preparation (session_01S16677s1SGeAJmN3QMz4ip, 13 Oct 02:13Z)

Head 4ccbaf628b9d5431982cbc3c210a86546c1534ae (PR main#350), detached worktree, CPython 3.12.3 (the driver refuses other runtimes with NOT_RUN "Python 3.12 runtime unavailable").

- nested-python3.12.log / nested-python3.12-O.log: `python3.12 -B -E [-O] -S -m unittest discover -s experiments/universality/two_parameter_cusp_population -p 'test_*.py' -v` -> 105 tests OK each.
- nested-python3.11*.log: the same under CPython 3.11.17 -> 97 OK, 8 runner-lifecycle tests fail with status NOT_RUN (runtime gate), kept as the negative control for the runtime requirement.
- replay-stdout.txt / replay-stderr.txt: the declared full command `python3.12 -B -S experiments/universality/two_parameter_cusp_population/run_global_h0_selector.py` in a fresh copy with results/global_selector_1 removed: exit 0, wall 12.31 s, status CERTIFIED, 18/18 fixtures in both executions.
- RESULTS.replayed.md: byte-identical to the PR's RESULTS.md (sha256 79333482...).
- CERTIFICATE.replayed.sha256: the replayed CERTIFICATE.json (4,431,198 B) differs from the PR's (4,431,203 B, sha256 c5320020...) in exactly 23 leaves, all timing: 20 phase_times[].seconds, work.monotonic_start, work.monotonic_deadline, elapsed_before_serialization_seconds. Every fixture value, certificate and status is identical.

Preparation only; not a verdict. Not hosted evidence.
