# Q-9006 bound project-import regression contracts

Test-only phase against unchanged production blob
c549e575f250fde68074f49c2083bb80372b4593. The original 23 helper methods,
18 CLI methods, no-token harness and all production inputs remain unchanged.

Two added CLI methods cover both node_status_refresh and validate_node_status:
an untracked sibling package with __init__.py, and an untracked timestamp-valid
__pycache__ entry with the current interpreter magic, optimization tag, exact
source mtime and exact source size. Nine selected .py/data bytes and actual
Git HEAD stay unchanged. These are project import-resolution controls, within
the trusted host/interpreter bootstrap boundary.

Each adverse wrapper writes an owned marker and executes the actual dependency
source to retain its real API. A separate child harness provides only the
literal SYNTHETIC_IMPORT_PROBE_NOT_A_CREDENTIAL value, denies Python urllib and
socket audit events/calls, and traces the actual tool's run_live call. The trace
records the real bound entry/context and raises before the function body,
observer construction or HTTPS. No live calls, retries or live credit occur.

Before asserting marker absence, each case requires the bound run_live entry,
structured FRESHNESS_CHECK_FAILED refusal, unchanged pre/post nine-file and
HEAD bindings, matching external report, and no freshness/observation credit.
An unrelated import/context/network failure therefore cannot pass the test.

Run the unchanged 23 helper controls and the now 20 CLI methods normally and
with -O. Against c549, expect all 41 original methods per mode to pass, and the
two new methods to produce four marker-presence subtest failures per mode
(two dependencies for each import substitution). Expected RED is these actual
test-body assertions, with zero errors/skips. This has not been executed here.

After parent independently observes that RED, the bounded production fix can
compile/execute captured, verified validator and refresh source bytes directly,
bypassing package and bytecode resolution while preserving the normal helper
import API. No production change is included in this test artifact.
