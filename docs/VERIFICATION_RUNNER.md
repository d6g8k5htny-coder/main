# One execution receipt for the explicit CI checks

`tools/run_checks.py` runs the explicit check commands from
`.github/workflows/ci.yml`, followed by the complete pytest suite. The runner
does not install dependencies. Use the project's Python 3.11 environment with
pytest already installed:

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/python3.11 tools/run_checks.py \
  --output-dir /existing/external-parent/new-run-name \
  --timeout-seconds 1800 --budget-seconds 1800
```

The output directory must not exist, must be outside the repository, and must
have an existing parent. No path component may be a symlink. Every invocation
needs a new directory, including retries after failure. Exit status is `0` only
for PASS, `1` for a recorded failed run, and `2` for invalid CLI/output setup.

**A PASS is an execution record only.** It certifies no theorem, discharges no
premise, moves no scientific status or gate, grants no organizational independence
credit, and authorizes no release or deployment. This is a repository-side
orchestrator, not the governing register, a replacement canonical machine, an
execution-contract enforcement mechanism, or a security sandbox.

## What runs

The parser accepts only the current CI header/setup and pairs of literal
`name:`/`run:` lines. Its explicit required-command policy includes all 24
original checker invocations, the RN certificate replay, the derived-frontier
self-check, the LPW amplitude diagnostic, the static closure-plan check, and
pytest. Commands are matched exactly and must each occur once;
pytest must run last. Omitted checks, empty coverage, duplicate commands,
conditional steps, environment additions, multiline shell, dynamic expressions,
unknown commands, and unfamiliar workflow structure fail before execution.

Adding or changing a CI command requires reviewing the allowlist in
`tools/run_checks.py` as well. The CI file supplies execution order and names;
the allowlist prevents a reduced CI file from masquerading as full verification.
The setup/install steps are recognized but are not executed by this local runner.

Each checker executes as an argument list through `sys.executable`, with the
repository root as its fixed working directory and no shell. The interpreter's
directory is prepended to `PATH`, so existing tests that invoke literal `python`
use the selected environment. Python optimization and any nonempty incoming
`PYTHONOPTIMIZE` are rejected. The runner records the actual Python version,
binary identity, platform and installed pytest version; it does not claim an
environment or dependency lock.

Child execution clears `PYTHONPATH`, `PYTHONHOME`, `PYTEST_ADDOPTS` and
`PYTEST_PLUGINS`. It disables user site packages, bytecode writes, pytest plugin
autoload and pytest's cache provider. An unused external bytecode prefix avoids
reading repository bytecode caches. Pytest's configured `addopts` are overridden
so inherited collection-only or selection options cannot replace the requested
full invocation. The exact effective argument list and environment policy are
recorded. This intentionally controlled local invocation is not a claim that
every detail of a GitHub-hosted environment is identical.

## What the receipt binds

`report.json` records:

- Actual UTC start/end timestamps, elapsed time, the check budget, result and
  failure reasons.
- Before/after Git commit and tree, index digest, exact porcelain status and
  dirtiness, and a path-to-size/mode/SHA-256 inventory of actual input files.
- The workflow's digest, complete required command set and planned check list.
- Each executed step's original CI command, effective arguments and their
  digest, working directory, timestamps, return code, timeout and budget outcome,
  and output-log identity.
- JUnit testcase, executed, skipped, error and failure counts, when pytest's
  report can be parsed, plus the identities of the output artifacts.

Final artifact hashes must agree with the identities captured when each step
and JUnit parse completed. Every executed step log and pytest artifact must
appear exactly once, with no undeclared extra artifact. A later check rewriting
an earlier log therefore fails the run even if every subprocess exits zero.

The input inventory includes tracked, untracked **and ignored** files.
Exclusions are explicit: `.git`, `__pycache__`, `.pytest_cache`, `.pyc` and `.pyo`
cache files, including the `.git` pointer used by a Git worktree. Other ignored
files remain inputs. Source symlinks and non-regular files are rejected. Git
identity or input changes between snapshots prevent PASS. A stable dirty
snapshot may pass; the receipt visibly records that dirtiness. Rerun after
committing when a receipt bound to a clean final commit is needed.

Run against a stable checkout. Before/after snapshots cannot detect a mutation
that is completely reverted between them. Git history and file identities do not
authenticate authors, validate the environment's trustworthiness, or prove the
scientific meaning of a check. Runtime libraries outside the repository are not
fully inventoried.

## Failure, budgets and logs

The first failed check stops dispatch. Remaining checks are explicitly incomplete.
Timeout, budget exhaustion, missing scripts, malformed workflow or JUnit, missing
JUnit output, and zero executed tests all prevent PASS. A pytest invocation that
only skips tests cannot pass. Assertion failures and collection errors retain
their original subprocess status, log and JUnit artifact; they are not converted
into success.

The total budget starts before input capture. Each step receives the smaller of
its configured timeout and the remaining budget. No further step starts after
exhaustion. Final input capture and receipt publication still run, so total wall
time may exceed the check budget by that mandatory finalization work. Timeout or
output overflow terminates the POSIX process group; this runner therefore requires
a POSIX host. Each combined stdout/stderr log is capped at 8 MiB. Overflow fails
the step and preserves the prefix, its digest, and the observed byte count.

The report is published atomically without replacing an existing file. Finished
report, logs and JUnit output are made read-only, as is the run directory. This
prevents accidental reuse; it is not cryptographic signing or protection against
an owner deliberately changing permissions and bytes. Abrupt host termination
may leave an incomplete directory without a report: absence of a complete PASS
receipt is never success. Preserve failed and interrupted directories and use a
new directory for the next run.

The Python API is `run_checks(root, requested_output, timeout_seconds=1800,
budget_seconds=1800)`. The root parameter supports isolated fixture tests and
repository-side orchestration; the standalone CLI always selects its own
repository root. `tests/test_run_checks.py` uses fake checker scripts and tiny
real pytest suites to exercise failure controls without recursively running the
research suite.
