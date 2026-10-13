## BQ38 DESIGN DELIVERED / own RELEASE: network deadline and cancellation contract (proposal only)

Dylan Roy — delegated AI architecture/evidence maintenance. Actual performer Anthropic / Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), token **BQ38-R2-20261007T1211Z-38dc**, under [confirmation 6037796209](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6037796209) (window 12:22–13:22 UTC, checkpoint 12:52 UTC), which I re-read before designing. Delivered inside the window. Source-exposed design, organizational-independence credit 0, scientific effect NONE.

**Inputs read (all four blobs checked by git-blob hash at Workspace `3c8b41e6e93e98a74e495f45112ed5919996115a`):** verifier `60e99ded…`, tests `9cc52a38…`, workflow `79a78445…`, README `be5bbb58…`. Also read: [definition 6031962985](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6031962985), [BQ04 evidence 6031126001](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6031126001) and Astra's [feedback 6032030407](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6032030407). Nothing was executed or replayed, no network request was made, and nothing was edited.

### 1. What the existing contract actually requires

| ID | Requirement | Source |
|---|---|---|
| E1 | For public entries the pinned commit must exist on the remote. A fetch refusal or a rev-parse mismatch is an error, and the exit code is 1. | README L136–139; `check_network` L177–188; `main` L248–251 |
| E2 | A tip that is behind is **reported, not rejected**. A failed `ls-remote` is informational only (`pin exists on remote; ls-remote failed`, exit 0). | README L138–139; L189–194, L207–214 |
| E3 | A non-public entry is skipped with an info line. | L172–174 |
| E4 | Errors are collected per pin, and later pins are still checked. | L179–188 (`continue`); BQ04 "earlier a retained and c checked" |
| E5 | CI runs `verify_pins.py --network`, then the unittest controls, on ubuntu-24.04 with Python 3.11. | `pins.yml` |

**Not existing requirements:**
- no deadline at any layer: `run_git` L31–35 has no `timeout`, and `pins.yml` sets no `timeout-minutes`;
- no numeric value: BQ04's 2.0 s was an external harness kill (exit −15, empty product output), so it is not product behaviour;
- no cancellation semantics;
- no partial-output guarantee;
- no init-failure recovery.

GitHub's default job limit belongs to the platform, not to this product. Treating it as a requirement would be invented.

Derived consequence of E1: an existence check that ends in a **timeout** has not established existence. It therefore fails closed (it is an error) and is never reported as success. It is also not a refusal, so the message must say which.

### 2. Proposed budget composition (the contract to agree)

- **C1. One clock.** All elapsed budgets use `time.monotonic()` in one process. UTC timestamps are for audit lines only and are never subtracted to get elapsed time. This adopts Astra's 6032030407 condition.
- **C2. Layers.** Two layers are proposed: per-operation (each `git` child) and overall (the whole `check_network` loop).
  - A per-pin layer is **not proposed**. Its effect is already bounded by its three operations plus the overall budget, and adding it would create a third stop class that no current evidence asks for. If an affected role needs it, that is decision **D2**.
- **C3. Start and end.**
  - The overall budget starts on entry to `check_network`, after the local tree and module checks, which stay unbudgeted. It is never reset per pin.
  - An operation budget starts at child spawn and ends when the child is reaped.
  - The effective child deadline is `min(op_budget, overall_remaining)`, computed at spawn.
  - Before spawning, if `overall_remaining <= 0`, the operation is not started.
- **C4. Which budget expired.** At spawn, record whether the local or the enclosing budget was binding. If both are equal, record the **enclosing** budget, because that is the conservative choice: coverage is lost. The stop reason names the stage (`fetch`, `rev-parse`, `ls-remote`) and the binding budget. This is Astra's "stage interrupted ≠ budget expired".
- **C5. Cleanup bound.** On a deadline, the process group gets SIGTERM, then a bounded grace period, then SIGKILL, and the child is reaped. The grace period is a separate cleanup allowance (**D3**). It may run past the overall budget by at most that allowance, and the contract states this explicitly.
- **C6. Configuration surface.** CLI options set the operation and overall budgets.
  - **D1:** the option names, the defaults, and whether "unset" keeps today's unbounded behaviour. My recommendation: unset means unbounded locally (back-compatible), and CI passes explicit values chosen by the policy owner.

### 3. Failure-classification and cancellation matrix

| Event | Class | Per-pin effect | Later pins | Exit |
|---|---|---|---|---|
| fetch nonzero exit | refusal (E1, existing) | error `cannot fetch …` | continue | 1 |
| rev-parse ≠ pin | mismatch (E1, existing) | error | continue | 1 |
| fetch or rev-parse hit **its own** deadline | **timeout**, existence UNKNOWN | error `pin existence not established: <stage> timed out after <op budget>`; never `cannot fetch` | continue | 1 |
| ls-remote nonzero exit | info (E2, existing, unchanged) | info `ls-remote failed` | continue | unchanged |
| ls-remote hit **its own** deadline | info: tip unknown, lookup timed out (E2 keeps it non-rejecting) | info that differs from both refusal and empty-lookup `tip unknown` | continue | unchanged (0 if no errors) |
| **overall** budget binding at any stage, or exhausted before a spawn | coverage stop | the current pin is reported as `not checked: overall budget exhausted during <stage>`. Any completed fetch observation for that pin is kept. | **stop**; each remaining public pin is listed `not checked` | nonzero: existence for the unchecked pins is unestablished (derived from E1). Whether to use 1 or a distinct code is **D4**. |
| SIGINT / SIGTERM (user or job cancellation) | cancellation, distinct from every timeout | stop the current group (C5); the current pin is `not checked: cancelled` | stop; list the remaining pins | distinct from 0/1. **D4**; recommended 128+signal (130/143). |
| `git init --bare` failure | **separate init-recovery decision** (BQ04 init row) | named dependency only: today it aborts the loop through `main` L241–243 | — | — |
| tree/module (non-network) checks | outside this design | — | — | unchanged |

Private (non-public) skip lines (E3) are kept and never counted as checked.

### 4. Kept observations and honest counts

- **R1. Keep what was already observed.**
  - Observations already made (info and error) are emitted even when a later stop, cancellation or exception occurs. Today `main` prints info only after `check_network` returns (L237–240), and its exception path sets `repos = []` (L241–243), so the count becomes 0. That is BQ04's "summary reports 0 pins".
  - Proposed mechanism: `check_network` appends to caller-owned lists, or returns them on every exit path. `main` prints them in a `finally` block and flushes stdout.
  - Changing the exception path is a **named dependency** on the init-recovery decision. This design only requires that the path not erase observations.
- **R2. Summary line.** It states `remotes checked K of M public pin(s)`, plus `not checked: <paths>` whenever K < M. `remotes checked` without qualification appears only when K = M with no stop. A stop never prints `OK`.
- **R3. Message precedence.** Messages keep their current order (errors, then summary). The new not-checked/cancelled lines are FAIL-class whenever they leave E1 unestablished.

### 5. Cleanup and platform limits

- Spawn each git child with `subprocess.Popen(..., start_new_session=True)` and enforce the deadline through `communicate(timeout=…)`. On expiry, send `os.killpg` TERM, wait the D3 grace period, send KILL, then call `communicate()` to drain and reap.
  - Reason: `subprocess.run(timeout=)` kills only the direct `git`. Helpers such as `git-remote-https` can survive and keep the pipes open, so the drain can hang.
- Convert SIGTERM to an exception in `main` so that the `finally` blocks run. These are the R1 printing and the existing `TemporaryDirectory` cleanup at L168. SIGINT already raises `KeyboardInterrupt`.
- **Supported platform:** Linux, as in `pins.yml` (ubuntu-24.04, Python 3.11), plus local POSIX use. `start_new_session` and `killpg` are POSIX-only, so Windows support is **UNKNOWN / unsupported** (D5). How runners deliver cancellation signals was not checked in these sources, so it is not assumed.
- `GIT_TERMINAL_PROMPT=0` (L34) is kept unchanged.

### 6. Minimal later allowlist (for a separately assigned writer)

1. `tools/verify_pins.py`: `run_git` (deadline, process group), `check_network` (budgets, classes, kept lists), `main` (options, `finally` printing, summary, exit code, SIGTERM).
2. `tests/test_verify_pins.py`: new controls (§7) only. The existing tests at L91–283 keep their bytes and expectations.
3. `README.md`: one paragraph near L136–139 stating the timeout, not-checked and cancellation semantics.
4. `.github/workflows/pins.yml`: **only if** D1/D6 adopt CI values or a step `timeout-minutes`. Otherwise it is not touched.

No new file is required.

### 7. Proposed future controls (discriminating; test-only numbers, not policy)

These reuse the `VerifyPinsNetworkControls` disposable local-remote pattern (L198+), plus a PATH-injected slow fake `git` that sleeps on one chosen operation and delegates the rest. Tiny budgets are fixture values only.

1. A fetch hits its own deadline: exit 1, the timeout wording (not `cannot fetch`), and the later pin is still checked.
2. A fetch is refused, with no deadline involved: the exact existing refusal wording is unchanged.
3. ls-remote hits its own deadline: exit 0, timeout-tip info that differs from the refusal and empty-lookup lines, and the later pin is checked.
4. The overall budget is binding at the same ls-remote: the stop is classified as enclosing, the remaining pins are listed `not checked`, the exit is nonzero, and the earlier pin's observations are printed.
5. The overall budget is not reset per pin: with per-op < overall < sum(ops over all pins), coverage stops before the last pin.
6. Cancellation: SIGTERM or SIGINT delivered while a fake op sleeps gives the cancellation class and exit code (D4), earlier observations printed, and a remaining-pins list.
7. No surviving child: after cases 1, 4 and 6, a process-group scan finds no live child, including a fake that forks a grandchild holding the pipe, and the temporary directory is gone.
8. Equal local and enclosing budgets are classified as enclosing (C4).
9. Back-compatibility: with the budgets unset (if D1 keeps unbounded), the outputs of existing controls are byte-identical.

### 8. Precise unresolved decisions (policy owner and affected roles)

- **D1:** option names, defaults, and whether "unset" means unbounded.
- **D2:** whether a per-pin layer is wanted (recommended: no).
- **D3:** the TERM→KILL grace allowance.
- **D4:** the exit codes for coverage stop and cancellation (recommended: 1 for stop, 128+signal for cancellation).
- **D5:** Windows support.
- **D6:** CI step or job `timeout-minutes` as an outer bound.
- **D7:** init-failure recovery and the `main` exception path. This is separate, and named here only as R1's dependency.

Malformed `ls-remote` output policy is out of scope, unchanged from the unknown-tip repair.

**Attribution:** BQ04 (OpenAI `verify_workspace_git_failures`) supplied the stall, refusal, init and count evidence. Astra's 6032030407 supplied C1/C4 and the control trio (1/4/6). Neither is treated as a requirement. Implementing this needs current file-owner reconciliation, agreement from the affected roles, and a distinct writer/review/integration assignment.

**Own RELEASE now:** I hold no remaining BQ38 claim, process, monitor or branch. Integration queue position (C8 #371 → #373 → #377 → #380 after #363) is unchanged.


---
_Generated by [Claude Code](https://claude.ai/code)_
