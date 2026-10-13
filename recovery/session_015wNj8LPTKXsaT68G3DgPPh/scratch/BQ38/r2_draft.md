## BQ38 design r2: author revision for BQ50 findings F1–F5 (design text only)

Dylan Roy — delegated AI architecture/evidence maintenance. Actual performer: Anthropic Claude, session `session_015wNj8LPTKXsaT68G3DgPPh` (C8), author of [design 6037851247](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6037851247). Credit 0. Scientific effect NONE.

This answers the nonauthor [BQ50 AMEND 6044164142](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6044164142), the coordinator's [handoff 6044203112](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6044203112) and Astra's [triage 6045211075](https://github.com/d6g8k5htny-coder/main/issues/275#issuecomment-6045211075). ACTIVATION-LINE. It changes no source, test, CI, setting or policy, and nothing was executed beyond reading.

**Source rebind.** The design was cut at Workspace `3c8b41e6`; current `main` is `a152b9d1`.
- `tools/verify_pins.py` `18dafd3b` differs only in `load_pins` (duplicate JSON members). `run_git` (L31–40), `check_network` (now L175–226) and `main` (now L229–265) are unchanged in content.
- Tests `bd993325` add duplicate-member and path controls. `assert_network_output` (L363–378) still compares the whole stdout, including `verify_pins: OK (4 pin(s) … remotes checked)`.
- `pins.yml` `79a78445` is unchanged.
- README `754e92bc` keeps E1/E2 (L152–155). It adds L89–90 (`--network` checks that public pinned commits exist) and L183–185 (a green run means the pins exist on their remotes).

All line numbers below refer to `a152b9d1`.

**The effective design is 6037851247 as amended here.** E1–E5, C1, C3's exclusion of the local checks and the §6 allowlist are unchanged. D1–D7 remain open choices.

| Finding | Paragraphs changed | Change |
|---|---|---|
| F1 cleanup bound | C3, C5, §5 | Absolute deadlines; every phase is either bounded or excluded; a new "cleanup unconfirmed" outcome |
| F2 required versus optional | §3 overall-stop row, R2 | Two outcome dimensions and a partition of M; E1 forces nonzero only when existence is not established |
| F3 output compatibility | R2, §6.2, §7.9 | Legacy mode keeps today's bytes; new lines appear only in budgeted mode or on stop paths; D1 decides |
| F4 init accounting | C2, C4, §3 | Four children per public pin; init and spawn placed in the accounting; budgets only inside `check_network` |
| F5 cancellation boundary | §3 cancellation row, §5, R1 | Flag-only handlers with sliced waits; three termination classes; a record is complete only when its summary line is present |

### F1 → C3, C5, §5

**Absolute deadlines.**
- On entry to `check_network`, take `t0 = monotonic()`. The overall deadline is the fixed instant `t0 + overall`.
- Before each spawn, compute that operation's deadline as the instant `min(now + op_budget, overall_deadline)`.
- Every wait uses `deadline − monotonic()`, recomputed at the wait itself, so time already spent on spawn or setup is never granted again.

**The six phases of one operation and their bounds:**
1. **Pre-spawn check.** If the overall deadline has passed, nothing is spawned.
2. **Spawn** (`Popen`). This step cannot be interrupted. Its duration is charged against the deadline afterwards; it is not bounded by it.
3. **Run.** Bounded by the operation deadline.
4. **Termination on expiry.** `killpg(TERM)`, then wait at most the grace allowance, then `killpg(KILL)`.
5. **Drain and reap.** `communicate(timeout=reap_allowance)`.
6. **Confirmation.** The direct child has been reaped, and `killpg(pgid, 0)` raises `ProcessLookupError`, meaning no member of the group is left.

**If phase 5 or 6 does not finish within its allowance:**
- close the parent's pipe ends;
- record the operation as **cleanup unconfirmed**, naming the stage and the pin;
- start no further operations;
- do not claim a bounded or clean run.

Its exit code falls under D4. I recommend nonzero, because the run cannot attest that it left no process behind. It does not change the E1 accounting of observations already completed.

**Assumptions and exclusions.**
- Containment covers only descendants that stay in the child's process group, that is, ones that do not call `setsid` or `setpgid` themselves.
- A process in uninterruptible sleep can outlive KILL.
- Both cases end as "cleanup unconfirmed", never as a false clean claim.
- The bound covers the network phase only. It excludes interpreter start-up, the pre-network local checks (C3), `TemporaryDirectory` removal and final output I/O.

**Revised C5:** the network phase ends within the overall budget plus the grace and reap allowances of the one in-flight operation, or the run reports cleanup unconfirmed. Both allowances belong to D3, and their numbers are undecided.

**Controls to add:**
- tests that tell apart direct-child exit, pipe end-of-file and a confirmed-empty group;
- a fake git whose descendant holds the pipe, which must yield "cleanup unconfirmed" within the stated bound.

### F2 → §3 overall-stop row, R2

**Two independent outcomes per public pin:**
- **Required (existence):**
  - `established`: the fetch succeeded and the SHA is equal;
  - `refused`: the fetch exited nonzero;
  - `mismatched`: the SHA differs;
  - `not established: <stage> <cause>`, where the cause is one of: own deadline, enclosing deadline, cancelled, not started, cleanup unconfirmed.
- **Optional (tip):** recorded only after `established`.
  - Today's outcomes: equal, behind, default mismatch, tip unknown, lookup failed.
  - New: `lookup timed out`, `lookup interrupted: enclosing budget`, `lookup cancelled`.

**Counts.**
- M is the number of public entries in the loaded manifest, frozen at the start. Private skips are outside M.
- The required outcomes partition M: `established + completed negative (refused + mismatched) + not established = M`. This is Astra's check from 6045211075.
- Optional outcomes are tallied separately.

**Exit consequence of E1 and README L183–185.** Exit 0 claims that every public pin exists, so nonzero is forced exactly when `completed negative + not established > 0`.
- **Stop on the last pin.** Suppose every required check is established and an enclosing stop interrupts the last pin's optional lookup. E1 forces nothing; whether an operational stop alone exits nonzero is D4.
- **Stop on a middle pin.** The same stop on a middle pin keeps that pin `established`. The later pins become `not established: not started`, so nonzero is forced.

**Controls to add:**
- Astra's pair: a final-pin optional stop with complete required coverage, against the same run plus one appended public pin that never starts. The earlier observations must be identical in both.
- One run that also contains a prior required failure and a private skip.

### F3 → R2, §6.2, §7.9

The design no longer promises both a new summary and unchanged bytes everywhere. The proposed boundary below is my recommendation; D1 decides.

- **Legacy mode** (no budget option given):
  - Every path that completes today keeps its exact output and exit code.
  - The existing controls, including L363–378, stay byte-for-byte unchanged.
  - Partial-coverage lines appear only on stop and cancellation paths. Those paths end today without product output (BQ04), so no existing receipt covers them.
- **Budgeted mode** (any budget option given): a new coverage summary (`required: E established, N negative, U not established of M public; optional: …`) replaces the legacy summary and has its own new controls.

The alternative for D1 is one unified new summary for all runs. It needs an explicitly reviewed change to the expectation at L369–370; earlier receipts stay valid for their own commits.

§7.9 now reads: in legacy mode, the existing controls' outputs are byte-identical.

### F4 → C2, C4, §3

**Corrected count.** Each public pin runs **four** Git children inside `check_network`: `init` (L187), then `fetch` (L188), then `rev-parse` (L194), then `ls-remote` (L200). All four go through the same budget wrapper and carry stage names.

**init and spawn in the accounting.**
- Own or enclosing deadline, pre-spawn exhaustion and cancellation each give `not established: init <cause>`.
- A nonzero `init` or a spawn `OSError` keeps today's recovery path (D7, separate and not decided here). That path has one named dependency: completed observations and the loaded denominator M must survive it. It must not return to the `repos = []` count loss at L252–254.

**Budget scope.** `run_git` gains an optional deadline argument that defaults to none. Only `check_network` passes one; the local `check_tree_and_modules` calls stay unbudgeted (C3).

**Controls to add:**
- an init-stage interruption after one completed pin;
- a budget exhausted before spawn, which must start no process.

### F5 → §3 cancellation row, §5, R1

**Signal handling.** Handlers only record the first signal; they never raise. Waits run in short slices that check that record, so cleanup code is never interrupted by an asynchronous exception. The slice length is an implementation constant, not a policy number.

**Three termination classes:**
- **Graceful cancellation:** a handled SIGINT or SIGTERM.
  - No new operation starts.
  - The in-flight group gets the F1 sequence.
  - The completed observations, the not-established list and a `cancelled` summary are printed.
  - The exit code falls under D4; I recommend 128 plus the signal number.
- **Cleanup interruption:** a second signal during finalization. The first cause is kept, the grace period is not restarted (the group goes straight to KILL), and no later pin starts.
- **External hard termination:** SIGKILL, a process-tree kill or out-of-memory. The product gives no guarantee here.

**Completeness rule.**
- Each observation line is flushed as it is produced.
- A run's record is complete only if its final `verify_pins:` summary line is present. Consumers must treat a record without it as incomplete.
- `finally` plus flush is not advertised as durable output after a hard termination.

**CI.** The documented runner sequence ([workflow cancellation](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation)) signals the step's entry process, which is the step shell. Graceful handling inside CI is therefore not assured. It is a separate evidence class that needs authorized runner validation (D6). Direct-signal and repeated-signal controls are local evidence only.

### Still open, and next step

- D1–D7 remain open. F1 and F5 add no new number beyond D3's grace and reap allowances.
- The §6 allowlist is unchanged. Its README paragraph must also state the completeness rule and the split between required and optional outcomes.
- No numeric default, implementation, CI, settings or scientific change is made here.

**Next:** a nonauthor delta readback of this revision against BQ50's F1–F5, by Astra or another nonauthor. After that come the D1–D7 decisions and owner reconciliation, before any writer is assigned. RELEASE-LINE

---
_Generated by [Claude Code](https://claude.ai/code)_
