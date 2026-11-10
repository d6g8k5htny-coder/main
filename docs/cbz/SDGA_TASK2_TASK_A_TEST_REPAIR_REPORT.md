# SDGA Task A test-only review repair

Date: 2026-11-10 UTC. Scientific effect NONE; qualification and classification remain BLOCKED; execution_authenticated remains false.

This source-only repair addresses the two P2 findings before publication. It supplies no production Component.lean or sdga_component.py. No native, Python, Lean, workflow, semantic-adverse, mutation, mathematical, or production-build result is claimed.

## Exact replacement scope

- tests/fixtures/sdga_component/Controls.lean: blob b8ec51fbcee8516fcfeb89a637fa08097d918602 replaces 0a7c3743f7d0aa7a815d8ade2a81764007f9f70d.
- tests/sdga_component_contract.py: blob d7f72edb1b7dac6458c568e9411edf21f2703476 replaces 9c44d907c6b1d37b198c34f4059e53f96b353c21.
- This short report is additive. No workflow change is required: result.json and commands.json already print and retain the separate cleanup evidence.

Unchanged inputs: Safe.lean 814c0f3169b4e967f5b4940d205a479effebeed1; workflow 613a7840949eb5acfa68f3318e7ebdc5a7511a48; lakefile 6f8645db9422a25732aa7848b1e942616f909656; toolchain ba8ebf2dbaf6a668cd2a0e086186d6d569b69ff5. Root-approved design 3487f54476b5e6a283839902797bc05aae2ce623 and plan 9eefcb144247aa3f358b735956e580777e2b8945 remain the authority.

## Independent CLI reconciliation

C02 retains every original assertion and emits additional oracle rows from each actual raw final ConstantInfo. Constructor matching determines kind; actual environment forward/reverse module indices determine owner; core Meta.isProp determines proposition status; raw exported membership and isPrivateName determine privacy; core CI flags determine unsafe/partial. These oracle rows do not copy SDGA.observeConstant.

The launcher compares actual inventory declaration name/module/kind/is_prop/is_private/unsafe/partial against those independent rows. It retains the complete final-name census and checks raw ordered OLean tables, measured request hashes, extra-name union and module/scope identity. Task 1 parsing still enforces the fixed JSONL structure, terminal counts, exact declaration population, private-part hash binding and axiom union. Bootstrap and C02 require exactly three selected targets, one proposition target, two data targets and no fixture findings. The permanent BLOCKED classifier state is still asserted; it is no longer the sole inventory acceptance condition. Diagnostic type_display remains governed by the unchanged Task 1 nonempty-string protocol; Task B strict axiom semantics are not claimed here.

## Owned process cleanup only

Every command uses Popen with a new session/process group and original stdout/stderr files. The launcher enables a Linux child subreaper, waits for the direct leader with a deadline, then probes and cleans its owned group on timeout or after normal leader exit. Cleanup sends bounded TERM then KILL as needed, lets Popen reap the leader first, reaps adopted descendants only with waitpid(-owned_pgid, WNOHANG), and requires the owned group to be absent. Every unexpected timeout raises after preserving the actual exit, streams and cleanup evidence, even if TERM produces exit zero. Only the exact fixed cleanup preflight label/arguments/cwd/three-second timeout may use the expected-timeout exception. The scoped reap loop checks its deadline after each result. An unconfirmed cleanup permanently blocks further scheduling. No PID 1 reaping, waitpid(-1), name-based kill, or unrelated process-group signal is used.

A separately labelled cleanup-only preflight runs fixed Python parent/descendant sources. Both ignore TERM; a readiness handshake records their common owned group; an actual timeout must exercise TERM and KILL, reap the direct parent, explicitly reap the descendant and observe the group absent. Its separate cleanup-only-preflight.json and command record remain outside all Lean methods and subcases. Its outcome provides no Lean, semantic-adverse or mathematical credit.

## Source verification and execution limits

Both replacement UTF-8 blobs were fetched back through GitHub MCP and compared exactly with the staged source strings. Static comparisons confirm an unchanged CASES dictionary with 39 IDs, the same six test method names and preservation of original source-before/source-after measurements. No filesystem, shell, Python execution, Lean execution, ref, commit, PR, comment or workflow rerun was used to prepare this repair.

Fresh independent scoped oracle/reconciliation and process-cleanup source reviews found no remaining actionable issues in Controls b8ec51fbcee8516fcfeb89a637fa08097d918602 and final launcher d7f72edb1b7dac6458c568e9411edf21f2703476. The first cleanup review identified timeout-as-success and an inner reap deadline gap; both were repaired before these final reviews. Review limits are source inspection only, including visual syntax inspection; no interpreter/parser, preflight or native execution is represented.

The initial test-first hosted observation remains required. While production Component.lean/controller are absent, bridge/setup failure is implementation absence only, with all 39 semantic subcases not_run. Cleanup-only execution, if later observed, must be reported separately. No test success is inferred from these source blobs.
