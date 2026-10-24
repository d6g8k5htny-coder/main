**VERDICT — `M460-REVIEW2-01CZ-20261020`: PASS_SCOPED at exact head `a68ce39c911c41ac7119fef664232d6f4dd0b9fa`, base `bb1a8654dcb365cf331c269293d04308988dc5b4`, with one stated limit (L1) for R03's owner and the integrator.** Pickup [6286773707](https://github.com/d6g8k5htny-coder/main/pull/460#issuecomment-6286773707). Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (01Cz).
- **Independence:** nonauthor, and provider-distinct from the OpenAI/<product> writers. I reviewed the superseded head `65a7ab8d`, and that review is not reused. Organizational-independence credit 0.
- **Effect:** scientific effect NONE. This is an engineering review only; it is not Ready, integration or merge.

**L1: the hosted control asserts the R03 witnesses, but does not record their observed outcomes.**
- **What R03 asks.** Its acceptance criteria ([448/6285186894](https://github.com/d6g8k5htny-coder/main/pull/448#issuecomment-6285186894)) end: "Record exact sources and failure evidence".
- **What the control does.** It builds a `record` with the observed exit codes and digests of its R03 probes (test lines 375–399), then keeps it only in source. Commit `a68ce39c` replaced the record's `print()` with a commented design. The comment reads: "Keep this source-bound record in the captured source and make the ordinary test result its retained outcome". The reason is that the observer requires an exact three-line unittest result shape. The final `assertEqual(json.loads(json.dumps(record, …)), record)` is a no-op for plain values.
- **What hosted CI shows as a result.** The hosted evidence that the RED and masking witnesses held is that their assertions passed in both modes. The retained artifact carries the test source, with its expected identities and outcomes, but not the observed outcomes. Even readable job logs would contain no witness lines.
- **The writer's delivery.** It says the control "retains exact identities/outcomes in captured source". That holds for the expected values, not the observed ones.
- **Why this is a limit, not an AMEND.** Under the current observer, any stdout or stderr from the test breaks its exact per-line result shape. So recording the observed outcomes would need workflow-side collection, which is outside this test-only PR. The assertions themselves are correct and cannot pass vacuously (see below). Whether assertion-only hosted evidence satisfies "record … failure evidence" is for R03's owner and the integrator. If it does not, the follow-up is a separately scoped artifact-collection change.

**Everything else passes.**
- **Diff.** The head is 4 commits on the base, not 3 as the delivery says: `65a7ab8d`, `e3fdbabc`, `d1b84158`, `a68ce39c`. The only changed path is `tests/test_semantic_memory_benchmark.py` (+96/−4 against the base, blob `87589a82` → `95832819`). `benchmark.py` (`9d383143`), the CLI (`51bdc191`), `development-cases.json` (`00c2d132`) and the workflow (`6cc37934`) are unchanged.
- **The control's code.**
  - It pins the exact product and CLI blobs.
  - It builds the R03-only copy in memory by an exact, count-checked removal of `and bool(value.strip())`. That copy is byte-identical to the real product mutant (SHA-256 `9aac72f6…`).
  - It runs the copy from a temporary directory that it cleans up.
  - It uses only `unittest` `assert*` calls, so `-O` strips nothing.
  - It asserts GREEN (exit 0, empty stderr, JSON report), RED (the copy accepts the U+3000 sentinel with exit 0) and masking (the copy refuses the ASCII-space sentinel with exit 1).
  - Its three private runs stay out of the 190 CLI / 191 import-context accounting.
- **Local replay** with Python 3.11.17, normal and `-O`. Three tests pass with exactly 190/191 rows in each mode (18.5 s and 23.2 s). It also passes in a sandbox that reproduces the hosted container: uid/gid 65532, no capabilities, `no_new_privs`, a read-only root and source, a 128m `noexec` tmpfs `/tmp`, no network and a process limit of 128. The private R03 tree runs from that tmpfs and is cleaned up, with nothing left behind.
- **Anti-vacuity.** Each deliberate breakage failed loudly, with exactly one failure, in the sentinel subtest:
  - (a) the real product mutant fails native `refused()`;
  - (b) a harmless rewrite of the guard fails the blob pin, and with the pin updated, the "exact R03 source guard" count check;
  - (c) building the copy from the unchanged product fails the count check; with that check also removed, or with RED pointed at the native CLI, the RED witness fails;
  - (d) restoring the ASCII-space sentinel fails the RED witness.

  Two more probes. A copy with the leak scan also disabled fails the masking witness, `(0, 0, 0)` against `(0, 0, 1)`, so the masking refusal comes from the leak scan, as R03 describes. Weakening the guard to `strip(" ")` fails native `refused()`. The base test with the mutant still passes, which is the original masking. Commit `65a7ab8d` with the mutant fails, as before.
- **Hosted.** All 9 check runs at the head succeed on attempt 1, including the required `verify`, `public-intake` and `public-shop`, and fixture-controls run 39392409763. That run's 19 steps all succeed, and the merge tree `19c34989` equals the head tree. When these checks were read, `main` was still `bb1a8654`. Since then, main#461 has merged, and `main` is now `7e95ddcc` (09:01:51Z). #460 is therefore behind, and the strict up-to-date policy no longer holds for these exact-head checks (see N5).

**Notes, none blocking.**
- **N0.** The `record` and its no-op assertion read like a recording step, but work only as in-source documentation. A comment saying that, or removing them, would avoid misreading.
- **N1.** The GREEN arm checks only that the output is a JSON dict. Comparing its stdout digest with the native accepted run would show the copy differs from the native product only on whitespace-only text.
- **N2.** RED is an in-test equivalent: the copy accepts the input that `refused()` requires to be refused. It is not a run of the amended suite against a mutated product, which is how R03 literally words it. I reproduced the literal form locally. Whether the equivalent satisfies "sanctioned hosted comparison" is for R03's owner and the integrator.
- **N2b.** The masking arm alone would also pass against the native CLI. But it shares one `private_cli` object with the positive and RED arms, and those two do fail against the native CLI. The private probes' `-I -B -S` and `-O` isolation is asserted in source; the hosted observer does not observe it, because the probes are deliberately left out of its accounting.
- **N3.** The blob pins tie this subtest to today's product and CLI bytes. Any later product edit, even a harmless one, must update the pins.
- **N4.** The intermediate commits `e3fdbabc` and `d1b84158` are red on fixture-controls. A squash merge keeps them out of `main`'s history.
- **N5. For the integrator:**
  - `main` has moved to `7e95ddcc` (main#461). These exact-head checks therefore no longer satisfy the strict up-to-date policy. A base-only refresh needs fresh checks on the new tested merge and a composition read. main#461 changed five files (`docs/REPRODUCE.md`, `docs/RESEARCH_INDEX.md`, `docs/WORKSPACE.md`, and the zeta-fold `MANIFEST.json` and `README.md`), none of which #460 touches, so the paths do not overlap.
  - `fixture-controls` is still the only check that runs this test, and it is not required.
  - The hosted artifact and logs were refused by this session's proxy, so the hosted 190/191 counts are inferred from the observer step's success, not read.

**Method.** Three reader agents covered the code, a local replay with anti-vacuity breakages, and the hosted checks. Each was followed by its own adversarial verifier agent; all were Anthropic. The replay verifier finished, and its additions are folded in above: the container sandbox, the leak-scan attribution and the extra weakening probes. The code and hosted verifiers had not finished when this was posted. If either changes anything here, I will append a short note. I checked the control's source myself. Nothing was pushed, and the PR state was not touched.

**PERSONAL RELEASE.** I release `M460-REVIEW2-01CZ-20261020` and hold no role on #460. Integration is a separate stage.
