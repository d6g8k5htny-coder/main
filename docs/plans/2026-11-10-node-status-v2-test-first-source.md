# Node-status v2 test-first source candidate

This document records the accepted T1 test and historical-fixture harness source
from plan `557e3d173316a55ffc25c8694a29d22c589297ba` and append
`366fda3c562e84d5267ee6017bf9c497c46f9dec`. Root authorized an unreferenced
candidate only after whole-source review and the final bounded fix closed.
The candidate must use fresh Main as its sole parent; its commit/tree and full
preservation guard are reported separately. The reserved branch name is
`codex/node-status-v2-test-first-r0-20261110`; this source record creates no
branch, pull request or workflow run.

The source cut observed on 10 November 2026 was Main
`8aaaa387d11dc21a0867b9cc41cc81743ca59975`, tree
`542e6ee34af4c99ba80c8b4d6acd2c6d5bc22f7c`, with 6,246 leaves in a complete
nontruncated recursive tree. Construction requires another current-ref check.
All 17 harness bodies were fetched and their Git blob identities, UTF-8
SHA-256 values, sizes and line counts independently recomputed in memory.
This is source verification; no Python, YAML or test execution occurred.

## Frozen source

| Path | Final Git blob |
| --- | --- |
| tests/test_node_status_live_v2.py | af0d51d8930d6b7ad4fa718dc0741bc6cae53146 |
| .github/workflows/node-status-v2-contract.yml | f77c11f9c16a574fc193ed9be0e2604f84f856ad |
| tests/node_status_legacy_v1_runner.py | 77e036b5e8e816cec67720318d61fb95d445bc47 |
| tests/test_node_status_live_check.py | 3591d1088f22148fea71e25f829ba27ae4524d08 |
| tests/test_node_status_live_cli.py | 4d0df19642d66ccd22eb008325c15d6b370ac7e8 |
| tests/fixtures/node_status_live_v1/bundle.json | 0723223a945bea1593b6e42cb82c957f64b5488b |

The other eleven additions are exact original payload copies under
`tests/fixtures/node_status_live_v1/payload/<original-path>.txt`. They total
1,072,811 bytes and reuse the original Git blobs. The
[complete line/oracle/payload trace](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/e943ac4b76d000ba2f83737bf1dd63e3b19f0bf8)
records every payload identity, all 12 new parent IDs and all 43 unchanged
legacy method IDs and original body locations. The 17 harness bodies total
1,246,504 bytes; all are ASCII, terminal-LF, mode 100644.

## Test-first scope and retained originals

The new test module imports `evaluate_fresh_records_v2` and `run_live_v2`
unconditionally from the existing live module at line 31. Those names remain
absent. The later automatic R0 must show the actual missing-name ImportError,
one loader FailedTest and zero of 12 behavioral bodies. Another error is not
absence evidence. Actual R0 and every behavioral outcome remain unobserved.

The existing outer classes retain all 23 helper and 20 CLI method names.
Each fixed wrapper starts a fresh same-CPython `-B[-O]-S` child and runs only
the identical original method with its original unittest lifecycle. The full
original modules and assertions stay byte-exact in the eleven-member bundle.
The harness retains original test and program stdout/stderr separately from
outer unittest stderr and captures the actual owned native-Git nine-input
program fixtures before cleanup. These are historical local TEST receipts;
they confer no authenticated HTTP or current-v2 evidence.

Shared bootstrap `ea5e25c0605bb9439a3275b207d86ad08918e988` remains byte-exact
and executes from one retained external file. Complete actual source captures
are retained once by SHA-256 with per-case exact references, while complete
before/after captures are still compared. All raw streams remain complete.
The 2 MiB readback cap remains 2,097,152 bytes. Missing, omitted or erroneous
material means HOLD; runtime and actual complete native material size remain
unobserved.

The final fix admits the temporary parent before any creation: literal
`/tmp` when RUNNER_TEMP is absent, otherwise the exact RUNNER_TEMP value,
through `_destination`. Empty, relative, aliased and checkout-contained
parents refuse. There is no `gettempdir` probe, TMPDIR selection or global
tempfile-cache mutation. Both callers still pass explicit admitted directories.
Only the runner and its two wrapper pins and three workflow pins changed in
the final 31-line delta; all other code, bootstrap and payload bytes stayed exact.

The separate workflow has normal and optimized CPython 3.11 jobs, fixed action
pins, 37 protected source pins and the existing 15-minute job limit. It retains
strict separate 23/20/12 outer streams and all inner evidence. It has no
manual-dispatch route or credential environment for the fixture code.

The new 12 parents freeze the closed seven-key manifest and finite profiles;
exact native Git source/input/mode/import binding; ancestor/orphan/historical
artifact controls with pure-v1 positives; refresh validation before observation
or writes; and the fixed native Math pinned-path/blob to local raw Graph,
current identity and complete embedded-data join. Coherent raw or commit
substitution against an unchanged native Graph must refuse. Synthetic API-shaped
fixture inputs traverse the real observer and supply no actual transport,
current-default freshness or scientific authority.

## Exact source evidence and review

- [Full 15-addition diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/dd838bbb3624ac562678d35ee2cba89af126cee0).
- [Helper loader diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/46a7e963044383516c8bc91c9ad8fcee3851893b) and [CLI loader diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/17d5b18e1e663aa1e650f76b39dbd674f5a3f06d).
- [Combined 17-path diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/ba798023a97bca25c0a028a9279c9bc141733987).
- [Final 31-line temporary-parent delta](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/b9a4486fc70580cd7f6e67b35f3034e015222562).
- Earlier bounded numeric/temp-admission and size-dedup deltas:
  `517c3bba57fd0939b9f15533c7f292fda6409863` and
  `368f2b99186e15eacdb8deeac549f135bba668f9`.

Root read the entire 17-file source and independently reconstructed all four
final blobs forward and inverse. Independent technical review by
`v2_newtest_source_review` closed at C0/I0/M0; Lean had read the entire original
legacy source and its 43 targets. The test/workflow author is
`v2_test_contract_source`; runner/wrappers and packet integration are by
`connected_inputs_refresh`. Organizational-independence credit is zero.

The candidate contains exactly the 17 harness paths, this document and a
minimal append to DECISIONS.md. It preserves every other Main leaf, the entire
old decision prefix, production tools and schemas, the four canonical records
and histories, projection, Graph source cut, all Acq/V1/G5/native controls and
owner state. It adds no production input manifest or record population.

Root's full candidate guard precedes publication. Later implementation requires
the actual R0 read and separate source authority. Actual nine-record adoption
and original Q9004/Q9005 current exercise remain later work. No current-v2,
full-Main, G5, CBZ or scientific qualification follows; scientific_effect NONE.

## Publication failure and bounded packaging correction

The preceding source record belongs to candidate
`0e81116ca8fcde518231a1d1c1ee616e21a58367`; it is retained unchanged.
Root subsequently passed its full tree guard and authorized only the fixed
branch and draft [PR #816](https://github.com/d6g8k5htny-coder/main/pull/816).
The automatic [run 42830214749/1](https://github.com/d6g8k5htny-coder/main/actions/runs/42830214749)
failed before creating any job. Its public annotation reports:

```text
(Line: 55, Col: 39): Unrecognized named-value: 'runner'. Located at position 1 within expression: runner.temp, (Line: 56, Col: 37): Unrecognized named-value: 'runner'. Located at position 1 within expression: runner.temp
```

This exact error was read from the public page at line 146 with
`mcp__codex_apps__search_service_web_run`. The
[small native-error receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/3d26680d2f2e1afc3758d59be1d8d0b136cfb271)
retains the public URL, tool operation, source binding and limits, without
private or authenticated HTML. No normal/optimized target job or missing-import
R0 ran. The zero-job failure remains historical failure evidence.

GitHub's [context table](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#context-availability)
omits `runner` from job-level env and permits it in step env. The same lesson
had already been recorded in [PR #803's source correction](https://github.com/d6g8k5htny-coder/main/commit/db3066b4950d5b88583ea4fd6e88b026203e3928);
this review missed that known platform constraint. Separately, the original
preparation run scalar measures 27,799 ASCII characters, above the
[documented 21,000-character limit](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstepsrun).
That second constraint is measured source evidence; it was not the reported
native annotation.

Root authorized only the packaging correction below, pending source review.
The fixed workflow is
`e7ca903c6df0d8eea44cb3d3f19526dcde2ca2eb` (8,829 ASCII bytes, 174 LF).
New non-discovered utility `tests/node_status_v2_controller.py` is
`70146dc623661ad6275881c6fc564bcb7e0864fc` (27,043 ASCII bytes, 460 LF;
SHA-256 `a96987b44609220a64ae74ab62ddc3edf8f49af73dbb8eabf9abb8e417925af3`).
Its entire body equals the former f77 workflow's runtime SOURCE literal after
YAML indentation removal. All 37 protected pins, controller logic, oracles,
streams and 2 MiB cap are byte-exact.

The short preparation source joins the utility's fixed path, 100644 mode, blob,
length and SHA-256 to native HEAD/GITHUB_SHA, native Git object bytes and actual
working bytes before an exclusive external copy. It rejects aliases, executable
or nonregular files and capture drift. Each of the four unchanged invocation
steps receives the same two evidence variables in its valid step env. Only the
new utility path is added to both existing path filters; events, commands,
conditions and artifact scope are unchanged. All five run scalars now measure
3,123 / 131 / 128 / 132 / 131 characters, each within 21,000.

Existing automatic legacy contract job 129878243834 did print all 23 helper and
20 CLI outer successes in both modes at signed native merge
`a22d6c05314c6aae25eea5076751dec75ad7c08c`, tree `cd37442f`.
Its [entire original job log](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/ccda5c00a54f3a172146d691e06c5550b415d8ba)
is retained. Its old workflow uploads only four outer logs, so full 43 inner
and program readback remains unobserved and receives no omission credit.
Those outer results retain historical local TEST scope and confer no current
HTTP authority.

This correction changes no af0d test, 77e runner, 3591/4d0d wrapper, ea5e
bootstrap, bundle/payload, production tool/schema/record/projection/owner or
Graph source cut. It supplies no successful new workflow execution or R0.
Candidate construction, ref movement and another automatic run require the
separate source and candidate gates; no dispatch or rerun is part of this
source packet. Scientific effect remains NONE.

## 2026-11-10 — Executed R0, bounded current-input source, and regression-first observer correction

The accepted plan557e3d173316a55ffc25c8694a29d22c589297ba remains the governing
design, qualified by the following actual source observations and Root rulings.

Actual dedicated run42834372722/1, normal job129892538635 and optimized
job129892538180, used signed-valid native879c8e352506e1eaae946c56cd99b0ce24d715b3,
tree1dfee4d5695a24276d1b26afef3270acd718f863, ordered parents[8aaaa387,8aa64f61].
Both exact1204-byte stderr originals have SHA256
1b012ea9a5a3bd2d85a2e03f2304710844c3938d15f773c85dd185f180e82543,
Git bloba1238bf882655969fdf5b3f1745e829c60c38fc2. Each genuinely reached the
line31 missing evaluate_fresh_records_v2 ImportError: one loader FailedTest,
native exit1, source unchanged, zero of twelve behavioral bodies. This closes
initial absence RED and supplies no semantic-negative or implementation credit.

Both complete337-frame readbacks were independently extent/SHA checked with
no missing/error/omission. All43 original outer/inner cases and37 program
captures per mode reconcile to actual historical local-Git nine-input fixtures,
the15 current source bindings, eleven immutable payloads and unchanged20403-byte
bootstrap. Normal original bodies total707343bytes; optimized708449bytes.
Complete normal archive5a649202a3dd68a931098f4c8bb7724012d67143 and receipt
38eef2030b46c19625443b2a7aea28530cd14d38; complete optimized archive
7697b4c33a0a27f32e58c2eb093bf974dc6d78de and receipt
ab9bd5471180da60c2d228506ee35c21c8aca9ab retain every body. Whole raw native logs
are762e125c7279287f1772ee58891643ac320bfff2 and
66862f62e7bae682e8ca5a8cae44765076818c78. The unchanged2MiB cap admitted source-
derived framed extents895671/896780bytes. GitHub masks the random stop-command
nonce, so no unmasked stream hash is claimed. Artifact metadata/digests remain
bound to this run/head; ZIP bytes/members were not read. Historical TEST
receipts confer no current HTTP, source-default freshness or scientific credit.

Root then authorized only the three existing production tools and the initial
closed seven-field manifest, retaining pure-v1 e63dc0da, both existing schemas,
all four canonical v0 records/history, projection, Graph source cut and owners.
Source proposals are live20420830d5385150160fc8faa98a75c62cb574b6,
validator1085a79cea8e5df5246c7584fbb8515689d4ff0d,
refresh4e50a8e50ffd2606618a4950d4b23124ed6574f0 and manifest
caf4dcef8bff2f7f3066428ac49be01652428772. The manifest explicitly selects
legacy-four-v0; actual nine-record adoption remains outside this stage.
The independent fixed path set is13 inputs for legacy4 and21 for graph9 TEST
fixtures; no fifth project Python module is introduced.

Capture binds all four code files, two schemas, manifest, snapshot/raw Graph,
the independently derived record directory and applicable three artifact files
to actual native Git mode/blob/working bytes before dependency execution.
Private current-entry bootstraps are separately native bound. Dependencies
compile into fresh captured ModuleTypes; the remaining observer exception is
identified below. Typed JSON/depth/duplicate checks, full embedded-data parity,
fixed dbe Math native-object join, artifact ancestor/path/body equality and
inert final recapture preserve the intended authority boundaries. The default
Graph lookup remains distinct from record URL counters, current-default equality,
whole-tree reconciliation freshness, proof and science. Schemas retain their
reviewed literal blob guards. Source review is not execution evidence.

Two existing freshness tests exposed an incompatible old CLI assumption:
f30b74891080293cd997ea6c92c92ae43530f7a2 invokes main on a manifestless,
untracked external one-record directory. Root explicitly authorized legacy_main,
preserving the entire old5fa7 main function body unchanged after name-only
rename. Current main has no legacy flag or fallback and requires native
manifest admission. Test source9020ee480e33bdd363898dc8885ced28d84a9f72 changes
only five n.main call sites at66/102/117/132/145 to n.legacy_main; all20 IDs,
payloads and assertions stay exact. Existing malformed-v1/v0-shaped helper
assertions remain unknown before observation. No historical helper result is
relabelled current admission.

Independent whole-production review found one Important issue: direct
run_live_v2 still constructs the ambient global ContentObserver instead of
its freshly captured module's default class. The original af0d Parent12 factory
positive expected that same global seam and therefore had an oracle gap.
Root requires a genuine regression RED before the production constructor fix.
Accordingly live20420830 deliberately retains that one known issue in this
regression-stage proposal; it is not production-ready and no green claim follows.

Root authorized only Parent12's five-line factory block to strengthen the
oracle. New whole testdc38155aaff65537a8d9d5c739ec83217469ecdb keeps all12 IDs,
all11 other entire methods and the remaining Parent12 report/context/output
assertions. A separate fresh injected positive preserves record-version
comparison, while a fresh healthy ambient decoy stays unused. A TEST-only
profile observes the real bound constructor's filename/qualname/AST line,
default api=None and interpreter mode. The sole stdlib HTTP call is denied
by a controlled sentinel; no capture/provider override or real network occurs.
The test requires LiveCheckError, zero decoy calls, one real constructor and
one fixed pinned-Math transport request. The old global constructor returns
the healthy decoy's report and should fail the actual assertion.

Required acyclic literal pin rolls are controller
937e8921b5fd9045d00d7470177f010c9651f7aa (only af0d→dc381 in the same37 pins),
then workflowf1b5f9e2ab2d149ba3c3dbd68744b35cea13edcb (only controller blob/SHA).
All commands, five legal run scalars, contexts, guards, artifact scope, original
43 bodies and2MiB cap remain unchanged. Full eight-path source diff
34626d410eb05dc5667929992ae42320b0e9fc67 and test proof
264dfcf9e45bdd2491121536b3de47564f9bceb0 supply exact forward/inverse recovery.

Next gates: Root and independent bounded source review; fresh peer-preserving
unreferenced candidate and Root full guard; ordinary expected-head CAS on
existing DraftPR816; then actual automatic normal/optimized regression read.
Only a source-bound Parent12 failure from the ambient-decoy path counts for
this issue. Syntax/import/source-guard or other failures must be diagnosed.
The nine later Parent12 CLI controls are not reached in the intended RED;
14 preceding import/refresh TEST invocations must not be reported as23.
After that actual RED, the one-line bound constructor correction requires
separate source/candidate gates and fresh automatic execution.

No current production proposal has been compiled or run here. No candidate,
ref update, dispatch, rerun, Ready, merge, actual HTTP refresh, data/owner change,
Graph11-body cut advance or scientific/G5/CBZ award follows from these sources.

Source plan continuation: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/541561fe1da8c7bc0eb84cbbe516625652e438a3
Exact continuation-only text: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/5ce3697857255473f3ed335fbccb517be7b86a1b

## 2026-11-10 — Actual constructor regression RED and bounded two-path correction

Root independently closed actual normal and optimized regression RED on run
42846487233/1, jobs129935214884/129935214340, at signed-valid native
522312e5327dde6c7c06649e6d1e5c4c5eae6a85/treec0105ea54807926a7450afd4a078df0826268396,
ordered parents[e7515c02,e4b0dcde]. All12 bodies ran: eleven passed and only
Parent12 failed at dc381 line1082 with LiveCheckError not raised. Both native
children exited1 without timeout, source unchanged, no errors or skips.
Normal stderr0ff726dc8cbf1ce043d2380f9a8955c0f22ba3a3 is2838bytes/SHA256
bf3bed043b8ac056c7601eaa07cf22e1636398f8f41144ea81c7b27326af4932;
optimized stderr4c0741cde4509420be967a54a09417d9571e7f57 is2838bytes/SHA256
80866c472a4af33eb162143f0e844f09ec4971443384042574013461568d9ce2.
Whole logs89c32952b2d458649872fe9827b951d7c5c57049 and
70cdcd9d465be47f66c605f4218d74b6dacc5649 retain every native line.

Both complete379-original readbacks passed extent/SHA/footer checks with no
missing, error or omission. Normal archivea195784eb963c618a4e1cef76bc11a39445b2dbe
retains810097 original bytes; optimized archive
4bfbd98fb8aaf241e1ca33fdb822c108a5125e59 retains811329 original bytes.
The43 historical outer/inner cases,37 program captures and actual14 TEST
invocations (6 imports plus8 refresh) retain their original local TEST scope.
Later Parent12 constructor/transport/record-version assertions and its nine
CLI controls were not reached; they receive no success or23-invocation credit.
Derived framed extents1017977/1019212bytes remain within the unchanged2MiB cap.
The nonce is masked in platform logs; unmasked stream hashes and ZIP members
remain unread. Optimized causal receipt014f21e17812bdc7cfec8a488b02509ad1e6d119
and the normal receipt referenced by the source packet preserve those limits.

Root then authorized the single production correction: live blob
9a08db343b640b33b3288adfbe20b18124f7c305 changes only run_live_v2's ambient
ContentObserver() expression to captured["modules"]["node_status_refresh"].ContentObserver().
Token validation and local native admission still precede exactly one default
constructor; no observer/provider argument or transcript override was added.
The whole dc381 test, all37 protected pins, controller937 and workflowf1b5 remain
unchanged. Independent source review closes I1 at SOURCE only, C0/I0/M0.
The corrected default route and full12 success have not run and are not claimed.

A separate actual Atlas failure exposed another old caller assumption.
Node-status run42846487855/1, job129935217291, reported10 cases:4passed/6failed.
The external temporary-record helper at b2ac line55 called the mandatory CURRENT
validator CLI and therefore received current-directory refusal before its old
v0 field diagnostics. The actual canonical CURRENT check at line134 passed.
The entire log6f1624dbd427ac31fbd609cb8f9cd02b02b1ceb6 retains that failure;
it is not recategorized as constructor RED.

Root explicitly authorized only that TEST helper's invocation to use the
preserved public validate_directory(root,tmp) API through a Python -c wrapper.
Atlas source1e3523d0b1869838efe6ba7e8999e3a35f401dfa preserves all ten complete
test bodies, payloads and assertions. It prints the same old problems to stderr
and returns1, or the same success text and returns0. The real canonical CURRENT
CLI statement stays byte-exact at its shifted line145. No validator API/current
main, production fallback, legacy flag, schema or data changes were introduced.

Full two-path diff fda749ad0c8bf0ee5924a5c7bb2ecaf3e4ff6393 (1751bytes/37lines)
reconstructs both whole old/new files forward and inverse exactly. The live
change is one expression; the Atlas change is one helper invocation. Bounded
independent review is C0/I0/M0, with corrected-route and Atlas GREEN explicitly
unobserved. Actual Atlas RED and the actual constructor RED provide separate
test-first causes. No assertion or test selection was weakened to accommodate
either current/native boundary.

Next: Root source/document gates, fresh actual Main plus complete existing
feature composition with every peer/canonical prefix preserved, an unreferenced
candidate and Root full guard, then ordinary expected-head CAS on DraftPR816.
Only fresh automatic results may establish corrected-route/full12 and Atlas
outcomes. No dispatch/rerun, Ready/merge, actual data refresh, record/owner
adoption, Graph source-cut advance, full-suite/G5/CBZ or scientific award follows.

Source continuation: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/306d24e7149e7571ae5752fe3fb021c21dc0c134
Continuation-only text: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/63ad2b5b0e0c3d796a7d723fd35edb332fb6f8d1

## 2026-11-10 — Offline v2 passed; V1 source-guard migration proposed

Automatic run 42853434373/1 checked signed commit
5707397c04bfa85a432d5aaf39dddd8ddecc6d4f, tree
64fe825c7919b26fe7a209dd6c21e268bee99339, with ordered parents e7515c02 and
814a2b9c. Both normal and optimized modes passed all 12 unchanged v2 methods,
including Parent12's captured-constructor, ambient-decoy, denied-transport,
record-version and output assertions. Each mode retained all 406 whole originals,
43 historical outer/inner cases, 37 program chains and 23 TEST invocations
(6 import, 8 refresh, 9 current CLI). Extent, SHA, footer and lineage checks
closed within the unchanged 2 MiB readback cap, with no omitted originals.
The constructor witness is established by the executed immutable test body;
separate in-memory trace files were not printed.

[Normal causal receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/2fc6fbd615c1bb09df563a4129f772ab25cbada3) and
[optimized causal receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/a03a05e1e173ae806d3403616defb4d3c2e869c7)
link every original stream and source binding. These are offline TEST contracts.
Artifact ZIP members and the unmasked framing nonce remain unread; current
production HTTP freshness and scientific qualification receive no credit.

Separately, Node status run 42853434576/1, job 129959976233, passed all 10 Atlas
tests and the five original real-record methods in each Python mode. Its
[whole log](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/a4b16245adf252d4f150371cb26fb5a99dc7d0a2)
retains the legacy-four-v0 population scope.

V1 run 42853434363/1 stopped in the source guard in both modes before producing
any V1 method or child stream. Its fixed validator pin still named e9d44b76,
while the reviewed current validator is 1085a79c. The complete 20-entry protected
map has one stale value; the other 19 match the native tree exactly.
Both original failure logs remain retained as
[normal](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/76379aa73ab8b642e77b618d662015730d1a46dd) and
[optimized](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/1877e47910bfafc78cc9aa15a2ec30135fd96782).

The proposed source-only correction changes that one fixed validator identity,
then changes the V2 controller's fixed V1-workflow identity, then rolls its two
fixed controller hashes in the V2 workflow. The V2 protected map remains 37
entries: one workflow identity changes and the other 36 stay exact. All 12 V1
oracles, v2 oracles, capture checks, commands, limits and workflow behavior stay
byte-exact outside those four literals. The validator's entire 15,252-byte
historical constant/API block and all 16 old function bodies are identical.
[Whole three-file forward/inverse diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/6405bc1a8036c44bce040b40b3eeb0c546fbced9).

Other CI holds remain separate: the original Q9006 method expects CLI schema v1
and receives v2. The site freshness-provenance step exited 1 and uploaded two
files. The fixed expected originals are controls-normal.stdout and
controls-normal.stderr in artifact 13782246777; their member bytes remain unread
here, and execution of either mutation driver is unproven. Neither hold is fixed
or awarded success by the V1 pin proposal. Root and independent source/document review, a fresh
Main/complete-feature candidate guard, and fresh automatic V1, V2 and all
applicable checks remain required. Records, owners, projection, the historical
Graph cut and actual data adoption remain unchanged. Scientific effect: NONE.

Linked source-plan continuation: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/4bc51f83b6faf1d4ba8901ebdb8f190e3fdd76ec
Continuation-only text: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/a8ff56dfa0a94241db837f73b254ce6de11f1ac4

## 2026-11-10 — Scope historical mutations to the legacy function

The isolated reader recovered both original files from site artifact 13782246777.
The [4409-byte stderr](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/a6ca526e78aa557b5e76c3675eacd2c5ff4652d7)
records 24 normal control methods: 23 passed and only
test_real_subject_assertion_is_not_a_kill failed at line 93 with 2 != 1.
The [96809-byte stdout](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/332a5ba0e4117394d551e21558e5a0236b1c78b0)
contains 22 classifier self-test mutation records.
Controls in optimized mode and both mutation drivers were never executed.
[Capture and custody receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/0c7c822ab6a3cc0515f4c4f95faa3ef0777b7549).

Current refresh source has one schema-preflight anchor inside legacy_main and
another in current validation. The two-file source correction selects the exact
top-level AST legacy_main span, rejects ambiguous, async or decorated definitions,
and requires one anchor inside that body. Both the subject-origin control and the
schema-preflight mutation use this same helper. Reconstruction preserves every
byte before and after the historical function, including current validation.
All 24 test IDs and assertions, 11 mutation definitions, the origin runner and
four provenance/classifier functions remain exact. The other ten mutation cases
retain their global uniqueness guards.
[Whole two-file forward/inverse diff](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/4862114abe5f29ede9ee0da12dc47bdca330a768).

One integration proposal collects seven existing source paths:
V1 workflow 1b2b0211, V2 controller 273f1539 and V2 workflow 395d4f1e;
Q9006 test module 472ae35d and live workflow d4d013cc;
site grader bb4c1519 and controls ebcaa0a9.
The [three-pin packet](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/40b36c3deb5a6d5df1abac07e65bb679ef0b0a8b)
and [Q9006 packet](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/48883a600226713db22abee6e9348f0ffab6599e)
retain their separate actual RED causes and source proofs.

The earlier v2 offline and Atlas success remains tied to native 5707397c/tree64fe.
Fixed site controls, both mutation drivers, migrated V1, migrated Q9006 and fresh
V2 results for the combined source cut are unobserved. Source/document review,
a fresh-Main plus complete-814a candidate guard, and automatic CI remain required.
The reader branch stays separate. Records, owner histories, manifest population,
schemas, production tools, projection and Graph cut stay unchanged. No actual
data adoption, current HTTP freshness, science, G5 or CBZ credit follows.

Linked source-plan continuation: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/e111e594746ecad185349cfe0a7d8285d2c6e5cb
Continuation-only text: https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/96698e831f6542e67d6b45e932946066f72d2b69


## Task4 Stage4A: original dependency and artifact RED

This source proposal starts from merged Main 5819d5536b300436792c56dc3ad247006c97f170, tree c1a5ee039e641ce9a3125ab7b2747ce9621d6051. It replaces only the two original dependency/artifact skip blocks with ordinary nonempty assertions. IDs, canonical loaders, following dependency/path checks and every other module byte remain exact. The four canonical records, projection, production tools, schemas and Graph are unchanged.

Root's retained [local receipt](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/281cc2d5f0a2e2c7bab7957780afcbde754a491f) records both actual modes: the original baseline had five OK and two skips; the proposed bodies had five OK and two ordinary FAIL, native exit 1, no errors/skips. Failures occur at the original bodies' new assertions, lines 832/839. This local Python 3.12.14 evidence does not qualify the unexecuted hosted observer.

The existing workflow preserves Atlas/current-validator commands and selects exactly seven original IDs in fresh same-Python normal/optimized children. Its fixed nineteen native source/data originals, complete four-record directory, absent namespace initializer and selected bytecode-cache absence are bound before/after. Large source bodies and the two fixed workflow-literal helpers stay in the artifact; all seventeen printed originals plus framing share one 2 MiB limit. Ordinary failure exits stay failures. Upload explicitly includes the fixed .github source original. Fetch depth zero supports the separate later historical-ancestor check.

The [complete source ledger](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/32b1ad2790d3ce7803bb4a1928598157b40af592) contains both full forward/inverse diffs, source map and syntax checks; [workflow source](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/601d9df5cbde35a236973fa993a46070a86c6624) and living module 276902524c812e3a625dcd3c8a7c84fdc691157a are unreferenced proposals. This narrow workflow establishes checked-byte custody and a reviewed static import/read boundary, not a runtime census of every open/import. No new tracked utility, profile/audit-hook framework or expected-RED green wrapper is added.

Stage4B six-edge/three-artifact/reader strengthening (73480f24243ab128a2c8f51b446028a8199dfed1) and actual-nine adoption remain separate. The Stage4A guard deliberately freezes four records/nineteen inputs; later data requires an explicit reviewed nine-record/twenty-four-input update. Hosted ordinary-body RED must be retained and reviewed before data adoption. No current remote freshness, owner, scientific, G5 or CBZ-0 qualification follows. Candidate/ref/PR/run actions remain outside this source proposal.
