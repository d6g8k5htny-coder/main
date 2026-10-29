# Prepared Node regression tool registration

Tool: tools/cbz_node_regression.py. Maintenance owner: none, self-contained.
Serves CBZ criteria 5, 9, 11 and 12. It closes the reproducibility gap between
raw Node discovery and the actual three-variable prepared fixture suite.

Usage from the clean main checkout:

~~~sh
python3.11 -B -S -m unittest discover -s tests -p test_cbz_node_regression.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_node_regression.py -v
python3.11 -B -S tools/cbz_node_regression.py --math-root /absolute/federation/Math- --query-root /absolute/federation/query- --output /absolute/new/evidence/node
~~~

The output directory must be absent and outside the three source trees. All
inputs are immutable Git blobs, verified against descriptor size/SHA256 and
declared Git-blob identity. No network endpoint, branch alias or current working
file is used as a missing-object fallback. All tracked tests/*.mjs are executed.
The report binds the actual command, fixture identities/environment, runtime,
exit, complete no-skip TAP summary, raw-log digest and unchanged owned source
endpoints. RUNNING/FAIL cannot count as PASS. Existing output is never overwritten.

The tests use actual temporary Git repositories and real Python/Node subprocesses.
Named source mutations cover byte identities, regular-file admission, pin URLs,
complete input materialization, hidden source/mode changes, zero/skip/ambiguous
summaries, subprocess exit, empty discovery and false-green error handling.
Harness controls reject zero exits, errors, foreign assertions, zero tests and
launch failures. The harness is itself mutated. A source/harness launch error
does not count as a killed mutant. Ordinary and optimized Python are separate.

Observed source b3eacf768fc402ad65c9bf41e3b6b8d3c670342c:
33/33 methods in each Python mode, 14/14 source mutants and 3/3 harness mutants
produced the named test assertion failures, and an unmutated copied-source baseline passes.
Native run 40943173611; normal job123691557774, optimized123691557243,
prepared123691557583. Prepared Node: 764/764, all failure/cancel/skip/todo counts
zero, exit0; 38 immutable fixture records; Nodev22.23.4.
node.tap SHA256 c644a9e64da171421fb6fbc64b2b747a372085eba7c7266f8f7f566e3e6ebe36.
The subsequent source and environment controls were observed in the newer run below.

All original RED results remain under red/. The first RED ran24 tests/mode with
24 assertion failures because the helper was absent. The first implementation
then exposed a redundant skip mutation and duplicate self-harness anchors;
these four failures/mode were preserved and repaired in the test harness.
No failing behavior was deleted or disabled.

Scope: operational test reproduction only. This tool is neither a mathematical
certifier nor a final G0–G11/CBZ certificate. It does not prove arbitrary ignored
dependencies, malicious simultaneous writers, runtime correctness, complete
mathematical coverage, or authenticity of supplied transcript text.
The full guide in docs/cbz/COLD_START.md records final gate gaps explicitly.
Root integration must register this tool in GOVERNANCE.md/TOOLING.md and add its
command to STATUS.md; this isolated branch does not overwrite shared canonical
status while other contributions are being composed.

## Subsequent source-bound observation

Source `dba9a9eaf65e190b2aee0445244eb26d6034b022`, hosted run 40948751545, job 123710906208: 41 tests in ordinary mode and 41 in optimized mode passed, including all 21 named source mutations and all four harness mutations. The prepared Node suite reported 764 tests passed, zero failures, skips, cancellations or todos, exit 0. It used 38 pinned fixture records, materialized as 39 files. The raw Node transcript is [dba-node.tap](green/dba-node.tap), SHA-256 `b2bc7b2b48068d475129118d07b62a0b8b217725ec1b68667ea584286e54450f`, Git blob `8cdeca3f173700f826563ec02bf30b48ea1680d5`. The console transport verified the digest and rejected a modified transcript; uploaded artifact 12929309903 reported ZIP SHA-256 `139ff5eec7300d582dc1af1d2504399fe84724a1a7a59dd2905185324c47b527`. The ZIP itself has not been independently downloaded and rehashed here. The source-bound observation does not certify the later evidence-only commit or the complete federation.

**Historical mutation-evidence limitation:** The older dba9 harness checked the selected test name and failure summary without binding the particular bypass observation. The later f4ad harness below repairs that limitation; results remain scoped to its exact source and finite control cases.

## Reason-bound control observation

On source `f4ad0f49dbf9963fcec849a3dc66f1de46f51ac3`, hosted run 40950290452 job 123716298060 passed all 43 tests in both Python modes. Its 21 retained source mutants each produced the corresponding control-side, per-run nonce-bound bypass receipt and expected named assertion failure; eight harness mutations and adverse controls for the wrong event, subject, nonce and missing receipt also passed. Prepared Node reported 764 tests passed, zero failures/skips/cancellations/todos and exit 0. The unmodified original TAP transcript [f4-node.tap](green/f4-node.tap) has SHA-256 `9b90d335aa4dc9345aaa1618d2aecc23c2bfd9f4c836f98ae71e784726303eae` and Git blob `bbeceb67e877902e1967d0bb5a86fb78613e2970`. Artifact 12931890855 was reported uploaded with ZIP SHA-256 `ef2fbff49b322fe1d2dd7182e5065915ed7b1dc397d39ecd920304ec88127086`; downloaded ZIP bytes have not been independently rechecked here. This is finite operational verification, not G1/G5/G9 or CBZ-0 certification. The stronger successor source and its observed hosted qualification are recorded below.

## Strengthened control observation

Source `2db895e866dcd37ae7343eedbdfc2272aeb82e6c`, run 40950731897, job 123717841712: all 43 tests passed in each Python mode, including 21 source mutants with matching control-side bypass receipts and eight harness mutations. The strengthened controls bind the specific input or outcome: the affected byte field, real Node skip, materialized fixture, exact preflight failure and child diagnostic where applicable. The prepared Node replay had 764 passes, zero failures/skips/cancellations/todos and exit 0. Its [original transcript](green/2db-node.tap) has SHA-256 `e0acd188a5a881b377742be0547ea0906a5ae20c5dcf4e2f0a1c0bd33b8fb3f0` and Git blob `dcbeca296f5d6d46180001ac5c83e3347366be66`. The [decoded hosted job log](green/2db-hosted-job.log) records both Python modes and the transcript transport. Artifact 12930309399 reported ZIP SHA-256 `992168bb5cd49ed47fcc87933751cc6eb19a95942d0b0bbec22026623a1575c1`; its downloaded bytes were not independently rechecked here. These results are scoped to the tested source and finite controls; no federation gate or CBZ-0 claim follows.

## Mixed-failure harness correction

An additional control exposed a remaining weakness in the earlier f4/2db matcher: an output containing both the expected test failure and an unrelated failure could count as a killed mutation. On PR head `e77502daab6abd0efcfe158b079ac2efce4eb997`, actual tested merge `802cb0294db93665f52b31f2d3b837c53ff28021`, run 40951199942 job 123719465476, the new control failed as intended in each Python mode: 44 tests run, exactly one assertion failure, no errors. The [decoded hosted job log](red/mixed-foreign-hosted.log) is preserved. The prepared Node step independently reported 764 passes; it does not repair the harness finding.

The successor matcher requires exit 1, a failures-only summary with a consistent failure count, and no foreign failing test names, in addition to the nonce-bound specific observation receipt. New controls preserve and exercise mixed failures, skips/expected failures, abnormal exits and inconsistent counts. **The successor source is not yet qualified until its own hosted execution completes.**

## Outcome, fixture and source-identity corrections

The stricter matcher at head `db7d4a9cef29f2927333b9979032200b144e29e7`, tested PR merge `fc874d3d6da55276180937bae8e239ab395570f8`, passed 47 tests per Python mode, 21 source mutants and nine harness mutants in run 40951498618. The separate prepared Node replay reported 764/764, zero skips and exit 0. Its [decoded job log](green/db7-hosted-job.log) and exact [Node transcript](green/db7-node.tap) are retained; the transcript's Git blob is `d73c8c6dc9ccb53ab39c629a3f14231a4f5bf071` and its reported SHA-256 is `380c35cb9c274ec44a63d373eb635b886cc2ab3c1f29258a707873da86255066`.

Further actual REDs found that a fixture receipt could be written after a nonzero Node exit (run 40951811637) and a timestamp/size-valid poisoned museum bytecode cache could execute while the unchanged source bytes were recorded (run 40952898593). Each failed in both Python modes; their decoded logs are retained in red/. The current candidate requires the actual zero exit and compiles museum source bytes directly, and the test loader also compiles source bytes. It adds corresponding negative controls and source/harness mutations. The poisoned test-loader RED was subsequently observed in both modes, as detailed below. The combined successor was then observed passing at its separately identified tested merge.

Observation receipts are emitted by the control in the same Python process that loads the chosen mutant. They distinguish the finite accidental-failure and wrong-observation cases exercised here, but are not a security boundary against malicious code that deliberately forges a receipt or a compromised interpreter. The helper provides operational regression evidence only; it does not establish G1, G5, G9, or CBZ-0.

## Test-loader poisoned-bytecode observation

On test-first head `3598f413da9be44263c2c6e853c7245012ee42f6`, actual tested merge `590c601766d112f15b61a3e9e6990f0945db3812`, run 40953041841 job 123725878794 ran 50 tests in each Python mode and had four assertion failures per mode with zero errors. The poisoned museum and test-runner cache controls, fixture-exit control and copied-source baseline reproduced the intended defects. The [decoded original job log](red/both-poisoned-bytecode-hosted.log) is retained. These are genuine RED observations; the subsequent source-only candidate's scoped execution is recorded below.

## Combined source-only repair observation

Source head `8aa7d353d6864b6481df5508aaea7b73f185abdf`, tested merge `e01ab09f9a198009de74a0fb1f017c96f5e85b7d`, hosted run 40953580427/job123727766200: all 51 tests passed in both Python modes, including 22 source mutants bound to specific per-run observation receipts and 11 harness mutations. The original poisoned museum and runner-cache controls passed. The prepared Node suite reported 764 passes with zero failures, skips, cancellations or todos and exit 0. Its [Node transcript](green/8aa-node.tap) has SHA-256 `4da55b5959ffba1c920db90eff091d04dbafcff4fac91bbe6f460b65dee781fb` and Git blob `2327ca95a56afdf4d7efd46ec469f26a53c15953`; the [decoded hosted job log](green/8aa-hosted-job.log) records both modes and the source-bound replay. Uploaded artifact 12933401209 reported ZIP SHA-256 `cc9c97e2d0326d5226be953ddc0bdbbd69e8a11d4b91b3bc90aef93c79d7fa75`; downloaded ZIP bytes were not independently checked here. The tested merge has parents c93 and 8aa, and its runner, tests, museum and workflow blob identities independently match the source head. Current/final federation gates remain pending.

## Additional reason-binding review (test-first)

Independent source review of dc49 found three remaining admission weaknesses: the matcher accepts a multi-test count even though a single control was selected; fixture receipts accept a substring of an incomplete or ambiguous Node TAP summary; and the loaded-source receipt accepts an arbitrary nonzero child exit accompanied by loose marker substrings. New adverse controls specify the exact single-test count, complete unique one-test TAP outcome, and exact loaded-source observation. Their execution and repair are pending. The earlier 22 source/11 harness counts and prepared Node passes remain historical observations of their exact tested versions; they do not qualify the uncorrected admission logic or a final gate.
