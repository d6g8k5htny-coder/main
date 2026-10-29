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

Source `2db895e866dcd37ae7343eedbdfc2272aeb82e6c`, run 40950731897, job 123717841712: all 43 tests passed in each Python mode, including 21 source mutants with matching control-side bypass receipts and eight harness mutations. The strengthened controls bind the specific input or outcome: the affected byte field, real Node skip, materialized fixture, exact preflight failure and child diagnostic where applicable. The prepared Node replay had 764 passes, zero failures/skips/cancellations/todos and exit 0. Its [original transcript](green/2db-node.tap) has SHA-256 `e0acd188a5a881b377742be0547ea0906a5ae20c5dcf4e2f0a1c0bd33b8fb3f0` and Git blob `dcbeca296f5d6d46180001ac5c83e3347366be66`. The [original hosted job log](green/2db-hosted-job.log) records both Python modes and the transcript transport. Artifact 12930309399 reported ZIP SHA-256 `992168bb5cd49ed47fcc87933751cc6eb19a95942d0b0bbec22026623a1575c1`; its downloaded bytes were not independently rechecked here. These results are scoped to the tested source and finite controls; no federation gate or CBZ-0 claim follows.
