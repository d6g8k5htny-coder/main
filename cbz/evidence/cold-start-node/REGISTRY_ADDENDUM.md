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

**Mutation-evidence limitation:** The current harness checks the selected test name and failure summary, but does not independently bind the failure to the particular bypass observation. An unrelated assertion failure inside the selected test could be misclassified as a killed source mutant. The 21 source-mutant outcomes are therefore named-test observations, not a fully reason-bound mutation certificate. Repair and adverse-test this limitation before admitting the helper as a completely verified gate tool.
