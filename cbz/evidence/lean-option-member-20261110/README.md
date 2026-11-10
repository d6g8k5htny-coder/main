# Lean kernel-option member detection: bounded adverse record

At Main commit `c32e6f27d318e25708939105285366a7a0daf971`, scanner blob
`0d53463de621b821519b71559db2834eef3a8d94` (SHA-256
`bf797f33d5df7a8e695f62e6a2d70dca05ec99956ac5c33aa0347164eac48410`)
returned exit 0 with zero findings on two tracked negative fixture files:
`kernel_replay/tests/fixtures/BadPublic.lean` and `BadPrivate.lean`. Both
contain the live token `debug.skipKernelTC.set` at line 13, column 41. Their
respective SHA-256 values are
`9b636915c4c868091d72b6c72337b6c255cab8ede30b8b394b43389e1fdd95d5`
and `fa95b63ae8037158cdc643a6542ed28dc1adde86a529a535cfc0c713033ec429`.
The exact original scanner is recoverable from that native Git blob; the
normal and optimized command outputs and exit statuses are retained here.

The root cause is that the lexer emits the entire dotted identifier, while
classification checked only its last component, `set`. The added control
exercises real fixture bytes, CLI exit and finding counts, literal and escaped
option components, a `.get` member, comments, strings, a different identifier,
and an escaped identifier containing a dot. Two selected controls failed in
both modes against the original scanner (`lean-option-member-red-*.log`),
with six assertion failures each. The first post-fix runs detected the
fixtures but errored because the new test tried to parse the CLI's findings
and summary as one JSON document; those errors are preserved in
`lean-option-member-green-*.log`. The test was corrected to parse the final
summary line, and both selected tests then passed in both modes
(`lean-option-member-green2-*.log`).

The scanner source SHA-256 after the fix is
`dd8ee1c3dcf1f263c77238d35d639ed000bdc1dd7f382512219ac281d80e4a17`.
Its conservative component check reports even read-only member spelling; it
does not resolve aliases, inspect generated Lean declarations, decide whether
an option is enabled, or establish full Layer A coverage. The historical
fixtures and their statements are untouched and remain negative test data;
this lexical check does not prove their transitive isolation.

Using that working scanner, Git-tree scans of Main `c32e6f27` report three
fixture violations and one `unsafe` review finding in 18 Lean blobs in each
mode; the report SHA-256 is
`53f6ed8b22ba5e2d1051d0b9a65f3304f9f2e2305c17c691108ab8b39e9823e9`.
Math `9bdfc54b76173b57912503bfcf901a5f9d43b09c` reports no listed tokens
in 1,974 Lean blobs in each mode; report SHA-256
`f8bc30f2df38fcf1b301b774c4c555df3540829348a1e97f1536a76dbb05100f`.
These byte-bound scans do not compile Lean or prove axiom, kernel, isolation,
scientific, or whole-federation status.

The local source-specific census campaign ran 85 top-level tests in each
Python 3.12.14 mode without top-level skips. Its independently read-back
reports contain 102 clean selected baselines and 102 authorized selected-body
mutant failures per mode: 75 Lean and 27 Python census mutants. The new mutant
removes the nonfinal option-component check; its selected test has both a
clean baseline and an actual authorized failure. The retained full-child
records include the explicitly checked two and four recursion-guard skips and
two deliberate syntax errors in each mode. Readback logs and top-level outputs
are included; raw child reports and source snapshots are retained separately
in the supplemental evidence archive. This is scoped test evidence, not final
G0 or G1.

The workflow's expected population is now 85 tests and 204 census reports per
mode. Its synthetic guard fixture first failed with the previous counts
(`lean-option-member-workflow-red-*.log`); after the count update, both workflow
guard tests passed in each mode (`lean-option-workflow-green-*.log`). Those
shell-guard tests use synthetic child outcomes; they are not additional real
scanner mutation executions. Exact clean-commit hosted qualification and
artifact readback remain required.

The first staged `git diff --check` reported trailing spaces in the captured
raw unittest failures. The command sequence still created commit `fd8b9e5c`;
that observed failed check and exit 2 are retained as `diff-check-first.*`.
The follow-up applies the repository's existing evidence-log whitespace
attribute narrowly to this directory, preserving the original log bytes rather
than editing the recorded failures. The complete diff must be checked again.
