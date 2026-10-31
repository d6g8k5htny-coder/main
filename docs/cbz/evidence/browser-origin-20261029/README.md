# Browser mutation-grader origin correction

Scoped engineering evidence, 2026-10-29. Original false acceptance was reproduced
against source tree6786ec405cef50e0fef73c7fcd9ca3945c555230. Monkeypatching only
the child class setUp to raise AssertionError made the old six-mutant meta-test
accept every failed setup as a kill. The actual test bodies never ran. Both
normal and optimized observations are retained. DD lines in those original
combined logs refer to the separately repaired Math541 harness, not browser scope.

Final local commands and exact limits are documented in
`docs/cbz/ATLAS_BROWSER_REGRESSIONS.md`. All45 combined browser-contract controls
passed in both modes, including30 controls in the amended module, original six
production source mutants and13 new lifecycle/origin controls. Six additional
grader guard mutations per mode were graded with a pristine independent copy
of the repaired structured classifier, not the mutant's own admission flags.
Raw stdout/stderr are preserved; warning in the lazy-body adverse mutant is not
a passing oracle. Full repository and actual browser execution on this source
remain unexecuted here. Production runner/browser/workflow files are unchanged.

Parent review independently reran30methods normally and optimized, checked the
raw-origin classifier and pristine meta-runner, and required explicit target
cardinality refusal under -O and portable repository-relative paths. Both were
corrected before final rerun. This is same-family engineering review, not a
formal PR approval, mathematical proof, universal harness soundness or CBZ-0.
