# Task B contract staging design — source only

Binding brief: Main blob56d807b42ff19f95e9a1813a93515b549c034ba1.
Base head e72c45ab8576bb0c81b0834b11952a7ca3ae1c00; tree f2355c08e4c01c30f2cbf7baf92a1be2ab1c7359.
No execution is claimed. Scientific NONE; qualification/classification BLOCKED; execution_authenticated false.

This adds C04–C07 (four methods, 13 subcases) to the original six/39.
Expected total is exactly ten methods/52 subcases. Every original Task A method
body and Lean control body is retained verbatim; the fixed Task A case map is
retained separately in TASK_A_CASES and census metadata. The 12 pure protocol
methods, Task1 JSONL/Name/classifier/assertions, frozen profile, historical source
and all99/1137 bindings are unchanged.

C04 uses genuine Fixture.BadDependency.BadProof plus compiled AxiomUse declarations.
The type-only assertion first requires actual typeOnly.type to retain TypeAlias and
its actual value not to mention BadProof. Definition and opaque value controls
separately require their actual values to mention that exact dependency axiom.
Actual strictAxioms is invoked for each root; native inventory closure is checked
separately. Existing strictAxioms is retained; no behavior is removed to force RED.

C05 alters only actual loaded Safe ConstantInfo fields in the test. Missing type
and missing value roots use separate alterations. The single missing-link subcase
checks both an appended actual inductive constructor link and an appended actual
recursor mutual link. Lookup replaces only the altered root and otherwise resolves
the actual loaded map/import environment. Exact missingConstant constructor and
lossless absent Name are asserted; no invalid source is compiled.

C06 compiles a genuine module/prelude-only OwnedAxiom with exact root propext.
Actual raw imports=[] and actual owned axiom CI are Lean subcases. The third
owned-axiom-block subcase has Python_controller provenance: it invokes unchanged
Task1 classify_inventory over a separately retained native inventory and requires
OWNED_AXIOM for exact propext while FORBIDDEN_AXIOM is absent for that same Name.
No classifier stub or new production policy helper is introduced.

C07 compiles ordinary unsafe and partial source. Pinned PreDefinition/Basic uses
Compiler.mkUnsafeRecName for the partial implementation and sets safety=.partial;
the public wrapper is opaque and is not credited as isPartial. Controls fetch that
exact generated CI and require native flags, owner/identity and retained native
inventory fields. Missing generated CI is setup failure, never a skip/pass.
Each of the four C07 subcases runs in its own fixed invocation, so missing guard
does not stop the separate partial guard observation.

Proposed production interface, accepted by root for staging:
SDGA.guardReplayConstant : Lean.ConstantInfo → Except SDGA.AuditError Unit.
A narrow test-only term elaborator checks the current Component import environment
for that exact name. When present, it elaborates and invokes that exact declaration
with its complete explicit function type. When absent, the produced IO function
only throws IMPLEMENTATION_ABSENCE: SDGA.guardReplayConstant; it cannot return
an expected AuditError or pass. The same tests automatically call the eventual
production declaration without assertion replacement. Wrong type blocks compile.
Pinned Term/TermElabM.elabTerm provides this typed elaboration path; no unsafe
evalConst, unchecked CI insertion or local imitation of safety policy is used.

The guard's eventual production implementation must also be called by actual
replayOwned before upstream replay. These tests do not establish that integration;
root source review must verify it. Task C kernel replay remains absent.

After root source review, Task A qualification and explicit execution authorization,
run from a fresh Main checkout using the pinned Lean4.34.1 commit
5045d0056413266e57c625dcd7c365b10e377c52:
python3 tests/sdga_component_contract.py --output "$RUNNER_TEMP/sdga-task-b-red" -v
Use a new output path for every subsequent invocation.
Fixture compile order is fixed and sequential: BadDependency, Safe, AxiomUse,
Skipped, OwnedAxiom; then Component separately; Controls only after prerequisites.
AxiomUse is not scheduled if BadDependency fails. Native inventory precedes each
B fixture's controls. Existing bounded owned-process cleanup, original stream
retention and source drift checks remain in force. Missing/skipped/unsupported/
partial/resource-failed outcomes block; expected policy rejections are asserted
causes, not coverage exemptions. C07 absence must remain actual failed evidence
before any minimal production guard addition. Preserve failures and all native logs.
