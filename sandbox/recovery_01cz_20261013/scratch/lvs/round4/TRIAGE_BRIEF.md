# Round 4 triage brief: which registered Props without a theorem are provable as stated?

Scientific effect of everything here: NONE. You are classifying formal statements for provability against a pinned
Mathlib; you decide no mathematical status, review nothing, accept nothing. Organizational-independence credit: 0.

## What exists

- Statement packet (the Props): $S/repos/math/specifications/lean_statements_20261010/Specifications/{Field,Lifetime,Side24,RN,P15}.lean
  (import names `Specifications.<Module>`), with S=<SCRATCH>.
  Every `def … : Prop` has a docstring with `Source:`, `Anchor:`, `Interface:` and `Does not claim:`. Interface
  `structure`s in the same modules carry the given objects (fields are hypotheses/given data, not constructions).
- The 82 Props still without a theorem, with their full Lean source (docstring + def), are listed in
  $S/lvs/round4/remaining_props.json (fields: module, declaration, claims, source). Read the module itself for the
  interface structures and auxiliary definitions the Prop uses.
- Proof packet (60 theorems already proved; style reference and reusable Aux lemmas):
  $S/repos/math/specifications/lean_statement_proofs_20261011/SpecificationProofs/*.lean and its README.md
  (table "Registered Props without a theorem here, and why" gives the previous, informal reasons; test them, do not
  copy them).
- Pinned Mathlib source (grep freely): $S/mathf/.lake/packages/mathlib/Mathlib ; Lean v4.34.1, Mathlib d13f23b7.
- Pinned informal sources: $S/repos/math/specifications/lean_statements_20261010/sources/<claim-id>/ (byte-pinned;
  read-only; cite by path and line).
- Toolchain: export PATH=$HOME/.elan/bin:$PATH. To test whether a Mathlib lemma exists or a statement typechecks,
  write a probe file under $S/lvs/round4/probes/<yourtag>_<n>.lean (imports `Specifications.<Module>` and Mathlib
  modules you need; use `#check`, `example : … := by sorry` ONLY in probe files, never in anything else) and run,
  from the proof packet directory:
      cd $S/repos/math/specifications/lean_statement_proofs_20261011 && lake env lean $S/lvs/round4/probes/<file>.lean
  Each run loads ~1.8 GB of oleans (30-120 s). Run at most one `lake env lean` at a time. Do not run `lake build`,
  `lake update`, `lake exe cache`, and do not write anything under $S/repos or $S/mathf.

## The question, per Prop

"Is there a `theorem t : <Prop applied to exactly its own binders>` that Lean v4.34.1 + Mathlib d13f23b7 can
check, with no extra hypothesis, no narrowed argument, no axiom beyond propext/Classical.choice/Quot.sound, no
sorry, no native_decide, and no edit to the statement packet?" Classify each Prop with exactly one verdict:

- PROVABLE_NOW: a concrete route exists with named Mathlib lemmas (verified to exist by grep or probe) and the
  packet's Aux lemmas; estimated effort small (≤ 1 agent-day); give the route in 3-8 sentences.
- PROVABLE_WITH_WORK: no missing theory, but a substantial development is needed (e.g. a page of real analysis,
  an explicit computation, a sizable induction); give the route and the hard step.
- BLOCKED_MISSING_MATHLIB: provable in principle but a specific result is absent from the pinned Mathlib (name
  the closest existing declarations and exactly what is missing; e.g. "no closed form for iteratedFDeriv of a
  Gaussian in arbitrary directions"; "Jacobi theta only in one dimension").
- BLOCKED_INTERFACE: the Prop quantifies over an interface structure whose given fields do not determine the
  conclusion (the Prop could be false for some inhabitant); state the field(s) and why. Distinguish carefully
  from an interface whose fields DO imply the conclusion (then it is PROVABLE_*).
- FALSE_OR_UNDERDETERMINED_AS_STATED: you found a concrete inhabitant/instance making the Prop false, or a
  definitional degeneracy (e.g. Set.ncard of an infinite set = 0, division by zero conventions) under which it
  fails. This is a FINDING about the statement: describe it precisely; never propose to edit the statement in
  the proof packet; the register's maintainers record findings.
- PREDICATE_VOCABULARY: a parametrised predicate with free arguments (`def IsX (…) : Prop`), the vocabulary of other Props; a theorem
  of its type would assert it for every argument, which is false or meaningless; not a target. (Check: does the def have
  explicit non-instance arguments beyond type/instance binders, and is it used as a hypothesis or field elsewhere?)
- VACUOUS_OR_TRIVIAL: provable for a degenerate reason (e.g. a hypothesis field is contradictory, or the Prop
  reduces to `True`); say so, because a theorem of it would be misleading evidence and the register should know.

Default to the more cautious verdict when unsure, and say what probe would settle it.

## Hard rules

- Never edit any file under $S/repos or $S/mathf. Never run anything that writes there.
- Do not propose weakening, strengthening or rewording a Prop; the only object is the Prop as it stands.
- Quote the Lean text you rely on (interface fields, definitions) by file and line.
- Vocabulary: a Prop is "stated", a theorem "proves the Prop as stated", a run receipt "kernel-checks". Never
  write "verified", "closed", "established", "accepted" about any informal claim.

## Output (StructuredOutput; one entry per Prop assigned to you, no omissions)

verdicts: [{declaration, module, verdict, route_or_blocker (3-12 sentences), mathlib_names_checked
([names you confirmed exist, with file paths]), interface_fields_relied_on, estimated_effort
("hours"/"days"/"weeks"/"n/a"), probe_files ([paths you created]), finding_for_register (string or "")}],
notes (anything cross-cutting; e.g. a Mathlib area that unlocks several Props).
