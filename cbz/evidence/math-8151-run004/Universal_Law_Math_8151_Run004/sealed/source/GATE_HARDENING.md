# Gate hardening RF-GATE-01: environment inventory instead of a source scan

Scientific effect: **NONE**. Dylan Roy, delegated AI work (Anthropic / Claude, Claude
Code session `session_015wNj8LPTKXsaT68G3DgPPh`). No Lean proof, manifest target,
`SCOPE.md` or status entry changes; `formalization_status` stays `proved` and
`alignment_status` stays `PENDING_INDEPENDENT_REVIEW`.

## The gap

Before this change `gate.py` built its target inventory by scanning the source modules
for lines beginning, at column 0, with `theorem` or `lemma`, and compared that list with
`manifest.json["targets"]`. Everything downstream (the `#print axioms` audit, the
elaborated-type dump, the Blueprint link check) ran on that list. A declaration the
scanner did not see was therefore never audited:

- an indented `theorem`, a `private theorem`, a `protected theorem`, or a `@[simp] theorem`
  on the same line;
- a `def`, `instance`, `opaque` or `abbrev` whose body uses `sorry` but is not a target;
- an `axiom` that no target mentions;
- a theorem keyword hidden in a block comment that the scanner mis-counted.

Such a declaration compiles (Lean only warns on `sorry`), lands in the package's `.olean`,
and is exported to every importer, while the gate reports the manifest targets as clean.
Finding RF-GATE-01 (main#229).

## The fix: two layers, the second authoritative

1. **Grammar pre-check on source (`grammar_check`).** The stripper first lexes string
   literals (with escapes), raw strings and character literals, so a `--` or `/-` inside a
   literal is not a comment; literal contents are kept, so refusals see them. Interpolated
   strings (`s!"…"`, and any plain string containing a brace, because macros such as
   `throwError` interpolate plain literals), unterminated literals and unterminated block
   comments are refused. Comments are then removed (nesting-aware), keeping newlines; a
   block comment becomes one space, so `opaque/-x-/hidden` stays two tokens. On the
   remaining code:
   - every line with a `theorem`/`lemma`/`def` keyword must be the one canonical column-0
     declaration on that line (`theorem <name>`, `lemma <name>`, `def <name>` or
     `noncomputable def <name>`);
   - declaration-bearing, metaprogramming and elaborator tokens are refused **anywhere** on a
     line, so wrappers such as `set_option … in run_elab`, `open Lean in …` and term-level
     `by_elab` are caught (`private`, `protected`, `opaque`, `abbrev`, `structure`, `class`,
     `instance`, `inductive`, `mutual`, `syntax`, `macro`, `elab`, `run_elab`, `run_cmd`,
     `run_meta`, `run_tac`, `by_elab`, `initialize`, `example`, `attribute`, `notation`,
     `section`, `Lean`, `MetaM`, `CoreM`, `addDecl`, every `#` command, and more); `local`
     and `scoped` are refused at the start of a line (`open scoped` stays allowed);
   - each module has exactly one column-0 `namespace ResearchFormalCoreR1` … `end
     ResearchFormalCoreR1` block, so a nested namespace cannot be mis-prefixed;
   - no declaration name repeats, and the manifest target list has no duplicates;
   - the tokens `sorry`, `axiom`, `native_decide`, `implemented_by`, `extern` and `unsafe`
     are refused outright, including inside string literals.

   This is a pre-check: it keeps the source legible for the rest of the pipeline and fails
   early, but it is not what the gate trusts. Admission does not depend on it (see the bound
   compiled inventory below).

2. **Environment inventory at execute time (`inventory_source`, `audit_inventory`).**
   After `lake build`, the gate writes and runs `Inventory.lean`, a Lean program that
   enumerates, from the compiled environment's own module tables
   (`env.header.moduleData[idx].constNames` for every module under the package prefix),
   **every** constant of every package module: public, private, compiler-generated, used
   or unused. For each constant it prints one `INV` line with its kind, its **provenance**
   and its module and name; one `MODULE` line per module with the module-system flag and
   constant counts; and then it axiom-closes all package constants together in one shared
   traversal that replicates `Lean.CollectAxioms.collect` (axiom: record and visit its
   type; def/theorem/opaque: type and value; quot: nothing; ctor/rec: type; inductive: type
   and constructors). The union is printed as one `AXIOMS` line.

   - **Provenance** is Lean's own record, `findDeclarationRanges?`: `user` when Lean stored
     a source range for the constant (every user-written declaration, private or not), `aux`
     when it did not (compiler-generated constants). On the current package this splits the
     116 constants into 63 `user` theorems (exactly the targets), 10 `user` definitions and
     43 `aux` theorems (`_proof_N`, `_simp_N_M`, `eq_N`).
   - **Names are encoded losslessly.** Each name is printed as its components joined by
     `/`, with `s<hex of UTF-8>` for a string component and `n<decimal>` for a numeric one.
     `Name.toString` is never parsed, so private names such as
     `_private.«.lake».«formal-evidence».hidden_sorry.0.ResearchFormalCoreR1.hiddenAdmission`,
     quoted components and names that differ only by escaping all stay distinct and
     parseable. Decoding is strict (canonical lowercase hex, valid UTF-8, canonical decimal).

   `audit_inventory` first parses the transcript strictly; any malformed line, a missing
   or repeated `AXIOMS`/`CLOSURE` record, or an empty inventory is an
   `InventoryProtocolError`, never a rejection. It then collects **every** policy finding
   as a `(code, name)` pair and raises `InventoryRejected` with the complete list:
   - `unregistered_module`, `missing_module`, `duplicate_module`, `module_system`,
     `root_not_empty`, `codegen_extra`: the module table is exactly the registered
     `source_modules` plus the empty root, none compiled under the module system, no
     code-generator extras;
   - `outside_module`, `duplicate_name`, `count_mismatch`, `package_axiom`: every constant
     sits in a registered module, names are unique, per-module counts match the table, and
     no constant is itself an `axiom`;
   - `extra_theorem`, `missing_theorem`: the `user` theorem-kind constants, private ones
     included, are exactly the manifest targets, whatever their names look like;
   - `unregistered_constant`, `missing_constant`: **the bound compiled inventory.**
     `manifest.json["compiled_inventory"]` lists, as reviewed `[kind, origin, module, name]`
     records, every constant of the package modules that is not a target theorem: today the
     10 user-written definitions and the 43 compiler-generated theorems. Every such constant
     in the compiled environment must match one record exactly (kind and provenance
     included), and every record must be present. A theorem a metaprogram adds without a
     declaration range, a definition, an `opaque` or anything else not reviewed into this
     list is rejected whatever its name, parent or provenance. The source gate checks the
     records' shape and that their user-written definitions are exactly the source's
     canonical `def` declarations. Adding a declaration therefore means updating this list
     in a reviewed manifest change, alongside `targets`;
   - `unbound_auxiliary`: in addition, every `aux` theorem-kind constant hangs below a `user`
     constant of the same module through compiler suffix components only (`_proof_N`,
     `_simp_N_M`, `eq_N`, `eq_def`, `match_N`). The name shape is used only together with
     Lean's provenance, a user-written parent and the bound list, never on its own;
   - `forbidden_axiom`, `reported_axiom_omitted`, `impossible_closure`: the axiom closure of
     the whole package is inside `{propext, Classical.choice, Quot.sound}` (so a `sorry`
     anywhere, in any kind of declaration, appears as `sorryAx`), contains every axiom the
     per-target `#print axioms` audit reported, and is at least as large as the inventory.

   Union-of-closure is equivalent to the per-constant condition: every constant's axioms
   are inside the allowed set if and only if their union is.

3. **Admission controls, each bound to its own reason (`check_admission`).** The execute
   path compiles four files that each append one declaration the old scanner could never
   see to the inventory program (with the current file's constants included):

   | Control | Injected declaration | Required findings, exactly |
   |---|---|---|
   | `hidden_sorry` | `private theorem hiddenAdmission … := by sorry` | `forbidden_axiom sorryAx` and `extra_theorem` (the private name) |
   | `hidden_theorem` | indented `@[simp] theorem indentedAdmission … := rfl` | `extra_theorem ResearchFormalCoreR1.indentedAdmission` |
   | `hidden_def` | unused `noncomputable def unusedAdmission … := by sorry` | `forbidden_axiom sorryAx` |
   | `outside_namespace` | column-0 `theorem outsideNamespaceAdmission` outside the namespace | `extra_theorem outsideNamespaceAdmission` |

   For each, the transcript must parse, the injected constant must appear exactly once with
   the expected kind, provenance, privacy and module, and the findings must equal the
   required ones by FULL decoded name and multiplicity: the short user-facing name only
   locates the injection, so an unrelated private constant with the same short name in
   another module is an extra finding and fails the control. A protocol error, a pass, a missing injection or any other set of findings
   fails the run; there is no "any `ValueError` counts" path.

4. **Compiled-package admission experiments (`compiled_admission`).** The controls above
   inject into the inventory program's own file. The gate also copies the package (sources
   plus the freshly built `.lake/build`, dependencies by symlink) into
   `.lake/compiled-admission`, outside the uploaded evidence directory, and audits the
   unmodified copy (it must pass and reproduce the real package's inventory digest, the
   positive baseline). Then, one at a time, it adds a declaration to
   `ResearchFormalCoreR1/AlgebraV2.lean` and runs the same stages as the real package on the
   modified copy: `lake build`, `leanchecker`, and the per-target `#print axioms` audit, each
   of which must succeed (an unrelated failure fails the run and is never counted as a
   rejection), and finally the inventory, which must contain the injected constant and
   reject with exactly the expected findings:

   | Experiment | Injected declaration | Required findings, exactly |
   |---|---|---|
   | `compiled_admission` | `private theorem compiledAdmission : (1 : Nat) = 1 := by sorry` | `forbidden_axiom sorryAx`, `extra_theorem` (private name) |
   | `compiled_metaprogram` | `set_option maxRecDepth 1000 in run_elab` + `Lean.addDecl` of `ResearchFormalCoreR1.ec005_fold_gap._proof_999 : True` | `unregistered_constant ResearchFormalCoreR1.ec005_fold_gap._proof_999` |

   The second is the compiled RF312-SUCCESSOR-COMMAND-004 probe (#312, 6009707611): Lean
   stores no declaration range for it, so it is `aux`, and it has a real user parent and a
   compiler-shaped suffix; only the bound list rejects it. The copy is removed afterwards;
   each experiment's four logs and the injected module's sha256 stay in the evidence.

## History of the admission evidence

The first helper revision (`dec3f646`) printed names with `Name.toString` and credited any
`ValueError` from `audit_inventory` as a rejection. Replaying its real logs showed 3 of 4
controls rejected for their intended reason; `hidden_sorry` stopped earlier, at a
malformed-line error on the quoted private name, although `sorryAx` was in that log's
closure (PR #312 comments 6008555025 and 6008954175). That defect and the name-shaped
auxiliary exemption (which let a user-written `ResearchFormalCoreR1._unregistered` theorem
pass, 6009106591) were repaired by the successor `9aee8fd1`. The historical transcript shape
is kept as a regression: it is a protocol error, and `check_admission` does not count it.

The source audit of `9aee8fd1` (#312, 6009593567) found two more gaps, both repaired here:
RF312-SUCCESSOR-COMMAND-004, wrapped and term-level elaborators passed the line-anchored
pre-check, and a theorem added by `Lean.addDecl` was then admitted as a range-less
auxiliary under a real parent (confirmed compiled in 6009707611); and
RF312-COMPILED-RECHECK-005, the modified package copy was not rechecked by `leanchecker` or
the axiom audit. The audit of `96606a6d` (6014639706) found RF312-CONTROL-PRIVATE-IDENTITY-006:
`check_admission` compared findings after shortening private names, so a second private
`compiledAdmission` in another module collapsed into the expected finding. Findings are now
compared by full name and multiplicity.

## What the receipt gains

`receipt.json` has an `inventory` object: number of package constants, number of public
theorems, number of compiler-generated theorems, the axiom closure, the closure size, a
sha256 of the inventory lines, the per-control outcomes with their exact findings, and the
compiled-package experiments' baseline digest and, per injection, outcome, findings,
module, module sha256 and the stages run.
`required_formal_check.py` binds fixed keys and tolerates the extra object. All earlier
receipt fields are unchanged.

## Cost and measurements (local, Lean 4.34.1, Mathlib `d13f23b7`)

| Step | Wall time |
|---|---|
| Inventory via whole-environment scan (`env.constants.map₁`) | about 8 minutes (interpreter) |
| Inventory via module tables, with provenance and encoding (shipped) | about 11 seconds |
| Four admission controls plus both compiled-package experiments (local, `leanchecker` skipped) | about 2 minutes |

The shared closure of all 116 package constants visits 26 853 constants.

## Limits

- Provenance is what Lean records, and a missing range is not proof of compiler origin. The
  gate therefore does not rely on it alone: every non-target constant must be in the
  reviewed `compiled_inventory`, so a metaprogram-added constant is rejected even if it
  slips past the pre-check. The cost is that the list must be refreshed, in review, when a
  proof change makes Lean generate different auxiliaries.
- The inventory covers constants that reach the environment through the package's
  modules. Constants a downstream file realizes lazily (reserved names such as `eq_1`)
  belong to that downstream file, not to the package.
- `extraConstNames` (code-generator auxiliaries) must be empty for every package module;
  a module that needs compiled code therefore fails this gate on purpose. Inductive types,
  structures and instances are refused by the pre-check and would need a reviewed
  extension of the auxiliary suffixes before they could be admitted.
- `leanchecker` replays the package, and each modified copy, in the execute path; on one
  local 15 GB container it was killed for memory (exit 137), so local precommit runs of
  this helper skipped those stages and say so. Complete executions with `leanchecker` exit 0
  are recorded separately.
- The gate still does not decide mathematics. A clean inventory says the package's
  compiled declarations are exactly the registered ones and depend on nothing beyond the
  three standard axioms; alignment with the prose remains a separate human/nonauthor
  review.

## Tests

`tests/test_inventory_gate.py` covers the string-aware stripper (comment markers inside
strings, escapes, character and raw literals, refused interpolation and unterminated
literals), the grammar pre-check (spellings, unsupported commands, namespace structure,
duplicates, forbidden tokens), the lossless name encoding (round trips, collisions that a
dotted rendering would make, strict decoding), the generated Lean program's shape,
`audit_inventory` on synthetic inventories (every finding code, the quoted private `sorry`
case, extra user theorems whatever their names, auxiliary binding, the bound compiled
inventory including the RF312-SUCCESSOR-COMMAND-004 probe, protocol errors),
`bound_inventory` validation and the live manifest binding, and `check_admission` (expected
reason, pass, other reason, protocol error, missing or misplaced injection). The pre-check
tests include every wrapped, nested, tactic- and term-level form reported in 6009593567. Sol's PR #312 source-gate regression cases (indented, private, attributed,
unused `sorry` definition, `opaque`, `axiom`, comment ghost, duplicate target, nested
namespace) are replayed against a copied formal tree. The live modules are checked to pass
the grammar and to reproduce `manifest.json["targets"]`.
