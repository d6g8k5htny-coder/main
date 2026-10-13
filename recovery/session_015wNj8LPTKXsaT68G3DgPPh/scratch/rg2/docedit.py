from pathlib import Path
p = Path('GATE_HARDENING.md'); s = p.read_text()
def rep(a, b):
    global s
    assert a in s, a[:70]; s = s.replace(a, b)
rep("""   - declaration-bearing and metaprogramming commands the scanner does not inventory are
     refused at the start of any line (`private`, `protected`, `opaque`, `abbrev`,
     `structure`, `class`, `instance`, `inductive`, `mutual`, `syntax`, `macro`, `elab`,
     `initialize`, `example`, `attribute`, `notation`, `local`, `scoped`, `section`, `run_cmd`,
     and every `#` command);""", """   - declaration-bearing, metaprogramming and elaborator tokens are refused **anywhere** on a
     line, so wrappers such as `set_option … in run_elab`, `open Lean in …` and term-level
     `by_elab` are caught (`private`, `protected`, `opaque`, `abbrev`, `structure`, `class`,
     `instance`, `inductive`, `mutual`, `syntax`, `macro`, `elab`, `run_elab`, `run_cmd`,
     `run_meta`, `run_tac`, `by_elab`, `initialize`, `example`, `attribute`, `notation`,
     `section`, `Lean`, `MetaM`, `CoreM`, `addDecl`, every `#` command, and more); `local`
     and `scoped` are refused at the start of a line (`open scoped` stays allowed);""")
rep("""   This is a pre-check: it keeps the source legible for the rest of the pipeline and fails
   early, but it is not what the gate trusts.""", """   This is a pre-check: it keeps the source legible for the rest of the pipeline and fails
   early, but it is not what the gate trusts. Admission does not depend on it (see the bound
   compiled inventory below).""")
rep("""   - `unbound_auxiliary`: every `aux` theorem-kind constant hangs below a `user` constant
     of the same module through compiler suffix components only (`_proof_N`, `_simp_N_M`,
     `eq_N`, `eq_def`, `match_N`). The name shape is used only in conjunction with Lean's
     provenance and a user-written parent, never on its own;""", """   - `unregistered_constant`, `missing_constant`: **the bound compiled inventory.**
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
     Lean's provenance, a user-written parent and the bound list, never on its own;""")
start = s.index("4. **Compiled-package admission experiment (`compiled_admission`).**")
end = s.index("## History of the admission evidence")
s = s[:start] + """4. **Compiled-package admission experiments (`compiled_admission`).** The controls above
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

""" + s[end:]
rep("""pass, 6009106591) are what this revision repairs. The historical transcript shape is kept as
a regression: it is a protocol error, and `check_admission` does not count it.""", """pass, 6009106591) were repaired by the successor `9aee8fd1`. The historical transcript shape
is kept as a regression: it is a protocol error, and `check_admission` does not count it.

The source audit of `9aee8fd1` (#312, 6009593567) found two more gaps, both repaired here:
RF312-SUCCESSOR-COMMAND-004, wrapped and term-level elaborators passed the line-anchored
pre-check, and a theorem added by `Lean.addDecl` was then admitted as a range-less
auxiliary under a real parent (confirmed compiled in 6009707611); and
RF312-COMPILED-RECHECK-005, the modified package copy was not rechecked by `leanchecker` or
the axiom audit.""")
rep("""sha256 of the inventory lines, the per-control outcomes with their exact findings, and the
compiled-package experiment's outcome, module and baseline digest.""", """sha256 of the inventory lines, the per-control outcomes with their exact findings, and the
compiled-package experiments' baseline digest and, per injection, outcome, findings,
module, module sha256 and the stages run.""")
rep("""| Four admission controls plus the compiled-package experiment | about 75 seconds |""", """| Four admission controls plus both compiled-package experiments (local, `leanchecker` skipped) | about 2 minutes |""")
rep("""- Provenance is what Lean records. A metaprogram that adds a theorem without a declaration
  range would be classified `aux`; it must still hang below a user-written constant through
  compiler suffixes, and its axioms are still in the closure. The pre-check refuses every
  `#` command, `run_cmd`, `elab`, `macro` and `initialize`, so the package sources cannot
  contain such a metaprogram.""", """- Provenance is what Lean records, and a missing range is not proof of compiler origin. The
  gate therefore does not rely on it alone: every non-target constant must be in the
  reviewed `compiled_inventory`, so a metaprogram-added constant is rejected even if it
  slips past the pre-check. The cost is that the list must be refreshed, in review, when a
  proof change makes Lean generate different auxiliaries.""")
rep("""- `leanchecker` replays the package in the execute path; on one local 15 GB container it
  was killed for memory (exit 137), so local precommit runs of this helper skipped that
  stage and say so. A complete execution with `leanchecker` exit 0 is recorded separately.""", """- `leanchecker` replays the package, and each modified copy, in the execute path; on one
  local 15 GB container it was killed for memory (exit 137), so local precommit runs of
  this helper skipped those stages and say so. Complete executions with `leanchecker` exit 0
  are recorded separately.""")
rep("""case, extra user theorems whatever their names, auxiliary binding, protocol errors), and
`check_admission` (expected reason, pass, other reason, protocol error, missing or misplaced
injection).""", """case, extra user theorems whatever their names, auxiliary binding, the bound compiled
inventory including the RF312-SUCCESSOR-COMMAND-004 probe, protocol errors),
`bound_inventory` validation and the live manifest binding, and `check_admission` (expected
reason, pass, other reason, protocol error, missing or misplaced injection). The pre-check
tests include every wrapped, nested, tactic- and term-level form reported in 6009593567.""")
p.write_text(s)
r = Path('README.md'); t = r.read_text()
for a, b in (("requires the user-written theorems to be exactly the manifest targets and every compiler-generated theorem to belong to a user-written parent,",
              "requires the user-written theorems to be exactly the manifest targets and every other constant to be one of the manifest's reviewed `compiled_inventory` records,"),
             ("and injects a private `sorry` theorem into a compiled copy of the package that must be rejected the same way;",
              "and injects a private `sorry` theorem and a metaprogram-added theorem into a compiled copy of the package, which is rebuilt, rechecked and axiom-audited before its inventory must reject each one;")):
    assert a in t, a[:50]; t = t.replace(a, b)
r.write_text(t)
print('ok')
