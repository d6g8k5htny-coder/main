#!/usr/bin/env python3
"""Adversarial probes of scan_module through the source CLI (check.py source) on a temporary git fixture.

Read-only on the repository worktree: runs against the git-archive copy under verify_fable/tree.
Each probe edits ClosedSuperlevel.lean in the fixture, regenerates SOURCE_FILES.json (the
owner-authorized regeneration the reviewer describes), commits, and runs `check.py source`.
A probe is ACCEPTED if the CLI exits 0 (the hidden form passed the scanner) and REFUSED otherwise.
Also runs scan_module directly for comparison. Executes no Lean.
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

TREE = Path(__file__).resolve().parent / 'tree'
SIDE = TREE / 'companions/torus_filtration_20261011'
sys.path.insert(0, str(SIDE))
spec = importlib.util.spec_from_file_location('test_torus_filtration', SIDE / 'test_torus_filtration.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
c = T.checker()
Fixture = T.Fixture

NS = 'namespace TorusFiltration\n'


def after_ns(line):
    """Insert `line` immediately after the namespace opening (the reviewer's G2 position)."""
    return lambda original: original.replace(NS, NS + line + '\n')


def before_end(line):
    return lambda original: original.replace('\nend TorusFiltration', '\n' + line + '\nend TorusFiltration')


def on_decl(suffix):
    """Append `suffix` to the first theorem's declaration line (same-line command form)."""
    return lambda original: original.replace('theorem superlevel_mono : True := trivial',
                                             'theorem superlevel_mono : True := trivial' + suffix)


def replace_def_body(body):
    return lambda original: original.replace('noncomputable def superlevel_set : Nat := 0',
                                             'noncomputable def superlevel_set : Nat := ' + body)


PROBES = [
    # --- the reviewer's exact G2 forms ---
    ('G2 indented alias after namespace', after_ns('  alias hidden := Nat.add_comm')),
    ('G2 same-line alias on a declaration line', on_decl(' alias hidden := Nat.add_comm')),
    ('G2 column-0 alias', before_end('alias hidden := Nat.add_comm')),
    ('alias with anonymous-constructor pattern (indented)', after_ns('  alias ⟨hidden_mp, hidden_mpr⟩ := Iff.rfl')),
    ('alias with deprecated attribute (indented)', after_ns('  @[deprecated] alias hidden := Nat.add_comm')),
    # --- commands absent from COMMANDS (Lean core / Batteries / Mathlib / Aesop at the pin) ---
    ('Mathlib mk_iff_of_inductive_prop (indented)', after_ns('  mk_iff_of_inductive_prop Nat.le hidden_iff')),
    ('Mathlib pp_extended_field_notation (indented)', after_ns('  pp_extended_field_notation superlevel_set')),
    ('Mathlib initialize_simps_projections (indented)', after_ns('  initialize_simps_projections Equiv (toFun → apply)')),
    ('Mathlib assert_exists (indented)', after_ns('  assert_exists Nat')),
    ('Mathlib assert_not_imported (indented)', after_ns('  assert_not_imported Mathlib.Foo')),
    ('Mathlib suppress_compilation (indented)', after_ns('  suppress_compilation')),
    ('Mathlib unset_option in (indented)', after_ns('  unset_option pp.all in def hidden : Nat := 0')),
    ('Mathlib variable? (indented)', after_ns('  variable? (x : Nat)')),
    ('Mathlib register_hint (indented)', after_ns('  register_hint trivial')),
    ('Mathlib register_tactic_tag (indented)', after_ns('  register_tactic_tag hidden_tag')),
    ('Mathlib to_additive via attribute cmd', after_ns('  attribute [to_additive hidden] Nat.add_comm')),
    ('Batteries register_label_attr (indented)', after_ns('  register_label_attr hidden_attr')),
    ('Batteries proof_wanted (indented)', after_ns('  proof_wanted hidden : True')),
    ('Batteries alias via open private', after_ns('  open private hidden from Nat.add_comm')),
    ('Aesop add_aesop_rules (indented)', after_ns('  add_aesop_rules safe [Nat.add_comm]')),
    ('Aesop erase_aesop_rules (indented)', after_ns('  erase_aesop_rules [Nat.add_comm]')),
    ('core builtin_simproc_decl (indented)', after_ns('  builtin_simproc_decl hidden (Nat.succ _) := fun _ => return .continue')),
    ('core builtin_dsimproc_decl (indented)', after_ns('  builtin_dsimproc_decl hidden (Nat.succ _) := fun _ => return .continue')),
    ('core simproc_pattern% (indented)', after_ns('  simproc_pattern% (Nat.succ _) => hidden')),
    ('core register_builtin_option (indented)', after_ns('  register_builtin_option hidden : Bool := { defValue := false }')),
    ('core register_parser_alias (indented)', after_ns('  register_parser_alias hidden')),
    ('core register_error_explanation (indented)', after_ns('  register_error_explanation hidden.err { summary := s!"x" }')),
    ('core declare_config_elab (indented)', after_ns('  declare_config_elab hiddenElab Nat')),
    ('core declare_simp_like_tactic (indented)', after_ns('  declare_simp_like_tactic hiddenSimp "hidden_simp" {}')),
    ('core gen_injective_theorems% (indented)', after_ns('  gen_injective_theorems% Nat')),
    ('core declare_uint_theorems (indented)', after_ns('  declare_uint_theorems UInt8')),
    ('core elab_stx_quot? no; `deriving instance` (indented)', after_ns('  deriving instance Repr for Nat')),
    ('core instance without name (indented)', after_ns('  instance : Inhabited Nat := ⟨0⟩')),
    ('core instance without name (column 0)', before_end('instance : Inhabited Nat := ⟨0⟩')),
    ('core where-clause auxiliary declaration', on_decl('\n  where hidden : True := trivial')),
    ('core where-clause auxiliary declaration, keyword where at col>0 on next line',
     lambda o: o.replace('theorem superlevel_mono : True := trivial', 'theorem superlevel_mono : True := hidden\nwhere\n  hidden : True := trivial')),
    ('core let rec auxiliary declaration in body', replace_def_body('hidden\n  where hidden : Nat := 0')),
    ('core nested namespace inside companion namespace (indented)', after_ns('  namespace Hidden\n  end Hidden')),
    ('core section block (indented)', after_ns('  section\n  end')),
    ('core noncomputable section ... end (indented)', after_ns('  noncomputable section\n  end')),
    ('core open ... in (indented)', after_ns('  open Nat in def hidden : Nat := 0')),
    ('core set_option ... in (indented)', after_ns('  set_option maxHeartbeats 0 in def hidden : Nat := 0')),
    ('core macro expansion (indented)', after_ns('  macro "hidden_cmd" : command => `(def hidden : Nat := 0)')),
    ('core line split: keyword then name on next line (column 0)', before_end('theorem\n  hidden : True := trivial')),
    ('core have/let inside tactic block (not a top-level declaration)', on_decl('\n  -- body replaced below')),
    ('tactic have inside by-block', lambda o: o.replace('theorem superlevel_mono : True := trivial',
                                                        'theorem superlevel_mono : True := by\n  have hidden : True := trivial\n  exact hidden')),
    # --- more core/Mathlib/Batteries commands absent from COMMANDS (found by enumerating `: command` syntax at the pin) ---
    ('core run_meta (addDecl path; run_cmd/run_elab are listed, run_meta is not)',
     after_ns('  run_meta Lean.addDecl (.thmDecl { name := `TorusFiltration.hidden, levelParams := [], type := Lean.mkConst ``True, value := Lean.mkConst ``True.intro })')),
    ('core cbv_simproc (indented)', after_ns('  cbv_simproc hidden (Nat.succ _) := fun _ => return .continue')),
    ('core cbv_simproc_decl (indented)', after_ns('  cbv_simproc_decl hidden (Nat.succ _) := fun _ => return .continue')),
    ('core grind_propagator (indented)', after_ns('  grind_propagator hidden ↓ (Nat.succ) := fun _ => return ()')),
    ('core register_linter_set (indented)', after_ns('  register_linter_set hidden := linter.all')),
    ('core reprove (indented)', after_ns('  reprove Nat.add_comm by omega')),
    ('core seal (indented)', after_ns('  seal superlevel_set')),
    ('core with_weak_namespace prefix (indented)', after_ns('  with_weak_namespace Hidden def hidden : Nat := 0')),
    ('core public theorem (column 0)', before_end('public theorem hidden : True := trivial')),
    ('core public theorem (indented)', after_ns('  public theorem hidden : True := trivial')),
    ('core meta def (column 0)', before_end('meta def hidden : Nat := 0')),
    ('core declare_config_elab_legacy (indented)', after_ns('  declare_config_elab_legacy hiddenElab Nat')),
    ('core elab_stx_quot (indented)', after_ns('  elab_stx_quot hidden')),
    ('core declare_eval_bin (indented)', after_ns('  declare_eval_bin hidden Nat.add')),
    ('core gen_injective_theorems% on companion-free type (indented)', after_ns('  gen_injective_theorems% Option')),
    ('Mathlib name_poly_vars (indented)', after_ns('  name_poly_vars X over Nat')),
    ('Mathlib to_dual_insert_cast (indented)', after_ns('  to_dual_insert_cast hidden := Nat.add_comm')),
    ('Mathlib assert_no_sorry (indented)', after_ns('  assert_no_sorry superlevel_set')),
    ('Mathlib extend_docs (indented)', after_ns('  extend_docs superlevel_set')),
    ('Mathlib deprecate to prefix (indented)', after_ns('  deprecate to hidden\n  theorem hidden2 : True := trivial')),
    # --- the same lookbehind class guards ADMISSION; report only (Lean-side viability of `0sorry` is not asserted) ---
    ('digit-prefixed sorry in a def body (ADMISSION lookbehind shares the digit class)', replace_def_body('0sorry')),
    # --- lexical tricks against COMMAND_TOKEN ---
    ('digit-prefixed alias on a def body line (Lean lexes `0alias` as numeral then keyword)', replace_def_body('0alias hidden := Nat.add_comm')),
    ('digit-prefixed theorem on a def body line', replace_def_body('0theorem hidden : True := trivial')),
    ('digit-prefixed alias indented after namespace', after_ns('  0alias hidden := Nat.add_comm')),
    ('unicode-letter-prefixed alias (Lean: one identifier, so not the command)', after_ns('  αalias hidden := Nat.add_comm')),
    ('guillemet identifier «alias» (Lean: identifier, not the command)', after_ns('  def «alias» : Nat := 0')),
    ('tab-indented alias', after_ns('\talias hidden := Nat.add_comm')),
    ('NBSP-indented alias (Python isspace True)', after_ns(' alias hidden := Nat.add_comm')),
    ('alias with zero-width space inside keyword', after_ns('  al​ias hidden := Nat.add_comm')),
    ('alias split by nested comment', after_ns('  alias/- -/hidden := Nat.add_comm')),
    ('alias behind a comment-closer trick', after_ns('  /- -/ alias hidden := Nat.add_comm')),
    ('alias after `--` in a string? (strings refused anyway)', after_ns('  def hidden : String := "-- alias x := y"')),
    ('alias preceded by a dot (Lean: dotted ident Foo.alias is an ident)', after_ns('  Nat.alias hidden := Nat.add_comm')),
    ('alias preceded by underscore (Lean: _alias is an ident)', after_ns('  _alias hidden := Nat.add_comm')),
    ('alias followed by prime (Lean: alias\' is an ident)', after_ns('  alias\' hidden := Nat.add_comm')),
    ('attribute via @[ (refused by @[ rule)', after_ns('  @[simp] def hidden : Nat := 0')),
    ('#-command (refused by # rule)', after_ns('  #eval 1')),
]


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        f = Fixture(c, tmp)
        module = f.side / 'ClosedSuperlevel.lean'
        original = module.read_text()
        baseline = f.run('source')
        assert baseline.returncode == 0, baseline.stderr
        for label, mutate in PROBES:
            text = mutate(original)
            if text == original:
                results.append((label, 'NOOP (mutation did not apply)', ''))
                continue
            module.write_text(text)
            f.restore()
            run = f.run('source')
            try:
                c.scan_module(text, 'ClosedSuperlevel')
                direct = 'scan_module ACCEPTED'
            except Exception as e:   # noqa: BLE001 - report the refusal text
                direct = 'scan_module refused: ' + str(e)[:110]
            verdict = 'ACCEPTED' if run.returncode == 0 else 'REFUSED'
            msg = ''
            for line in run.stderr.splitlines():
                if 'TORUS_FILTRATION_FAIL' in line:
                    msg = line.strip()[:150]
            results.append((label, verdict + ' rc=%d' % run.returncode, msg or direct))
        module.write_text(original)
        f.restore()
        assert f.run('source').returncode == 0
    width = max(len(r[0]) for r in results)
    for label, verdict, msg in results:
        print('%-*s | %-16s | %s' % (width, label, verdict, msg))
    print('\noptimize flag of this interpreter:', sys.flags.optimize)
    print('child_python():', T.child_python())


if __name__ == '__main__':
    main()
