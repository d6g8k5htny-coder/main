"""Adversarial module texts through `check.py source` on the test module's own Fixture (archive copy)."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

TREE = Path(sys.argv[1])
SIDE = TREE / 'companions/torus_filtration_20261011'
spec = importlib.util.spec_from_file_location('tft', SIDE / 'test_torus_filtration.py')
tft = importlib.util.module_from_spec(spec)
sys.argv = [str(SIDE / 'test_torus_filtration.py')]
spec.loader.exec_module(tft)
c = tft.checker()
good = tft.placeholder_module('ClosedSuperlevel')


def inject(line):
    return good.replace('\nend TorusFiltration', '\n' + line + '\nend TorusFiltration')


def body(line):
    return good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := ' + line)


def after_ns(line):
    return good.replace('namespace TorusFiltration\n', 'namespace TorusFiltration\n' + line + '\n')


PROBES = [
    # --- numeral forms Lean lexes as one numeral token followed by a keyword (expect: scanner ACCEPTS = gap)
    ('hex 0x1Ftheorem', body('0x1Ftheorem hidden : True := trivial')),
    ('hex 0x1Ftheorem proof_1 (internal name)', body('0x1Ftheorem proof_1 : True := trivial')),
    ('sci 1e5theorem', body('1e5theorem hidden : True := trivial')),
    ('sci 1E5theorem', body('1E5theorem hidden : True := trivial')),
    ('bin 0b1def', body('0b1def hidden : Nat := 0')),
    ('oct 0o7instance', body('0o7instance hidden : Inhabited Nat := ⟨0⟩')),
    ('sep 1_0theorem', body('1_0theorem hidden : True := trivial')),
    ('bin 0b1axiom proof_1 (hidden axiom, internal name)', body('0b1axiom proof_1 : False')),
    ('sci 1e5axiom hidden', body('1e5axiom hidden : False')),
    ('hex 0x1Fset_option maxHeartbeats 0', body('0x1Fset_option maxHeartbeats 0')),
    ('bin 0b1attribute [simp] superlevel_set', body('0b1attribute [simp] superlevel_set')),
    ('hex 0x1Fprivate theorem (second keyword still a token)', body('0x1Fprivate theorem hidden : True := trivial')),
    ('open Lean.Elab.Command + 0x1Fprotected aux_def', good.replace('import Mathlib\n', 'import Mathlib\nopen Lean.Elab.Command\n')
     .replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0x1Fprotected aux_def hidden : Nat := 42')),
    # --- controls that must still be REFUSED
    ('dec 0alias (control)', body('0alias hidden := Nat.add_comm')),
    ('dec 0theorem (control)', body('0theorem hidden : True := trivial')),
    ('dec 1.theorem (Lean error anyway)', body('1.theorem hidden : True := trivial')),
    ('dec 1.5theorem', body('1.5theorem hidden : True := trivial')),
    ('dec 1e-5theorem', body('1e-5theorem hidden : True := trivial')),
    ('nested /- /- -/ -/ splice', inject('  theorem/- /- x -/ -/hidden : True := trivial')),
    ('multi-line block splice', inject('  theorem/-\n-/hidden : True := trivial')),
    ('doc-comment splice /-- -/', inject('  theorem/-- d -/hidden : True := trivial')),
    ('-- at line end then indented keyword', inject('theorem a_hidden : True := trivial -- x\n  theorem hidden : True := trivial')),
    ('-- hides /- opener, next line keyword', inject('theorem a_hidden : True := trivial -- /-\n  theorem hidden : True := trivial')),
    (')theorem', body('(0)theorem hidden : True := trivial')),
    (']theorem', body('[0].length]theorem hidden : True := trivial')),
    (',theorem', body('(0,theorem hidden : True := trivial')),
    ('«theorem» (Lean: escaped ident; scanner: safe false refusal)', inject('  «theorem» hidden : True := trivial')),
    ('x.«theorem»', body('x.«theorem»')),
    ('.theorem adjacency', body('0 .theorem hidden : True := trivial')),
    ('unicode αtheorem (Lean ident; safe false refusal)', body('αtheorem')),
    ('unicode theoremα', body('theoremα')),
    ('NBSP-indented theorem', inject(' theorem hidden : True := trivial')),
    ('zero-width-space col0 theorem', inject('​theorem hidden : True := trivial')),
    ('col-0 block-comment prefix /- c -/theorem (now refused: safe direction)',
     good.replace('theorem superlevel_mono : True := trivial', '/- c -/theorem superlevel_mono : True := trivial')),
    ('_theorem identifier (Lean ident)', body('_theorem')),
    ("theorem' identifier (Lean ident)", body("theorem'")),
    ('theorem.hidden dotted identifier (Lean ident)', body('theorem.hidden')),
    ('theorem?', body('theorem?')),
    ('1.theorem in where', inject('theorem a_hidden : True := trivial\n  where x : Nat := 1.theorem')),
    # --- forms that produce declarations and are not commands (expect: scanner ACCEPTS; census decides)
    ('where hidden (non-internal: census refuses)', good.replace('theorem superlevel_mono : True := trivial',
                                                                 'theorem superlevel_mono : True := trivial\n  where hidden : True := trivial')),
    ('where proof_1 (internal: census excludes)', good.replace('theorem superlevel_mono : True := trivial',
                                                                'theorem superlevel_mono : True := trivial\n  where proof_1 : True := trivial')),
    ('where _hidden (internal)', good.replace('theorem superlevel_mono : True := trivial',
                                              'theorem superlevel_mono : True := trivial\n  where _hidden : True := trivial')),
    ('where eq_ (internal: empty suffix)', good.replace('theorem superlevel_mono : True := trivial',
                                                        'theorem superlevel_mono : True := trivial\n  where eq_ : True := trivial')),
    ('where match_1 omega_2 eq_3', good.replace('theorem superlevel_mono : True := trivial',
                                                'theorem superlevel_mono : True := trivial\n  where\n    match_1 : True := trivial\n    omega_2 : True := trivial\n    eq_3 : True := trivial')),
    ('where «proof_1»', good.replace('theorem superlevel_mono : True := trivial',
                                     'theorem superlevel_mono : True := trivial\n  where «proof_1» : True := trivial')),
    ('where eq_def (NOT internal)', good.replace('theorem superlevel_mono : True := trivial',
                                                 'theorem superlevel_mono : True := trivial\n  where eq_def : True := trivial')),
    # --- absent commands reachable without a refused keyword
    ('register_parser_alias indented', after_ns('  register_parser_alias hidden Lean.Parser.ident')),
    ('initialize_simps_projections indented', after_ns('  initialize_simps_projections Equiv (toFun → apply)')),
    ('lrat_proof (string-guarded)', after_ns('  lrat_proof hidden "p cnf 1 1 1 0" "1 0 0"')),
    ('init_quot indented', after_ns('  init_quot')),
    ('seal indented (no decl)', after_ns('  seal superlevel_set')),
    ('grind_pattern indented (no decl)', after_ns('  grind_pattern superlevel_mono => superlevel_set')),
]

with tempfile.TemporaryDirectory() as tmp:
    f = tft.Fixture(c, tmp)
    module = f.side / 'ClosedSuperlevel.lean'
    original = module.read_text()
    assert f.run('source').returncode == 0
    for label, text in PROBES:
        module.write_text(text)
        f.restore()
        run = f.run('source')
        try:
            direct = c.scan_module(text, 'ClosedSuperlevel')
            direct_msg = 'scan ok'
        except ValueError as e:
            direct_msg = 'scan refused: ' + str(e)[:90]
        verdict = 'ACCEPTED rc=0' if run.returncode == 0 else 'REFUSED rc=%d %s' % (
            run.returncode, [l for l in run.stderr.splitlines() if 'FAIL' in l][:1])
        print('%-70s | %s | %s' % (label, verdict, direct_msg))
    module.write_text(original)
    f.restore()
    # census_auxiliary whitelist abuse: a user-named where-clause auxiliary accepted by metadata(), source, and finish
    print('---- census_auxiliary abuse')
    for aux in (['TorusFiltration.torus_periodBasis.congr_simp', 'TorusFiltration.superlevel_mono.hidden'],
                ['TorusFiltration.torus_periodBasis.congr_simp', 'TorusFiltration..x'],
                ['TorusFiltration.torus_periodBasis.congr_simp', 'TorusFiltration._private.x'],
                ['TorusFiltration.torus_periodBasis.congr_simp', 'TorusFiltration.a.b.c.d']):
        f.manifest(lambda m: m.__setitem__('census_auxiliary', aux))
        f.regenerate(); f.commit()
        run = f.run('source')
        print('aux=%s source rc=%d %s' % (aux[1], run.returncode, [l for l in run.stderr.splitlines() if 'FAIL' in l][:1]))
    f.manifest(lambda m: m.__setitem__('census_auxiliary', ['TorusFiltration.torus_periodBasis.congr_simp', 'TorusFiltration.superlevel_mono.hidden']))
    f.regenerate(); f.commit()
    out = f.evidence()   # evidence() writes census.log from targets + census_auxiliary
    fin = f.run('finish', str(out))
    print('finish with whitelisted superlevel_mono.hidden: rc=%d' % fin.returncode,
          (json.loads(fin.stdout)['census'] if fin.returncode == 0 else fin.stderr[-200:]))
