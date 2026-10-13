from pathlib import Path
side = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011')
def patch(name, pairs):
    p = side / name; s = p.read_text(encoding='utf-8')
    for old, new in pairs:
        assert s.count(old) == 1, (name, s.count(old), old[:90])
        s = s.replace(old, new)
    p.write_text(s, encoding='utf-8'); print(name, 'patched')

patch('check.py', [
    # numerals as Lean lexes them
    ("""# Token scan for keyword detection. ASCII identifiers (dotted components, primes, `!`, `?`) and numerals are whole
# tokens, as Lean lexes them: `0alias` is the numeral `0` followed by the keyword `alias` (refused), `foo2alias` is
# one identifier (accepted). A component after a dot is a name component, not a keyword. Any non-ASCII character
# separates tokens, a refusal in the safe direction when a keyword abuts a unicode letter.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_'!?]*(?:\\.[A-Za-z_][A-Za-z0-9_'!?]*)*|[0-9]+")
""",
     """# Token scan for keyword detection. ASCII identifiers (dotted components, primes, `!`, `?`) and numerals are whole
# tokens, with numerals in Lean's own grammar (Lean/Parser/Basic.lean numberFnAux: decimal with `_` separators,
# fraction and exponent; `0x`/`0b`/`0o` prefixes): `0alias` is the numeral `0` followed by the keyword `alias`
# (refused), `0x1Ftheorem` is `0x1F` then `theorem` (refused), `foo2alias` is one identifier (accepted). A component
# after a dot is a name component, not a keyword. Any non-ASCII character separates tokens, a refusal in the safe
# direction when a keyword abuts a unicode letter.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_'!?]*(?:\\.[A-Za-z_][A-Za-z0-9_'!?]*)*"
                   r"|0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|[0-9][0-9_]*(?:\\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9]+)?")
"""),
    ("""            'name_power_vars', 'with_weak_namespace', 'meta', 'public', 'aux_def', 'lrat_proof')
""",
     """            'name_power_vars', 'with_weak_namespace', 'meta', 'public', 'aux_def', 'lrat_proof', 'reprove',
            'declare_aesop_exception', 'declare_uint_simprocs', 'declare_sint_simprocs', 'seal', 'unseal', 'unset_option',
            'suppress_compilation', 'grind_pattern', 'grind_annotated', 'extend_docs', 'init_quot',
            'initialize_simps_projections')
"""),
    ("""# The exact membership of the evidence directory when `finish` runs (C262 gate review G3): the nine files emit()
# writes, a .log/.status pair per Lean process, the two protocol-test transcripts and the four environment records.
""",
     """# The exact membership of the evidence directory when `finish` runs (C262 gate review G3): the ten files emit()
# writes, a .log/.status pair per Lean process, the two protocol-test transcripts and the four environment records.
"""),
    ("""# The elaborated declaration census (C262 G2 closure): Lean lists every constant that the companion modules add to
# the environment and that is not an internal detail (`Name.isInternalDetail`: `_`-prefixed, `eq_n`, `match_n`,
# `proof_n`, `omega_n`, numeric components). `finish` requires the list to equal the inventory plus the named
# compiler-generated auxiliaries, so a declaration the scanner missed (an unlisted command, a `where` clause) is
# refused by the receipt, whatever its spelling.
CENSUS = '''import TorusFiltration
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let modules : Array Lean.Name := #[`TorusTopology, `ClosedSuperlevel, `TorusFiltration]
  let idxs := modules.filterMap env.getModuleIdx?
  if idxs.size != modules.size then throwError "census: a companion module is not imported"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if idxs.contains i && !n.isInternalDetail then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    IO.println n
  IO.println s!"CENSUS_END {names.size}"
'''
""",
     """# The elaborated declaration census (C262 G2 closure): Lean lists EVERY constant the companion modules add to the
# environment, each tagged `internal`/`user` (`Name.isInternalDetail`) and `range`/`generated` (whether Lean recorded
# a declaration range, which every declaration elaborated from source syntax has and compiler-generated constants
# such as `congr_simp` lemmas, abstracted proofs and equation lemmas lack). `finish` requires: the user-named ranged
# constants equal the inventory exactly (a `where`-clause or unlisted-command declaration is refused whatever its
# spelling); no ranged constant carries an internal-style name (a user-written `proof_1` is refused); the user-named
# generated constants equal `census_auxiliary` exactly (so that whitelist can name only compiler-generated constants);
# internal generated constants are counted and otherwise ignored.
CENSUS = '''import TorusFiltration
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let modules : Array Lean.Name := #[`TorusTopology, `ClosedSuperlevel, `TorusFiltration]
  let idxs := modules.filterMap env.getModuleIdx?
  if idxs.size != modules.size then throwError "census: a companion module is not imported"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if idxs.contains i then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    let ranged ← Lean.findDeclarationRanges? n
    IO.println s!"{n} {if n.isInternalDetail then "internal" else "user"} {if ranged.isSome then "range" else "generated"}"
  IO.println s!"CENSUS_END {names.size}"
'''
"""),
    ("""def census(text, names, auxiliary):
    \"\"\"Parse census.log: one constant per line, then `CENSUS_END <count>`; the set must equal names ∪ auxiliary.\"\"\"
    lines = text.strip('\\n').split('\\n') if text.strip() else []
    require(lines and re.fullmatch(r'CENSUS_END [0-9]+', lines[-1]), 'census terminator absent')
    listed = lines[:-1]
    require(all(re.fullmatch(NAME, n) for n in listed), 'unrecognized census output')
    require(int(lines[-1].split()[1]) == len(listed) and len(set(listed)) == len(listed), 'census count mismatch')
    expected = set(names) | set(auxiliary)
    extra, missing = sorted(set(listed) - expected), sorted(expected - set(listed))
    require(not extra and not missing, 'declaration census differs: extra %s, missing %s' % (extra, missing))
    return {'constants': len(listed), 'auxiliary': sorted(auxiliary)}
""",
     """CENSUS_LINE = re.compile(r'(' + NAME + r') (internal|user) (range|generated)')


def census(text, names, auxiliary):
    \"\"\"Parse census.log (`<name> internal|user range|generated` per constant, then `CENSUS_END <count>`).

    user+range must equal `names` exactly; internal+range is refused; user+generated must equal `auxiliary` exactly.\"\"\"
    lines = text.strip('\\n').split('\\n') if text.strip() else []
    require(lines and re.fullmatch(r'CENSUS_END [0-9]+', lines[-1]), 'census terminator absent')
    rows = [CENSUS_LINE.fullmatch(line) for line in lines[:-1]]
    require(all(rows), 'unrecognized census output')
    listed = [m.group(1) for m in rows]
    require(int(lines[-1].split()[1]) == len(listed) and len(set(listed)) == len(listed), 'census count mismatch')
    kinds = {m.group(1): (m.group(2), m.group(3)) for m in rows}
    ranged_user = sorted(n for n, k in kinds.items() if k == ('user', 'range'))
    ranged_internal = sorted(n for n, k in kinds.items() if k == ('internal', 'range'))
    generated_user = sorted(n for n, k in kinds.items() if k == ('user', 'generated'))
    require(not ranged_internal, 'user-written declarations with internal-style names: %s' % ranged_internal)
    extra, missing = sorted(set(ranged_user) - set(names)), sorted(set(names) - set(ranged_user))
    require(not extra, 'user-written declarations outside the inventory: %s' % extra)
    require(not missing, 'inventory targets absent from the census: %s' % missing)
    unlisted, absent = sorted(set(generated_user) - set(auxiliary)), sorted(set(auxiliary) - set(generated_user))
    require(not unlisted, 'compiler-generated constants not listed in census_auxiliary: %s' % unlisted)
    require(not absent, 'census_auxiliary entries absent or not compiler-generated: %s' % absent)
    return {'constants': len(listed), 'user_declarations': len(ranged_user), 'generated_user': generated_user,
            'internal_generated': sum(1 for k in kinds.values() if k == ('internal', 'generated'))}


def write_new_file(path, text):
    \"\"\"Create `path` exclusively (O_EXCL, O_NOFOLLOW): an existing file or a planted symlink fails the open instead of
    being replaced or followed; nothing is truncated.\"\"\"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(text)
"""),
    ("""    # Exclusive creation (O_EXCL, O_NOFOLLOW): a receipt that appeared meanwhile, or a symlink planted at the path,
    # fails the open instead of being replaced or followed.
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
""",
     """    write_new_file(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
"""),
    ("""    require(isinstance(aux, list) and all(isinstance(n, str) and re.fullmatch(NAME, n) and n.startswith(NAMESPACE + '.')
                                          and n.count('.') >= 2 for n in aux) and len(aux) == len(set(aux)),
            'census_auxiliary must list dotted auxiliary names under the companion namespace')
""",
     """    # Dotted names under the namespace; that each is compiler-generated is checked by the census (`finish` requires
    # every entry to appear without a declaration range), not by this schema rule.
    require(isinstance(aux, list) and all(isinstance(n, str) and re.fullmatch(NAME, n) and n.startswith(NAMESPACE + '.')
                                          and n.count('.') >= 2 for n in aux) and len(aux) == len(set(aux)),
            'census_auxiliary must list dotted auxiliary names under the companion namespace')
"""),
])

patch('test_torus_filtration.py', [
    # fixture census.log in the new format
    ("""        listed = sorted(targets + list(meta['census_auxiliary']))
        (out / 'census.log').write_text(''.join(n + '\\n' for n in listed) + 'CENSUS_END %d\\n' % len(listed))
        return out
""",
     """        rows = sorted([(n, 'user range') for n in targets] + [(n, 'user generated') for n in meta['census_auxiliary']]
                      + [(n, 'internal generated') for n in INTERNAL_GENERATED])
        (out / 'census.log').write_text(''.join('%s %s\\n' % r for r in rows) + 'CENSUS_END %d\\n' % len(rows))
        return out
"""),
    ("""PYTHON_LINE = 'TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize
""",
     """PYTHON_LINE = 'TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize
# Compiler-generated internal constants as the real census lists them (abstracted proofs, equation lemmas); the fixture
# census carries a few so the parser's `internal generated` branch is exercised.
INTERNAL_GENERATED = ('TorusFiltration.superlevel_set.eq_1', 'TorusFiltration.torus_periodBasis._proof_1',
                      'TorusFiltration.torus_proj_eq_iff._abel_1_1')
"""),
    # numeral splices (scanner) and the new commands
    ("""            (good.replace(':= trivial', ':= by\\n  exact/- -/sorry', 1), 'admission token in ClosedSuperlevel: sorry'),
""",
     """            (good.replace(':= trivial', ':= by\\n  exact/- -/sorry', 1), 'admission token in ClosedSuperlevel: sorry'),
            # Lean's non-decimal numerals end where Lean ends them (hex, binary, octal, exponent, `_` separators)
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0x1Ftheorem hidden : True := trivial'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 1e5theorem hidden : True := trivial'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 1_0theorem hidden : True := trivial'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0o7instance hidden : Inhabited Nat := ⟨0⟩'), command),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0b1axiom hidden : False'), 'admission token in ClosedSuperlevel: axiom'),
            (good.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0x1Fset_option maxHeartbeats 0'), command),
            (inject('  reprove superlevel_mono by trivial'), command),
            (inject('  declare_aesop_exception hidden'), command),
            (inject('  seal superlevel_set'), command),
            (inject('  unset_option maxHeartbeats'), command),
"""),
    # CLI numeral splice controls
    ("""                     (original.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
""",
     """                     (original.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0alias hidden := Nat.add_comm'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0x1Ftheorem hidden : True := trivial'),
                      'command token outside a column-0 declaration in ClosedSuperlevel:'),
                     (original.replace('noncomputable def superlevel_set : Nat := 0', 'noncomputable def superlevel_set : Nat := 0b1axiom hidden : False'),
                      'admission token in ClosedSuperlevel: axiom'),
"""),
    # emit test: census content
    ("""            for needed in ('isInternalDetail', 'getModuleIdxFor?', '`TorusTopology', '`ClosedSuperlevel', '`TorusFiltration', 'CENSUS_END'):
""",
     """            for needed in ('isInternalDetail', 'findDeclarationRanges?', 'getModuleIdxFor?', '`TorusTopology', '`ClosedSuperlevel',
                           '`TorusFiltration', 'CENSUS_END', '"internal" else "user"', '"range" else "generated"'):
"""),
    # finish receipt field
    ("""            self.assertEqual(receipt['census'], {'constants': 58 + len(self.c.metadata()['census_auxiliary']),
                                                 'auxiliary': sorted(self.c.metadata()['census_auxiliary'])})
""",
     """            self.assertEqual(receipt['census'], {'constants': 58 + len(self.c.metadata()['census_auxiliary']) + len(INTERNAL_GENERATED),
                                                 'user_declarations': 58,
                                                 'generated_user': sorted(self.c.metadata()['census_auxiliary']),
                                                 'internal_generated': len(INTERNAL_GENERATED)})
"""),
    # finish weakening cases for the census, rewritten for the new format
    ("""                # C262 G2 closure: the elaborated census must equal the inventory plus the named auxiliaries exactly
                ('census.log', (out / 'census.log').read_text().replace('CENSUS_END', 'TorusFiltration.superlevel_mono.hidden\\nCENSUS_END'),
                 'census count mismatch'),
                ('census.log', (out / 'census.log').read_text().replace('\\nCENSUS_END %d' % (58 + len(meta['census_auxiliary'])),
                                                                        '\\nTorusFiltration.superlevel_mono.hidden\\nCENSUS_END %d' % (59 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra ['TorusFiltration.superlevel_mono.hidden'], missing []"),
                ('census.log', (out / 'census.log').read_text().replace('TorusFiltration.superlevel_mono\\n', '').replace(
                    'CENSUS_END %d' % (58 + len(meta['census_auxiliary'])), 'CENSUS_END %d' % (57 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra [], missing ['TorusFiltration.superlevel_mono']"),
                ('census.log', (out / 'census.log').read_text().replace('TorusFiltration.torus_periodBasis.congr_simp\\n', '').replace(
                    'CENSUS_END %d' % (58 + len(meta['census_auxiliary'])), 'CENSUS_END %d' % (57 + len(meta['census_auxiliary']))),
                 "declaration census differs: extra [], missing ['TorusFiltration.torus_periodBasis.congr_simp']"),
                ('census.log', ''.join((out / 'census.log').read_text().splitlines(True)[:-1]), 'census terminator absent'),
                ('census.log', 'warning: noise\\n' + (out / 'census.log').read_text(), 'unrecognized census output'),
                ('census.status', '1\\n', 'missing successful process: census'),
""",
     """                # C262 G2 closure: the elaborated census. N = the fixture's row count.
                ('census.log', census_with(('TorusFiltration.superlevel_mono.hidden', 'user range')), "user-written declarations outside the inventory: ['TorusFiltration.superlevel_mono.hidden']"),
                ('census.log', census_with(('TorusFiltration.superlevel_mono.proof_1', 'internal range')), "user-written declarations with internal-style names: ['TorusFiltration.superlevel_mono.proof_1']"),
                ('census.log', census_with(('TorusFiltration.superlevel_mono._hidden', 'internal range')), "user-written declarations with internal-style names: ['TorusFiltration.superlevel_mono._hidden']"),
                ('census.log', census_with(('TorusFiltration.superlevel_mono.other_simp', 'user generated')), "compiler-generated constants not listed in census_auxiliary: ['TorusFiltration.superlevel_mono.other_simp']"),
                ('census.log', census_without('TorusFiltration.superlevel_mono'), "inventory targets absent from the census: ['TorusFiltration.superlevel_mono']"),
                ('census.log', census_without('TorusFiltration.torus_periodBasis.congr_simp'), "census_auxiliary entries absent or not compiler-generated: ['TorusFiltration.torus_periodBasis.congr_simp']"),
                ('census.log', (out / 'census.log').read_text().replace('TorusFiltration.torus_periodBasis.congr_simp user generated', 'TorusFiltration.torus_periodBasis.congr_simp user range'),
                 "user-written declarations outside the inventory: ['TorusFiltration.torus_periodBasis.congr_simp']"),
                ('census.log', (out / 'census.log').read_text().replace('CENSUS_END', 'TorusFiltration.superlevel_mono.hidden user range\\nCENSUS_END'), 'census count mismatch'),
                ('census.log', ''.join((out / 'census.log').read_text().splitlines(True)[:-1]), 'census terminator absent'),
                ('census.log', 'warning: noise\\n' + (out / 'census.log').read_text(), 'unrecognized census output'),
                ('census.log', (out / 'census.log').read_text().replace(' user range\\n', '\\n', 1), 'unrecognized census output'),
                ('census.status', '1\\n', 'missing successful process: census'),
"""),
    # helpers for the census cases, defined before the cases list
    ("""            cases = [
                ('build-TorusTopology.status', '1\\n', 'missing successful process: build-TorusTopology'),
""",
     """            def census_with(row):
                lines = (out / 'census.log').read_text().splitlines(True)
                body = sorted(lines[:-1] + ['%s %s\\n' % row])
                return ''.join(body) + 'CENSUS_END %d\\n' % len(body)

            def census_without(name):
                lines = (out / 'census.log').read_text().splitlines(True)
                body = [l for l in lines[:-1] if not l.startswith(name + ' ')]
                return ''.join(body) + 'CENSUS_END %d\\n' % len(body)
            cases = [
                ('build-TorusTopology.status', '1\\n', 'missing successful process: build-TorusTopology'),
"""),
    # exclusive-write control
    ("""            # A pre-existing receipt (even a symlink to nothing) is never replaced.
""",
     """            # The exclusive write itself (not only the pre-check) refuses an existing file and a planted symlink, intact.
            frozen = b'frozen receipt bytes\\n'
            (out / 'planted.json').write_bytes(frozen)
            with self.assertRaises(FileExistsError):
                self.c.write_new_file(out / 'planted.json', 'new\\n')
            self.assertEqual((out / 'planted.json').read_bytes(), frozen)
            (out / 'planted.json').unlink()
            os.symlink('absent-target', out / 'planted.json')
            with self.assertRaises(OSError):
                self.c.write_new_file(out / 'planted.json', 'new\\n')
            self.assertFalse((out / 'absent-target').exists())
            (out / 'planted.json').unlink()
            # A pre-existing receipt (even a symlink to nothing) is never replaced.
"""),
])

patch('README.md', [
    ("""`public`, `aux_def`, `lrat_proof`).
""",
     """`public`, `aux_def`, `lrat_proof`, `reprove`, `declare_aesop_exception`, `declare_uint_simprocs`,
`declare_sint_simprocs`, `seal`, `unseal`, `unset_option`, `suppress_compilation`, `grind_pattern`,
`grind_annotated`, `extend_docs`, `init_quot`, `initialize_simps_projections`; Mathlib spells
`compile_inductive%`, `compile_def%` and `gen_injective_theorems%` with a `%`, which the token scan
does not need; a few listed commands only attach documentation or notes rather than declarations,
and are refused all the same).
"""),
    ("""`name_poly_vars`, `add_aesop_rules`, and more, all now listed), and the `where` clause of a
declaration adds a dotted auxiliary declaration that no keyword list can refuse. The lexer also
treats a closed block comment as a token separator (`theorem/- -/hidden` is `theorem hidden`)
and ends a numeral at its first non-digit (`0alias` is `0` then `alias`), both found by that
verification. **The declaration census the receipt stands on is therefore not the scanner but
`Census.lean`**, emitted by `check.py emit` and run by the replay: Lean itself lists every
constant the three companion modules add to the environment that is not an internal detail
(`Name.isInternalDetail`: `_`-prefixed, `eq_n`, `match_n`, `proof_n`, `omega_n` or numeric
components), and `check.py finish` requires that list to equal the 58 inventoried names plus
the compiler-generated auxiliaries named in `SOURCE_FILES.json` (`census_auxiliary`, today
exactly `TorusFiltration.torus_periodBasis.congr_simp`, the simp-congruence lemma Lean generates
for `torus_periodBasis`), nothing more and nothing less. A declaration the scanner missed is
refused by the receipt whatever its spelling. A compile fix that needs a refused form must
extend the scanner, the inventory and, for a new auxiliary, `census_auxiliary` explicitly.
""",
     """`name_poly_vars`, `add_aesop_rules`, `aux_def`, `reprove`, and more, listed as found), and the
`where` clause of a declaration adds a dotted auxiliary declaration that no keyword list can
refuse. The lexer also treats a closed block comment as a token separator (`theorem/- -/hidden`
is `theorem hidden`) and lexes numerals as Lean does (`0alias` is `0` then `alias`; `0x1Ftheorem`
is `0x1F` then `theorem`), both found by that verification. **The declaration census the receipt
stands on is therefore not the scanner but `Census.lean`**, emitted by `check.py emit` and run
by the replay: Lean itself lists every constant the three companion modules add to the
environment, tagged `internal` or `user` (`Name.isInternalDetail`: `_`-prefixed, `eq_n`,
`match_n`, `proof_n`, `omega_n` or numeric components) and `range` or `generated` (whether Lean
recorded a declaration range, which every declaration elaborated from source syntax carries and
compiler-generated constants lack). `check.py finish` requires the user-named ranged constants
to equal the 58 inventoried names exactly, refuses any ranged constant with an internal-style
name (a `where`-clause `proof_1` written by hand), and requires the user-named generated
constants to equal `census_auxiliary` in `SOURCE_FILES.json` exactly (today
`TorusFiltration.torus_periodBasis.congr_simp`, the simp-congruence lemma Lean generates for
`torus_periodBasis`), so that list can name only constants Lean generated; internal generated
constants (abstracted proofs, equation lemmas) are counted and otherwise ignored. A declaration
the scanner missed is refused by the receipt whatever its spelling, because it was elaborated
from source and therefore carries a range. A compile fix that needs a refused form must extend
the scanner, the inventory and, for a new compiler-generated constant, `census_auxiliary`.
"""),
])
patch('SCOPE.md', [
    ("""lock, that Lean elaborated them with warnings as errors, that Lean's own census of the
constants the three modules add to the environment (`Census.lean`, non-internal names) is
exactly these 58 plus the one named compiler-generated auxiliary, that
""",
     """lock, that Lean elaborated them with warnings as errors, that Lean's own census of the
constants the three modules add to the environment (`Census.lean`, with declaration-range and
internal-name tags) has exactly these 58 as its user-written declarations and exactly the one
named compiler-generated auxiliary, that
"""),
    ("""constant, a wrong census count or terminator, and a `check.py` child running in a different
""",
     """constant, a user-written declaration with an internal-style name, a `census_auxiliary` entry
that Lean did not generate, a keyword spliced to a hex/binary/octal/exponent numeral, a wrong
census count or terminator, and a `check.py` child running in a different
"""),
])
