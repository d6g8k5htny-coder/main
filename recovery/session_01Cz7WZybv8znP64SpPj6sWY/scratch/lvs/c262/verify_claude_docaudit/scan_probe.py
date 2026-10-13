import importlib.util, sys
spec = importlib.util.spec_from_file_location('check_new', sys.argv[1]); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
base = 'import Mathlib\n\nnamespace TorusFiltration\n{}theorem superlevel_mono : True := trivial\nend TorusFiltration\n'
probes = {
  'alias (indented, after namespace)': '  alias hidden := Nat.add_comm\n',
  'alias (column 0)': 'alias hidden := Nat.add_comm\n',
  'mk_iff_of_inductive_prop (indented)': '  mk_iff_of_inductive_prop Foo hidden\n',
  'lrat_proof (indented)': '  lrat_proof hidden\n',
  'aux_def (indented)': '  aux_def hidden : Nat := 0\n',
  'builtin_simproc_decl (indented)': '  builtin_simproc_decl hidden (x + 0) := fun _ => return .continue\n',
  'builtin_dsimproc_decl (indented)': '  builtin_dsimproc_decl hidden (x + 0) := fun _ => return .continue\n',
  'cbv_simproc_decl (indented)': '  cbv_simproc_decl hidden (x + 0) := fun _ => return .continue\n',
  'register_builtin_option (indented)': '  register_builtin_option hidden : Bool := { defValue := true }\n',
  'register_label_attr (indented)': '  register_label_attr hidden\n',
  'initialize_simps_projections (indented)': '  initialize_simps_projections Foo\n',
  'declare_aesop_exception (indented)': '  declare_aesop_exception hidden\n',
  'simproc_decl (indented, listed)': '  simproc_decl hidden (x + 0) := fun _ => return .continue\n',
  'irreducible_def (indented, listed)': '  irreducible_def hidden : Nat := 0\n',
}
for label, inj in probes.items():
    try:
        scan = c.scan_module(base.format(inj), 'M')
        print('ACCEPTED (uncounted) ::', label, '-> inventory', {k: v for k, v in scan.items() if v})
    except Exception as e:
        print('refused ::', label, '->', str(e)[:90])
