#!/bin/bash
# G12 census over every Lean group in Math- at dfa9b30a with dependency definitions (--defs-from): each group pools
# the definitions of the groups its import lines name, transitively (dependency table in CENSUS.md); raw evidence
# snapshots excluded; copies deduplicated.
S=${CENSUS_SCRATCH:-/tmp/g12-census}
M=$S/math            # git -C <Math- clone> archive dfa9b30a frontiers companions specifications reviews formal | tar -x -C $M
OUT=$S/out; mkdir -p $OUT
cd "$(git rev-parse --show-toplevel)"
run() {
  g=$1; shift; name=$(echo "$g" | tr '/' '_')
  python3 -I tools/cbz/statement_fuzz.py --lean-dir "$M/$g" "$@" --exclude /raw --exclude raw- --json "$OUT/$name.json" > "$OUT/$name.stdout.txt" 2>&1
  python3 - "$OUT/$name.json" "$g" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1])); c = j['status_counts']
print(f"{sys.argv[2]:58} targets {j['targets']:4}  PASS {c.get('PASS', 0):4} CORR {c.get('CORROBORATED', 0):4} FAIL {c.get('FAIL', 0):3} "
      f"UNSUP {c.get('UNSUPPORTED', 0):4}  defs-from {len(j.get('defs_from', {}).get('dirs', []))} unresolved {sorted(j['unresolved_imports'])}")
EOF
}

run formal
run frontiers/source_derivative_gaussian_20261025 --defs-from "$M/companions/torus_filtration_20261011" --defs-from "$M/frontiers/asymmetric_elder_bridge_20261017" --defs-from "$M/frontiers/determinant_moment_20261025" --defs-from "$M/frontiers/source_lifetime_realization_20261017" --defs-from "$M/frontiers/source_smooth_realization_20261011" --defs-from "$M/specifications/lean_statements_20261010"
run frontiers/source_lifetime_realization_20261017 --defs-from "$M/frontiers/source_smooth_realization_20261011"
run specifications/lean_statement_proofs_20261011 --defs-from "$M/specifications/lean_statements_20261010"
run frontiers/source_smooth_realization_20261011
run frontiers/determinant_moment_20261025
run frontiers/compact_morse_elder_adapter_20261018 --defs-from "$M/companions/torus_filtration_20261011" --defs-from "$M/frontiers/asymmetric_elder_bridge_20261017" --defs-from "$M/frontiers/source_lifetime_realization_20261017" --defs-from "$M/frontiers/source_smooth_realization_20261011" --defs-from "$M/specifications/lean_statements_20261010"
run frontiers/regular_superlevel_geometry_20261020
run frontiers/asymmetric_elder_bridge_20261017 --defs-from "$M/companions/torus_filtration_20261011" --defs-from "$M/specifications/lean_statements_20261010"
run companions/torus_filtration_20261011 --defs-from "$M/specifications/lean_statements_20261010"
run specifications/lean_statements_20261010
run reviews/w3_side24_source_proposal_20261008 --defs-from "$M/formal"
run companions/d2_square_schur_20261006 --defs-from "$M/formal"
run companions/d2_atom_integral_20261006 --defs-from "$M/companions/d2_moment_bridge_20261006" --defs-from "$M/companions/d2_square_schur_20261006" --defs-from "$M/formal"
run companions/d2_moment_bridge_20261006 --defs-from "$M/formal"
run frontiers/cap_first_exit_lean_20261002
