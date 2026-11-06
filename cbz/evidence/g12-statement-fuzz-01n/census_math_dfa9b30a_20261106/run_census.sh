#!/bin/bash
# G12 census over every Lean group in Math- at dfa9b30a (raw evidence snapshots excluded; copies deduplicated)
S=${CENSUS_SCRATCH:-/tmp/g12-census}
M=$S/math            # `git -C <Math- clone> archive dfa9b30a frontiers companions specifications reviews formal | tar -x -C $M`
OUT=$S/out; mkdir -p $OUT
cd "$(git rev-parse --show-toplevel)"
for g in formal frontiers/source_derivative_gaussian_20261025 frontiers/source_lifetime_realization_20261017 \
         specifications/lean_statement_proofs_20261011 frontiers/source_smooth_realization_20261011 \
         frontiers/determinant_moment_20261025 frontiers/compact_morse_elder_adapter_20261018 \
         frontiers/regular_superlevel_geometry_20261020 frontiers/asymmetric_elder_bridge_20261017 \
         companions/torus_filtration_20261011 specifications/lean_statements_20261010 \
         reviews/w3_side24_source_proposal_20261008 companions/d2_square_schur_20261006 \
         companions/d2_atom_integral_20261006 companions/d2_moment_bridge_20261006 frontiers/cap_first_exit_lean_20261002; do
  name=$(echo $g | tr '/' '_')
  python3 -I tools/cbz/statement_fuzz.py --lean-dir $M/$g --exclude /raw --exclude raw- --json $OUT/$name.json > $OUT/$name.stdout.txt 2>&1
  python3 -c "
import json,sys; j=json.load(open('$OUT/$name.json')); c=j['status_counts']
print(f\"{'$g':58} files {j['files']:4} (eval {j['files_evaluated']:3}) targets {j['targets']:4}  PASS {c.get('PASS',0):4} CORR {c.get('CORROBORATED',0):4} FAIL {c.get('FAIL',0):3} UNSUP {c.get('UNSUPPORTED',0):4}\")"
done
