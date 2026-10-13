"""Write SOURCE_MAP.json and SOURCE_FILES.json for the packet (run from the worktree root)."""
import hashlib, json, os, sys, subprocess
PKT = 'frontiers/c8_lifetime_difference_constant_20261001'
MAIN = '3e0a91b69892b9717564c592e7a966492a81566a'
def blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()
def ident(data):
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'git_blob': blob(data)}
def show(spec):
    return subprocess.run(['git', 'show', spec], capture_output=True, check=True).stdout
def rev(ref):
    return subprocess.run(['git', 'rev-parse', ref], capture_output=True, check=True, text=True).stdout.strip()
src = []
LP = 'imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md'
CAP = 'imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md'
src.append(dict(path=LP, **ident(open(LP, 'rb').read()), id='LP', commit=MAIN, current_required=True, on_main=True,
    role='section 1: the model in every dimension and Theorem B with (1.2); section 3: the preconditioned pins (3.1)-(3.3); section 8: '
         '1 - p_r <= Q^W(G_r^c); sections 9-11: the weighted Kac-Rice interface, the radial ledger (10.1)-(10.2) with '
         'A_r = 12 pi_r Z_r/r^2, and the lifetime pushforward (11.1)-(11.3); section 12: (12.1) and (12.3). Sections 8-12 are '
         'consumed as statements; section 11 records that the convergence A_r -> A_0 was not assigned a numerical rate, and '
         'section 16 lists numerical constants among the items not supplied'))
src.append(dict(path=CAP, **ident(open(CAP, 'rb').read()), id='CAP', commit=MAIN, current_required=True, on_main=True,
    role='section 1: the cylinder D, the partial-block norms M_3, M_4 and condition (1) defining G_r; consumed through the box-bound '
         'engines of Math-#206 and Math-#212'))
for pid, br, path, copy, pr, role in (
    ('E', 'origin/claude/c8-theorem-a-constants-planar-20260930', 'frontiers/c8_theorem_a_constants_planar_20260930/theorem_a.py',
     'engine_e2.py', 206, 'the planar certificate of Theorem E: exact covariance series and torus remainder, interval band laws, '
     'box_bound for sup Q^W(G_r^c)/r^3 on (b, k) boxes, band list; carried byte for byte as engine_e2.py and used as a library'),
    ('E3', 'origin/claude/c8-theorem-a-constants-d3-20261001', 'frontiers/c8_theorem_a_constants_d3_20261001/theorem_a3.py',
     'engine_e3.py', 212, 'the d = 3 certificate of Theorem E3: exact 3D covariance series, band laws with the regression on the '
     'transverse Hessian, contact moments E[det^a tr^c 1{B > 0}], box_bound, band list; carried byte for byte as engine_e3.py')):
    c = rev(br)
    data = show('%s:%s' % (c, path))
    src.append(dict(path=path, **ident(data), id=pid, commit=c, pull_request=pr, packet_copy=copy, current_required=False,
                    on_main=False, role=role))
for pid, br, path, pr, role in (
    ('W', 'origin/claude/c8-window-coefficient-planar-20260930', 'frontiers/c8_window_coefficient_planar_20260930/NOTE.md', 197,
     'read-only cross-check: the reference planar c_{B,K} = 0.0023869380211529483 on B x K (Owen-T closed form), which the '
     'certified enclosure of c^ref_{B,K} must contain'),
    ('W3', 'origin/claude/c8-window-d3-torus-20260930', 'frontiers/c8_window_coefficient_d3_torus_20260930/NOTE.md', 205,
     'read-only cross-check: the reference d = 3 c_{B,K} = 0.00097438236602425 on B x K'),
    ('G', 'origin/claude/c8-cap-failure-coefficient-planar-20260930', 'frontiers/c8_cap_failure_coefficient_planar_20260930/NOTE.md', 203,
     'read-only comparison: c_G = R(b) J(k) with J(k) = 2359296 k^3 (1 + delta(k)), for the floating cap-route limit of C_{B,K}'),
    ('G3', 'origin/claude/c8-cap-failure-coefficient-dims-20261001', 'frontiers/c8_cap_failure_coefficient_dims_20261001/RESULTS.json', 208,
     'read-only comparison: the floating R_3(b) R(b) values and the J^(3)/J ratios, for the floating d = 3 cap-route limit')):
    c = rev(br)
    data = show('%s:%s' % (c, path))
    src.append(dict(path=path, **ident(data), id=pid, commit=c, pull_request=pr, current_required=False, on_main=False, role=role))
smap = {'schema': 1, 'object': 'CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1', 'scientific_effect': 'NONE', 'certified': True,
        'review_status': 'AUTHOR_SIDE_THEOREM_WITH_CERTIFIED_CONSTANTS', 'source_repository': 'd6g8k5htny-coder/Math-',
        'main_at_creation': MAIN, 'note': sys.argv[1] if len(sys.argv) > 1 else '', 'sources': src}
open(os.path.join(PKT, 'SOURCE_MAP.json'), 'w').write(json.dumps(smap, indent=1) + '\n')
files = []
for name in sorted(os.listdir(PKT)):
    if name == 'SOURCE_FILES.json':
        continue
    data = open(os.path.join(PKT, name), 'rb').read()
    files.append(dict(path=name, **ident(data)))
man = {'schema': 1, 'scientific_effect': 'NONE', 'certified': True, 'review_status': 'AUTHOR_SIDE_THEOREM_WITH_CERTIFIED_CONSTANTS',
       'object': 'CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1', 'files': files, 'nonauthor_reads': []}
open(os.path.join(PKT, 'SOURCE_FILES.json'), 'w').write(json.dumps(man, indent=1) + '\n')
print(json.dumps([(f['path'], f['bytes']) for f in files]))
