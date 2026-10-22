#!/usr/bin/env python3
"""Static scan (no import, no execution) of Python sources for their use of the H2
library cov_exact.py.  For each file: how it loads the library, its configure() calls,
and every call to an evaluator whose iv path carries the truncated-normalizer defect
(e_sum, s_sum, k1_image, and through them k1_spectral, k1, cov, moment_a), plus reads
of the interval lattice.  Usage: python3 -I -B scan.py <root> <out.json>"""
import ast, hashlib, json, os, re, sys

AFFECTED = {'e_sum', 's_sum', 'k1_image', 'k1_spectral', 'k1', 'cov', 'moment_a'}
OTHER = {'configure', 'lattice', 'get_config', 'moment_a_poisson', 'one_minus_a2_poisson',
         'spectral_tail_1d', 'image_tail_1d', 'planar_moment', 'hermite_he'}
# position of the mode argument when passed positionally
MODE_POS = {'e_sum': 2, 's_sum': 2, 'k1_image': 2, 'k1_spectral': 2, 'k1': 3, 'cov': 5,
            'moment_a': 2, 'moment_a_poisson': 1, 'one_minus_a2_poisson': 0, 'hermite_he': 2}
LOADRX = re.compile(r'cov_exact|f08c1c5f')


def seg(src, node):
    s = ast.get_source_segment(src, node) or ''
    return ' '.join(s.split())[:240]


def mode_of(call, name):
    for kw in call.keywords:
        if kw.arg == 'mode':
            return ast.unparse(kw.value)
    p = MODE_POS.get(name)
    if p is not None and len(call.args) > p:
        return ast.unparse(call.args[p])
    return "default('mpf')" if name in MODE_POS else None


def scan(path):
    raw = open(path, 'rb').read()
    src = raw.decode('utf-8', 'replace')
    rec = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
           'load_lines': [], 'calls': [], 'lattice_attr': [], 'defs': [], 'parse_error': None}
    for i, line in enumerate(src.splitlines(), 1):
        if LOADRX.search(line):
            rec['load_lines'].append([i, ' '.join(line.split())[:200]])
        if re.search(r'_LATT_IV|_LATT\b', line):
            rec['lattice_attr'].append([i, ' '.join(line.split())[:200]])
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rec['parse_error'] = str(e)
        return rec
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in AFFECTED | OTHER:
            rec['defs'].append([node.lineno, node.name])
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Attribute):
            name, recv = f.attr, ast.unparse(f.value)
        elif isinstance(f, ast.Name):
            name, recv = f.id, None
        else:
            continue
        if name in AFFECTED or name in OTHER:
            rec['calls'].append({'line': node.lineno, 'name': name, 'recv': recv,
                                 'mode': mode_of(node, name), 'text': seg(src, node)})
    rec['calls'].sort(key=lambda c: c['line'])
    return rec


def main():
    root, out = sys.argv[1], sys.argv[2]
    res = {}
    for dp, dns, fns in os.walk(root):
        dns.sort()
        for fn in sorted(fns):
            if fn.endswith('.py'):
                p = os.path.join(dp, fn)
                src = open(p, 'rb').read().decode('utf-8', 'replace')
                if LOADRX.search(src):
                    res[os.path.relpath(p, root)] = scan(p)
    json.dump(res, open(out, 'w'), indent=1, sort_keys=True)
    for k, r in sorted(res.items()):
        iv = [c for c in r['calls'] if c['name'] in AFFECTED and c['mode'] and 'iv' in c['mode']]
        aff = [c for c in r['calls'] if c['name'] in AFFECTED]
        cfg = [c['text'] for c in r['calls'] if c['name'] == 'configure']
        print('%-70s aff=%3d iv=%3d latt=%2d cfg=%s%s' % (k[13:], len(aff), len(iv), len(r['lattice_attr']),
              cfg[:3], ' PARSE_ERR' if r['parse_error'] else ''))


main()
