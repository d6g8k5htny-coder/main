"""Empirical behaviour matrix for the generated-JSON scientific_status_authority predicate.

Read-only behaviour report on meta-framework origin/main b4576cb9 (checker blob adf2945b,
tests blob 8548a2f0). Runs against an exported copy in ./repo; never touches the clone.
Fixture reproduces tests/test_architecture_conformance.py ArchitectureConformance.setUp.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, sys, traceback
from pathlib import Path

W = Path(__file__).resolve().parent
REPO = W / 'repo'
SCRIPT = REPO / 'tools/architecture_conformance.py'
CASES = W / 'cases'
PY = '/usr/bin/python3'
sys.path.insert(0, str(REPO))
from tools.architecture_conformance import check_workspace  # noqa: E402

REPOS = ['Math-', 'google-drive', 'governance-', 'main', 'meta-framework', 'query-', 'sandbox', 'trial']
M = '"scientific_status_authority"'


def valid_authority():
    return {'required_authority_ids': ['claims_firewall', 'scientific_state_architecture'],
            'authorities': {'claims_firewall': {'id': 'claims_firewall', 'owns': ['grade']}},
            'this_package': {'id': 'scientific_state_architecture', 'owns': ['schema_contract'],
                             'never_writes': ['status', 'grade', 'classification', 'controlling',
                                              'lemma_closed', 'prizes_solved', 'independence_credit']}}


def build_fixture(root: Path):
    # verbatim reproduction of ArchitectureConformance.setUp
    for name in REPOS:
        (root / name).mkdir(parents=True)
    registry = {'schema_version': 1, 'scientific_status_authority': False,
                'repositories': {name: {'full_name': 'd6g8k5htny-coder/' + name, 'role': name,
                                        'visibility': 'private' if name == 'sandbox' else 'public'}
                                 for name in REPOS}, 'artifacts': []}
    (root / 'meta-framework/registry.json').write_text(json.dumps(registry))
    a = root / 'main/architecture/scientific_state/v1'
    a.mkdir(parents=True)
    (a / 'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))


def snapshot(root: Path):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def category(v: str) -> str:
    if v.startswith('generated scientific-status authority: '):
        return 'AUTH'
    if v.startswith('generated JSON parse error: '):
        return 'PARSE'
    return 'OTHER'


def run_cli(ws: Path, optimize: bool):
    cmd = [PY] + (['-O'] if optimize else []) + [str(SCRIPT), '--workspace', str(ws)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(W))
    return cmd, p.returncode, p.stdout, p.stderr


def run_api_O(ws: Path):
    code = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
            'from tools.architecture_conformance import check_workspace as c;'
            'print(json.dumps(c(sys.argv[2]),sort_keys=True))')
    p = subprocess.run([PY, '-O', '-c', code, str(REPO), str(ws)], capture_output=True, text=True,
                       timeout=120, cwd=str(W))
    if p.returncode != 0:
        return {'exception': p.stderr.strip().splitlines()[-1] if p.stderr.strip() else 'rc=%d' % p.returncode,
                'rc': p.returncode}
    return json.loads(p.stdout)


def run_case(case_id: str, files: dict[str, bytes], meta: dict):
    cdir = CASES / case_id
    if cdir.exists():
        shutil.rmtree(cdir)
    ws = cdir / 'ws'
    build_fixture(ws)
    for rel, raw in files.items():
        p = ws / 'meta-framework/generated' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        inp = cdir / 'input' / rel
        inp.parent.mkdir(parents=True, exist_ok=True)
        inp.write_bytes(raw)
    before = snapshot(ws)
    rec = dict(meta, case=case_id, files={k: v.decode('utf-8', 'replace') if len(v) < 400 else
                                           v[:200].decode('utf-8', 'replace') + '...(%d bytes)' % len(v)
                                           for k, v in files.items()})
    try:
        api = check_workspace(ws)
        rec['api'] = api
        rec['api_categories'] = [category(v) for v in api['violations']]
    except BaseException as exc:  # noqa: BLE001 - record any escape
        rec['api'] = {'exception': type(exc).__name__ + ': ' + str(exc)[:300]}
        rec['api_categories'] = ['EXC:' + type(exc).__name__]
        (cdir / 'api_traceback.txt').write_text(traceback.format_exc()[-4000:])
    (cdir / 'api.json').write_text(json.dumps(rec['api'], indent=2, sort_keys=True))
    api_o = run_api_O(ws)
    rec['api_O'] = api_o
    (cdir / 'api_O.json').write_text(json.dumps(api_o, indent=2, sort_keys=True))
    rec['api_O_equal'] = (api_o == rec['api'])
    for tag, opt in (('cli_normal', False), ('cli_O', True)):
        cmd, rc, out, err = run_cli(ws, opt)
        (cdir / f'{tag}.cmd').write_text(' '.join(cmd) + '\n')
        (cdir / f'{tag}.rc').write_text(str(rc) + '\n')
        (cdir / f'{tag}.stdout').write_text(out)
        (cdir / f'{tag}.stderr').write_text(err)
        try:
            parsed = json.loads(out)
        except ValueError:
            parsed = None
        rec[tag] = {'rc': rc, 'stdout_matches_api': (parsed == rec['api']),
                    'stdout_byte_equal_expected': (out == json.dumps(rec['api'], indent=2, sort_keys=True) + '\n')
                    if 'exception' not in rec['api'] else None,
                    'stderr_tail': err.strip().splitlines()[-1] if err.strip() else '',
                    'violations': parsed.get('violations') if isinstance(parsed, dict) else None}
    after = snapshot(ws)
    rec['input_unchanged'] = (before == after)
    (cdir / 'record.json').write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


VALUES = [
    ('absent', None), ('true', 'true'), ('false', 'false'), ('0', '0'), ('1', '1'), ('0.0', '0.0'),
    ('1.0', '1.0'), ('-0.0', '-0.0'), ('""', '""'), ('"true"', '"true"'), ('"false"', '"false"'),
    ('null', 'null'), ('[]', '[]'), ('[true]', '[true]'), ('{}', '{}'),
    ('{"ssa":true}', '{' + M + ':true}'), ('2', '2'), ('"1"', '"1"'),
]


SLUG = {'absent': 'absent', 'true': 'true', 'false': 'false', '0': 'int0', '1': 'int1', '0.0': 'float0',
        '1.0': 'float1', '-0.0': 'negzero', '""': 'str_empty', '"true"': 'str_true', '"false"': 'str_false',
        'null': 'null', '[]': 'list_empty', '[true]': 'list_true', '{}': 'obj_empty',
        '{"ssa":true}': 'obj_marker_true', '2': 'int2', '"1"': 'str_1'}
assert len(set(SLUG.values())) == len(SLUG)


def marker_obj(v):
    return '{"k":"v"}' if v is None else '{"k":"v",' + M + ':' + v + '}'


PLACEMENTS = [
    ('root', 'root object', lambda d: d),
    ('nested2', 'object two levels deep {"a":{"b":D}}', lambda d: '{"a":{"b":' + d + '}}'),
    ('list_root', 'object inside a root list [D]', lambda d: '[' + d + ']'),
    ('list_in_dict', 'object inside a list inside the root object {"items":[D]}', lambda d: '{"items":[' + d + ']}'),
    ('deep_mixed', 'object inside list/list/dict/list/dict {"a":[{"b":[[D]]}]}', lambda d: '{"a":[{"b":[[' + d + ']]}]}'),
]


def nest(depth: int, inner: str, kind: str):
    if kind == 'dict':
        return '{"a":' * depth + inner + '}' * depth
    return '[' * depth + inner + ']' * depth


def extras():
    E = []
    def add(cid, desc, text=None, files=None):
        E.append((cid, desc, files if files is not None else {'x.json': text.encode('utf-8') if isinstance(text, str) else text}))
    # controls
    add('ctrl_empty_object', 'control: generated/x.json = {}', '{}')
    add('ctrl_no_generated_dir', 'control: no generated directory (test fixture as-is)', files={})
    # malformed containers carrying a nested marker
    add('mc_dict_true', 'malformed container: {"ssa":{"ssa":true}}', '{' + M + ':{' + M + ':true}}')
    add('mc_list_true', 'malformed container list: {"ssa":[{"ssa":true}]}', '{' + M + ':[{' + M + ':true}]}')
    add('mc_dict_one', 'malformed container with nested 1: {"ssa":{"ssa":1}}', '{' + M + ':{' + M + ':1}}')
    add('mc_list_one', 'malformed container list with nested 1: {"ssa":[{"ssa":1}]}', '{' + M + ':[{' + M + ':1}]}')
    add('mc_deep_true', 'malformed container 3-deep: {"ssa":{"ssa":{"ssa":true}}}', '{' + M + ':{' + M + ':{' + M + ':true}}}')
    add('false_plus_nested_true', 'outer false, sibling nested true: {"ssa":false,"x":{"ssa":true}}', '{' + M + ':false,"x":{' + M + ':true}}')
    # siblings
    add('sib_true_then_1', 'siblings {"a":{"ssa":true},"b":{"ssa":1}}', '{"a":{' + M + ':true},"b":{' + M + ':1}}')
    add('sib_1_then_true', 'siblings {"a":{"ssa":1},"b":{"ssa":true}}', '{"a":{' + M + ':1},"b":{' + M + ':true}}')
    add('sib_list_true_1', 'list siblings [{"ssa":true},{"ssa":1}]', '[{' + M + ':true},{' + M + ':1}]')
    add('sib_list_1_true', 'list siblings [{"ssa":1},{"ssa":true}]', '[{' + M + ':1},{' + M + ':true}]')
    # duplicate keys
    add('dup_false_true', 'duplicate keys {"ssa":false,"ssa":true}', '{' + M + ':false,' + M + ':true}')
    add('dup_true_false', 'duplicate keys {"ssa":true,"ssa":false}', '{' + M + ':true,' + M + ':false}')
    add('dup_1_false', 'duplicate keys {"ssa":1,"ssa":false}', '{' + M + ':1,' + M + ':false}')
    add('dup_false_1', 'duplicate keys {"ssa":false,"ssa":1}', '{' + M + ':false,' + M + ':1}')
    add('dup_nested_false_true', 'nested duplicate {"a":{"ssa":false,"ssa":true}}', '{"a":{' + M + ':false,' + M + ':true}}')
    add('dup_nested_true_false', 'nested duplicate {"a":{"ssa":true,"ssa":false}}', '{"a":{' + M + ':true,' + M + ':false}}')
    # non-object roots
    for cid, txt in (('root_list_empty', '[]'), ('root_list_true', '[true]'), ('root_list_marker_string', '["scientific_status_authority",true]'),
                     ('root_scalar_true', 'true'), ('root_scalar_false', 'false'), ('root_scalar_1', '1'),
                     ('root_scalar_0', '0'), ('root_scalar_null', 'null'), ('root_scalar_str', '"scientific_status_authority"'),
                     ('root_scalar_float', '1.5'), ('root_scalar_emptystr', '""'), ('root_scalar_NaN', 'NaN')):
        add(cid, 'non-object root: ' + txt, txt)
    # NaN / Infinity literals (Python json extension, not RFC 8259 JSON)
    for cid, lit in (('nan_root', 'NaN'), ('inf_root', 'Infinity'), ('neginf_root', '-Infinity')):
        add(cid, 'marker literal ' + lit + ' at root object', '{' + M + ':' + lit + '}')
    add('nan_nested2', 'marker NaN two levels deep', '{"a":{"b":{' + M + ':NaN}}}')
    add('nan_list_root', 'marker NaN in root list', '[{' + M + ':NaN}]')
    add('nan_with_nested_true', 'NaN marker with nested true sibling', '{' + M + ':NaN,"x":{' + M + ':true}}')
    # key spelling (exact-key behaviour, outside the requested matrix)
    add('key_unicode_escape', 'key spelled with \\u escape for "a": decodes to exact key', '{"scientific_status_\\u0061uthority":true}')
    add('key_case_variant', 'key case variant {"Scientific_Status_Authority":true}', '{"Scientific_Status_Authority":true}')
    add('key_trailing_space', 'key with trailing space {"scientific_status_authority ":true}', '{"scientific_status_authority ":true}')
    # file scope / glob (outside the requested matrix)
    add('multi_file', 'two files: x.json true, y.json 1, sub/z.json true',
        files={'x.json': ('{' + M + ':true}').encode(), 'y.json': ('{' + M + ':1}').encode(),
               'sub/z.json': ('{' + M + ':true}').encode()})
    add('ext_upper', 'X.JSON (upper-case extension) with true', files={'X.JSON': ('{' + M + ':true}').encode()})
    add('hidden_file', '.x.json (dot file) with true', files={'.x.json': ('{' + M + ':true}').encode()})
    add('hidden_dir', '.hid/x.json (dot dir) with true', files={'.hid/x.json': ('{' + M + ':true}').encode()})
    # parse boundary
    add('parse_truncated', 'truncated JSON "{"', '{')
    add('parse_empty_file', 'empty file', '')
    add('parse_bom', 'UTF-8 BOM before {"ssa":true}', b'\xef\xbb\xbf' + ('{' + M + ':true}').encode())
    add('parse_bad_utf8', 'invalid UTF-8 byte inside a string', b'{"k":"\xff",' + M.encode() + b':true}')
    # nesting depth (traversal completeness)
    for depth in (100, 300, 400, 450, 490, 500, 600, 900, 1000, 2000, 5000, 100000):
        add(f'depth_dict_{depth}', f'true marker under {depth} nested objects', nest(depth, '{' + M + ':true}', 'dict'))
    for depth in (450, 490, 500, 900, 1000, 5000):
        add(f'depth_list_{depth}', f'true marker under {depth} nested lists', nest(depth, '{' + M + ':true}', 'list'))
    for depth in (490, 500, 1000, 5000):
        add(f'depth_dict_benign_{depth}', f'no marker, {depth} nested objects', nest(depth, '{"k":"v"}', 'dict'))
    return E


def main():
    CASES.mkdir(exist_ok=True)
    records = []
    for vname, vlit in VALUES:
        for pname, pdesc, wrap in PLACEMENTS:
            text = wrap(marker_obj(vlit))
            cid = 'm__' + pname + '__' + SLUG[vname]
            rec = run_case(cid, {'x.json': text.encode()}, {'kind': 'matrix', 'value': vname, 'placement': pname,
                                                           'placement_desc': pdesc})
            records.append(rec)
            print(cid, rec['api_categories'], rec['cli_normal']['rc'], rec['cli_O']['rc'], flush=True)
    for cid, desc, files in extras():
        rec = run_case('x__' + cid, files, {'kind': 'extra', 'desc': desc})
        records.append(rec)
        print('x__' + cid, rec['api_categories'], rec['cli_normal']['rc'], rec['cli_O']['rc'], flush=True)
    (W / 'results.json').write_text(json.dumps(records, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
