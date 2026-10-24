"""Probe the unchanged checker's generated-JSON marker behaviour (API + CLI, normal and -O)."""
import json, subprocess, sys, tempfile, hashlib
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))
from tools.architecture_conformance import check_workspace  # noqa: E402

REPOS = ['Math-', 'google-drive', 'governance-', 'main', 'meta-framework', 'query-', 'sandbox', 'trial']
AUTH = {'required_authority_ids': ['claims_firewall', 'scientific_state_architecture'],
        'authorities': {'claims_firewall': {'id': 'claims_firewall', 'owns': ['grade']}},
        'this_package': {'id': 'scientific_state_architecture', 'owns': ['schema_contract'],
                         'never_writes': ['status', 'grade', 'classification', 'controlling', 'lemma_closed',
                                          'prizes_solved', 'independence_credit']}}


def make_ws(root: Path):
    for n in REPOS:
        (root / n).mkdir(parents=True, exist_ok=True)
    reg = {'schema_version': 1, 'scientific_status_authority': False,
           'repositories': {n: {'full_name': 'd6g8k5htny-coder/' + n, 'role': n,
                                'visibility': 'private' if n == 'sandbox' else 'public'} for n in REPOS},
           'artifacts': []}
    (root / 'meta-framework/registry.json').write_text(json.dumps(reg))
    a = root / 'main/architecture/scientific_state/v1'
    a.mkdir(parents=True, exist_ok=True)
    (a / 'AUTHORITY_MAP.json').write_text(json.dumps(AUTH))


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}


def cli(root, opt):
    flags = ['-B', '-S'] + (['-O'] if opt else [])
    p = subprocess.run([sys.executable, *flags, str(SRC / 'tools/architecture_conformance.py'), '--workspace', str(root)],
                       capture_output=True, text=True, timeout=60)
    try:
        rep = json.loads(p.stdout)
    except Exception:
        rep = None
    return p.returncode, rep, p.stderr


CASES = [
    ('absent (no generated dir)', None),
    ('empty object', b'{}'),
    ('root false', b'{"scientific_status_authority": false}'),
    ('root true', b'{"scientific_status_authority": true}'),
    ('nested-dict true (test L73-74 shape)', b'{"nested": {"scientific_status_authority": true}}'),
    ('nested-list true', b'{"nested": [{"scientific_status_authority": true}]}'),
    ('root-list true', b'[{"scientific_status_authority": true}]'),
    ('deep list-in-list true', b'{"a": [[[{"b": {"scientific_status_authority": true}}]]]}'),
    ('nested false', b'{"nested": [{"scientific_status_authority": false}]}'),
    ('int 1', b'{"nested": [{"scientific_status_authority": 1}]}'),
    ('int 0', b'{"nested": [{"scientific_status_authority": 0}]}'),
    ('float 1.0', b'{"nested": [{"scientific_status_authority": 1.0}]}'),
    ('float 0.0', b'{"nested": [{"scientific_status_authority": 0.0}]}'),
    ('string "true"', b'{"nested": [{"scientific_status_authority": "true"}]}'),
    ('string ""', b'{"nested": [{"scientific_status_authority": ""}]}'),
    ('null', b'{"nested": [{"scientific_status_authority": null}]}'),
    ('empty list', b'{"scientific_status_authority": []}'),
    ('list [true]', b'{"scientific_status_authority": [true]}'),
    ('empty object value', b'{"scientific_status_authority": {}}'),
    ('NaN', b'{"scientific_status_authority": NaN}'),
    ('Infinity', b'{"scientific_status_authority": Infinity}'),
    ('-Infinity', b'{"scientific_status_authority": -Infinity}'),
    ('container marker holding nested true',
     b'{"scientific_status_authority": {"scientific_status_authority": true}}'),
    ('list marker holding dict true', b'{"scientific_status_authority": [{"scientific_status_authority": true}]}'),
    ('dup keys true then false', b'{"scientific_status_authority": true, "scientific_status_authority": false}'),
    ('dup keys false then true', b'{"scientific_status_authority": false, "scientific_status_authority": true}'),
    ('escaped key true', b'{"scientific\\u005fstatus_authority": true}'),
    ('case-variant key true', b'{"Scientific_Status_Authority": true}'),
    ('key with trailing space true', b'{"scientific_status_authority ": true}'),
    ('root scalar true', b'true'),
    ('root scalar 1', b'1'),
    ('root null', b'null'),
    ('utf-8 BOM + true', b'\xef\xbb\xbf{"scientific_status_authority": true}'),
    ('utf-8 BOM + false', b'\xef\xbb\xbf{"scientific_status_authority": false}'),
    ('invalid utf-8', b'{"x": "\xff"}'),
    ('utf-16 true', '{"scientific_status_authority": true}'.encode('utf-16')),
    ('malformed json', b'{"scientific_status_authority": tru'),
    ('payload-bearing parse error', b'{"secret_payload_value_XYZ": tru'),
    ('true marker after malformed sibling marker',
     b'{"a": {"scientific_status_authority": "yes"}, "b": {"scientific_status_authority": true}}'),
]


def run_case(name, raw, filename='X.json', subdir=''):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        make_ws(root)
        if raw is not None:
            g = root / 'meta-framework/generated' / subdir
            g.mkdir(parents=True, exist_ok=True)
            (g / filename).write_bytes(raw)
        before = snapshot(root)
        try:
            api = check_workspace(root)
            api_s = (api['ok'], api['violations'])
        except BaseException as e:  # noqa
            api_s = ('EXC', type(e).__name__ + ': ' + str(e)[:120])
        rc0, rep0, err0 = cli(root, False)
        rc1, rep1, err1 = cli(root, True)
        after = snapshot(root)
        same = (rep0 == rep1) and rc0 == rc1
        print(f'{name!r:52} API={api_s!r}')
        print(f'{"":52} CLI normal rc={rc0} optimized rc={rc1} reports_equal={same} '
              f'stderr_empty={err0 == "" and err1 == ""} inputs_unchanged={before == after}')
        if err0:
            print(f'{"":52} stderr tail: {err0.strip().splitlines()[-1][:160]}')


if __name__ == '__main__':
    for name, raw in CASES:
        run_case(name, raw)
    print('--- discovery probes')
    run_case('uppercase .JSON extension true', b'{"scientific_status_authority": true}', filename='X.JSON')
    run_case('hidden dotfile .x.json true', b'{"scientific_status_authority": true}', filename='.x.json')
    run_case('deep subdir a/b/c true', b'{"scientific_status_authority": true}', subdir='a/b/c')
    run_case('.jsonl extension true', b'{"scientific_status_authority": true}', filename='X.jsonl')
