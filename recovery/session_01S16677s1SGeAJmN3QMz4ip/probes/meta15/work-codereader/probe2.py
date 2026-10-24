"""Second probe: recursion depth, multiple files/order, symlinks, dir-named-json, registry predicate."""
import json, os, subprocess, sys, tempfile
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import make_ws, cli  # noqa: E402
from tools.architecture_conformance import check_workspace, _contains_status_authority  # noqa: E402

print('python', sys.version.split()[0], 'recursionlimit', sys.getrecursionlimit())


def ws_with(files: dict, extra=None):
    t = tempfile.TemporaryDirectory()
    root = Path(t.name)
    make_ws(root)
    for rel, raw in files.items():
        p = root / 'meta-framework/generated' / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    if extra:
        extra(root)
    return t, root


def summarize(root):
    try:
        r = check_workspace(root)
        api = (r['ok'], r['violations'])
    except BaseException as e:
        api = ('EXC', type(e).__name__)
    rc0, rep0, err0 = cli(root, False)
    rc1, rep1, err1 = cli(root, True)
    tail0 = err0.strip().splitlines()[-1][:140] if err0 else ''
    return api, (rc0, rep0 is not None, tail0), (rc1, rep1 is not None)


print('--- depth (nested lists wrapping a true marker): json.loads vs helper')
for depth in (300, 450, 500, 990, 2000, 5000, 20000):
    raw = '[' * depth + '{"scientific_status_authority": true}' + ']' * depth
    try:
        data = json.loads(raw)
        parsed = 'parsed'
    except RecursionError as e:
        parsed = 'json RecursionError'
        data = None
    helper = None
    if data is not None:
        try:
            helper = _contains_status_authority(data)
        except RecursionError:
            helper = 'helper RecursionError'
    t, root = ws_with({'deep.json': raw.encode()})
    api, c0, c1 = summarize(root)
    t.cleanup()
    print(f'depth={depth:6} load={parsed:20} helper={helper!s:22} API={str(api)[:110]}')
    print(f'{"":14} CLI normal rc={c0[0]} report={c0[1]} stderr_tail={c0[2]!r}  optimized rc={c1[0]} report={c1[1]}')

print('--- multiple files: one violation per file, order follows glob order')
t, root = ws_with({'b.json': b'{"scientific_status_authority": true}',
                   'a.json': b'{"x":[{"scientific_status_authority": true},{"scientific_status_authority": true}]}',
                   'c/z.json': b'{"scientific_status_authority": true}',
                   'bad.json': b'{'})
api, c0, c1 = summarize(root)
print('API', api)
print('glob order', [str(p.relative_to(root)) for p in (root / 'meta-framework/generated').glob('**/*.json')])
t.cleanup()

print('--- directory named x.json')
t, root = ws_with({}, extra=lambda r: (r / 'meta-framework/generated/dir.json').mkdir(parents=True))
print(summarize(root)[0]); t.cleanup()

print('--- generated is a regular file, not a directory')
def gen_file(r):
    (r / 'meta-framework/generated').write_text('{"scientific_status_authority": true}')
t, root = ws_with({}, extra=gen_file)
print(summarize(root)[0]); t.cleanup()

print('--- symlinked file pointing outside the workspace / symlinked subdirectory')
outside = tempfile.TemporaryDirectory()
od = Path(outside.name)
(od / 'evil.json').write_text('{"scientific_status_authority": true}')
(od / 'sub').mkdir()
(od / 'sub/inner.json').write_text('{"scientific_status_authority": true}')
def links(r):
    g = r / 'meta-framework/generated'
    g.mkdir(parents=True, exist_ok=True)
    os.symlink(od / 'evil.json', g / 'linkfile.json')
    os.symlink(od / 'sub', g / 'linkdir')
t, root = ws_with({}, extra=links)
print(summarize(root)[0]); t.cleanup()
def gen_symlink(r):
    os.symlink(od / 'sub', r / 'meta-framework/generated')
t, root = ws_with({}, extra=gen_symlink)
print('generated itself a symlink ->', summarize(root)[0]); t.cleanup()
outside.cleanup()

print('--- registry top-level predicate (line 60) by value type; and nested registry marker')
for label, val in [('missing', '<missing>'), ('false', False), ('true', True), ('0', 0), ('null', None),
                   ('"false"', 'false'), ('0.0', 0.0)]:
    def mut(r, val=val):
        p = r / 'meta-framework/registry.json'
        d = json.loads(p.read_text())
        if val == '<missing>':
            d.pop('scientific_status_authority')
        else:
            d['scientific_status_authority'] = val
        p.write_text(json.dumps(d))
    t, root = ws_with({}, extra=mut)
    print(f'registry {label:8}', check_workspace(root)['violations']); t.cleanup()

def dup_registry(r):
    p = r / 'meta-framework/registry.json'
    txt = p.read_text()
    p.write_text('{"scientific_status_authority": true, ' + txt[1:])
t, root = ws_with({}, extra=dup_registry)
print('registry dup key true then (later) false ->', check_workspace(root)['violations']); t.cleanup()

def nested_registry_true(r):
    p = r / 'meta-framework/registry.json'
    d = json.loads(p.read_text())
    d['repositories']['main']['scientific_status_authority'] = True
    d['extra'] = {'scientific_status_authority': True}
    p.write_text(json.dumps(d))
t, root = ws_with({}, extra=nested_registry_true)
print('registry nested true markers (not top-level) ->', check_workspace(root)['violations']); t.cleanup()

print('--- argparse usage error exit code')
p = subprocess.run([sys.executable, '-B', '-S', str(SRC / 'tools/architecture_conformance.py')], capture_output=True, text=True)
print('no --workspace rc', p.returncode, 'stdout empty', p.stdout == '', 'stderr', p.stderr.strip().splitlines()[-1])
