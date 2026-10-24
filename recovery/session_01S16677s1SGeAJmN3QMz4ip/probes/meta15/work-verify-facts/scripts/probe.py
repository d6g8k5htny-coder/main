"""Fact-check probe: build synthetic workspaces from the test setUp fixture plus generated files,
run the real CLI (normal and -O) and the in-process API, print compact results."""
import json, subprocess, sys, tempfile, os, hashlib
from pathlib import Path

SRC = Path(sys.argv[1])
PY = sys.argv[2]
SCRIPT = SRC / 'tools/architecture_conformance.py'
REPOS = ['Math-','google-drive','governance-','main','meta-framework','query-','sandbox','trial']
SSA = '"scientific_status_authority"'

def valid_authority():
    return {'required_authority_ids':['claims_firewall','scientific_state_architecture'],'authorities':{'claims_firewall':{'id':'claims_firewall','owns':['grade']}},'this_package':{'id':'scientific_state_architecture','owns':['schema_contract'],'never_writes':['status','grade','classification','controlling','lemma_closed','prizes_solved','independence_credit']}}

def make_ws(files):
    tmp = tempfile.mkdtemp(dir=str(Path(__file__).resolve().parents[1] / 'ws'))
    root = Path(tmp)
    for n in REPOS: (root/n).mkdir(parents=True)
    registry={'schema_version':1,'scientific_status_authority':False,'repositories':{n:{'full_name':'d6g8k5htny-coder/'+n,'role':n,'visibility':'private' if n=='sandbox' else 'public'} for n in REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(registry))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(valid_authority()))
    for rel, raw in files.items():
        p = root/'meta-framework/generated'/rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw if isinstance(raw, bytes) else raw.encode('utf-8'))
    return root

def gen_hashes(root):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}

def run_cli(root, opt):
    flags = ['-B','-S'] + (['-O'] if opt else [])
    env = dict(os.environ); env.pop('PYTHONOPTIMIZE', None)
    p = subprocess.run([PY, *flags, str(SCRIPT), '--workspace', str(root)], capture_output=True, text=True, env=env, timeout=120)
    return p

def run_api(root, opt):
    code = ('import sys,json;sys.path.insert(0,%r);from tools.architecture_conformance import check_workspace\n'
            'try:\n r=check_workspace(%r);print(json.dumps(r))\n'
            'except BaseException as e:\n print(json.dumps({"EXC":type(e).__name__}))') % (str(SRC), str(root))
    flags = ['-B','-S'] + (['-O'] if opt else [])
    env = dict(os.environ); env.pop('PYTHONOPTIMIZE', None)
    p = subprocess.run([PY, *flags, '-c', code], capture_output=True, text=True, env=env, timeout=120)
    return json.loads(p.stdout)

def short(v):
    out=[]
    for x in v:
        if x.startswith('generated scientific-status authority: '): out.append('AUTH('+x.split('/')[-1]+')')
        elif x.startswith('generated JSON parse error'): out.append('PARSE')
        else: out.append('OTHER:'+x)
    return ','.join(out) or 'none'

def summarize(name, files):
    root = make_ws(files)
    before = gen_hashes(root)
    res = []
    for opt in (False, True):
        p = run_cli(root, opt)
        api = run_api(root, opt)
        if 'EXC' in api:
            apis = 'EXC:'+api['EXC']
        else:
            apis = short(api['violations'])
            # stdout byte equality
            if p.returncode in (0,2) and p.stdout != json.dumps(api, indent=2, sort_keys=True)+'\n':
                apis += '[STDOUT MISMATCH]'
        tail = p.stderr.strip().splitlines()[-1] if p.stderr.strip() else ''
        res.append((apis, p.returncode, len(p.stdout), tail[:70]))
    after = gen_hashes(root)
    same = 'hash-ok' if before == after else 'HASH CHANGED'
    n, o = res
    print(f'{name:60s} | normal: api={n[0]} exit={n[1]} stdout={n[2]} {n[3]} | -O: api={o[0]} exit={o[1]} stdout={o[2]} {o[3]} | {same}')

def D(v):
    return '{"k":"v"}' if v is None else '{"k":"v",' + SSA + ':' + v + '}'

PLACE = {
    'root': lambda d: d,
    'nested': lambda d: '{"a":{"b":' + d + '}}',
    'rootlist': lambda d: '[' + d + ']',
    'listindict': lambda d: '{"items":[' + d + ']}',
    'deepmixed': lambda d: '{"a":[{"b":[[' + d + ']]}]}',
}

if __name__ == '__main__':
    mode = sys.argv[3]
    if mode == 'values':
        vals = [None,'false','true','0','1','2','0.0','1.0','-0.0','NaN','Infinity','-Infinity','""','"true"','"false"','"1"','null','[]','[true]','{}','{' + SSA + ':true}']
        for v in vals:
            for pn, pf in PLACE.items():
                summarize(f'V={v if v is not None else "absent"} @{pn}', {'x.json': pf(D(v))})
    elif mode == 'extra':
        cases = {
            'ssa:[{ssa:true}]': {'x.json': '{' + SSA + ':[{' + SSA + ':true}]}'},
            'ssa:false,x:{ssa:true}': {'x.json': '{' + SSA + ':false,"x":{' + SSA + ':true}}'},
            'siblings 1 then true': {'x.json': '{"a":{' + SSA + ':1},"b":{' + SSA + ':true}}'},
            'siblings true then 1': {'x.json': '{"a":{' + SSA + ':true},"b":{' + SSA + ':1}}'},
            'siblings list 1,true': {'x.json': '[{' + SSA + ':1},{' + SSA + ':true}]'},
            'root true': {'x.json': 'true'}, 'root 1': {'x.json': '1'}, 'root null': {'x.json': 'null'},
            'root ""': {'x.json': '""'}, 'root []': {'x.json': '[]'}, 'root [true]': {'x.json': '[true]'},
            'escaped key true': {'x.json': '{"\\u0073cientific_status_authority":true}'},
            'case key true': {'x.json': '{"Scientific_Status_Authority":true}'},
            'trailing space key true': {'x.json': '{"scientific_status_authority ":true}'},
            'dup true,false': {'x.json': '{' + SSA + ':true,' + SSA + ':false}'},
            'dup false,true': {'x.json': '{' + SSA + ':false,' + SSA + ':true}'},
            'dup true,null': {'x.json': '{' + SSA + ':true,' + SSA + ':null}'},
            'dup null,true': {'x.json': '{' + SSA + ':null,' + SSA + ':true}'},
            'dup 1,false': {'x.json': '{' + SSA + ':1,' + SSA + ':false}'},
            'dup false,1': {'x.json': '{' + SSA + ':false,' + SSA + ':1}'},
            'unrelated dup + nested true': {'x.json': '{"a":1,"a":2,"n":{' + SSA + ':true}}'},
            'truncated': {'x.json': '{"a":'}, 'empty': {'x.json': ''},
            'BOM': {'x.json': b'\xef\xbb\xbf{"a":1}'}, 'invalid utf8': {'x.json': b'{"a":"\xff"}'},
            'X.JSON true': {'X.JSON': '{' + SSA + ':true}'}, 'jsonl true': {'x.jsonl': '{' + SSA + ':true}'},
            'three files': {'x.json': '{' + SSA + ':true}', 'y.json': '{' + SSA + ':1}', 'sub/z.json': '{' + SSA + ':true}'},
            'hidden .x.json': {'.x.json': '{' + SSA + ':true}'},
            'hidden dir': {'.hid/x.json': '{' + SSA + ':true}'},
            'deep1000 then b true': {'x.json': '{"a":' + '{"a":'*1000 + '1' + '}'*1000 + ',"b":{' + SSA + ':true}}'},
            'b true then deep1000': {'x.json': '{"b":{' + SSA + ':true},"a":' + '{"a":'*1000 + '1' + '}'*1000 + '}'},
        }
        for n, f in cases.items(): summarize(n, f)
    elif mode == 'depth':
        for n in [int(x) for x in sys.argv[4].split(',')]:
            for inner in ('{' + SSA + ':true}', '{' + SSA + ':1}', '{"k":"v"}'):
                summarize(f'depth n={n} inner={inner[-6:]}', {'x.json': '{"a":'*n + inner + '}'*n})
