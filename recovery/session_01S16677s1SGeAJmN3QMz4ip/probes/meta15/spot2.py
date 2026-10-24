import sys,json,subprocess,tempfile,importlib.util,pathlib
root0=pathlib.Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('t',root0/'tests/test_architecture_conformance.py'); sys.path.insert(0,str(root0)); t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
def ws(text):
    d=tempfile.mkdtemp(); root=pathlib.Path(d)
    for name in t.REPOS:(root/name).mkdir(parents=True)
    registry={'schema_version':1,'scientific_status_authority':False,'repositories':{name:{'full_name':'d6g8k5htny-coder/'+name,'role':name,'visibility':'private' if name=='sandbox' else 'public'} for name in t.REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(registry))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(t.valid_authority()))
    g=root/'meta-framework/generated'; g.mkdir(); (g/'x.json').write_text(text); return d
def run(py,text):
    p=subprocess.run([py,'-B','-S',str(root0/'tools/architecture_conformance.py'),'--workspace',ws(text)],capture_output=True,text=True)
    return p.returncode, (json.loads(p.stdout)['violations'] if p.stdout else 'NOSTDOUT')
deep=lambda n,inner:'{"a":'*n+inner+'}'*n
T='{"scientific_status_authority":true}'
for py,ns in (('python3.11',(332,333,991,992)),('python3',(498,499))):
    for n in ns: print(py,n,run(py,deep(n,T)))
print('escaped key',run('python3','{"scientific_status_\\u0061uthority":true}'))
print('case key',run('python3','{"Scientific_Status_Authority":true}'))
print('two true',run('python3','{"a":{"scientific_status_authority":true},"b":[{"scientific_status_authority":true}]}'))
print('dup true,false',run('python3','{"scientific_status_authority":true,"scientific_status_authority":false}'))
print('Infinity root',run('python3','{"scientific_status_authority":Infinity}'))
