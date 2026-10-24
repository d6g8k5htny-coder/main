import sys,json,subprocess,tempfile,importlib.util,pathlib
root0=pathlib.Path(sys.argv[1]); sys.path.insert(0,str(root0))
spec=importlib.util.spec_from_file_location('t',root0/'tests/test_architecture_conformance.py'); t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
for py in ('python3','python3.11'):
  for n,val in ((900,'true'),(300,'true'),(900,'1')):
    d=tempfile.mkdtemp(); root=pathlib.Path(d)
    for name in t.REPOS:(root/name).mkdir(parents=True)
    registry={'schema_version':1,'scientific_status_authority':False,'repositories':{name:{'full_name':'d6g8k5htny-coder/'+name,'role':name,'visibility':'private' if name=='sandbox' else 'public'} for name in t.REPOS},'artifacts':[]}
    (root/'meta-framework/registry.json').write_text(json.dumps(registry))
    a=root/'main/architecture/scientific_state/v1';a.mkdir(parents=True);(a/'AUTHORITY_MAP.json').write_text(json.dumps(t.valid_authority()))
    g=root/'meta-framework/generated'; g.mkdir()
    (g/'x.json').write_text('{"a":'*n+'{"scientific_status_authority":'+val+'}'+'}'*n)
    p=subprocess.run([py,'-B','-S',str(root0/'tools/architecture_conformance.py'),'--workspace',d],capture_output=True,text=True)
    print(py,n,val,'exit',p.returncode,'stdout',len(p.stdout),'stderr',p.stderr.strip().splitlines()[-1:] if p.stderr else '', (json.loads(p.stdout)['violations'] if p.stdout else ''))
