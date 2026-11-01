"""Execute the unchanged runner entry against an explicitly named source fixture."""
import argparse, importlib.util, sys, json, hashlib, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--block-three',action='store_true');p.add_argument('--no-reflow',action='store_true');a=p.parse_args()
repo=Path('/workspace/atlas-hosted-admission-20261029');sys.path.insert(0,str(repo/'tools'))
spec=importlib.util.spec_from_file_location('atlas_hosted_entry',repo/'tools/current_site_browser_check.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.ROOT=a.fixture
if a.block_three or a.no_reflow:
 from playwright.sync_api import Browser
 original=Browser.new_context
 def new_context(self,*args,**kwargs):
  context=original(self,*args,**kwargs)
  if a.block_three:context.route('**/three.module.js*',lambda route:route.abort())
  if a.no_reflow:
   def no_reflow(route):
    response=route.fetch();raw=response.body();old=b'onChange({write:false,graphOnly:true});'
    if raw.count(old)!=1:raise RuntimeError('Ambiguous no-reflow mutation')
    route.fulfill(response=response,body=raw.replace(old,b''))
   context.route('**/atlas-spatial-shell.mjs*',no_reflow)
  return context
 Browser.new_context=new_context
sys.argv=[str(repo/'tools/current_site_browser_check.py'),'--browser','/usr/bin/chromium','--output',str(a.output)]
started=time.perf_counter()
try:raise SystemExit(m.main())
finally:
 a.output.mkdir(parents=True,exist_ok=True)
 binding={'fixture':str(a.fixture),'code_root':str(repo),'block_three':a.block_three,'no_reflow':a.no_reflow,'elapsed_seconds':time.perf_counter()-started,'scope':'Wrapper changes only source fixture root, argv, and optional real Three.js network abort; existing entry assertions unchanged.','source_files':{str(q.relative_to(repo)):{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()}for q in [repo/'tools/current_site_browser_check.py',repo/'tools/atlas_spatial_admission.py']}}
 (a.output/'CONTROL-BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
