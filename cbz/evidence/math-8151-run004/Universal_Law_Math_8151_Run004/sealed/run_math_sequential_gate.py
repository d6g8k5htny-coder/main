"""Execute the unchanged pinned Math gate with sequential exact-module replay.

Only execution scheduling/launching changes: build every registered module in sequence,
then confirm the ordinary default build; replace the single prefix-replay call
with the pinned LeanChecker.replayFromImports API once per exact module. All
original source/axiom/inventory/admission/negative-control gates still execute.
A 7GiB per-process-group RSS guard and900s per-command timeout fail closed.
Lean subprocesses use the same pinned executable and10 documented Lake environment
fields captured once per working directory, eliminating a resident Lake parent.
This is named-module kernel replay, NOT --fresh dependency-closure replay.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,re,shutil,signal,subprocess,time,sys,platform

O=Path('/workspace/cbz0/lean-foundation');R=Path('/workspace/cbz0/repos/Math-/formal')
T=Path('/workspace/research/audit/formal/toolchain/bin')
ENV=dict(os.environ,PATH=str(T)+os.pathsep+os.environ.get('PATH',''),LEAN_NUM_THREADS='1')
RSS_LIMIT=7*1024**3;TIME_LIMIT=900
index=1
while (O/f'math-run-{index:03}').exists():index+=1
OUT=O/f'math-run-{index:03}';OUT.mkdir()
records=[];LAKE_ENVIRONMENTS={}

def require(value,msg):
 if not value:raise RuntimeError(msg)

def hashes(root):
 return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in root.rglob('*')if p.is_file()and'.lake'not in p.parts and'__pycache__'not in p.parts}

def group_usage(pgid):
 stats=subprocess.check_output(['ps','-eo','pgid=,stat=,rss='],text=True)
 live=[row.split() for row in stats.splitlines()if int(row.split()[0])==pgid and not row.split()[1].startswith('Z')]
 return sum(int(row[2])*1024 for row in live),len(live)

def lake_environment_keys():
 return ('LAKE','LAKE_HOME','LEAN_SYSROOT','LEAN_AR','LEAN_CC','LEAN_PATH','LEAN_SRC_PATH','PATH','DYLD_LIBRARY_PATH','LD_LIBRARY_PATH')

def unique_lake_pairs(pairs):
 result={}
 for key,value in pairs:
  require(key not in result,'duplicate Lake environment key')
  result[key]=value
 return result

def lean_launch(command,label,cwd):
 if command[:3]!=['lake','env','lean']:return list(command),ENV,None
 cache_key=str(Path(cwd).resolve())
 if cache_key not in LAKE_ENVIRONMENTS:
  keys=lake_environment_keys()
  program='import os,json;print(json.dumps({k:os.environ.get(k) for k in '+repr(keys)+'},sort_keys=True))'
  code,text=run_process([str(T/'lake'),'env',sys.executable,'-c',program],label+'-lake-environment',cwd)
  require(code==0,'Lake environment capture failed')
  captured=json.loads(text,object_pairs_hook=unique_lake_pairs)
  require(isinstance(captured,dict)and set(captured)==set(keys),'Lake environment field set')
  require(all(value is None or isinstance(value,str)for value in captured.values()),'Lake environment field value')
  require(captured['LEAN_SYSROOT']==str(T.parent)and captured['LAKE']==str(T/'lake'),'Lake environment toolchain')
  require(bool(captured['LEAN_PATH']),'Lake environment missing Lean path')
  LAKE_ENVIRONMENTS[cache_key]=captured
 captured=LAKE_ENVIRONMENTS[cache_key];env=dict(ENV)
 for key,value in captured.items():
  if value is None:env.pop(key,None)
  else:env[key]=value
 return [str(T/'lean'),*command[3:]],env,captured

def stop_group(p,grace_seconds=2):
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 monitor_error=None
 try:
  deadline=time.monotonic()+grace_seconds
  while time.monotonic()<deadline:
   p.poll()
   _,live=group_usage(p.pid)
   if not live:break
   time.sleep(.05)
  _,live=group_usage(p.pid)
 except BaseException as exc:
  monitor_error=exc;live=1
 if live or monitor_error:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 p.wait(timeout=5)
 if monitor_error:raise monitor_error
 _,live=group_usage(p.pid)
 require(live==0,'task-owned process group cleanup incomplete')

def run_process(args,label,cwd):
 path=OUT/(re.sub(r'[^A-Za-z0-9_.-]','_',label)+'.log');start=time.monotonic();peak=0;reason=None;error=None;cleanup_error=None
 executed,process_env,lake_fields=lean_launch(args,label,cwd)
 with path.open('w')as log:
  p=subprocess.Popen(executed,cwd=cwd,env=process_env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   while p.poll()is None:
    rss,_=group_usage(p.pid)
    peak=max(peak,rss)
    if rss>RSS_LIMIT:reason='RSS_LIMIT';break
    if time.monotonic()-start>TIME_LIMIT:reason='COMMAND_TIMEOUT';break
    time.sleep(.25)
   if reason is None:
    rss,live=group_usage(p.pid);peak=max(peak,rss)
    if live:reason='DESCENDANTS_AFTER_EXIT'
  except BaseException as exc:
   error=exc;reason='MONITOR_EXCEPTION'
  finally:
   if reason:
    try:stop_group(p)
    except BaseException as exc:cleanup_error=exc
   code=p.wait(timeout=5)
 row={'args':[str(x)for x in args],'executed_args':[str(x)for x in executed],'lake_environment':lake_fields,'label':label,'cwd':str(cwd),'exit_code':code,'stop_reason':reason,'process_group_id':p.pid,'peak_group_rss_bytes':peak,'elapsed_seconds':time.monotonic()-start,'log':str(path),'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};records.append(row)
 (OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(label,'exit',code,'RSSMiB',round(peak/1024**2,1),'seconds',round(row['elapsed_seconds'],1),flush=True)
 if error:raise error
 if cleanup_error:raise cleanup_error
 require(reason is None,'resource boundary: '+label+' '+str(reason))
 return code,path.read_text()

spec=importlib.util.spec_from_file_location('math_gate',R/'gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
m,digest=gate.source_check();before=hashes(R)
MODULES=[p[:-5].replace('/','.')for p in m['source_modules']]+['ResearchFormalCoreR1']
require(len(set(MODULES))==len(MODULES),'duplicate module list')
require(all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*',n)for n in MODULES),'invalid module name')
TEMPLATE=(O/'OneModuleReplay.lean').read_text()
ADAPTER_INPUTS=[Path(__file__),O/'OneModuleReplay.lean',O/'LeanChecker-pinned.lean',T/'lean',T/'lake',T/'leanchecker',T.parent/'lib/lean/LeanChecker.olean',T.parent/'lib/lean/libleanshared.so']
ADAPTER_HASHES={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in ADAPTER_INPUTS}

def exact_replay(module,label,cwd):
 require(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*',module),'unsafe module name')
 source=OUT/(label.replace('/','_')+'.lean');source.write_text(TEMPLATE.replace('CBZ_EXACT_MODULE',module))
 code,log=run_process(['lake','env','lean',str(source)],label,cwd)
 return code,log

# Actual scheduling controls: exact existing module succeeds; absent module fails
# for the module identity diagnostic. They do not certify any Math theorem.
probe=Path('/workspace/universal-law/website-repair-20261028/formal')
code,log=exact_replay('UniversalLaw.Side24.Endpoints','selftest-existing-exact-module',probe)
require(code==0 and'CBZ_EXACT_REPLAY_PASS UniversalLaw.Side24.Endpoints'in log,'exact replay positive control failed')
code,log=exact_replay('CBZMissingExactModule','selftest-missing-exact-module',probe)
require(code!=0 and('unknown module prefix'in log or'does not exist'in log),'missing replay control failed for unrelated reason')

# Keep the current gate and every check; bound each phase and serialize only
# the documented build and prefix-replay operations, including admission copies.
def sequential_run(command,label,out,expect_success=True,cwd=None):
 cwd=R if cwd is None else Path(cwd)
 logs=[]
 if command==['lake','build']:
  for i,module in enumerate(MODULES):
   code,text=run_process(['lake','build','+'+module],f'{label}-module-{i:02}',cwd);logs.append(text)
   require(code==0,'registered module build failed: '+module)
  code,text=run_process(command,label+'-default-confirmation',cwd);logs.append(text)
 elif command==['lake','env','leanchecker','ResearchFormalCoreR1']:
  for i,module in enumerate(MODULES):
   code,text=exact_replay(module,f'{label}-exact-{i:02}',cwd);logs.append(text)
   require(code==0,'exact named-module kernel replay failed: '+module)
  code=0
 else:
  code,text=run_process(command,label,cwd);logs.append(text)
 text='\n'.join(logs);(out/(label+'.log')).write_text(text)
 require((code==0)==expect_success,'unexpected original gate phase outcome: '+label)
 return text

gate.run=sequential_run
evidence=R/'.lake/formal-evidence'
if evidence.exists():shutil.move(str(evidence),OUT/'preexisting-formal-evidence')
result={'status':'RUNNING','source_root':str(R),'manifest_effective_sha256':digest,'exact_modules':MODULES,'execution_change':'Sequential builds plus default-build confirmation; pinned LeanChecker exact-module API called in separate resource-bounded processes instead of concurrent prefix replay. No original predicate/control disabled. No --fresh Math dependency-closure claim.','source_hashes_before':before,'adapter_input_sha256':ADAPTER_HASHES,'execution_environment':{'python':sys.version,'platform':platform.platform(),'LEAN_NUM_THREADS':ENV['LEAN_NUM_THREADS']},'rss_limit_bytes':RSS_LIMIT,'command_timeout_seconds':TIME_LIMIT,'pinned_gate_sha256':hashlib.sha256((R/'gate.py').read_bytes()).hexdigest(),'pinned_gate_helper_sha256':hashlib.sha256((R/'gate_helper.py').read_bytes()).hexdigest()}
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 receipt=gate.execute(m,digest)
 require(hashes(R)==before,'pinned Math source bytes changed during execution')
 require(all(hashlib.sha256(p.read_bytes()).hexdigest()==ADAPTER_HASHES[str(p)]for p in ADAPTER_INPUTS),'adapter or pinned runtime bytes changed during execution')
 result.update(status='PASS_CURRENT_PINNED_GATE_WITH_SEQUENTIAL_EXECUTION',receipt=receipt,source_unchanged=True)
except BaseException as exc:
 live_receipt=R/'.lake/formal-evidence/receipt.json'
 if live_receipt.exists():shutil.move(str(live_receipt),OUT/'refused-inner-receipt.json')
 result.update(status='INCOMPLETE_OR_FAILED',error_type=type(exc).__name__,error=str(exc),source_unchanged=hashes(R)==before)
finally:
 result['commands']=records
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('FINAL',result['status'],OUT,flush=True)
raise SystemExit(0 if result['status'].startswith('PASS_')else 1)
