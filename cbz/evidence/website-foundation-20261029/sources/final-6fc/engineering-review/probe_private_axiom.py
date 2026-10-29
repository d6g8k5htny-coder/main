from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util,json,os,subprocess
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/review-tools');lean='/workspace/research/audit/formal/toolchain/bin/lean'
spec=importlib.util.spec_from_file_location('cbz',ROOT/'tools/cbz_lean_check.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report=[]
with TemporaryDirectory(prefix='cbz-review-private-',dir=OUT) as tmp:
 p=Path(tmp);env=dict(os.environ,LEAN_PATH=str(p))
 for label,header in [('legacy',''),('module','module\n')]:
  name='Private_'+label;(p/(name+'.lean')).write_text(header+'import Lean\nprivate axiom hidden : False\ntheorem honest : True := True.intro\n')
  c=subprocess.run([lean,'-o',str(p/(name+'.olean')),str(p/(name+'.lean'))],cwd=p,env=env,capture_output=True,text=True)
  row={'case':label,'compile_exit':c.returncode,'compile_log':c.stdout+c.stderr}
  if c.returncode==0:
   a=p/('Audit_'+label+'.lean');a.write_text(m.audit_program((ROOT/'tools/CBZNonvacuity.lean').read_text(),name))
   r=subprocess.run([lean,str(a)],cwd=p,env=env,capture_output=True,text=True);row.update(audit_exit=r.returncode,audit_log=r.stdout+r.stderr)
  report.append(row)
(OUT/'private-axiom-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
