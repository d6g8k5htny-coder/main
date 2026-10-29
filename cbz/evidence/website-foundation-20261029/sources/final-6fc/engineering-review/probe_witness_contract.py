from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util,json,os,subprocess
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/review-tools')
spec=importlib.util.spec_from_file_location('cbz',ROOT/'tools/cbz_lean_check.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lean=Path('/workspace/research/audit/formal/toolchain/bin/lean');report={}
with TemporaryDirectory(prefix='cbz-review-witness-',dir=OUT) as tmp:
 p=Path(tmp);source='import Lean\nnamespace CBZNonvacuity\n'+''.join('theorem '+name.split('.')[-1]+' : True := True.intro\n' for name in m.WITNESSES)+'end CBZNonvacuity\n'
 (p/'FakeWitness.lean').write_text(source)
 (p/'Audit.lean').write_text(m.audit_program((ROOT/'tools/CBZNonvacuity.lean').read_text(),'FakeWitness'))
 env=dict(os.environ,LEAN_PATH=str(p))
 compile=subprocess.run([str(lean),'-o',str(p/'FakeWitness.olean'),str(p/'FakeWitness.lean')],cwd=p,env=env,capture_output=True,text=True)
 audit=subprocess.run([str(lean),str(p/'Audit.lean')],cwd=p,env=env,capture_output=True,text=True)
 report={'scope':'Real Lean disposable audit stage only, not a full CBZ wrapper replay or scientific truth review','compile_exit':compile.returncode,'compile_output':compile.stdout+compile.stderr,'audit_exit':audit.returncode,'audit_output':audit.stdout+audit.stderr}
 if compile.returncode==0 and audit.returncode==0:
  result=m.validate_census(audit.stdout,set(m.WITNESSES),{'FakeWitness'})
  report['witness_census_accepted']=True;report['statements']={row['name']:row['statement'] for row in result['proofs']}
(OUT/'witness-contract-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
