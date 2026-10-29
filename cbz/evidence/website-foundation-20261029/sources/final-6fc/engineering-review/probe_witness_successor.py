from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util,json,os,subprocess,hashlib
ROOT=Path('/workspace/universal-law/website-repair-20261028');OUT=Path('/workspace/cbz0/review-tools');lean='/workspace/research/audit/formal/toolchain/bin/lean'
spec=importlib.util.spec_from_file_location('cbz',ROOT/'tools/cbz_lean_check.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);source=(ROOT/'tools/CBZNonvacuity.lean').read_text();cases=[]
with TemporaryDirectory(prefix='cbz-review-witness-fixed-',dir=OUT) as tmp:
 p=Path(tmp);env=dict(os.environ,LEAN_PATH=str(p)+':'+str(ROOT/'formal/.lake/build/lib/lean'))
 for label in ['TrueOnly','ActualWitness']:
  content=source[:source.index('open Lean Meta Elab Command')] if label=='ActualWitness' else 'import Lean\nnamespace CBZNonvacuity\n'+''.join('theorem '+n.split('.')[-1]+' : True := True.intro\n' for n in m.WITNESSES)+'end CBZNonvacuity\n'
  (p/(label+'.lean')).write_text(content)
  c=subprocess.run([lean,'-o',str(p/(label+'.olean')),str(p/(label+'.lean'))],cwd=p,env=env,capture_output=True,text=True);assert c.returncode==0,c.stdout+c.stderr
  audit=p/('Audit'+label+'.lean');audit.write_text(m.audit_program(source,label));a=subprocess.run([lean,str(audit)],cwd=p,env=env,capture_output=True,text=True);assert a.returncode==0,a.stdout+a.stderr
  row={'case':label,'compile_exit':c.returncode,'audit_exit':a.returncode}
  try:m.validate_census(a.stdout,set(m.WITNESSES),{label});row['census_accepted']=True
  except ValueError as err:row.update(census_accepted=False,census_error=str(err))
  contract=p/('Contract'+label+'.lean');contract.write_text(m.witness_contract_program(label));k=subprocess.run([lean,str(contract)],cwd=p,env=env,capture_output=True,text=True);row.update(contract_exit=k.returncode,contract_output=k.stdout+k.stderr)
  assert row['census_accepted']==(label=='ActualWitness');assert (k.returncode==0)==(label=='ActualWitness')
  cases.append(row)
report={'scope':'Independent actual Lean audit-stage and exact-type controls, not a repeated full wrapper/kernel run','cases':cases,'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['tools/cbz_lean_check.py','tools/CBZNonvacuity.lean','tests/test_cbz_lean_check.py']}}
(OUT/'witness-successor-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
