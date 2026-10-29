#!/usr/bin/env python3
"""Finite adversarial source-mutation verification for coverage_companion.py."""
import hashlib,json,os,pathlib,re,subprocess,sys,tempfile
base=pathlib.Path(__file__).resolve().parent;source=base/'coverage_companion.py';tests=base/'test_coverage_companion.py';text=source.read_text()
mutations=[
 ('drop_members',"result['members'].append(row)",'pass'),
 ('disable_lean_scan',"for code,line,detail in a.lean_findings(tokens)","for code,line,detail in []"),
 ('disable_budget',"if info.file_size>limit or total>MAX_TOTAL:","if False:"),
 ('ignore_outer_sha',"if row['sha256']!=gap['sha256']:","if False:"),
 ('ignore_numeric_payload_length',"if len(payload)!=count*itemsize:","if False:"),
 ('force_preservation',"'all_preserved':bool(rows) and all(r['preserved'] for r in rows)","'all_preserved':True"),
 ('drop_notebook_outputs',"for output in cell.get('outputs',[]):","for output in []:"),
 ('ignore_control_log_hash',"if sha(log)!=receipt.get('logs',{}).get(name+'.log'):","if False:"),
 ('suppress_archive_residuals',"return result\n\ndef inspect_gzip","result['residuals']=[]\n return result\n\ndef inspect_gzip")]
def run(path,optimized=False):
 env=dict(os.environ,CBZ_COVERAGE_UNDER_TEST=str(path),PYTHONDONTWRITEBYTECODE='1')
 cmd=[sys.executable,'-B']+(['-O'] if optimized else [])+[str(tests)]
 p=subprocess.run(cmd,env=env,capture_output=True,text=True);return {'exit_code':p.returncode,'output':p.stdout+p.stderr}
receipt={'schema_version':1,'tool_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tests_sha256':hashlib.sha256(tests.read_bytes()).hexdigest(),'baseline':run(source),'optimized':run(source,True),'threshold':{'numerator':1,'denominator':1},'mutants':[]}
with tempfile.TemporaryDirectory(prefix='cbz-coverage-mutants-') as folder:
 for name,before,after in mutations:
  if text.count(before)!=1:receipt['mutants'].append({'name':name,'killed':False,'error':'nonunique mutation'});continue
  path=pathlib.Path(folder)/(name+'.py');path.write_text(text.replace(before,after,1));row=run(path);row['name']=name
  row['killed']=row['exit_code']!=0 and bool(re.search(r'FAILED \(failures=[1-9][0-9]*\)',row['output'])) and 'ERROR:' not in row['output'];receipt['mutants'].append(row)
receipt['killed']=sum(m['killed'] for m in receipt['mutants']);receipt['total']=len(mutations)
receipt['result']='PASS' if receipt['baseline']['exit_code']==receipt['optimized']['exit_code']==0 and receipt['total']>0 and receipt['killed']==receipt['total'] else 'FAIL'
print(json.dumps(receipt,indent=2));sys.exit(0 if receipt['result']=='PASS' else 1)
