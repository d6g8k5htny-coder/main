#!/usr/bin/env python3
"""Read-only evidence assembly; conclusions are scoped in COVERAGE_REPORT.md."""
import ast,datetime,hashlib,importlib.util,json,pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1]).resolve();base='f20cc54d4d136e854608853372249882eb4c949f'
parent=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('coverage',parent/'coverage_companion.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.AUDITOR_PATH=root/'tools/cbz_audit.py'
def git(*args):return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True).stdout
def binding(path):
 raw=(root/path).read_bytes();return {'path':path,'sha256':c.sha(raw),'bytes':len(raw)}
manifest=json.loads((root/'formal/manifest.json').read_text());kernel_path='formal/.lake/cbz-evidence/receipt.json';kernel=json.loads((root/kernel_path).read_text())
source_bindings=[]
for path,expected in manifest['files'].items():
 row=binding(path);row['manifest_sha256']=expected;row['base_sha256']=c.sha(git('show',base+':'+path));row['parent_sha256']=c.sha(git('show',base+'^:'+path));row['all_equal']=row['sha256']==expected==row['base_sha256']==row['parent_sha256'];source_bindings.append(row)
imports={};dynamic=[];lean_files=[]
for path in sorted(p for p in manifest['files'] if p.endswith('.lean')):
 source=(root/path).read_text();tokens,gaps=c.auditor().lean_tokens(source);lean_files.append(binding(path))
 imports[path]=re.findall(r'^import ([A-Za-z0-9_.]+)$',source,re.M)
 for kind,value,line in tokens:
  if value in {'elab','macro','syntax','initialize','builtin_initialize','run_tac','unsafe','extern','implemented_by','include_str','include_bytes','IO','native_decide','axiom','sorry','admit'}:dynamic.append({'path':path,'line':line,'token':value})
 for line,detail in gaps:dynamic.append({'path':path,'line':line,'lexical_gap':detail})
logs=[]
for name,expected in kernel['logs'].items():
 row=binding('formal/.lake/cbz-evidence/'+name);row['receipt_sha256']=expected;row['matched']=row['sha256']==expected;logs.append(row)
audit_inputs=[]
for path,expected in kernel['audit_input_sha256'].items():
 row=binding(path);row['receipt_sha256']=expected;row['matched']=row['sha256']==expected;audit_inputs.append(row)
def tests(raw):
 tree=ast.parse(raw);result={}
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef):
   for f in cls.body:
    if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)) and f.name.startswith('test_'):result[cls.name+'.'+f.name]=c.sha(ast.dump(f,include_attributes=False).encode())
 return result
path='tests/test_node_status_refresh.py';before=tests(git('show',base+'^:'+path));after=tests(git('show',base+':'+path))
source_manifest='cbz/evidence/website-foundation-20261029/source-manifest.json'
out={
 'schema_version':1,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Main31 literal theorem/source and archival isolation boundary; no source-parent statement alignment acceptance',
 'assembler_sha256':c.sha(pathlib.Path(__file__).read_bytes()),'companion_sha256':c.sha((parent/'coverage_companion.py').read_bytes()),
 'base_commit':base,'observed_head':git('rev-parse','HEAD').decode().strip(),
 'kernel_receipt':binding(kernel_path),'kernel_baseline_receipt_expected_sha256':kernel['baseline_receipt_sha256'],'kernel_baseline_receipt_current':binding('formal/.lake/formal-evidence/receipt.json'),'kernel_baseline_receipt_matches_current':kernel['baseline_receipt_sha256']==c.sha((root/'formal/.lake/formal-evidence/receipt.json').read_bytes()),'receipt_interpretation':'Point-in-time source/binding observations only; this assembler does not issue a fresh kernel execution or complete current receipt verdict','kernel_fresh_status':kernel['fresh_kernel'],'kernel_checked_commit':kernel['checked_commit'],'kernel_manifest_bound':kernel['manifest_sha256']==c.sha((root/'formal/manifest.json').read_bytes()),'kernel_audit_inputs':audit_inputs,'kernel_logs':logs,
 'kernel_proof_count':len(kernel['proofs']),'kernel_constant_count':len(kernel['all_project_constants']),
 'kernel_axiom_sets':{p['name']:p['axioms'] for p in kernel['all_project_constants']},
 'kernel_source_alignment':kernel['source_alignment'],'kernel_scope':kernel['scope'],
 'manifest_alignment_status':manifest['alignment_status'],
 'source_preservation':source_bindings,'all_manifest_sources_equal_parent_f20_and_current':all(r['all_equal'] for r in source_bindings),
 'project_import_graph':imports,'implicit_core_dependency':'Lean Init, supplied by exact kernel toolchain; no project Math dependency',
 'lake_manifest':binding('formal/lake-manifest.json'),'lake_packages':json.loads((root/'formal/lake-manifest.json').read_text())['packages'],
 'lean_source_files':lean_files,'dynamic_source_observations':dynamic,
 'preservation_from_f20':c.preservation(root,base),
 'f20_parent_to_f20_test_change':{'path':path,'old_count':len(before),'new_count':len(after),'removed_names':sorted(set(before)-set(after)),'added_names':sorted(set(after)-set(before)),'changed_retained_names':[n for n in sorted(set(before)&set(after)) if before[n]!=after[n]],'old_ast_hashes':before,'new_ast_hashes':after,'interpretation':'One old moving-ref behavior test was explicitly replaced with discussion-timestamp coverage and exact Git-object freshness regression tests; no claim of zero removed names. Existing formal tests are byte-preserved.'},
 'statement_alignment_disposition':'QUARANTINED: analytic/source parent claims and historic scientific labels are not established by Main31 integer arithmetic',
 'archival_disposition':'Historical negative-control inputs and proposed Math/legacy source copies are not imported into any of the three live project modules; archived token hits remain visible.',
 'remaining_unknowns':['Transitive source-parent analytic alignment is not established.','Historical/dynamic Python programs are not executed or proved safe by member inventory.','Fresh final runner must rebind receipts after any source change.'],
 'mathematical_status_promotion':False}
if (root/source_manifest).exists():out['namespaced_quarantine_source_manifest']=binding(source_manifest)
print(json.dumps(out,indent=2,sort_keys=True))
