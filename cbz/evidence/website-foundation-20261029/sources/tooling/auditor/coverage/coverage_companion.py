#!/usr/bin/env python3
"""Bounded nine-gap CBZ companion: inventory bytes, never execute archive code.

python coverage_companion.py --root REPO --g1 G1.json
Tests: CBZ_AUDITOR=REPO/tools/cbz_audit.py python test_coverage_companion.py
Output is a static coverage receipt, not a total Layer A verdict. All Lean hits,
Python floating-point observations, active HTML and semantic unknowns survive.
"""
from __future__ import annotations
import argparse,ast,base64,gzip,hashlib,importlib.util,io,json,os,re,stat,struct,subprocess,sys,zipfile
from html.parser import HTMLParser
from pathlib import Path,PurePosixPath

MAX_BYTES=64*1024*1024
MAX_TOTAL=128*1024*1024
MAX_MEMBERS=4096
AUDITOR_PATH=Path(os.environ.get('CBZ_AUDITOR',Path.cwd()/'tools/cbz_audit.py'))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def strict_json(raw):
 def pairs(rows):
  out={}
  for key,value in rows:
   if key in out:raise ValueError('duplicate JSON key: '+key)
   out[key]=value
  return out
 def invalid(value):raise ValueError('nonfinite JSON: '+value)
 # Decimal tokens are represented verbatim; no binary float arithmetic occurs.
 return json.loads(raw,object_pairs_hook=pairs,parse_float=lambda x:{'$decimal_token':x},parse_constant=invalid)
def auditor():
 spec=importlib.util.spec_from_file_location('cbz_static',AUDITOR_PATH)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class ActiveHTML(HTMLParser):
 def __init__(self):super().__init__();self.hits=[]
 def handle_starttag(self,tag,attrs):
  if tag in {'script','iframe','object','embed'}:self.hits.append({'tag':tag,'line':self.getpos()[0]})
  for name,value in attrs:
   if name.lower().startswith('on') or (value or '').strip().lower().startswith('javascript:'):
    self.hits.append({'attribute':name,'line':self.getpos()[0]})

def inspect_npy(raw):
 if not raw.startswith(b'\x93NUMPY') or raw[6:8] not in {b'\x01\x00',b'\x02\x00',b'\x03\x00'}:raise ValueError('unsupported NPY version')
 width=2 if raw[6]==1 else 4;offset=8+width
 size=int.from_bytes(raw[8:offset],'little')
 if size>65536 or offset+size>len(raw):raise ValueError('invalid NPY header size')
 header=ast.literal_eval(raw[offset:offset+size].decode('utf-8' if raw[6]==3 else 'latin1').strip())
 if type(header) is not dict or set(header)!={'descr','shape','fortran_order'}:raise ValueError('invalid NPY header')
 dtype=header['descr'];shape=header['shape']
 if type(dtype) is not str or not re.fullmatch(r'[<>=|][biufc][124816]+',dtype) or type(shape) is not tuple or any(type(x) is not int or x<0 for x in shape) or type(header['fortran_order']) is not bool:raise ValueError('unsupported NPY object/structured dtype or shape')
 itemsize=int(dtype[2:]);count=1
 if itemsize not in {1,2,4,8,16}:raise ValueError('unsupported NPY item size')
 for x in shape:count*=x
 payload=raw[offset+size:]
 if len(payload)!=count*itemsize:raise ValueError('NPY payload length mismatch')
 return {'dtype':dtype,'shape':list(shape),'fortran_order':header['fortran_order'],'elements':count,'header_sha256':sha(raw[:offset+size]),'payload_bytes':len(payload),'payload_sha256':sha(payload),'floating_storage':dtype[1] in 'fc','values_used_for_certification':False}

def inspect_payload(name,raw):
 suffix=PurePosixPath(name).suffix.lower()
 if suffix=='.npy':return {'format':'npy','numeric_storage':inspect_npy(raw)}
 if suffix=='.ipynb':return inspect_notebook(raw)
 text=raw.decode('utf-8')
 out={'format':suffix or 'text','utf8':True,'lines':len(text.splitlines())}
 if suffix=='.lean':
  a=auditor();tokens,gaps=a.lean_tokens(text)
  out['lean_findings']=[{'code':code,'line':line,'detail':detail} for code,line,detail in a.lean_findings(tokens)]
  out['lexical_gaps']=[{'line':line,'detail':detail} for line,detail in gaps]
 elif suffix=='.py':
  tree=ast.parse(text,filename=name);a=auditor();hits,gaps=a.certification_findings(name,text,[])
  out['float_observations']=[{'code':code,'line':line} for code,line,_ in hits]
  out['calls']=sorted({ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)})
  out['dynamic_operations']=[{'line':n.lineno,'operation':ast.unparse(n.func)} for n in ast.walk(tree) if isinstance(n,ast.Call) and (ast.unparse(n.func).split('.')[-1] in {'eval','exec','compile','__import__','run','Popen','system','load','loads'})]
  out['mathematical_certification_inferred']=False
 elif suffix=='.json':
  data=strict_json(raw);out['top_keys']=sorted(data) if type(data) is dict else None
  out['top_type']=type(data).__name__
  out['top_counts']={k:len(v) for k,v in data.items() if type(v) in {list,dict}} if type(data) is dict else {}
 elif suffix in {'.html','.htm'}:
  parser=ActiveHTML();parser.feed(text);out['active_html']=parser.hits
 return out

def inspect_zip(raw,limit=MAX_BYTES):
 result={'format':'zip','members':[],'residuals':[]}
 try:
  with zipfile.ZipFile(io.BytesIO(raw)) as archive:
   infos=archive.infolist();total=0;seen=set()
   if len(infos)>MAX_MEMBERS:raise ValueError('member count exceeds finite budget')
   for info in infos:
    path=PurePosixPath(info.filename);row={'path':info.filename,'declared_bytes':info.file_size,'compressed_bytes':info.compress_size}
    result['members'].append(row)
    try:
     if path.is_absolute() or '..' in path.parts or '\\' in info.filename or info.filename in seen:raise ValueError('unsafe or duplicate archive member')
     seen.add(info.filename)
     if stat.S_ISLNK(info.external_attr>>16) or info.flag_bits&1:raise ValueError('symlink/encrypted archive member')
     total+=info.file_size
     if info.file_size>limit or total>MAX_TOTAL:raise ValueError('member/total byte budget exceeded')
     with archive.open(info) as member:data=member.read(limit+1)
     if len(data)!=info.file_size or len(data)>limit:raise ValueError('member byte count mismatch/budget')
     row.update(bytes=len(data),sha256=sha(data),read=True)
     if info.is_dir():row['inspection']={'format':'directory'};continue
     if auditor().is_archive(info.filename,data):raise ValueError('nested archive remains explicit residual')
     row['inspection']=inspect_payload(info.filename,data)
    except (OSError,ValueError,SyntaxError,UnicodeError,RuntimeError,zipfile.BadZipFile) as error:
     row['error']=str(error);result['residuals'].append({'path':info.filename,'reason':str(error)})
 except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as error:result['residuals'].append({'path':'<container>','reason':str(error)})
 return result

def inspect_gzip(raw,limit=MAX_BYTES):
 out={'format':'gzip','members':[],'residuals':[]}
 try:
  with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:data=stream.read(limit+1)
  if len(data)>limit:raise ValueError('gzip expanded byte budget exceeded')
  out['members'].append({'path':'<decompressed-json-stream>','sha256':sha(data),'bytes':len(data),'read':True,'inspection':inspect_payload('payload.json',data)})
 except (OSError,EOFError,ValueError,UnicodeError) as error:out['residuals'].append({'path':'<gzip-stream>','reason':str(error)})
 return out

def inspect_notebook(raw):
 data=strict_json(raw)
 if type(data) is not dict or data.get('nbformat')!=4 or type(data.get('cells')) is not list:raise ValueError('unsupported notebook structure')
 rows=[]
 for index,cell in enumerate(data['cells']):
  if type(cell) is not dict or cell.get('cell_type') not in {'code','markdown','raw'}:raise ValueError('invalid cell')
  source=cell.get('source');text=source if type(source) is str else ''.join(source) if type(source) is list and all(type(s) is str for s in source) else None
  if text is None:raise ValueError('invalid cell source')
  raw_source=text.encode();row={'cell':index,'cell_type':cell['cell_type'],'cell_canonical_sha256':sha(canonical(cell)),'source_sha256':sha(raw_source),'source_bytes':len(raw_source),'outputs':[],'attachments':[]}
  row['inspection']=inspect_payload('cell.py',raw_source) if cell['cell_type']=='code' else {'format':cell['cell_type'],'source_text':text}
  def media_entries(media):
   entries=[]
   for mime,value in media.items():
    encoded=value if type(value) is str else ''.join(value) if type(value) is list else json.dumps(value,sort_keys=True)
    m={'mime':mime,'encoded_sha256':sha(encoded.encode())}
    if mime.startswith('image/') and mime!='image/svg+xml':
     decoded=base64.b64decode(encoded,validate=True);m.update(decoded_sha256=sha(decoded),decoded_bytes=len(decoded))
    entries.append(m)
   return entries
  for output in cell.get('outputs',[]):
   row['outputs'].append({'type':output.get('output_type'),'canonical_sha256':sha(canonical(output)),'media':media_entries(output.get('data',{}))})
  for name,attachment in cell.get('attachments',{}).items():row['attachments'].append({'name':name,'media':media_entries(attachment)})
  rows.append(row)
 return {'format':'notebook','cells':rows,'metadata_canonical_sha256':sha(canonical(data.get('metadata',{}))),'residuals':[]}

POLICIES={
 'audits/formal_enforcement/2026-09-27/Main188_landed_formal_36358139870.zip':'historical_main_negative_control_evidence',
 'audits/formal_enforcement/2026-09-27/Math96_landed_downstream_36359933550.zip':'historical_downstream_mutation_evidence',
 'audits/formal_enforcement/2026-09-27/Math96_landed_formal_36359933550.zip':'historical_math_negative_control_evidence',
 'docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb':'explicit_noncertifying_coefficient_viewer',
 'docs/reports/atlas-evidence-accessibility-20261026/qualified-observation-5596.json.gz':'historical_ui_observation',
 'docs/site/evidence/side24-main-3b39/verified-execution.zip':'historical_main_negative_control_evidence',
 'docs/site/snapshots/38ac44c585feb2f9a480d8e54ad1a7fe56ad35aff19e2eab929861ea81974644/archive.zip':'historical_readonly_site_snapshot',
 'docs/site/snapshots/7c583e28878d2bf2c94bf37325820bd5eb1e7f3f557e8a78dd27881a27f6548c/archive.zip':'historical_readonly_site_snapshot',
 'reviews/zeta_fold_slice_c_openai_20261026/fresh/raw/rs_cheb_coeffs.npz':'explicit_noncertifying_numerical_review_data'}

def control_bindings(raw):
 with zipfile.ZipFile(io.BytesIO(raw)) as archive:
  names=archive.namelist();candidates=[p for p in names if PurePosixPath(p).name=='receipt.json']
  if len(candidates)!=1:raise ValueError('historical control receipt not unique')
  path=candidates[0];receipt_raw=archive.read(path);receipt=strict_json(receipt_raw);prefix=str(PurePosixPath(path).parent)
  prefix='' if prefix=='.' else prefix+'/'
  controls=receipt.get('negative_controls')
  if type(controls) is not dict or not controls:raise ValueError('historical control inventory empty')
  rows=[]
  for name,status in controls.items():
   if status not in {'REJECTED_BY_AXIOM_GATE','REJECTED_BY_LEAN'} or '/' in name or '..' in name:raise ValueError('unrecognized historical control status/name')
   source_path=prefix+name+'.lean';log_path=prefix+name+'.log';source=archive.read(source_path);log=archive.read(log_path)
   if sha(log)!=receipt.get('logs',{}).get(name+'.log'):raise ValueError('historical control log SHA mismatch')
   rows.append({'control':name,'historical_status':status,'source_member':source_path,'source_sha256':sha(source),'log_member':log_path,'log_sha256':sha(log),'log_text':log.decode('utf-8'),'scientific_disposition':'QUARANTINED_TEST_INPUT_NOT_CLOSED_THEOREM'})
  return {'receipt_member':path,'receipt_sha256':sha(receipt_raw),'checked_commit':receipt.get('checked_commit'),'alignment_status':receipt.get('alignment_status'),'controls':rows,'fresh_execution_claim':False}

def inventory(root,g1_path):
 gaps=strict_json(g1_path.read_bytes())['coverage_gaps'];rows=[];failures=[]
 actual={g['path'] for g in gaps}
 if actual!=set(POLICIES) or len(gaps)!=len(POLICIES):failures.append('input gap set is not the exact nine scoped containers')
 for gap in gaps:
  path=gap['path'];row={'path':path,'expected_sha256':gap.get('sha256'),'classification':POLICIES.get(path,'UNKNOWN'),'mathematical_status':'NO_PROMOTION'};rows.append(row)
  try:
   if path not in POLICIES:raise ValueError('out-of-scope path remains residual')
   raw=(root/path).read_bytes();row.update(sha256=sha(raw),bytes=len(raw))
   if row['sha256']!=gap['sha256']:raise ValueError('G1 exact byte binding changed')
   inspected=inspect_notebook(raw) if path.endswith('.ipynb') else inspect_gzip(raw) if path.endswith('.gz') else inspect_zip(raw)
   row.update(inspected)
   if row['classification'] in {'historical_main_negative_control_evidence','historical_math_negative_control_evidence'}:
    row['historical_control_bindings']=control_bindings(raw)
   for member in row.get('members',[]):
    member['classification']=row['classification']
    if member.get('inspection',{}).get('format')=='.lean':member['scientific_disposition']='QUARANTINED_HISTORICAL_SOURCE_OR_CONTROL; findings preserved; not a current theorem'
   if row.get('residuals'):failures.append(path+': content residuals')
  except (OSError,ValueError,KeyError,SyntaxError,UnicodeError) as error:row['error']=str(error);failures.append(path+': '+str(error))
 return {'schema_version':1,'tool':'cbz-nine-gap-coverage-1','tool_sha256':sha(Path(__file__).read_bytes()),'scanner_sha256':sha(AUDITOR_PATH.read_bytes()),'g1_receipt_sha256':sha(g1_path.read_bytes()),'containers':rows,'residuals':failures,'coverage_result':'INVENTORIED' if not failures and len(rows)==9 else 'INCOMPLETE','mathematical_soundness_certified':False,'semantic_result':'REQUIRES_EXACT_LIVE_IMPORT_BOUNDARY_AND_STATUS_RECEIPTS','limits':{'member_bytes':MAX_BYTES,'total_archive_bytes':MAX_TOTAL,'members_per_archive':MAX_MEMBERS},'unknowns':['Python dynamic operations are observed, not executed or proven safe.','Historical source/claim alignment is not accepted by archival inventory.','Numeric data and display floats supply no mathematical certification.','Current CLOSED dependency isolation must be independently bound to fresh kernel/source receipts.']}

def preservation(root,base):
 """Exact-byte test/formal/workflow preservation, not semantic equivalence."""
 def git(*args):return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True).stdout
 commit=git('rev-parse',base).decode().strip()
 paths=[os.fsdecode(p) for p in git('ls-tree','-r','--name-only','-z',commit).split(b'\0') if p]
 rows=[]
 for path in paths:
  if not (path.startswith(('formal/','.github/workflows/')) or PurePosixPath(path).name.startswith('test_')):continue
  before=git('show',commit+':'+path)
  try:current=(root/path).read_bytes();actual=sha(current)
  except OSError:actual=None
  rows.append({'path':path,'base_sha256':sha(before),'current_sha256':actual,'preserved':actual==sha(before)})
 return {'base_commit':commit,'rows':rows,'count':len(rows),'all_preserved':bool(rows) and all(r['preserved'] for r in rows)}

def main():
 global AUDITOR_PATH
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--g1',type=Path,required=True);args=p.parse_args();AUDITOR_PATH=args.root/'tools/cbz_audit.py'
 try:r=inventory(args.root,args.g1)
 except (OSError,ValueError,KeyError,TypeError) as error:r={'coverage_result':'INCOMPLETE','error':str(error),'mathematical_soundness_certified':False}
 print(json.dumps(r,indent=2,sort_keys=True,allow_nan=False));return 0 if r['coverage_result']=='INVENTORIED' else 2
if __name__=='__main__':sys.exit(main())
