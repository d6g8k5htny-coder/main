from pathlib import Path
import zipfile,hashlib,json
p=Path(__file__).with_name('main808-preparation.zip')
raw=p.read_bytes(); expected='dc9956304b8d0f0dc5bed6a1966f8eeef67ac4a904c1ed4cc1006613f4b38b48'
if len(raw)!=231420 or hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('archive identity')
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=157 or len(set(names))!=len(names) or z.testzip() is not None:raise ValueError('archive members')
 listed={}
 for line in z.read('ARTIFACT_SHA256SUMS').decode().splitlines():
  h,n=line.split('  ',1)
  if n in listed or hashlib.sha256(z.read(n)).hexdigest()!=h:raise ValueError('artifact member '+n)
  listed[n]=h
 if set(listed)!=set(names)-{'ARTIFACT_SHA256SUMS'}:raise ValueError('manifest coverage')
 receipt=json.loads(z.read('receipt.json'));prep=json.loads(z.read('preparation-result.json'));context=json.loads(z.read('context.json'))
 if context!={'checked_commit':'02973f3975a1c4cc91b7c51d372f65717c9b3c5c','repository':'d6g8k5htny-coder/main','run_attempt':'1','run_id':'42781606883'}:raise ValueError('native context')
 if prep['status']!='ACCEPTED_PREPARATION' or prep['preparation_error'] is not None or prep['recovery_error'] is not None:raise ValueError('preparation outcome')
 for row in receipt['originals']:
  b=z.read(row['path'])
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('receipt original '+row['path'])
 for name,row in prep['available_originals'].items():
  b=z.read(name)
  if len(b)!=row['bytes'] or hashlib.sha256(b).hexdigest()!=row['sha256']:raise ValueError('preparation original '+name)
 controls=prep['controls']; counts={s:sum(x['status']==s for x in controls) for s in set(x['status'] for x in controls)}
 out={'artifact':13746905242,'sha256':expected,'bytes':len(raw),'members':len(names),'artifact_hashes':len(listed),'receipt_originals':len(receipt['originals']),'preparation_originals':len(prep['available_originals']),'context':context,'status':prep['status'],'controls':counts,'acquisition_origin':prep['acquisition_origin'],'custody_authentication':prep['custody_authentication'],'docker_hub_fallback':prep['docker_hub_fallback'],'organizational_independence':prep['organizational_independence'],'scope':'original artifact byte reconciliation; no scientific acceptance or final G5'}
 Path(__file__).with_name('main808-preparation-verification.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out))
