import hashlib,json,zipfile
from pathlib import Path
p=Path(__file__).with_name('main808-architecture.zip'); raw=p.read_bytes()
def sha(b): return hashlib.sha256(b).hexdigest()
if len(raw)!=49489 or sha(raw)!='2f20158405e6529755c98c0955c32934446f54781df5fc61a7cecdf768eac2aa': raise ValueError('archive identity')
sandbox=('test_kernel_enforces_cpu_memory_and_process_limits','test_network_namespace_has_no_external_interface_or_route','test_nonroot_and_no_new_privileges','test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space','test_source_mount_is_read_only_and_environment_has_no_publishing_secret')
commit='02973f3975a1c4cc91b7c51d372f65717c9b3c5c'
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=13 or len(set(names))!=13 or z.testzip() is not None: raise ValueError('archive membership')
 if (z.read('checked-commit.txt').decode().strip()!=commit or z.read('run-id.txt').decode().strip()!='42781606883' or z.read('run-attempt.txt').decode().strip()!='1' or z.read('container-exit-status.txt').decode().strip()!='0'):raise ValueError('identity')
 receipt=json.loads(z.read('architecture-run-receipt.json'))
 if receipt['checked_commit']!=commit or receipt['run_id']!='42781606883':raise ValueError('receipt identity')
 for name,digest in receipt['evidence_file_sha256'].items():
  if sha(z.read(name))!=digest:raise ValueError('receipt hash '+name)
 listed={}
 for line in z.read('ARTIFACT_SHA256SUMS').decode().splitlines():
  digest,name=line.split('  ',1)
  if name in listed or sha(z.read(name))!=digest:raise ValueError('artifact hash '+name)
  listed[name]=digest
 if set(listed)!=set(names)-{'ARTIFACT_SHA256SUMS'}:raise ValueError('manifest coverage')
 for mode in ('normal','optimized'):
  text=z.read('tests-'+mode+'.log').decode()
  if 'Ran 363 tests in ' not in text or not text.rstrip().endswith('OK') or ' ... skipped ' in text or 'skipped=' in text:raise ValueError('test population')
  for name in sandbox:
   if not any(line.startswith(name+' (') and line.endswith(' ... ok') for line in text.splitlines()):raise ValueError('missing sandbox case '+name)
out={'run':42781606883,'attempt':1,'job':129712720640,'artifact':13747021504,'sha256':sha(raw),'bytes':len(raw),'members':13,'artifact_hashes':len(listed),'receipt_hashes':len(receipt['evidence_file_sha256']),'commit':commit,'tests_per_mode':363,'skips':0,'selected_sandbox_cases_per_mode':5,'scope':'Exact bounded component; does not erase discovery skips or establish final G5.'}
Path(__file__).with_name('main808-architecture-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
