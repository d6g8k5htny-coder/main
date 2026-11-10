import hashlib,json,zipfile
from pathlib import Path
p=Path(__file__).with_name('main811-architecture-attempt3.zip'); raw=p.read_bytes()
def sha(b): return hashlib.sha256(b).hexdigest()
if len(raw)!=49489 or sha(raw)!='438fc8b6bae87e0fff8257e2ba79621ebd15ed369ee08b1db5d8b87aa033d164': raise ValueError('archive identity')
sandbox=('test_kernel_enforces_cpu_memory_and_process_limits','test_network_namespace_has_no_external_interface_or_route','test_nonroot_and_no_new_privileges','test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space','test_source_mount_is_read_only_and_environment_has_no_publishing_secret')
commit='b285dd7a31ad7b297abc9f2aa93267300111a4e9'
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=13 or len(set(names))!=13 or z.testzip() is not None: raise ValueError('archive membership')
 if (z.read('checked-commit.txt').decode().strip()!=commit or z.read('run-id.txt').decode().strip()!='42778804824' or z.read('run-attempt.txt').decode().strip()!='3' or z.read('container-exit-status.txt').decode().strip()!='0'):raise ValueError('identity')
 receipt=json.loads(z.read('architecture-run-receipt.json'))
 if receipt['checked_commit']!=commit or receipt['run_id']!='42778804824':raise ValueError('receipt identity')
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
out={'run':42778804824,'attempt':3,'job':129711248328,'artifact':13747405278,'sha256':sha(raw),'bytes':len(raw),'members':13,'artifact_hashes':len(listed),'receipt_hashes':len(receipt['evidence_file_sha256']),'commit':commit,'tests_per_mode':363,'skips':0,'selected_sandbox_cases_per_mode':5,'scope':'Exact bounded component; does not erase discovery skips or establish final G5.'}
Path(__file__).with_name('main811-architecture-attempt3-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
