from pathlib import Path
import hashlib,json,re,zipfile
p=Path(__file__).with_name('main808-navigation.zip');raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
if len(raw)!=72378 or digest!='de278f73d8f25dfe98df32340a1592ba0b65bfc803c3b9cb41fe4a596baaf2eb':raise ValueError('archive identity')
with zipfile.ZipFile(p) as z:
 names=z.namelist()
 if len(names)!=7 or len(set(names))!=7 or z.testzip() is not None:raise ValueError('archive members')
 if z.read('commit.txt').decode().strip()!='02973f3975a1c4cc91b7c51d372f65717c9b3c5c':raise ValueError('commit')
 log=z.read('tests-normal.log').decode();skips=[x for x in log.splitlines() if ' ... skipped ' in x]
 if not re.search(r'Ran 2774 tests in [0-9.]+s\s+OK \(skipped=23\)',log) or len(skips)!=23:raise ValueError('discovery population')
 if json.loads(z.read('navigation.json'))['problems']:raise ValueError('navigation problems')
 for n,count in [('history-reader.log',20),('graph525-optimized.log',1)]:
  text=z.read(n).decode()
  if not re.search(r'Ran '+str(count)+r' tests? in [0-9.]+s\s+OK\s*$',text) or ' ... skipped ' in text:raise ValueError(n)
old=Path('/workspace/scratch/7abea556a9be/cbz_v50_build/private/main811-postmerge-navigation.zip')
with zipfile.ZipFile(old) as z:previous=[x for x in z.read('tests-normal.log').decode().splitlines() if ' ... skipped ' in x]
if set(previous)!=set(skips):raise ValueError('skip populations differ')
out={'run':42781607023,'job':129712723363,'artifact':13746702283,'sha256':digest,'members':7,'crc':'valid','commit':'02973f3975a1c4cc91b7c51d372f65717c9b3c5c','discovery_tests':2774,'skips':23,'same_skip_lines_as_prior_source':True,'history_tests':20,'graph_tests':1,'scope':'Original source-bound navigation run; skips remain; not final G5.'}
Path(__file__).with_name('main808-navigation-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
