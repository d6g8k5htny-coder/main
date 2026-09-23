#!/usr/bin/env python3
"""Validate exact export transport, stage the unchanged importer, create only a Git blob."""
from pathlib import Path
import argparse, base64, hashlib, json, os, platform, subprocess, sys, tempfile
import urllib.request, zlib
from rebuild_export import rebuild, TARGET_SHA256, TARGET_BYTES, require

IMPORTER_SHA = 'e263f8600571648b84bf575f380cc94fce3395ff12eeb093b9a493ee1e2555cb'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def snapshot(path):
    return {p.relative_to(path).as_posix(): digest(p.read_bytes()) for p in path.rglob('*') if p.is_file()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--publish', action='store_true')
    a=p.parse_args(); root=Path(__file__).resolve().parent
    a.output.mkdir(parents=True, exist_ok=False)
    manifest=json.loads((root/'payload_manifest.json').read_text())
    chunks=[]
    for index,row in enumerate(manifest['chunks']):
        require(row['index']==index and row['path']==f'payload/{index:02d}.bin','Unexpected chunk path')
        data=(root/row['path']).read_bytes()
        require(len(data)==row['bytes'] and digest(data)==row['sha256'],'Chunk identity mismatch')
        chunks.append(data)
    payload=b''.join(chunks)
    require(len(payload)==manifest['payload_bytes'] and digest(payload)==manifest['payload_sha256'],'Payload mismatch')
    old=a.research/'registers/source/GP-REG-032_v1.2_export_2026-09-18.xlsx'
    importer=a.research/'tools/registers_import.py'
    old_bytes=old.read_bytes(); importer_bytes=importer.read_bytes()
    require(digest(importer_bytes)==IMPORTER_SHA,'Importer identity mismatch')
    result=rebuild(old_bytes,payload)
    source=a.output/'GP-REG-032_v1.2_export_2026-09-23_R1.xlsx'
    source.write_bytes(result)
    refused=[]
    for name,bad_source,bad_payload in [('source_hash',old_bytes+b'X',payload),('payload_hash',old_bytes,payload+b'X')]:
        try:
            rebuild(bad_source,bad_payload)
        except ValueError:
            refused.append(name)
        else:
            raise RuntimeError('Transport negative control did not refuse')
    command=[sys.executable,str(importer),'--source',str(source)]
    executions=[]
    def run(out, check=False, expected=0, diagnostic=None):
        args=command+['--out-json',str(out/'json'),'--out-csv',str(out/'csv')]+(['--check'] if check else [])
        r=subprocess.run(args,capture_output=True,text=True,timeout=180)
        executions.append({'check':check,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
        require(r.returncode==expected,'Importer returned an unexpected code: '+r.stdout+r.stderr)
        if diagnostic:
            require(diagnostic in r.stdout+r.stderr,'Missing expected rejection diagnostic')
    with tempfile.TemporaryDirectory(prefix='register-import-') as work:
        first=Path(work)/'first'; second=Path(work)/'second'
        run(first);run(second)
        fingerprints=snapshot(first)
        require(fingerprints==snapshot(second) and len(fingerprints)==88,'Repeatability/file-count failure')
        run(first,True)
        queue_path=first/'json/review_queue.json'; original_queue=queue_path.read_bytes()
        queue=json.loads(original_queue)
        keys=['RV-H3-SOLVER-20260921-01','RV-WITHDRAWAL-20260921-01','RV-ROLLOUT-R2-20260921-01']
        counts={key:sum(row[0]==key for row in queue['rows']) for key in keys}
        require(len(queue['rows'])==40 and all(v==1 for v in counts.values()),'Required review routes missing or duplicated')
        queue_path.write_bytes(original_queue+b' ');run(first,True,1,'review_queue.json');queue_path.write_bytes(original_queue)
        missing=first/'csv/review_queue.csv'; original_csv=missing.read_bytes();missing.unlink()
        run(first,True,1,'review_queue.csv');missing.write_bytes(original_csv)
        extra=first/'json/stale_transport_control.json';extra.write_text('{}\n')
        run(first,True,1,'stale_transport_control.json');extra.unlink()
        run(first,True)
        require(snapshot(first)==fingerprints,'Restoration failed')
    require(old.read_bytes()==old_bytes and importer.read_bytes()==importer_bytes,'Source/importer changed')
    require(source.read_bytes()==result,'Rebuilt source changed')
    blob_sha=hashlib.sha1(f'blob {len(result)}\0'.encode()+result).hexdigest()
    report={'source_commit':'b02efe21d8d7475ac9b4aa25b3dcc26a5b6f4bf4','runtime':platform.python_version(),'zlib':zlib.ZLIB_RUNTIME_VERSION,'bytes':len(result),'sha256':digest(result),'git_blob_sha':blob_sha,'importer_sha256':IMPORTER_SHA,'tabs':44,'generated_files':88,'repeated_generation_identical':True,'review_queue_rows':40,'review_key_counts':counts,'transport_negative_controls':refused,'import_negative_controls':['altered_json','missing_csv','extra_stale_json'],'source_and_importer_unchanged':True,'canonical_import_completed':False,'scientific_status_changed':False,'blob_created_in_repository':False,'github_run_id':os.environ.get('GITHUB_RUN_ID'),'executions':executions}
    if a.publish:
        require(sys.version_info[:2]==(3,11),'Publishing requires repository Python 3.11')
        require(os.environ.get('GITHUB_REPOSITORY')=='d6g8k5htny-coder/main','Wrong repository')
        require(os.environ.get('GITHUB_REF')=='refs/heads/transport/register-r1-20260923','Wrong transport branch')
        token=os.environ['GH_TOKEN']
        request=urllib.request.Request('https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs',data=json.dumps({'encoding':'base64','content':base64.b64encode(result).decode()}).encode(),headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','Content-Type':'application/json','X-GitHub-Api-Version':'2022-11-28'},method='POST')
        with urllib.request.urlopen(request,timeout=60) as response:
            created=json.load(response)
        require(created['sha']==blob_sha,'Remote Git blob identity mismatch')
        report['blob_created_in_repository']=True
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='executions'},sort_keys=True))

if __name__=='__main__':
    main()
