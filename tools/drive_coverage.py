#!/usr/bin/env python3
"""Verify scoped Drive coverage and inert packed bytes; no scientific verdict.

This validates an explicitly pinned inventory plus dated additive sources. It
cannot prove discovery completeness, native export fidelity, review approval,
or the mathematical status of copied material. Imported source is never run.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.manifest_integrity_check import local_file, strict_object, reject_constant

DEFAULT='drive/deltas/2026-09-19/DG-MIGRATION-20260919'
DIGEST=re.compile(r'[0-9a-f]{64}\Z')
STORED={'EXACT_SOURCE_BYTES','EXISTING_EXACT_BYTES','NATIVE_EXPORT','RAW_SOURCE_SNAPSHOT'}
UNSTORED={'FOLDER_METADATA','HELD_NO_OPEN','HELD_PERSONAL_SCOPE','HELD_SOURCE_DRIFT','HELD_PUBLICATION_REVIEW','FETCH_FAILED','MISSING'}


def load(path):
    return json.loads(path.read_text(),object_pairs_hook=strict_object,parse_constant=reject_constant)


def json_lines(path):
    return [json.loads(line,object_pairs_hook=strict_object,parse_constant=reject_constant)
            for line in path.read_text().splitlines() if line.strip()]


def require(condition,message):
    if not condition:raise ValueError(message)


def hash_bytes(data):return hashlib.sha256(data).hexdigest()


class Store:
    """Validated, read-only chunk store. Does not extract or execute ZIP members."""
    def __init__(self,root:Path,bundle:str):
        self.root=root
        self.base=local_file(root,bundle+'/objects.json').parent
        self.index=load(self.base/'objects.json')
        require(type(self.index.get('schema_version')) is int and self.index['schema_version']==1,'unsupported object schema')
        require(type(self.index.get('chunk_size')) is int and 0<self.index['chunk_size']<=1048576,'invalid chunk size')
        self.cache={}
        self.packs=set()

    def chunk(self,digest):
        require(isinstance(digest,str) and DIGEST.fullmatch(digest),'invalid chunk identity')
        if digest in self.cache:return self.cache[digest]
        record=self.index['chunks'][digest]
        require(type(record['bytes']) is int and 0<record['bytes']<=self.index['chunk_size'],'invalid chunk size')
        pack=local_file(self.base,record['pack'])
        require(pack.parent==self.base/'packs' and DIGEST.fullmatch(pack.stem) and pack.suffix=='.zip','invalid pack path')
        if pack not in self.packs:
            require(hash_bytes(pack.read_bytes())==pack.stem,'pack digest mismatch')
            with zipfile.ZipFile(pack) as archive:
                names=archive.namelist()
                expected={key for key,value in self.index['chunks'].items() if value['pack']==record['pack']}
                require(len(names)==len(set(names)) and set(names)==expected,'unindexed or duplicate pack member')
            self.packs.add(pack)
        with zipfile.ZipFile(pack) as archive:
            matches=[x for x in archive.infolist() if x.filename==digest]
            require(len(matches)==1,'missing or duplicate packed chunk')
            info=matches[0]
            require(info.file_size==record['bytes'] and not info.is_dir(),'chunk size mismatch')
            require((info.external_attr>>16)&0o170000 != 0o120000,'symlink chunk')
            data=archive.read(info)
        require(len(data)==record['bytes'] and hash_bytes(data)==digest,'chunk digest mismatch')
        # At most one chunk retained: avoid a whole-corpus memory cache.
        self.cache={digest:data}
        return data

    def object_chunks(self,digest):
        require(isinstance(digest,str) and DIGEST.fullmatch(digest),'invalid object identity')
        record=self.index['objects'][digest]
        require(type(record['bytes']) is int and record['bytes']>=0,'invalid object bytes')
        require(isinstance(record['chunks'],list),'invalid chunk order')
        total=0;hasher=hashlib.sha256()
        for i,key in enumerate(record['chunks']):
            data=self.chunk(key)
            if i<len(record['chunks'])-1:require(len(data)==self.index['chunk_size'],'short interior chunk')
            total+=len(data);hasher.update(data)
            yield data
        require(total==record['bytes'] and hasher.hexdigest()==digest,'object digest/size mismatch')

    def verify(self):
        for digest in self.index['objects']:
            for _ in self.object_chunks(digest):pass
        for digest in self.index['chunks']:self.chunk(digest)


def check(root:Path=ROOT,bundle:str=DEFAULT):
    scope=load(local_file(root,bundle+'/scope.json'))
    require(type(scope.get('schema_version')) is int and scope['schema_version']==1,'unsupported scope schema')
    inventory=local_file(root,scope['inventory']['path'])
    require(hash_bytes(inventory.read_bytes())==scope['inventory']['sha256'],'inventory identity drift')
    sources=json_lines(inventory)+scope['additional_sources']
    require(all(isinstance(s,dict) and isinstance(s.get('id'),str) and s['id'] for s in sources),'invalid source identity')
    by_id={s['id']:s for s in sources}
    require(len(by_id)==len(sources),'duplicate source identity')
    ledger=json_lines(local_file(root,bundle+'/coverage.jsonl'))
    rows={r['id']:r for r in ledger}
    require(len(rows)==len(ledger),'duplicate coverage identity')
    require(set(rows)==set(by_id),'coverage does not account for exactly the scoped identities')
    store=Store(root,bundle);store.verify()
    file_cache={}
    for row in ledger:
        source=by_id[row['id']];status=row['status']
        require(status in STORED|UNSTORED,'unknown coverage status')
        require(row.get('context')==source['context'],'source context changed')
        folder=source['mimeType']=='application/vnd.google-apps.folder'
        require((status=='FOLDER_METADATA')==folder,'folder counted as payload')
        if not folder:
            if 'DO_NOT_OPEN' in source['path']:require(status=='HELD_NO_OPEN','protected source consumed')
            elif source['context']=='PERSONAL_EARLIER' or 'PERSONAL_EARLIER' in source['title']:
                require(status=='HELD_PERSONAL_SCOPE','personal source consumed')
        if status not in STORED:
            require('storage' not in row,'non-stored row has a payload')
            require(isinstance(row.get('reason'),str) and row['reason'].strip(),'missing disposition reason')
            continue
        require(type(row.get('bytes')) is int and row['bytes']>=0,'invalid stored byte count')
        storage=row['storage'];require(set(storage) in ({'object'},{'file'}),'ambiguous storage reference')
        if 'object' in storage:
            digest=storage['object'];record=store.index['objects'][digest];size=record['bytes']
        else:
            path=local_file(root,storage['file'])
            if path not in file_cache:
                data=path.read_bytes();file_cache[path]=(hash_bytes(data),len(data))
            digest,size=file_cache[path]
        require(row.get('sha256')==digest and row['bytes']==size,'stored identity mismatch')
        native=source['mimeType'].startswith('application/vnd.google-apps.')
        require((status=='NATIVE_EXPORT')==native,'native export misrepresented as raw exact bytes')
        if status in {'EXACT_SOURCE_BYTES','EXISTING_EXACT_BYTES'}:
            require(source.get('sha256')==digest and source.get('bytes')==size,'inventory source identity mismatch')
        elif status=='NATIVE_EXPORT':
            require(isinstance(row.get('export_mime_type'),str) and row['export_mime_type'],'missing export format')
            require(isinstance(row.get('drive_modified_time'),str) and row['drive_modified_time'],'missing native observation')
            revision=row.get('native_revision_observation')
            require(isinstance(revision,dict) and revision.get('id')==row['id'] and isinstance(revision.get('current_revision_id'),str) and revision['current_revision_id'] and revision.get('revision_pages_complete') is True,'missing complete native revision metadata')
        elif status=='RAW_SOURCE_SNAPSHOT':
            require(source.get('sha256') is None,'snapshot used to conceal an inventory mismatch')
    counts=dict(sorted(Counter(r['status'] for r in ledger).items()))
    unresolved=sum(counts.get(k,0) for k in ('MISSING','FETCH_FAILED','HELD_SOURCE_DRIFT','HELD_PUBLICATION_REVIEW'))
    return {'scoped_objects':len(sources),'counts':counts,'packed_objects':len(store.index['objects']),
            'packs':len(store.packs),'unresolved':unresolved,
            'eligible_sources_accounted_for':unresolved==0,
            'scope':'Declared source identities and bytes only; no discovery, native fidelity or scientific verdict.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--bundle',default=DEFAULT)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--search')
    parser.add_argument('--extract-id')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        report=check(args.root,args.bundle)
        if args.search:
            scope=load(local_file(args.root,args.bundle+'/scope.json'))
            sources=json_lines(local_file(args.root,scope['inventory']['path']))+scope['additional_sources']
            rows={r['id']:r for r in json_lines(local_file(args.root,args.bundle+'/coverage.jsonl'))}
            report['matches']=[{'id':s['id'],'title':s['title'],'path':s['path'],'status':rows[s['id']]['status']} for s in sources if args.search.casefold() in (s['title']+' '+s['path']).casefold()]
        if args.extract_id:
            require(args.output is not None,'extraction needs --output')
            require(not args.output.exists(),'refusing to replace output')
            row=next(r for r in json_lines(local_file(args.root,args.bundle+'/coverage.jsonl')) if r['id']==args.extract_id)
            require(row['status'] in STORED,'source is not stored')
            storage=row['storage']
            if 'object' in storage:
                data=b''.join(Store(args.root,args.bundle).object_chunks(storage['object']))
            else:data=local_file(args.root,storage['file']).read_bytes()
            # Exclusive creation after full digest verification; never execute.
            with args.output.open('xb') as stream:stream.write(data)
        print(json.dumps(report,sort_keys=True) if args.json or args.search else 'ok: Drive coverage '+json.dumps(report['counts'],sort_keys=True))
        return 0
    except (OSError,ValueError,KeyError,TypeError,StopIteration,zipfile.BadZipFile) as error:
        print(f'FAIL: {error}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
