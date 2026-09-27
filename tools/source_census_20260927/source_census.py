#!/usr/bin/env python3
"""Offline, content-addressed source census. Never execute source files or macros.

Acquisition/authentication remains with the caller. A missing match in a supplied
baseline is NOT proof of absence from a remote service. Optional PyMuPDF and
Pillow expand metadata/text coverage; unsupported types stay explicit.
"""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import os
import plistlib
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import xml.etree.ElementTree as ET
import zipfile

VERSION = '1.2'


def safe_member(name):
    """Accept relative archive names only; never normalize a traversal into safety."""
    p = PurePosixPath(name)
    return bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name and not re.match(r'^[A-Za-z]:', name) and '\x00' not in name


def xml_text(data):
    declaration_view = data.replace(b'\x00', b'').upper()
    if b'<!DOCTYPE' in declaration_view or b'<!ENTITY' in declaration_view:
        raise ValueError('DTD/entity declarations are not accepted')
    root = ET.fromstring(data)
    return '\n'.join(s for s in root.itertext() if s.strip())


class Scanner:
    def __init__(self, cache_dir=None, max_member_bytes=64*1024*1024,
                 max_container_bytes=256*1024*1024, max_file_bytes=128*1024*1024,
                 max_depth=8, max_members=20000, max_text_chars=32*1024*1024):
        self.cache_dir=Path(cache_dir) if cache_dir else None
        self.max_member_bytes=max_member_bytes; self.max_container_bytes=max_container_bytes
        self.max_file_bytes=max_file_bytes; self.max_depth=max_depth;self.max_members=max_members
        self.max_text_chars=max_text_chars
        self.records=[]; self.issues=[];self.texts={};self.cache={};self.parse_count=0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
        try:
            import fitz
            self.pdf=fitz
        except ImportError:
            self.pdf=None
        try:
            from PIL import Image
            self.image=Image
        except ImportError:
            self.image=None
        self.profile=f'{VERSION}:{max_text_chars}:{bool(self.pdf)}:{bool(self.image)}'

    def issue(self,path,reason,detail=''):
        self.issues.append({'path':path,'reason':reason,'detail':str(detail)[:500]})

    def scan_root(self,label,root):
        root=Path(root)
        if root.is_symlink():
            self.issue(label,'symlink');return
        if root.is_file():
            self._file(label,root);return
        if not root.is_dir():
            self.issue(label,'missing_root');return
        for base,dirs,files in os.walk(root,followlinks=False):
            for d in list(dirs):
                p=Path(base)/d
                if p.is_symlink():
                    self.issue(label+'/'+p.relative_to(root).as_posix(),'symlink');dirs.remove(d)
            dirs.sort()
            for n in sorted(files):
                p=Path(base)/n;self._file(label+'/'+p.relative_to(root).as_posix(),p)

    def _file(self,label,p):
        try:
            if p.is_symlink():self.issue(label,'symlink');return
            if not p.is_file():self.issue(label,'nonregular_file');return
            size=p.stat().st_size
            if size>self.max_file_bytes:
                h=hashlib.sha256();g=hashlib.sha1(f'blob {size}\0'.encode())
                with p.open('rb') as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk);g.update(chunk)
                self.records.append({'path':label,'bytes':size,'sha256':h.hexdigest(),'git_blob':g.hexdigest(),
                                     'kind':'uninspected_large_file','status':'size_limit','depth':0,'cache_hit':False})
                self.issue(label,'file_size_limit',size);return
            self.inspect_bytes(label,p.read_bytes())
        except (OSError,ValueError) as e:self.issue(label,'file_read_error',e)

    def inspect_bytes(self,path,data,depth=0):
        digest=hashlib.sha256(data).hexdigest()
        cached=self.cache.get(digest)
        cachefile=self.cache_dir/(digest+'.json') if self.cache_dir else None
        if cached is None and cachefile and cachefile.is_file():
            try:
                candidate=json.loads(cachefile.read_text(encoding='utf-8'))
                if candidate.get('profile')==self.profile:cached=candidate
            except (OSError,ValueError):pass
        hit=cached is not None
        if cached is None:
            self.parse_count+=1
            try:kind,status,text,metadata=self._parse(data)
            except Exception as e:kind,status,text,metadata='unknown','parse_error','',{'error':str(e)[:300]}
            cached={'profile':self.profile,'kind':kind,'status':status,'text':text,'metadata':metadata}
            if cachefile:
                temp=cachefile.with_suffix('.tmp');temp.write_text(json.dumps(cached,ensure_ascii=False),encoding='utf-8');temp.replace(cachefile)
        self.cache[digest]=cached; self.texts[digest]=cached['text']
        r={'path':path,'bytes':len(data),'sha256':digest,
           'git_blob':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),
           'kind':cached['kind'],'status':cached['status'],'depth':depth,'cache_hit':hit,
           'text_sha256':hashlib.sha256(cached['text'].encode()).hexdigest() if cached['text'] else '',
           'text_chars':len(cached['text']), 'metadata':cached['metadata']}
        self.records.append(r)
        kind=cached['kind']
        if kind not in ('zip','tar','gzip','ooxml'):return
        if depth>=self.max_depth:self.issue(path,'depth_limit');return
        try:
            if kind in ('zip','ooxml'):self._zip(path,data,depth,embedded_only=(kind=='ooxml'))
            elif kind=='tar':self._tar(path,data,depth)
            else:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as f:out=f.read(self.max_member_bytes+1)
                if len(out)>self.max_member_bytes:self.issue(path,'member_size_limit');return
                self.inspect_bytes(path+'!/<decompressed>',out,depth+1)
        except Exception as e:self.issue(path,'container_error',e)

    def _parse(self,data):
        metadata={}; text=''
        if data[:4] in (b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08'):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names=z.namelist()
                office='[Content_Types].xml' in names and any(n.startswith(('word/','xl/','ppt/')) for n in names)
                if not office:return 'zip','container_inspected','',{'members':len(names)}
                # Read inert XML only. Never open relationships, execute macros, or calculate formulas.
                parts=[i for i in z.infolist() if i.filename.endswith('.xml') and i.filename.startswith(('word/','xl/','ppt/'))]
                if len(parts)>self.max_members or sum(i.file_size for i in parts)>self.max_container_bytes:
                    return 'ooxml','content_limit','',{'members':len(names)}
                chunks=[];blocked=[];counts=collections.Counter(names)
                for i in parts:
                    if i.file_size>self.max_member_bytes or not safe_member(i.filename) or i.flag_bits&1 or counts[i.filename]>1:
                        blocked.append(i.filename);continue
                    try:chunks.append('['+i.filename+']\n'+xml_text(z.read(i)))
                    except Exception:blocked.append(i.filename)
                text='\n'.join(chunks);metadata={'members':len(names),'blocked_parts':blocked,
                    'visual_parts_unreviewed':sum('/media/' in n for n in names),
                    'macros_present':any('vbaproject' in n.lower() for n in names),
                    'equivalence':'XML text is a reading extract, not source-byte equivalence'}
                status='ooxml_text_extracted' if not blocked else 'ooxml_partial'
                return self._bounded('ooxml',status,text,metadata)
        if data.startswith(b'bplist00'):
            value=plistlib.loads(data)
            text=json.dumps(value,ensure_ascii=False,default=repr)
            return self._bounded('plist','plist_data_extracted',text,{'execution':'none'})
        if data.startswith(b'%PDF-'):
            if self.pdf is None:return 'pdf','parser_unavailable','',{}
            doc=self.pdf.open(stream=data,filetype='pdf')
            try:
                if doc.needs_pass:return 'pdf','encrypted_pdf','',{}
                chunks=[];images=0
                for page in doc:
                    chunks.append(f'[Page {page.number+1}]\n'+page.get_text())
                    images+=len(page.get_images())
                text='\n'.join(chunks);metadata={'pages':len(doc),'image_occurrences':images,'visual_review':'not_performed_by_scanner'}
                status='pdf_text_visual_review_pending' if text.strip() else 'pdf_no_text_visual_review_pending'
                return self._bounded('pdf',status,text,metadata)
            finally:doc.close()
        if data.startswith(b'\x1f\x8b'):return 'gzip','container_inspected','',{}
        if len(data)>262 and data[257:262]==b'ustar':return 'tar','container_inspected','',{}
        if data.startswith((b'\x89PNG\r\n\x1a\n',b'\xff\xd8\xff',b'GIF87a',b'GIF89a')) or (data[:4]==b'RIFF' and data[8:12]==b'WEBP'):
            if self.image:
                with self.image.open(io.BytesIO(data)) as im:metadata={'format':im.format,'size':list(im.size),'frames':getattr(im,'n_frames',1)}
            return 'image','image_metadata_only','',metadata
        if data.startswith((b'7z\xbc\xaf\x27\x1c',b'Rar!',b'BZh',b'\xfd7zXZ',b'\xd0\xcf\x11\xe0')):
            return 'binary_container','unsupported_binary','',{}
        try:
            encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig'
            text=data.decode(encoding)
            bad=sum(ord(c)<32 and c not in '\n\r\t' for c in text)
            if bad>max(0,len(text)//100):raise ValueError('binary control characters')
            metadata={'encoding':encoding}
            # Notebook/JSON remain inert text: no deserialization of executable objects.
            return self._bounded('text','text_extracted',text,metadata)
        except (UnicodeDecodeError,ValueError):return 'binary','unsupported_binary','',{}

    def _bounded(self,kind,status,text,metadata):
        if len(text)>self.max_text_chars:
            metadata['untruncated_text_chars']=len(text);text=text[:self.max_text_chars];status='text_truncated'
        return kind,status,text,metadata

    def _zip(self,path,data,depth,embedded_only=False):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            members=z.infolist(); counts=collections.Counter(i.filename for i in members)
            if len(members)>self.max_members:self.issue(path,'member_count_limit',len(members));return
            total=0
            for i in members:
                p=path+'!/'+i.filename
                if embedded_only and not any(x in i.filename for x in ('/embeddings/','/media/')):continue
                if not safe_member(i.filename):self.issue(p,'unsafe_path');continue
                if counts[i.filename]>1:self.issue(p,'duplicate_member_name');continue
                if i.is_dir():continue
                mode=(i.external_attr>>16)&0xFFFF
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                    self.issue(p,'nonregular_member');continue
                if i.flag_bits&1:self.issue(p,'encrypted_member');continue
                if i.file_size>self.max_member_bytes:self.issue(p,'member_size_limit',i.file_size);continue
                total+=i.file_size
                if total>self.max_container_bytes:self.issue(p,'container_size_limit');break
                self.inspect_bytes(p,z.read(i),depth+1) # CRC is checked by zipfile.

    def _tar(self,path,data,depth):
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as t:
            members=t.getmembers();counts=collections.Counter(i.name for i in members)
            if len(members)>self.max_members:self.issue(path,'member_count_limit',len(members));return
            total=0
            for i in members:
                p=path+'!/'+i.name
                if not safe_member(i.name):self.issue(p,'unsafe_path');continue
                if counts[i.name]>1:self.issue(p,'duplicate_member_name');continue
                if i.isdir():continue
                if not i.isfile():self.issue(p,'nonregular_member');continue
                if i.size>self.max_member_bytes:self.issue(p,'member_size_limit',i.size);continue
                total+=i.size
                if total>self.max_container_bytes:self.issue(p,'container_size_limit');break
                f=t.extractfile(i)
                with f:self.inspect_bytes(p,f.read(),depth+1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',action='append',required=True,help='LABEL=local file or directory; repeatable')
    parser.add_argument('--out',required=True);parser.add_argument('--cache')
    parser.add_argument('--term',action='append',default=[])
    args=parser.parse_args();out=Path(args.out)
    roots=[]
    for value in args.root:
        label,sep,p=value.partition('=')
        if not sep:parser.error('--root requires LABEL=PATH')
        root=Path(p).resolve()
        if root.is_dir() and (out.resolve()==root or root in out.resolve().parents):parser.error('output must be outside scanned roots')
        if args.cache and root.is_dir():
            cache=Path(args.cache).resolve()
            if cache==root or root in cache.parents:parser.error('cache must be outside scanned roots')
        roots.append((label,root))
    out.mkdir(parents=True,exist_ok=True)
    s=Scanner(cache_dir=args.cache)
    for label,root in roots:s.scan_root(label,root)
    with (out/'inventory.jsonl').open('w',encoding='utf-8') as f:
        for row in s.records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'issues.json').write_text(json.dumps(s.issues,indent=2,ensure_ascii=False),encoding='utf-8')
    hits=[]
    for row in s.records:
        text=s.texts.get(row['sha256'],'')
        for term in args.term:
            if term.casefold() in row['path'].casefold() or term.casefold() in text.casefold():
                hits.append({'path':row['path'],'sha256':row['sha256'],'term':term,'filename_match':term.casefold() in row['path'].casefold()})
    (out/'term_hits.json').write_text(json.dumps(hits,indent=2,ensure_ascii=False),encoding='utf-8')
    summary={'version':VERSION,'file_occurrences':len(s.records),'unique_hashes':len({r['sha256'] for r in s.records}),
             'unique_payloads_parsed':s.parse_count,'cache_hits':sum(r['cache_hit'] for r in s.records),
             'kind_counts':dict(collections.Counter(r['kind'] for r in s.records)),
             'status_counts':dict(collections.Counter(r['status'] for r in s.records)),
             'issue_count':len(s.issues),'scope':'supplied local roots only; no remote absence or semantic-review claims',
             'optional_parsers':{'pymupdf':bool(s.pdf),'pillow':bool(s.image)}}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
    return 0 if not s.issues else 2

if __name__=='__main__':
    raise SystemExit(main())
