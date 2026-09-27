"""Declared navigation pages/anchors and optional bounded public-source readback.
No repository writes or retrieved-code execution. Not a whole-GitHub crawler.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
PUBLIC={'main','Math-','meta-framework','query-','google-drive','trial','governance-'}
LINK=re.compile(r'\[[^\]\n]*\]\(([^\s)]+)\)')


def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def lines_outside_fences(text):
    fence=None
    for line in text.splitlines():
        match=re.match(r'\s*(`{3,}|~{3,})',line)
        if match:
            token=match.group(1)
            if fence is None: fence=token
            elif token[0]==fence[0] and len(token)>=len(fence): fence=None
            continue
        if fence is None: yield line


def anchors(text):
    result=set()
    for line in lines_outside_fences(text):
        result.update(re.findall(r'<a\s+(?:name|id)=[\"\']([^\"\']+)[\"\']',line))
        match=re.match(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$',line)
        if match:
            title=re.sub(r'[`*_~]','',match.group(1).strip()).lower()
            slug=re.sub(r'[^\w -]','',title).replace(' ','-')
            candidate=slug; n=0
            while candidate in result:
                n+=1; candidate=f'{slug}-{n}'
            result.add(candidate)
    return result


def safe_file(root, rel):
    root=Path(root).resolve(strict=True)
    if not isinstance(rel,str) or not rel or '\\' in rel or ':' in rel:
        raise ValueError('invalid local path')
    raw=PurePosixPath(rel)
    if raw.is_absolute(): raise ValueError('absolute path refused')
    path=root
    for part in raw.parts:
        path=path/part
        if path.is_symlink(): raise ValueError('symlink refused')
    path=path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('missing/outside local file')
    return path


def load(root, name='docs/NAVIGATION.json'):
    path=safe_file(root,name)
    if path.stat().st_size>1_000_000: raise ValueError('manifest too large')
    data=json.loads(path.read_text(),object_pairs_hook=unique)
    if not isinstance(data,dict) or type(data.get('version')) is not int or data['version']!=1:
        raise ValueError('unknown navigation schema')
    pages=data.get('pages')
    if not isinstance(pages,list) or not 1<=len(pages)<=32 or not all(isinstance(p,str) for p in pages) or len(set(pages))!=len(pages):
        raise ValueError('invalid page list')
    rows=data.get('public_targets')
    if not isinstance(rows,list) or len(rows)>64: raise ValueError('invalid public targets')
    seen=set()
    for row in rows:
        if not isinstance(row,dict) or row.get('repository') not in PUBLIC: raise ValueError('nonpublic target refused')
        path=row.get('path')
        if not isinstance(path,str) or not path or '\\' in path or ':' in path: raise ValueError('invalid public path')
        pp=PurePosixPath(path)
        if pp.is_absolute() or '..' in pp.parts or str(pp)!=path: raise ValueError('unsafe public path')
        if row.get('ref')!='main': raise ValueError('this manifest checks default main targets')
        if type(row.get('bytes')) is not int or not 0<=row['bytes']<=1_000_000: raise ValueError('invalid size')
        if not isinstance(row.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}',row['sha256']): raise ValueError('invalid hash')
        key=(row['repository'],path)
        if key in seen: raise ValueError('duplicate public target')
        seen.add(key)
    return data


def public_readback(row):
    url=f"https://raw.githubusercontent.com/d6g8k5htny-coder/{row['repository']}/main/{row['path']}"
    with urllib.request.urlopen(url,timeout=20) as response:
        raw=response.read(row['bytes']+1)
    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
        raise ValueError('public source changed: '+row['repository']+'/'+row['path'])
    return row['repository']+'/'+row['path']


def check(root=ROOT, verify_public=False):
    root=Path(root).resolve(strict=True)
    data=load(root)
    problems=[]; local=fragments=external=0; readback=[]
    links=set()
    for page in data['pages']:
        text=safe_file(root,page).read_text()
        for line in lines_outside_fences(text):
            for target in LINK.findall(line):
                links.add(target)
                parsed=urlsplit(target)
                if parsed.scheme in ('https','http','mailto'):
                    external+=1; continue
                if parsed.scheme or parsed.netloc:
                    problems.append(page+': unsupported link '+target); continue
                try:
                    rel=str(PurePosixPath(page).parent/unquote(parsed.path)) if parsed.path else page
                    dest=safe_file(root,rel)
                    local+=1
                    if parsed.fragment:
                        fragments+=1
                        if unquote(parsed.fragment) not in anchors(dest.read_text()): raise ValueError('missing fragment '+parsed.fragment)
                except (ValueError,OSError) as error:
                    problems.append(page+': '+target+': '+str(error))
    for row in data['public_targets']:
        target=f"https://github.com/d6g8k5htny-coder/{row['repository']}/blob/main/{row['path']}"
        if target not in links: problems.append('listed public source has no direct navigation link: '+target)
    if verify_public:
        for row in data['public_targets']:
            try: readback.append(public_readback(row))
            except (ValueError,OSError) as error: problems.append(str(error))
    return {'pages':len(data['pages']),'local_links':local,'fragment_links':fragments,
            'external_links_not_crawled':external,'public_sources_verified':readback,
            'problems':problems,'scope':'declared inline links/ATX anchors and optional named public bytes only; not research acceptance'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--verify-public',action='store_true')
    args=p.parse_args()
    try: result=check(args.root,args.verify_public)
    except (ValueError,OSError,TypeError,KeyError) as error:
        print(json.dumps({'error':str(error)})); return 2
    print(json.dumps(result,indent=2,sort_keys=True))
    return 1 if result['problems'] else 0


if __name__=='__main__': raise SystemExit(main())
