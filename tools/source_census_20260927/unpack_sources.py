#!/usr/bin/env python3
"""Extract inert q0 carrier payloads only after a full SHA256 and size match.

The carrier's manifest establishes declared byte identity, not truth, acceptance,
original authorship, or global source availability. No input code is executed.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

LIMIT = 32 * 1024 * 1024
FULL_HASH = re.compile(r'^[0-9a-f]{64}$')
BEGIN = re.compile(rb'<!-- BEGIN SOURCE: ([^\r\n|]+?)\s*\|\s*sha256:([0-9a-f]+)[^\r\n]*?-->')
END = re.compile(rb'<!-- END SOURCE: ([^\r\n]+?)\s*-->')


def safe_name(name):
    return isinstance(name,str) and bool(name) and not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and '\\' not in name and '\x00' not in name and not re.match(r'^[A-Za-z]:',name)


def unique_object(pairs):
    result={}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate JSON key: '+k)
        result[k]=v
    return result


def load_json(data):
    if len(data)>LIMIT: raise ValueError('input exceeds byte limit')
    return json.loads(data,object_pairs_hook=unique_object)


def identity(data):
    return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'git_blob':hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()}


def accept(name,data,expected,method,**extra):
    if not safe_name(name) or not isinstance(expected,dict): return None
    if not FULL_HASH.fullmatch(str(expected.get('sha256',''))): return None
    if type(expected.get('bytes')) is not int or expected['bytes']<0: return None
    if len(data)>LIMIT or len(data)!=expected['bytes']: return None
    facts=identity(data)
    if facts['sha256']!=expected['sha256']: return None
    return {'name':name,**facts,'method':method,'payload':data,
            'source_disposition':expected.get('disposition','not_declared'),**extra}


def extract_marked(data,manifest):
    if len(data)>LIMIT: raise ValueError('input exceeds byte limit')
    records=[];issues=[];pos=0
    for start in BEGIN.finditer(data):
        if start.start()<pos: continue
        name=start.group(1).decode('utf-8').strip(); finish=END.search(data,start.end())
        nested=BEGIN.search(data,start.end())
        if not finish or finish.group(1).decode('utf-8').strip()!=name or (nested and nested.start()<finish.start()):
            issues.append({'name':name,'reason':'missing_mismatched_or_nested_end'});continue
        pos=finish.end();raw=data[start.end():finish.start()];expected=manifest.get(name)
        matched=None
        # Remove only 0..4 literal LF delimiter bytes at each boundary. No other
        # normalization is performed; the full external digest selects the slice.
        for left in range(5):
            if raw[:left]!=b'\n'*left: continue
            for right in range(5):
                if right and raw[-right:]!=b'\n'*right: continue
                if left+right>len(raw):continue
                candidate=raw[left:len(raw)-right if right else len(raw)]
                r=accept(name,candidate,expected,'literal_marker_slice',byte_start=start.end()+left,byte_end=finish.start()-right,declared_prefix=start.group(2).decode())
                if r and r['sha256'].startswith(start.group(2).decode()):matched=r;break
            if matched:break
        if matched: records.append(matched)
        else:issues.append({'name':name,'reason':'no_full_hash_and_size_match','declared_prefix':start.group(2).decode()})
    # These marked sections explicitly identify themselves as OCR/text extracts.
    # Their original-file prefix does not identify the bytes of the reading copy.
    derived=re.compile(rb'<!-- BEGIN SOURCE: ([^\r\n|]+?)\s*\|\s*original-sha256:([0-9a-f]+)[^\r\n]*?-->')
    for marker in derived.finditer(data):
        issues.append({'name':marker.group(1).decode('utf-8').strip(),
                       'reason':'derived_text_not_original_file_bytes',
                       'declared_original_prefix':marker.group(2).decode(),
                       'marker_byte_offset':marker.start()})
    return records,issues


def extract_python(data,manifest):
    if len(data)>LIMIT: raise ValueError('input exceeds byte limit')
    maps={'_S64':{},'HASHES':{}};issues=[];records=[]
    tree=ast.parse(data)
    for node in tree.body:
        if not isinstance(node,ast.Assign):continue
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in maps:
                try:
                    value=ast.literal_eval(node.value)
                    if not isinstance(value,dict) or maps[target.id]:raise ValueError('nonempty reassignment')
                    maps[target.id]=value
                except (ValueError,TypeError,SyntaxError):issues.append({'name':target.id,'reason':'nonliteral_or_duplicate_assignment'})
            elif isinstance(target,ast.Subscript) and isinstance(target.value,ast.Name) and target.value.id in maps:
                label=target.value.id
                try:
                    name=ast.literal_eval(target.slice);value=ast.literal_eval(node.value)
                    if not isinstance(name,str) or not isinstance(value,str) or name in maps[label]:raise ValueError('invalid or repeated payload assignment')
                    maps[label][name]=value
                except (ValueError,TypeError,SyntaxError):issues.append({'name':label,'reason':'nonliteral_or_duplicate_assignment'})
    for name,value in maps['_S64'].items():
        try:
            if not isinstance(value,str):raise ValueError('not a literal string')
            payload=base64.b64decode(value,validate=True)
            declared=maps['HASHES'].get(name)
            if not isinstance(declared,str) or not re.fullmatch(r'[0-9a-f]{12,64}',declared):raise ValueError('missing or malformed script hash declaration')
            if not hashlib.sha256(payload).hexdigest().startswith(declared):raise ValueError('script hash mismatch')
            expected=manifest.get(name)
            # A prefix is only corroboration. A full external manifest hash AND
            # size are required to call this an original source recovery.
            scope='original_manifest' if expected else 'container_hash_only'
            if expected is None:
                if not FULL_HASH.fullmatch(declared):raise ValueError('hash prefix without full original manifest')
                expected={'sha256':declared,'bytes':len(payload),'disposition':'container-declared; original-manifest absent'}
            r=accept(name,payload,expected,'ast_literal_base64',identity_scope=scope)
            if r:records.append(r)
            else:raise ValueError('manifest identity mismatch')
        except (ValueError,TypeError,UnicodeError) as e:issues.append({'name':str(name),'reason':str(e)})
    return records,issues


def extract_machine(data):
    obj=load_json(data);manifest=obj.get('manifest',{});records=[];issues=[]
    if not isinstance(manifest,dict):raise ValueError('manifest must be an object')
    for group in ('json_artifacts','csv_artifacts','text_artifacts','png_artifacts_base64'):
        entries=obj.get(group,{})
        if not isinstance(entries,dict):raise ValueError('artifact group must be an object')
        for name,value in entries.items():
            candidates=[]
            if group=='json_artifacts':
                if isinstance(value,dict) and isinstance(value.get('_raw_text'),str):
                    candidates.append(('stored_raw_text',value['_raw_text'].encode()))
                for indent in (None,1,2,4):
                    for ascii_only in (True,False):
                        for sorted_keys in (False,True):
                            text=json.dumps(value,indent=indent,ensure_ascii=ascii_only,sort_keys=sorted_keys)
                            for ending in ('','\n'):
                                candidates.append((f'json_indent={indent};ascii={ascii_only};sort={sorted_keys};lf={bool(ending)}',(text+ending).encode()))
                candidates.append(('json_compact',json.dumps(value,separators=(',',':')).encode()))
            elif group=='png_artifacts_base64':
                try:candidates.append(('stored_base64',base64.b64decode(value,validate=True)))
                except (ValueError,TypeError):pass
            elif isinstance(value,str):candidates.append(('stored_text',value.encode()))
            matched=None
            for method,payload in candidates:
                matched=accept(name,payload,manifest.get(name),method,json_pointer='/'+group+'/'+name)
                if matched:break
            if matched:records.append(matched)
            else:issues.append({'name':name,'group':group,'reason':'no_full_hash_and_size_match; semantic content not promoted'})
    return records,issues


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master',required=True);p.add_argument('--ledger',required=True)
    p.add_argument('--machine',required=True);p.add_argument('--scripts',required=True)
    p.add_argument('--out',required=True);p.add_argument('--remaining')
    args=p.parse_args();out=Path(args.out).resolve()
    inputs={label:Path(getattr(args,label)).absolute() for label in ('master','ledger','machine','scripts')}
    if out.exists():raise ValueError('output must be a new directory; no overwrite')
    if any(path.is_symlink() or not path.is_file() or path.stat().st_size>LIMIT for path in inputs.values()):raise ValueError('invalid or oversized input')
    data={label:path.read_bytes() for label,path in inputs.items()}
    manifest=load_json(data['machine'])['manifest'];all_records=[];all_issues=[]
    for label,payload in data.items():
        if label in ('master','ledger'):records,issues=extract_marked(payload,manifest)
        elif label=='machine':records,issues=extract_machine(payload)
        else:records,issues=extract_python(payload,manifest)
        parent={'container':inputs[label].name,'container_sha256':identity(payload)['sha256']}
        all_records += [dict(r,**parent) for r in records]
        all_issues += [dict(r,**parent) for r in issues]
    out.mkdir(parents=True);(out/'payloads').mkdir();receipt=[]
    for r in all_records:
        payload=r.pop('payload');extension=Path(r['name']).suffix
        if not re.fullmatch(r'\.[A-Za-z0-9]{1,12}',extension):extension='.bin'
        target=out/'payloads'/(r['sha256']+extension)
        if not target.exists():target.write_bytes(payload)
        r['output_path']=target.relative_to(out).as_posix();receipt.append(r)
    (out/'EXTRACTIONS.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    (out/'UNRESOLVED_EMBEDDED.json').write_text(json.dumps(all_issues,ensure_ascii=False,indent=2)+'\n')
    summary={'extracted_occurrences':len(receipt),'unique_hashes':len({r['sha256'] for r in receipt}),'unresolved_embedded_occurrences':len(all_issues),'execution':'none','identity_is_not_acceptance':True,'inputs':{label:dict(path=inputs[label].name,**identity(raw)) for label,raw in data.items()}}
    if args.remaining:
        with open(args.remaining,newline='') as f:rows=list(csv.DictReader(f))
        recovered={r['sha256'] for r in receipt};found=[r for r in rows if r['sha256'] in recovered];left=[r for r in rows if r['sha256'] not in recovered]
        for filename,contents in (('R2_RESOLVED.csv',found),('R2_REMAINING.csv',left)):
            with (out/filename).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['file','sha256']);w.writeheader();w.writerows(contents)
        summary.update(r2_baseline_rows=len(rows),r2_resolved_rows=len(found),r2_resolved_hashes=len({r['sha256'] for r in found}),r2_remaining_rows=len(left),r2_remaining_hashes=len({r['sha256'] for r in left}))
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
