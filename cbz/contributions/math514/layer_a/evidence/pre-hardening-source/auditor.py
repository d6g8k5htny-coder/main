#!/usr/bin/env python3
"""G1 bounded static auditor. `STATIC_PASS` is not proof closure. See SPEC.md."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import unicodedata

from source_checks import check_lean, check_python

APPROVED_AXIOMS = frozenset({'propext', 'Classical.choice', 'Quot.sound'})
KINDS = frozenset({'project_lean','math_python','engineering_python','negative_fixture','quarantined_source',
                   'external_pinned','binary_proof','documentation','data'})
RETAINED = frozenset({'negative_fixture','quarantined_source'})
SOURCE_EXTENSIONS = frozenset({'.lean','.py','.pyi','.sh','.bash','.js','.ts','.c','.cc','.cpp','.h','.rs'})
BINARY_EXTENSIONS = frozenset({'.olean','.ilean','.o','.so','.a','.dll','.dylib','.wasm'})
HEX = re.compile(r'[0-9a-f]{64}')
ENTRY_KEYS = frozenset({'kind','sha256','module','language','vacuity','expected_findings','quarantine',
                       'fixture_purpose','pin','nonmath_review','bindings','status_evidence'})


CLASS_KEYS = {
 'project_lean': {'module','vacuity'},
 'math_python': {'module'},
 'engineering_python': {'module','nonmath_review'},
 'negative_fixture': {'language','module','expected_findings','fixture_purpose'},
 'quarantined_source': {'language','module','expected_findings','quarantine'},
 'external_pinned': {'language','module','vacuity','pin','bindings'},
 'binary_proof': {'bindings','pin'},
 'documentation': {'status_evidence'},
 'data': {'status_evidence'},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError('duplicate JSON key: ' + k)
            result[k] = v
        return result
    def refuse(token):
        raise ValueError('noninteger/nonfinite JSON number refused: ' + token)
    return json.loads(text, object_pairs_hook=unique, parse_float=refuse, parse_constant=refuse)


def normalized_path(value):
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        return False
    p = PurePosixPath(value)
    return (not p.is_absolute() and '..' not in p.parts and '.' not in p.parts and str(p) == value and
            unicodedata.normalize('NFC', value) == value and not any(ord(c) < 32 for c in value))


def audit(manifest):
    findings, retained_findings, observations = [], [], []
    records, roots, graph, source_results = {}, {}, {}, {}
    classified_count, scanned_count = 0, 0

    def add(code, detail, root='', path='', line=0):
        findings.append({'code':code, 'root':root, 'path':path, 'line':line, 'detail':detail})

    def nonempty(value):
        return isinstance(value, str) and bool(value.strip())

    def binding(value, owner, label):
        if not isinstance(value, dict) or set(value) != {'root','path','sha256'}:
            add('EVIDENCE_BINDING', 'malformed '+label, *owner); return None
        ref = value.get('root','') + ':' + value.get('path','') if isinstance(value.get('root'),str) and isinstance(value.get('path'),str) else ''
        target = records.get(ref)
        if target is None or target.get('data') is None or value.get('sha256') != target['entry'].get('sha256') or digest(target['data']) != value.get('sha256'):
            add('EVIDENCE_BINDING', 'missing/stale/unreadable '+label, *owner); return None
        try:
            return strict_json(target['data'].decode('utf-8'))
        except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
            add('EVIDENCE_BINDING', 'non-JSON '+label+': '+str(exc), *owner); return None

    if not isinstance(manifest, dict) or set(manifest) != {'schema_version','approved_axioms','roots','module_map'} or type(manifest.get('schema_version')) is not int or manifest.get('schema_version') != 1:
        add('MANIFEST','schema v1 exact top-level keys required')
        manifest = manifest if isinstance(manifest,dict) else {}
    approved = manifest.get('approved_axioms')
    if not isinstance(approved,list) or any(not isinstance(a,str) for a in approved) or len(approved)!=3 or set(approved)!=APPROVED_AXIOMS:
        add('FOUNDATION_PROFILE','approved imported axiom set must be exactly propext, Classical.choice, Quot.sound')
    raw_roots=manifest.get('roots')
    if not isinstance(raw_roots,list) or not raw_roots:
        add('MANIFEST','nonempty roots required');raw_roots=[]
    root_realpaths=[]
    for r in raw_roots:
        if not isinstance(r,dict) or set(r)!={'id','path','tree_id','tracked_files','files'}:
            add('MANIFEST','root exact keys required');continue
        rid=r.get('id')
        if not isinstance(rid,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',rid) or rid in roots:
            add('MANIFEST','invalid or duplicate root id');continue
        if not isinstance(r.get('tree_id'),str) or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',r['tree_id']) is None:
            add('MANIFEST','immutable tree identifier required',rid)
        rp=r.get('path')
        if not isinstance(rp,str) or not Path(rp).is_absolute() or str(Path(rp))!=rp or unicodedata.normalize('NFC',rp)!=rp:
            add('PATH','root must be normalized absolute path',rid);continue
        root=Path(rp)
        if not root.is_dir():
            add('FILE_MISSING','root directory missing',rid);continue
        if any(p.is_symlink() for p in [root,*root.parents]):
            add('PATH_SYMLINK','root or ancestor is a symlink',rid);continue
        real=root.resolve()
        if any(real==other or real.is_relative_to(other) or other.is_relative_to(real) for other in root_realpaths):
            add('PATH','overlapping/aliased roots are forbidden',rid);continue
        root_realpaths.append(real);roots[rid]=r
        tracked, files=r.get('tracked_files'),r.get('files')
        if not isinstance(tracked,list) or not tracked or any(not isinstance(p,str) for p in tracked) or len(set(tracked))!=len(tracked) or not isinstance(files,dict):
            add('MANIFEST','nonempty unique tracked paths and file object required',rid);continue
        if set(tracked)!=set(files):
            add('SCOPE_CLASSIFICATION','tracked/classified path sets differ: missing='+repr(sorted(set(tracked)-set(files)))+' extra='+repr(sorted(set(files)-set(tracked))),rid)
        for path in sorted(set(tracked)|set(files)):
            if not normalized_path(path):
                add('PATH','unsafe, nonnormalized or non-NFC file path',rid,str(path));continue
            if path not in files:
                continue
            entry=files[path];classified_count+=1
            if not isinstance(entry,dict) or set(entry)-ENTRY_KEYS or not {'kind','sha256'}<=set(entry):
                add('CLASSIFICATION','malformed classification or unknown fields',rid,path);continue
            if not isinstance(entry.get('kind'),str) or entry.get('kind') not in KINDS:
                add('CLASSIFICATION','unknown classification: '+str(entry.get('kind')),rid,path);continue
            if set(entry)-{'kind','sha256'}-CLASS_KEYS.get(entry['kind'],set()):
                add('CLASSIFICATION','fields not applicable to declared classification',rid,path)
            if 'language' in entry and (not isinstance(entry['language'],str) or entry['language'] not in {'lean','python'}):
                add('LANGUAGE_GAP','unsupported/nonstring language classification',rid,path);continue
            if 'module' in entry and not isinstance(entry['module'],str):
                add('MODULE_ALIAS','module identity must be a string',rid,path);continue
            if not isinstance(entry.get('sha256'),str) or HEX.fullmatch(entry['sha256']) is None:
                add('FILE_HASH','missing or malformed SHA256',rid,path)
            target=root/path
            record={'root':rid,'path':path,'entry':entry,'data':None};records[rid+':'+path]=record
            if any((root.joinpath(*PurePosixPath(path).parts[:i])).is_symlink() for i in range(1,len(PurePosixPath(path).parts)+1)):
                add('PATH_SYMLINK','symlink file/ancestor aliases are forbidden',rid,path);continue
            try:
                info=target.stat()
                if not stat.S_ISREG(info.st_mode):
                    add('FILE_MISSING','not a regular file',rid,path);continue
                if not info.st_mode & (stat.S_IRUSR|stat.S_IRGRP|stat.S_IROTH):
                    add('FILE_UNREADABLE','no read permissions even under privileged runner',rid,path);continue
                data=target.read_bytes()
            except FileNotFoundError:
                add('FILE_MISSING','declared file is absent',rid,path);continue
            except OSError as exc:
                add('FILE_UNREADABLE',str(exc),rid,path);continue
            record['data']=data;scanned_count+=1
            if digest(data)!=entry.get('sha256'):
                add('FILE_HASH','content differs from manifest pin',rid,path)
    module_map=manifest.get('module_map')
    if not isinstance(module_map,dict):
        add('MANIFEST','module_map must be an object');module_map={}
    for name, ref in module_map.items():
        if not isinstance(name,str) or not re.fullmatch(r'(lean|python):[^\s:]+',name) or not isinstance(ref,str) or ref not in records:
            add('MODULE_ALIAS','module map points outside classified scope: '+str(name));continue
        if records[ref]['entry'].get('kind') in {'documentation','data','binary_proof'}:
            add('MODULE_ALIAS','module identity must point to actual classified source: '+name)
        if records[ref]['entry'].get('module')!=name:
            add('MODULE_ALIAS','module map alias does not equal target declared identity: '+name)
    for ref, record in records.items():
        rid,path,entry=record['root'],record['path'],record['entry']
        owner=(rid,path);kind=entry['kind'];data=record['data'];suffix=Path(path).suffix.lower()
        language=entry.get('language')
        if kind=='project_lean': language='lean'
        if kind in {'math_python','engineering_python'}: language='python'
        source_like=suffix in SOURCE_EXTENSIONS or path.lower().endswith(('.lean.txt','.py.txt','.sh.txt','.js.txt'))
        if kind in {'documentation','data'} and (source_like or suffix in BINARY_EXTENSIONS):
            add('SOURCE_CLASSIFICATION','source/proof object cannot be relabeled documentation/data',rid,path)
        if kind=='binary_proof' and source_like:
            add('SOURCE_CLASSIFICATION','source cannot be relabeled binary proof',rid,path)
        if kind in {'project_lean','math_python','engineering_python'} and suffix not in ({'.lean'} if language=='lean' else {'.py','.pyi'}):
            add('SOURCE_CLASSIFICATION','source kind does not match extension',rid,path)
        if kind in RETAINED|{'external_pinned'} and source_like and language not in {'lean','python'}:
            add('LANGUAGE_GAP','source needs supported explicit lean/python classification',rid,path)
        expected_language='lean' if path.lower().endswith(('.lean','.lean.txt')) else 'python' if path.lower().endswith(('.py','.pyi','.py.txt')) else None
        if expected_language is not None and language is not None and language!=expected_language:
            add('SOURCE_CLASSIFICATION','declared language disagrees with source extension',rid,path)
        if language is not None and language not in {'lean','python'}:
            add('LANGUAGE_GAP','unsupported language classification',rid,path)
        if kind in RETAINED and (not isinstance(entry.get('expected_findings'),dict) or any(not isinstance(k,str) or type(v) is not int or v<1 for k,v in entry.get('expected_findings',{}).items())):
            add('RETAINED_FINDING_MISMATCH','exact positive finding-count map required',rid,path)
        if kind=='negative_fixture' and not nonempty(entry.get('fixture_purpose')):
            add('NEGATIVE_FIXTURE','retained test purpose required',rid,path)
        if kind=='quarantined_source':
            q=entry.get('quarantine')
            if not isinstance(q,dict) or set(q)!={'id','boundary','blocking_reason','owner','maintenance'} or any(not nonempty(q.get(k)) for k in ['id','boundary','blocking_reason']) or q.get('owner')!='none' or q.get('maintenance')!='zero':
                add('QUARANTINE','exact id/boundary/blocking reason and terminal owner:none maintenance:zero required',rid,path)
        if kind=='external_pinned' or 'pin' in entry:
            pin=entry.get('pin')
            if not isinstance(pin,dict) or set(pin)!={'revision','origin'} or not nonempty(pin.get('origin')) or not isinstance(pin.get('revision'),str) or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}',pin['revision']) is None:
                add('EXTERNAL_PIN','immutable 40/64 hex revision and nonempty origin required',rid,path)
        if kind=='engineering_python':
            review=binding(entry.get('nonmath_review'),owner,'nonmathematical review')
            if not isinstance(review,dict) or set(review)!={'type','source_sha256','scope','reviewer','rationale'} or review.get('type')!='nonmathematical_review' or review.get('scope')!='nonmathematical_execution_only' or review.get('source_sha256')!=entry['sha256'] or not nonempty(review.get('reviewer')) or not nonempty(review.get('rationale')):
                add('NONMATH_REVIEW','explicit hash-bound reviewed nonmathematical scope required',rid,path)
        if kind=='binary_proof' or suffix in BINARY_EXTENSIONS:
            bs=entry.get('bindings')
            if not isinstance(bs,dict) or set(bs)!={'G2','G4'}:
                add('BINARY_BINDING','binary proof requires exact G2/G4 bindings',rid,path)
            else:
                for gate,value in bs.items():
                    proof=binding(value,owner,gate+' binary proof evidence')
                    if not isinstance(proof,dict) or proof.get('type')!='kernel_evidence' or proof.get('gate')!=gate or proof.get('subject_sha256')!=entry['sha256'] or proof.get('observed_status')!='PASS' or type(proof.get('exit_code')) is not int or proof.get('exit_code')!=0 or not isinstance(proof.get('command'),list) or not proof['command'] or not nonempty(proof.get('run_id')):
                        add('BINARY_BINDING','missing/stale/nonpassing '+gate+' subject evidence',rid,path)
        if data is None:
            continue
        text=None
        if kind!='binary_proof' and suffix not in BINARY_EXTENSIONS:
            try:
                text=data.decode('utf-8')
            except UnicodeError:
                add('FILE_ENCODING','nonbinary classified file is not strict UTF-8',rid,path);continue
        if language in {'lean','python'}:
            if text is None:
                add('SOURCE_CLASSIFICATION','binary object cannot be declared Lean/Python source',rid,path);continue
            module=entry.get('module')
            if not isinstance(module,str) or not module.startswith(language+':') or module_map.get(module)!=ref:
                add('MODULE_ALIAS','source module must have one exact registered identity',rid,path)
            result=check_lean(text) if language=='lean' else check_python(text, kind!='engineering_python')
            source_results[ref]=result
            raw_findings=list(result['findings'])
            if kind in RETAINED:
                actual=dict(Counter(f['code'] for f in raw_findings))
                if actual!=entry.get('expected_findings'):
                    add('RETAINED_FINDING_MISMATCH','expected '+repr(entry.get('expected_findings'))+' actual '+repr(actual),rid,path)
                for f in raw_findings:
                    retained_findings.append({'root':rid,'path':path,**f})
            else:
                for f in raw_findings:
                    add(f['code'],f['detail'],rid,path,f['line'])
            graph[ref]=[]
            for imported in result['imports']:
                if language=='python' and imported=='fractions' and kind=='math_python' and 'python:fractions' not in module_map:
                    continue  # Explicit exact-arithmetic trust profile; G7 binds runtime/stdlib.
                target=module_map.get(language+':'+imported)
                if target not in records:
                    add('IMPORT_UNRESOLVED','unknown/unsupported dependency '+language+':'+imported,rid,path)
                else:
                    graph[ref].append(target)
                    if kind=='math_python' and records[target]['entry']['kind']=='engineering_python':
                        add('MATH_DEPENDENCY_GAP','mathematical code cannot acquire values from nonmathematical-only source '+target,rid,path)
                if language=='python':
                    parts=imported.split('.')
                    for i in range(1,len(parts)):
                        prefix=module_map.get('python:'+'.'.join(parts[:i]))
                        if prefix in records:
                            graph[ref].append(prefix)
            for candidate in result.get('import_candidates',[]):
                target=module_map.get('python:'+candidate)
                if target in records:
                    graph[ref].append(target)
            if language=='lean' and kind not in RETAINED:
                annotations=entry.get('vacuity')
                declarations=result['declarations']
                keys=[d['key'] for d in declarations]
                if not isinstance(annotations,dict) or len(keys)!=len(set(keys)) or set(annotations)!=set(keys):
                    add('VACUITY_COVERAGE','annotation keys must exactly cover theorem/lemma/example source locations',rid,path)
                    annotations=annotations if isinstance(annotations,dict) else {}
                for declaration in declarations:
                    key=declaration['key'];a=annotations.get(key)
                    if not isinstance(a,dict) or not isinstance(a.get('status'),str) or a.get('status') not in {'conditional','nonvacuous','vacuous'} or not nonempty(a.get('reason')):
                        add('VACUITY_ANNOTATION','required status/reason for '+key,rid,path,declaration['line']);continue
                    if set(a)-{'status','reason','premises','evidence'}:
                        add('VACUITY_ANNOTATION','unknown annotation keys for '+key,rid,path,declaration['line'])
                    if a['status']=='conditional' and (not isinstance(a.get('premises'),list) or any(not nonempty(p) for p in a['premises'])):
                        add('VACUITY_ANNOTATION','explicit premises list required for '+key,rid,path,declaration['line'])
                    if declaration['contradictions'] and a['status']!='vacuous':
                        add('VACUITY_CONTRADICTION','syntactic contradiction candidates require vacuous: '+key+': '+repr(declaration['contradictions']),rid,path,declaration['line'])
                    if a['status']=='nonvacuous':
                        certificate=binding(a.get('evidence'),owner,'nonvacuity certificate '+key)
                        if not isinstance(certificate,dict) or certificate.get('type')!='nonvacuity_certificate' or certificate.get('subject_sha256')!=entry['sha256'] or certificate.get('declaration')!=key or not nonempty(certificate.get('witness_description')):
                            add('VACUITY_EVIDENCE','content-bound nonvacuity witness certificate required for '+key,rid,path,declaration['line'])
                    observations.append({'root':rid,'path':path,'declaration':key,'status':a['status'],'contradiction_candidates':declaration['contradictions']})
        if entry.get('status_evidence') is not None:
            if entry['status_evidence'] is not True:
                add('STATUS_EVIDENCE','status_evidence must be true, never disable/skip',rid,path)
            try:
                s=strict_json(text)
            except (ValueError,TypeError,RecursionError) as exc:
                add('STATUS_EVIDENCE','invalid status JSON: '+str(exc),rid,path);continue
            required={'schema_version','gate','observed_status','run_id','command','exit_code','inputs'}
            if not isinstance(s,dict) or not required<=set(s) or set(s)-required-{'evidence'} or type(s.get('schema_version')) is not int or s.get('schema_version')!=1 or not isinstance(s.get('gate'),str) or s.get('gate') not in {'G'+str(i) for i in range(12)} or not isinstance(s.get('observed_status'),str) or s.get('observed_status') not in {'PASS','FAIL','NOT_RUN','ERROR','INACCESSIBLE'} or not nonempty(s.get('run_id')) or not isinstance(s.get('command'),list) or not s['command'] or any(not nonempty(c) for c in s['command']) or not isinstance(s.get('inputs'),dict) or not s['inputs']:
                add('STATUS_EVIDENCE','invalid exact observed status schema',rid,path);continue
            for input_ref,value in s['inputs'].items():
                target=records.get(input_ref)
                if target is None or value!=target['entry'].get('sha256') or target['data'] is None or digest(target['data'])!=value:
                    add('STATUS_EVIDENCE','status input missing/stale: '+input_ref,rid,path)
            if s['observed_status']=='NOT_RUN':
                if s['exit_code'] is not None or 'evidence' in s:
                    add('STATUS_EVIDENCE','NOT_RUN cannot carry execution exit/evidence',rid,path)
            else:
                evidence=binding(s.get('evidence'),owner,'observed status receipt')
                comparable={k:v for k,v in s.items() if k!='evidence'}
                if evidence!=comparable or type(s['exit_code']) is not int or (s['observed_status']=='PASS' and s['exit_code']!=0) or (s['observed_status']=='FAIL' and s['exit_code']==0):
                    add('STATUS_EVIDENCE','status differs from bound observed execution receipt',rid,path)
    for start in sorted(graph):
        if records[start]['entry']['kind'] in RETAINED:
            continue
        seen,pending=set(),list(graph[start])
        while pending:
            target=pending.pop()
            if target in seen:
                continue
            seen.add(target)
            if records[target]['entry']['kind'] in RETAINED:
                add('QUARANTINE_IMPORT','closed source depends on isolated source '+target,records[start]['root'],records[start]['path'])
            pending.extend(graph.get(target,[]))
    findings.sort(key=lambda f:(f['root'],f['path'],f['line'],f['code'],f['detail']))
    retained_findings.sort(key=lambda f:(f['root'],f['path'],f['line'],f['code'],f['detail']))
    manifest_bytes=json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    inventories={rid:digest(json.dumps(sorted(r['tracked_files']),ensure_ascii=False,separators=(',',':')).encode()) for rid,r in roots.items() if isinstance(r.get('tracked_files'),list) and all(isinstance(p,str) for p in r['tracked_files'])}
    obligations=[{'gate':g,'required_work':work} for g,work in [
        ('G2','All elaborated owned declarations, private/generated/import-expanded declarations included; approved imported axioms exactly propext/Classical.choice/Quot.sound.'),
        ('G3','Actual Lean build and no sorry warnings; source scanning does not execute Lean.'),
        ('G4','Kernel recheck every closed declaration and all bound binary proof objects.'),
        ('G5','Execute retained regression tests; source profile does not establish runtime success or historical test retention.'),
        ('G7','Authenticate immutable dependency revisions and Python exact-runtime/fractions implementation; verify cross-repository consistency.'),
        ('G8','Authenticate command receipts, review/certificate content, semantic status truth, statement nonvacuity and historical bound strength.'),
        ('G10','Authenticate full final tracked-file inventory and actual immutable tree identifiers; scanner does not run Git.')]]
    return {'schema_version':1,'result':'STATIC_PASS' if not findings else 'FAIL','claim':'bounded static rules only; not CBZ-0 or mathematical proof closure',
            'manifest_sha256':digest(manifest_bytes),'inventory_sha256':inventories,'classified_count':classified_count,
            'scanned_count':scanned_count,'retained_count':len(retained_findings),'findings':findings,
            'fatal_gaps':[f for f in findings if f['code'].endswith('_GAP') or f['code'] in {'MANIFEST','SCOPE_CLASSIFICATION','CLASSIFICATION','FILE_MISSING','FILE_UNREADABLE','IMPORT_UNRESOLVED','EVIDENCE_BINDING'}],
            'retained_findings':retained_findings,'vacuity_observations':observations,'remaining_obligations':obligations,
            'files':[{'root':r['root'],'path':r['path'],'kind':r['entry']['kind'],'expected_sha256':r['entry']['sha256'],'actual_sha256':digest(r['data']) if r['data'] is not None else None} for _,r in sorted(records.items())]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('audit',help='audit exact controller-supplied tracked scope')
    p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--output',type=Path,help='also write deterministic report JSON')
    args=parser.parse_args(argv)
    try:
        raw=args.manifest.read_bytes()
        manifest=strict_json(raw.decode('utf-8'))
        report=audit(manifest)
        report['manifest_file_sha256']=digest(raw)
        code=0 if report['result']=='STATIC_PASS' else 1
    except (OSError,ValueError,UnicodeError,RecursionError) as exc:
        report={'schema_version':1,'result':'INPUT_ERROR','findings':[{'code':'INPUT','detail':str(exc)}]};code=2
    encoded=json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+'\n'
    if args.output:
        try:
            args.output.write_text(encoded,encoding='utf-8')
        except OSError as exc:
            report={'schema_version':1,'result':'OUTPUT_ERROR','findings':[{'code':'OUTPUT','detail':str(exc)}]}
            encoded=json.dumps(report,sort_keys=True,indent=2)+'\n';code=3
    sys.stdout.write(encoded)
    return code


if __name__=='__main__':
    raise SystemExit(main())
