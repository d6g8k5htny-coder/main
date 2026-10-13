"""Fail-closed spectral packet replay. --local-only excludes project Git objects."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


def read_json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique)


def safe_path(path):
    if not isinstance(path,str) or not path or '\\' in path or ':' in path or '\x00' in path:
        raise ValueError('unsafe source path')
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or '.' in path.split('/') or str(p)!=path:
        raise ValueError('unsafe source path')
    return path


def identity(data,entry):
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if (type(entry['bytes']) is not int or len(data)!=entry['bytes']
        or hashlib.sha256(data).hexdigest()!=entry['sha256'] or blob!=entry['git_blob']):
        raise ValueError('identity mismatch: '+entry['path'])


def verify_sources(root):
    entries=read_json(root/'SOURCES.json')['sources']
    ids=[e['id'] for e in entries]
    if not entries or len(ids)!=len(set(ids)):
        raise ValueError('missing or duplicate source inventory')
    for entry in entries:
        commit,path=entry['commit'],safe_path(entry['path'])
        if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('invalid source commit')
        # The pinned tree, not a followed worktree symlink, is the authority.
        out=subprocess.run(['git','ls-tree','-z',commit,'--',path],cwd=root.parents[1],
                           check=True,capture_output=True,timeout=60).stdout
        records=[x for x in out.split(b'\0') if x]
        if len(records)!=1: raise ValueError('missing or ambiguous pinned path: '+path)
        header,actual=records[0].split(b'\t',1)
        mode,kind,blob=header.decode().split()
        if actual.decode()!=path or mode!='100644' or kind!='blob':
            raise ValueError('pinned source must be a regular nonexecutable blob: '+path)
        if blob!=entry['git_blob']: raise ValueError('tree blob mismatch: '+path)
        data=subprocess.run(['git','cat-file','blob',blob],cwd=root.parents[1],check=True,
                            capture_output=True,timeout=60).stdout
        identity(data,entry)
    return len(entries)


def verify_inventory(root):
    if (root/'MANIFEST.json').is_symlink(): raise ValueError('manifest symlink')
    entries=read_json(root/'MANIFEST.json')['files']
    names=[e['path'] for e in entries]
    if len(names)!=len(set(names)): raise ValueError('duplicate packet path')
    if sorted(p.name for p in root.iterdir())!=sorted(names+['MANIFEST.json']):
        raise ValueError('packet inventory mismatch')
    for entry in entries:
        name=safe_path(entry['path']); path=root/name
        if PurePosixPath(name).name!=name or path.is_symlink() or not path.is_file():
            raise ValueError('flat regular packet file required')
        identity(path.read_bytes(),entry)
    return len(entries)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only',action='store_true')
    args=parser.parse_args(); root=Path('/tmp/claude-0/-home-user/469e2708-1be1-5731-8b45-31b653d1d9d3/scratchpad/wt162/frontiers/spectral_cluster_closure_20260929')
    verify_inventory(root)
    source_count=0 if args.local_only else verify_sources(root)
    expected=(root/'RESULTS.json').read_bytes()
    mutants=['drop-spectral-r','drop-soft-vandermonde','wrong-hard-power',
             'drop-far-branch','reverse-cutoff','include-zero',
             'drop-mixed-term','swap-full-normalizer']
    for flags in (['-B','-S'],['-B','-O','-S']):
        subprocess.run([sys.executable,*flags,'-m','unittest','discover','-v'],cwd=root.parents[1],
                       check=True,timeout=120)
        cmd=[sys.executable,*flags,str(root/'check.py')]
        out=subprocess.run(cmd,check=True,capture_output=True,timeout=30)
        if out.stdout!=expected or out.stderr: raise ValueError('deterministic replay mismatch')
        for mutant in mutants+['unknown']:
            out=subprocess.run(cmd+['--mutant',mutant],capture_output=True,timeout=30)
            if out.returncode!=(2 if mutant=='unknown' else 1):
                raise ValueError('negative control not rejected: '+mutant)
    verify_inventory(root)
    print(json.dumps({'passed':True,'source_pins_checked':not args.local_only,
                      'source_count':source_count,'scientific_effect':'NONE',
                      'mathematical_acceptance':False},sort_keys=True))


if __name__=='__main__': main()
