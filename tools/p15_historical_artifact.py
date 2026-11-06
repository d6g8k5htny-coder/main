#!/usr/bin/env python3
"""Verify the bounded historical P15 receipt and archived evidence bytes.

This checker verifies byte/identity consistency only; it does not execute Lean,
prove source-to-prose alignment, or certify the current federation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import zipfile

HEX64 = re.compile(r"[0-9a-f]{64}")

def digest(data):
    return hashlib.sha256(data).hexdigest()

def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key]=value
    return result

def safe(name):
    if not isinstance(name,str) or not name or "\\" in name:
        raise ValueError("unsafe member path")
    path=PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path)+("/" if name.endswith("/") else "")!=name:
        raise ValueError("unsafe member path")

def verify(archive, manifest, source_root, expected_zip, expected_receipt, expected_manifest,
           checked_commit, expected_total=59, expected_p15=28, expected_run="38096827443"):
    if Path(archive).stat().st_size>64*1024*1024:
        raise ValueError("archive bounds exceeded")
    raw=Path(archive).read_bytes()
    if digest(raw)!=expected_zip:  # MUT_ARCHIVE
        raise ValueError("archive SHA256 mismatch")
    source=Path(manifest).read_bytes()
    if digest(source)!=expected_manifest:  # MUT_MANIFEST
        raise ValueError("manifest SHA256 mismatch")
    data=json.loads(source,object_pairs_hook=unique)
    source_files=data.get("files")
    if not isinstance(source_files,dict) or not source_files:
        raise ValueError("empty source file inventory")
    bound={}
    root=Path(source_root)
    for name,want in source_files.items():
        safe(name)
        if not isinstance(want,str) or not HEX64.fullmatch(want):
            raise ValueError("invalid source file digest")
        path=root/name
        if any(root.joinpath(*PurePosixPath(name).parts[:i]).is_symlink()
               for i in range(1,len(PurePosixPath(name).parts)+1)):
            raise ValueError("symlinked source file")
        source_data=path.read_bytes()
        if digest(source_data)!=want:  # MUT_SOURCE_FILE
            raise ValueError("source file SHA256 mismatch")
        bound[name]=want
    targets=[row["name"] for row in data["targets"]]
    if len(targets)!=expected_total or len(set(targets))!=expected_total:
        raise ValueError("target inventory mismatch")
    p15=[name for name in targets if name.startswith("UniversalLaw.P15.")]
    if len(p15)!=expected_p15:  # MUT_TARGETS
        raise ValueError("P15 target count mismatch")
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        if not entries or len(entries)>10000 or sum(e.file_size for e in entries)>128*1024*1024:
            raise ValueError("archive bounds exceeded")
        files={}
        for e in entries:
            safe(e.filename)
            if e.filename in files:
                raise ValueError("duplicate archive entry")
            if stat.S_IFMT(e.external_attr >> 16) not in (0,stat.S_IFREG,stat.S_IFDIR):
                raise ValueError("special archive entry")
            if e.flag_bits & 1:
                raise ValueError("encrypted archive entry")
            files[e.filename]=e
        member_hashes={name:digest(z.read(info)) for name,info in files.items() if not info.is_dir()}
        receipts=[name for name,e in files.items() if not e.is_dir() and
                  (name in {"receipt.json","receipt/receipt.json"} or name.endswith("/receipt/receipt.json"))]
        if len(receipts)!=1:
            raise ValueError("receipt inventory mismatch")
        receipt_name=receipts[0]
        receipt_bytes=z.read(files[receipt_name])
        if digest(receipt_bytes)!=expected_receipt:  # MUT_RECEIPT
            raise ValueError("receipt SHA256 mismatch")
        receipt=json.loads(receipt_bytes,object_pairs_hook=unique)
        if type(receipt.get("schema_version")) is not int or receipt.get("schema_version")!=1 or receipt.get("checked_commit")!=checked_commit or receipt.get("workflow_run_id")!=expected_run or receipt.get("repository")!="d6g8k5htny-coder/main":
            raise ValueError("receipt source identity mismatch")
        if receipt.get("manifest_sha256")!=expected_manifest or receipt.get("formalization_status")!="kernel-checked":
            raise ValueError("receipt manifest/status mismatch")
        axioms=receipt.get("axioms")
        if not isinstance(axioms,dict) or set(axioms)!=set(targets):
            raise ValueError("receipt theorem inventory mismatch")
        if any(axioms[name]!=[] for name in targets):  # MUT_AXIOMS
            raise ValueError("unexpected theorem axiom")
        logs=receipt.get("logs")
        if not isinstance(logs,dict) or not logs:
            raise ValueError("empty or invalid log inventory")
        prefix=receipt_name.rsplit("/",1)[0]+"/" if "/" in receipt_name else ""
        actual_logs={name[len(prefix):] for name,e in files.items()
                     if not e.is_dir() and name.startswith(prefix)
                     and "/" not in name[len(prefix):] and name.endswith(".log")}
        if actual_logs!=set(logs):
            raise ValueError("receipt log inventory mismatch")
        observed={}
        for name,want in logs.items():
            if not isinstance(name,str) or PurePosixPath(name).name!=name or not name.endswith(".log") or not isinstance(want,str) or not HEX64.fullmatch(want):
                raise ValueError("unsafe log identity")
            got=digest(z.read(files[prefix+name]))
            if got!=want:  # MUT_LOG
                raise ValueError("receipt log SHA256 mismatch")
            observed[name]=got
    return {"schema":"cbz.p15-historical-byte-readback.v1","status":"PASS",
            "scope":"historical downloaded byte/identity consistency; not a new Lean execution or prose alignment",
            "archive_sha256":expected_zip,"receipt_sha256":expected_receipt,
            "manifest_sha256":expected_manifest,"checked_commit":checked_commit,
            "target_count":len(targets),"p15_count":len(p15),"p15_names":p15,"source_files":bound,
            "axioms":{name:axioms[name] for name in p15},
            "log_count":len(observed),"logs":observed,"member_sha256":member_hashes,
            "elaborated_types_log":"elaborated-types.log"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",required=True);p.add_argument("--manifest",required=True);p.add_argument("--source-root",required=True);p.add_argument("--output",required=True)
    args=p.parse_args()
    report=verify(args.archive,args.manifest,args.source_root,
      "7a72c7a3405976c15437e14c9d7201997378704c16003db7bce45444b592ee45",
      "115533a9ba6a5e7dc264376f25821bbc09168c6bdf546424f99cfa37f09485c5",
      "d039ef64a841ac382a9ee7fbea9576f8691322a5dc87cf01eab127084a4394e1",
      "3c02eb7d004c4b7a44f5ce7b04c2c5de94e47c0c")
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))

if __name__=="__main__":
    main()
