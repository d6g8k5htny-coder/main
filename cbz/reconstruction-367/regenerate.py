#!/usr/bin/env python3
"""Regenerate a source-bound classification from published literal evidence.
This builder does not execute Lean or authenticate historical receipts.
Run the existing cbz_inventory generate/verify commands to check its output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

MAIN = "82d9e480a51b946b3e74525ccf32139129d37d28"
MATH = "9f8dbb4d18b9f3ebd9ffac6b0caa42df9f343966"
EVIDENCE_COMMIT = "3ccd631c45f247268995860669ba08a1cb0c4c1c"
APPROVED = {"Classical.choice", "Quot.sound", "propext"}

def demand(condition, message):
    if not condition:
        raise ValueError(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def git(repo, *args):
    env = {k:v for k,v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_NO_REPLACE_OBJECTS="1", GIT_TERMINAL_PROMPT="0")
    return subprocess.check_output(["git", "--no-replace-objects", "--literal-pathspecs",
                                    "-C", str(repo), *args], env=env, timeout=120)

def bound(repo, name, commit, path, expected=None):
    demand(path and not path.startswith("/") and "\\" not in path and
           all(part not in ("", ".", "..") for part in path.split("/")),
           "unsafe binding path")
    spec = commit + ":" + path
    oid = git(repo, "rev-parse", "--verify", spec).decode("ascii").strip()
    raw = git(repo, "cat-file", "blob", oid)
    checksum = digest(raw)
    demand(expected is None or checksum == expected, "source/evidence digest mismatch: " + path)
    demand(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == oid,
           "Git blob identity mismatch: " + path)
    return {"repository":name, "commit":commit, "path":path,
            "git_object":oid, "sha256":checksum}

def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            demand(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))

def validate_family(entries, manifest, receipt, count, commit):
    demand(len(entries) == count and len({row["id"] for row in entries}) == count,
           "unexpected family size or duplicate id")
    demand(manifest["schema"] == "cbz.kernel-targets.v1" and
           manifest["source_repository"] == "Math-" and manifest["checked_commit"] == commit,
           "manifest identity mismatch")
    demand(receipt["schema"] == "cbz.kernel-replay.receipt.v1" and
           receipt["formalization_status"] == "kernel-checked" and
           receipt["checked_repository"] == "Math-" and receipt["checked_commit"] == commit and
           receipt["source_sha256"] == manifest["files"], "receipt identity mismatch")
    targets = manifest["targets"]
    demand(len(targets) == count and len({x["name"] for x in targets}) == count,
           "unexpected or duplicate target")
    target_paths = {x["name"]:x["path"] for x in targets}
    demand(set(receipt["axioms"]) == set(target_paths), "receipt axiom target set differs")
    demand({x["declaration"] for x in entries} == set(target_paths),
           "literal index target set differs")
    for row in entries:
        name = row["declaration"]
        demand(row["status"] == "CLOSED" and row["repository"] == "Math-" and
               row["source_commit"] == commit and row["path"] == target_paths[name] and
               row["source_sha256"] == manifest["files"][row["path"]],
               "literal source binding mismatch: " + row["id"])
        axioms = receipt["axioms"][name]
        demand(isinstance(axioms,list) and len(axioms)==len(set(axioms)) and
               set(axioms) <= APPROVED and row["axioms"] == axioms,
               "literal axiom mismatch: " + row["id"])

def build(root):
    main, math = root / "main", root / "Math-"
    demand(git(main, "rev-parse", "HEAD").decode().strip() == MAIN, "wrong main source")
    demand(git(math, "rev-parse", "HEAD").decode().strip() == MATH, "wrong Math source")
    existing = read_json(main / "cbz/classifications.json")["obligations"]
    originals = read_json(main / "cbz/evidence/kernel/closed-declarations.json")["entries"]
    demand(len(existing) == len(originals) == 94, "unexpected original index size")
    old = {row["id"]:row for row in originals}
    demand(len(old)==94 and {r["id"] for r in existing}==set(old), "original id set mismatch")
    for entry in existing:
        row=old[entry["id"]]
        demand(entry["status"]=="CLOSED" and entry["kernel_evidence"]["targets"]==[row["declaration"]],
               "original literal target mismatch")
        source=entry["source_bindings"]
        demand(len(source)==1 and source[0]["repository"]==row["repository"] and
               source[0]["commit"]==row["source_commit"] and source[0]["path"]==row["path"] and
               source[0]["sha256"]==row["source_sha256"] and source[0]["git_object"]==row["git_object"],
               "original literal source mismatch")
        if entry["id"].startswith("K-MAIN-"):
            entry["kernel_evidence"]["manifest"]["scope"]="historical"
    result=list(existing)
    for family,count,commit in (("additional",207,"8151f592918a845bf19eb8eafaf8444540a32734"),
                                ("determinant",66,"74565f013575b7b05ecfc4034cd5d638ee355353")):
        prefix="cbz/evidence/kernel/"+family+"/"
        manifest_path=prefix+"kernel-targets.json"
        receipt_path=prefix+"receipt.json"
        manifest=read_json(main/manifest_path)
        receipt=read_json(main/receipt_path)
        entries=read_json(main/(prefix+"closed-declarations.json"))["entries"]
        validate_family(entries,manifest,receipt,count,commit)
        manifest_row=bound(main,"main",EVIDENCE_COMMIT,manifest_path,receipt["manifest_sha256"])
        receipt_row=bound(main,"main",EVIDENCE_COMMIT,receipt_path)
        sources=[bound(math,"Math-",commit,path,checksum)
                 for path,checksum in sorted(manifest["files"].items())]
        source_by_path={row["path"]:row for row in sources}
        raw=[bound(main,"main",EVIDENCE_COMMIT,prefix+path,checksum)
             for path,checksum in sorted(receipt["evidence_sha256"].items())]
        for row in entries:
            primary=source_by_path[row["path"]]
            demand(primary["git_object"]==row["git_object"], "literal Git object mismatch: "+row["id"])
            result.append({"id":row["id"],"status":"CLOSED","source_bindings":[primary],
                "kernel_evidence":{"format":"cbz.kernel-replay.v1","checked_commit":commit,
                                   "targets":[row["declaration"]],"source_bindings":sources,
                                   "manifest":manifest_row,"receipt":receipt_row,"execution_evidence":raw}})
    demand(len(result)==367 and len({x["id"] for x in result})==367,
           "367 distinct context-bound identities required")
    combined=read_json(main/"cbz/evidence/kernel/all-closed-declarations.json")
    combined=combined["entries"] if isinstance(combined,dict) else combined
    expected={x["id"]:x for x in combined}
    demand(len(expected)==367 and set(expected)=={x["id"] for x in result},
           "combined literal index identity mismatch")
    for entry in result:
        row=expected[entry["id"]]
        source=entry["source_bindings"][0]
        demand(entry["kernel_evidence"]["targets"]==[row["declaration"]] and
               source["repository"]==row["repository"] and source["commit"]==row["source_commit"] and
               source["path"]==row["path"] and source["git_object"]==row["git_object"] and
               source["sha256"]==row["source_sha256"], "combined literal binding mismatch: "+row["id"])
    return {"obligations":result}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    demand(not args.output.exists() and not args.output.is_symlink(), "output exists")
    data=build(args.root)
    raw=(json.dumps(data,sort_keys=True,indent=2)+"\n").encode()
    with args.output.open("xb") as file:
        file.write(raw)
    print(json.dumps({"count":len(data["obligations"]),"sha256":digest(raw),
                      "main_source":MAIN,"math_source":MATH,
                      "historical_evidence_commit":EVIDENCE_COMMIT,
                      "scope":"binding reconstruction only; no new Lean execution or CBZ claim"}))
if __name__=="__main__":
    main()
