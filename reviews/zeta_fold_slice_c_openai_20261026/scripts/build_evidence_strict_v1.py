#!/usr/bin/env python3
"""Package a completed NON-CERTIFYING run; never promote scientific status.

Reuses #451's exact timing masks, preserves all 66 output byte streams and
full nonidentical diffs, and independently verifies command/stream coverage.
"""
import argparse
from collections import Counter
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent.parent
REL = "experiments/zeta_fold_exploration"
SOURCE = "0a7e5145b55f020e02e7782fed318266d13c2447"
REVIEW = "8f1d846daeb03990a84a92fa6e42c9903575c412"
RPATH = "reviews/zeta_fold_slice_c_rerun_20261019"
XAI = "78976ae7d412427ac15d2ce01786ca1c24d6b066"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def classify(a, b, mask):
    if a == b:
        return "IDENTICAL"
    return "TIMING-ONLY" if mask(a.decode()) == mask(b.decode()) else "DIFFERENT"


def validate_environment(environment):
    if environment["python"].split()[0] != "3.13.16" or environment["numpy"] != "2.5.3" or environment["mpmath"] != "1.4.1":
        raise ValueError("recorded dependency pins differ")
    if environment["mpmath_backend"] != "python":
        raise ValueError("recorded mpmath backend differs")
    for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        if environment["thread_environment"].get(key) != "1":
            raise ValueError("recorded thread pin differs: " + key)


def validate_jobs(jobs, expected, log, interpreter, root):
    signatures = lambda rows: Counter((j["name"], tuple(j["argv"][1:])) for j in rows)
    if len(jobs) != 59 or signatures(jobs) != signatures(expected):
        raise ValueError("job multiset does not equal all 59 expected commands")
    for j in jobs:
        if j["argv"][0] != interpreter:
            raise ValueError("interpreter mismatch: " + j["name"])
        expected_cwd = root if j["name"] == "fold_controls" else root / REL
        if Path(j["cwd"]).resolve() != expected_cwd.resolve():
            raise ValueError("working directory mismatch: " + j["name"])
        if j["rc"] != 0:
            raise ValueError("nonzero exit: " + j["name"])
        p = log / (j["name"] + ".stdout")
        if sha(p.read_bytes()) != j["stdout_sha256"]:
            raise ValueError("stdout hash mismatch: " + j["name"])
        if (log / (j["name"]+".stderr")).stat().st_size != j["stderr_bytes"]:
            raise ValueError("stderr size mismatch: " + j["name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkout", required=True, type=Path)
    a = ap.parse_args()
    root = a.checkout.resolve()
    pkt = root / REL
    log = HERE / "logs"
    os.environ.update(ZF_PKT=str(pkt),ZF_LOG=str(log))
    spec = importlib.util.spec_from_file_location("frozen_compare", HERE/"scripts/frozen_compare.py")
    comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparison)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=root)
    show = lambda commit,path: git("show",commit+":"+path)
    original = json.loads(show(REVIEW,RPATH+"/comparison/hashes.json"))
    expected = json.loads(show(REVIEW,RPATH+"/comparison/ledger.json"))
    if isinstance(expected, dict):
        expected = expected["jobs"]
    jobs = [json.loads(s) for lane in ("dbn","stats")
            for s in (log/("ledger_"+lane+".jsonl")).read_text().splitlines()]
    environment = json.loads((HERE/"environment.json").read_text())
    validate_environment(environment)
    validate_jobs(jobs, expected, log, environment["executable"], root)
    lanes = json.loads((HERE/"lanes.json").read_text())["lanes"]
    if len(lanes) != 2 or {r["lane"] for r in lanes} != {"dbn","stats"} or any(r["rc"] != 0 for r in lanes):
        raise ValueError("parent lane coverage/exit mismatch")
    if comparison.rebuild_dt_sensitivity() != "rebuilt":
        raise ValueError("missing dt logs")
    xai = {r["path"]:r for r in json.loads(show(XAI,"reviews/zeta_fold_slice_c_r2_replay_20261020/outputs.json"))}
    rows = []
    for r in original["files"]:
        p = r["path"]
        old = show(SOURCE,REL+"/"+p)
        if sha(old) != r["committed_sha256"]:
            raise ValueError("original hash mismatch: "+p)
        same = old if r["class"] == "IDENTICAL" else show(REVIEW,RPATH+"/reproduced/"+p)
        if sha(same) != r["rerun_sha256"]:
            raise ValueError("same-provider hash mismatch: "+p)
        new = (pkt/p).read_bytes()
        target = HERE/"reproduced"/p
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(new)
        row = dict(path=p,committed_blob=r["committed_blob"],original_sha256=sha(old),
            same_provider_sha256=sha(same),openai_sha256=sha(new),openai_bytes=len(new),
            original_bytes=len(old),same_provider_bytes=len(same),
            vs_original=classify(old,new,comparison.mask),
            vs_same_provider=classify(same,new,comparison.mask),
            xai_sha256=xai[p]["sha256"],xai_class=xai[p]["class"],
            byte_identical_to_xai=(sha(new)==xai[p]["sha256"]))
        for name,data in (("original",old),("same_provider",same)):
            if data != new:
                target = HERE/name/p
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(data)
                diff = HERE/"diffs"/(p.replace("/","__")+".vs_"+name+".diff")
                diff.parent.mkdir(exist_ok=True)
                diff.write_text("".join(difflib.unified_diff(data.decode().splitlines(True),
                    new.decode().splitlines(True),fromfile=name,tofile="openai")))
        rows.append(row)
    sources = []
    manifest = json.loads(show(SOURCE,REL+"/MANIFEST.json"))
    for r in manifest["committed_files"]:
        if r["path"].startswith("results/"):
            continue
        data = (pkt/r["path"]).read_bytes()
        if sha(data) != r["sha256"] or len(data) != r["bytes"]:
            raise ValueError("executed source changed: "+r["path"])
        sources.append(r)
    external = "experiments/universality/fold_controls.py"
    data = (root/external).read_bytes()
    if data != show(SOURCE,external):
        raise ValueError("external fold-controls source changed")
    sources.append(dict(path="../../"+external,bytes=len(data),sha256=sha(data)))
    raw = []
    for r in manifest["raw_files"]:
        data = (pkt/r["path"]).read_bytes()
        raw.append(dict(path=r["path"],expected_bytes=r["bytes"],bytes=len(data),
            expected_sha256=r["sha256"],sha256=sha(data),
            match=sha(data)==r["sha256"] and len(data)==r["bytes"]))
    summary = dict(label="NON-CERTIFYING",scientific_effect="NONE",review_status="REVIEW_REQUIRED",
        source=SOURCE,frozen_review=REVIEW,prior_xai=XAI,jobs=len(jobs),
        command_multiset_matches=True,all_rc_zero=True,source_files_verified=len(sources),
        outputs=len(rows),vs_original=dict(Counter(r["vs_original"] for r in rows)),
        vs_same_provider=dict(Counter(r["vs_same_provider"] for r in rows)),
        raw_samples=len(raw),raw_matches=sum(r["match"] for r in raw),
        raw_binary_retention="203573297 bytes regenerated; hashes retained here, large samples excluded from Git as in source packet")
    for name,data in (("comparisons.json",rows),("raw_hashes.json",raw),("source_files.json",sources),
                      ("jobs.json",jobs),("summary.json",summary)):
        (HERE/name).write_text(json.dumps(data,indent=2)+"\n")
    for p in pkt.glob("results/dbn/track_log_dt*.txt"):
        target = HERE/"reproduced/uncommitted"/p.name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
