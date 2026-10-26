#!/usr/bin/env python3
"""NON-CERTIFYING execution wrapper; this is numerical evidence, not acceptance.

Use the frozen #451 driver without altering its 59 jobs. Run in a disposable
detached checkout, not the evidence branch. All subprocess streams are retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent.parent
SOURCE = "0a7e5145b55f020e02e7782fed318266d13c2447"
ENV_SCRIPT = r'''
import contextlib,hashlib,io,json,os,platform,sys
import numpy as np
import mpmath as mp
from mpmath.libmp import BACKEND
b=io.StringIO()
with contextlib.redirect_stdout(b): np.show_config()
print(json.dumps(dict(python=sys.version,executable=sys.executable,
 executable_sha256=hashlib.sha256(open(sys.executable,"rb").read()).hexdigest(),
 numpy=np.__version__,mpmath=mp.__version__,mpmath_backend=BACKEND,
 platform=platform.platform(),machine=platform.machine(),cpu_count=os.cpu_count(),
 processor=platform.processor(),numpy_config=b.getvalue(),
 thread_environment={k:os.environ.get(k) for k in
 ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","PYTHONDONTWRITEBYTECODE"]}),indent=2))
'''


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkout", required=True, type=Path)
    ap.add_argument("--python", required=True, type=Path)
    args = ap.parse_args()
    root = args.checkout.resolve()
    py = str(args.python.absolute())
    pkt = root / "experiments/zeta_fold_exploration"
    log = HERE / "logs"
    log.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    env.pop("ZETA_FOLD_RAW", None)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != SOURCE:
        raise SystemExit("wrong source commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise SystemExit("source checkout is not initially clean")
    records = []

    def capture(name, argv, cwd):
        out, err = log / (name + ".stdout"), log / (name + ".stderr")
        t = time.time()
        with out.open("wb") as fo, err.open("wb") as fe:
            rc = subprocess.run(argv, cwd=cwd, env=env, stdout=fo, stderr=fe,
                                timeout=1200).returncode
        rec = dict(name=name, argv=argv, cwd=str(cwd), start_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(t)), wall_seconds=time.time()-t,
            rc=rc, stdout_sha256=digest(out), stderr_sha256=digest(err),
            stdout_bytes=out.stat().st_size, stderr_bytes=err.stat().st_size)
        records.append(rec)
        (HERE / "setup_and_representative.json").write_text(json.dumps(records, indent=2)+"\n")
        print(name, "rc", rc, "seconds", round(rec["wall_seconds"], 2), flush=True)
        if rc:
            raise SystemExit("failed " + name)
        return out

    environment = capture("environment", [py, "-B", "-c", ENV_SCRIPT], root)
    (HERE / "environment.json").write_bytes(environment.read_bytes())
    capture("manifest_before", [py, "-B", "make_manifest.py", "--check"], pkt)
    representative = capture("representative_count", [py, "-B", "dbn/count_check.py", "1", "103", "0"], pkt)
    original = pkt / "results/dbn/count_1_103_0.out"
    if representative.read_bytes() != original.read_bytes():
        raise SystemExit("representative stdout differs; preserve and investigate before extending")
    print("representative equals original bytes", digest(representative), flush=True)
    children = []
    started = time.time()
    try:
        for lane in ("dbn", "stats"):
            e = dict(env, ZF_PKT=str(pkt), ZF_PY=py, ZF_LOG=str(log), ZF_LANE=lane)
            argv = [py, "-B", str(HERE / "scripts/frozen_driver.py")]
            fo = (log / ("lane_" + lane + ".stdout")).open("wb")
            fe = (log / ("lane_" + lane + ".stderr")).open("wb")
            child = subprocess.Popen(argv, cwd=root, env=e, stdout=fo, stderr=fe, start_new_session=True)
            children.append((lane, child, fo, fe, argv))
        while any(p.poll() is None for _, p, *_ in children):
            if time.time() - started > 2400:
                raise TimeoutError("40-minute full replay ceiling")
            time.sleep(1)
    finally:
        for _, p, fo, fe, _ in children:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=30)
            fo.close()
            fe.close()
        (HERE / "lanes.json").write_text(json.dumps(dict(start_utc=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), wall_seconds=time.time()-started,
            lanes=[dict(lane=l,argv=a,rc=p.returncode,cwd=str(root)) for l,p,_,_,a in children]),indent=2)+"\n")
    jobs = []
    for lane in ("dbn", "stats"):
        jobs.extend(json.loads(line) for line in (log / ("ledger_"+lane+".jsonl")).read_text().splitlines())
    success = len(jobs) == 59 and len({j["name"] for j in jobs}) == 59 and all(j["rc"] == 0 for j in jobs)
    print("full replay jobs", len(jobs), "all_rc_zero", success, flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
