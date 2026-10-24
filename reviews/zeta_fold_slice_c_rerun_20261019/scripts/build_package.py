#!/usr/bin/env python3
"""Assemble the slice-C rerun record as a self-contained directory for an engineering PR.

Inputs: the detached worktree of main 0a7e5145 after the rerun (ZF_WT), the run logs (ZF_LOG),
the scripts directory (ZF_SCRIPTS) and the drafted README.md/HANDOFF.md (ZF_DOCS).
Output: ZF_OUT/<package dir> with originals, reproductions, comparison, experiments, logs,
scripts, environment.json and MANIFEST.json (bytes + sha256 of every file).
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path(os.environ["ZF_WT"]).resolve()
PKT = WT / "experiments/zeta_fold_exploration"
REL = "experiments/zeta_fold_exploration"
LOG = Path(os.environ["ZF_LOG"]).resolve()
SCRIPTS = Path(os.environ["ZF_SCRIPTS"]).resolve()
DOCS = Path(os.environ["ZF_DOCS"]).resolve()
OUT = Path(os.environ["ZF_OUT"]).resolve()
PY = os.environ["ZF_PY"]
MAIN = "0a7e5145b55f020e02e7782fed318266d13c2447"
RECOVERY = "5d298b9e"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(*a, binary=False):
    r = subprocess.run(["git", *a], cwd=WT, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def put(dst, data: bytes):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# 1. comparison artefacts and full hash table of every committed output
compare = json.loads((LOG / "compare.json").read_text())
cls = {r["path"]: r["class"] for r in compare["changed"]}
rows = []
for rel in git("ls-files", "--", f"{REL}/results").splitlines():
    rel_pkt = rel[len(REL) + 1:]
    committed = git("show", f"HEAD:{rel}", binary=True)
    rerun_p = PKT / rel_pkt
    rows.append(dict(path=rel_pkt, committed_blob=git("rev-parse", f"HEAD:{rel}").strip(),
                     committed_bytes=len(committed), committed_sha256=hashlib.sha256(committed).hexdigest(),
                     rerun_bytes=rerun_p.stat().st_size, rerun_sha256=sha(rerun_p),
                     **{"class": cls.get(rel_pkt, "IDENTICAL")}))
    if rel_pkt in cls:  # changed: keep both byte streams side by side
        put(OUT / "original" / rel_pkt, committed)
        put(OUT / "reproduced" / rel_pkt, rerun_p.read_bytes())
summary = {k: sum(1 for r in rows if r["class"] == k) for k in ("IDENTICAL", "TIMING-ONLY", "DIFFERENT")}
(OUT / "comparison").mkdir()
(OUT / "comparison/hashes.json").write_text(json.dumps(dict(
    main=MAIN, committed_output_files=len(rows), summary=summary, files=rows,
    raw=compare["raw"]), indent=1) + "\n")
for name in ("COMPARE.md", "compare.json", "ledger.json"):
    src = LOG / name if (LOG / name).exists() else LOG.parent / "package" / name
    shutil.copy(src, OUT / "comparison" / name)
import importlib.util as _iu
_spec = _iu.spec_from_file_location("cmp", str(SCRIPTS / "compare.py"))
_cmp = _iu.module_from_spec(_spec)
os.environ.setdefault("ZF_PKT", str(PKT))
_spec.loader.exec_module(_cmp)
(OUT / "comparison/timing_mask_rules.txt").write_text(
    "Lines are compared after these regex substitutions (see scripts/compare.py TIMING):\n" +
    "\n".join(f"  {rx.pattern!r} -> {rep!r}" for rx, rep in _cmp.TIMING) + "\n")

# 2. reproduced extras: uncommitted event logs and the independent check
for p in sorted((PKT / "results/dbn").glob("track_log_dt*.txt")):
    put(OUT / "reproduced/uncommitted" / p.name, p.read_bytes())
shutil.copy(LOG / "indep_check.json", OUT / "reproduced/indep_check.json")

# 3. the environment / version / script-version experiments on indep_ht
exp = OUT / "experiments"
exp.mkdir()
committed = {k: [l for l in git("show", f"HEAD:{REL}/results/dbn/indep_ht_{k}.out").splitlines() if not l.startswith("time")]
             for k in ("c910", "c3435")}
rerun = {k: [l for l in (PKT / f"results/dbn/indep_ht_{k}.out").read_text().splitlines() if not l.startswith("time")]
         for k in ("c910", "c3435")}
index = []
for p in sorted(LOG.glob("*_c910.out")) + sorted(LOG.glob("*_c3435.out")):
    k = "c910" if p.name.endswith("c910.out") else "c3435"
    lines = [l for l in p.read_text().splitlines() if not l.startswith("time")]
    verdict = ("equals committed" if lines == committed[k] else
               "equals pinned rerun" if lines == rerun[k] else "differs from both")
    shutil.copy(p, exp / p.name)
    index.append(dict(file=p.name, collision=k, verdict_modulo_time=verdict, sha256=sha(p)))
for p in sorted(LOG.glob("*_env.txt")) + sorted(LOG.glob("gmpy_backend.txt")):
    shutil.copy(p, exp / p.name)
(exp / "INDEX.json").write_text(json.dumps(index, indent=1) + "\n")

# 4. raw run logs (every job's stdout/stderr, lane logs)
logs = OUT / "logs"
logs.mkdir()
for p in sorted(LOG.iterdir()):
    if p.suffix in (".stdout", ".stderr") or p.name.startswith("lane_") or p.name in ("START",) or p.suffix == ".jsonl":
        shutil.copy(p, logs / p.name)

# 5. scripts and the exact command list
sc = OUT / "scripts"
sc.mkdir()
for name in ("driver.py", "compare.py", "indep_check.py", "make_identity.py", "build_package.py"):
    shutil.copy(SCRIPTS / name, sc / name)
ledger = json.loads((OUT / "comparison/ledger.json").read_text())
cmds = ["#!/usr/bin/env bash", "# Exact commands of the rerun, from comparison/ledger.json. $PY = pinned interpreter",
        "# (Python 3.13.16 venv with numpy==2.5.3 mpmath==1.4.1); $WT = clean checkout of main " + MAIN + ".",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1", ""]
for j in ledger:
    argv = [("$PY" if i == 0 else a) for i, a in enumerate(j["argv"])]
    cwd = j["cwd"].replace(str(WT), "$WT")
    cmds.append(f"# [{j['lane']}] {j['name']}  start {j['start']}  wall {j['wall_s']}s  rc {j['rc']}")
    cmds.append(f"(cd {cwd} && " + " ".join(argv) + (f")  # stdout -> {j['mapped']}" if j["mapped"] else ")"))
(sc / "commands.sh").write_text("\n".join(cmds) + "\n")

# 6. environment
venv_py = Path(PY)
freeze = subprocess.run([str(venv_py.parent / "pip"), "freeze"], capture_output=True, text=True).stdout.split()
pyver = subprocess.run([PY, "-c", "import sys;print(sys.version.split()[0])"], capture_output=True, text=True).stdout.strip()
blas = subprocess.run([PY, "-c", "import numpy,json;numpy.show_config(mode='dicts') and None"], capture_output=True, text=True)
blas_cfg = subprocess.run([PY, "-c", "import numpy,json;print(json.dumps(numpy.show_config(mode='dicts')['Build Dependencies']['blas']))"],
                          capture_output=True, text=True).stdout.strip()
backend = subprocess.run([PY, "-c", "import mpmath.libmp as l;print(l.BACKEND)"], capture_output=True, text=True).stdout.strip()
env = dict(
    pinned_run=dict(python=pyver, packages=freeze, mpmath_backend=backend, numpy_blas=json.loads(blas_cfg or "{}"),
                    interpreter="/usr/bin/python3.13 venv", thread_env=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                    pillow_installed=False),
    host=dict(platform=platform.platform(), machine=platform.machine(), cpus=os.cpu_count(),
              cpu_model=next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None),
              mem_gib=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)),
    extra_test_environments=[p.read_text().strip() for p in sorted(LOG.glob("*_env.txt"))] + [(LOG / "gmpy_backend.txt").read_text().strip()],
    source=dict(repository="d6g8k5htny-coder/main", commit=MAIN, worktree="detached git worktree, manifest check ok before run",
                recovery_commit_for_script_versions=RECOVERY),
)
(OUT / "environment.json").write_text(json.dumps(env, indent=1) + "\n")

# 7. docs
for name in ("README.md", "HANDOFF.md"):
    shutil.copy(DOCS / name, OUT / name)

# 8. manifest
files = []
for p in sorted(OUT.rglob("*")):
    if p.is_file() and p.name != "MANIFEST.json":
        files.append(dict(path=p.relative_to(OUT).as_posix(), bytes=p.stat().st_size, sha256=sha(p)))
blobs = {f: git("rev-parse", f"HEAD:{REL}/{f}").strip() for f in
         ("README.md", "MANIFEST.json", "CLAIMS.md", "requirements.txt", "dbn/indep_ht.py",
          "results/dbn/indep_ht_c910.out", "results/dbn/indep_ht_c3435.out")}
manifest = dict(
    schema=1, object="ZETA-FOLD-SLICE-C-RERUN-20261019", status="review-record", scientific_effect="NONE",
    review_status="REVIEW_REQUIRED",
    sources=dict(main=MAIN, packet_blobs=blobs,
                 recovery_branch="claude/recovery-S-20261013", recovery_commit=RECOVERY,
                 recovery_indep_ht_blobs={"review-math": "c561d95bcf753e2b6fac0621b29b8604b77cffb4",
                                          "apply-baseline": "c152559bcfb5d19e488876aa17550cada1328e5e"}),
    generated_by="scripts/build_package.py", total_bytes=sum(f["bytes"] for f in files), files=files)
(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
print(f"{len(files)} files, {manifest['total_bytes']} bytes -> {OUT}")
print("summary", summary)
