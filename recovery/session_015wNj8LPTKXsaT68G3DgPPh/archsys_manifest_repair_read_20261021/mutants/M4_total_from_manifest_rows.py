"""Write or check MANIFEST.json for this packet (standard library only).

    python3 make_manifest.py            # rewrite MANIFEST.json
    python3 make_manifest.py --check    # exit 1 if any committed file differs from the manifest
    python3 make_manifest.py --raw-dir DIR   # also re-hash raw samples found in DIR

"Committed files" are all files under this directory except MANIFEST.json itself, the contents
of results/raw/ (other than its .gitignore) and Python bytecode caches. Raw samples are
regenerable and not committed; their entries record the SHA-256 and size from the exploratory
run (kept from the existing manifest unless --raw-dir is given). Regeneration is expected to be
byte-identical only with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (BLAS summation order; see
README section 8).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST.json"

RAW_SOURCES = {
    "results/raw/zeta_A.npz": "python3 stats/rs_zeta.py A 1e4 1e5",
    "results/raw/zeta_B.npz": "python3 stats/rs_zeta.py B 1e5 1e6",
    "results/raw/rs_cheb_coeffs.npz": "written on first import of stats/rs_zeta.py",
    "results/raw/gauss.npz": "python3 stats/gauss_proc.py 500 4",
    "results/raw/cue12.npz": "python3 stats/cue.py 12 86016 4",
    "results/raw/cue128.npz": "python3 stats/cue.py 128 8064 4",
    "results/raw/ecdf_samples.npz": "python3 stats/analyze.py",
    "results/raw/Z_grid_600_900.npz": "python3 stats/sample_Z.py 600 900 0.02",
    "results/raw/Z_grid_5000_5200.npz": "python3 stats/sample_Z.py 5000 5200 0.015",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def committed_files():
    out = []
    for p in sorted(HERE.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(HERE).as_posix()
        if rel == "MANIFEST.json" or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if rel.startswith("results/raw/") and rel != "results/raw/.gitignore":
            continue
        out.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--raw-dir", type=Path)
    args = ap.parse_args()
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    files = committed_files()
    if args.check:
        want = {}
        duplicates = set()
        for f in old.get("committed_files", []):
            if f["path"] in want:
                duplicates.add(f["path"])
            want[f["path"]] = f
        have = {f["path"]: f for f in files}
        bad = sorted(set(want) ^ set(have)) + sorted(k for k in want.keys() & have.keys() if want[k] != have[k])
        bad.extend("duplicate committed path: " + path for path in sorted(duplicates))
        total = old.get("committed_total_bytes")
        if type(total) is not int or total != sum(f["bytes"] for f in old.get("committed_files", [])):
            bad.append("committed_total_bytes")
        for k in bad:
            print("MISMATCH", k)
        print("ok" if not bad else f"{len(bad)} mismatches")
        return 1 if bad else 0
    raw = {r["path"]: r for r in old.get("raw_files", [])}
    if args.raw_dir:
        for rel in RAW_SOURCES:
            p = args.raw_dir / Path(rel).name
            if p.exists():
                raw[rel] = dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p))
    raw_list = []
    for rel, cmd in RAW_SOURCES.items():
        if rel in raw:
            raw_list.append(dict(path=rel, bytes=raw[rel]["bytes"], sha256=raw[rel]["sha256"], regenerate=cmd))
    data = dict(
        schema=1,
        object="ZETA-FOLD-EXPLORATION-20261012",
        status="exploratory",
        scientific_effect="NONE",
        sources=dict(main="7588716d347fc7100ead086ef1049bb290f875e0",
                     math="79cb314d5b9f6b8b410bfd0df200414314f6b8c7"),
        generated_by="make_manifest.py",
        note=("committed_files lists every file meant to be committed (this manifest excluded). "
              "raw_files are regenerable samples written to results/raw/ and not committed; "
              "their hashes are from the exploratory run; regeneration is expected to be byte-identical only with "
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 (README section 8)."),
        committed_total_bytes=sum(f["bytes"] for f in files),
        committed_files=files,
        raw_total_bytes=sum(r["bytes"] for r in raw_list),
        raw_files=raw_list,
    )
    MANIFEST.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(files)} committed files, {data['committed_total_bytes']} bytes; {len(raw_list)} raw entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
