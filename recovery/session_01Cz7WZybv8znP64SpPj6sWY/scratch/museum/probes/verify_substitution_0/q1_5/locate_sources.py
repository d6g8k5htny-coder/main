# Locate byte-identical copies of the pinned Math- sources in the local shallow clone (read-only).
import ast, hashlib, json, pathlib, subprocess, sys, re
src = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r"SOURCES = json.loads\(r'''(.*?)'''\)", src, re.S)
sources = json.loads(m.group(1))
extra = re.search(r'SOURCES\["README.md"\] = (\{.*?\})\n', src).group(1)
sources["README.md"] = json.loads(extra)
math = pathlib.Path(sys.argv[2]); out = pathlib.Path(sys.argv[3])
for path, ident in sources.items():
    if not ident["repository"].endswith("Math-"): continue
    raw = None
    r = subprocess.run(["git", "cat-file", "-p", ident["blob"]], cwd=math, capture_output=True)
    if r.returncode == 0: raw = r.stdout; how = "blob-object"
    elif (math / path).exists(): raw = (math / path).read_bytes(); how = "worktree"
    ok = raw is not None and hashlib.sha256(raw).hexdigest() == ident["sha256"] and len(raw) == ident["bytes"]
    if ok:
        (out / "Math-" / path).parent.mkdir(parents=True, exist_ok=True); (out / "Math-" / path).write_bytes(raw)
    print(("OK  " if ok else "MISS"), how if raw is not None else "-", path)
