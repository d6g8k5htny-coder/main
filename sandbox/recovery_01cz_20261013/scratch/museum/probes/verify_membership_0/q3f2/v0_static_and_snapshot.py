"""V0 (static + snapshot facts, read-only).
(a) In every in-scope file at cd32303, is there ANY code that enumerates incoming/ or uses MAIN_COMMIT?
(b) The packet list emitted by packet_descriptors() and the generator's summary line, read from the AST.
(c) git ls-tree of incoming/ at the declared pin 71400b94 and at a12c178c (the #150 main merge),
    and whether every main-repository pinned source is byte-identical (same blob id) at a12c178c."""
import ast, importlib.util, json, os, re, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; CD = H.parents[2] / "cd32303"
env = dict(os.environ, GIT_OPTIONAL_LOCKS="0"); G = ["git", "-C", "/home/user/main"]
files = ["tools/museum_data.py", "tools/museum_check.py", "docs/site/museum.mjs", "docs/site/geometry.mjs",
         "tests/test_museum_data.py", "tests/test_museum_frontend.mjs", "tests/test_museum_geometry.mjs"]
pat = re.compile(r"ls-tree|iterdir|listdir|scandir|os\.walk|\bglob\b|\.glob\(|readdir|MAIN_COMMIT")
print("(a) enumeration / MAIN_COMMIT hits in the 7 in-scope files:")
for f in files:
    for n, line in enumerate((CD / f).read_text().splitlines(), 1):
        if pat.search(line): print(f"    {f}:{n}: {line.strip()[:110]}")
src = (CD / "tools/museum_data.py").read_text(); tree = ast.parse(src)
uses = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "MAIN_COMMIT" and isinstance(n.ctx, ast.Load)]
print("    MAIN_COMMIT load-uses in museum_data.py (AST):", uses)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "packet_descriptors")
ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return)][0]
print(f"(b) packet_descriptors() lines {fn.lineno}-{fn.end_lineno}: docstring={ast.get_docstring(fn)!r}; "
      f"returns {type(ret.value).__name__} literal with {len(ret.value.elts)} element(s); args={len(fn.args.args)}")
pr = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"]
for p in pr: print(f"    print at line {p.lineno}: arg is {type(p.args[0]).__name__} -> {ast.unparse(p.args[0])[:120]}")
spec = importlib.util.spec_from_file_location("md", CD / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
pin = md.STATUS["commit"]; new = "a12c178c0f857a130cf434e9efd44233a038195b"
def lstree(c): return subprocess.run(G + ["ls-tree", "-d", "--name-only", c, "incoming/"], capture_output=True, text=True, env=env, check=True).stdout.split()
print("(c) declared main pin (STATUS/packet commit):", pin)
print("    ls-tree -d", pin[:8], "incoming/ :", lstree(pin))
print("    museum.json packets          :", [p["id"] + "@" + p["result"]["commit"][:8] for p in json.loads((CD / "docs/site/museum.json").read_text())["packets"]])
print("    ls-tree -d", new[:8], "incoming/ :", lstree(new))
print("    a12c178c subject:", subprocess.run(G + ["log", "-1", "--format=%s | %ad", new], capture_output=True, text=True, env=env).stdout.strip())
for path, ident in md.SOURCES.items():
    if ident["repository"].endswith("/main"):
        out = subprocess.run(G + ["ls-tree", new, "--", path], capture_output=True, text=True, env=env).stdout.split()
        print(f"    main source {path}: blob@a12c178c == pinned blob: {bool(out) and out[2] == ident['blob']}")
