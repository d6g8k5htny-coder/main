"""V1: cd32303 museum_data.read_source on a (commit, path) whose path is NOT in that commit.
main_root = ./wt_base (cd32303 tree + discovered .git with all main objects, as in CI). No GIT_DIR.
Controls: (a) declared pin 71400b94 (member); (b) bd9fc9d6 (real main commit, parent of the packet
landing commit d10c2fb, no incoming/); (c) nonexistent commit; (d) bd9fc9d6 with the working-tree
file removed -> shows the accepted bytes came from the working tree, not from the declared commit."""
import importlib.util, os, shutil, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; CD = H.parents[1] / "cd32303"
os.environ.pop("GIT_DIR", None); os.environ["GIT_OPTIONAL_LOCKS"] = "0"
spec = importlib.util.spec_from_file_location("md", CD / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
wt = H / "wt_base"; path = "incoming/side24-identity-replay-20260926/RESULT.md"; pin = md.SOURCES[path]
def member(c): return subprocess.run(["git", "cat-file", "-e", f"{c}:{path}"], cwd=wt, capture_output=True).returncode == 0
def run(label, commit, root):
    ident = dict(pin, commit=commit)
    try: raw = md.read_source(ident, root, None); r = f"ACCEPTED {len(raw)} bytes, sha256 ok"
    except Exception as e: r = f"REFUSED {type(e).__name__}: {e}"
    print(f"{label}: commit={commit[:12]} path-in-commit-tree={member(commit)} -> read_source {r}")
run("(a) declared pin", pin["commit"], wt)
run("(b) bd9fc9d6", "bd9fc9d603fd70946de364afcc3c902f09a5bf29", wt)
run("(c) no such commit", "f" * 40, wt)
wt2 = H / "wt_nofile"; shutil.rmtree(wt2, ignore_errors=True); shutil.copytree(wt, wt2, symlinks=True)
(wt2 / path).unlink()
run("(d) bd9fc9d6, working-tree file removed", "bd9fc9d603fd70946de364afcc3c902f09a5bf29", wt2)
run("(e) declared pin, working-tree file removed", pin["commit"], wt2)
