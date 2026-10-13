"""P1 (unit): museum_data.read_source() accepts a (commit, path) pin whose path is NOT in that commit.

main_root = a plain copy of the cd32303 working tree; GIT_DIR points read-only at the main object
store via ./objstore.git (bare; objects/info/alternates -> /home/user/main/.git/objects). This is exactly a full CI checkout: objects for every commit, working
tree = PR head. read_source runs `git show <commit>:<path>`; on ANY failure it silently reads the
working-tree file instead, so only the bytes are checked, never membership of path in the declared commit.
"""
import os, shutil, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load, CD, HERE
os.environ.update(GIT_DIR=str(HERE / "objstore.git"), GIT_OPTIONAL_LOCKS="0")  # bare alternates store; cannot write to /home/user/main
md = load()
wt = HERE / "wt_plain"
if not wt.exists(): shutil.copytree(CD, wt)
path = "incoming/side24-identity-replay-20260926/RESULT.md"
pinned = md.SOURCES[path]
def present(commit):
    return subprocess.run(["git", "cat-file", "-e", f"{commit}:{path}"], capture_output=True).returncode == 0
def strict(identity, root):                 # what raw.githubusercontent.com/<repo>/<commit>/<path> does
    r = subprocess.run(["git", "show", identity["commit"] + ":" + identity["path"]], cwd=root, capture_output=True)
    if r.returncode: raise ValueError("not in declared commit: " + r.stderr.decode().strip())
    return md.verify_bytes(r.stdout, identity)
cases = [("71400b94 (declared snapshot, control)", pinned["commit"]),
         ("bd9fc9d6 (main, parent of the packet's landing commit d10c2fb)", "bd9fc9d603fd70946de364afcc3c902f09a5bf29"),
         ("4aa316f (merged side-branch commit of #150, packet never in its tree)", "4aa316f" ),
         ("ffff... (no such commit)", "f" * 40)]
for label, commit in cases:
    full = subprocess.run(["git", "rev-parse", "-q", "--verify", commit + "^{commit}"], capture_output=True, text=True).stdout.strip() or commit
    ident = dict(pinned, commit=full)
    try: md.read_source(ident, wt, None); got = "ACCEPTED (bytes returned)"
    except Exception as e: got = "REFUSED: " + str(e)
    try: strict(ident, wt); s = "accepted"
    except Exception as e: s = "REFUSED"
    print(f"{label}\n   path in commit tree: {present(full)} | museum_data.read_source: {got} | strict commit:path read: {s}")
