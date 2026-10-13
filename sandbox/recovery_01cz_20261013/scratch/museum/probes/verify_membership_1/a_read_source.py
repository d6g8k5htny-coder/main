"""A: cd32303 museum_data.read_source(identity, main_root=<full-history checkout>, None) on the landed packet's
RESULT.md with only identity["commit"] varied. Reports whether <commit>:<path> exists, what read_source does,
and (control) what it does once the working-tree file is moved aside, which shows where accepted bytes came from."""
import importlib.util, os, subprocess, sys
from pathlib import Path
V = Path(__file__).resolve().parent; CD = V.parents[1] / "cd32303"; CK = V / "ck"
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
path = "incoming/side24-identity-replay-20260926/RESULT.md"; pin = md.SOURCES[path]
def git(*a): return subprocess.run(["git", "-C", str(CK), *a], capture_output=True, text=True)
def full(c): return git("rev-parse", "-q", "--verify", c + "^{commit}").stdout.strip() or c
cases = [("71400b94 (declared pin)", pin["commit"]),
         ("bd9fc9d6 (parent of packet landing commit d10c2fb)", "bd9fc9d603fd70946de364afcc3c902f09a5bf29"),
         ("4aa316f", "4aa316f"), ("0123...(nonexistent)", "0123456789abcdef0123456789abcdef01234567")]
def attempt(ident):
    try: md.read_source(ident, CK, None); return "ACCEPTED"
    except Exception as e: return "REFUSED " + type(e).__name__
wt = CK / path; aside = CK / (path + ".aside")
for label, c in cases:
    c = full(c); ident = dict(pin, commit=c)
    inside = git("cat-file", "-e", c + ":" + path).returncode == 0
    r1 = attempt(ident)
    os.rename(wt, aside)
    try: r2 = attempt(ident)
    finally: os.rename(aside, wt)
    print(f"{label}: {path} in tree of {c[:8]}: {inside} | read_source: {r1} | read_source with working-tree file moved aside: {r2}")
