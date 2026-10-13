"""Q2-F3 verifier-1 probe. Independent of probes/pointers/ci_sim.py.
1. Re-verify every fixture file byte-for-byte (size, git blob sha1, sha256) against cd32303 tools/museum_data.SOURCES.
2. For each mutation: copy the cd32303 tree to sim/<name>, apply the edit, regenerate museum.json with the
   (possibly edited) generator when the mutation is a generator edit, then run the exact commands of the
   'Museum source projection and refusal controls' step of cd32303 .github/workflows/public-shop.yml,
   with --math-root pointed at the verified local fixture (no network: urlopen is never reached because a root is given).
Never writes into the extracted cd32303 tree."""
import importlib.util, os, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../../cd32303").resolve()
FIX = (HERE / "../../pointers/fixture").resolve()
MATH = FIX / "Math-"
spec = importlib.util.spec_from_file_location("md_cd", CD / "tools/museum_data.py"); md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
n = 0
for path, ident in md.SOURCES.items():
    sub = "main" if ident["repository"].endswith("/main") else "Math-"
    md.verify_bytes((FIX / sub / ident["path"]).read_bytes(), ident); n += 1
print(f"fixture re-verified against cd32303 SOURCES: {n} files")
NOTICE = 'review_notice="The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor."'
GUARD = "||!nonempty(source.review_notice)"
LABEL = "'Review pointer in the verified index'"
M = {
  "control": [],
  "G1_notice_byte_frozen": [("tools/museum_data.py", NOTICE, 'review_notice="Byte-frozen review: the linked comment is sealed by the SHA-256 shown below."')],
  "G2_notice_stripped_to_fixture_wording": [("tools/museum_data.py", NOTICE, 'review_notice="Pointer only."')],
  "G3_guard_removed": [("docs/site/museum.mjs", GUARD, "")],
  "G4_guard_removed_and_notice_empty": [("docs/site/museum.mjs", GUARD, ""), ("tools/museum_data.py", NOTICE, 'review_notice=""')],
  "G5_notice_empty_guard_intact": [("tools/museum_data.py", NOTICE, 'review_notice=""')],
  "G6_handedit_json_only": "json",
  "G8_replay_notice_claims_acceptance": [("tools/museum_data.py", '"notice": ("Finite algebra or implementation replay only; it is not theorem acceptance."', '"notice": ("Replay certifies theorem acceptance."')],
  "G7_hardcoded_label_relabelled": [("docs/site/museum.mjs", LABEL, "'Byte-verified review'")],
}
name = sys.argv[1]
sim = HERE / "sim" / name
if sim.exists(): shutil.rmtree(sim)
shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
regen = False
if M[name] == "json":
    p = sim / "docs/site/museum.json"; s = p.read_text(encoding="utf-8")
    old = "the linked review is not byte-frozen by this descriptor."; assert s.count(old) == 6
    p.write_text(s.replace(old, "the linked comment is sealed by the SHA-256 shown below.", 1), encoding="utf-8")
else:
    for rel, old, new in M[name]:
        p = sim / rel; s = p.read_text(encoding="utf-8"); assert s.count(old) == 1, (rel, old); p.write_text(s.replace(old, new), encoding="utf-8")
        regen = regen or rel.endswith(".py")
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ.get("PATH", "/usr/bin:/bin"))
steps = ([["python3", "-B", "tools/museum_data.py", "--math-root", str(MATH)]] if regen else []) + [
  ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
  ["python3", "-B", "tools/museum_check.py", "--math-root", str(MATH)],
  ["node", "--test", "tests/test_museum_geometry.mjs"],
]
ok = True
for cmd in steps:
    r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True)
    lines = (r.stdout + r.stderr).strip().splitlines()
    keep = [l.strip() for l in lines if l.strip().startswith(("# pass", "# fail", "OK", "FAILED", "Ran ", "Museum", "ValueError", "not ok"))][-5:]
    ok = ok and r.returncode == 0
    print(f"[{name}] exit={r.returncode} :: {' '.join(c for c in cmd if not c.startswith('/'))} :: {' | '.join(keep)}")
print(f"[{name}] MUSEUM CI STEP: {'GREEN' if ok else 'RED'}")
