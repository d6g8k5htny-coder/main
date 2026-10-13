"""Q2-F3 verifier 2 (independent of the finder's ci_sim.py).
1. Re-verify every fixture byte string against cd32303 museum_data.SOURCES (size, git blob, sha256)
   using cd32303's own verify_bytes.
2. Copy the cd32303 tree to sim/<variant>, apply ONE edit, regenerate museum.json with the generator
   itself (except for the hand-edit control), then run the cd32303 public-shop.yml step
   "Museum source projection and refusal controls" verbatim, the only change being
   --math-root <verified fixture> so museum_check.py reads pinned bytes from disk instead of the network.
No network, no git state changes; the extracted cd32303 bytes are never written."""
import base64, json, os, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CD = (HERE / "../../../cd32303").resolve()
FX = (HERE / "../fixture").resolve()
sys.path.insert(0, str(CD / "tools")); import museum_data as md
fxmap = json.loads((FX / "verified-inputs.json").read_text())
for path, ident in md.SOURCES.items():
    raw = (FX / ident["repository"].split("/")[-1] / ident["path"]).read_bytes()
    md.verify_bytes(raw, ident); assert base64.b64decode(fxmap[md.descriptor(path)["url"]]) == raw
print(f"fixture: {len(md.SOURCES)} sources re-verified against cd32303 SOURCES pins")
OLD = '"The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor."'
VARIANTS = {
  "control": ("gen", None),
  "notice_byte_frozen": ("gen", '"Byte-frozen review: the linked comment is sealed by the SHA-256 shown below."'),
  "notice_limitation_dropped": ("gen", '"The pinned source binds this review pointer."'),
  "notice_absent_pointer_only_1": ("gen2", None),
  "handedit_json_control": ("json", "Byte-frozen review: the linked comment is sealed by the SHA-256 shown below."),
}
steps = [
  ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"],
  ["python3", "-B", "tools/museum_check.py", "--math-root", str(FX / "Math-")],
  ["node", "--test", "tests/test_museum_geometry.mjs"],
]
env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"]); env.pop("MUSEUM_MATH_ROOT", None); env.pop("MUSEUM_FIXTURE", None)
for name in (sys.argv[1:] or VARIANTS):
    kind, new = VARIANTS[name]
    sim = HERE / "sim" / name
    if sim.exists(): shutil.rmtree(sim)
    shutil.copytree(CD, sim, ignore=shutil.ignore_patterns("__pycache__"))
    if kind == "gen":
        if new:
            gen = (sim / "tools/museum_data.py").read_text(); assert gen.count(OLD) == 1
            (sim / "tools/museum_data.py").write_text(gen.replace(OLD, new))
        r = subprocess.run(["python3", "-B", "tools/museum_data.py", "--math-root", str(FX / "Math-")], cwd=sim, env=env, capture_output=True, text=True)
        print(f"[{name}] regenerate exit={r.returncode} :: {(r.stdout+r.stderr).strip().splitlines()[-1]}")
    elif kind == "gen2":
        gen = (sim / "tools/museum_data.py").read_text()
        old2 = 'return descriptor(path, pointer_only=True, review_url=target,\n        review_notice=' + OLD + ')'
        assert gen.count(old2) == 1
        (sim / "tools/museum_data.py").write_text(gen.replace(old2, 'return descriptor(path, pointer_only=1, review_url=target)'))
        r = subprocess.run(["python3", "-B", "tools/museum_data.py", "--math-root", str(FX / "Math-")], cwd=sim, env=env, capture_output=True, text=True)
        print(f"[{name}] regenerate exit={r.returncode} :: {(r.stdout+r.stderr).strip().splitlines()[-1]}")
    else:
        j = sim / "docs/site/museum.json"; t = j.read_text()
        old = "The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor."
        j.write_text(t.replace(old, new)); print(f"[{name}] hand-edited museum.json ({t.count(old)} notices), generator untouched")
    changed = (sim / "docs/site/museum.json").read_bytes() != (CD / "docs/site/museum.json").read_bytes()
    print(f"[{name}] museum.json differs from cd32303: {changed}")
    for cmd in steps:
        r = subprocess.run(cmd, cwd=sim, env=env, capture_output=True, text=True)
        lines = (r.stdout + r.stderr).strip().splitlines()
        keep = [l for l in lines if l.startswith(("# pass", "# fail", "# skip", "OK", "FAILED", "Ran ", "Museum")) or "ValueError" in l or "not ok" in l][-5:]
        print(f"[{name}] exit={r.returncode} :: {' '.join(cmd[1:4]) if cmd[0]=='python3' else ' '.join(cmd)} :: {' | '.join(keep)}")
