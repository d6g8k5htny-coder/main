"""Q2-F4, verifier 2: is the guard gap reachable through the generator, i.e. past CI?
For each one-line generator edit, build a throwaway copy of the relevant cd32303 files (never the extracted
bytes themselves), regenerate museum.json with the edited museum_data.py, then run the cd32303 CI step
`python3 -B tools/museum_check.py --math-root <fixture>` (export check + node frontend test), and finally render
the regenerated manifest with the cd32303 museum.mjs. No network: Math-/main bytes come from the
re-verified fixture (v0). Usage: python3 -B v2_generator_ci_sim.py <cd32303> <verified-inputs.json>"""
import base64, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
CD = Path(sys.argv[1]).resolve(); FIX = Path(sys.argv[2]).resolve(); HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(CD / "tools")); import museum_data as md
fx = json.loads(FIX.read_text())
P15_REVIEW = "reviews/p15_full_price_nonauthor_20260926/REVIEW.md"
real_p15 = base64.b64decode(fx[md.descriptor(P15_REVIEW)["url"]])
revised_p15 = real_p15.replace(b"ACCEPT", b"AMEND")
OTHER = "1" * 40

def ident_block(path, ident):
    return json.dumps(ident, indent=4).replace("\n", "\n  ")

VARIANTS = {
  "G0-unmodified": [],
  "G1-d2-review-is-own-proof": [(
    '"https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206",\n    "D2"',
    '"frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",\n    "D2"')],
  "G2-d6-review-repinned-to-other-commit": [(
    '"commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",\n    "path": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",\n    "blob": "07db19f826763b8faa7414ba2e403d9c45bc0b6e",\n    "bytes": 15569,\n    "sha256": "691ea0db5c0f073326c9f7208c9804c6f5d0daf92ba7467529c9d57e08b15eb7"',
    f'"commit": "{OTHER}",\n    "path": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",\n    "blob": "{hashlib.sha1(b"blob %d" % len(revised_p15) + bytes([0]) + revised_p15).hexdigest()}",\n    "bytes": {len(revised_p15)},\n    "sha256": "{hashlib.sha256(revised_p15).hexdigest()}"')],
  "G3-d5pin-review-target-PROOF_INDEX": [(
    '"https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md")',
    '"PROOF_INDEX.md")')],
  "G4-d5-review-path-bare-REVIEW.md": [(
    '"reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md",\n    null,',
    '"REVIEW.md",\n    null,'), (
    'SOURCES["README.md"] =',
    'SOURCES["REVIEW.md"] = dict(SOURCES["reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"], path="REVIEW.md")\nSOURCES["README.md"] =')],
}
src = (CD / "tools/museum_data.py").read_text()
for name, edits in VARIANTS.items():
    w = HERE / "sim" / name
    shutil.rmtree(w, ignore_errors=True)
    for rel in ["docs/site", "tests/test_museum_frontend.mjs", "tools/museum_check.py"]:
        (w / rel).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if (CD / rel).is_dir() else shutil.copy)(CD / rel, w / rel)
    text = src
    for old, new in edits:
        assert text.count(old) == 1, (name, old[:60]); text = text.replace(old, new)
    (w / "tools/museum_data.py").write_text(text)
    math = w / "_math"
    for path, ident in md.SOURCES.items():
        raw = base64.b64decode(fx[md.descriptor(path)["url"]])
        root = w if ident["repository"].endswith("/main") else math
        (root / path).parent.mkdir(parents=True, exist_ok=True); (root / path).write_bytes(raw)
    if name.startswith("G2"): (math / P15_REVIEW).write_bytes(revised_p15)
    if name.startswith("G4"): (math / "REVIEW.md").write_bytes(base64.b64decode(fx[md.descriptor("reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md")["url"]]))
    env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
    gen = subprocess.run([sys.executable, "-B", "tools/museum_data.py", "--main-root", str(w), "--math-root", str(math)], cwd=w, capture_output=True, text=True, env=env)
    print(f"[{name}] generator: exit {gen.returncode} | {(gen.stdout + gen.stderr).strip().splitlines()[-1]}")
    if gen.returncode: continue
    ci = subprocess.run([sys.executable, "-B", "tools/museum_check.py", "--math-root", str(math)], cwd=w, capture_output=True, text=True, env=env)
    tail = [l for l in (ci.stdout + ci.stderr).splitlines() if l.startswith(("# pass", "# fail", "# skipped", "Museum source export"))]
    print(f"[{name}] CI step museum_check.py: exit {ci.returncode} | {' ; '.join(tail)}")
    # fixture for rendering = exactly the bytes the edited generator pinned
    sys.path.insert(0, str(w / "tools"))
    mapping = {}
    for path, ident in json.loads(json.dumps(__import__("runpy").run_path(str(w / "tools/museum_data.py"))["SOURCES"])).items():
        root = w if ident["repository"].endswith("/main") else math
        base = ident["repository"] + "/" + ident["commit"] + "/" + "/".join(__import__("urllib.parse").parse.quote(p) for p in path.split("/"))
        mapping["https://raw.githubusercontent.com/" + base] = base64.b64encode((root / path).read_bytes()).decode()
    (w / "render-inputs.json").write_text(json.dumps(mapping))
    r = subprocess.run(["/opt/node22/bin/node", str(HERE / "v3_render_one.mjs"), str(w), str(w / "render-inputs.json")], capture_output=True, text=True)
    print("   " + (r.stdout + r.stderr).strip().replace("\n", "\n   "))
