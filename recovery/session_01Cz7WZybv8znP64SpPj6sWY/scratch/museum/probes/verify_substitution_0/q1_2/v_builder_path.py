# Verifier probe for Q1-2 (builder path, end to end through what CI regenerates).
# Read-only against cd32303 tools/museum_data.py; no network (urlopen stubbed).
# Scenario: the single SOURCES pin for a review path is moved to another Math- commit whose bytes
# differ (synthetic stand-in). Question: does read_source/verify_bytes + project() accept it, and is
# the resulting manifest (= shipped museum.json with claims replaced, which is all build() changes)
# then accepted by the frontend? Writes manifests for v_frontend.mjs into ./out/.
import copy, importlib.util, json, pathlib, shutil, sys
sys.dont_write_bytecode = True
SCRATCH = pathlib.Path("<SCRATCH>/museum")
TREE = SCRATCH / "cd32303"; HERE = SCRATCH / "probes/verify_substitution_0/q1_2"
spec = importlib.util.spec_from_file_location("museum_data_cd32303", TREE / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
md.urlopen = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NETWORK BLOCKED"))
import hashlib
INDEX = (HERE / "PROOF_INDEX.md").read_bytes(); STATUS = (TREE / "STATUS.md").read_bytes()
shipped = json.loads((TREE / "docs/site/museum.json").read_text())
assert md.project(INDEX, STATUS) == shipped["claims"], "baseline project() != shipped claims"
print("baseline: project(pinned INDEX, STATUS) == shipped museum.json claims: True")
out = HERE / "out"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
ALT = "a" * 40
cases = {  # claim index -> review path
    10: "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",      # d6-p15-full-price (replay source = README.md)
    5: "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",  # d5-two-scale (no replay)
    3: "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md",   # d5-all-height-annulus (replay source = review)
}
for i, rpath in cases.items():
    real = (HERE / "fixture/Math-" / rpath).read_bytes()
    alt = real.replace(b"ACCEPT", b"AMEND", 1) + b"\n\nSYNTHETIC ALTERNATE VERSION (not the bytes PROOF_INDEX@d6628da cites).\n"
    assert alt != real
    saved = copy.deepcopy(md.SOURCES[rpath])
    md.SOURCES[rpath] = dict(saved, commit=ALT, blob=hashlib.sha1(b"blob %d\0" % len(alt) + alt).hexdigest(),
                             bytes=len(alt), sha256=hashlib.sha256(alt).hexdigest())
    # museum_check/read_source path: plain fixture folder (no git object) holding the alt bytes
    root = out / f"math_{i}"; (root / rpath).parent.mkdir(parents=True); (root / rpath).write_bytes(alt)
    try:
        md.read_source(md.SOURCES[rpath], None, root); rs = "accepted"
    except Exception as e:
        rs = f"refused: {e}"
    try:
        claims = md.project(INDEX, STATUS); c = claims[i]
        print(f"[{c['id']}] read_source(alt pin): {rs}; project(): ACCEPTED, review {c['review']['repository']}@{c['review']['commit'][:8]} "
              f"vs index {md.INDEX['commit'][:8]}; replay.source = {c['replay'].get('source', {}).get('path')}")
        m = dict(shipped, claims=claims)
        (out / f"manifest_{i}.json").write_text(json.dumps(m)); (out / f"alt_{i}.bin").write_bytes(alt)
    except ValueError as e:
        print(f"[{i}] project(): REFUSED {e}")
    md.SOURCES[rpath] = saved
