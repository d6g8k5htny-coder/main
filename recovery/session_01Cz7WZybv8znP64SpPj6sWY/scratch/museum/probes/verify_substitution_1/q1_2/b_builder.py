# Verifier-2 for Q1-2 (builder half). Loads the cd32303 builder by path (network blocked), uses the
# PROOF_INDEX bytes embedded in cd32303 tests/test_museum_data.py and cd32303 STATUS.md (both checked
# against the builder's own pins by common.py). Mutates only in-memory copies of SOURCES.
import copy, hashlib, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_1")
import common
m = common.museum
RP = "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"
PP = "frontiers/rn_annulus_bridge_20260925/PROOF.md"
real = (common.FIX / RP).read_bytes(); m.verify_bytes(real, m.SOURCES[RP])
alt = real.replace(b"| R7 crossover, absorption, and full cover (17)\xe2\x80\x93(20), (2) | **ACCEPT** |",
                   b"| R7 crossover, absorption, and full cover (17)\xe2\x80\x93(20), (2) | **AMEND** |")
print("alt differs:", alt != real, "| alt has AMEND verdict row:", b"| **AMEND** |" in alt)
def ident(repo, commit, path, raw):
    return {"repository": repo, "commit": commit, "path": path, "bytes": len(raw),
            "blob": hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest(), "sha256": hashlib.sha256(raw).hexdigest()}
saved = copy.deepcopy(m.SOURCES)
print("baseline: every Math- SOURCES pin at INDEX commit:",
      all(v["commit"] == m.INDEX["commit"] for v in saved.values() if v["repository"].endswith("/Math-")))
for label, repo, commit in (("other Math- commit", "d6g8k5htny-coder/Math-", "1" * 40),
                            ("main repository", "d6g8k5htny-coder/main", "2" * 40),
                            ("query- repository", "d6g8k5htny-coder/query-", "3" * 40)):
    m.SOURCES.clear(); m.SOURCES.update(copy.deepcopy(saved)); m.SOURCES[RP] = ident(repo, commit, RP, alt)
    try:
        c = m.project(common.INDEX, common.STATUS)[3]
        pj = f"project() ACCEPTS: {c['id']} class={c['class']} review={c['review']['repository']}@{c['review']['commit'][:8]} index={m.INDEX['repository']}@{m.INDEX['commit'][:8]}"
    except ValueError as e:
        pj = f"project() REFUSES: {e}"
    try:
        m.verify_bytes(alt, m.SOURCES[RP]); vb = "verify_bytes (read_source/build path) ACCEPTS alt bytes"
    except ValueError as e:
        vb = f"verify_bytes (read_source/build path) REFUSES: {e}"
    print(f"[{label}] {pj}\n    {vb}")
m.SOURCES.clear(); m.SOURCES.update(copy.deepcopy(saved))
m.SOURCES[PP] = dict(saved[PP], commit="4" * 40)
c = m.project(common.INDEX, common.STATUS)[3]
print(f"[contrast] builder project() also accepts PROOF at {c['proof']['commit'][:8]} (no proof binding in the builder either)")
m.SOURCES.clear(); m.SOURCES.update(saved)
