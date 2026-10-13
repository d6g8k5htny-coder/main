# Q1/P1: the builder binds a claim's review only by "](review_path)" occurring somewhere in the
# claim's own bullet. The bullet also links the claim's PROOF (and, for D2, an open issue), so a
# CLAIM_SPECS entry whose review is the proof itself, or the open issue, is accepted.
import copy, json, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/substitution")
from builder import museum, INDEX, STATUS, HERE
cases = {
  "control: d5-all-height review := pr16 review (not in bullet)": (3, "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md"),
  "A: d5-two-scale review := its own proof": (5, "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md"),
  "B: d2 review := its own proof": (0, "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md"),
  "C: d2 review := bare main#67 ('remains open for other bundled fronts')": (0, "https://github.com/d6g8k5htny-coder/main/issues/67"),
  "D: d5-fixed-transverse review := its proof under reviews/": (7, "reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md"),
}
out = {}
for name, (i, review) in cases.items():
    specs = copy.deepcopy(museum.CLAIM_SPECS); specs[i][2] = review
    saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
    try:
        claims = museum.project(INDEX, STATUS)
        c = claims[i]; r = c["review"]
        same = {k: r.get(k) for k in ("path","sha256")} == {k: c["proof"].get(k) for k in ("path","sha256")}
        print(f"ACCEPTED  {name}\n          -> {c['id']} review={r.get('review_url') or r['path']} pointer_only={r.get('pointer_only', False)} review_is_proof={same}")
        out[name] = c
    except ValueError as e:
        print(f"REFUSED   {name}: {e}")
    finally:
        museum.CLAIM_SPECS = saved
json.dump(out, open(HERE / "p1_builder_claims.json", "w"), indent=1)
