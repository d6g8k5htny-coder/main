# V2a: dump the exact claims the cd32303 builder emits for the accepted, CI-unit-test-passing
# substitutions (A: d5-two-scale review := proof; P: p15 review := LANDING_CLAIMS.json), so the
# frontend probe consumes builder bytes rather than hand-built objects.
import copy, json, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_1")
from common import museum, INDEX, STATUS, HERE
out = {}
for key, i, review in [("A", 5, "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md"),
                       ("P", 9, "claims/LANDING_CLAIMS.json"),
                       ("ctl", 5, "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md")]:
    specs = copy.deepcopy(museum.CLAIM_SPECS); specs[i][2] = review
    saved = museum.CLAIM_SPECS; museum.CLAIM_SPECS = specs
    try:
        out[key] = museum.project(INDEX, STATUS)[i]
        print(key, "builder emitted claim", out[key]["id"], "review.path =", out[key]["review"]["path"],
              "| review dict == proof dict:", out[key]["review"] == out[key]["proof"])
    except ValueError as e:
        print(key, "builder REFUSED:", e)
        # for the frontend control, hand the frontend the descriptor the builder would have made
        museum.CLAIM_SPECS = saved
        out[key] = dict(museum.project(INDEX, STATUS)[i], review=museum.descriptor(review))
    finally:
        museum.CLAIM_SPECS = saved
(HERE / "v2_builder_claims.json").write_text(json.dumps(out, indent=1))
