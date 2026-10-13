# Apply one named textual patch to a COPY of cd32303 tools/museum_data.py (never the original).
# Every patch asserts its anchor occurs exactly once, so a silent no-op patch is impossible.
import hashlib, pathlib, sys

copy_root = pathlib.Path(sys.argv[1]); variant = sys.argv[2]
f = copy_root / "tools/museum_data.py"; src = f.read_text()

def sub(old, new):
    global src
    assert src.count(old) == 1, ("anchor count", src.count(old), old[:80])
    src = src.replace(old, new)

# Fixture artifact (all variants, including the baseline): README.md@d6628da is not available offline.
# Re-pin SOURCES["README.md"] to the locally available README bytes (Math- 10e1f191), which record
# the same three replay command lines. This touches only the D2/D4/D6 replay-source identity.
readme = (pathlib.Path(sys.argv[3])).read_bytes()
blob = hashlib.sha1(b"blob %d\0" % len(readme) + readme).hexdigest()
sub('"blob":"abe16871da7740fa28ae4dbb28fb31853a5868bd","bytes":6023,"sha256":"89d4c618ad4560373f8fba3c787522cea8f0191fd8e176f3e0741c6230390730"',
    f'"blob":"{blob}","bytes":{len(readme)},"sha256":"{hashlib.sha256(readme).hexdigest()}"')

if variant == "baseline":
    pass
elif variant == "A_two_scale_review_is_its_proof":
    sub('''    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",''',
        '''    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",''')
elif variant == "B_d2_review_is_its_proof":
    sub('''    "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",
    "https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206",''',
        '''    "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",
    "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",''')
elif variant == "C_d5open_review_is_index_no_pointer":
    sub('''        ("d5-pin-neighborhoods-open", "PROOF_INDEX.md",
         "https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md"),''',
        '''        ("d5-pin-neighborhoods-open", "PROOF_INDEX.md", "PROOF_INDEX.md"),''')
elif variant == "CONTROL_two_scale_review_is_other_claims_review":
    sub('''    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",''',
        '''    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md",''')
else:
    raise SystemExit("unknown variant")
f.write_text(src)
print("patched", variant)
