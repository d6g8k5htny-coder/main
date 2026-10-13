# Q1/P2 (builder half): project() never compares a review descriptor's commit/repository with the
# PROOF_INDEX snapshot whose bullet names it. The relative link "](reviews/...)" in the index means
# "this path at the index's own commit", but any SOURCES pin for that path is accepted.
import copy, sys
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_0/q1_2")
from builder import museum, INDEX, STATUS
path = "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md"
saved = copy.deepcopy(museum.SOURCES[path])
for label, change in (("other Math- commit", {"commit": "1" * 40}), ("query- repository", {"repository": "d6g8k5htny-coder/query-", "commit": "2" * 40})):
    museum.SOURCES[path] = dict(saved, **change)
    try:
        c = museum.project(INDEX, STATUS)[3]
        print(f"ACCEPTED ({label}): {c['id']} review {c['review']['repository']}@{c['review']['commit'][:8]} "
              f"vs index {museum.INDEX['repository']}@{museum.INDEX['commit'][:8]}")
    except ValueError as e:
        print(f"REFUSED ({label}): {e}")
museum.SOURCES[path] = saved
# For contrast the builder has no proof-commit binding either:
ppath = "frontiers/rn_annulus_bridge_20260925/PROOF.md"; psaved = copy.deepcopy(museum.SOURCES[ppath])
museum.SOURCES[ppath] = dict(psaved, commit="3" * 40)
c = museum.project(INDEX, STATUS)[3]
print(f"builder also ACCEPTS proof at {c['proof']['commit'][:8]} (the frontend refuses that one, see p2_review_snapshot.mjs)")
