# V3: the D5-pin case via a one-entry builder edit. open_specs is a local literal inside project(),
# so this probe copies cd32303 tools/museum_data.py into the probe dir, changes only the d5-pin
# review_target from the D5_OPEN_REVIEW https URL to "PROOF_INDEX.md", and asks: does the builder
# bind open-row reviews at all, and does the CI unit suite notice?
import difflib, io, json, pathlib, sys, unittest
sys.path.insert(0, "<SCRATCH>/museum/probes/verify_substitution_1")
from common import INDEX, STATUS, TREE, HERE, load
orig = (TREE / "tools/museum_data.py").read_text()
old = '"https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md"),'
assert orig.count(old) == 1
edited = orig.replace(old, '"PROOF_INDEX.md"),')
(HERE / "museum_data_open_edit.py").write_text(edited)
print("".join(l for l in difflib.unified_diff(orig.splitlines(True), edited.splitlines(True), "cd32303", "edited", n=0) if l[0] in "+-" and not l.startswith(("+++", "---"))), end="")
m = load("museum_data_open_edit", HERE / "museum_data_open_edit.py"); m.urlopen = None
c = m.project(INDEX, STATUS)[12]
print("project ACCEPTS:", c["id"], "review.path =", c["review"]["path"], "pointer_only =", c["review"].get("pointer_only", False))
tests = load("test_museum_data_cd32303_v3", TREE / "tests/test_museum_data.py"); tests.museum = m
names = ["test_exact_fourteen_projections_and_distinct_d5_scopes", "test_misdirected_review_pointer_refused",
         "test_replay_commands_bind_source_and_checkout", "test_review_pointers_do_not_claim_comment_byte_custody",
         "test_check_rejects_quote_identity_and_class_tampering"]
r = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite(tests.MuseumDataTests(n) for n in names))
print("unit suite:", "PASSES" if r.wasSuccessful() else [t.id() for t, _ in r.failures + r.errors])
(HERE / "v3_d5pin_claim.json").write_text(json.dumps(c, indent=1))
