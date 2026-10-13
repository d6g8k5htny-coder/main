# Verifier 2 of Q1-6. Read-only against the cd32303 bytes; no network (urlopen stubbed).
# For each HYPOTHETICAL successor STATUS (labelled as such in the row text), re-pin STATUS.md
# (and the main commit) exactly as a maintainer would edit SOURCES, then:
#   (1) run tools/museum_data.py project() at cd32303,
#   (2) run the cd32303 unittest suite tests/test_museum_data.py (which reads live STATUS.md) with
#       the successor as live STATUS and the re-pinned module, to see whether CI catches it,
#   (3) write a manifest for the frontend half (v1_frontend.mjs).
import hashlib, importlib.util, io, json, pathlib, sys, unittest
sys.dont_write_bytecode = True
S = pathlib.Path("<SCRATCH>/museum")
T = S / "cd32303"; H = S / "probes/verify_substitution_1"
INDEX = (S / "probes/substitution/fixture/Math-/PROOF_INDEX.md").read_bytes()
STATUS = (T / "STATUS.md").read_bytes()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m

def blocked(*a, **k): raise RuntimeError("NETWORK BLOCKED")
NEW = "5" * 40
def repin(m, new):
    for ident in m.SOURCES.values():
        if ident["repository"] == "d6g8k5htny-coder/main": ident["commit"] = NEW
    m.SOURCES["STATUS.md"].update(bytes=len(new), sha256=hashlib.sha256(new).hexdigest(),
        blob=hashlib.sha1(b"blob %d\0" % len(new) + new).hexdigest())
    m.STATUS = m.SOURCES["STATUS.md"]

base = load("md_base", T / "tools/museum_data.py"); base.urlopen = blocked
assert hashlib.sha256(INDEX).hexdigest() == base.INDEX["sha256"] and hashlib.sha256(STATUS).hexdigest() == base.STATUS["sha256"]
print("pins: PROOF_INDEX and STATUS bytes equal the cd32303 pinned identities")

lines = STATUS.decode().splitlines()
d1 = next(l for l in lines if l.startswith("| **D1 —")); d5 = next(l for l in lines if l.startswith("| **D5 — pin"))
sard = next(l for l in lines if l.startswith("| **SARD-G"))
A = ("| **D5 — shrinking multiple-witness collision** | HYPOTHETICAL successor row: the shrinking-separation "
     "factorial-moment/collision estimate is not supplied. | [Math proof index](https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md) |")
B = ("| **D5 — inner microdisk collar (HYPOTHETICAL split)** | HYPOTHETICAL successor row: the summed pin-neighborhood "
     "bound has moved to its own record; the inner microdisk collar estimate is a separate open obligation with no review yet. "
     "| [Math issue #999 (hypothetical)](https://github.com/d6g8k5htny-coder/Math-/issues/999) |")
variants = {
    "A_finding_row":        STATUS.decode().replace(d5, A),
    "B_row_mentions_inner_microdisk": STATUS.decode().replace(d5, B),
    "C_swap_D5_SARD":       STATUS.decode().replace(d5, "@@").replace(sard, d5).replace("@@", sard),
}
for name, txt in variants.items():
    new = txt.encode(); assert new != STATUS
    (H / f"{name}_STATUS.md").write_bytes(new)
    m = load("md_" + name, T / "tools/museum_data.py"); m.urlopen = blocked; repin(m, new)
    try:
        claims = m.project(INDEX, new); c = claims[12]
        print(f"[{name}] builder project(): ACCEPTED -> claims[12] id={c['id']!r} title={c['title']!r}")
        print(f"    proof={c['proof']['path']} availability={c['proof'].get('availability')!r}")
        print(f"    review={c['review'].get('review_url') or c['review']['path']}")
        print(f"    claims[13] id={claims[13]['id']!r} title={claims[13]['title']!r} review={claims[13]['review']['path']}")
        man = json.loads((T / "docs/site/museum.json").read_text())
        man.update(claims=claims, status_source=m.descriptor("STATUS.md"), packets=m.packet_descriptors())
        (H / f"{name}_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    except Exception as e:
        print(f"[{name}] builder project(): REFUSED {type(e).__name__}: {e}")
    # (2) the CI unit tests at cd32303, with this successor as live STATUS and the re-pinned module
    tm = load("tm_" + name, T / "tests/test_museum_data.py")
    tm.museum.urlopen = blocked; repin(tm.museum, new)
    orig = tm.MuseumDataTests.setUp
    tm.MuseumDataTests.setUp = lambda self, new=new: setattr(self, "status", new)
    names = ["test_exact_fourteen_projections_and_distinct_d5_scopes", "test_status_promotion_refused",
             "test_check_rejects_quote_identity_and_class_tampering", "test_review_pointers_do_not_claim_comment_byte_custody"]
    suite = unittest.TestSuite(tm.MuseumDataTests(n) for n in names)
    r = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    bad = {t.id().split(".")[-1]: tb.strip().splitlines()[-1] for t, tb in r.failures + r.errors}
    for n in names: print(f"    unittest {n}: {'FAIL ' + bad[n] if n in bad else 'ok'}")
