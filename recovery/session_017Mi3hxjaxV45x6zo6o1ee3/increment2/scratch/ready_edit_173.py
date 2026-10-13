import pathlib, sys, hashlib, subprocess, json, re
W = pathlib.Path(sys.argv[1]); D = W / "reviews/c6_residual_closure_20260930"
def patch(path, pairs):
    p = W / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, ("MISS", path, old[:70], s.count(old))
        s = s.replace(old, new)
    p.write_text(s); print("patched", path)
READY = "`reviews/register_execution_readiness_20260930`"
# ---- checker: the installed-state simulation (and its drift mutant) in every register state
patch("reviews/c6_residual_closure_20260930/residual_check.py", [
("""    if not installed:
        # the installed state, simulated (Codex 4140125558): the executed graph passes LIVE and re-building it is a no-op
        sim = build(spec, graph, "final")[1]
        if MUT == "installed-drift":
            sim["nodes"]["math.c6r-component.spectral-closure-proof"]["fingerprint"] = "0" * 64
        sim_ok, sim_st = live_ok(sim, sel, spec)
        ok &= sim_ok and sim_st["residual"] == "PROVED_REVIEWED" and sim_st["present"] == sorted(cids)
        ok &= sim_st["edges_present"] and not sim_st["mismatched"]
        o2, n2 = build(spec, sim, "final")
        ok &= o2 == n2
        builds["installed-simulated"] = (o2, n2)
    return bool(ok), builds
""", """    # the installed state, simulated (Codex 4140125558): the executed graph passes LIVE and re-building it is a no-op;
    # on the installed register the simulation is the live graph itself, so the drift mutant is rejected in either state
    sim = build(spec, graph, "final")[1]
    if MUT == "installed-drift":
        sim["nodes"]["math.c6r-component.spectral-closure-proof"]["fingerprint"] = "0" * 64
    sim_ok, sim_st = live_ok(sim, sel, spec)
    ok &= sim_ok and sim_st["residual"] == "PROVED_REVIEWED" and sim_st["present"] == sorted(cids)
    ok &= sim_st["edges_present"] and not sim_st["mismatched"]
    o2, n2 = build(spec, sim, "final")
    ok &= o2 == n2
    if not installed:
        builds["installed-simulated"] = (o2, n2)
    return bool(ok), builds
"""),
])
# ---- record text (v1.6)
patch("reviews/c6_residual_closure_20260930/RECONCILIATION.md", [
("""`8d48d02` re-verifies after Math-#160 merged (no content change); v1.5 at `02772ec` applies the direct-statement amendment""",
 """`8d48d02` re-verifies after Math-#160 merged (no content change); v1.6 at `dda8991` (register execution readiness,
""" + READY + """) keeps the installed-state simulation and its `installed-drift` mutant active on the installed
register too (before, the mutant was a no-op once the residual was live `PROVED_REVIEWED`, so the workflow's mutant rule
would have failed after execution), prints `register_state`, and the workflow compares the checker's output with
`RESULTS.json` in the baseline state and with `RESULTS_INSTALLED.json` in the installed state (the per-stage gate report
differs between the two by construction); no other content changes; v1.5 at `02772ec` applies the direct-statement amendment"""),
("""  The same test runs on the executed graph built from the proposal (installed state, simulated) in every run, so that
  execution cannot turn this check red (Codex 4140125558).""",
 """  The same test runs on the executed graph built from the proposal (installed state, simulated) in every run, so that
  execution cannot turn this check red (Codex 4140125558); on the installed register that simulation is the live graph
  itself (v1.6)."""),
("""review basis must fail `TRANSITIONS` and `REVIEWED`), `reviewed-text-drift` (a changed reviewed text must fail `REVIEWED`).""",
 """review basis must fail `TRANSITIONS` and `REVIEWED`), `reviewed-text-drift` (a changed reviewed text must fail `REVIEWED`).
The workflow runs the checker in the state the repository is in: it compares the output with `RESULTS.json` when the
checker reports `register_state: baseline` and with `RESULTS_INSTALLED.json` when it reports `installed`; the mutants
must be rejected in either state (the composition of Math-#167, #160 and #173 on a scratch copy, with every mutant, is
replayed by """ + READY + """)."""),
])
# ---- repin the record's own bytes (INVENTORY sha256 + blob; the record node's fingerprint)
data = (D / "RECONCILIATION.md").read_bytes()
sha = hashlib.sha256(data).hexdigest()
blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
old_sha = "b647a186d0cfd1662fc69e9d31dfd447dbb1b2f6ef4364bbd59c4956f18aed25"; old_blob = "3cfd17c5ef0356dbd3e8f98a2d08cb404ddd982c"
for name in ("residual_check.py", "PROPOSED_TRANSITIONS.json"):
    p = D / name; s = p.read_text()
    n1, n2 = s.count(old_sha), s.count(old_blob)
    s = s.replace(old_sha, sha).replace(old_blob, blob); p.write_text(s)
    print(name, "sha repinned x%d, blob x%d" % (n1, n2))
print("new sha", sha, "blob", blob)
# the record node's fingerprint must equal the file's sha (as the checker requires)
spec = json.loads((D / "PROPOSED_TRANSITIONS.json").read_text())
rec = next(c for c in spec["proposed_graph_nodes"] if c["id"] == "math.c6r-component.residual-closure-record")
assert rec["fingerprint"] == sha and rec["source"].endswith("RECONCILIATION.md"), rec
