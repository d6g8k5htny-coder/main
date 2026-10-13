"""Dry run: compose the #167, #160 and #173 declarative proposals on a scratch copy of GRAPH.json / SELECTOR_REGION.json,
validate with the hard gate, then run the three merged checkers against the executed state. Nothing is committed."""
import copy, importlib.util, json, pathlib, subprocess, sys, hashlib
WT = pathlib.Path("<SCRATCH>/wtdry")
def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, WT / rel); m = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]; spec.loader.exec_module(m); return m
hg = load("hard_gate", "frontiers/downstream_gate_20260925/hard_gate.py")
al = load("alignment_check", "reviews/register_alignment_20260930/alignment_check.py")
rc = load("residual_check", "reviews/c6_residual_closure_20260930/residual_check.py")
GRAPH = WT / "frontiers/downstream_gate_20260925/GRAPH.json"; SEL = WT / "frontiers/downstream_gate_20260925/SELECTOR_REGION.json"
g0 = json.loads(GRAPH.read_text()); sel0 = json.loads(SEL.read_text())
print("live nodes", len(g0["nodes"]), "edges", len(g0["edges"]))
print("SELECTOR keys:", list(sel0.keys()), "| regions:", sel0.get("regions"), "| cells sample:", json.dumps(sel0.get("cells", sel0))[:300])
rep0 = hg.closure_report(g0); holds0 = {h["node"] for h in rep0["hold_proposals"]}
print("baseline gate_ok", rep0["gate_ok"], "holds", sorted(holds0))
# ---------- step A: #167 ----------
spec167 = al.load_spec(WT)
g1 = al.apply(spec167, g0)
print("after #167: nodes", len(g1["nodes"]), "edges", len(g1["edges"]), "flipped:", [(t["node"], t["proposed"]["classification"]) for t in spec167["transitions"]])
# ---------- step B: #160 ----------
spec160 = json.loads((WT / "reviews/c6_witness_collision_reconciliation_20260929/PROPOSED_TRANSITIONS.json").read_text())
WIT = "math.rn-region.witness-collision"; RES = "math.rn-region.witness-collision.leading-mass-localization"
wit_entry = next(e for e in spec160["graph"] if e["node"] == WIT)
print("#160 WITNESS entry keys:", list(wit_entry.keys())); print("   ", json.dumps({k: v for k, v in wit_entry.items() if k not in ("required", "supporting")})[:700])
g2 = copy.deepcopy(g1)
for n in spec160["proposed_graph_nodes"]:                       # steps 1-3: residual (OPEN_ACTIVE), supersession, aggregate, components
    if n["id"] in g2["nodes"]:
        print("  #160 node already live (kept):", n["id"]); continue
    g2["nodes"][n["id"]] = {k: v for k, v in n.items() if k != "id"}
have = {(e["from"], e["to"], e["relation"]) for e in g2["edges"]}
for e in spec160["proposed_graph_edges"]:                        # steps 2-4 edges (including the old node's, step 4)
    key = (e["from"], e["to"], e["relation"])
    if key not in have:
        g2["edges"].append({"from": e["from"], "to": e["to"], "required": e["required"], "relation": e["relation"]}); have.add(key)
# step 4: move the old node (keep its fingerprint, as #173 requires; set classification and note)
w = g2["nodes"][WIT]; w_before = copy.deepcopy(w)
w["classification"] = "PROVED_REVIEWED"
w["notes"] = (w.get("notes", "") + " | Math-#160 section 4 discharge (count estimate at existential fixed-d compact-mark window scope); residual (Res) closed by Math-#173").strip(" |")
for k in ("scope", "explicit_limits", "review_disposition", "review_sources", "reading_rule", "required_nodes"):
    if k in wit_entry and wit_entry[k]: w[k] = wit_entry[k]
# selector (#160 selector_region_proposal)
sel2 = copy.deepcopy(sel0); srp = spec160["selector_region_proposal"]
if isinstance(sel2.get("regions"), list):
    for r in srp.get("regions", []):
        if r not in sel2["regions"]: sel2["regions"].append(r)
if isinstance(sel2.get("cells"), dict) and isinstance(srp.get("cells"), dict):
    for k, v in srp["cells"].items(): sel2["cells"][k] = v
print("after #160: nodes", len(g2["nodes"]), "edges", len(g2["edges"]))
# ---------- step C: #173 ----------
spec173 = rc.load_spec(WT)
old173, g3 = rc.build(spec173, g2, "final")                      # RES pre-exists OPEN_ACTIVE -> promoted; palm node reused
have = {(e["from"], e["to"], e["relation"]) for e in g3["edges"]}
for e in spec173["proposed_graph_edges"]:                        # the deferred old-node edges now apply (both presuppositions hold)
    if e.get("deferred_with"):
        key = (e["from"], e["to"], e["relation"])
        if key not in have:
            g3["edges"].append({"from": e["from"], "to": e["to"], "required": e["required"], "relation": e["relation"]}); have.add(key)
print("after #173: nodes", len(g3["nodes"]), "edges", len(g3["edges"]))
# ---------- validate ----------
try:
    hg.validate_graph_fail_closed(g3); print("validate_graph_fail_closed: OK")
except Exception as ex:
    print("validate_graph_fail_closed FAILED:", repr(ex)[:400]); sys.exit(1)
rep3 = hg.closure_report(g3); holds3 = {h["node"]: h.get("unsatisfied_required") for h in rep3["hold_proposals"]}
print("executed gate_ok", rep3["gate_ok"], "illegal_controlling", rep3.get("illegal_controlling"))
print("new holds:", {k: v for k, v in holds3.items() if k not in holds0}, "| cleared holds:", sorted(holds0 - set(holds3)))
for nid in (WIT, RES, "math.c6-witness-collision-factorial-moment", "regional.shrinking-regions.analytic-route") + tuple(t["node"] for t in spec167["transitions"]):
    n = g3["nodes"].get(nid, {}); print("  %-58s %-24s held=%s" % (nid, n.get("classification"), nid in holds3))
snap_old, snap_new = rc.snapshot(WT, g0), rc.snapshot(WT, g3)
ri = hg.reverse_impact_between(g0, g3, old_sources=snap_old, new_sources=snap_new)
print("reverse impact: changed", len(ri["changed_nodes"]), "impacted", len(ri["impacted"]), "snapshots", ri["source_snapshots_supplied"])
dc = hg.d4_region_complement(g3) if hasattr(hg, "d4_region_complement") else None
if dc: print("d4_region_complement:", json.dumps(dc)[:600])
# ---------- write executed state into the scratch worktree and run the three checkers ----------
GRAPH.write_text(json.dumps(g3, indent=2, ensure_ascii=False) + "\n"); SEL.write_text(json.dumps(sel2, indent=2, ensure_ascii=False) + "\n")
for chk in ("reviews/register_alignment_20260930/alignment_check.py", "reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py", "reviews/c6_residual_closure_20260930/residual_check.py"):
    r = subprocess.run([sys.executable, "-B", "-S", chk], cwd=WT, capture_output=True, text=True, timeout=600)
    try: out = json.loads(r.stdout); checks = out.get("checks")
    except Exception: checks = None
    print("\n==", chk.split("/")[1], "rc", r.returncode, "checks:", checks)
    if r.returncode and r.stderr: print("   stderr:", r.stderr.strip()[-600:])
