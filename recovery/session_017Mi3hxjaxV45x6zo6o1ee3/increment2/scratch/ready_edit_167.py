"""v1.4 edit of alignment_check.py: installed-state acceptance (exact replacements; fails loudly on a miss)."""
import pathlib, sys
P = pathlib.Path(sys.argv[1]) / "reviews/register_alignment_20260930/alignment_check.py"
s = P.read_text()
def sub1(old, new):
    global s
    assert s.count(old) == 1, ("MISS", old[:80], s.count(old))
    s = s.replace(old, new)

sub1('''Checks: IDENTITIES, VERDICTS, BASELINE, TRANSITIONS, GATE, PROPAGATION, INTERFACE, NEGATIVES, D4_DUPLICATE. No mathematics
is re-proved; the proposed graph is replayed through the downstream hard gate's own validators, and two-commit source
mutations through its reverse-impact rule. Nothing is written to the repository.
"""''', '''Checks: IDENTITIES, VERDICTS, BASELINE, TRANSITIONS, GATE, PROPAGATION, INTERFACE, NEGATIVES, D4_DUPLICATE. No mathematics
is re-proved; the proposed graph is replayed through the downstream hard gate's own validators, and two-commit source
mutations through its reverse-impact rule. Nothing is written to the repository.

Register states (v1.4). The live register is accepted in exactly two states: 'baseline' (none of the proposal is live: the
eight nodes carry their recorded current classification and fingerprint, no component node, no proposed edge) and
'installed' (all of it is live exactly: every transition node carries every proposed field, every component node is
present as proposed, every proposed edge is live). Anything in between, or any drift on a proposed key, is rejected. In
either state the proposal is replayed from the register without it (strip) through the hard gate; in the installed state
apply() is additionally the identity and this record's objects are unheld. The printed report names the state; the
replay's impacted set is printed only in the baseline state (post-execution it depends on the other landed records).
"""''')

sub1('''           "skip-interface-premise", "holder-reversed", "baseline-drift", "duplicate-drift", "region-unlinked",
           "review-metadata-only")''', '''           "skip-interface-premise", "holder-reversed", "baseline-drift", "duplicate-drift", "region-unlinked",
           "review-metadata-only", "installed-drift", "partial-install")''')

# --- check_baseline: accept baseline or installed, fail-closed otherwise
sub1('''def check_baseline(root, spec):
    live = json.loads((root / GRAPH).read_text(encoding="utf-8"))["nodes"]
    ok = True
    for t in spec["transitions"]:
        n = live.get(t["node"])
        if not isinstance(n, dict):
            return False
        ok &= n.get("classification") == t["current"]["classification"]
        ok &= n.get("fingerprint") == t["current"]["fingerprint"]
        ok &= n.get("kind") == t["kind"] and n.get("controlling") is False
        if t["kind"] == "candidate":
            ok &= n.get("source") == t["proposed"]["source"]
''', '''def proposed_fields(p):
    """The live fields the proposal writes on a transition node (apply's normative form)."""
    f = {"classification": p["classification"], "controlling": p["controlling"], "fingerprint": p["fingerprint"],
         "scope": p["scope"], "explicit_limits": p["explicit_limits"], "notes": p["notes"],
         "review_disposition": p["review_disposition"], "review_sources": [r["ref"] for r in p["review_sources"]],
         "review_basis": [{"review": r["ref"], "provider": r.get("provider", "unstated"), "verdict": r.get("verdict", "")}
                          for r in p["review_sources"]]}
    if p.get("source"):
        f["source"] = p["source"]
    if p.get("coverage_source"):
        f["coverage_source"] = p["coverage_source"]
    return f


def component_fields(comp):
    return {k: v for k, v in comp.items() if k != "id"}


def matches(live, fields):
    return isinstance(live, dict) and all(live.get(k) == v for k, v in fields.items())


def installed_state(graph, spec):
    """How much of the proposal the live graph carries. Fail-closed: a transition node is 'flipped' only if it carries
    every proposed field exactly and 'baseline' only if it carries the recorded current classification and fingerprint;
    a live node on a component source that is not the proposed node on every proposed key is a mismatch, as is a
    proposed id carrying anything else. The state is 'installed' (everything, exactly), 'baseline' (nothing) or
    'partial' (rejected by BASELINE and TRANSITIONS)."""
    nodes = graph["nodes"]
    flipped, baseline, mismatched, present = set(), set(), set(), set()
    for t in spec["transitions"]:
        n = nodes.get(t["node"])
        if matches(n, proposed_fields(t["proposed"])):
            flipped.add(t["node"])
        elif matches(n, {"classification": t["current"]["classification"], "fingerprint": t["current"]["fingerprint"]}):
            baseline.add(t["node"])
        else:
            mismatched.add(t["node"])
    by_source = {c["source"]: c for c in spec["proposed_graph_nodes"]}
    for nid, n in nodes.items():
        if not isinstance(n, dict) or n.get("source") not in by_source:
            continue
        comp = by_source[n["source"]]
        (present if nid == comp["id"] and matches(n, component_fields(comp)) else mismatched).add(nid)
    for c in spec["proposed_graph_nodes"]:
        if c["id"] in nodes and c["id"] not in present:
            mismatched.add(c["id"])
    live_edges = {(e["from"], e["to"], e["required"], e["relation"]) for e in graph["edges"]}
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in spec["proposed_graph_edges"]}
    n_edges = len(wanted & live_edges)
    ids = sorted(t["node"] for t in spec["transitions"])
    cids = sorted(c["id"] for c in spec["proposed_graph_nodes"])
    if not mismatched and sorted(flipped) == ids and sorted(present) == cids and n_edges == len(wanted):
        state = "installed"
    elif not mismatched and not flipped and not present and n_edges == 0:
        state = "baseline"
    else:
        state = "partial"
    return {"state": state, "flipped": sorted(flipped), "baseline": sorted(baseline), "mismatched": sorted(mismatched),
            "present": sorted(present), "edges_present": n_edges, "edges_proposed": len(wanted)}


def check_baseline(root, spec):
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    live = graph["nodes"]
    st = installed_state(graph, spec)
    ok = st["state"] in ("baseline", "installed")
    for t in spec["transitions"]:
        n = live.get(t["node"])
        if not isinstance(n, dict):
            return False
        expect = t["proposed"] if st["state"] == "installed" else t["current"]
        ok &= n.get("classification") == expect["classification"]
        ok &= n.get("fingerprint") == expect["fingerprint"]
        ok &= n.get("kind") == t["kind"] and n.get("controlling") is False
        if t["kind"] == "candidate":
            ok &= n.get("source") == t["proposed"]["source"]
        ok &= t["current"]["classification"] != t["proposed"]["classification"]   # a transition is a change
''')

# --- apply: idempotent; strip: pre-image
sub1('''def apply(spec, graph):
    """The proposal applied to a copy of the live graph: the eight flips with their scope exclusions and replacement
    notes, plus the review-record / consumed-source component nodes and the required edges that carry the evidence."""
    new = copy.deepcopy(graph)
    for t in spec["transitions"]:
        n = new["nodes"][t["node"]]
        p = t["proposed"]
        n["classification"] = p["classification"]
        n["controlling"] = p["controlling"]
        n["fingerprint"] = p["fingerprint"]
        if p.get("source"):
            n["source"] = p["source"]
        n["scope"] = p["scope"]
        n["explicit_limits"] = p["explicit_limits"]
        n["notes"] = p["notes"]
        n["review_disposition"] = p["review_disposition"]
        n["review_sources"] = [r["ref"] for r in p["review_sources"]]
        n["review_basis"] = [{"review": r["ref"], "provider": r.get("provider", "unstated"), "verdict": r.get("verdict", "")}
                             for r in p["review_sources"]]
        if p.get("coverage_source"):
            n["coverage_source"] = p["coverage_source"]
    for c in spec["proposed_graph_nodes"]:
        new["nodes"][c["id"]] = {k: v for k, v in c.items() if k != "id"}
    new["edges"] = list(new["edges"]) + [dict(e) for e in spec["proposed_graph_edges"]]
    return new
''', '''def apply(spec, graph):
    """The proposal applied to a copy of the live graph, idempotently: the eight flips with their scope exclusions and
    replacement notes (a no-op on a node that already carries them), the review-record / consumed-source component
    nodes (not re-created when present as proposed) and the required edges that carry the evidence (not re-added when
    live). On the installed register apply is the identity."""
    new = copy.deepcopy(graph)
    for t in spec["transitions"]:
        new["nodes"][t["node"]].update(proposed_fields(t["proposed"]))
    present = set(installed_state(graph, spec)["present"])
    for c in spec["proposed_graph_nodes"]:
        if c["id"] not in present:
            new["nodes"][c["id"]] = component_fields(c)
    have = {(e["from"], e["to"], e["required"], e["relation"]) for e in new["edges"]}
    new["edges"] = list(new["edges"])
    for e in spec["proposed_graph_edges"]:
        key = (e["from"], e["to"], e["required"], e["relation"])
        if key not in have:
            new["edges"].append(dict(e))
            have.add(key)
    return new


def strip(spec, graph):
    """The register without the proposal, the pre-image the hard-gate replay starts from: the eight nodes at their
    recorded current classification and fingerprint, the component nodes and every edge touching them removed, the
    proposed edges removed. On the baseline register strip is the identity; apply(strip(installed)) reproduces every
    installed object on every proposed key."""
    old = copy.deepcopy(graph)
    for t in spec["transitions"]:
        old["nodes"][t["node"]]["classification"] = t["current"]["classification"]
        old["nodes"][t["node"]]["fingerprint"] = t["current"]["fingerprint"]
    cids = {c["id"] for c in spec["proposed_graph_nodes"]}
    for cid in cids:
        old["nodes"].pop(cid, None)
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in spec["proposed_graph_edges"]}
    old["edges"] = [e for e in old["edges"] if (e["from"], e["to"], e["required"], e["relation"]) not in wanted
                    and e["from"] not in cids and e["to"] not in cids]
    return old
''')

# --- check_transitions: live-dependent rules by state
sub1('''def check_transitions(root, spec):
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    ok = spec.get("declarative") is True and spec.get("executed") is False and spec.get("edges_unchanged") is False
''', '''def check_transitions(root, spec):
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    st = installed_state(graph, spec)
    installed = st["state"] == "installed"
    ok = spec.get("declarative") is True and spec.get("executed") is False and spec.get("edges_unchanged") is False
    ok &= st["state"] in ("baseline", "installed")
''')
sub1('''    # component nodes: one per byte identity, fingerprinted to the inventory, PROVED_REVIEWED evidence records,
    # none for bytes the live graph already carries with a fingerprint (Codex 4139312869 on Math-#160)
    comps = spec["proposed_graph_nodes"]
    cids = [c["id"] for c in comps]
    live_fp_sources = {n.get("source") for n in graph["nodes"].values()
                       if isinstance(n, dict) and n.get("fingerprint") and n.get("source")}
    ok &= len(cids) == len(set(cids)) and not (set(cids) & set(graph["nodes"])) and len(comps) == spec["nodes_added"]
''', '''    # component nodes: one per byte identity, fingerprinted to the inventory, PROVED_REVIEWED evidence records,
    # none for bytes the live graph already carries with a fingerprint (Codex 4139312869 on Math-#160); once installed,
    # the only live node on a component source is the component itself
    comps = spec["proposed_graph_nodes"]
    cids = [c["id"] for c in comps]
    live_fp_sources = {n.get("source") for nid, n in graph["nodes"].items()
                       if isinstance(n, dict) and n.get("fingerprint") and n.get("source") and nid not in cids}
    ok &= len(cids) == len(set(cids)) and len(comps) == spec["nodes_added"]
    ok &= (set(cids) <= set(graph["nodes"])) if installed else not (set(cids) & set(graph["nodes"]))
''')
sub1('''    edges_new = spec["proposed_graph_edges"]
    known = set(graph["nodes"]) | set(cids)
    live_keys = {(e["from"], e["to"], e["relation"]) for e in graph["edges"]}
''', '''    edges_new = spec["proposed_graph_edges"]
    known = set(graph["nodes"]) | set(cids)
    base = strip(spec, graph)                                         # the register without the proposal
    live_keys = {(e["from"], e["to"], e["relation"]) for e in base["edges"]}
''')
sub1('''    if not ok:
        return False, None
    new = apply(spec, graph)
    nodes = new["nodes"]
    edges = new["edges"]
    ok &= edges[:len(graph["edges"])] == graph["edges"] and len(edges) == len(graph["edges"]) + len(edges_new)
    ok &= all(graph["nodes"][k] == nodes[k] for k in graph["nodes"] if k not in ids)   # only the eight live nodes change
''', '''    if not ok:
        return False, None
    new = apply(spec, graph)
    replay = apply(spec, base)
    nodes = new["nodes"]
    edges = new["edges"]
    if installed:
        ok &= new == graph                                            # the live graph carries the proposal exactly: apply is a no-op
    else:
        ok &= base == graph and replay == new                         # strip is a no-op on the baseline register
    ok &= edges[:len(graph["edges"])] == graph["edges"] and len(edges) == len(graph["edges"]) + (0 if installed else len(edges_new))
    ok &= all(graph["nodes"][k] == nodes[k] for k in graph["nodes"] if k not in ids)   # only the eight live nodes change
    for t in spec["transitions"]:                                     # the replay from the pre-image reproduces every object
        ok &= matches(replay["nodes"][t["node"]], proposed_fields(t["proposed"]))
        ok &= matches(nodes[t["node"]], proposed_fields(t["proposed"]))
    for c in comps:
        ok &= matches(replay["nodes"][c["id"]], component_fields(c)) and matches(nodes[c["id"]], component_fields(c))
''')
sub1('''    for c in comps:                                                   # component nodes' own required premises
        req = [e["to"] for e in edges if e["from"] == c["id"] and e["required"] is True]
        ok &= all(nodes[d]["classification"] == "PROVED_REVIEWED" for d in req)
    return bool(ok), new
''', '''    for c in comps:                                                   # component nodes' own required premises
        req = [e["to"] for e in edges if e["from"] == c["id"] and e["required"] is True]
        ok &= all(nodes[d]["classification"] == "PROVED_REVIEWED" for d in req)
    # the installed state, simulated on the baseline register (the live one once installed): the executed graph is
    # accepted exactly, and re-applying the proposal to it is a no-op; a drifted or partial install is rejected
    sim = copy.deepcopy(new)
    if MUT == "installed-drift":
        sim["nodes"][cids[0]]["fingerprint"] = "0" * 64
    if MUT == "partial-install":
        t4 = next(t for t in spec["transitions"] if t["node"] == D4_NODE)
        sim["nodes"][D4_NODE]["classification"] = t4["current"]["classification"]
    sim_st = installed_state(sim, spec)
    ok &= sim_st["state"] == "installed" and apply(spec, sim) == sim and sim_st["mismatched"] == []
    return bool(ok), {"new": new, "base": base, "replay": replay, "state": st["state"]}
''')

# --- check_gate: replay from the pre-image; installed extras
sub1('''def check_gate(root, spec, new):
    if new is None:
        return False, {}
    hg = load_hard_gate(root)
    old = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    try:
        hg.validate_graph_fail_closed(new)
        rep_new = hg.closure_report(new)
        rep_old = hg.closure_report(old)
        ri = hg.reverse_impact_between(old, new)
    except Exception:
        return False, {}
    ok = rep_new.get("gate_ok") is True and not rep_new.get("illegal_controlling")
    ok &= rep_old.get("gate_ok") is True
    ok &= not any(n.get("controlling") for n in new["nodes"].values())
    changed = sorted([t["node"] for t in spec["transitions"]] + [c["id"] for c in spec["proposed_graph_nodes"]])
    imp = ri.get("impacted") if isinstance(ri, dict) else None
    names = []
    if isinstance(imp, list):
        for x in imp:
            names.append(x if isinstance(x, str) else str(x.get("node") or x.get("id") or x))
    # loss-only rule: every changed node is itself in the impacted set
    ok &= isinstance(ri, dict) and set(changed) <= set(names)
    return bool(ok), {"changed_nodes": changed, "impacted": sorted(set(names)), "reverse_impact_keys": sorted(ri.keys()) if isinstance(ri, dict) else []}
''', '''def check_gate(root, spec, builds):
    """The proposal replayed from the register without it (strip) through the hard gate: in the baseline state that is
    the live graph against the proposed one; in the installed state the pre-image against its re-application, plus
    the installed-state facts (nothing left to apply, this record's objects unheld)."""
    if builds is None:
        return False, {}
    new, old, replay, state = builds["new"], builds["base"], builds["replay"], builds["state"]
    hg = load_hard_gate(root)
    live = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    try:
        hg.validate_graph_fail_closed(new)
        hg.validate_graph_fail_closed(replay)
        rep_new = hg.closure_report(replay)
        rep_old = hg.closure_report(old)
        rep_live = hg.closure_report(new)
        ri = hg.reverse_impact_between(old, replay)
        ri_live = hg.reverse_impact_between(live, new)
    except Exception:
        return False, {}
    ok = rep_new.get("gate_ok") is True and not rep_new.get("illegal_controlling")
    ok &= rep_old.get("gate_ok") is True and rep_live.get("gate_ok") is True
    ok &= not any(n.get("controlling") for n in new["nodes"].values())
    changed = sorted([t["node"] for t in spec["transitions"]] + [c["id"] for c in spec["proposed_graph_nodes"]])
    imp = ri.get("impacted") if isinstance(ri, dict) else None
    names = []
    if isinstance(imp, list):
        for x in imp:
            names.append(x if isinstance(x, str) else str(x.get("node") or x.get("id") or x))
    # loss-only rule: every changed node is itself in the impacted set, and the replay changes every one of them
    ok &= isinstance(ri, dict) and set(changed) <= set(names) and set(changed) <= set(ri.get("changed_nodes", []))
    # this record's objects are unheld once applied: every required premise PROVED_REVIEWED
    held = {h.get("node") for h in rep_live.get("hold_proposals", [])}
    ok &= not (set(changed) & held)
    if state == "installed":
        ok &= ri_live.get("changed_nodes") == [] and ri_live.get("impacted") == []   # nothing left to apply
        return bool(ok), {"register_state": "installed",
                          "replay": "the proposal replayed from the pre-image (strip) on the installed register: loss-only rule "
                                    "and unheld objects asserted; the impacted set is not pinned post-execution"}
    ok &= ri_live.get("changed_nodes") == ri.get("changed_nodes") and ri_live.get("impacted") == ri.get("impacted")
    return bool(ok), {"register_state": "baseline", "changed_nodes": changed, "impacted": sorted(set(names)),
                      "reverse_impact_keys": sorted(ri.keys())}
''')

# --- d4 duplicate: expected classification by state
sub1('''    ok &= any("main#76" in str(b.get("review")) and b.get("verdict") == "ACCEPT" for b in dup.get("review_basis", []))
    ok &= old.get("fingerprint") == dup.get("fingerprint") and old.get("classification") == t["current"]["classification"]
    return bool(ok)
''', '''    ok &= any("main#76" in str(b.get("review")) and b.get("verdict") == "ACCEPT" for b in dup.get("review_basis", []))
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    expect = t["proposed" if installed_state(graph, spec)["state"] == "installed" else "current"]["classification"]
    ok &= old.get("fingerprint") == dup.get("fingerprint") and old.get("classification") == expect
    return bool(ok)
''')

# --- propagation takes builds
sub1('''def check_propagation(root, spec, new):
    """Two-commit mutations on the proposed graph through the production loss-only rule (reverse_impact_between with
    source snapshots): evidence and consumed-source changes must reach the objects they support (OpenAI 5360320845,
    5360327058; Codex 4139807558, 4139807567). Nothing scientific is decided here."""
    if new is None:
        return False, {}
    hg = load_hard_gate(root)
    g = copy.deepcopy(new)
''', '''def check_propagation(root, spec, builds):
    """Two-commit mutations on the proposed graph through the production loss-only rule (reverse_impact_between with
    source snapshots): evidence and consumed-source changes must reach the objects they support (OpenAI 5360320845,
    5360327058; Codex 4139807558, 4139807567). Nothing scientific is decided here."""
    if builds is None:
        return False, {}
    hg = load_hard_gate(root)
    g = copy.deepcopy(builds["new"])
''')

# --- main: state in the report
sub1('''    ident = identities_ok(root)
    trans_ok, new = check_transitions(root, spec)
    gate_ok, gate = check_gate(root, spec, new)
    prop_ok, prop = check_propagation(root, spec, new)
''', '''    ident = identities_ok(root)
    trans_ok, builds = check_transitions(root, spec)
    gate_ok, gate = check_gate(root, spec, builds)
    prop_ok, prop = check_propagation(root, spec, builds)
    state = installed_state(json.loads((root / GRAPH).read_text(encoding="utf-8")), spec)["state"]
''')
sub1('''    print(json.dumps({"object": "REGISTER-ALIGNMENT-20260930-v1", "checks": checks, "passed": passed,
                      "inventory_files": len(INVENTORY), "transitions": len(spec["transitions"]),''',
     '''    print(json.dumps({"object": "REGISTER-ALIGNMENT-20260930-v1", "checks": checks, "passed": passed,
                      "register_state": state,
                      "inventory_files": len(INVENTORY), "transitions": len(spec["transitions"]),''')
sub1('''                      "scope": "identity, verdict-row, live-baseline, transition-shape, hard-gate replay, two-commit propagation, "
                               "interface arithmetic (exact and numerical parts labelled), filesystem-negative and live D4-duplicate "
                               "checks; no mathematics is re-proved; nothing is written"}, indent=2, sort_keys=True))''',
     '''                      "scope": "identity, verdict-row, live-register (baseline or exactly installed), transition-shape, hard-gate "
                               "replay from the pre-image, two-commit propagation, interface arithmetic (exact and numerical parts "
                               "labelled), filesystem-negative and live D4-duplicate checks; no mathematics is re-proved; nothing is "
                               "written"}, indent=2, sort_keys=True))''')
P.write_text(s)
print("ok", len(s))
