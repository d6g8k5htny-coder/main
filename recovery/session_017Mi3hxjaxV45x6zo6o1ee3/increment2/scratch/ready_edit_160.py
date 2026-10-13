"""v1.8 edit of reconciliation_check.py: installed-state acceptance (exact replacements; fails loudly on a miss)."""
import pathlib, sys
P = pathlib.Path(sys.argv[1]) / "reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py"
s = P.read_text()
def sub1(old, new):
    global s
    assert s.count(old) == 1, ("MISS", old[:80], s.count(old))
    s = s.replace(old, new)

sub1('"clone-live-node", "open-cell-left")', '"clone-live-node", "open-cell-left",\n           "installed-drift", "partial-install")')
if "\nimport copy\n" not in s:
    sub1("\nimport hashlib\n", "\nimport copy\nimport hashlib\n")

sub1('''def check_transitions(root):
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    d5 = json.loads((root / D5_TRANSITIONS).read_text(encoding="utf-8"))
    live = json.loads((root / GRAPH).read_text(encoding="utf-8"))["nodes"]
''', '''def matches(live, fields):
    return isinstance(live, dict) and all(live.get(k) == v for k, v in fields.items())


def installed_state(graph, proposed_nodes, proposed_edges):
    """How much of section 7's proposal the live graph carries, on the main path (the residual node created separately;
    Math-#173 may since have promoted it). Fail-closed: a proposed id carrying anything but the proposal on every
    proposed key is a mismatch (the residual: this record's node OPEN_ACTIVE with its notes, or PROVED_REVIEWED as
    Math-#173 proposes, with layer, kind, fingerprint and controlling unchanged); so is any other live fingerprinted
    node on a proposed source. 'installed': every node present, every edge not leaving the witness node live, and the
    witness node OPEN_ACTIVE (step 4 pending) or PROVED_REVIEWED with all its proposed edges live and their required
    targets PROVED_REVIEWED. 'baseline': nothing of the proposal live. Anything else is 'partial' and rejected."""
    nodes = graph["nodes"]
    proposed = {n["id"]: n for n in proposed_nodes}
    present, mismatched = set(), set()
    for nid, comp in proposed.items():
        n = nodes.get(nid)
        if n is None:
            continue
        fields = {k: v for k, v in comp.items() if k != "id"}
        if nid == RESIDUAL:
            fields = {k: fields[k] for k in ("layer", "kind", "fingerprint", "controlling")}
            fields_ok = (n.get("classification") == "PROVED_REVIEWED"
                         or (n.get("classification") == "OPEN_ACTIVE" and n.get("notes") == comp["notes"]))
        else:
            fields_ok = True
        (present if matches(n, fields) and fields_ok else mismatched).add(nid)
    by_source = {n["source"]: nid for nid, n in proposed.items() if n.get("source")}
    for nid, n in nodes.items():
        if isinstance(n, dict) and n.get("fingerprint") and n.get("source") in by_source and nid != by_source[n["source"]]:
            mismatched.add(nid)
    live_edges = {(e["from"], e["to"], e["required"], e["relation"]) for e in graph["edges"]}
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in proposed_edges}
    wit_edges = {e for e in wanted if e[0] == WITNESS}
    other = wanted - wit_edges
    n_other, n_wit = len(other & live_edges), len(wit_edges & live_edges)
    wit = nodes.get(WITNESS) if isinstance(nodes.get(WITNESS), dict) else {}
    cls = wit.get("classification")
    wit_ok = cls == "OPEN_ACTIVE" or (cls == "PROVED_REVIEWED" and n_wit == len(wit_edges) and all(
        nodes.get(e[1], {}).get("classification") == "PROVED_REVIEWED" for e in wit_edges if e[2] is True))
    if not mismatched and set(present) == set(proposed) and n_other == len(other) and wit_ok:
        state = "installed"
    elif not mismatched and not present and n_other == 0 and n_wit == 0:
        state = "baseline"
    else:
        state = "partial"
    return {"state": state, "present": sorted(present), "mismatched": sorted(mismatched),
            "edges_present": n_other + n_wit, "edges_proposed": len(wanted), "witness": cls}


def register_state(root):
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    return installed_state(graph, spec["proposed_graph_nodes"], spec["proposed_graph_edges"])["state"]


def check_transitions(root):
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    d5 = json.loads((root / D5_TRANSITIONS).read_text(encoding="utf-8"))
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    live = graph["nodes"]
''')

sub1('''    ok = spec.get("declarative") is True and spec.get("executed") is False
    # shape rules of the hard gate, applied to the proposed nodes and edges
''', '''    ok = spec.get("declarative") is True and spec.get("executed") is False
    # the live register carries nothing of the proposal, or all of it exactly (v1.8; Math-#173 may have promoted the residual)
    st = installed_state(graph, list(nodes.values()), edges)
    ok &= st["state"] in ("baseline", "installed")
    # shape rules of the hard gate, applied to the proposed nodes and edges
''')

sub1('''    # no proposed node duplicates a source the live graph already carries with a fingerprint (Codex 4139312869)
    ok &= not ({n.get("source") for n in nodes.values() if "fingerprint" in n and n.get("source")}
               & {n["source"] for n in live_fp.values()})
''', '''    # no proposed node duplicates a source the live graph already carries with a fingerprint (Codex 4139312869); once
    # installed, this record's own nodes are the live carriers of their sources and nothing else is (installed_state)
    ok &= not ({n.get("source") for n in nodes.values() if "fingerprint" in n and n.get("source")}
               & {n["source"] for nid, n in live_fp.items() if nid not in nodes})
''')

sub1('''    # the D4 bytes are carried by two live nodes (the D5 proposal landed as written); they must agree byte for byte
    dup = live.get("math.d5-component.remote-window-proof")
    if isinstance(dup, dict):
        ok &= dup.get("fingerprint") == live.get("math.rn-fixed-remote-window", {}).get("fingerprint") \\
            == INVENTORY["frontiers/remote_window_20260924/PROOF.md"][0]
    return bool(ok)
''', '''    # the D4 bytes are carried by two live nodes (the D5 proposal landed as written); they must agree byte for byte
    dup = live.get("math.d5-component.remote-window-proof")
    if isinstance(dup, dict):
        ok &= dup.get("fingerprint") == live.get("math.rn-fixed-remote-window", {}).get("fingerprint") \\
            == INVENTORY["frontiers/remote_window_20260924/PROOF.md"][0]
    # the installed state, simulated on the baseline register (the live one once installed): steps 1-3 applied, the
    # witness node left as it is; accepted exactly, and a drifted or partial install is rejected
    sim = copy.deepcopy(graph)
    for nid, n in nodes.items():
        if nid not in sim["nodes"]:
            sim["nodes"][nid] = {k: v for k, v in n.items() if k != "id"}
    have = {(e["from"], e["to"], e["required"], e["relation"]) for e in sim["edges"]}
    for e in edges:
        key = (e["from"], e["to"], e["required"], e["relation"])
        if key not in have and e["from"] != WITNESS:
            sim["edges"].append({"from": e["from"], "to": e["to"], "required": e["required"], "relation": e["relation"]})
            have.add(key)
    if MUT == "installed-drift":
        sim["nodes"]["math.c6-component.palm-proof"]["fingerprint"] = "0" * 64
    if MUT == "partial-install":
        sim["edges"] = [e for e in sim["edges"] if e["from"] != SUPERSESSION]
    sim_st = installed_state(sim, list(nodes.values()), edges)
    ok &= sim_st["state"] == "installed" and sim_st["mismatched"] == [] and sim_st["present"] == sorted(nodes)
    return bool(ok)
''')

sub1('''def check_selector(root):
    """Applied to the live selector table, the proposal leaves no open cell in its regions and moves exactly those
    region ids from open to covered (Codex 4139312863)."""
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    live = json.loads((root / SELECTOR).read_text(encoding="utf-8"))
    prop = spec["selector_region_proposal"]
    cells = dict(prop["cells"])
    if MUT == "open-cell-left":
        cells.pop("ENV-RESCOV", None)
    regions = list(prop["regions"])
    ok = set(regions) <= set(live["regions"]) and len(regions) == 3
    ok &= set(cells) == set(live["selectors"])
    ok &= all(v not in OPEN_CELLS for v in cells.values())
    table = {s: dict(row) for s, row in live["selectors"].items()}
    for s, v in cells.items():
        for reg in regions:
            table[s][reg] = v
    def is_open(reg):
        return any(table[s].get(reg) in OPEN_CELLS for s in live["selectors"])
    ok &= not any(is_open(reg) for reg in regions)
    ids = ["math.rn-region." + reg for reg in regions]
    covered = list(live["covered_region_ids"]) + [i for i in ids if i not in live["covered_region_ids"]]
    opened = [i for i in live["open_region_ids"] if i not in ids]
    ok &= sorted(prop["resulting_covered_region_ids"]) == sorted(covered)
    ok &= sorted(prop["resulting_open_region_ids"]) == sorted(opened)
    ok &= all(("math.rn-region." + reg) in live["open_region_ids"] for reg in regions)
    ok &= prop.get("under_node") == SUPERSESSION
    return bool(ok)
''', '''def check_selector(root):
    """Applied to the live selector table, the proposal leaves no open cell in its regions and moves exactly those
    region ids from open to covered (Codex 4139312863). Once the table carries every proposed cell, the three region
    ids must be covered and not open (installed); a table carrying part of the proposal is rejected (v1.8)."""
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    live = json.loads((root / SELECTOR).read_text(encoding="utf-8"))
    prop = spec["selector_region_proposal"]
    cells = dict(prop["cells"])
    if MUT == "open-cell-left":
        cells.pop("ENV-RESCOV", None)
    regions = list(prop["regions"])
    ok = set(regions) <= set(live["regions"]) and len(regions) == 3
    ok &= set(cells) == set(live["selectors"])
    ok &= all(v not in OPEN_CELLS for v in cells.values())
    ids = ["math.rn-region." + reg for reg in regions]
    applied = sum(1 for s, v in cells.items() for reg in regions if live["selectors"].get(s, {}).get(reg) == v)
    table = {s: dict(row) for s, row in live["selectors"].items()}
    if applied == len(cells) * len(regions):                          # installed: cells and region lists as proposed
        ok &= set(ids) <= set(live["covered_region_ids"]) and not (set(ids) & set(live["open_region_ids"]))
    else:                                                             # baseline: the proposal applied to the live table
        ok &= applied == 0
        for s, v in cells.items():
            for reg in regions:
                table[s][reg] = v
        covered = list(live["covered_region_ids"]) + [i for i in ids if i not in live["covered_region_ids"]]
        opened = [i for i in live["open_region_ids"] if i not in ids]
        ok &= sorted(prop["resulting_covered_region_ids"]) == sorted(covered)
        ok &= sorted(prop["resulting_open_region_ids"]) == sorted(opened)
        ok &= all(i in live["open_region_ids"] for i in ids)
    def is_open(reg):
        return any(table[s].get(reg) in OPEN_CELLS for s in live["selectors"])
    ok &= not any(is_open(reg) for reg in regions)
    ok &= prop.get("under_node") == SUPERSESSION
    return bool(ok)
''')

sub1('''    print(json.dumps({"object": "C6-WITNESS-COLLISION-RECONCILIATION-20260929-v1", "checks": checks, "passed": passed,
                      "inventory_files": len(INVENTORY),
                      "scope": "identity, verdict-row, obligation, filesystem-negative, transition-chain, monotonicity, "
                               "open-item and selector-cell checks; no mathematics is re-proved"}, indent=2, sort_keys=True))''',
     '''    print(json.dumps({"object": "C6-WITNESS-COLLISION-RECONCILIATION-20260929-v1", "checks": checks, "passed": passed,
                      "inventory_files": len(INVENTORY), "register_state": register_state(root),
                      "scope": "identity, verdict-row, obligation, filesystem-negative, transition-chain (live register baseline "
                               "or exactly installed), monotonicity, open-item and selector-cell checks; no mathematics is "
                               "re-proved"}, indent=2, sort_keys=True))''')
P.write_text(s)
print("ok", len(s))
