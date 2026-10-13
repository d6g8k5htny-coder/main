"""Math-#183: fixes for Codex review 5365065925 (threads 4143661492/496/501/511/519/525). Exact replacements."""
import pathlib, sys, json, hashlib
W = pathlib.Path(sys.argv[1])
def patch(path, pairs):
    p = W / path; s = p.read_text()
    for old, new in pairs:
        assert s.count(old) == 1, ("MISS", path, old[:80], s.count(old))
        s = s.replace(old, new)
    p.write_text(s); print("patched", path)

# ---------------------------------------------------------------- Math-#167 alignment_check.py (4143661511)
TEXT_FIELDS = ("scope", "explicit_limits", "notes", "review_disposition", "review_sources", "review_basis", "coverage_source", "source")
g = json.loads((W / "frontiers/downstream_gate_20260925/GRAPH.json").read_text())
spec167 = json.loads((W / "reviews/register_alignment_20260930/PROPOSED_TRANSITIONS.json").read_text())
digests = {}
for t in spec167["transitions"]:
    n = g["nodes"][t["node"]]
    digests[t["node"]] = hashlib.sha256(json.dumps({k: n.get(k) for k in TEXT_FIELDS}, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
lines = ",\n".join('    "%s": "%s"' % (k, v) for k, v in digests.items())
patch("reviews/register_alignment_20260930/alignment_check.py", [
('''           "review-metadata-only", "installed-drift", "partial-install")''',
 '''           "review-metadata-only", "installed-drift", "partial-install", "text-preinstalled")'''),
('''D4 = "frontiers/remote_window_20260924/PROOF.md"
''', '''D4 = "frontiers/remote_window_20260924/PROOF.md"
# the proposal-written fields of a transition node, and their digest per node as live at dda8991 (json, sorted keys,
# absent = null): the baseline state requires them unchanged, so the proposal's metadata installed under the current
# classification is a partial install, not baseline (Codex 4143661511 on Math-#183)
TEXT_FIELDS = ("scope", "explicit_limits", "notes", "review_disposition", "review_sources", "review_basis",
               "coverage_source", "source")
BASELINE_TEXT = {
''' + lines + '''
}
'''),
('''def matches(live, fields):
    return isinstance(live, dict) and all(live.get(k) == v for k, v in fields.items())


def installed_state(graph, spec):''', '''def matches(live, fields):
    return isinstance(live, dict) and all(live.get(k) == v for k, v in fields.items())


def text_digest(node):
    return hashlib.sha256(json.dumps({k: node.get(k) for k in TEXT_FIELDS}, sort_keys=True,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def at_baseline(node, t):
    """The recorded current classification and fingerprint, and the proposal-written fields as they were at dda8991."""
    ok = matches(node, {"classification": t["current"]["classification"], "fingerprint": t["current"]["fingerprint"]})
    if MUT != "text-preinstalled":
        ok = ok and text_digest(node) == BASELINE_TEXT[t["node"]]
    return ok


def installed_state(graph, spec):'''),
('''    """How much of the proposal the live graph carries. Fail-closed: a transition node is 'flipped' only if it carries
    every proposed field exactly and 'baseline' only if it carries the recorded current classification and fingerprint;
    a live node''', '''    """How much of the proposal the live graph carries. Fail-closed: a transition node is 'flipped' only if it carries
    every proposed field exactly and 'baseline' only if it carries the recorded current classification and fingerprint
    with the proposal-written fields unchanged since dda8991 (BASELINE_TEXT); a live node'''),
('''        if matches(n, proposed_fields(t["proposed"])):
            flipped.add(t["node"])
        elif matches(n, {"classification": t["current"]["classification"], "fingerprint": t["current"]["fingerprint"]}):
            baseline.add(t["node"])''', '''        if matches(n, proposed_fields(t["proposed"])):
            flipped.add(t["node"])
        elif isinstance(n, dict) and at_baseline(n, t):
            baseline.add(t["node"])'''),
('''    sim_st = installed_state(sim, spec)
    ok &= sim_st["state"] == "installed" and apply(spec, sim) == sim and sim_st["mismatched"] == []
    return bool(ok), {"new": new, "base": base, "replay": replay, "state": st["state"]}''',
 '''    sim_st = installed_state(sim, spec)
    ok &= sim_st["state"] == "installed" and apply(spec, sim) == sim and sim_st["mismatched"] == []
    # the pre-image of the applied graph carries the proposed metadata under the current classifications: a partial
    # install, never baseline (Codex 4143661511)
    ok &= installed_state(strip(spec, new), spec)["state"] == "partial"
    return bool(ok), {"new": new, "base": base, "replay": replay, "state": st["state"]}'''),
])

# ---------------------------------------------------------------- Math-#160 reconciliation_check.py (4143661492, 4143661519)
patch("reviews/c6_witness_collision_reconciliation_20260929/reconciliation_check.py", [
('''           "installed-drift", "partial-install")''', '''           "installed-drift", "partial-install", "selector-ahead", "witness-edges-partial")'''),
('''    witness node OPEN_ACTIVE (step 4 pending) or PROVED_REVIEWED with all its proposed edges live and their required
    targets PROVED_REVIEWED. 'baseline': nothing of the proposal live. Anything else is 'partial' and rejected."""''',
 '''    witness node OPEN_ACTIVE with none of its proposed edges live (step 4 pending) or PROVED_REVIEWED with all of them
    live and their required targets PROVED_REVIEWED (Codex 4143661519). 'baseline': nothing of the proposal live.
    Anything else is 'partial' and rejected."""'''),
('''    wit_ok = cls == "OPEN_ACTIVE" or (cls == "PROVED_REVIEWED" and n_wit == len(wit_edges) and all(
        nodes.get(e[1], {}).get("classification") == "PROVED_REVIEWED" for e in wit_edges if e[2] is True))''',
 '''    targets_ok = all(nodes.get(e[1], {}).get("classification") == "PROVED_REVIEWED" for e in wit_edges if e[2] is True)
    if MUT == "witness-edges-partial":
        wit_ok = cls == "OPEN_ACTIVE" or (cls == "PROVED_REVIEWED" and targets_ok)
    else:
        wit_ok = (cls == "OPEN_ACTIVE" and n_wit == 0) or (cls == "PROVED_REVIEWED" and n_wit == len(wit_edges) and targets_ok)'''),
('''    sim_st = installed_state(sim, list(nodes.values()), edges)
    ok &= sim_st["state"] == "installed" and sim_st["mismatched"] == [] and sim_st["present"] == sorted(nodes)
    return bool(ok)
''', '''    sim_st = installed_state(sim, list(nodes.values()), edges)
    ok &= sim_st["state"] == "installed" and sim_st["mismatched"] == [] and sim_st["present"] == sorted(nodes)
    # one witness edge toggled (added while the witness node is open, removed once it is PROVED_REVIEWED) is a partial
    # step 4, never an accepted state (Codex 4143661519)
    wit_edges = [e for e in edges if e["from"] == WITNESS]
    if wit_edges:
        tog = copy.deepcopy(sim)
        first = wit_edges[0]
        key = (first["from"], first["to"], first["required"], first["relation"])
        if tog["nodes"].get(WITNESS, {}).get("classification") == "PROVED_REVIEWED":
            tog["edges"] = [e for e in tog["edges"] if (e["from"], e["to"], e["required"], e["relation"]) != key]
        else:
            tog["edges"].append({"from": key[0], "to": key[1], "required": key[2], "relation": key[3]})
        ok &= installed_state(tog, list(nodes.values()), edges)["state"] == "partial"
    else:
        ok = False
    return bool(ok)
'''),
('''def check_selector(root):
    """Applied to the live selector table, the proposal leaves no open cell in its regions and moves exactly those
    region ids from open to covered (Codex 4139312863). Once the table carries every proposed cell, the three region
    ids must be covered and not open (installed); a table carrying part of the proposal is rejected (v1.8)."""''',
 '''def selector_state_ok(applied_all, graph_state):
    """A table carrying every proposed cell is accepted only when the graph carries the proposal (Codex 4143661492)."""
    if MUT == "selector-ahead":
        return True
    return (not applied_all) or graph_state == "installed"


def check_selector(root):
    """Applied to the live selector table, the proposal leaves no open cell in its regions and moves exactly those
    region ids from open to covered (Codex 4139312863). Once the table carries every proposed cell, the three region
    ids must be covered and not open, and the graph must carry the proposal (installed); a table carrying part of the
    proposal, or the proposal ahead of the graph, is rejected (v1.8)."""'''),
('''    applied = sum(1 for s, v in cells.items() for reg in regions if live["selectors"].get(s, {}).get(reg) == v)
    table = {s: dict(row) for s, row in live["selectors"].items()}
    if applied == len(cells) * len(regions):                          # installed: cells and region lists as proposed
        ok &= set(ids) <= set(live["covered_region_ids"]) and not (set(ids) & set(live["open_region_ids"]))''',
 '''    applied = sum(1 for s, v in cells.items() for reg in regions if live["selectors"].get(s, {}).get(reg) == v)
    table = {s: dict(row) for s, row in live["selectors"].items()}
    ok &= selector_state_ok(applied == len(cells) * len(regions), register_state(root))
    ok &= not selector_state_ok(True, "baseline") and selector_state_ok(False, "baseline")   # the rule itself
    if applied == len(cells) * len(regions):                          # installed: cells and region lists as proposed
        ok &= set(ids) <= set(live["covered_region_ids"]) and not (set(ids) & set(live["open_region_ids"]))'''),
])

# ---------------------------------------------------------------- Math-#173 residual_check.py (4143661501)
patch("reviews/c6_residual_closure_20260930/residual_check.py", [
('''    res = nodes.get(RES)
    live_edges = {(e["from"], e["to"], e["required"], e["relation"]) for e in graph["edges"]}
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in spec["proposed_graph_edges"]
              if not e.get("deferred_with")}
    return {"present": sorted(present), "mismatched": sorted(mismatched),
            "residual": res.get("classification") if isinstance(res, dict) else None,
            "edges_present": wanted <= live_edges}
''', '''    res = nodes.get(RES)
    live_edges = {(e["from"], e["to"], e["required"], e["relation"]) for e in graph["edges"]}
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in spec["proposed_graph_edges"]
              if not e.get("deferred_with")}
    res_cls = res.get("classification") if isinstance(res, dict) else None
    all_ids = sorted(c["id"] for c in spec["proposed_graph_nodes"])
    # exactly two accepted states (Codex 4143661501): 'installed' (residual PROVED_REVIEWED, every component and edge
    # live) and 'baseline' (residual absent or OPEN_ACTIVE, no edge of this record, no component except the
    # cross-record palm node that Math-#160 creates); anything else, the interim shape included, is 'partial'
    if not mismatched and sorted(present) == all_ids and wanted <= live_edges and res_cls == "PROVED_REVIEWED":
        state = "installed"
    elif not mismatched and set(present) <= {PALM_NODE} and not (wanted & live_edges) and res_cls in (None, "OPEN_ACTIVE"):
        state = "baseline"
    else:
        state = "partial"
    if MUT == "partial-install" and state == "partial":
        state = "baseline"
    return {"present": sorted(present), "mismatched": sorted(mismatched), "residual": res_cls,
            "edges_present": wanted <= live_edges, "state": state}
'''),
('''    st = installed_state(graph, spec)
    ok &= not st["mismatched"]
    if st["residual"] == "PROVED_REVIEWED":
        # a promoted residual must carry the whole proposal: every component by id and bytes, every edge, the reading rule
        ok &= set(c["id"] for c in spec["proposed_graph_nodes"]) <= set(st["present"]) and st["edges_present"]
        ok &= set(res.get("reading_rule", [])) >= set(spec["transitions"][0]["proposed"]["reading_rule"])
    return bool(ok), st
''', '''    st = installed_state(graph, spec)
    ok &= not st["mismatched"] and st["state"] in ("baseline", "installed")
    if st["residual"] == "PROVED_REVIEWED":
        # a promoted residual must carry the whole proposal: every component by id and bytes, every edge, the reading rule
        ok &= set(c["id"] for c in spec["proposed_graph_nodes"]) <= set(st["present"]) and st["edges_present"]
        ok &= set(res.get("reading_rule", [])) >= set(spec["transitions"][0]["proposed"]["reading_rule"])
    return bool(ok), st
'''),
('''    o2, n2 = build(spec, sim, "final")
    ok &= o2 == n2
    if not installed:
        builds["installed-simulated"] = (o2, n2)
    return bool(ok), builds
''', '''    o2, n2 = build(spec, sim, "final")
    ok &= o2 == n2
    # the installed graph without one component (and its edges) is a partial install, never baseline (Codex 4143661501)
    part = copy.deepcopy(sim)
    part["nodes"].pop("math.c6r-component.spectral-closure-proof", None)
    part["edges"] = [e for e in part["edges"] if "math.c6r-component.spectral-closure-proof" not in (e["from"], e["to"])]
    ok &= installed_state(part, spec)["state"] == "partial"
    if not installed:
        builds["installed-simulated"] = (o2, n2)
    return bool(ok), builds
'''),
('''                      "register_state": "installed" if live_state.get("residual") == "PROVED_REVIEWED" else "baseline",''',
 '''                      "register_state": live_state.get("state"),'''),
('''MUTANTS = ("allow-symlink", "no-hash", "stale-fingerprint", "drop-required-edge", "executed-flag", "controlling-true",''',
 '''MUTANTS = ("partial-install", "allow-symlink", "no-hash", "stale-fingerprint", "drop-required-edge", "executed-flag", "controlling-true",'''),
])

# ---------------------------------------------------------------- readiness execution_dryrun.py (4143661496, 4143661525)
patch("reviews/register_execution_readiness_20260930/execution_dryrun.py", [
('''  --out DIR              keep the executed copy in DIR (must be outside the repository); default: a temporary directory''',
 '''  --out DIR              keep the executed copy in DIR: a path outside the repository (not inside it, not one of its
                         parents) that does not exist or is an empty directory; nothing is ever deleted; default: a
                         temporary directory'''),
('''  --write-installed-results   maintenance of this packet only: write each checker's installed-state output into its packet
                              as RESULTS_INSTALLED.json (then the normal run compares against it)''',
 '''  --write-installed-results   maintenance of this packet only: write each checker's installed-state output into its packet
                              as RESULTS_INSTALLED.json, and only after every checker (and every mutant, if requested)
                              has passed (then the normal run compares against it)'''),
('''def run_checkers(root, out, mutants, write_installed):
    results = {}
    ok = True
    for name, (pdir, script) in PACKETS.items():
        entry = {}
        outs = {}
        for flags in (("-B", "-S"), ("-B", "-O", "-S")):
            r = subprocess.run([sys.executable, *flags, str(out / pdir / script)], cwd=out, capture_output=True, timeout=900)
            outs[flags] = r.stdout
            try:
                payload = json.loads(r.stdout)
            except Exception:
                payload = {}
            entry[" ".join(flags)] = {"rc": r.returncode, "checks": payload.get("checks"), "passed": payload.get("passed"),
                                      "register_state": payload.get("register_state")}
            ok &= r.returncode == 0 and payload.get("passed") is True and payload.get("register_state") == "installed"
        entry["modes_identical"] = outs[("-B", "-S")] == outs[("-B", "-O", "-S")]
        ok &= entry["modes_identical"]
        pinned = root / pdir / "RESULTS_INSTALLED.json"
        if write_installed:
            pinned.write_bytes(outs[("-B", "-S")])
            entry["written"] = str(pinned.relative_to(root))
        if pinned.exists():
            entry["matches_RESULTS_INSTALLED"] = pinned.read_bytes() == outs[("-B", "-S")]
            entry["RESULTS_INSTALLED_sha256"] = hashlib.sha256(pinned.read_bytes()).hexdigest()
            ok &= entry["matches_RESULTS_INSTALLED"]
        else:
            entry["matches_RESULTS_INSTALLED"] = None
            ok = False
        if mutants:
            src = (out / pdir / script).read_text(encoding="utf-8")
            names = re.findall(r'"([a-z0-9-]+)"', re.search(r"MUTANTS = \\((.*?)\\)\\n", src, re.S).group(1))
            rejected, not_rejected = [], []
            for m in names:
                r = subprocess.run([sys.executable, "-B", "-S", str(out / pdir / script), "--mutant", m], cwd=out,
                                   capture_output=True, timeout=900)
                (rejected if r.returncode == 1 else not_rejected).append(m)
            entry["mutants_rejected"] = len(rejected)
            entry["mutants_not_rejected"] = not_rejected
            ok &= not not_rejected and len(rejected) == len(names)
        results[name] = entry
    return ok, results
''', '''def run_checkers(root, out, mutants, write_installed):
    """Run every checker (both interpreter modes, and every mutant if asked) on the executed copy; then, only if all of
    that passed, write the pinned installed outputs if asked; then compare with the pinned files."""
    results, outputs = {}, {}
    valid = True
    for name, (pdir, script) in PACKETS.items():
        entry = {}
        outs = {}
        for flags in (("-B", "-S"), ("-B", "-O", "-S")):
            r = subprocess.run([sys.executable, *flags, str(out / pdir / script)], cwd=out, capture_output=True, timeout=900)
            outs[flags] = r.stdout
            try:
                payload = json.loads(r.stdout)
            except Exception:
                payload = {}
            entry[" ".join(flags)] = {"rc": r.returncode, "checks": payload.get("checks"), "passed": payload.get("passed"),
                                      "register_state": payload.get("register_state")}
            valid &= r.returncode == 0 and payload.get("passed") is True and payload.get("register_state") == "installed"
        entry["modes_identical"] = outs[("-B", "-S")] == outs[("-B", "-O", "-S")]
        valid &= entry["modes_identical"]
        if mutants:
            src = (out / pdir / script).read_text(encoding="utf-8")
            names = re.findall(r'"([a-z0-9-]+)"', re.search(r"MUTANTS = \\((.*?)\\)\\n", src, re.S).group(1))
            rejected, not_rejected = [], []
            for m in names:
                r = subprocess.run([sys.executable, "-B", "-S", str(out / pdir / script), "--mutant", m], cwd=out,
                                   capture_output=True, timeout=900)
                (rejected if r.returncode == 1 else not_rejected).append(m)
            entry["mutants_rejected"] = len(rejected)
            entry["mutants_not_rejected"] = not_rejected
            valid &= not not_rejected and len(rejected) == len(names)
        results[name] = entry
        outputs[name] = outs[("-B", "-S")]
    ok = valid
    for name, (pdir, script) in PACKETS.items():
        entry = results[name]
        pinned = root / pdir / "RESULTS_INSTALLED.json"
        if write_installed and valid:                       # never on a failed run (Codex 4143661525)
            pinned.write_bytes(outputs[name])
            entry["written"] = str(pinned.relative_to(root))
        elif write_installed:
            entry["written"] = None
        if pinned.exists():
            entry["matches_RESULTS_INSTALLED"] = pinned.read_bytes() == outputs[name]
            entry["RESULTS_INSTALLED_sha256"] = hashlib.sha256(pinned.read_bytes()).hexdigest()
            ok &= entry["matches_RESULTS_INSTALLED"]
        else:
            entry["matches_RESULTS_INSTALLED"] = None
            ok = False
    return ok, results
'''),
('''    tmp = None
    if args.out:
        out = pathlib.Path(args.out).resolve()
        if root == out or root in out.parents:
            print(json.dumps({"error": "--out must lie outside the repository"}))
            return 2
        if out.exists():
            shutil.rmtree(out)
    else:''', '''    tmp = None
    if args.out:
        out = pathlib.Path(args.out).resolve()
        if root == out or root in out.parents or out in root.parents:
            print(json.dumps({"error": "--out must lie outside the repository (neither inside it nor one of its parents)"}))
            return 2
        if out.exists() and (not out.is_dir() or any(out.iterdir())):   # nothing is ever deleted (Codex 4143661496)
            print(json.dumps({"error": "--out must not exist or must be an empty directory; nothing is deleted"}))
            return 2
    else:'''),
])
print(json.dumps(digests, indent=1))
