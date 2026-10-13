import json, re, pathlib, hashlib
WT = pathlib.Path("<SCRATCH>/wt167")
R = WT / "reviews/register_alignment_20260930"
HERE = "reviews/register_alignment_20260930"
md_p, js_p, py_p, rv_p, wf_p = R/"RECONCILIATION.md", R/"PROPOSED_TRANSITIONS.json", R/"alignment_check.py", R/"RN_COUNT_INTERFACE_REVIEW.md", WT/".github/workflows/register-alignment.yml"
md, raw, py, rv, wf = md_p.read_text(), js_p.read_text(), py_p.read_text(), rv_p.read_text(), wf_p.read_text()
js = json.loads(raw)
def rep(s, old, new, n=1):
    assert s.count(old) == n, ("COUNT", s.count(old), old[:100]); return s.replace(old, new)
def dump(ref):
    for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2, ensure_ascii=False, sort_keys=True), dict(indent=2), dict(indent=2, sort_keys=True)):
        if json.dumps(json.loads(ref), **kw) + "\n" == ref: return lambda o: json.dumps(o, **kw) + "\n"
    raise SystemExit("json format not detected")
dj = dump(raw)
def sha(p): return hashlib.sha256((WT / p).read_bytes()).hexdigest()
def blob(p):
    d = (WT / p).read_bytes(); return hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
live = json.loads((WT / "frontiers/downstream_gate_20260925/GRAPH.json").read_text())["nodes"]
# ======================= RN_COUNT_INTERFACE_REVIEW.md (evidence labels; consumed premises) =======================
rv = rep(rv, "**Reviewer:** Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3`.",
"**Revision:** v1.3 (30 September 2026) corrects the evidence labels after OpenAI review 5360320845 item 2: the checks\n"
"that use floating point are now called numerical corroborations, the rational ones exact; and names the note's consumed\n"
"premises (item 3 there; OpenAI 5360327058 item 2). Verdicts unchanged.\n"
"**Reviewer:** Anthropic Claude, Claude Code session `session_017Mi3hxjaxV45x6zo6o1ee3`.")
rv = rep(rv, "no RN numerical certificate and no evaluation of the Gaussian triple integral are claimed by the note or accepted\nhere. Scientific effect: NONE.",
"no RN numerical certificate and no evaluation of the Gaussian triple integral are claimed by the note or accepted\nhere. Scientific effect: NONE.\n"
"**Promoted scope (v1.3):** (N1)–(N5) and the §3 counterexamples as self-contained probability statements; (N6)–(N7)\n"
"and §5 as formula-level implications conditional on the note's two consumed premises, the parent's regression law `Q`\n"
"with its §9 Borel-mark convention (live node `math.uniform-matrix-cap-lifetime`, read with its four reading-rule\n"
"components) and the cap support source of §5 (live node `math.d1-component.marked-cylinder-cap`, SHA256 `0bf922b9…`, the\n"
"digest the note quotes); both are required edges in `PROPOSED_TRANSITIONS.json`. Nothing quantitative.")
rv = rep(rv, "whose supremum over `t ≥ 1` is finite (the logarithmic derivative `−3 + p/(t+1)` is eventually negative). Checked exactly on `r = e^(−t)`, `t = 1, …, 12`, and the bound function's maximum located. |",
"whose supremum over `t ≥ 1` is finite (the logarithmic derivative `−3 + p/(t+1)` is eventually negative). The argument is the log-derivative; `INTERFACE` corroborates it numerically, in floating point, on `r = e^(−t)`, `t = 1, …, 12`, and on a grid against the located maximum (labelled `numerical_corroboration`, not exact). |")
rv = rep(rv, "The (N4) example has the uniform tail `P(N_r > t) ≤ e³ e^(−3t)` (`A = e³`, `B = 1/3`), so the logarithmic loss in (N5) is not removable from a tail hypothesis alone, as the note says. Both facts checked in `INTERFACE`. |",
"The (N4) example has the uniform tail `P(N_r > t) ≤ e³ e^(−3t)` (`A = e³`, `B = 1/3`), so the logarithmic loss in (N5) is not removable from a tail hypothesis alone, as the note says. The identity is proved by the antiderivative; `INTERFACE` corroborates it by floating-point Simpson quadrature at three parameter points and checks the tail comparison on a floating-point grid (numerical corroboration, not exact). |")
rv = rep(rv, "**ACCEPT as a formula at the note's stated level of rigour** (bounded continuous cylinder marks, extension of finite measures, truncation of `W_r`, monotone convergence; compact `B` off the pins, exhaustion). The rigorous statement of exactly this identity, with the kernel convention fixed, is Lemma 5.1 of `frontiers/c6_palm_route_20260929/PROOF.md` (Math-#145, reviewed), and its fixed-remote instance is the reviewed formula (12)–(14) of the D4 proof, accepted in main#76 comment 5841783172 items 2–3. Nothing quantitative is asserted by (N6) and nothing quantitative is accepted here. |",
"**ACCEPT as a formula at the note's stated level of rigour** (bounded continuous cylinder marks, extension of finite measures, truncation of `W_r`, monotone convergence; compact `B` off the pins, exhaustion), conditional on the note's consumed premise: the parent's regression law `Q` and its §9 Borel-mark convention (live node `math.uniform-matrix-cap-lifetime`, read with its four reading-rule components; a required edge in the proposal). Its fixed-remote instance is the reviewed formula (12)–(14) of the D4 proof, accepted in main#76 comment 5841783172 items 2–3. A later rigorous statement of the same identity with the kernel convention fixed is Lemma 5.1 of `frontiers/c6_palm_route_20260929/PROOF.md` (Math-#145); it is cited for the reader and is not consumed here. Nothing quantitative is asserted by (N6) and nothing quantitative is accepted here. |")
rv = rep(rv, "**ACCEPT as a scope statement**; it consumes the cap source `0bf922b9…` for the containment `{M, S}` on `G_r` and adds nothing to it. |",
"**ACCEPT as a scope statement**; it consumes the cap source `0bf922b9…` for the containment `{M, S}` on `G_r` (the live node `math.d1-component.marked-cylinder-cap` carries exactly that SHA256; a required edge in the proposal) and adds nothing to it. |")
rv = rep(rv, "**Overall: ACCEPT at the logical-interface scope.** The note proves exactly what it claims and no more: correct\ninequalities and counterexamples, and the form of the full-pin Kac–Rice expression that the fixed-remote theorem later\nevaluated. It is not an expected-count rate and is not treated as one.",
"**Overall: ACCEPT at the logical-interface scope.** The note proves exactly what it claims and no more: correct\ninequalities and counterexamples, and the form of the full-pin Kac–Rice expression that the fixed-remote theorem later\nevaluated. It is not an expected-count rate and is not treated as one. Evidence labels: the (N1)–(N3) and sharp-`p`\nchecks in `INTERFACE` are exact rational arithmetic; the (N4) grid, the (N5) quadrature and the tail grid are floating-point\nnumerical corroborations of arguments proved above, not exact checks.")
rv_p.write_text(rv)
# ======================= inventory additions =======================
NEW_INV = {
 "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md": ("079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d", "89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73"),
 "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md": ("04cf4c986f29ac7ac33b3cc87feac23f3cf494e0c1b3bf4476e4e6666de46574", "8988251631cab8e65eb2b01a08f7fc593b018570"),
 "frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md": ("c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9", "173881916ccd0e738bdb41279e835e78e520fcbc"),
 HERE + "/RN_COUNT_INTERFACE_REVIEW.md": (sha(HERE + "/RN_COUNT_INTERFACE_REVIEW.md"), blob(HERE + "/RN_COUNT_INTERFACE_REVIEW.md")),
 HERE + "/EXTERNAL_REVIEWS.md": (sha(HERE + "/EXTERNAL_REVIEWS.md"), blob(HERE + "/EXTERNAL_REVIEWS.md")),
}
for p, (s, b) in list(NEW_INV.items())[:3]:
    assert sha(p) == s and blob(p) == b, ("hash mismatch", p, sha(p), blob(p))
IFACE_REV_SHA = NEW_INV[HERE + "/RN_COUNT_INTERFACE_REVIEW.md"][0]
EXT_SHA = NEW_INV[HERE + "/EXTERNAL_REVIEWS.md"][0]
inv_tail = '    "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md": [\n        "9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7",\n        "dfed3b8d318a3ab1950957f393307733a4bef3f2"\n    ]\n}\n'
add = "".join('    "%s": [\n        "%s",\n        "%s"\n    ],\n' % (p, s, b) for p, (s, b) in NEW_INV.items())
py = rep(py, inv_tail, inv_tail[:-3].rstrip() + ",\n" + add.rstrip(",\n") + "\n}\n")
# ======================= VERDICTS additions =======================
py = rep(py, '        "| Cutoff `r^{1/24}` | **ACCEPT** |"],', '        "| Cutoff `r^{1/24}` | **ACCEPT** |",\n        "| Inner two-scale stitching | **ACCEPT** |",\n        "`TWO_SCALE_ADDENDUM.md` blob `89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73`, 15902 bytes"],')
py = rep(py, '        "| 1. Coordinatewise hazard transform, separate concavity, chord, `p=0,1` | **ACCEPT** |"],', '        "| 1. Coordinatewise hazard transform, separate concavity, chord, `p=0,1` | **ACCEPT** |",\n        "SHA256 `c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`. That is the digest named in the proof.",\n        "| 4. Full-block cover, same palette, realized family | **ACCEPT** |"],')
py = rep(py, '        "- D6 P15 full price theorem: proof [frontiers/full_price_20260924/PROOF.md]"],\n}',
'        "- D6 P15 full price theorem: proof [frontiers/full_price_20260924/PROOF.md]"],\n'
'    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md": ["**Object:** D5-TWO-SCALE-20260925-v1."],\n'
'    "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md": [\n'
'        "blob `89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73`, 15902 bytes, SHA256 `079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d`",\n'
'        "| S6 | **ACCEPT** |"],\n'
'    "frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md": ["**Object:** P15-REALIZED-COVERS-20260924-v1."],\n'
'    HERE + "/RN_COUNT_INTERFACE_REVIEW.md": [\n'
'        "**Overall: ACCEPT at the logical-interface scope.**",\n'
'        "numerical corroborations of arguments proved above, not exact checks"],\n'
'    HERE + "/EXTERNAL_REVIEWS.md": [\n'
'        "issuecomment-5841270276", "issuecomment-5841782206", "issuecomment-5841269490", "issuecomment-5841779222",\n'
'        "issuecomment-5841783172", "issuecomment-5841861362", "issuecomment-5842112010",\n'
'        "R5 and R6 are **ACCEPT**. Theorem R is accepted at its stated O(1) remainder scope.",\n'
'        "All six coefficient interfaces check out. The disposition is **COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN**.",\n'
'        "This accepts only the fixed-ρ and fixed-η interfaces below. It does not accept a full RN or 24-jet theorem.",\n'
'        "D6 ANALYTIC REVIEW COMPLETE",\n'
'        "An issue comment is a mutable external object"],\n}')
# ======================= mutants, docstring =======================
py = rep(py, '"skip-interface-premise", "holder-reversed", "baseline-drift", "duplicate-drift")', '"skip-interface-premise", "holder-reversed", "baseline-drift", "duplicate-drift", "region-unlinked",\n           "review-metadata-only")')
py = rep(py, "Checks: IDENTITIES, VERDICTS, BASELINE, TRANSITIONS, GATE, INTERFACE, NEGATIVES. No mathematics is re-proved; the\nproposed graph is replayed through the downstream hard gate's own validators. Nothing is written to the repository.",
"Checks: IDENTITIES, VERDICTS, BASELINE, TRANSITIONS, GATE, PROPAGATION, INTERFACE, NEGATIVES, D4_DUPLICATE. No mathematics\nis re-proved; the proposed graph is replayed through the downstream hard gate's own validators, and two-commit source\nmutations through its reverse-impact rule. Nothing is written to the repository.")
# ======================= apply / check_transitions / check_gate =======================
NEW_APPLY = '''def apply(spec, graph):
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
'''
NEW_TRANS = '''def check_transitions(root, spec):
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    ok = spec.get("declarative") is True and spec.get("executed") is False and spec.get("edges_unchanged") is False
    ids = [t["node"] for t in spec["transitions"]]
    ok &= len(ids) == len(set(ids)) and len(ids) >= 1
    for t in spec["transitions"]:
        p = t["proposed"]
        ok &= p["classification"] == "PROVED_REVIEWED" and p["controlling"] is False
        ok &= bool(p.get("scope")) and bool(p.get("explicit_limits")) and bool(p.get("notes"))
        ok &= len(p.get("review_sources", [])) >= 1
        if t["kind"] == "candidate":
            ok &= p.get("source") in INVENTORY and p["fingerprint"] == INVENTORY[p["source"]][0]
        else:
            ok &= t["kind"] == "region" and p.get("coverage_source") in ids
        ok &= not t["node"].startswith(("hist.", "eng.", "regional."))
    # component nodes: one per byte identity, fingerprinted to the inventory, PROVED_REVIEWED evidence records,
    # none for bytes the live graph already carries with a fingerprint (Codex 4139312869 on Math-#160)
    comps = spec["proposed_graph_nodes"]
    cids = [c["id"] for c in comps]
    live_fp_sources = {n.get("source") for n in graph["nodes"].values()
                       if isinstance(n, dict) and n.get("fingerprint") and n.get("source")}
    ok &= len(cids) == len(set(cids)) and not (set(cids) & set(graph["nodes"])) and len(comps) == spec["nodes_added"]
    ok &= len({c["source"] for c in comps}) == len(comps)
    for c in comps:
        ok &= c["kind"] == "reading_rule_component" and c["classification"] == "PROVED_REVIEWED" and c["controlling"] is False
        ok &= c["source"] in INVENTORY and c["fingerprint"] == INVENTORY[c["source"]][0]
        ok &= c["source"] not in live_fp_sources
        ok &= bool(c.get("component_role")) and bool(c.get("author_provider")) and len(c.get("review_basis", [])) >= 1
        consumers = c.get("components_of") or [c.get("component_of")]
        ok &= all(x in ids + cids for x in consumers) and len(consumers) >= 1
    edges_new = spec["proposed_graph_edges"]
    known = set(graph["nodes"]) | set(cids)
    live_keys = {(e["from"], e["to"], e["relation"]) for e in graph["edges"]}
    seen = set()
    for e in edges_new:
        ok &= {"from", "to", "required", "relation"} <= set(e) and type(e["required"]) is bool
        ok &= e["from"] in known and e["to"] in known and bool(e["relation"].strip())
        key = (e["from"], e["to"], e["relation"])
        ok &= key not in seen and key not in live_keys
        seen.add(key)
        ok &= e["from"] in ids or e["from"] in cids                 # every added edge leaves one of this record's objects
    ok &= len(edges_new) == spec["edges_added"]
    if not ok:
        return False, None
    new = apply(spec, graph)
    nodes = new["nodes"]
    edges = new["edges"]
    ok &= edges[:len(graph["edges"])] == graph["edges"] and len(edges) == len(graph["edges"]) + len(edges_new)
    ok &= all(graph["nodes"][k] == nodes[k] for k in graph["nodes"] if k not in ids)   # only the eight live nodes change
    for t in spec["transitions"]:
        nid = t["node"]
        req = [e["to"] for e in edges if e["from"] == nid and e["required"] is True]
        ok &= all(nodes[d]["classification"] == "PROVED_REVIEWED" for d in req)
        ok &= sorted(req) == sorted(t["required_premises_after"])
        if t["kind"] == "region":
            ok &= t["proposed"]["coverage_source"] in req              # a region depends on its covering node
            ok &= nodes[t["proposed"]["coverage_source"]]["classification"] == "PROVED_REVIEWED"
        n = nodes[nid]
        ok &= n.get("explicit_limits") == t["proposed"]["explicit_limits"] and n.get("notes") == t["proposed"]["notes"]
        ok &= "review open" not in n["notes"] and "candidate only" not in n["notes"]
        ok &= isinstance(n.get("review_basis"), list) and len(n["review_basis"]) >= 1
    for c in comps:                                                   # component nodes' own required premises
        req = [e["to"] for e in edges if e["from"] == c["id"] and e["required"] is True]
        ok &= all(nodes[d]["classification"] == "PROVED_REVIEWED" for d in req)
    return bool(ok), new
'''
py = re.sub(r"def apply\(spec, graph\):.*?(?=\ndef )", NEW_APPLY.rstrip("\n") + "\n", py, count=1, flags=re.S)
py = re.sub(r"def check_transitions\(root, spec\):.*?(?=\ndef )", NEW_TRANS.rstrip("\n") + "\n", py, count=1, flags=re.S)
py = rep(py, '    changed = sorted(t["node"] for t in spec["transitions"])',
          '    changed = sorted([t["node"] for t in spec["transitions"]] + [c["id"] for c in spec["proposed_graph_nodes"]])')
# ======================= propagation =======================
NEW_PROP = '''def snapshot(root, graph):
    """Source snapshot in the shape of git_transition_audit.read_snapshot, taken from the working tree."""
    out = {}
    for nid, node in graph["nodes"].items():
        ref = node.get("source")
        if ref is None:
            out[nid] = {"kind": "record_only"}
            continue
        if ref.startswith(("https://", "http://", "external:")):
            out[nid] = {"kind": "external_unresolved", "reference": ref}
            continue
        p = root / ref.rstrip("/")
        if p.is_file():
            data = p.read_bytes()
            out[nid] = {"kind": "blob", "reference": ref, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        elif p.is_dir():
            h = hashlib.sha256()
            for f in sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
                h.update(str(f.relative_to(p)).encode("utf-8") + b"\\0" + hashlib.sha256(f.read_bytes()).digest())
            out[nid] = {"kind": "tree", "reference": ref, "sha256": h.hexdigest()}
        else:
            out[nid] = {"kind": "missing", "reference": ref}
    return out


def check_propagation(root, spec, new):
    """Two-commit mutations on the proposed graph through the production loss-only rule (reverse_impact_between with
    source snapshots): evidence and consumed-source changes must reach the objects they support (OpenAI 5360320845,
    5360327058; Codex 4139807558, 4139807567). Nothing scientific is decided here."""
    if new is None:
        return False, {}
    hg = load_hard_gate(root)
    g = copy.deepcopy(new)
    if MUT == "region-unlinked":
        g["edges"] = [e for e in g["edges"] if e.get("relation") != "covered_by"]
    if MUT == "review-metadata-only":
        drop = {c["id"] for c in spec["proposed_graph_nodes"] if str(c.get("component_role", "")).startswith("review_record")}
        g["nodes"] = {k: v for k, v in g["nodes"].items() if k not in drop}
        g["edges"] = [e for e in g["edges"] if e["from"] not in drop and e["to"] not in drop]
    base = snapshot(root, g)
    by_source = {}
    for nid, n in g["nodes"].items():
        if n.get("source"):
            by_source.setdefault(n["source"], []).append(nid)

    def run(edit):
        g2, s2 = copy.deepcopy(g), copy.deepcopy(base)
        edit(g2, s2)
        try:
            return set(hg.reverse_impact_between(g, g2, old_sources=base, new_sources=s2)["impacted"])
        except Exception:
            return None

    def source_edit(path, kind="blob"):
        def f(g2, s2):
            for nid in by_source.get(path, []):
                s2[nid] = ({"kind": "blob", "reference": path, "bytes": 1, "sha256": "1" * 64} if kind == "blob"
                           else {"kind": "missing", "reference": path})
        return f

    def fp_edit(g2, s2):
        g2["nodes"][D4_NODE]["fingerprint"] = "2" * 64

    eight = {t["node"] for t in spec["transitions"]}
    cases = {
        "d4-proof-edit-reaches-region-and-consumer": (source_edit(D4), {D4_NODE, "math.rn-region.fixed-remote", "math.rn-mesoscopic-reduction"}),
        "annulus-proof-edit-reaches-region": (source_edit("frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md"),
                                              {"math.rn-fixed-annulus-window", "math.rn-region.fixed-annulus-window"}),
        "interface-review-deleted-reaches-interface-consumers-regions": (source_edit(HERE + "/RN_COUNT_INTERFACE_REVIEW.md", "missing"),
                                              {IFACE, D4_NODE, "math.rn-fixed-annulus-window", "math.rn-region.fixed-remote", "math.rn-region.fixed-annulus-window"}),
        "external-review-record-edit-reaches-d2-d3-d4-d6": (source_edit(HERE + "/EXTERNAL_REVIEWS.md"),
                                              {"math.lifetime-remainder", "math.side24-coefficient", D4_NODE, "math.p15-full-price", "math.rn-region.fixed-remote"}),
        "two-scale-addendum-edit-reaches-annulus-and-region": (source_edit("frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md"),
                                              {"math.rn-fixed-annulus-window", "math.rn-region.fixed-annulus-window"}),
        "realized-covers-edit-reaches-d6": (source_edit("frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md"), {"math.p15-full-price"}),
        "d4-fingerprint-edit-reaches-region": (fp_edit, {D4_NODE, "math.rn-region.fixed-remote"}),
    }
    results = {}
    for name, (edit, must) in cases.items():
        imp = run(edit)
        results[name] = imp is not None and must <= imp
    imp = run(source_edit("reviews/d5_collar_count_20260928/REVIEW.md"))
    results["unrelated-edit-leaves-the-eight-untouched"] = imp is not None and not (imp & eight)
    imp = run(lambda g2, s2: None)
    results["no-op-impacts-nothing"] = imp == set()
    return bool(all(results.values()) and len(results) == 9), results


def main():'''
py = rep(py, "def main():", NEW_PROP)
# ======================= interface labels =======================
py = rep(py, 'def check_interface():\n    """Exact finite content of RN_COUNT_INTERFACE_REVIEW.md."""\n    ok = True\n',
          'def check_interface():\n    """Finite content of RN_COUNT_INTERFACE_REVIEW.md: exact rational checks, and floating-point corroborations\n    labelled as such (OpenAI 5360320845 item 2)."""\n    ok = True\n    num = True\n')
py = rep(py, '    # (N4): r = e^-t, t = 1..12: E N / r^3 = ceil(t) increases; E N^p = ceil(t)^p e^-3t <= (t+1)^p e^-3t <= finite sup\n    for p in (1, 2, 4, 7):\n        vals = [math.ceil(t) ** p * math.exp(-3 * t) for t in range(1, 13)]\n        bound = [(t + 1) ** p * math.exp(-3 * t) for t in range(1, 13)]\n        ok &= all(v <= b + 1e-15 for v, b in zip(vals, bound))\n        tmax = max(1.0, p / 3 - 1)  # critical point of (t+1)^p e^{-3t}\n        sup = (tmax + 1) ** p * math.exp(-3 * tmax)\n        grid = [(t / 100 + 1) ** p * math.exp(-3 * (t / 100)) for t in range(100, 400000, 7)]\n        ok &= max(grid) <= sup + 1e-9 and math.isfinite(sup)\n        ok &= all(math.ceil(t) > math.ceil(t - 1) for t in range(2, 13))  # E N_r / r^3 -> infinity along the grid\n',
          '    # (N4), NUMERICAL (floating point): r = e^-t, t = 1..12: E N / r^3 = ceil(t) increases; ceil(t)^p e^-3t <= (t+1)^p e^-3t,\n    # and the grid maximum of (t+1)^p e^-3t stays below the located critical value. The proof is the log-derivative argument.\n    for p in (1, 2, 4, 7):\n        vals = [math.ceil(t) ** p * math.exp(-3 * t) for t in range(1, 13)]\n        bound = [(t + 1) ** p * math.exp(-3 * t) for t in range(1, 13)]\n        num &= all(v <= b + 1e-15 for v, b in zip(vals, bound))\n        tmax = max(1.0, p / 3 - 1)  # critical point of (t+1)^p e^{-3t}\n        sup = (tmax + 1) ** p * math.exp(-3 * tmax)\n        grid = [(t / 100 + 1) ** p * math.exp(-3 * (t / 100)) for t in range(100, 400000, 7)]\n        num &= max(grid) <= sup + 1e-9 and math.isfinite(sup)\n        ok &= all(math.ceil(t) > math.ceil(t - 1) for t in range(2, 13))  # E N_r / r^3 -> infinity along the grid (exact integers)\n')
py = rep(py, '    # (N5): int_0^inf min(q, A e^{-t/B}) dt = B q (1 + log(A/q)), q <= A; numeric quadrature at three points\n',
          '    # (N5), NUMERICAL (floating point): int_0^inf min(q, A e^{-t/B}) dt = B q (1 + log(A/q)), q <= A; Simpson quadrature at\n    # three parameter points corroborates the antiderivative identity proved in the review\n')
py = rep(py, "        ok &= abs(num - B * q * (1 + math.log(A / q))) <= 1e-6 * B * q * (1 + math.log(A / q))\n",
          "        num &= abs(quad - B * q * (1 + math.log(A / q))) <= 1e-6 * B * q * (1 + math.log(A / q))\n")
py = rep(py, "        num = s * h / 3\n", "        quad = s * h / 3\n")
py = rep(py, '    # uniform tail of the (N4) example: P(N_r > t) = r^3 1{t < ceil(log 1/r)} <= e^3 e^{-3t}\n    for s_ in range(1, 15):\n        r3 = math.exp(-3 * s_)\n        for t10 in range(0, 10 * s_ + 30):\n            t = t10 / 10\n            p_tail = r3 if t < math.ceil(s_) else 0.0\n            ok &= le(p_tail, math.exp(3) * math.exp(-3 * t) + 1e-300)\n    return bool(ok)',
          '    # uniform tail of the (N4) example, NUMERICAL (floating point): P(N_r > t) = r^3 1{t < ceil(log 1/r)} <= e^3 e^{-3t}\n    for s_ in range(1, 15):\n        r3 = math.exp(-3 * s_)\n        for t10 in range(0, 10 * s_ + 30):\n            t = t10 / 10\n            p_tail = r3 if t < math.ceil(s_) else 0.0\n            num &= le(p_tail, math.exp(3) * math.exp(-3 * t) + 1e-300)\n    return {"exact": bool(ok), "numerical_corroboration": bool(num)}')
# ======================= main =======================
NEW_MAIN = '''def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    MUT = ap.parse_args().mutant
    root = pathlib.Path(".").resolve()
    spec = load_spec(root)
    ident = identities_ok(root)
    trans_ok, new = check_transitions(root, spec)
    gate_ok, gate = check_gate(root, spec, new)
    prop_ok, prop = check_propagation(root, spec, new)
    iface = check_interface()
    checks = {"IDENTITIES": ident, "VERDICTS": ident and check_verdicts(root), "BASELINE": check_baseline(root, spec),
              "TRANSITIONS": trans_ok, "GATE": gate_ok, "PROPAGATION": prop_ok,
              "INTERFACE": iface["exact"] and iface["numerical_corroboration"], "NEGATIVES": check_negatives(root),
              "D4_DUPLICATE": check_d4_duplicate(root, spec)}
    passed = all(checks.values()) and len(checks) == 9
    print(json.dumps({"object": "REGISTER-ALIGNMENT-20260930-v1", "checks": checks, "passed": passed,
                      "inventory_files": len(INVENTORY), "transitions": len(spec["transitions"]),
                      "component_nodes": len(spec["proposed_graph_nodes"]), "edges_added": len(spec["proposed_graph_edges"]),
                      "gate": gate, "propagation": prop,
                      "interface_evidence": {"exact_rational": ["(N1) Holder p=2,3 on finite rational spaces", "(N2) exponent identity",
                                                                "(N3) conditional-mean identity", "sharp-p example at r=2^-pm",
                                                                "(N4) integer monotonicity of ceil(t)"],
                                             "numerical_corroboration_floating_point": ["(N4) grid maximum of (t+1)^p e^-3t",
                                                                                        "(N5) Simpson quadrature at three parameter points",
                                                                                        "(N4) example uniform tail on a grid"],
                                             "results": iface},
                      "scope": "identity, verdict-row, live-baseline, transition-shape, hard-gate replay, two-commit propagation, "
                               "interface arithmetic (exact and numerical parts labelled), filesystem-negative and live D4-duplicate "
                               "checks; no mathematics is re-proved; nothing is written"}, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
'''
py = re.sub(r"def main\(\):.*", NEW_MAIN, py, count=1, flags=re.S)
py_p.write_text(py)
# ======================= PROPOSED_TRANSITIONS.json =======================
T = {t["node"]: t for t in js["transitions"]}
def layer(nid): return live[nid].get("layer", "D5")
EXT = "math.align-component.external-issue-reviews"
comps = [
 {"id": "math.align-component.rn-interface-review", "layer": layer("math.rn-count-interface"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.rn-count-interface", "component_role": "review_record", "author_provider": "Anthropic (nonauthor; session 017Mi3hx)",
  "fingerprint": IFACE_REV_SHA, "source": HERE + "/RN_COUNT_INTERFACE_REVIEW.md",
  "scope": "ACCEPT at the logical-interface scope: (N1)-(N5) and the section 3 counterexamples self-contained; (N6)-(N7) and section 5 as formula-level implications conditional on the parent regression law / section 9 convention and the cap support source",
  "review_basis": [{"review": HERE + "/RN_COUNT_INTERFACE_REVIEW.md", "provider": "Anthropic", "verdict": "ACCEPT at the logical-interface scope", "depth": "full at that scope; exact rational checks plus labelled floating-point corroborations in alignment_check.py"}],
  "notes": "This packet's own review record; a nonauthor read by a second provider was not available at v1.3 (xAI 5360240245 did not re-read the note)."},
 {"id": EXT, "layer": "D2", "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": None, "components_of": ["math.lifetime-remainder", "math.side24-coefficient", "math.rn-fixed-remote-window", "math.p15-full-price"],
  "component_role": "review_record_external", "author_provider": "xAI/Grok 4.7 via Cursor (posted by cursor[bot]) and owner reconciliations; transcribed by Anthropic from the GitHub API read-back of 2026-09-30",
  "fingerprint": EXT_SHA, "source": HERE + "/EXTERNAL_REVIEWS.md",
  "scope": "seven main-repository issue comments: main#67 5841270276 and 5841782206 (D2), main#65 5841269490 and 5841779222 (D3), main#76 5841783172 and 5841861362 (D4), main#74 5842112010 (D6), with comment id, URL, posting account, created_at, updated_at and body",
  "review_basis": [
    {"review": "main#67 comment 5841270276", "provider": "xAI/Grok 4.7 via Cursor (session bc-81faa729)", "verdict": "R1-R3 ACCEPT, R4 ACCEPT for (R12), R5-R6 IMPORTED-OPEN", "created_at": "2026-09-25T23:59:05Z", "updated_at": "2026-09-26T00:07:25Z"},
    {"review": "main#67 comment 5841782206", "provider": "xAI/Grok 4.7 via Cursor (session bc-dfbce246)", "verdict": "R5 and R6 ACCEPT; Theorem R accepted at its stated O(1) remainder scope", "created_at": "2026-09-26T01:04:21Z", "updated_at": "2026-09-26T01:07:56Z"},
    {"review": "main#65 comment 5841269490", "provider": "xAI/Grok 4.7 via Cursor (run bc-be5dc8d9)", "verdict": "C1-C6 VERIFIED; COEFFICIENT-CALC-REVIEWED / PARENT-IMPORTED-OPEN", "created_at": "2026-09-25T23:58:59Z", "updated_at": "2026-09-26T00:09:32Z"},
    {"review": "main#65 comment 5841779222", "provider": "owner", "verdict": "reconciliation; #65 closed as the coefficient-review work package", "created_at": "2026-09-26T01:03:54Z"},
    {"review": "main#76 comment 5841783172", "provider": "xAI/Grok via Cursor (session bc-593a8246; provider named by the owner's reconciliation)", "verdict": "items 1-7 ACCEPT at fixed-rho/fixed-eta scope; not a full RN or 24-jet theorem", "created_at": "2026-09-26T01:04:29Z", "updated_at": "2026-09-26T01:12:36Z"},
    {"review": "main#76 comment 5841861362", "provider": "owner", "verdict": "reconciliation; ACCEPT on all seven interfaces; #76 closed as the fixed-remote REVIEW work package only", "created_at": "2026-09-26T01:15:52Z"},
    {"review": "main#74 comment 5842112010", "provider": "owner", "verdict": "D6 ANALYTIC REVIEW COMPLETE; cross-provider nonauthor ACCEPT on all six slices (the review itself is reviews/p15_full_price_nonauthor_20260926/REVIEW.md); #74 closed", "created_at": "2026-09-26T01:48:31Z"}],
  "authentication": "mutable external evidence, read back through the session's GitHub connector and transcribed; not byte-authenticated offline (stdlib checker, no network); OpenAI 5360327058 reports direct authentication of 5841270276, 5841782206, 5841269490 and 5841783172; claims/LANDING_CLAIMS.json cites the same issues as review objects",
  "notes": "One node for one byte object (this file); loss-only: any edit to the record revalidates D2, D3, D4 and D6."},
 {"id": "math.align-component.side24-review-claude", "layer": layer("math.side24-coefficient"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.side24-coefficient", "component_role": "review_record", "author_provider": "Anthropic (session 01NMeKE)",
  "fingerprint": "258c414c46be93142ee552cfa4c32c03dd89b6bf7b0b470334f152e1e0a57eef", "source": "reviews/side24_v1_coefficient_claude_20260929/REVIEW.md",
  "scope": "second-provider read of coefficients/side24_v1/PROOF.md: ACCEPT at the arithmetic-enclosure scope",
  "review_basis": [{"review": "reviews/side24_v1_coefficient_claude_20260929/REVIEW.md", "provider": "Anthropic", "verdict": "ACCEPT at the arithmetic-enclosure scope", "depth": "full at that scope"}],
  "notes": "Makes D3 the two-provider row (with main#65 5841269490)."},
 {"id": "math.align-component.pr22-annulus-review", "layer": layer("math.rn-fixed-annulus-window"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.rn-fixed-annulus-window", "component_role": "review_record", "author_provider": "xAI/Grok 4.7 via Cursor",
  "fingerprint": "29a0c6d00ce739baf1d29a3163977d8e707e2079e216fba4fee3669cebec127d", "source": "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md",
  "scope": "ACCEPT of FIXED_ANNULUS_CANDIDATE.md (blob 081abc13, SHA256 1fd9fe71) at the exact scoped theorem: covariance floor, cutoff r^{1/24}, rare target, conditioned C^3 sixth moment, inner two-scale stitching; PR28 and later blobs excluded",
  "review_basis": [{"review": "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "ACCEPT at the exact scoped theorem", "depth": "full"}],
  "notes": "Single nonauthor provider."},
 {"id": "math.align-component.two-scale-addendum", "layer": layer("math.rn-fixed-annulus-window"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.rn-fixed-annulus-window", "component_role": "consumed_proof", "author_provider": "OpenAI",
  "fingerprint": "079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d", "source": "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
  "scope": "S6-S21 and (S3)-(S5): the inner two-scale stitching input (S4) consumed by FIXED_ANNULUS_CANDIDATE.md for |t| < h_r on A <= |u| <= B (claims/LANDING_CLAIMS.json dependency two-scale-s6-s21)",
  "review_basis": [{"review": "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "ACCEPT S6-S21 and (S3)-(S5) at blob 89cae3a9", "depth": "full"},
                   {"review": "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "inner two-scale stitching ACCEPT; the addendum read as the imported inner bound, not re-derived", "depth": "gluing step"}],
  "notes": "The addendum's own review record is the next node; the addendum requires it."},
 {"id": "math.align-component.two-scale-review", "layer": layer("math.rn-fixed-annulus-window"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.align-component.two-scale-addendum", "component_role": "review_record", "author_provider": "xAI/Grok 4.7 via Cursor (session bc-ff620630)",
  "fingerprint": "04cf4c986f29ac7ac33b3cc87feac23f3cf494e0c1b3bf4476e4e6666de46574", "source": "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",
  "scope": "ACCEPT of TWO_SCALE_ADDENDUM.md S6-S21 and (S3)-(S5) at blob 89cae3a9 (Math- PR34, commit bc7d7545)",
  "review_basis": [{"review": "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "ACCEPT", "depth": "full"}],
  "notes": "Single nonauthor provider."},
 {"id": "math.align-component.p15-review", "layer": layer("math.p15-full-price"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.p15-full-price", "component_role": "review_record", "author_provider": "xAI/Grok 4.7 via Cursor (Math- issue 57 lane)",
  "fingerprint": "691ea0db5c0f073326c9f7208c9804c6f5d0daf92ba7467529c9d57e08b15eb7", "source": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",
  "scope": "ACCEPT of Theorem F on the realized disjoint capacity/clutter family (six slices) at blob 582180e4, SHA256 87521901; checked the setwise (P1), (P2), Theorem P2 dependency of P15_REALIZED_COVERS.md",
  "review_basis": [{"review": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "ACCEPT Theorem F on the realized family", "depth": "full"}],
  "notes": "Single nonauthor provider; owner reconciliation main#74 5842112010 is in the external record."},
 {"id": "math.align-component.p15-realized-covers-setwise", "layer": layer("math.p15-full-price"), "kind": "reading_rule_component", "classification": "PROVED_REVIEWED", "controlling": False,
  "component_of": "math.p15-full-price", "component_role": "consumed_interface", "author_provider": "OpenAI",
  "fingerprint": "c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9", "source": "frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md",
  "scope": "setwise interfaces only: realized-family definition (P1), actual local restrictions (P2) and Theorem P2 full-block cover at K >= K_H(d), as consumed by Theorem F with d_i >= 2 (claims/LANDING_CLAIMS.json dependency p15-realized-covers:setwise-interfaces); the note's c_v <= p_v price theorem and any broader P15 statement are not included",
  "review_basis": [{"review": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md", "provider": "xAI/Grok 4.7 via Cursor", "verdict": "read as the cited setwise dependency and checked for the local restriction and the full-block cover; the c_v <= p_v price theorem not accepted", "depth": "setwise interfaces"}],
  "notes": "The realized-covers claim itself stays HOLD_WITH_DOMAIN in the claim manifest; only the consumed setwise interfaces are represented here, at the D6 review's scope."},
]
edges = [
 {"from": "math.rn-count-interface", "to": "math.align-component.rn-interface-review", "required": True, "relation": "requires_evidence"},
 {"from": "math.rn-count-interface", "to": "math.uniform-matrix-cap-lifetime", "required": True, "relation": "uses_parent_regression_law_and_section9_convention"},
 {"from": "math.rn-count-interface", "to": "math.d1-component.marked-cylinder-cap", "required": True, "relation": "consumes_cap_support_scope_section5"},
 {"from": "math.lifetime-remainder", "to": EXT, "required": True, "relation": "requires_evidence"},
 {"from": "math.side24-coefficient", "to": EXT, "required": True, "relation": "requires_evidence"},
 {"from": "math.side24-coefficient", "to": "math.align-component.side24-review-claude", "required": True, "relation": "requires_evidence"},
 {"from": "math.rn-fixed-remote-window", "to": EXT, "required": True, "relation": "requires_evidence"},
 {"from": "math.rn-region.fixed-remote", "to": "math.rn-fixed-remote-window", "required": True, "relation": "covered_by"},
 {"from": "math.rn-fixed-annulus-window", "to": "math.align-component.pr22-annulus-review", "required": True, "relation": "requires_evidence"},
 {"from": "math.rn-fixed-annulus-window", "to": "math.align-component.two-scale-addendum", "required": True, "relation": "consumes_s6_s21_stitching_input"},
 {"from": "math.align-component.two-scale-addendum", "to": "math.align-component.two-scale-review", "required": True, "relation": "requires_evidence"},
 {"from": "math.rn-region.fixed-annulus-window", "to": "math.rn-fixed-annulus-window", "required": True, "relation": "covered_by"},
 {"from": "math.p15-full-price", "to": EXT, "required": True, "relation": "requires_evidence"},
 {"from": "math.p15-full-price", "to": "math.align-component.p15-review", "required": True, "relation": "requires_evidence"},
 {"from": "math.p15-full-price", "to": "math.align-component.p15-realized-covers-setwise", "required": True, "relation": "consumes_setwise_interfaces_P1_P2_TheoremP2"},
 {"from": "math.align-component.p15-realized-covers-setwise", "to": "math.align-component.p15-review", "required": True, "relation": "requires_evidence"},
]
js["proposed_graph_nodes"] = comps; js["proposed_graph_edges"] = edges
js["nodes_added"] = len(comps); js["edges_added"] = len(edges); js["edges_unchanged"] = False
js["edges_removed"] = 0
REQ = {
 "math.rn-count-interface": ["math.align-component.rn-interface-review", "math.uniform-matrix-cap-lifetime", "math.d1-component.marked-cylinder-cap"],
 "math.lifetime-remainder": ["math.uniform-matrix-cap-lifetime", EXT],
 "math.side24-coefficient": ["math.uniform-matrix-cap-lifetime", EXT, "math.align-component.side24-review-claude"],
 "math.rn-fixed-remote-window": ["math.rn-count-interface", EXT],
 "math.rn-region.fixed-remote": ["math.rn-fixed-remote-window"],
 "math.rn-fixed-annulus-window": ["math.rn-count-interface", "math.align-component.pr22-annulus-review", "math.align-component.two-scale-addendum"],
 "math.rn-region.fixed-annulus-window": ["math.rn-fixed-annulus-window"],
 "math.p15-full-price": [EXT, "math.align-component.p15-review", "math.align-component.p15-realized-covers-setwise"],
}
NOTES = {
 "math.rn-count-interface": "PROVED_REVIEWED at the logical-interface scope stated in scope; see explicit_limits. Evidence: this packet's review record; premises: the live D1 node and cap component. Register alignment v1.3.",
 "math.lifetime-remainder": "PROVED_REVIEWED at Theorem R's existential O(1)-remainder scope; see explicit_limits. Evidence: main#67 comments (external record node). Single nonauthor provider. Register alignment v1.3.",
 "math.side24-coefficient": "PROVED_REVIEWED at the arithmetic-enclosure scope for d = 2, 3; see explicit_limits. Evidence: main#65 comment (external record node) and the in-repo Anthropic review; two providers. Register alignment v1.3.",
 "math.rn-fixed-remote-window": "PROVED_REVIEWED at fixed rho, fixed eta, between-pin window; see explicit_limits. Evidence: main#76 comments (external record node). Same bytes as the live math.d5-component.remote-window-proof. Single nonauthor provider. Register alignment v1.3.",
 "math.rn-region.fixed-remote": "PROVED_REVIEWED region: covered by math.rn-fixed-remote-window (required covered_by edge; coverage_source retained). Register alignment v1.3.",
 "math.rn-fixed-annulus-window": "PROVED_REVIEWED at the fixed d = 2 scaled-annulus window scope; see explicit_limits. Evidence: pr22 review record; consumed S6-S21 stitching input with its review. Single nonauthor provider. Register alignment v1.3.",
 "math.rn-region.fixed-annulus-window": "PROVED_REVIEWED region: covered by math.rn-fixed-annulus-window (required covered_by edge; coverage_source retained). Register alignment v1.3.",
 "math.p15-full-price": "PROVED_REVIEWED on the realized family with d_i >= 2; see explicit_limits. Evidence: p15 review record, consumed setwise interfaces of P15_REALIZED_COVERS.md, main#74 reconciliation (external record node). Single nonauthor provider. Register alignment v1.3.",
}
for nid, t in T.items():
    t["required_premises_after"] = REQ[nid]
    t["proposed"]["notes"] = NOTES[nid]
i = T["math.rn-count-interface"]["proposed"]
i["scope"] = ("(N1)-(N5) and the section 3 counterexamples as self-contained probability statements; (N6)-(N7) and section 5 as formula-level "
              "implications conditional on the parent's regression law Q and section 9 Borel-mark convention (live D1 node) and on the cap support "
              "source (live cap component); nothing quantitative")
i["explicit_limits"] = ("not an expected-count rate; no RN or 24-jet certificate; no evaluation of the Gaussian triple integral; (N6) is a formula, "
                        "not a numerical RN/JETMOD certificate; the C6 Palm route's Lemma 5.1 is cited for the reader and not consumed")
T["math.rn-count-interface"]["notes"] = rep(T["math.rn-count-interface"]["notes"], "Unreviewed until this packet;", "Unreviewed until this packet (its review record is now a component node);")
T["math.rn-region.fixed-remote"]["notes"] = ("Covered by a PROVED_REVIEWED node after the previous transition; coverage_source retained and a required covered_by edge added "
    "so that a change to the covering proof reaches the region (Codex 4139807558; OpenAI 5360327058 item 1). The current hard gate reports PROVED_REVIEWED regions in its "
    "own proved_reviewed_regions bucket since ec6db8c (Math-#151), so the earlier region-reporting complaint (Codex 4139807545) is superseded.")
T["math.rn-region.fixed-annulus-window"]["notes"] = "Covered by a PROVED_REVIEWED node after the previous transition; coverage_source retained and a required covered_by edge added (as for the fixed-remote region)."
T["math.rn-fixed-annulus-window"]["notes"] += " Its consumed S6-S21 stitching input (TWO_SCALE_ADDENDUM.md) and that input's review are component nodes since v1.3 (Codex 4139807567; OpenAI 5360327058 item 2)."
T["math.p15-full-price"]["notes"] += " The consumed setwise interfaces (P1), (P2), Theorem P2 of P15_REALIZED_COVERS.md are a component node since v1.3, at the D6 review's scope only."
js["external_evidence"] = {"file": HERE + "/EXTERNAL_REVIEWS.md", "sha256": EXT_SHA, "node": EXT,
    "status": "mutable external evidence, read back on 2026-09-30 through the session's GitHub connector and transcribed with comment id, URL, posting account, created_at, updated_at and body (Cursor footer HTML omitted); not byte-authenticated offline; OpenAI 5360327058 reports direct authentication of 5841270276, 5841782206, 5841269490 and 5841783172; this is not an additional review and not an additional provider"}
js["execution_order"] = ["0. create the component nodes and edges of proposed_graph_nodes / proposed_graph_edges (review records, consumed sources, covered_by, the interface's two live premises); they are evidence records and change no classification of an existing node"] + js["execution_order"]
js["review_record"] += [
 {"lane": "OpenAI/ChatGPT", "id": 5360320845, "kind": "source-custody and RN-interface review", "at": "f2250bf",
  "verdict": "historical lag and logical-interface inequalities sound at stated scopes; AMEND REQUIRED before the proposal is represented as a complete executable evidence-bound fold: (1) review bytes not carried by the graph, production read_snapshot reads node.source only; (2) N4/N5 checks are floating, relabel; (3) the interface is not a quantitative rate; N6 may be recorded via the canonical-kernel interface",
  "answered_by": "v1.3: review records and consumed sources as fingerprinted component nodes with required edges; external comments preserved in EXTERNAL_REVIEWS.md and labelled mutable; PROPAGATION two-commit cases; INTERFACE exact/numerical labels; interface scope narrowed with its premises as edges"},
 {"lane": "OpenAI/Codex (delegated)", "id": 5360327058, "kind": "engineering/interface review", "at": "f2250bf",
  "verdict": "ACCEPT the bounded historical inventory; AMEND the executable proposal; HOLD execution: (1) covering-proof loss does not reach the promoted regions; (2) TWO_SCALE_ADDENDUM, P15_REALIZED_COVERS, the interface's consumed C6-Palm/cap items are not tracked; (3) apply() drops explicit_limits and keeps stale notes; region-reporting complaint superseded on main; D2/D3/D4 comments directly authenticated",
  "answered_by": "v1.3: covered_by edges; consumed-source component nodes; interface premises as edges to the live D1 node and cap component (Lemma 5.1 cited, not consumed); apply() carries explicit_limits, replacement notes and review_basis; PROPAGATION regressions; the superseded complaint recorded, not replayed"},
 {"lane": "OpenAI/Codex (bot review 5360253242)", "id": "4139807545/4139807551/4139807558/4139807562/4139807567", "kind": "review threads", "at": "670e4ae",
  "verdict": "region reporting (superseded on main); stale notes on apply; region-to-covering-proof dependency; authenticate external verdicts; track consumed source dependencies",
  "answered_by": "v1.3 (all five); 4139807545 by Math-#151's hard_gate.py proved_reviewed_regions bucket, recorded in the region transition notes"}]
js["revision"] = {"version": "v1.3", "applies": js["revision"]["applies"] + [
  "OpenAI 5360320845 and 5360327058 and Codex threads 4139807545-67: eight review-record / consumed-source component nodes and sixteen required edges (edges_unchanged false); covered_by edges from regions; interface premises as edges to the live D1 node and cap component and its scope narrowed; apply() carries explicit_limits, replacement notes, review_basis; PROPAGATION check with nine two-commit cases and mutants region-unlinked / review-metadata-only; INTERFACE exact versus numerical labels; RN_COUNT_INTERFACE_REVIEW.md evidence labels; EXTERNAL_REVIEWS.md preserved and labelled mutable"],
  "unchanged": "the eight classifications and every scope exclusion; the ten original inventory identities"}
js_p.write_text(dj(js))
# ======================= RECONCILIATION.md =======================
md = rep(md, "node (`D4_DUPLICATE` check, `duplicate-drift` mutant) and records that the HOLD's Math-#151 precondition is met;\ntransitions unchanged.",
"node (`D4_DUPLICATE` check, `duplicate-drift` mutant) and records that the HOLD's Math-#151 precondition is met;\ntransitions unchanged. v1.3 answers the two OpenAI AMEND reviews (5360320845, 5360327058) and the five Codex threads\n(4139807545–67): review records and consumed sources become fingerprinted component nodes with required edges, regions\ngain `covered_by` edges, `apply()` carries scope exclusions and replacement notes, a nine-case `PROPAGATION` check replays\ntwo-commit source mutations through the production reverse-impact rule, the interface's promoted scope is narrowed to\nself-contained inequalities plus conditional implications with its consumed premises as edges, the interface evidence\nlabels separate exact from floating-point checks, and the external issue comments are preserved in `EXTERNAL_REVIEWS.md`\nand labelled as mutable evidence; the eight classifications are unchanged.")
md = rep(md, "lane record the single-provider rows as such, is already stated per row in §4 and in `PROPOSED_TRANSITIONS.json` and is\nthat lane's to carry out.",
"lane record the single-provider rows as such, is already stated per row in §4 and in `PROPOSED_TRANSITIONS.json` and is\nthat lane's to carry out.\n\n"
"OpenAI source-custody / RN-interface review 5360320845 and OpenAI-Codex engineering review 5360327058 (both at `f2250bf`,\n"
"with Codex threads 4139807545/51/58/62/67 at `670e4ae`): **the historical classification lag and the logical-interface\n"
"inequalities are sound at their stated scopes; AMEND the executable proposal; HOLD execution.** Their findings: the\n"
"acceptance evidence lived only in `review_sources` metadata, which the production loss-only audit\n"
"(`git_transition_audit.read_snapshot`, `node.source` only) never sees; a covering-proof change did not reach the promoted\n"
"regions; the annulus's `TWO_SCALE_ADDENDUM.md`, D6's `P15_REALIZED_COVERS.md` and the interface's consumed cap / parent\n"
"items were not tracked; `apply()` dropped `explicit_limits` and kept stale notes; the (N4)/(N5) checks were floating\n"
"point while labelled exact; external issue-comment verdicts were quoted, not preserved. v1.3 answers each (§4, §6; the\n"
"eight classifications and scope exclusions unchanged). The region-reporting complaint (4139807545) is superseded on main\n"
"since `ec6db8c` (`hard_gate.d4_region_complement` has a `proved_reviewed_regions` bucket) and is not replayed. Neither\n"
"OpenAI read is a new mathematical acceptance of the six sources; both keep the single-provider caveat.")
md = rep(md, "All six sources are unchanged since their landing commits (`git log` on each path shows one commit). Identities are\nchecked by `alignment_check.py` (`IDENTITIES`); the quoted verdict lines are checked as substrings where the review is\nin this repository (`VERDICTS`).",
"All six sources, and the two consumed sources and the addendum's review added in v1.3, are unchanged since their landing\ncommits. Identities of all fifteen inventoried files (six sources, the D1 parent, four in-repo reviews, two consumed\nsources, this packet's two review records) are checked by `alignment_check.py` (`IDENTITIES`); the quoted verdict lines\nare checked as substrings where the review is in this repository (`VERDICTS`). The issue-comment reviews are preserved\nas a transcribed read-back in `EXTERNAL_REVIEWS.md` (comment id, URL, posting account, timestamps, body) and are\nlabelled there as mutable external evidence that an offline check cannot re-authenticate; OpenAI 5360327058 reports\nhaving directly authenticated the D2, D3 and D4 comments.")
d1row = [l for l in md.splitlines() if l.startswith("| D1 parent (premise of D2, D3) |")]
assert len(d1row) == 1
md = rep(md, d1row[0], d1row[0] + "\n"
"| consumed by D5 annulus | `frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md`, 15902 B, SHA256 `079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d`, blob `89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73` | `reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md` (10073 B, SHA256 `04cf4c98…`, blob `89882516`): \"blob `89cae3a9…`, 15902 bytes, SHA256 `079f9399…`\"; the pr22 review: \"Read as the imported inner bound, not re-derived: `TWO_SCALE_ADDENDUM.md` blob `89cae3a9…`\" | S6–S21 and (S3)–(S5) **ACCEPT** (xAI, Math- PR34); pr22 row \"Inner two-scale stitching **ACCEPT**\". Component nodes `…two-scale-addendum`, `…two-scale-review` (§4). |\n"
"| consumed by D6 | `frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md`, 11467 B, SHA256 `c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`, blob `173881916ccd0e738bdb41279e835e78e520fcbc` | the p15 review: \"Read as the cited setwise dependency, and checked for the local restriction and the full-block cover: … SHA256 `c0dbb821…`. That is the digest named in the proof. The check uses (P1), (P2), and Theorem P2 there. It does not accept that note's `c_v<=p_v` price theorem\" | setwise interfaces only, at the D6 review's scope. Component node `…p15-realized-covers-setwise` (§4). |\n"
"| external comments (D2, D3, D4, D6) | `reviews/register_alignment_20260930/EXTERNAL_REVIEWS.md` (this packet) | seven comments with id, account, `created_at`, `updated_at`, body; the four `cursor[bot]` reviews and three owner reconciliations | preserved read-back; mutable external evidence, labelled. Component node `…external-issue-reviews` (§4). |")
md = re.sub(r"The finite parts are checked exactly \(`INTERFACE`\):.*?the D4 formula \(12\)–\(14\)\.",
"The finite parts are checked in `INTERFACE` in two labelled parts: exact rational arithmetic (Hölder on finite rational\n"
"distributions for `p = 2, 3`; the exponent identity of (N2); the (N3) identity; the single-`p` sharp example at\n"
"`r = 2^(−pm)`, where `E N^p = 1` and `E N = r^(3−3/p)` exactly) and floating-point numerical corroborations of arguments\n"
"proved in the review (the (N4) example's moments and the located maximum of `(t+1)^p e^(−3t)`; the identity\n"
"`∫ min(q, A e^(−t/B)) dt = B q [1 + log(A/q)]` by Simpson quadrature; the uniform tail `P(N_r > t) ≤ e³ e^(−3t)` of the\n"
"(N4) example on a grid); OpenAI 5360320845 item 2 asked for exactly this separation. The note's consumed premises are the\n"
"parent's regression law and §9 Borel-mark convention (live node `math.uniform-matrix-cap-lifetime`, with its four\n"
"reading-rule components) and the cap support source of §5 (live node `math.d1-component.marked-cylinder-cap`, SHA256\n"
"`0bf922b9…`, the digest the note quotes); both are required edges of the proposal, and the promoted scope is narrowed to\n"
"(N1)–(N5) with the counterexamples as self-contained statements and (N6)–(N7), §5 as conditional implications at formula\n"
"level. A later rigorous statement of (N6) is Lemma 5.1 of `frontiers/c6_palm_route_20260929/PROOF.md` (Math-#145),\n"
"cited for the reader and not consumed.", md, count=1, flags=re.S)
SEC4 = open("<SCRATCH>/sec4.md").read()
SEC6 = open("<SCRATCH>/sec6.md").read()
md = re.sub(r"## 4\. Proposed transitions\n.*?(?=## 5\. What is not proposed)", SEC4, md, count=1, flags=re.S)
md = re.sub(r"## 6\. Checks\n.*?(?=## 7\. Relation to other lanes)", SEC6, md, count=1, flags=re.S)
md = rep(md, "- **Math-#160 (this session, C6 reconciliation):** already references the live D4 node with supporting edges; nothing\n  there changes.",
"- **Math-#160 (this session, C6 reconciliation):** already references the live D4 node with supporting edges; nothing\n  there changes.\n"
"- **OpenAI reviews 5360320845 and 5360327058; Codex review 5360253242 (threads 4139807545–67):** answered by v1.3 (§0,\n  §4, §6); the region-reporting thread is superseded on main since `ec6db8c` and not replayed. The single-provider caveat\n  and the HOLD on execution stand until the executing lane decides.")
md_p.write_text(md)
# ======================= workflow =======================
wf = rep(wf, "'holder-reversed', 'baseline-drift', 'duplicate-drift']", "'holder-reversed', 'baseline-drift', 'duplicate-drift', 'region-unlinked', 'review-metadata-only']")
wf = rep(wf, "      - 'reviews/side24_v1_coefficient_claude_20260929/**'\n",
"      - 'reviews/side24_v1_coefficient_claude_20260929/**'\n      - 'frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md'\n      - 'reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md'\n      - 'frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md'\n")
wf_p.write_text(wf)
print("v1.3 edits written; iface review sha", IFACE_REV_SHA[:12], "external sha", EXT_SHA[:12])
