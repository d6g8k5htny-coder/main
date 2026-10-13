import json, pathlib, re
P = pathlib.Path("reviews/c6_witness_collision_reconciliation_20260929")
LIVE_LP = "math.uniform-matrix-cap-lifetime"
LP = "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md"
CATALOG = "reviews/candidates_pending_20260928/CANDIDATES.md"
# ------------------------------------------------------------------ RECONCILIATION.md
md = (P/"RECONCILIATION.md").read_text(encoding="utf-8")
def rep(old, new, count=1):
    global md
    assert md.count(old) == count, ("anchor", md.count(old), old[:90])
    md = md.replace(old, new)
rep("candidates after Math-#161 merged. §§1 and 3, the §4 statement, the 22-file inventory, every node id and every edge\nare unchanged.\n",
    "candidates after Math-#161 merged; v1.3 answers Codex review 5359693191 (the [LP] evidence reuses the live node\n"
    "`math.uniform-matrix-cap-lifetime`; the selector proposal resolves all six cells per region and is checked; the\n"
    "catalog is bound by the `OBLIGATION` check and the workflow paths; the residual is defined at scale `r`). §§1 and\n"
    "3, the §4 statement and the 22-file inventory are unchanged; v1.3 removes one proposed node in favour of a live one.\n")
rep("And catalog C6 (`reviews/candidates_pending_20260928/CANDIDATES.md`) asked for",
    "And catalog C6 (`reviews/candidates_pending_20260928/CANDIDATES.md`, read at blob `e3fc1a31`, unchanged through\n"
    "`5c484bf`; bound by the `OBLIGATION` check and the workflow paths) asked for")
rep("The one genuinely open regional statement is the\n"
    "  **localization of the leading-order mass**: (A4) shows pairs within `C_d r` of the midpoint carry `≥ c r³`; [P_η]\n"
    "  shows pairs with both witnesses at distance `≥ η` from the pins carry `O(r⁵)`; an `o(r³)` bound for the **mixed pairs**\n"
    "  (one witness within `η` of a pin, the other remote) is **not supplied by the landed reviewed chain**: [PALM] §6\n"
    "  bounds each regime by `C r³`, not `o(r³)`. This is a further theorem, not a gap in Theorem Q.",
    "The one genuinely open regional statement is the\n"
    "  **localization of the leading-order mass at scale `r`**, defined (with `o` the midpoint) as\n"
    "  `lim_{R→∞} limsup_{r→0} r⁻³ E_{Q_r^W} #{ordered pairs (X, X′) of distinct window points : not both |X − o| ≤ R r\n"
    "  and |X′ − o| ≤ R r} = 0`. What the landed chain gives: (A4) shows pairs within `C_d r` of `o` carry `≥ c r³`, so\n"
    "  the scale-`r` mass is of order `r³`, and local–local pairs between `C_d r` and `R r` belong to it, not to the\n"
    "  residual; [P_η] shows pairs with both witnesses at distance `≥ s_0` from the pins carry `O(r⁵)`; [PALM] §6 R3, the\n"
    "  marked shell bound with `q = 2` whose dyadic sum is `C r³(A_0⁻² + s_0²)`, shows that every pair with a witness in\n"
    "  the shells `R r ≤ |X − o| ≤ s_0` carries at most `C r³(R⁻² + s_0²)`. What it does not give: an `o(r³)` bound for the\n"
    "  **mixed pairs**, one witness within `R r` of `o` and the other at distance `≥ s_0`, where [PALM] §6 R4 gives `C r³`\n"
    "  only. By the three facts above the residual is equivalent to `lim_{s_0→0} lim_{R→∞} limsup_{r→0} r⁻³ M(R, s_0) = 0`\n"
    "  for that mixed count `M(R, s_0)`, and this bound is **not supplied by the landed reviewed chain**. This is a\n"
    "  further theorem, not a gap in Theorem Q.")
rep("to the reviewed regions and scopes only. The `SELECTOR_REGION.json` matrix entries for those regions become\n"
    "`BYPASSED_BY_ANALYTIC_ROUTE` under that node; that file edit is for the executing lane.",
    "to the reviewed regions and scopes only. The `SELECTOR_REGION.json` cells of those three regions are all resolved\n"
    "under that node, so that `hard_gate.py::selector_region_report` shows no open cell for them: `CH-LIFT`,\n"
    "`Piece-2-annulus`, `OBL-H5-JETMOD` and `H5-REMOTE-THRESHOLD` become `BYPASSED_BY_ANALYTIC_ROUTE`; `ALLCELL-FDZ-Q4`\n"
    "becomes `NOT_REQUIRED` and `ENV-RESCOV` `REDERIVED_QUALITATIVELY`, the values the `fixed-remote` row already carries\n"
    "for the same reason (the analytic route uses the positive Fourier spectrum and the joint floors of [PALM] Lemma 6.1 /\n"
    "Lemma E_d, not the historical cell carrier or the historical covariance predicate). The three region ids then move\n"
    "from `open_region_ids` to `covered_region_ids`; `mesoscopic-scaled-annulus` stays open. The `SELECTOR` check applies\n"
    "the proposal to the live table and verifies exactly this. The file edit is for the executing lane; if it prefers the\n"
    "node-only precedent of `regional.fixed-annulus.high-jet-route` (cells untouched), the lists stay unchanged and §7's\n"
    "selector row does not apply.")
rep("and `required: true` edges from the region node and the aggregate to each; plus a required edge to the planar reading-rule node `math.d5-pin-neighborhood-first-moment` (created by Math-#151) |",
    "and `required: true` edges from the region node and the aggregate to each; the [LP] evidence is the live reconciled node `math.uniform-matrix-cap-lifetime` (same SHA256, four required reading-rule edges), not a new node; plus a required edge to the planar reading-rule node `math.d5-pin-neighborhood-first-moment` (created by Math-#151) |")
rep("`math.rn-region.witness-collision.leading-mass-localization`, `OPEN_ACTIVE`, required by nothing, with the §5 candidate note (Math-#161, Math-#159: unreviewed, not premises) |",
    "`math.rn-region.witness-collision.leading-mass-localization`, `OPEN_ACTIVE`, required by nothing, defined as the scale-`r` localization of §5 with the mixed local/remote count as its open piece, with the §5 candidate note (Math-#161, Math-#159: not premises) |")
rep("| `SELECTOR_REGION.json` | `Piece-2-annulus` / `CH-LIFT` `OPEN_ACTIVE` for the shrinking regions | `BYPASSED_BY_ANALYTIC_ROUTE` for `pin-collision`, `intermediate-r-to-rho`, `witness-collision`; `covered_region_ids` / `open_region_ids` updated |",
    "| `SELECTOR_REGION.json` | `Piece-2-annulus` / `CH-LIFT` `OPEN_ACTIVE`, `ENV-RESCOV` / `ALLCELL-FDZ-Q4` `OPEN_HISTORICAL` for the shrinking regions | all six cells resolved for `pin-collision`, `intermediate-r-to-rho`, `witness-collision` (§6: four `BYPASSED_BY_ANALYTIC_ROUTE`, `ALLCELL-FDZ-Q4` `NOT_REQUIRED`, `ENV-RESCOV` `REDERIVED_QUALITATIVELY`); the three region ids move to `covered_region_ids`; checked by `SELECTOR` |")
rep("  `PROOF_INDEX.md` carries either the line-45 obligation sentence or a reference to this record.\n",
    "  `PROOF_INDEX.md` carries either the line-45 obligation sentence or a reference to this record;\n"
    "  `reviews/candidates_pending_20260928/CANDIDATES.md` carries the C6 heading and the quoted open-task sentence, or a\n"
    "  reference to this record.\n")
rep("  cross-record node is one the D5 proposal defines; every proposed node passes the hard gate's shape rules;\n"
    "  `declarative` is true and `executed` is false.\n",
    "  cross-record node is one the D5 proposal defines; a source already carried by a live fingerprinted node ([LP],\n"
    "  `math.uniform-matrix-cap-lifetime`) is referenced through that node, with equal SHA256 and its four required\n"
    "  reading-rule edges present, and no proposed node duplicates a live source; every proposed node passes the hard\n"
    "  gate's shape rules; `declarative` is true and `executed` is false.\n"
    "- **SELECTOR.** The selector proposal names every selector of the live table for its three regions, assigns each a\n"
    "  value the hard gate classifies as covered or bypassed, and, applied to the live table, moves exactly those three\n"
    "  region ids from `open_region_ids` to `covered_region_ids`.\n")
rep("- **OPEN_RESIDUAL.** §5's open items are recorded (numerical constants; the residual node; the mixed-pair bound\n",
    "- **OPEN_RESIDUAL.** §5's open items are recorded (numerical constants; the residual node with its scale-`r`\n  definition; the mixed-pair bound\n")
rep("Eight mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,\n`close-residual`, `drop-lower-bound`, `regional-strict`.",
    "Ten mutants must fail: `allow-symlink`, `no-hash`, `drop-edge`, `stale-fingerprint`, `executed-flag`,\n`close-residual`, `drop-lower-bound`, `regional-strict`, `clone-live-node`, `open-cell-left`.")
assert "not supplied by the landed reviewed chain" in md and "candidates only" in md
(P/"RECONCILIATION.md").write_text(md, encoding="utf-8")

# ------------------------------------------------------------------ PROPOSED_TRANSITIONS.json
d = json.loads((P/"PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
old_lp = [n for n in d["proposed_graph_nodes"] if n.get("source") == LP]
assert len(old_lp) == 1 and old_lp[0]["id"] == "math.c6-component.lifetime-parent"
d["proposed_graph_nodes"] = [n for n in d["proposed_graph_nodes"] if n.get("source") != LP]
new_edges = []
for e in d["proposed_graph_edges"]:
    if e["to"] == "math.c6-component.lifetime-parent":
        e = dict(e, to=LIVE_LP, relation="requires_evidence_via_live_reconciled_node")
    new_edges.append(e)
d["proposed_graph_edges"] = new_edges
assert sum(1 for e in new_edges if e["to"] == LIVE_LP and e["required"]) == 2
wit = next(e for e in d["graph"] if e["node"] == "math.rn-region.witness-collision")
if LIVE_LP not in wit["required_nodes"]:
    wit["required_nodes"].append(LIVE_LP)
d["live_nodes_reused"] = [{"id": LIVE_LP, "source": LP,
                           "fingerprint": "9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7",
                           "reading_rule_required": ["math.d1-component.congruence-erratum",
                                                     "math.d1-component.section9-replacement-v1_1",
                                                     "math.d1-component.marked-cylinder-cap",
                                                     "math.d1-component.reconciliation-record"],
                           "why": "the live graph already carries these bytes with four required reading-rule edges (D1 reconciliation, Math-#126); a second terminal node would not inherit a later hold on them (Codex 4139312869)"}]
d["selector_region_proposal"] = {
    "regions": ["pin-collision", "intermediate-r-to-rho", "witness-collision"],
    "cells": {"CH-LIFT": "BYPASSED_BY_ANALYTIC_ROUTE", "Piece-2-annulus": "BYPASSED_BY_ANALYTIC_ROUTE",
              "OBL-H5-JETMOD": "BYPASSED_BY_ANALYTIC_ROUTE", "H5-REMOTE-THRESHOLD": "BYPASSED_BY_ANALYTIC_ROUTE",
              "ALLCELL-FDZ-Q4": "NOT_REQUIRED", "ENV-RESCOV": "REDERIVED_QUALITATIVELY"},
    "under_node": "regional.shrinking-regions.analytic-route",
    "resulting_covered_region_ids": ["math.rn-region.fixed-remote", "math.rn-region.pin-collision",
                                     "math.rn-region.intermediate-r-to-rho", "math.rn-region.witness-collision"],
    "resulting_open_region_ids": ["math.rn-region.mesoscopic-scaled-annulus"],
    "precedent": "fixed-remote row: ALLCELL-FDZ-Q4 NOT_REQUIRED, ENV-RESCOV REDERIVED_QUALITATIVELY; the analytic route uses the positive Fourier spectrum and the joint floors (PALM Lemma 6.1 / Lemma E_d), not the historical carriers. hist.ENV-RESCOV and hist.ALLCELL-FDZ-Q4 keep their classifications.",
    "alternative": "node-only supersession as regional.fixed-annulus.high-jet-route (cells untouched, lists unchanged); the executing lane decides."}
R = "math.rn-region.witness-collision.leading-mass-localization"
g = next(e for e in d["graph"] if e["node"] == R)
g["scope"] = ("RECONCILIATION.md section 5: leading-mass localization at scale r, i.e. lim_{R->oo} limsup_{r->0} r^-3 E #{ordered pairs "
              "of distinct window points not both within R r of the midpoint} = 0. Known: pairs within C_d r carry >= c r^3 (EDL (A4)), so "
              "local-local pairs between C_d r and R r are localized mass, not residual; both-remote pairs carry O(r^5) (P_eta); pairs with "
              "a witness in the shells R r <= |X - o| <= s_0 carry <= C r^3 (R^-2 + s_0^2) (PALM section 6 R3, marked shell bound, q = 2). "
              "Open piece: the mixed count M(R, s_0) of pairs with one witness within R r of the midpoint and the other at distance >= s_0, "
              "where PALM R4 gives C r^3 only; the residual is equivalent to lim_{s_0->0} lim_{R->oo} limsup_r r^-3 M(R, s_0) = 0, not supplied "
              "by the landed reviewed chain. " + g["scope"].split("Author-side candidates:")[1].join(["Author-side candidates:", ""]) if False else
              "RECONCILIATION.md section 5: leading-mass localization at scale r, i.e. lim_{R->oo} limsup_{r->0} r^-3 E #{ordered pairs "
              "of distinct window points not both within R r of the midpoint} = 0. Known: pairs within C_d r carry >= c r^3 (EDL (A4)), so "
              "local-local pairs between C_d r and R r are localized mass, not residual; both-remote pairs carry O(r^5) (P_eta); pairs with "
              "a witness in the shells R r <= |X - o| <= s_0 carry <= C r^3 (R^-2 + s_0^2) (PALM section 6 R3, marked shell bound, q = 2). "
              "Open piece: the mixed count M(R, s_0) of pairs with one witness within R r of the midpoint and the other at distance >= s_0, "
              "where PALM R4 gives C r^3 only; the residual is equivalent to lim_{s_0->0} lim_{R->oo} limsup_r r^-3 M(R, s_0) = 0, not supplied "
              "by the landed reviewed chain. Author-side candidates:" + g["scope"].split("Author-side candidates:")[1])
n = next(x for x in d["proposed_graph_nodes"] if x["id"] == R)
n["fingerprint"] = "leading-order r^3 mass of E N(N-1): scale-r localization; open piece = mixed local/remote pairs M(R, s_0)"
n["notes"] = ("Residual after the count-level discharge of math.rn-region.witness-collision. Definition: lim_{R->oo} limsup_{r->0} r^-3 "
              "E #{ordered pairs not both within R r of the midpoint} = 0. Known: pairs within C_d r carry >= c r^3 (EDL (A4)); both-remote "
              "pairs O(r^5) (P_eta); pairs with a shell witness <= C r^3 (R^-2 + s_0^2) (PALM section 6 R3, q = 2). Open: the mixed "
              "local/remote count M(R, s_0), o(r^3) in the iterated limit, not supplied by the landed reviewed chain; " +
              n["notes"].split("author-side candidates:")[1].strip().rstrip(".") + ". Required by no theorem; recorded so that the register keeps a precise statement.")
d["register_read"] = {CATALOG: {"git_blob": "e3fc1a310c89ac9fd9249af26e28da509649ffd5",
                                "sha256": "3809e6147321df260085815b2d9e995094d63492534afa4ad1cd05b9a168ce4d",
                                "binding": "OBLIGATION check: C6 heading and the quoted open-task sentence present, or this record referenced; workflow re-runs on change"}}
d["revision"] = {"version": "v1.3",
                 "applies": ["v1.1: OpenAI review 5359701805 items 1-3; OpenAI pickup 5900929519 residual wording",
                             "v1.2: merge of Math- main 5c484bf; candidate status after Math-#161 merged",
                             "v1.3: Codex review 5359693191 threads 4139312857 (catalog bound), 4139312869 (live LP node reused), 4139312863 (all six selector cells resolved, SELECTOR check), 4139312874 (residual defined at scale r)"],
                 "unchanged": "22-file inventory, status-row statement, historical predicates; one proposed node removed in favour of the live node"}
(P/"PROPOSED_TRANSITIONS.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ reconciliation_check.py
c = (P/"reconciliation_check.py").read_text(encoding="utf-8")
def crep(old, new):
    global c
    assert c.count(old) == 1, ("checker anchor", c.count(old), old[:80])
    c = c.replace(old, new)
crep('"drop-lower-bound", "regional-strict")', '"drop-lower-bound", "regional-strict", "clone-live-node", "open-cell-left")')
crep('D5_AGGREGATE = "math.d5-pin-neighborhood-first-moment"\n',
     'D5_AGGREGATE = "math.d5-pin-neighborhood-first-moment"\n'
     'LIVE_LP = "math.uniform-matrix-cap-lifetime"\n'
     'LP = "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md"\n'
     'CATALOG = "reviews/candidates_pending_20260928/CANDIDATES.md"\n'
     'OPEN_CELLS = ("OPEN_ACTIVE", "OPEN_HISTORICAL", "NOT_DISCHARGED", "PARTIAL_COVER_ONLY", "PARTIAL_PR7_REDUCTION",\n'
     '              "REOPENED", "PARTIAL_COVER_DECLARED_REGION")  # hard_gate.py::selector_region_report\n')
crep('    ok &= ("NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate." in index\n'
     '           or HERE in index)\n    return ok\n',
     '    ok &= ("NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate." in index\n'
     '           or HERE in index)\n'
     '    catalog = (root / CATALOG).read_text(encoding="utf-8")\n'
     '    ok &= ("## C6. Torus-wide second factorial moment" in catalog\n'
     '           and "a valid full-window factorial upper" in catalog) or HERE in catalog\n'
     '    return ok\n')
crep('    if MUT == "stale-fingerprint":\n',
     '    if MUT == "clone-live-node":\n'
     '        nodes["math.c6-component.lifetime-parent"] = dict(nodes[AGGREGATE], id="math.c6-component.lifetime-parent",\n'
     '                                                          kind="reading_rule_component", source=LP,\n'
     '                                                          fingerprint=INVENTORY[LP][0])\n'
     '    if MUT == "stale-fingerprint":\n')
crep('    evidence = dict(nodes)\n    for n in d5["proposed_graph_nodes"]:\n        evidence.setdefault(n["id"], n)\n',
     '    # a source already carried by a live fingerprinted node is referenced through that node; then this record\'s\n'
     '    # nodes; then the D5 proposal\'s (cross-record reuse, same bytes)\n'
     '    live_fp = {nid: n for nid, n in live.items() if isinstance(n, dict) and n.get("fingerprint") and n.get("source")}\n'
     '    evidence = dict(live_fp)\n'
     '    for nid, n in nodes.items():\n        evidence.setdefault(nid, n)\n'
     '    for n in d5["proposed_graph_nodes"]:\n        evidence.setdefault(n["id"], n)\n')
crep('    for nid, n in evidence.items():                             # this record\'s nodes come first and win ties\n',
     '    for nid, n in evidence.items():                             # live nodes first, then this record\'s, then D5\'s\n')
crep('    ok &= all(evidence[nid]["fingerprint"] == INVENTORY[src][0] for src, nid in by_source.items())\n',
     '    ok &= all(evidence[nid]["fingerprint"] == INVENTORY[src][0] for src, nid in by_source.items())\n'
     '    # no proposed node duplicates a source the live graph already carries with a fingerprint (Codex 4139312869)\n'
     '    ok &= not ({n.get("source") for n in nodes.values() if "fingerprint" in n and n.get("source")}\n'
     '               & {n["source"] for n in live_fp.values()})\n'
     '    # the reused live node is PROVED_REVIEWED with its four required reading-rule components, all PROVED_REVIEWED\n'
     '    lp = live.get(LIVE_LP, {})\n'
     '    ok &= lp.get("classification") == "PROVED_REVIEWED" and lp.get("source") == LP and lp.get("fingerprint") == INVENTORY[LP][0]\n'
     '    rr = lp.get("reading_rule", [])\n'
     '    ok &= len(rr) == 4 and all(live.get(x, {}).get("classification") == "PROVED_REVIEWED" for x in rr)\n'
     '    ok &= by_source.get(LP) == LIVE_LP\n')
crep('    ok &= (WITNESS, D5_AGGREGATE) in have_req and (AGGREGATE, D5_AGGREGATE) in have_req\n    return bool(ok)\n',
     '    ok &= (WITNESS, D5_AGGREGATE) in have_req and (AGGREGATE, D5_AGGREGATE) in have_req\n'
     '    ok &= (WITNESS, LIVE_LP) in have_req and (AGGREGATE, LIVE_LP) in have_req\n'
     '    return bool(ok)\n\n\n'
     'def check_selector(root):\n'
     '    """Applied to the live selector table, the proposal leaves no open cell in its regions and moves exactly those\n'
     '    region ids from open to covered (Codex 4139312863)."""\n'
     '    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))\n'
     '    live = json.loads((root / SELECTOR).read_text(encoding="utf-8"))\n'
     '    prop = spec["selector_region_proposal"]\n'
     '    cells = dict(prop["cells"])\n'
     '    if MUT == "open-cell-left":\n'
     '        cells.pop("ENV-RESCOV", None)\n'
     '    regions = list(prop["regions"])\n'
     '    ok = set(regions) <= set(live["regions"]) and len(regions) == 3\n'
     '    ok &= set(cells) == set(live["selectors"])\n'
     '    ok &= all(v not in OPEN_CELLS for v in cells.values())\n'
     '    table = {s: dict(row) for s, row in live["selectors"].items()}\n'
     '    for s, v in cells.items():\n'
     '        for reg in regions:\n'
     '            table[s][reg] = v\n'
     '    def is_open(reg):\n'
     '        return any(table[s].get(reg) in OPEN_CELLS for s in live["selectors"])\n'
     '    ok &= not any(is_open(reg) for reg in regions)\n'
     '    ids = ["math.rn-region." + reg for reg in regions]\n'
     '    covered = list(live["covered_region_ids"]) + [i for i in ids if i not in live["covered_region_ids"]]\n'
     '    opened = [i for i in live["open_region_ids"] if i not in ids]\n'
     '    ok &= sorted(prop["resulting_covered_region_ids"]) == sorted(covered)\n'
     '    ok &= sorted(prop["resulting_open_region_ids"]) == sorted(opened)\n'
     '    ok &= all(("math.rn-region." + reg) in live["open_region_ids"] for reg in regions)\n'
     '    ok &= prop.get("under_node") == SUPERSESSION\n'
     '    return bool(ok)\n')
crep('              "MONOTONE": check_monotone(), "OPEN_RESIDUAL": check_open_residual(root)}\n    passed = all(checks.values()) and len(checks) == 7\n',
     '              "MONOTONE": check_monotone(), "OPEN_RESIDUAL": check_open_residual(root),\n'
     '              "SELECTOR": check_selector(root)}\n    passed = all(checks.values()) and len(checks) == 8\n')
crep('"scope": "identity, verdict-row, obligation, filesystem-negative, transition-chain, monotonicity "\n'
     '                               "and open-item checks; no mathematics is re-proved"',
     '"scope": "identity, verdict-row, obligation, filesystem-negative, transition-chain, monotonicity, "\n'
     '                               "open-item and selector-cell checks; no mathematics is re-proved"')
crep('    ok = "**Numerical constants**" in text and "leading-mass-localization" in text\n',
     '    ok = "**Numerical constants**" in text and "leading-mass-localization" in text\n'
     '    ok &= "localization of the leading-order mass at scale `r`" in text and "M(R, s_0)" in text\n')
(P/"reconciliation_check.py").write_text(c, encoding="utf-8")

# ------------------------------------------------------------------ workflow
W = pathlib.Path(".github/workflows/c6-witness-collision-reconciliation.yml")
w = W.read_text(encoding="utf-8")
assert w.count("      - 'PROOF_INDEX.md'\n") == 1
w = w.replace("      - 'PROOF_INDEX.md'\n", "      - 'PROOF_INDEX.md'\n      - 'reviews/candidates_pending_20260928/CANDIDATES.md'\n")
old = "'drop-lower-bound', 'regional-strict']"
assert w.count(old) == 1
w = w.replace(old, "'drop-lower-bound', 'regional-strict', 'clone-live-node', 'open-cell-left']")
W.write_text(w, encoding="utf-8")
print("v1.3 edits applied")
