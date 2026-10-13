import json, re, pathlib, subprocess
WT = pathlib.Path("<SCRATCH>/wt160")
R = WT / "reviews/c6_witness_collision_reconciliation_20260929"
md_p, js_p, py_p = R/"RECONCILIATION.md", R/"PROPOSED_TRANSITIONS.json", R/"reconciliation_check.py"
md, raw, py = md_p.read_text(), js_p.read_text(), py_p.read_text()
js = json.loads(raw)
def rep(s, old, new, n=1):
    assert s.count(old) == n, ("COUNT", s.count(old), old[:90]); return s.replace(old, new)
def dump(ref):
    for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2, ensure_ascii=False, sort_keys=True), dict(indent=2), dict(indent=2, sort_keys=True)):
        if json.dumps(json.loads(ref), **kw) + "\n" == ref: return lambda o: json.dumps(o, **kw) + "\n"
    raise SystemExit("json format not detected")
dj = dump(raw)
g = lambda *a: subprocess.check_output(["git", *a], cwd=WT).decode().strip()
old_pi_blob = g("rev-parse", "bb429d3:PROOF_INDEX.md"); old_graph_blob = g("rev-parse", "bb429d3:frontiers/downstream_gate_20260925/GRAPH.json")
# ---------------- RECONCILIATION.md
md = rep(md, "`bb429d3` (Math-#157 and #165 merged); no inventoried path changed at any of them.",
"`bb429d3` (Math-#157 and #165 merged); no inventoried path changed at any of them. Re-verified at `ec6db8c` (Math-#151\nmerged, 30 September 00:56Z): no inventoried path changed; three register surfaces did (§1a).")
md = rep(md, "v1.6 records xAI 5901915195.\n",
"v1.6 records xAI 5901915195. v1.7 re-verifies at `ec6db8c`,\nquotes the register as Math-#151 rewrote it (§1a), rebinds the `OBLIGATION` check to the current text, lists the D5 nodes\nnow live, and records that Math-#151 took the keep-open alternative without defining the regional mechanism.\n")
sec1a = '''### 1a. The register at `ec6db8c` (after Math-#151)

Math-#151 merged at `ec6db8c` on 30 September (00:56Z) and rewrote three of the surfaces quoted above. The quotes in
§1 remain the obligation as the register recorded it when this record was opened and reviewed (OpenAI 5359701805,
xAI 5901915195 read that text); the `OBLIGATION` check now binds either text, so the workflow fails on any third
rewrite.

1. **`PROOF_INDEX.md`** (blob `030447a1`), open-obligations section:
   > D5 shrinking multiple-witness collision: fixed-separation machinery is `frontiers/remote_collision_20260928/PROOF.md`.
   > The reviewed Fourier upper bound and the later merged sharp Palm route establish global count statements by routes
   > that do not identify this regional shrinking-separation mechanism. **NO COMPLETE REGIONAL PROOF YET:**
   > `math.rn-region.witness-collision` remains `OPEN_ACTIVE`; the global `Theta(r^3)` result is not a proof of that node.
2. **`GRAPH.json`** (blob `220b5686`), node `math.rn-region.witness-collision`: still `OPEN_ACTIVE`, fingerprint
   unchanged ("eta->0 mutual witness separation"); new fields `scope: "catalog C6"`,
   `review_source: reviews/d5_reconciliation_20260929/RECONCILIATION.md`, `review_disposition: "OPEN"`,
   `reading_rule: [math.d5-component.offpin-second-moment-review]` with a `required` edge to that component; notes:
   > Off-pin fixed-separation second-moment evidence is reviewed and sharp global order is reviewed and merged through
   > Math-#145, but the regional shrinking pin/witness-collision mechanism and its source-bound graph fold remain open.
3. **Catalog C6** (`CANDIDATES.md`, blob `02455f05`), retitled "Torus-wide factorial moments: reviewed global order;
   regional mechanism open": it records [PALM]'s `Θ(r³)` in every fixed `d ≥ 2` at its packet scope, then:
   > They do not close the regional shrinking-witness mechanism: the planar graph node `math.rn-region.witness-collision`
   > remains `OPEN_ACTIVE`. […] Open task: prove or refute the shrinking-separation witness-collision mechanism encoded by
   > the open graph node, without inferring it from the global Palm bound.

**Reading.** The count sentence of §1 ("shrinking-separation factorial-moment/collision estimate") is no longer in the
register: Math-#151 replaced it by a "regional shrinking pin/witness-collision mechanism" that no surface defines, i.e.
it took the alternative anticipated in §5 (keep the old node open) but under a new, undefined meaning and the old
fingerprint. Three consequences for the executing lane. (i) The count obligation, as the register recorded it through
`bb429d3` and as OpenAI 5359701805 and xAI 5901915195 read it, is discharged at the §4 scope; nothing in the new text
disputes that: "the global `Theta(r^3)` result is not a proof of that node" is a statement about the node's new
meaning, not about the count. (ii) The only definition on file for the new meaning is §5's residual, the scale-`r`
localization with the mixed count `M(R, s_0)` as its open piece; the new open task ("without inferring it from the
global Palm bound") is consistent with it and supplies no other. (iii) The proposal of §7 therefore stands, and its two
executions are register-equivalent: create the residual node and move the old node, or keep the old node open and give
it the residual's definition and fingerprint. Either way the reading rule now live on the old node
(`math.d5-component.offpin-second-moment-review`) is preserved; this record already lists that component as supporting
evidence of the old node and of the aggregate.

'''
md = rep(md, "§5 returns to that point.\n\n## 2. What is now proved and reviewed", "§5 returns to that point.\n\n" + sec1a + "## 2. What is now proved and reviewed")
md = rep(md, "Math-#151's draft\n  catalog and proof-index text keep `math.rn-region.witness-collision` open on the ground that",
"Math-#151's catalog\n  and proof-index text (merged at `ec6db8c`; quoted in §1a) keep `math.rn-region.witness-collision` open on the ground that")
md = rep(md, "| Surface | Current (`8e61fc4`) | Proposed |", "| Surface | Current (`8e61fc4`; `ec6db8c` where Math-#151 changed it) | Proposed |")
md = rep(md, "| `PROOF_INDEX.md` line 45 | \"NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate\" | move to the reviewed section",
"| `PROOF_INDEX.md` open-obligations bullet | at `8e61fc4`: \"NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate\"; at `ec6db8c`: \"NO COMPLETE REGIONAL PROOF YET: `math.rn-region.witness-collision` remains `OPEN_ACTIVE`; the global `Theta(r^3)` result is not a proof of that node\" (§1a) | move to the reviewed section")
md = rep(md, "| `reviews/candidates_pending_20260928/CANDIDATES.md` C6 | Math-#151's draft keeps \"regional mechanism open\" |",
"| `reviews/candidates_pending_20260928/CANDIDATES.md` C6 | at `ec6db8c` (Math-#151): retitled \"reviewed global order; regional mechanism open\", open task \"prove or refute the shrinking-separation witness-collision mechanism encoded by the open graph node\" (§1a) |")
md = rep(md, "| `GRAPH.json` `math.rn-region.witness-collision` | `OPEN_ACTIVE` (also in Math-#151) |",
"| `GRAPH.json` `math.rn-region.witness-collision` | `OPEN_ACTIVE` (kept by Math-#151 at `ec6db8c`, with `reading_rule` `[math.d5-component.offpin-second-moment-review]` and `review_disposition` `OPEN`; §1a) |")
md = rep(md, "**Ordering.** These transitions presuppose Math-#151's execution of the planar D5 fold (its 14 nodes and 30 edges);\nthe required edge to `math.d5-pin-neighborhood-first-moment` refers to that node. They can be executed in the same\nlane, after it, or folded into #151's restack if that lane prefers; either way the executing lane, not this record,\ndecides. Within the execution the order is fixed (xAI 5901345590):",
"**Ordering.** These transitions presupposed Math-#151's execution of the planar D5 fold; it is on main at `ec6db8c`\n(14 nodes, 30 edges), so the nodes this record reuses are live: `math.d5-pin-neighborhood-first-moment`\n(`PROVED_REVIEWED`, aggregate, no fingerprint), `math.d5-component.offpin-second-moment-review` (`PROVED_REVIEWED`,\nfingerprint `43beb4ef…`, now also the old node's live `reading_rule`) and `math.d5-component.remote-window-proof`\n(`PROVED_REVIEWED`, fingerprint `a332bae9…`, a second node for the `math.rn-fixed-remote-window` bytes; Math-#167\nproposes the alignment of the older node). The executing lane, not this record, decides. Within the execution the\norder is fixed (xAI 5901345590):")
md = rep(md, "the xAI bounded read of [PALM] §§4–6 asked\nfor in §0 precedes that last step. Nothing here edits",
"the xAI bounded read of [PALM] §§4–6 asked\nfor in §0 precedes that last step. The keep-open execution (§1a, point (iii)) performs the first three steps, records\nthe discharge on the aggregate node and then, instead of moving the old node, gives it the residual's definition and\nfingerprint (the residual node is then not created separately); both executions leave the same question open. Nothing here edits")
m = re.search(r"- \*\*OBLIGATION\.\*\*.*?(?=\n- \*\*)", md, re.S); assert m, "OBLIGATION bullet"
md = md[:m.start()] + ("- **OBLIGATION.** The live graph still carries `math.rn-region.witness-collision` as an `OPEN_ACTIVE` region with the\n"
"  fingerprint quoted in §1 (or as `PROVED_REVIEWED` after execution); while open, its `review_disposition`, if any, is\n"
"  `OPEN` and every `reading_rule` component is live `PROVED_REVIEWED`; the selector still lists the region;\n"
"  `PROOF_INDEX.md` carries the §1 sentence, or the §1a sentence (\"NO COMPLETE REGIONAL PROOF YET\" with the node named\n"
"  `OPEN_ACTIVE`), or a reference to this record; the catalog carries the §1 C6 heading with its open-task sentence, or\n"
"  the §1a heading with the node named `OPEN_ACTIVE`, or a reference to this record. Any other rewrite of those surfaces\n"
"  fails the check, and the workflow re-runs on their change.") + md[m.end():]
md = rep(md, "cross-record node is one the D5 proposal defines;", "cross-record node is one the D5 proposal defines and, since `ec6db8c`, live with that proposal's fingerprint;")
md = rep(md, "- **Math-#151 (Codex; head `64a170f`; xAI wording hold lifted at 5900436844):** executes the planar D5 fold and\n  keeps witness-collision open. This record does not conflict with its graph edits; it supersedes only its *catalog and proof-index wording* about C6, and only if the\n  integrating lane accepts §§1 and 4. If that lane prefers to keep the old node open, §5 asks that the reason be\n  written as the residual node's definition. One register point for that lane: the live graph already carries\n  `math.rn-fixed-remote-window` (fingerprint = the remote_window bytes, classification `AUTHOR_SIDE_CANDIDATE` despite\n  the main#76 review), and the D5 proposal creates `math.d5-component.remote-window-proof` for the same bytes; this\n  record references the live node (supporting edges only) and leaves the collapse of the pair, and the classification\n  repair, to the executing lane.",
"- **Math-#151 (Codex; merged at `ec6db8c`, 30 September 00:56Z, final head `609e559`):** executed the planar D5 fold\n  (14 nodes, 30 edges; the three planar first-moment regions to `PROVED_REVIEWED`) and kept witness-collision open under\n  the rewritten text quoted in §1a. This record does not conflict with its graph edits and now points at its live nodes;\n  it supersedes only the catalog and proof-index wording about C6, and only if the executing lane accepts §§1 and 4\n  (OpenAI 5359701805 and xAI 5901915195 do). The D4 duplicate (`math.d5-component.remote-window-proof` beside\n  `math.rn-fixed-remote-window`, same bytes) landed uncollapsed; Math-#167 (this session) proposes the alignment of the\n  older node, and this record keeps referencing the older live node with supporting edges only.")
md = rep(md, "states the §5 residual exactly, conditional on Math-#162; B and C accepted (other Claude session), A under review;\n  not consumed here; the residual closes by a successor record after both land.",
"states the §5 residual exactly, conditional on Math-#162; Slices A, B and C carry the other Claude session's verdicts\n  (5360227991, 5360178611, 5360216551; one provider); unmerged; not consumed here; the residual closes by a successor\n  record after both land.")
md = rep(md, "- **main#207:** the pickup for this record is comment 5900687550. The register execution stays with a non-Claude lane.",
"- **main#207 (closed 30 September 00:57Z, after Math-#151 merged):** the pickup for this record is comment 5900687550\n  and the delivery 5900887059. The register execution stays with a non-Claude lane.")
# ---------------- PROPOSED_TRANSITIONS.json
js["reverified_at"] = ("ec6db8c (Math- main after Math-#151 merged 2026-09-30T00:56:58Z; no inventoried path changed; PROOF_INDEX.md, "
                       "GRAPH.json and reviews/candidates_pending_20260928/CANDIDATES.md changed as quoted in RECONCILIATION.md section 1a; "
                       "SELECTOR_REGION.json unchanged)")
old_cat = js["register_read"]["reviews/candidates_pending_20260928/CANDIDATES.md"]
js["register_read"] = {
  "reviews/candidates_pending_20260928/CANDIDATES.md": {"git_blob": "02455f0561d51990e1486e5d4fbb4229c8a34f5a",
      "sha256": "cbaa71b55190ac234b8363614b7e57387f60c33b393ea102408eb88ec1e44513", "at": "ec6db8c (rewritten by Math-#151; section 1a)",
      "previous": {"git_blob": old_cat["git_blob"], "sha256": old_cat["sha256"], "through": "bb429d3"},
      "binding": "OBLIGATION check: the section 1 C6 heading with its open-task sentence, or the section 1a heading with the node named OPEN_ACTIVE, or this record referenced; workflow re-runs on change"},
  "PROOF_INDEX.md": {"git_blob": "030447a1130b52a09230e21c039409f87ef2ba32", "sha256": "a1259b8bff4d73d97ee0835ea5c82568d4fac39fae352f5a3a2422c6bf0f8a0c",
      "at": "ec6db8c (rewritten by Math-#151; section 1a)", "previous": {"git_blob": old_pi_blob, "through": "bb429d3"},
      "binding": "OBLIGATION check: the section 1 sentence, or the section 1a sentence, or this record referenced; workflow re-runs on change"},
  "frontiers/downstream_gate_20260925/GRAPH.json": {"git_blob": "220b56867f24c5ef6f55a3cce70490e9777cf3af", "sha256": "8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09",
      "at": "ec6db8c (Math-#151 fold: 14 nodes and 30 edges added; witness node gains scope, review_source, review_disposition OPEN, reading_rule and notes; classification and fingerprint unchanged)",
      "previous": {"git_blob": old_graph_blob, "through": "bb429d3"}, "binding": "OBLIGATION, TRANSITIONS and SELECTOR checks; workflow re-runs on change"},
  "frontiers/downstream_gate_20260925/SELECTOR_REGION.json": {"git_blob": "ebffdb0969c7d3484f8fd1f4352ea290346f0318", "sha256": "53fa0ec88e417882eff2a40e4b575c639d43b7e8787e7fb8917aa22ace79390d",
      "at": "unchanged through ec6db8c", "binding": "OBLIGATION and SELECTOR checks; workflow re-runs on change"}}
js["presupposes"] = ["reviews/d5_reconciliation_20260929/PROPOSED_TRANSITIONS.json executed by Math-#151, on main at ec6db8c: math.d5-pin-neighborhood-first-moment (PROVED_REVIEWED, aggregate) and its 13 component nodes are live; bound by TRANSITIONS"]
js["live_nodes_reused"] += [
  {"id": "math.d5-pin-neighborhood-first-moment", "live_since": "ec6db8c (Math-#151)", "classification": "PROVED_REVIEWED", "kind": "candidate (aggregate, no fingerprint)",
   "source": "reviews/d5_reconciliation_20260929/RECONCILIATION.md", "role": "required evidence of the witness node and of the aggregate (planar reading rule)"},
  {"id": "math.d5-component.offpin-second-moment-review", "live_since": "ec6db8c (Math-#151)", "classification": "PROVED_REVIEWED",
   "fingerprint": "43beb4ef7da5bf71e15164d5c4bc1862db387d83f906aebdf9831625b1ca6369", "source": "reviews/d5_offpin_second_moment_20260928/REVIEW.md",
   "role": "supporting only; since ec6db8c also the live reading_rule of math.rn-region.witness-collision"},
  {"id": "math.d5-component.remote-window-proof", "live_since": "ec6db8c (Math-#151)", "classification": "PROVED_REVIEWED",
   "fingerprint": "a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7", "source": "frontiers/remote_window_20260924/PROOF.md",
   "role": "not referenced; second live node for the math.rn-fixed-remote-window bytes (Math-#151 landed the D5 proposal as written); Math-#167 proposes the alignment of the older node"}]
eo = js["execution_order"]; assert eo[3].startswith("4. only after Math-#151's fold is on main")
eo[3] = ("4. Math-#151's fold is on main (ec6db8c). Remaining precondition for this step: the xAI bounded read of PALM sections 4-6 (per xAI 5901915195) "
         "or an explicit integrator decision that OpenAI 5358116559 plus the section 1 reading suffices; then move math.rn-region.witness-collision to "
         "PROVED_REVIEWED and update STATUS/PROOF_INDEX/catalog rows. Register-equivalent alternative (section 1a): keep the node OPEN_ACTIVE, give it the "
         "residual's definition and fingerprint, record the discharge on the aggregate node, and do not create the residual node separately")
js["note"] = rep(js["note"], "Execution is for a non-Claude lane, after Math-#151's planar fold, which defines math.d5-pin-neighborhood-first-moment and math.d5-component.offpin-second-moment-review (reused here rather than duplicated); the [LP] and remote_window bytes are referenced through the live nodes math.uniform-matrix-cap-lifetime and math.rn-fixed-remote-window.",
  "Execution is for a non-Claude lane. Math-#151's planar fold is on main (ec6db8c) and defines math.d5-pin-neighborhood-first-moment and math.d5-component.offpin-second-moment-review as live nodes (reused here rather than duplicated); the [LP] and remote_window bytes are referenced through the live nodes math.uniform-matrix-cap-lifetime and math.rn-fixed-remote-window (the latter now duplicated by the live math.d5-component.remote-window-proof; Math-#167 proposes the alignment).")
js["revision"]["version"] = "v1.7"
js["revision"]["applies"].append("v1.7: re-verified at ec6db8c (Math-#151 merged); register surfaces re-quoted (section 1a); OBLIGATION check rebound to the current PROOF_INDEX/catalog text and the open node's reading rule; live D5 nodes bound to the D5 proposal's fingerprints; keep-open alternative recorded")
# ---------------- reconciliation_check.py
py = rep(py, '''    if node.get("classification") == "OPEN_ACTIVE":
        ok = node.get("fingerprint") == "eta->0 mutual witness separation"
    else:
        ok = node.get("classification") == "PROVED_REVIEWED"
''', '''    if node.get("classification") == "OPEN_ACTIVE":
        ok = node.get("fingerprint") == "eta->0 mutual witness separation"
        # since ec6db8c (Math-#151) the open node carries a review disposition and a reading rule
        ok &= node.get("review_disposition") in (None, "OPEN")
        ok &= all(graph["nodes"].get(x, {}).get("classification") == "PROVED_REVIEWED" for x in node.get("reading_rule", []))
    else:
        ok = node.get("classification") == "PROVED_REVIEWED"
''')
py = rep(py, '''    ok &= ("NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate." in index
           or HERE in index)
    catalog = (root / CATALOG).read_text(encoding="utf-8")
    ok &= ("## C6. Torus-wide second factorial moment" in catalog
           and "a valid full-window factorial upper" in catalog) or HERE in catalog
    return ok''', '''    ok &= ("NO COMPLETE PROOF YET: shrinking-separation factorial-moment/collision estimate." in index      # through bb429d3
           or "**NO COMPLETE REGIONAL PROOF YET:** `math.rn-region.witness-collision` remains `OPEN_ACTIVE`" in index  # ec6db8c
           or HERE in index)
    catalog = (root / CATALOG).read_text(encoding="utf-8")
    ok &= (("## C6. Torus-wide second factorial moment" in catalog                                              # through bb429d3
            and "a valid full-window factorial upper" in catalog)
           or ("## C6. Torus-wide factorial moments: reviewed global order; regional mechanism open" in catalog  # ec6db8c
               and "`math.rn-region.witness-collision` remains `OPEN_ACTIVE`" in catalog)
           or HERE in catalog)
    return ok''')
py = rep(py, '''    by_source = {}
    for nid, n in evidence.items():                             # live nodes first, then this record's, then D5's
''', '''    by_source = {}
    for r in spec["live_nodes_reused"]:                         # the live nodes this record names come first
        n = live.get(r["id"])
        ok &= isinstance(n, dict) and (not r.get("fingerprint") or n.get("fingerprint") == r["fingerprint"])
        if isinstance(n, dict) and r.get("fingerprint") and n.get("source") in INVENTORY:
            by_source.setdefault(n["source"], r["id"])
    for nid, n in evidence.items():                             # then other live nodes, this record's, then D5's
''')
py = rep(py, '''    ok &= (WITNESS, LIVE_LP) in have_req and (AGGREGATE, LIVE_LP) in have_req
    return bool(ok)''', '''    ok &= (WITNESS, LIVE_LP) in have_req and (AGGREGATE, LIVE_LP) in have_req
    # since ec6db8c (Math-#151) the reused D5 nodes are live: bound to the D5 proposal's classification and fingerprint
    d5_by_id = {n["id"]: n for n in d5["proposed_graph_nodes"]}
    for nid in (D5_AGGREGATE, "math.d5-component.offpin-second-moment-review"):
        ln = live.get(nid, {})
        ok &= ln.get("classification") == "PROVED_REVIEWED" == d5_by_id[nid].get("classification")
        ok &= (ln.get("fingerprint") or None) == (d5_by_id[nid].get("fingerprint") or None)
    # the D4 bytes are carried by two live nodes (the D5 proposal landed as written); they must agree byte for byte
    dup = live.get("math.d5-component.remote-window-proof")
    if isinstance(dup, dict):
        ok &= dup.get("fingerprint") == live.get("math.rn-fixed-remote-window", {}).get("fingerprint") \\
            == INVENTORY["frontiers/remote_window_20260924/PROOF.md"][0]
    return bool(ok)''')
md_p.write_text(md); js_p.write_text(dj(js)); py_p.write_text(py)
print("edits written")
