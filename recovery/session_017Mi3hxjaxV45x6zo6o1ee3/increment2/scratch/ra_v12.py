import json, re, pathlib, subprocess, hashlib, sys, textwrap
WT = pathlib.Path("<SCRATCH>/wt167")
R = WT / "reviews/register_alignment_20260930"
md_p, js_p, py_p, wf_p = R/"RECONCILIATION.md", R/"PROPOSED_TRANSITIONS.json", R/"alignment_check.py", WT/".github/workflows/register-alignment.yml"
md, raw, py, wf = md_p.read_text(), js_p.read_text(), py_p.read_text(), wf_p.read_text()
js = json.loads(raw)
FP = "a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7"
def rep(s, old, new, n=1):
    assert s.count(old) == n, ("COUNT", s.count(old), old[:80]); return s.replace(old, new)
def dump(obj, ref):
    for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2, ensure_ascii=False, sort_keys=True), dict(indent=2), dict(indent=2, sort_keys=True)):
        if json.dumps(json.loads(ref), **kw) + "\n" == ref: return lambda o: json.dumps(o, **kw) + "\n"
    raise SystemExit("json format not detected")
dj = dump(js, raw)
# ---- RECONCILIATION.md
md = rep(md, "**Read at:** Math- `main` `aa288385af9392c509233c25a0f4012c9ba023e5` (30 September 2026); main `STATUS.md` at `7bbbdc9`.",
"**Read at:** Math- `main` `aa288385af9392c509233c25a0f4012c9ba023e5` (30 September 2026); main `STATUS.md` at `7bbbdc9`.\n"
"Re-verified at `ec6db8c` (Math-#151 merged, 30 September 00:56Z): the ten inventoried sources and the eight live nodes of\n"
"§4 are unchanged; the live graph now also carries `math.d5-component.remote-window-proof` (`PROVED_REVIEWED`, fingerprint\n"
"`a332bae9…`, `review_basis` \"STATUS D4 (main#76, external)\") beside the still-`AUTHOR_SIDE_CANDIDATE`\n"
"`math.rn-fixed-remote-window`: the D4 dual-node inconsistency is now live, and the `D4_DUPLICATE` check binds it.")
md = rep(md, "**Revision:** v1.1 records the two xAI reads of `670e4ae` and makes the Math-#151-first ordering of the D4 flips\nexplicit; transitions unchanged.",
"**Revision:** v1.1 records the two xAI reads of `670e4ae` and makes the Math-#151-first ordering of the D4 flips\nexplicit; transitions unchanged. v1.2 re-verifies at `ec6db8c` after Math-#151 merged, binds the now-live duplicate D4\nnode (`D4_DUPLICATE` check, `duplicate-drift` mutant) and records that the HOLD's Math-#151 precondition is met;\ntransitions unchanged.")
md = rep(md, "This record agrees\non every point; v1.1 makes the #151-first ordering explicit in §4 and `execution_order`.",
"This record agrees\non every point; v1.1 makes the #151-first ordering explicit in §4 and `execution_order`. Math-#151 merged at `ec6db8c`\n(30 September 00:56Z; main#207 closed at 00:57Z), so the first HOLD condition is met; the second, that the executing\nlane record the single-provider rows as such, is already stated per row in §4 and in `PROPOSED_TRANSITIONS.json` and is\nthat lane's to carry out.")
md = rep(md, "accepted. The register is also internally inconsistent on D4: the D5 proposal (Math-#138, executed by Math-#151)\ncreates `math.d5-component.remote-window-proof` for the same bytes as `PROVED_REVIEWED`, with `review_basis`\n\"STATUS D4 (main#76, external), nonauthor, ACCEPT, full\". The same review either suffices for those bytes or it does\nnot; this record proposes that it does, on the live node, and asks the executing lane to collapse the duplicate (§7).",
"accepted. The register is also internally inconsistent on D4: since `ec6db8c` the live graph carries\n`math.d5-component.remote-window-proof` (Math-#151, executing the D5 proposal of Math-#138) for the same bytes as\n`PROVED_REVIEWED`, with `review_basis` \"STATUS D4 (main#76, external), nonauthor, ACCEPT, full\", beside\n`math.rn-fixed-remote-window` at `AUTHOR_SIDE_CANDIDATE`. The same review either suffices for those bytes or it does\nnot; this record proposes that it does, on the older live node, so that the two agree; collapsing the pair is a\nseparate one-edge follow-up for the executing lane (§7). The `D4_DUPLICATE` check binds the duplicate's classification,\nfingerprint, source and review basis to the D4 transition.")
md = rep(md, "the D4 theorem node\nand its region are executed only after Math-#151 is on main, so that the D4 bytes are represented in one direction\n(collapse `math.d5-component.remote-window-proof` onto the live node, or reference it) before the flip (xAI\n5901912476, 5360240245).",
"the D4 theorem node\nand its region are executed only after Math-#151 is on main (satisfied at `ec6db8c`): the D4 bytes are now carried by\ntwo live nodes on the same review, and the flip aligns the older one with the newer one; the executing lane may then\ncollapse `math.d5-component.remote-window-proof` onto the live node or keep both (xAI 5901912476, 5360240245).")
i = md.index("\nNine mutants must fail:")
assert md[:i].endswith("\n"), "no blank line before mutants"
md = md[:i].rstrip("\n") + "\n- **D4_DUPLICATE.** The live graph carries `math.d5-component.remote-window-proof` as `PROVED_REVIEWED`,\n  non-controlling, with the same SHA256 fingerprint and source as the D4 transition and as the inventory, a `review_basis`\n  citing main#76 with verdict ACCEPT, `component_of` `math.d5-pin-neighborhood-first-moment`; and the older live node\n  `math.rn-fixed-remote-window` still carries that fingerprint at `AUTHOR_SIDE_CANDIDATE`, the lag this record aligns.\n" + md[i:]
md = rep(md, "Nine mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,\n`executed-flag`, `skip-interface-premise`, `holder-reversed`, `baseline-drift`.",
"Ten mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-review-source`, `controlling-true`,\n`executed-flag`, `skip-interface-premise`, `holder-reversed`, `baseline-drift`, `duplicate-drift`.")
md = rep(md, "- **Math-#151 (Codex, D5 fold):** its proposal creates `math.d5-component.remote-window-proof` for the D4 bytes. If\n  this record is executed, that node should be collapsed to a `required` edge onto the live\n  `math.rn-fixed-remote-window` (coordination note posted on #151, comment 5901742635); if #151 lands first, the\n  collapse is a one-edge follow-up.",
"- **Math-#151 (Codex, D5 fold; merged at `ec6db8c`, 30 September 00:56Z, final head `609e559`):** landed as proposed,\n  so the live graph carries `math.d5-component.remote-window-proof` (`PROVED_REVIEWED`, `component_of`\n  `math.d5-pin-neighborhood-first-moment`, `review_basis` main#76) for the D4 bytes beside `math.rn-fixed-remote-window`\n  (`AUTHOR_SIDE_CANDIDATE`); the coordination note (comment 5901742635) was not taken up before the merge. This record's\n  D4 flip aligns the two; collapsing the component node to a `required` edge onto the live node is a one-edge follow-up\n  for the executing lane, not proposed here.")
md = rep(md, "- **main#207:** register coordination issue; this record is a separate, additive proposal for the same non-Claude lane.",
"- **main#207 (closed 30 September 00:57Z, after Math-#151 merged):** the delivery note for this record is comment\n  5901889845. The executing lane for this record remains non-Claude; any new coordination thread is that lane's to open.")
# ---- PROPOSED_TRANSITIONS.json
js["reverified_at"] = ("ec6db8c (Math- main after Math-#151 merged 2026-09-30T00:56:58Z; the ten inventoried sources and the eight live "
                       "nodes unchanged; math.d5-component.remote-window-proof live as PROVED_REVIEWED with fingerprint " + FP +
                       " beside math.rn-fixed-remote-window at AUTHOR_SIDE_CANDIDATE)")
js["live_duplicate"] = {"node": "math.d5-component.remote-window-proof", "since": "ec6db8c (Math-#151)", "classification": "PROVED_REVIEWED",
    "controlling": False, "fingerprint": FP, "source": "frontiers/remote_window_20260924/PROOF.md",
    "component_of": "math.d5-pin-neighborhood-first-moment", "review_basis": "STATUS D4 (main#76, external), nonauthor, ACCEPT, full",
    "relation_to_this_record": "same bytes and same review as the math.rn-fixed-remote-window transition; the flip aligns the older node with it; collapsing the pair is a one-edge follow-up for the executing lane, not proposed here",
    "check": "D4_DUPLICATE; mutant duplicate-drift"}
eo = js["execution_order"]
assert eo[2].startswith("3. only after Math-#151 is on main")
eo[2] = ("3. Math-#151 is on main (ec6db8c): the D4 bytes are carried by two live nodes (math.rn-fixed-remote-window AUTHOR_SIDE_CANDIDATE; "
         "math.d5-component.remote-window-proof PROVED_REVIEWED; same fingerprint, same main#76 review); execute math.rn-fixed-remote-window -> "
         "PROVED_REVIEWED, then math.rn-region.fixed-remote; collapsing the component node onto the live node is an optional one-edge follow-up")
for t in js["transitions"]:
    if t["node"] == "math.rn-fixed-remote-window":
        assert t["notes"].endswith("Single nonauthor provider.")
        t["notes"] += " At ec6db8c that duplicate is live (Math-#151 merged); the D4_DUPLICATE check binds it to this transition."
for r in js["review_record"]:
    r["hold_precondition_math_151"] = "satisfied: Math-#151 merged at ec6db8c (2026-09-30T00:56:58Z)"
js["revision"] = {"version": "v1.2", "applies": js["revision"]["applies"] + ["re-verified at ec6db8c (Math-#151 merged); live D4 duplicate bound (D4_DUPLICATE check, duplicate-drift mutant); HOLD precondition on Math-#151 recorded as met"],
                  "unchanged": "all eight transitions, inventory"}
# ---- alignment_check.py
py = rep(py, '"skip-interface-premise", "holder-reversed", "baseline-drift")', '"skip-interface-premise", "holder-reversed", "baseline-drift", "duplicate-drift")')
py = rep(py, 'D1 = "math.uniform-matrix-cap-lifetime"\n', 'D1 = "math.uniform-matrix-cap-lifetime"\nD4_DUP = "math.d5-component.remote-window-proof"\n')
fn = '''def check_d4_duplicate(root, spec):
    """Since ec6db8c (Math-#151) the live graph carries the D4 bytes twice; bind the duplicate to the D4 transition."""
    live = json.loads((root / GRAPH).read_text(encoding="utf-8"))["nodes"]
    dup = live.get(D4_DUP)
    old = live.get(D4_NODE)
    t = next(t for t in spec["transitions"] if t["node"] == D4_NODE)
    want = spec["live_duplicate"]
    fp = t["proposed"]["fingerprint"]
    if MUT == "duplicate-drift":
        fp = "1" * 64
    if not isinstance(dup, dict) or not isinstance(old, dict):
        return False
    ok = dup.get("classification") == "PROVED_REVIEWED" and dup.get("controlling") is False
    ok &= dup.get("fingerprint") == fp == want["fingerprint"] == INVENTORY[t["proposed"]["source"]][0]
    ok &= dup.get("source") == t["proposed"]["source"] == want["source"]
    ok &= dup.get("component_of") == want["component_of"]
    ok &= any("main#76" in str(b.get("review")) and b.get("verdict") == "ACCEPT" for b in dup.get("review_basis", []))
    ok &= old.get("fingerprint") == dup.get("fingerprint") and old.get("classification") == t["current"]["classification"]
    return bool(ok)


def main():'''
py = rep(py, "def main():", fn)
py = rep(py, '"TRANSITIONS": trans_ok, "GATE": gate_ok, "INTERFACE": check_interface(), "NEGATIVES": check_negatives(root)}',
          '"TRANSITIONS": trans_ok, "GATE": gate_ok, "INTERFACE": check_interface(), "NEGATIVES": check_negatives(root),\n              "D4_DUPLICATE": check_d4_duplicate(root, spec)}')
py = rep(py, "len(checks) == 7", "len(checks) == 8")
py = rep(py, '"scope": "identity, verdict-row, live-baseline, transition-shape, hard-gate replay, interface arithmetic "\n                               "and filesystem-negative checks; no mathematics is re-proved; nothing is written"',
          '"scope": "identity, verdict-row, live-baseline, transition-shape, hard-gate replay, interface arithmetic, "\n                               "filesystem-negative and live D4-duplicate checks; no mathematics is re-proved; nothing is written"')
wf = rep(wf, "'holder-reversed', 'baseline-drift']", "'holder-reversed', 'baseline-drift', 'duplicate-drift']")
md_p.write_text(md); js_p.write_text(dj(js)); py_p.write_text(py); wf_p.write_text(wf)
print("edits written")
