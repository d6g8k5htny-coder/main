import json, hashlib, pathlib, re, subprocess, sys
S = pathlib.Path(".").resolve()
WT = S / "wtres"; R = WT / "reviews/c6_residual_closure_20260930"; V = S / "v13"

def sha(b): return hashlib.sha256(b).hexdigest()
def blob(b): return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()
def sub1(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)

# ---------- 1. reviewed section file ----------
block = (V / "REVIEWED_SECTION_3_b3fac79.body").read_bytes()
assert block.startswith(b"## 3. ") and b"Route C" not in block and block.endswith(b"Exponent ledgers checked exactly.\n")
block_sha = sha(block)
header = (b"<!-- Verbatim copy of RECONCILIATION.md section 3, lines 1-34 ((D1), Route A, Route B), at head "
          b"b3fac79875f28bacd135c0aa5e65a47a41ae0fdf, as read by Math-#173 review 5360645884 (OpenAI / GPT-5.6 Sol). "
          b"Not edited; the checker's REVIEWED test binds the text after this comment to the proposal's reviewed_text_sha256 "
          b"and requires it verbatim in the current RECONCILIATION.md. -->\n\n")
(R / "REVIEWED_SECTION_3_b3fac79.md").write_bytes(header + block)

# ---------- 2. EXTERNAL_REVIEWS.md ----------
sol = (V / "sol_5360645884_body.txt").read_bytes()
sol_sha = sha(sol); assert len(sol) == 2781
E = (R / "EXTERNAL_REVIEWS.md").read_text(encoding="utf-8")
E = sub1(E, "# External review preserved for C6-RESIDUAL-CLOSURE-20260930-v1", "# External reviews preserved for C6-RESIDUAL-CLOSURE-20260930-v1")
E = sub1(E, "proposed graph node `math.c6r-component.cluster-law-acceptance`).",
         "proposed graph node `math.c6r-component.cluster-law-acceptance`; since v1.3 also the preserved copy of Math-#173 review\n5360645884, the non-Claude read of this record's §3 on which `math.c6r-component.residual-closure-record` rests).")
E = sub1(E, "| 1 | Math-#159 `frontiers/c6_cluster_law_20260929/PROOF.md`, blob `ba492c8e…` | Math-#159 | 5360192822 |",
         "| 1 | Math-#159 `frontiers/c6_cluster_law_20260929/PROOF.md`, blob `ba492c8e…` | Math-#159 | 5360192822 |")
idx_row2 = ("| 2 | this record, `RECONCILIATION.md` §3 at head `b3fac79` ((D1), Route A, Route B; preserved in `REVIEWED_SECTION_3_b3fac79.md`) "
            "| Math-#173 | 5360645884 | `d6g8k5htny-coder` (owner account; the body names OpenAI / GPT-5.6 Sol) "
            "| OpenAI / GPT-5.6 Sol, source-exposed nonauthor (integrator of Math-#162/#166) | `b3fac79875f28bacd135c0aa5e65a47a41ae0fdf` "
            "| 2026-09-30T02:06:27Z | \"MATHEMATICAL VERDICT ON §3: ACCEPT at the stated fixed-d, fixed-torus, compact-positive-gap window scope.\" |\n")
anchor = "\nThe earlier reviews on the same pull request (5359646638 on v1.0,"
assert E.count(anchor) == 1
E = E.replace(anchor, idx_row2 + anchor)
sec2 = ("\n\n## 2. Math-#173 review 5360645884 (OpenAI / GPT-5.6 Sol, ACCEPT of §3 at head `b3fac79`)\n\n"
        "URL: https://github.com/d6g8k5htny-coder/Math-/pull/173#pullrequestreview-5360645884 · posted by `d6g8k5htny-coder` · "
        "commit `b3fac79875f28bacd135c0aa5e65a47a41ae0fdf` · submitted 2026-09-30T02:06:27Z · state COMMENTED (the verdict is in the body) · "
        f"body {len(sol)} bytes as read back through the connector (JSON-decoded, no trailing newline), SHA256 `{sol_sha}`\n\n"
        "What it covers: §3 as it stood at `b3fac79`, i.e. (D1), Route A and Route B (items 1–5 of the body); the four sources re-read on\n"
        "main `fa1cf7b`. It does not cover Route C (added in v1.1; OpenAI-authored, checked in Math-#160 comment 5902616692) or the\n"
        "Theorem L paragraph (Math-#166's own reviewed statement). Its amend list (Codex findings; stale \"pending\" wording; promotion of\n"
        "the record node on the reviewed-record path with the #166 path as a distinct alternative) is answered in v1.3\n"
        "(`RECONCILIATION.md` §§5–7). The reviewer states source exposure and that the read is not organizational independence.\n"
        "Codex review 5360636576 (three inline engineering threads 4140125558, 4140125562, 4140125565 on `residual_check.py`) is cited\n"
        "in `RECONCILIATION.md` §7 and answered in v1.3; it is not preserved here.\n\n"
        "`````text\n" + sol.decode("utf-8") + "\n`````\n")
E = E.rstrip("\n") + sec2
(R / "EXTERNAL_REVIEWS.md").write_text(E, encoding="utf-8")

# ---------- 3. RECONCILIATION.md ----------
M = (R / "RECONCILIATION.md").read_text(encoding="utf-8")
M = sub1(M, "`REVIEW_RECORD.md` extended and `REVIEW_RECORD.json` added; register surfaces unchanged).",
         "`REVIEW_RECORD.md` extended and `REVIEW_RECORD.json` added; register surfaces unchanged); main is still `fe2c3ff` at v1.3,\n"
         "and the integrator refreshed Math-#160's branch onto it (`dd4f4f8`, its six packet blobs preserved from `eba6045`).")
M = sub1(M, "review 5360227991. Pickup: Math-#160 comment 5902470285; delivery 5902585446.",
         "review 5360227991; v1.3 at `fe2c3ff` answers Math-#173's reviews: this record's own node is promoted on the reviewed-record\n"
         "path (OpenAI / GPT-5.6 Sol 5360645884, a non-Claude bounded read of §3 at head `b3fac79`, whose paragraphs are preserved\n"
         "verbatim in `REVIEWED_SECTION_3_b3fac79.md` and checked unchanged), the two evidence paths are modelled separately, and\n"
         "the checker accepts the installed state and replays the hard gate with source snapshots (Codex 5360636576). Pickup:\n"
         "Math-#160 comment 5902470285; delivery 5902585446.")
M = sub1(M, "  Math-#159 / [DL] / [C6], source-exposed on [RC].\n",
         "  Math-#159 / [DL] / [C6], source-exposed on [RC].\n"
         "- Math-#173's own reviews: OpenAI / GPT-5.6 Sol 5360645884, a bounded non-Claude read of §3 at head `b3fac79` (ACCEPT at\n"
         "  the stated scope; source-exposed to the OpenAI side of the chain and the integrator of Math-#162/#166; not organizational\n"
         "  independence, by its own statement), and OpenAI/Codex 5360636576 (three engineering findings on the checker, all repaired\n"
         "  in v1.3). The paragraphs that read covers ((D1), Route A, Route B) are preserved byte-for-byte in\n"
         "  `REVIEWED_SECTION_3_b3fac79.md`; Route C and the Theorem L paragraph were not part of it.\n")
M = sub1(M, "| the preserved 5360192822 |\n",
         "| the preserved 5360192822 and 5360645884 |\n"
         f"| this packet | `reviews/c6_residual_closure_20260930/REVIEWED_SECTION_3_b3fac79.md` | see `SOURCE_FILES.json`; reviewed text (after the provenance comment) 2779 · `{block_sha}` | verbatim copy of §3 at `b3fac79` | read by OpenAI 5360645884 (preserved in `EXTERNAL_REVIEWS.md`) | the paragraphs that read covers, checked unchanged in §3 (`REVIEWED`) |\n")
M = sub1(M, "regression, no Kac–Rice step, no new estimate. Since v1.1 the direct statement [TSL] carries the flip and this record's\n"
            "own node is supporting; a non-Claude read of this section remains welcome (the reconciler is source-exposed, §0) but\n"
            "is not gating.\n",
         "regression, no Kac–Rice step, no new estimate. The direct statement [TSL] and this section are two separate evidence\n"
         "paths (§5). This section's (D1), Route A and Route B paragraphs were read by OpenAI / GPT-5.6 Sol (Math-#173 review\n"
         "5360645884, at head `b3fac79`, ACCEPT at the stated scope; the reconciler is source-exposed, §0); those paragraphs are\n"
         "preserved byte-for-byte in `REVIEWED_SECTION_3_b3fac79.md`, and `REVIEWED` checks that they stand unchanged above. Route C\n"
         "is OpenAI-authored and Claude-checked (5902616692); the Theorem L paragraph is [TSL]'s own reviewed statement. A revision\n"
         "that alters the reviewed paragraphs must obtain a new read or return this record's node to `AUTHOR_SIDE_CANDIDATE`.\n")
M = sub1(M, "| `math.c6r-component.residual-closure-record` | this packet's `RECONCILIATION.md` | reading-rule record: the §3 corollary and Route C | Anthropic (this session; source-exposed) | `AUTHOR_SIDE_CANDIDATE` (the reconciler's own text; a non-Claude read is welcome); **supporting only** since v1.1, because [TSL] Theorem L is the direct statement |",
         "| `math.c6r-component.residual-closure-record` | this packet's `RECONCILIATION.md` | reading-rule record: the §3 corollary ((D1), Routes A, B) and Route C | Anthropic (this session; source-exposed) | **OpenAI / GPT-5.6 Sol 5360645884** (bounded non-Claude read of §3 at `b3fac79`; reviewed text preserved and checked, `REVIEWED`); Route C: OpenAI-authored, checked 5902616692. `PROVED_REVIEWED` on the reviewed-record path since v1.3 |")
M = sub1(M, "Edges: from the residual node to each component, `required: true` (`requires_evidence`), except the record node, which\n"
            "is supporting (`required: false`) since [TSL] is bound. From the old node (deferred transition): `required` edges to the\n"
            "residual node and to Math-#160's aggregate node.\n",
         "Edges: from the residual node to each of the ten components, `required: true` (`requires_evidence`). From the old node\n"
         "(deferred transition): `required` edges to the residual node and to Math-#160's aggregate node.\n\n"
         "**Two evidence paths** (`PROPOSED_TRANSITIONS.json` `evidence_paths`), each sufficient for (Res) on its own: the **direct\n"
         "statement** ([TSL] Theorem L with its in-repo review record and JSON binding: three nodes) and the **reviewed corollary**\n"
         "(§3 from [SC], [CL], [PALM] and this record, read by 5360645884: seven nodes). The residual's required edges take the\n"
         "union, so that a later byte change on either path holds the residual (fail-closed); `GATE` replays each path alone (the\n"
         "other path's edges supporting) as well as the union, and replays the negative in which the record node is required\n"
         "without its read: the gate then holds the residual on exactly that node.\n")
M = sub1(M, "(0) satisfied at `ab13a08f`: Math-#166 is on main and its\n"
            "Theorem L is a required component node; a non-Claude read of §3 remains welcome but is not gating; (1) the component nodes and their edges; (2) the residual node\n",
         "(0) satisfied twice: at `ab13a08f`\n"
         "Math-#166's Theorem L is on main as a required component (direct path), and at `b3fac79` OpenAI 5360645884 read §3\n"
         "(reviewed-corollary path); (1) the component nodes and their edges; (2) the residual node\n")
M = sub1(M, "**Reverse-impact.** Every component node carries the SHA256 of its source; a later change to [SC], [CL], [PALM], the\n"
            "review record, the source map or the preserved acceptance reaches the residual node (and, once executed, the old\n"
            "node) through `reverse_impact_between` (`GATE` replays it in both executions).\n",
         "**Reverse-impact.** Every component node carries the SHA256 of its source; `GATE` calls `reverse_impact_between` with exact\n"
         "source snapshots of both graph states (every node: `record_only`, `external_unresolved`, `blob` with bytes and SHA256,\n"
         "or `tree`), asserts `source_snapshots_supplied`, and then, for each component in turn, changes only that component's\n"
         "snapshot digest and checks that the residual node (and the old node, once it depends on the residual) is impacted. A\n"
         "later byte change to [SC], [CL], [PALM], [TSL], a review record, the source map, the preserved reviews or this record\n"
         "therefore reaches the residual through the graph, not through the fingerprint strings alone.\n")
M = sub1(M, "- **IDENTITIES.** All twelve inventoried files", "- **IDENTITIES.** All thirteen inventoried files")
M = sub1(M, "and the JSON binding's native id, proof path and body digest; the review id, verdict line and mutability label in `EXTERNAL_REVIEWS.md`.\n",
         "and the JSON binding's native id, proof path and body digest; the review ids, verdict lines, the `b3fac79` head and the\n"
         "  mutability label in `EXTERNAL_REVIEWS.md`; the heading and the Route A / Route B headers of the reviewed-section file.\n")
M = sub1(M, "- **LIVE.** `math.rn-region.witness-collision` is a live `OPEN_ACTIVE` region with its recorded fingerprint (or\n"
            "  `PROVED_REVIEWED` after execution); the residual node is absent or `OPEN_ACTIVE` with Math-#160's fingerprint (or\n"
            "  `PROVED_REVIEWED` after execution); the selector lists the region; no live fingerprinted node carries a component\n"
            "  source of §5 other than `math.c6-component.palm-proof` if Math-#160 has been executed.\n",
         "- **LIVE.** `math.rn-region.witness-collision` is a live `OPEN_ACTIVE` region with its recorded fingerprint (or\n"
         "  `PROVED_REVIEWED` after execution); the residual node is absent, `OPEN_ACTIVE` with Math-#160's fingerprint, or\n"
         "  `PROVED_REVIEWED`; the selector lists the region. A live node on a component source is accepted only if it is the\n"
         "  proposed node on every proposed key (the palm node on its cross-record keys: kind, source, fingerprint,\n"
         "  classification, controlling); anything else on such a source, or anything else under a proposed id, is a mismatch. If\n"
         "  the residual is live `PROVED_REVIEWED`, every component, every edge and the reading rule must be present as proposed.\n"
         "  The same test runs on the executed graph built from the proposal (installed state, simulated) in every run, so that\n"
         "  execution cannot turn this check red (Codex 4140125558).\n"
         "- **REVIEWED.** The record node's first review basis is Math-#173 review 5360645884 (provider OpenAI, ACCEPT, head\n"
         "  `b3fac79`); `REVIEWED_SECTION_3_b3fac79.md`, after its one-line provenance comment, has the SHA256 the proposal states,\n"
         "  begins at the §3 heading, contains Route A and Route B and not Route C, and is a verbatim substring of the current\n"
         "  `RECONCILIATION.md`; the review id and head are in `EXTERNAL_REVIEWS.md`.\n")
M = sub1(M, "  record node `AUTHOR_SIDE_CANDIDATE` and reached by a supporting edge only; every required premise of the residual node\n"
            "  `PROVED_REVIEWED`; the old-node transition marked deferred with its two presuppositions; the palm node marked\n"
            "  cross-record with Math-#160's id.\n",
         "  every component `PROVED_REVIEWED` with at least one review basis and the residual's required edges covering all ten;\n"
         "  the two evidence paths partition the components, the record node on the corollary path and Theorem L on the direct\n"
         "  path; the old-node transition marked deferred with its two presuppositions; the palm node marked cross-record with\n"
         "  Math-#160's id. The builds are idempotent: installed components and edges are not re-created, and the installed state\n"
         "  rebuilds to itself.\n")
M = sub1(M, "- **GATE.** In both executions (residual node created directly; residual node pre-existing `OPEN_ACTIVE` as\n"
            "  Math-#160 proposes), the proposed graph passes `validate_graph_fail_closed`, `closure_report` gives `gate_ok` with no\n"
            "  illegal controlling node, and `reverse_impact_between` names the residual node and the components as changed and\n"
            "  impacted; the deferred old-node transition is not applied (its presupposition is absent live) and is reported.\n",
         "- **GATE.** In every execution shape (interim; final; residual node pre-existing `OPEN_ACTIVE` as Math-#160 proposes;\n"
         "  each evidence path alone; the installed state, simulated; and the negative with the record node required but unread),\n"
         "  the proposed graph passes `validate_graph_fail_closed`, `closure_report` gives `gate_ok` with no illegal controlling\n"
         "  node, `reverse_impact_between` runs with source snapshots of both states and names the residual node and the new\n"
         "  components as changed and impacted (nothing, in the installed state); the residual is not held except in the negative,\n"
         "  where the gate holds it on exactly the record node; a single-component source drift reaches the residual for every\n"
         "  component; the deferred old-node transition is not applied (its presupposition is absent live) and is reported.\n")
M = sub1(M, "Ten mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-required-edge`, `executed-flag`,\n"
            "`controlling-true`, `tail-reversed`, `pair-identity-broken`, `route-c-broken`, `drop-review-needle`.\n",
         "Fourteen mutants must fail: `allow-symlink`, `no-hash`, `stale-fingerprint`, `drop-required-edge`, `executed-flag`,\n"
         "`controlling-true`, `tail-reversed`, `pair-identity-broken`, `route-c-broken`, `drop-review-needle`, and since v1.3\n"
         "`installed-drift` (an installed component with another fingerprint must fail `LIVE`), `snapshot-omitted` (the\n"
         "reverse-impact replay without snapshots must fail `GATE`), `record-unreviewed` (the record node `PROVED_REVIEWED` with no\n"
         "review basis must fail `TRANSITIONS` and `REVIEWED`), `reviewed-text-drift` (a changed reviewed text must fail `REVIEWED`).\n")
M = sub1(M, "  and composes with either execution of its §7. Math-#160 is not changed.\n",
         "  and composes with either execution of its §7. Math-#160 is not changed by this record; the integrator refreshed its\n"
         "  branch onto `fe2c3ff` (`dd4f4f8`) with the six packet blobs preserved.\n"
         "- **Math-#173 reviews (on v1.0 `b3fac79`):** OpenAI/Codex 5360636576, three findings: (i) `LIVE` rejected the installed\n"
         "  components (4140125558), repaired: the installed state is accepted when exact and simulated in every run; (ii) the\n"
         "  reverse-impact replay ran without source snapshots (4140125562), repaired: snapshots of both states and a\n"
         "  per-component drift test; (iii) the `final` build promoted this record without a read on the Math-#166 path\n"
         "  (4140125565), repaired: the two paths are modelled separately, the record node's promotion rests on 5360645884, and\n"
         "  the unread case is replayed as a negative. OpenAI / GPT-5.6 Sol 5360645884: ACCEPT of §3 at the stated scope with the\n"
         "  amend list above; the record node is promoted on that reviewed-record path, the #166 path stays a distinct alternative\n"
         "  with `TWO_SCALE_LAW.md` and its exact review binding as evidence, and the \"pending\" wording is gone.\n")
(R / "RECONCILIATION.md").write_text(M, encoding="utf-8")
rec_b = (R / "RECONCILIATION.md").read_bytes(); assert block in rec_b
ext_b = (R / "EXTERNAL_REVIEWS.md").read_bytes()
rvw_b = (R / "REVIEWED_SECTION_3_b3fac79.md").read_bytes()

# ---------- 4. PROPOSED_TRANSITIONS.json ----------
raw = (R / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8")
s = json.loads(raw)
fmt = None
for kw in (dict(indent=2, ensure_ascii=False), dict(indent=2), dict(indent=2, ensure_ascii=False, sort_keys=True)):
    if json.dumps(s, **kw) + "\n" == raw: fmt = kw
assert fmt, "json format"
RES = "math.rn-region.witness-collision.leading-mass-localization"; REC = "math.c6r-component.residual-closure-record"
s["residual_definition"]["defined_in"] = ("Math-#160 reviews/c6_witness_collision_reconciliation_20260929/RECONCILIATION.md section 5 "
    "(head eba6045; refreshed by the integrator onto fe2c3ff as dd4f4f8 with the packet bytes preserved; open)")
p = s["transitions"][0]["proposed"]
p["explicit_limits"] = sub1(p["explicit_limits"], "no configuration or moving-cutoff statement (Math-#166, pending)",
    "no configuration statement (Math-#166 Theorem T, not consumed) and no d_TV statement (Math-#166 (V1), not consumed); the moving-cutoff form is Theorem L at its own scope")
p["notes"] = sub1(p["notes"], "Route C pair-count corollary. C6 residual closure v1.1.",
    "Route C pair-count corollary. Two evidence paths: the direct statement (Theorem L) and the reviewed corollary (section 3, read by OpenAI 5360645884). C6 residual closure v1.3.")
assert REC not in p["reading_rule"]; p["reading_rule"].append(REC)
p["review_sources"].append({"ref": "Math-#173 review 5360645884 (preserved in EXTERNAL_REVIEWS.md)", "provider": "OpenAI (GPT-5.6 Sol; source-exposed, integrator of Math-#162/#166)",
                            "verdict": "ACCEPT section 3 ((D1), Route A, Route B) at the stated scope, at head b3fac79"})
t0 = s["transitions"][0]; assert REC not in t0["required_premises_after"]; t0["required_premises_after"].append(REC)
t0["notes"] = ("Created PROVED_REVIEWED if Math-#160 step 1 has not run; moved OPEN_ACTIVE -> PROVED_REVIEWED if it has. Every required "
               "premise is PROVED_REVIEWED on both evidence paths (direct statement: Math-#166 Theorem L; reviewed corollary: section 3, "
               "read by OpenAI 5360645884); the required edges take the union (fail-closed).")
for n in s["proposed_graph_nodes"]:
    if n["id"] == REC:
        n["classification"] = "PROVED_REVIEWED"
        n["fingerprint"] = sha(rec_b)
        n["scope"] = ("section 3: the corollary (D1) + steps 1-3 (Route A) and Route B, read by 5360645884 at b3fac79 and preserved verbatim in "
                      "REVIEWED_SECTION_3_b3fac79.md; Route C (OpenAI-authored, checked 5902616692)")
        n["review_basis"] = [
            {"review": "Math-#173 review 5360645884", "provider": "OpenAI (GPT-5.6 Sol; source-exposed to the OpenAI side of the chain; integrator of Math-#162/#166; no authorship of this record)",
             "verdict": "ACCEPT section 3 at the stated fixed-d, fixed-torus, compact-positive-gap window scope",
             "depth": "bounded read of section 3 at head b3fac79 with the four sources re-read on main fa1cf7b",
             "reviewed_head": "b3fac79875f28bacd135c0aa5e65a47a41ae0fdf",
             "reviewed_text": "reviews/c6_residual_closure_20260930/REVIEWED_SECTION_3_b3fac79.md",
             "reviewed_text_sha256": block_sha,
             "covers": "(D1), Route A, Route B; not Route C and not the Theorem L paragraph",
             "preserved_in": "reviews/c6_residual_closure_20260930/EXTERNAL_REVIEWS.md section 2", "body_sha256": sol_sha, "submitted_at": "2026-09-30T02:06:27Z"},
            {"review": "Math-#160 comment 5902616692", "provider": "Anthropic Claude (this session; provider-distinct nonauthor of the OpenAI-authored corollary 5901994240; source-exposed on RC)",
             "verdict": "ACCEPT Route C at the stated scope", "depth": "bounded", "covers": "Route C only"}]
        n.pop("required_until", None)
        n["notes"] = ("reviewed-corollary evidence path; alternative to the direct-statement path; the promotion rests on 5360645884 and on the "
                      "reviewed paragraphs standing unchanged (REVIEWED); a revision that alters them must obtain a new read or return this node to AUTHOR_SIDE_CANDIDATE")
    if n["id"] == "math.c6r-component.cluster-law-acceptance":
        n["fingerprint"] = sha(ext_b)
        n["scope"] = "preserved transcriptions (5360192822 for [CL]; 5360645884 for this record's section 3); mutable external evidence, labelled; not an additional review or provider"
for e in s["proposed_graph_edges"]:
    if e["from"] == RES and e["to"] == REC:
        assert e["required"] is False
        e["required"] = True; e["relation"] = "requires_evidence"
direct = ["math.c6r-component.two-scale-law-proof", "math.c6r-component.two-scale-review-record", "math.c6r-component.two-scale-review-binding"]
corollary = [c["id"] for c in s["proposed_graph_nodes"] if c["id"] not in direct]
s["evidence_paths"] = {
    "direct_statement": {"components": direct, "statement": "Math-#166 Theorem L (L1) at q = 2 with delta_r = R r, and (L2)",
                         "review": "Claude 5360227991 (in-repo REVIEW_RECORD.md and REVIEW_RECORD.json body binding); xAI 5901945424 as reported"},
    "reviewed_corollary": {"components": corollary, "statement": "RECONCILIATION.md section 3: (D1) with Route A (Math-#162 (26)/(18)/(20), PALM Theorem Q), Route B (Math-#159 Corollary Lambda / Proposition 4.4 / Lemma 5.2), Route C",
                           "review": "OpenAI / GPT-5.6 Sol 5360645884 at b3fac79 ((D1), Route A, Route B; reviewed text preserved); Route C checked in 5902616692"},
    "rule": ("either path alone establishes (Res) at the section 4 scope; the residual node's required edges take the union so that a later "
             "source change on either path holds the residual (fail-closed); GATE replays each path alone, the union, and the negative with the record node required but unread")}
s["execution_order"][0] = ("0. satisfied twice: (a) at ab13a08f Math-#166 Theorem L is on main as a required component (direct-statement path); "
    "(b) at b3fac79 OpenAI / GPT-5.6 Sol 5360645884 read section 3 (reviewed-corollary path), so math.c6r-component.residual-closure-record is "
    "PROVED_REVIEWED on that basis; either path alone suffices and the required edges take the union")
s["review_record"].extend([
    {"lane": "OpenAI/Codex", "id": 5360636576, "kind": "review of v1.0 at b3fac79: three engineering findings (4140125558 LIVE after execution; 4140125562 source snapshots; 4140125565 record promoted without a read on the Math-#166 path)", "disposition": "all repaired in v1.3"},
    {"lane": "OpenAI (GPT-5.6 Sol)", "id": 5360645884, "kind": "bounded non-Claude read of section 3 at b3fac79 (sources re-read at fa1cf7b)",
     "verdict": "ACCEPT at the stated fixed-d, fixed-torus, compact-positive-gap window scope; amend list repaired in v1.3",
     "preserved_in": "EXTERNAL_REVIEWS.md section 2", "reviewed_text": "REVIEWED_SECTION_3_b3fac79.md"}])
s["exposure"] += "; Math-#173's own reviewers are OpenAI Sol (source-exposed; integrator) and OpenAI/Codex"
s["revision"]["version"] = "v1.3"
s["revision"]["applies"].append("fe2c3ff (v1.3): Math-#173 reviews answered: record node PROVED_REVIEWED on the reviewed-record path (5360645884; reviewed section preserved and checked), evidence_paths, installed-state acceptance in LIVE, source snapshots and per-component drift in GATE, four new mutants")
s["reverified_at"] += "; unchanged at v1.3 (Math-#160 refreshed onto fe2c3ff as dd4f4f8 by the integrator)"
(R / "PROPOSED_TRANSITIONS.json").write_text(json.dumps(s, **fmt) + "\n", encoding="utf-8")

# ---------- 5. checker ----------
C = (V / "residual_check.py").read_text(encoding="utf-8")
for tag, b in (("EXT", ext_b), ("REC", rec_b), ("RVW", rvw_b)):
    C = sub1(C, f'"{tag}_SHA"', f'"{sha(b)}"'); C = sub1(C, f'"{tag}_BLOB"', f'"{blob(b)}"')
(R / "residual_check.py").write_text(C, encoding="utf-8")

# ---------- 6. workflow ----------
W = (WT / ".github/workflows/c6-residual-closure.yml").read_text(encoding="utf-8")
W = sub1(W, "'route-c-broken', 'drop-review-needle']", "'route-c-broken', 'drop-review-needle', 'installed-drift', 'snapshot-omitted', 'record-unreviewed', 'reviewed-text-drift']")
(WT / ".github/workflows/c6-residual-closure.yml").write_text(W, encoding="utf-8")
print("block_sha", block_sha, "sol_sha", sol_sha, "rec", sha(rec_b)[:12], "ext", sha(ext_b)[:12], "rvw", sha(rvw_b)[:12])
