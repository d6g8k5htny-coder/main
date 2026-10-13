"""Checks for C6-RESIDUAL-CLOSURE-20260930-v1 (RECONCILIATION.md, PROPOSED_TRANSITIONS.json, EXTERNAL_REVIEWS.md,
REVIEWED_SECTION_3_b3fac79.md).

Standard library only. Run from the repository root:  python -B -S reviews/c6_residual_closure_20260930/residual_check.py
Checks: IDENTITIES, VERDICTS, LIVE, REVIEWED, DEDUCTION, TRANSITIONS, GATE, NEGATIVES. No mathematics is re-proved beyond the
exact finite bookkeeping of section 3 (pair identities, tails, ledgers); the proposed graph is replayed through the downstream
hard gate's own validators in every execution shape (interim, final, pre-existing residual node, each evidence path alone, the
unreviewed-record negative, and the installed state), with source snapshots. Nothing is written to the repository.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import tempfile
from fractions import Fraction as F

MUTANTS = ("allow-symlink", "no-hash", "stale-fingerprint", "drop-required-edge", "executed-flag", "controlling-true",
           "tail-reversed", "pair-identity-broken", "route-c-broken", "drop-review-needle",
           "installed-drift", "snapshot-omitted", "record-unreviewed", "reviewed-text-drift")
MUT = None
HERE = "reviews/c6_residual_closure_20260930"
GRAPH = "frontiers/downstream_gate_20260925/GRAPH.json"
SELECTOR = "frontiers/downstream_gate_20260925/SELECTOR_REGION.json"
HARD_GATE = "frontiers/downstream_gate_20260925/hard_gate.py"
RES = "math.rn-region.witness-collision.leading-mass-localization"
WIT = "math.rn-region.witness-collision"
AGG = "math.c6-witness-collision-factorial-moment"
RECORD_NODE = "math.c6r-component.residual-closure-record"
PALM_NODE = "math.c6-component.palm-proof"
SC = "frontiers/spectral_cluster_closure_20260929/PROOF.md"
REVIEWED = HERE + "/REVIEWED_SECTION_3_b3fac79.md"
REVIEWED_HEAD = "b3fac79875f28bacd135c0aa5e65a47a41ae0fdf"
SOL_REVIEW = "Math-#173 review 5360645884"
FP_RES = "leading-order r^3 mass of E N(N-1): scale-r localization; open piece = mixed local/remote pairs M(R, s_0)"
FP_WIT = "eta->0 mutual witness separation"
INVENTORY = {
    "frontiers/spectral_cluster_closure_20260929/PROOF.md": [
        "e971b2cbe50a8a06b47c1219191201e0d1037c9adac23e5e3b7d2051bb70c2eb",
        "16c56821b52fd76b0be791622b9c3809eafde75a"
    ],
    "frontiers/spectral_cluster_closure_20260929/REVIEW_RECORD.md": [
        "6031d36697789097185a8e65bfbdf80f447e229689eac5a706ee8b0558db64a5",
        "9bcae183fecdf0ffe4d612a209438a4fb1fa0edd"
    ],
    "frontiers/spectral_cluster_closure_20260929/README.md": [
        "b1e2a0722945a6189d8338c2e5b1f1a74b3b5646b8dc10fe29168c165508342c",
        "895dded8aa8e405ad8e9640d3ffebc6449a108d0"
    ],
    "frontiers/c6_cluster_law_20260929/PROOF.md": [
        "96b8e2d59defee6059bf151a896b7fc7c91fc529b385a3db8a87574901197b38",
        "ba492c8e62e58bfc055fc8254c790e893d346ba5"
    ],
    "frontiers/c6_cluster_law_20260929/README.md": [
        "e4b35f2b0e919ba57ebfb121c403896a32782d58dd99d7ec6a3d6d9e5107c6ac",
        "1166376f7b4d917b1bda3f9438adfd94d8831139"
    ],
    "frontiers/c6_cluster_law_20260929/SOURCE_MAP.json": [
        "0584c3808b633fceb3c308186b07d862eb4916053daf4e7bf1110592ec41f107",
        "ef51c582d0b1c874aabb01fce3b35c10631f7fc6"
    ],
    "frontiers/c6_palm_route_20260929/PROOF.md": [
        "aa37f16040ab9e05bb2e4e3f1d678367497e8dc2cf43eef197a4d6592c2f129b",
        "89eb8adf08fe7afc2cab9662d3a9875c05ae5cc5"
    ],
    "reviews/c6_residual_closure_20260930/EXTERNAL_REVIEWS.md": [
        "EXT_SHA",
        "EXT_BLOB"
    ],
    "reviews/c6_residual_closure_20260930/RECONCILIATION.md": [
        "REC_SHA",
        "REC_BLOB"
    ],
    "reviews/c6_residual_closure_20260930/REVIEWED_SECTION_3_b3fac79.md": [
        "RVW_SHA",
        "RVW_BLOB"
    ],
    "frontiers/two_scale_cluster_geometry_20260929/TWO_SCALE_LAW.md": [
        "e81d7fe09d25c3266eb8e62922756f54761d7f74d692fb35d9a8071fffd73769",
        "a32fd5f7d941bbe1fe943df045b1e0fbec8d691c"
    ],
    "frontiers/two_scale_cluster_geometry_20260929/REVIEW_RECORD.md": [
        "af1540308f37220e6592b700bf42181205b1d4588a040a5c151bf0a6f861e728",
        "31896de23fc56822c184375d0dc2b1c7e27fcae4"
    ],
    "frontiers/two_scale_cluster_geometry_20260929/REVIEW_RECORD.json": [
        "958d2e1ae8ea5d034d0d87efce6cde9be57ebb9d7c0b671e69d3d6598a6675ca",
        "a4b0fb99c24e2edf41973ea2a5e5176584602855"
    ]
}
VERDICTS = {
    "frontiers/spectral_cluster_closure_20260929/PROOF.md": [
        "sum_(n>=1) n^q |r^-3 Q_r^W(N_r=n)-nu1*1{n=1}-nu2*1{n=2}| -> 0",
        "r^-3 Q_W(N_R=j) -> a_j^R=integral 1{n_R=j}dM",
        "Dominated convergence proves a_j^R->a_j",
        "Q_W(N_R>0,N_far^rho>0)=o_(R,rho)(r^3)",
        "E (N)_2/r^3 -> 2nu2"],
    "frontiers/spectral_cluster_closure_20260929/REVIEW_RECORD.md": [
        "Git blob16c56821b52fd76b0be791622b9c3809eafde75a",
        "A — Anthropic Claude5359968447",
        "B — Anthropic Claude5359845136",
        "C — OpenAI/Codex5359879627",
        "ACCEPT Sections7-8, compact-origin-jet far conditioning, full mixed o(r^3), radius"],
    "frontiers/spectral_cluster_closure_20260929/README.md": [
        "Core conditional proof has completed scoped nonauthor A/B/C acceptance."],
    "frontiers/c6_cluster_law_20260929/PROOF.md": [
        "**Corollary Lambda (asymptotic factorial moments).**",
        "Lambda_2 = 2 nu(2)",
        "Moreover `nu_near^A(n) -> nu_near(n)` as `A -> oo`",
        "**Lemma 5.2 (no near–far cross term).**",
        "C(A, rho) r^(9/2)"],
    "frontiers/c6_cluster_law_20260929/README.md": ["# C6 cluster law (CL-C6-CLUSTER-LAW-20260929-v1.2)"],
    "frontiers/c6_cluster_law_20260929/SOURCE_MAP.json": ["nonauthor_acceptance", "review 5360192822"],
    "frontiers/c6_palm_route_20260929/PROOF.md": [
        "**Object:** CL-C6-PALM-20260929-v1.",
        "**Theorem Q (factorial moments, every fixed dimension).**"],
    HERE + "/EXTERNAL_REVIEWS.md": [
        "pullrequestreview-5360192822",
        "VERDICT: ACCEPT Theorem N, Corollaries Lambda/S",
        "A pull-request review is a mutable external object",
        "pullrequestreview-5360645884",
        "MATHEMATICAL VERDICT ON §3: ACCEPT at the stated fixed-d, fixed-torus, compact-positive-gap window scope.",
        REVIEWED_HEAD],
    HERE + "/RECONCILIATION.md": ["**Object:** C6-RESIDUAL-CLOSURE-20260930-v1.", "(Res)", "(D1)", "(D2)", "5360645884"],
    REVIEWED: ["## 3. The corollary: (Res) from the reviewed statements", "(D1)",
               "**Route A ([SC], Math-#162).**", "**Route B ([CL], Math-#159).**"],
    "frontiers/two_scale_cluster_geometry_20260929/TWO_SCALE_LAW.md": [
        "**Theorem L (factorial localization).** For every fixed integer q>=2,",
        "r^-3 E_r[(N_r)_q-(N_in)_q] -> 0.",
        "E_r[N_R N_far^rho]=o_(R,rho)(r^3),"],
    "frontiers/two_scale_cluster_geometry_20260929/REVIEW_RECORD.md": [
        "Anthropic Claude review5360227991",
        "bloba32fd5f7d941bbe1fe943df045b1e0fbec8d691c",
        "ACCEPT of Theorem T, Theorem L, ordered-pair law L3"],
    "frontiers/two_scale_cluster_geometry_20260929/REVIEW_RECORD.json": [
        "\"native_review_id\": 5360227991",
        "\"proof_path\": \"TWO_SCALE_LAW.md\"",
        "\"body_sha256\": \"c63efc292d7cf209e2770419cc159276897d0ba955c32ee1d2af88c8aa7f0ed8\"",
        "\"git_blob\": \"a32fd5f7d941bbe1fe943df045b1e0fbec8d691c\""],
}
EXTERNAL_PREFIXES = ("https://", "http://", "external:")


def git_blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def no_symlink_on_path(root, path):
    parts = pathlib.PurePosixPath(path).parts
    return not any((root / pathlib.PurePosixPath(*parts[:i])).is_symlink() for i in range(1, len(parts) + 1))


def identities_ok(root):
    for path, (sha, blob) in INVENTORY.items():
        p = root / path
        if not os.path.lexists(p):
            return False
        if MUT != "allow-symlink" and not no_symlink_on_path(root, path):
            return False
        if not p.is_file():
            return False
        data = p.read_bytes()
        if MUT == "no-hash":
            continue
        if hashlib.sha256(data).hexdigest() != sha or git_blob(data) != blob:
            return False
    return True


def check_verdicts(root):
    needles = {k: list(v) for k, v in VERDICTS.items()}
    if MUT == "drop-review-needle":
        needles[SC].append("THIS NEEDLE IS NOT IN THE SOURCE")
    for path, ns in needles.items():
        text = (root / path).read_text(encoding="utf-8")
        if not all(n in text for n in ns):
            return False
    return True


def load_spec(root):
    spec = json.loads((root / HERE / "PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
    if MUT == "executed-flag":
        spec["executed"] = True
    if MUT == "controlling-true":
        spec["transitions"][0]["proposed"]["controlling"] = True
    if MUT == "stale-fingerprint":
        for n in spec["proposed_graph_nodes"]:
            if n["id"] == PALM_NODE:
                n["fingerprint"] = "0" * 64
    if MUT == "drop-required-edge":
        spec["proposed_graph_edges"] = [e for e in spec["proposed_graph_edges"]
                                        if not (e["from"] == RES and e["to"] == "math.c6r-component.spectral-closure-proof")]
    if MUT == "record-unreviewed":
        for n in spec["proposed_graph_nodes"]:
            if n["id"] == RECORD_NODE:
                n["review_basis"] = []          # a PROVED_REVIEWED record with no nonauthor read must be rejected
    return spec


def load_live(root):
    graph = json.loads((root / GRAPH).read_text(encoding="utf-8"))
    sel = json.loads((root / SELECTOR).read_text(encoding="utf-8"))
    return graph, sel


def component_keys(comp):
    """Keys a live node must reproduce to count as the proposed component. The palm node is cross-record (Math-#160
    defines its other fields); every node of this record must reproduce the proposal byte-for-byte on every proposed key."""
    if comp["id"] == PALM_NODE:
        return ("kind", "source", "fingerprint", "classification", "controlling")
    return tuple(k for k in comp if k not in ("id", "create_if_absent", "cross_record"))


def matches_proposal(live, comp):
    return isinstance(live, dict) and all(live.get(k) == comp[k] for k in component_keys(comp))


def installed_state(graph, spec):
    """How much of the proposal the live graph already carries (Codex 4140125558). Fail-closed: a live node on a
    component source that is not the proposed node on every proposed key is a mismatch, as is a proposed id carrying
    anything else."""
    nodes = graph["nodes"]
    by_source = {c["source"]: c for c in spec["proposed_graph_nodes"]}
    present, mismatched = set(), set()
    for nid, n in nodes.items():
        if not isinstance(n, dict) or n.get("source") not in by_source:
            continue
        comp = by_source[n["source"]]
        (present if nid == comp["id"] and matches_proposal(n, comp) else mismatched).add(nid)
    for c in spec["proposed_graph_nodes"]:
        if c["id"] in nodes and c["id"] not in present:
            mismatched.add(c["id"])
    res = nodes.get(RES)
    live_edges = {(e["from"], e["to"], e["required"], e["relation"]) for e in graph["edges"]}
    wanted = {(e["from"], e["to"], e["required"], e["relation"]) for e in spec["proposed_graph_edges"]
              if not e.get("deferred_with")}
    return {"present": sorted(present), "mismatched": sorted(mismatched),
            "residual": res.get("classification") if isinstance(res, dict) else None,
            "edges_present": wanted <= live_edges}


def live_ok(graph, sel, spec):
    nodes = graph["nodes"]
    wit = nodes.get(WIT)
    ok = isinstance(wit, dict) and wit.get("kind") == "region" and wit.get("fingerprint") == FP_WIT
    ok &= wit.get("classification") in ("OPEN_ACTIVE", "PROVED_REVIEWED")
    res = nodes.get(RES)
    if res is not None:
        ok &= res.get("kind") == "region" and res.get("fingerprint") == FP_RES
        ok &= res.get("classification") in ("OPEN_ACTIVE", "PROVED_REVIEWED")
    ok &= "witness-collision" in sel.get("regions", [])
    st = installed_state(graph, spec)
    ok &= not st["mismatched"]
    if st["residual"] == "PROVED_REVIEWED":
        # a promoted residual must carry the whole proposal: every component by id and bytes, every edge, the reading rule
        ok &= set(c["id"] for c in spec["proposed_graph_nodes"]) <= set(st["present"]) and st["edges_present"]
        ok &= set(res.get("reading_rule", [])) >= set(spec["transitions"][0]["proposed"]["reading_rule"])
    return bool(ok), st


def check_live(root, spec):
    graph, sel = load_live(root)
    return live_ok(graph, sel, spec)


def reviewed_block(data):
    """The reviewed text: the file after its one-line provenance comment."""
    if data.startswith(b"<!--"):
        _, sep, rest = data.partition(b"-->\n\n")
        return rest if sep else b""
    return data


def check_reviewed(root, spec):
    """The record node is PROVED_REVIEWED on the strength of a non-Claude read of section 3 (Math-#173 review 5360645884
    at head b3fac79): the paragraphs that review read are preserved verbatim, fingerprinted in the proposal, and present
    unchanged in the current RECONCILIATION.md; the review's identity is preserved in EXTERNAL_REVIEWS.md."""
    rec = next((c for c in spec["proposed_graph_nodes"] if c["id"] == RECORD_NODE), None)
    if rec is None or not rec.get("review_basis"):
        return False
    b0 = rec["review_basis"][0]
    ok = rec["classification"] == "PROVED_REVIEWED" and b0.get("review") == SOL_REVIEW
    ok &= str(b0.get("provider", "")).startswith("OpenAI") and "ACCEPT" in str(b0.get("verdict", ""))
    ok &= b0.get("reviewed_head") == REVIEWED_HEAD and b0.get("reviewed_text") == REVIEWED
    block = reviewed_block((root / REVIEWED).read_bytes())
    if MUT == "reviewed-text-drift":
        block = block[:-1] + b"?"
    ok &= len(block) > 1000 and hashlib.sha256(block).hexdigest() == b0.get("reviewed_text_sha256")
    ok &= block.startswith(b"## 3. ") and b"**Route A ([SC], Math-#162).**" in block
    ok &= b"**Route B ([CL], Math-#159).**" in block and b"Route C" not in block     # the read does not cover Route C
    ok &= block in (root / HERE / "RECONCILIATION.md").read_bytes()                  # verbatim in the current record
    ext = (root / HERE / "EXTERNAL_REVIEWS.md").read_text(encoding="utf-8")
    ok &= "pullrequestreview-5360645884" in ext and REVIEWED_HEAD in ext
    return bool(ok)


def check_deduction():
    """Exact finite bookkeeping of RECONCILIATION.md section 3."""
    ok = True
    f2 = lambda n: n * (n - 1)
    f3 = lambda n: n * (n - 1) * (n - 2)
    # (D1): (N)_2 - (N_R)_2 = 2 N_R N_out + (N_out)_2 >= 2 N_R N_out >= 0
    for nr in range(13):
        for no in range(13):
            rhs = 2 * nr * no + (no * no if MUT == "pair-identity-broken" else f2(no))
            ok &= f2(nr + no) - f2(nr) == rhs and rhs >= 2 * nr * no >= 0
    # tail: for integers n > M >= 3, n(n-1) <= n(n-1)(n-2)/(M-1)
    for M in range(3, 11):
        for n in range(M + 1, 41):
            lhs, rhs = f2(n) * (M - 1), f3(n)
            ok &= (lhs >= rhs) if MUT == "tail-reversed" else (lhs <= rhs)
    # (D2), Route C: 0 <= (N)_2 - (A)_2 <= N^2 [1{B>0} + 1{A>0, C>0} + 1{C>=2}] for all counts A, B, C
    for a in range(9):
        for b in range(9):
            for c in range(9):
                n = a + b + c
                ind = (1 if b > 0 else 0) + (1 if (a > 0 and c > 0) else 0) + (0 if MUT == "route-c-broken" else (1 if c >= 2 else 0))
                ok &= 0 <= f2(n) - f2(a) <= n * n * ind
    ok &= F(3, 2) + F(9, 4) - 3 == F(3, 4) and F(3, 2) + F(5, 2) - 3 == F(1)
    # Stirling: n^4 = (n)_4 + 6 (n)_3 + 7 (n)_2 + (n)_1
    for n in range(0, 21):
        ok &= n ** 4 == n * (n - 1) * (n - 2) * (n - 3) + 6 * f3(n) + 7 * f2(n) + n
    # exponent ledgers of the two direct routes
    ok &= F(3) + F(3, 2) == F(9, 2) and F(9, 2) > 3
    ok &= F(3, 2) + F(3, 4) + F(5, 4) == F(7, 2) and F(7, 2) > 3
    # truncation error of a finite distribution against E(N)_3 / (M - 1)
    probs = [F(1, 2), F(1, 4), F(1, 8), F(1, 16), F(1, 32), F(1, 64), F(1, 128), F(1, 256), F(1, 256)]
    assert sum(probs) == 1
    e3 = sum(p * f3(n) for n, p in enumerate(probs))
    for M in (3, 4, 5, 6):
        tail = sum(p * f2(n) for n, p in enumerate(probs) if n > M)
        ok &= tail <= e3 / (M - 1)
    # iterated limit, exact rational instance: a_2 = 1, a_2^R = 1 - 1/R^2, limsup_r r^-3 M_R = 2 a_2 - 2 a_2^R = 2/R^2 -> 0
    vals = [2 - 2 * (1 - F(1, R * R)) for R in range(4, 40)]
    ok &= all(v == F(2, R * R) for v, R in zip(vals, range(4, 40)))
    ok &= all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)) and vals[-1] < F(1, 500)
    return bool(ok)


def evidence_paths(spec):
    ep = spec["evidence_paths"]
    return list(ep["direct_statement"]["components"]), list(ep["reviewed_corollary"]["components"])


def build(spec, graph, stage):
    """Apply the proposal to a copy of the live graph, idempotently: a component the live graph already carries
    byte-for-byte is not re-created, an edge already present is not re-added. stage: 'interim' (before step 2: residual
    node OPEN_ACTIVE with its evidence attached), 'final' (after step 2: residual PROVED_REVIEWED, both evidence paths
    required), 'final-preexisting' (Math-#160 step 1 already created the residual node OPEN_ACTIVE), 'final-direct-only'
    and 'final-corollary-only' (the other path's edges supporting: each path alone), 'final-record-unreviewed' (the
    record node without its read but still required: the residual must be held). Returns (old, new)."""
    old = copy.deepcopy(graph)
    if stage == "final-preexisting":
        old["nodes"][RES] = {"layer": "D5", "kind": "region", "classification": "OPEN_ACTIVE", "controlling": False,
                             "fingerprint": FP_RES, "notes": "Math-#160 execution_order step 1"}
    new = copy.deepcopy(old)
    present = set(installed_state(old, spec)["present"])
    live_by_source = {n.get("source"): nid for nid, n in old["nodes"].items()
                      if isinstance(n, dict) and n.get("fingerprint") and n.get("source")}
    id_map = {}
    for c in spec["proposed_graph_nodes"]:
        nid = c["id"]
        if nid in present:
            id_map[nid] = nid                                   # already installed as proposed
            continue
        if c.get("create_if_absent") and c["source"] in live_by_source:
            id_map[nid] = live_by_source[c["source"]]           # cross-record node already created by Math-#160
            continue
        new["nodes"][nid] = {k: v for k, v in c.items() if k != "id"}
        id_map[nid] = nid
    t = spec["transitions"][0]
    p = t["proposed"]
    res = dict(old["nodes"].get(RES) or {})
    res.update({"layer": p["layer"], "kind": "region",
                "classification": p["classification"] if stage.startswith("final") else "OPEN_ACTIVE",
                "controlling": p["controlling"], "fingerprint": p["fingerprint"], "scope": p["scope"],
                "explicit_limits": p["explicit_limits"], "review_disposition": p["review_disposition"],
                "review_sources": [r["ref"] for r in p["review_sources"]],
                "reading_rule": [id_map.get(x, x) for x in p["reading_rule"]], "notes": p["notes"]})
    new["nodes"][RES] = res
    have = {(e["from"], e["to"], e["required"], e["relation"]) for e in new["edges"]}
    added = []
    for e in spec["proposed_graph_edges"]:
        if e.get("deferred_with"):
            continue                                            # the old-node transition presupposes Math-#160 step 3
        key = (e["from"], id_map.get(e["to"], e["to"]), e["required"], e["relation"])
        if key not in have:
            added.append({"from": key[0], "to": key[1], "required": key[2], "relation": key[3]})
            have.add(key)
    direct, corollary = evidence_paths(spec)
    if stage == "final-direct-only":
        for e in added:
            if e["from"] == RES and e["to"] in corollary:
                e["required"] = False
    if stage == "final-corollary-only":
        for e in added:
            if e["from"] == RES and e["to"] in direct:
                e["required"] = False
    if stage == "final-record-unreviewed" and RECORD_NODE not in present:
        new["nodes"][RECORD_NODE]["classification"] = "AUTHOR_SIDE_CANDIDATE"
        new["nodes"][RECORD_NODE]["review_basis"] = []
    new["edges"].extend(added)
    return old, new


def check_transitions(root, spec):
    graph, sel = load_live(root)
    ok = spec.get("declarative") is True and spec.get("executed") is False
    t0, t1 = spec["transitions"]
    p = t0["proposed"]
    ok &= t0["node"] == RES and t0["kind"] == "region" and p["classification"] == "PROVED_REVIEWED"
    ok &= p["controlling"] is False and p["fingerprint"] == FP_RES
    ok &= bool(p.get("scope")) and bool(p.get("explicit_limits")) and len(p.get("review_sources", [])) >= 2
    ok &= bool(p.get("notes")) and p.get("review_disposition") == "ACCEPT_AT_EXISTENTIAL_SCOPE"
    comps = spec["proposed_graph_nodes"]
    cids = [c["id"] for c in comps]
    ok &= len(cids) == len(set(cids)) and len({c["source"] for c in comps}) == len(comps)
    for c in comps:
        ok &= c["kind"] == "reading_rule_component" and c["controlling"] is False
        ok &= c["source"] in INVENTORY and c["fingerprint"] == INVENTORY[c["source"]][0]
        ok &= bool(c.get("component_role")) and bool(c.get("author_provider")) and len(c.get("review_basis", [])) >= 1
        ok &= c["component_of"] == RES and c["classification"] == "PROVED_REVIEWED"
        if c["id"] == PALM_NODE:
            ok &= c.get("create_if_absent") is True and "Math-#160" in str(c.get("cross_record"))
    # two evidence paths, each sufficient, partitioning the components; the residual requires the union (fail-closed)
    direct, corollary = evidence_paths(spec)
    ok &= sorted(direct + corollary) == sorted(cids) and not set(direct) & set(corollary)
    ok &= RECORD_NODE in corollary and "math.c6r-component.two-scale-law-proof" in direct and len(direct) >= 2
    ok &= sorted(p["reading_rule"]) == sorted(cids) and sorted(t0["required_premises_after"]) == sorted(cids)
    req = {(e["from"], e["to"]) for e in spec["proposed_graph_edges"] if e["required"] is True and not e.get("deferred_with")}
    ok &= all((RES, cid) in req for cid in cids)
    seen = set()
    for e in spec["proposed_graph_edges"]:
        key = (e["from"], e["to"], e["relation"])
        ok &= key not in seen and type(e["required"]) is bool and bool(e["relation"])
        seen.add(key)
        ok &= e["from"] in (RES, WIT)
    # the deferred old-node transition
    ok &= t1["node"] == WIT and t1.get("deferred") is True and t1["proposed"]["classification"] == "PROVED_REVIEWED"
    ok &= sorted(t1["required_premises_after"]) == sorted([RES, AGG]) and t1["proposed"]["controlling"] is False
    ok &= AGG not in graph["nodes"] or graph["nodes"][AGG].get("classification") == "PROVED_REVIEWED"
    if not ok:
        return False, None
    st = installed_state(graph, spec)
    installed = st["residual"] == "PROVED_REVIEWED"
    stages = ("installed",) if installed else ("interim", "final", "final-preexisting", "final-direct-only",
                                                "final-corollary-only", "final-record-unreviewed")
    expected_required = {"final-direct-only": len(direct), "final-corollary-only": len(corollary)}
    builds = {}
    for stage in stages:
        old, new = build(spec, graph, "final" if stage == "installed" else stage)
        nodes, edges = new["nodes"], new["edges"]
        ok &= all(old["nodes"][k] == nodes[k] for k in old["nodes"] if k != RES)   # no live node other than the residual changes
        ok &= edges[:len(old["edges"])] == old["edges"]
        req_after = [e["to"] for e in edges if e["from"] == RES and e["required"] is True]
        if stage == "interim":
            ok &= nodes[RES]["classification"] == "OPEN_ACTIVE"
        else:
            ok &= nodes[RES]["classification"] == "PROVED_REVIEWED"
        if stage == "installed":
            ok &= new == old                                    # the live graph carries the proposal exactly: a no-op
        if stage == "final-record-unreviewed":
            ok &= nodes[RECORD_NODE]["classification"] == "AUTHOR_SIDE_CANDIDATE"
        else:
            ok &= nodes[RECORD_NODE]["classification"] == "PROVED_REVIEWED"
            ok &= all(nodes[d]["classification"] == "PROVED_REVIEWED" for d in req_after)
        ok &= len(req_after) == expected_required.get(stage, len(cids))
        builds[stage] = (old, new)
    if not installed:
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


def load_hard_gate(root):
    hg_spec = importlib.util.spec_from_file_location("hard_gate", root / HARD_GATE)
    hg = importlib.util.module_from_spec(hg_spec)
    hg_spec.loader.exec_module(hg)
    return hg


def snapshot(root, graph):
    """Exact source snapshot covering every node of a graph, in the shape git_transition_audit.read_snapshot produces,
    taken from the working tree: record_only, external_unresolved, blob (bytes and SHA256), tree (sorted file digests),
    or missing."""
    out = {}
    for nid, n in graph["nodes"].items():
        ref = n.get("source") if isinstance(n, dict) else None
        if ref is None:
            out[nid] = {"kind": "record_only"}
            continue
        if ref.startswith(EXTERNAL_PREFIXES):
            out[nid] = {"kind": "external_unresolved", "reference": ref}
            continue
        p = root / ref.rstrip("/")
        if p.is_file():
            data = p.read_bytes()
            out[nid] = {"kind": "blob", "reference": ref, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        elif p.is_dir():
            items = [[q.relative_to(p).as_posix(), hashlib.sha256(q.read_bytes()).hexdigest()]
                     for q in sorted(p.rglob("*")) if q.is_file() and "__pycache__" not in q.parts]
            out[nid] = {"kind": "tree", "reference": ref, "files": len(items),
                        "sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}
        else:
            out[nid] = {"kind": "missing", "reference": ref}
    return out


def check_gate(root, spec, builds):
    if builds is None:
        return False, {}
    hg = load_hard_gate(root)
    out = {}
    ok = True
    comp_ids = [c["id"] for c in spec["proposed_graph_nodes"]]
    for stage, (old, new) in builds.items():
        try:
            hg.validate_graph_fail_closed(new)
            rep_new = hg.closure_report(new)
            rep_old = hg.closure_report(old)
            old_src, new_src = snapshot(root, old), snapshot(root, new)
            if MUT == "snapshot-omitted":
                ri = hg.reverse_impact_between(old, new)
            else:
                ri = hg.reverse_impact_between(old, new, old_sources=old_src, new_sources=new_src)
        except Exception:
            return False, {}
        ok &= ri.get("source_snapshots_supplied") is True and ri.get("promotion_permission") is False
        ok &= rep_new.get("gate_ok") is True and not rep_new.get("illegal_controlling") and rep_old.get("gate_ok") is True
        ok &= not any(n.get("controlling") for n in new["nodes"].values())
        changed, impacted = set(ri["changed_nodes"]), set(ri["impacted"])
        pre = set(installed_state(old, spec)["present"])
        if stage in ("installed", "installed-simulated"):
            ok &= not changed and not impacted                  # nothing to apply: the proposal is live as declared
        else:
            ok &= RES in changed and RES in impacted
            ok &= all((cid in changed) or (cid not in new["nodes"]) or (cid in pre) for cid in comp_ids)
        ok &= WIT not in changed                                 # the deferred transition is not applied
        held = {h["node"]: h.get("unsatisfied_required", []) for h in rep_new.get("hold_proposals", [])}
        if stage == "interim":
            ok &= RES not in held or new["nodes"][RES]["classification"] == "OPEN_ACTIVE"
        elif stage == "final-record-unreviewed":
            ok &= RES in held and RECORD_NODE in held[RES]     # the gate itself holds a residual resting on an unread record
        else:
            ok &= RES not in held                                # every required premise PROVED_REVIEWED
        # a later source-byte change on any component reaches the residual (and the old node once it depends on it)
        wit_depends = any(e["from"] == WIT and e["to"] == RES for e in new["edges"])
        tested = 0
        for cid in comp_ids:
            if cid not in new["nodes"]:
                continue
            drift = copy.deepcopy(new_src)
            drift[cid]["sha256"] = "0" * 64
            try:
                ri2 = hg.reverse_impact_between(new, new, old_sources=new_src, new_sources=drift)
            except Exception:
                return False, {}
            ok &= set(ri2["changed_nodes"]) == {cid} and RES in ri2["impacted"]
            ok &= (not wit_depends) or WIT in ri2["impacted"]
            tested += 1
        ok &= tested == sum(1 for cid in comp_ids if cid in new["nodes"]) >= len(comp_ids) - 1
        out[stage] = {"changed_nodes": sorted(changed), "impacted": sorted(impacted),
                      "residual_classification": new["nodes"][RES]["classification"],
                      "residual_held": RES in held, "source_snapshots_supplied": True,
                      "source_drift_reaches_residual": tested,
                      "old_node_transition": "deferred: presupposes Math-#160 step 3 (aggregate node absent live)"}
    return bool(ok), out


def check_negatives(src):
    rejected = []
    for fault in ("delete", "change", "symlink", "symlink-parent"):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for path in INVENTORY:
                dst = root / path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src / path, dst)
            target = root / SC
            if fault == "delete":
                target.unlink()
            elif fault == "change":
                data = bytearray(target.read_bytes())
                data[0] ^= 1
                target.write_bytes(bytes(data))
            elif fault == "symlink":
                real = target.with_name("real_" + target.name)
                target.rename(real)
                os.symlink(real.name, target)
            else:
                parent = target.parent
                real = parent.with_name("real_" + parent.name)
                parent.rename(real)
                os.symlink(real.name, parent)
            rejected.append(not identities_ok(root))
    return all(rejected)


def main():
    global MUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutant", choices=MUTANTS)
    MUT = ap.parse_args().mutant
    root = pathlib.Path(".").resolve()
    spec = load_spec(root)
    ident = identities_ok(root)
    live, live_state = check_live(root, spec)
    trans_ok, builds = check_transitions(root, spec)
    gate_ok, gate = check_gate(root, spec, builds)
    checks = {"IDENTITIES": ident, "VERDICTS": ident and check_verdicts(root), "LIVE": live,
              "REVIEWED": ident and check_reviewed(root, spec), "DEDUCTION": check_deduction(),
              "TRANSITIONS": trans_ok, "GATE": gate_ok, "NEGATIVES": check_negatives(root)}
    passed = all(checks.values()) and len(checks) == 8
    print(json.dumps({"object": "C6-RESIDUAL-CLOSURE-20260930-v1", "checks": checks, "passed": passed,
                      "inventory_files": len(INVENTORY), "component_nodes": len(spec["proposed_graph_nodes"]),
                      "live_state": live_state, "gate": gate,
                      "scope": "identity, verdict-row, live-register (including the installed state), reviewed-text, exact "
                               "finite bookkeeping of section 3, transition-shape, hard-gate replay with source snapshots in "
                               "every execution shape, and filesystem-negative checks; the Gaussian limits are the sources' "
                               "reviewed statements and are not re-proved; nothing is written"},
                     indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
