import json, pathlib, sys
RID = sys.argv[1]  # this session's #158 Slice C review id
P = pathlib.Path("reviews/c6_witness_collision_reconciliation_20260929")
md = (P/"RECONCILIATION.md").read_text(encoding="utf-8")
def rep(old, new):
    global md
    assert md.count(old) == 1, ("anchor", md.count(old), old[:80])
    md = md.replace(old, new)
rep("inventoried path).\n**Revision:**",
    "inventoried path); re-verified again at `5c484bf` (Math-#161 merged; no inventoried path changed).\n**Revision:**")
rep("**Revision:** v1.1 applies the scope corrections and the residual wording of OpenAI nonauthor review 5359701805\n"
    "(at `6a0e3b9`) and pickup 5900929519; §§1 and 3, the §4 statement, the 22-file inventory, every node id and every\n"
    "edge are unchanged.\n",
    "**Revision:** v1.1 applies the scope corrections and the residual wording of OpenAI nonauthor review 5359701805\n"
    "(at `6a0e3b9`) and pickup 5900929519; v1.2 merges Math- main `5c484bf` and updates the §5/§9 status of the\n"
    "candidates after Math-#161 merged. §§1 and 3, the §4 statement, the 22-file inventory, every node id and every edge\n"
    "are unchanged.\n")
rep("  candidates for it exist and are recorded here as **candidates only**: Math-#161 (OpenAI, draft head `e7f8929`,\n"
    "  planar; its Lemma M claims joint probability `o(r³)` for the local and fixed-remote nonempty events, with the\n"
    "  global one/two law conditional on Math-#158) and Math-#159 (Anthropic Claude, session `015wNj8L…`, v1.1 head\n"
    "  `7188bfa`, every fixed `d`; its Lemma 5.2 claims a near–far cross term `O(r^{9/2})`). Neither is reviewed, neither\n"
    "  is a premise of this record, neither is inventoried, and neither is consumed by any proposed node or edge; the\n"
    "  residual closes only when one of them, or a successor, carries a nonauthor verdict and is folded by an executing\n"
    "  lane.",
    "  candidates for it exist and are recorded here as **candidates only**: Math-#161 (OpenAI, head `e7f8929`, merged\n"
    "  to main at `5c484bf`; planar; its Lemma M gives joint probability `o(r³)` for the local and fixed-remote nonempty\n"
    "  events, and its §§6–7 one/two law is conditional on Math-#158 Theorems C/G/F) and Math-#159 (Anthropic Claude,\n"
    "  session `015wNj8L…`, v1.1 head `7188bfa`, every fixed `d`; its Lemma 5.2 claims a near–far cross term\n"
    "  `O(r^{9/2})`). Status at `5c484bf`: Math-#161 carries an Anthropic analytic ACCEPT (5359733142, by the session\n"
    "  that authored its consumed [PALM] and [DL], so source-exposed) and a Codex engineering ACCEPT (5359797442); the\n"
    "  four review slices of Math-#158 carry verdicts (A xAI 5359488967; B Codex 5359738045; C Anthropic, this session,\n"
    "  " + RID + "; D Codex 5359601495) while Math-#158 itself is not merged; Math-#159 is unreviewed. Neither candidate\n"
    "  is a premise of this record, neither is inventoried, and neither is consumed by any proposed node or edge. The\n"
    "  residual stays `OPEN_ACTIVE` for every fixed `d`: the Math-#161 chain is planar and source-exposed, so an executing\n"
    "  lane could at most record a planar sub-status from it, after its own fold; this record proposes none.")
rep("- **Math-#161 (OpenAI, draft `e7f8929`) and Math-#159 (Anthropic Claude, other session, `7188bfa`):** author-side\n"
    "  candidates for the §5 residual; unreviewed, not premises, not inventoried, not consumed (§5). Their appearance\n"
    "  changes none of this record's status rows, nodes or edges; their acceptance would be a further transition on the\n"
    "  residual node, by an executing lane.\n",
    "- **Math-#161 (OpenAI, `e7f8929`, merged at `5c484bf`) and Math-#159 (Anthropic Claude, other session, `7188bfa`):**\n"
    "  author-side candidates for the §5 residual; not premises, not inventoried, not consumed (§5, with their review\n"
    "  status at `5c484bf`). Their appearance changes none of this record's status rows, nodes or edges; a planar\n"
    "  sub-status of the residual would be a further transition by an executing lane, after its own fold.\n"
    "- **Math-#158 (OpenAI, planar cubic cluster law, `bfcc67dc`, open):** premise of Math-#161 §§6–7; this session\n"
    "  reviewed its Slice C (" + RID + "); not consumed here.\n")
assert "not supplied by the landed reviewed chain" in md and "candidates only" in md and "draft head" not in md and "draft `e7f8929`" not in md
(P/"RECONCILIATION.md").write_text(md, encoding="utf-8")

d = json.loads((P/"PROPOSED_TRANSITIONS.json").read_text(encoding="utf-8"))
d["reverified_at"] = "5c484bf (Math- main after Math-#155, #156 and #161; no inventoried path changed)"
d["revision"] = {"version": "v1.2",
                 "applies": ["v1.1: OpenAI review 5359701805 items 1-3; OpenAI pickup 5900929519 residual wording",
                             "v1.2: merge of Math- main 5c484bf; candidate status after Math-#161 merged (Anthropic 5359733142 source-exposed; Codex engineering 5359797442); Math-#158 slice verdicts A/B/C/D"],
                 "unchanged": "inventory, node ids, edges, status-row statement, historical predicates"}
R = "math.rn-region.witness-collision.leading-mass-localization"
g = next(e for e in d["graph"] if e["node"] == R)
old = ("Author-side candidates: Math-#161 (OpenAI, draft e7f8929, planar, global law conditional on Math-#158) and "
       "Math-#159 (Claude, 7188bfa, every fixed d); unreviewed, not premises, not inventoried, not consumed.")
assert old in g["scope"]
g["scope"] = g["scope"].replace(old,
    "Author-side candidates: Math-#161 (OpenAI, e7f8929, merged 5c484bf; planar; Anthropic analytic ACCEPT 5359733142 by the "
    "PALM/DL author session, source-exposed; Codex engineering ACCEPT 5359797442; sections 6-7 conditional on Math-#158, whose "
    "slices A/B/C/D carry verdicts while #158 is unmerged) and Math-#159 (Claude, 7188bfa, every fixed d, unreviewed); not "
    "premises, not inventoried, not consumed. Stays OPEN_ACTIVE for every fixed d; a planar sub-status is for an executing lane.")
n = next(x for x in d["proposed_graph_nodes"] if x["id"] == R)
old = ("author-side candidates Math-#161 (e7f8929) and Math-#159 (7188bfa) are unreviewed and not premises.")
assert old in n["notes"]
n["notes"] = n["notes"].replace(old,
    "author-side candidates: Math-#161 (e7f8929, merged 5c484bf, planar, Anthropic 5359733142 source-exposed + Codex "
    "engineering 5359797442, conditional on Math-#158) and Math-#159 (7188bfa, unreviewed); not premises, not consumed.")
(P/"PROPOSED_TRANSITIONS.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
print("v1.2 edits applied; max line", max(len(l) for l in md.splitlines() if not l.startswith("|")))
