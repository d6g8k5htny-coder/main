#!/usr/bin/env python3
"""Fill the placeholders of docs/LEAN_VERIFICATION_SPECIFICATION.md from REGISTER.json and
formal/manifest.json. Scientific effect: NONE. The script renders inventories; it decides nothing.

Prose sections (COVERAGE SUMMARY, ROADMAP, MATHLIB BUILD TIMING) are read from sibling files in
this directory so that the numbers they quote can be asserted against the register here.
"""
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRATCH = HERE.parent.parent
DOC = SCRATCH / "lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md"
MANIFEST = SCRATCH / "lvs_main/formal/manifest.json"
REGISTER = SCRATCH / "repos/math/specifications/lean_statements_20261010/REGISTER.json"
PACKET = "specifications/lean_statements_20261010"

KIND_ORDER = [
    ("landing_claim", "Landing claims", "disposition transcribed verbatim from `claims/LANDING_CLAIMS.json`"),
    ("graph_node", "Downstream graph nodes", "every `math.*` node of `frontiers/downstream_gate_20260925/GRAPH.json`; classifications are source identity"),
    ("open_obligation", "Open obligations", "stated as targets where a Prop exists; nothing asserts they hold"),
    ("proof_index_result", "PROOF_INDEX results", "reviewed scoped results and the packets PROOF_INDEX lists by id only"),
    ("existing_lean_package", "Existing Lean packages", "their theorems are not Props stating a research claim of this register"),
    ("main_experiment_proof", "main experiment proofs", "prose proofs on `main`, not in the Math- registers"),
    ("exact_certificate", "Exact certificates", "L2 exact-arithmetic Python evidence, not Lean"),
]

NE_CAP = 230
NS_CAP = 200
HEAD_CAP = 72
STMT_SHOW = 7


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def is_idonly(row):
    return row["kind"] == "proof_index_result" and row["not_established"].startswith("Statement not extracted")


def first_sentence(text, cap):
    """Leading whole sentences of `text` that fit in `cap` characters (at least one); a lone
    over-long sentence is cut at a word boundary and marked with an ellipsis."""
    text = " ".join(str(text).split())
    parts = re.split(r"(?<=[a-z\)\]])\.\s+(?=[A-Z])", text)
    acc = parts[0] if parts[0].endswith(".") or len(parts) == 1 else parts[0] + "."
    for nxt in parts[1:]:
        nxt = nxt if nxt.endswith(".") else nxt + "."
        if len(acc) + 1 + len(nxt) <= cap:
            acc += " " + nxt
        else:
            break
    if len(acc) > cap:
        cut = acc[:cap].rsplit(" ", 1)[0]
        acc = cut.rstrip(",;:.") + " …"
    return esc(acc)


def short_decl(st):
    decl = st["declaration"]
    if st["package"] == PACKET:
        name = decl.replace("UniversalLaw.Spec.", "")
        return f"`{name}`"
    # main formal/ theorem: UniversalLaw.P15.x or UniversalLaw.Side24.x
    name = decl.replace("UniversalLaw.", "")
    return f"main:`{name}` ({st['formal_progress']})"


def render_statements(stmts):
    if not stmts:
        return "—"
    names = [short_decl(s) for s in stmts]
    if len(names) <= STMT_SHOW:
        return ", ".join(names)
    n_spec = sum(1 for s in stmts if s["package"] == PACKET)
    n_main = len(stmts) - n_spec
    head = ", ".join(names[:3])
    tail = ", ".join(names[-3:])
    mid = f"… (+{len(names) - 6} more) …"
    summary = f"{n_spec} packet statements" + (f" + {n_main} main theorems (`proved`)" if n_main else "")
    return f"{summary}: {head}, {mid}, {tail}"


def render_layer0(row, sources):
    l0 = row.get("layer0") or {}
    sid = l0.get("source_id")
    heading = l0.get("heading") or ""
    heading = " ".join(heading.split())
    if len(heading) > HEAD_CAP:
        heading = heading[:HEAD_CAP].rsplit(" ", 1)[0] + " …"
    if sid and sid in sources:
        s = sources[sid]
        return esc(f"`{s['commit'][:8]}:{s['path']}` — {heading}") if heading else esc(f"`{s['commit'][:8]}:{s['path']}`")
    if heading:
        return esc(f"not pinned — {heading}")
    return "not pinned"


def render_table(reg):
    sources = {s["id"]: s for s in reg["sources"]}
    rows = reg["claims"]
    by_kind = OrderedDict((k, []) for k, _, _ in KIND_ORDER)
    for r in rows:
        by_kind[r["kind"]].append(r)
    assert sum(len(v) for v in by_kind.values()) == len(rows)
    out = []
    out.append(
        "Column key. **Claim**: the register `claim_id`. **Layer 0 source**: `commit:path` of the "
        "byte-pinned source under `sources/` in the packet (SHA-256 and byte count are in `REGISTER.json` "
        "`sources[]`), followed by the heading the row anchors to; *not pinned* means the row has no pinned "
        "bytes in this packet and therefore no Lean statement. **Transcribed disposition**: verbatim from "
        "`LANDING_CLAIMS.json`, `—` where the source carries none; it is never re-derived. **Lean "
        "statement(s)**: `Module.Declaration` in namespace `UniversalLaw.Spec` of the packet, all at "
        "`specified`; `main:` prefixes a theorem of the `main` core-Lean package `formal/` listed beneath the "
        "claim with its own `proved` manifest label (an arithmetic skeleton, never the claim). **Formal "
        "progress**: the claim-level label; no claim is `proved` or `kernel-checked`. **Not established** "
        "and **Next step** are abridged here to their first sentence; the full fields in `REGISTER.json` "
        "and the packet's `SCOPE.md` are the authoritative text and are longer, not shorter, in what they "
        "exclude. Alignment of every statement with its anchor is `PENDING_INDEPENDENT_REVIEW`."
    )
    out.append("")
    sec = 0
    for kind, title, note in KIND_ORDER:
        group = by_kind[kind]
        sec += 1
        fp = Counter(r["formal_progress"] for r in group)
        fp_txt = ", ".join(f"{n} `{k}`" for k, n in sorted(fp.items(), key=lambda kv: (kv[0] != "specified", kv[0])))
        out.append(f"#### 4.{sec} {title} ({len(group)}: {fp_txt}) — {note}")
        out.append("")
        if kind == "proof_index_result":
            n_idonly = sum(1 for r in group if is_idonly(r))
            out.append(
                f"{n_idonly} of these rows are packets that PROOF_INDEX lists by id only: their bytes are not pinned "
                "in this packet, their statement is recorded as *not extracted*, and no verdict is implied by the "
                "row's existence. They are kept as rows rather than omitted so that the register's inventory is "
                "two-way complete against PROOF_INDEX."
            )
            out.append("")
        out.append("| Claim | Layer 0 source (commit:path) | Transcribed disposition | Lean statement(s) (module.declaration) | Formal progress | Not established | Next step |")
        out.append("|---|---|---|---|---|---|---|")
        for r in group:
            disp = r.get("disposition_transcribed")
            out.append(
                "| "
                + " | ".join(
                    [
                        f"`{esc(r['claim_id'])}`",
                        render_layer0(r, sources),
                        f"`{esc(disp)}`" if disp else "—",
                        render_statements(r["statements"]),
                        f"`{r['formal_progress']}`",
                        first_sentence(r["not_established"], NE_CAP),
                        first_sentence(r["next_step"], NS_CAP) if r["next_step"] else "—",
                    ]
                )
                + " |"
            )
        out.append("")
    return "\n".join(out).rstrip()


def counts(reg, manifest):
    rows = reg["claims"]
    c = {}
    c["claims"] = len(rows)
    c["fp"] = Counter(r["formal_progress"] for r in rows)
    c["kind"] = Counter(r["kind"] for r in rows)
    c["kind_fp"] = Counter((r["kind"], r["formal_progress"]) for r in rows)
    stmts = [s for r in rows for s in r["statements"]]
    c["statements"] = len(stmts)
    c["stmt_fp"] = Counter(s["formal_progress"] for s in stmts)
    spec = {(s["module"], s["declaration"], s["kind"]) for s in stmts if s["package"] == PACKET}
    c["spec_distinct"] = len(spec)
    c["spec_kind"] = Counter(k for _, _, k in spec)
    c["spec_module"] = Counter(m.split("/")[-1].replace(".lean", "") for m, _, _ in spec)
    main = {s["declaration"] for s in stmts if s["package"] != PACKET}
    c["main_distinct"] = len(main)
    c["main_targets"] = len(manifest["targets"])
    c["main_modules"] = Counter(t["module"] for t in manifest["targets"])
    c["main_controls"] = len(manifest["negative_controls"])
    c["sources"] = len(reg["sources"])
    c["idonly"] = sum(1 for r in rows if is_idonly(r))
    c["pir_none_not_idonly"] = sorted(r["claim_id"] for r in rows if r["kind"] == "proof_index_result" and r["formal_progress"] == "none" and not is_idonly(r))
    c["graph_none"] = sorted(r["claim_id"] for r in rows if r["kind"] == "graph_node" and r["formal_progress"] == "none")
    c["claims_with_stmts"] = sum(1 for r in rows if r["statements"])
    return c


def coverage_tables(reg, c):
    rows = reg["claims"]
    stmts = [s for r in rows for s in r["statements"]]
    stmt_fp = Counter(s["formal_progress"] for s in stmts)
    counts_tbl = [
        "| Formal-progress label | Claims (171) | Statement entries (256) | Where the label may appear |",
        "|---|---:|---:|---|",
        f"| `none` | {c['fp']['none']} | — | claim rows only |",
        f"| `specified` | {c['fp']['specified']} | {stmt_fp['specified']} | claim rows; every statement of the packet |",
        f"| `proved` | {c['fp'].get('proved', 0)} | {stmt_fp['proved']} | statements of other packages (the `main` theorems listed beneath claims); no claim |",
        f"| `kernel-checked` | {c['fp'].get('kernel-checked', 0)} | {stmt_fp.get('kernel-checked', 0)} | run receipts only, never in the register |",
    ]
    kind_tbl = [
        "| Kind | Rows | `specified` | `none` |",
        "|---|---:|---:|---:|",
    ]
    for kind, title, _ in KIND_ORDER:
        n = c["kind"][kind]
        kind_tbl.append(f"| `{kind}` ({title}) | {n} | {c['kind_fp'].get((kind, 'specified'), 0)} | {c['kind_fp'].get((kind, 'none'), 0)} |")
    kind_tbl.append(f"| **all** | **{c['claims']}** | **{c['fp']['specified']}** | **{c['fp']['none']}** |")
    spec_list = []
    for kind, title, _ in KIND_ORDER:
        ids = [r["claim_id"] for r in rows if r["kind"] == kind and r["formal_progress"] == "specified"]
        if ids:
            spec_list.append(f"- `{kind}` ({len(ids)}): " + ", ".join(f"`{i}`" for i in ids))
    assert sum(len(l.split(", ")) for l in spec_list) == c["fp"]["specified"]
    return {
        "{{COUNTS_TABLE}}": "\n".join(counts_tbl),
        "{{KIND_TABLE}}": "\n".join(kind_tbl),
        "{{SPECIFIED_LIST}}": "\n".join(spec_list),
    }


def main():
    reg = json.loads(REGISTER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    c = counts(reg, manifest)
    if "--counts" in sys.argv:
        for k, v in c.items():
            print(k, dict(v) if isinstance(v, Counter) else v)
        return 0
    # invariants the prose relies on
    assert c["claims"] == 171 and c["statements"] == 256, (c["claims"], c["statements"])
    assert c["fp"] == Counter({"none": 138, "specified": 33}), c["fp"]
    assert "proved" not in c["fp"] and "kernel-checked" not in c["fp"]
    assert c["spec_distinct"] == 150 and c["spec_kind"] == Counter({"def_prop": 141, "structure": 9})
    assert c["main_targets"] == 59 and c["main_distinct"] == 59 and c["stmt_fp"]["proved"] == 59
    assert c["spec_module"] == Counter({"P15": 58, "Side24": 28, "RN": 28, "Lifetime": 24, "Field": 12}), c["spec_module"]
    assert c["idonly"] == 82, c["idonly"]
    assert c["sources"] == 23
    assert c["pir_none_not_idonly"] == sorted([
        "two-scale-s6-s21", "axial-density", "fixed-transverse-chart", "OA-C6-FOURIER-20260929-v1",
        "OA-C6-RARE-CLUSTER-20260929-v1", "CL-D5-REMOTE-COLLISION-20260928-v1", "cumulative-transfer-correction"]), c["pir_none_not_idonly"]
    assert len(c["graph_none"]) == 15 and sum(1 for g in c["graph_none"] if g.startswith("math.d1-component.")) == 4 \
        and sum(1 for g in c["graph_none"] if g.startswith("math.d5-component.")) == 8, c["graph_none"]

    doc = DOC.read_text()
    table = render_table(reg)
    coverage = (HERE / "coverage.md").read_text().strip()
    for token, text in coverage_tables(reg, c).items():
        assert coverage.count(token) == 1, token
        coverage = coverage.replace(token, text)
    assert "{{" not in coverage
    pieces = {
        "<<MAIN TARGET COUNT>> theorems, all by `decide`":
            "59 theorems (31 SIDE24 + 28 P15), by `decide` except one positivity conjunct by core `Nat.pow_pos`",
        "<<SPEC DECLARATION COUNT>> `def … : Prop` / interface structures, 0 theorems":
            "150 registered declarations (141 `def … : Prop`, 9 interface `structure`s; the modules' other 123 "
            "declarations are non-Prop definitions), 0 theorems",
        "<<CLAIM TABLE>>": table,
        "<<COVERAGE SUMMARY>>": coverage,
        "<<ROADMAP>>": (HERE / "roadmap.md").read_text().strip(),
        "<<MATHLIB BUILD TIMING>>": (HERE / "timing.md").read_text().strip(),
        # check.py has no positional `source` argument (argparse: --root, --repo, --execute, --allow-dirty);
        # the plain invocation is the source mode.
        "`python3 check.py source` / `--execute`": "`python3 check.py` / `--execute`",
        'python3 -B -S "$P/check.py" source': 'python3 -B -S "$P/check.py"',
    }
    for k, v in pieces.items():
        assert doc.count(k) == 1, (k, doc.count(k))
        doc = doc.replace(k, v)
    assert not re.search(r"<<[^<>]*>>", doc), "placeholder left"
    for bad in ("<<", ">>"):
        assert bad not in doc, bad
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} bytes); table rows {c['claims']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
