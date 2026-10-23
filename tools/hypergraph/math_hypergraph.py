#!/usr/bin/env python3
"""Math artifact hypergraph (Hypergraph-DB). Organization only: credit 0, OBL OPEN, no flags.

One hyperedge per coverage-ledger row, linking unit, source/review files,
implementations, graph nodes, lane and record. Inputs are all in-repo:
  data/hypergraph/math_coverage_ledger_20261023.md      (ledger copy)
  docs/site/dependency-source/GRAPH.json                (graph nodes, #527)
  data/hypergraph/graph_binding_report_20261023.json    (#527 binding report)

Usage:
  python tools/hypergraph/math_hypergraph.py build [--png]
  python tools/hypergraph/math_hypergraph.py reach <vertex> [depth]
  python tools/hypergraph/math_hypergraph.py gaps
  python tools/hypergraph/math_hypergraph.py shared-impl [n]
  python tools/hypergraph/math_hypergraph.py validate
"""
import argparse, collections, json, re, sys
from pathlib import Path
from hyperdb import HypergraphDB

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "hypergraph"
LEDGER = DATA / "math_coverage_ledger_20261023.md"
GRAPH = ROOT / "docs" / "site" / "dependency-source" / "GRAPH.json"
REPORT = DATA / "graph_binding_report_20261023.json"
HUBS = ("lane", "record")
PLACEHOLDER = re.compile(r"in PR head|per census", re.I)
REQUIRED_ROLES = ("unit",)


def parse_ledger(path=LEDGER):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        rows.append(dict(n=int(c[0]), rec=c[1], repo=c[2], unit=c[3].strip("`"), lane=c[4],
                         src=c[5], impl=c[6], node=c[7], bound=c[8], status=c[9]))
    return rows


def _split(cell, sep=";"):
    if cell in ("", "—") or cell.upper().startswith("MISSING"):
        return []
    return [s.strip() for s in cell.split(sep) if s.strip()]


def build(ledger=LEDGER, graph=GRAPH, report=REPORT):
    G = json.loads(Path(graph).read_text())["nodes"]
    R = json.loads(Path(report).read_text())
    bound_nodes = {b["node_id"] for b in R["bound"]}
    hg = HypergraphDB()

    def V(k, typ, **d):
        if not hg.has_v(k):
            hg.add_v(k, {"type": typ, **d})
        return k

    for nid, nd in G.items():
        V("node:" + nid, "graph_node", layer=nd.get("layer"), kind=nd.get("kind"),
          bound_in_527=nid in bound_nodes)
    for r in parse_ledger(ledger):
        u = V(f"unit:{r['repo']}:{r['unit']}", "unit", row=r["n"], status=r["status"], bound=r["bound"] == "Y")
        members = {u: "unit"}
        members[V("record:" + r["rec"], "record")] = "review"
        members[V("lane:" + r["lane"], "lane")] = "lane"
        srcs = _split(r["src"])
        for s in srcs:
            role = "review" if re.search(r"REVIEW", s, re.I) else "source"
            members[V("src:" + (s if "/" in s else r["unit"] + "/" + s), "source_file")] = role
        imps = _split(r["impl"])
        placeholders = []
        for s in imps:
            if s.startswith("(") or s.endswith(")") or PLACEHOLDER.search(s):
                placeholders.append(s)  # text, not an artifact
                continue
            members[V("impl:" + s, "implementation")] = "impl"
        for s in _split(r["node"], ","):
            members[V("node:" + s, "graph_node")] = "node"
        hg.add_e(tuple(members), {
            "kind": "binding", "row": r["n"], "roles": members, "bound": r["bound"] == "Y",
            "has_source": bool(srcs), "has_impl": bool(imps),
            "impl_placeholder": bool(imps) and not any(k.startswith("impl:") for k in members),
            "placeholders": placeholders, "credit": 0, "obl": "OPEN"})
    return hg, G


def nbr_vertices(hg, v):
    # workaround: hyperdb nbr_v()/nbr_e_of_v() raise on isolated vertices
    if not hg.degree_v(v):
        return set()
    return set().union(*[set(e) for e in hg.nbr_e_of_v(v)]) - {v}


def reach(hg, start, depth=2):
    if not hg.has_v(start):
        raise KeyError(start)
    seen, front = {start}, {start}
    for _ in range(depth):
        nxt = set()
        for v in front:
            if hg.v(v)["type"] in HUBS and v != start:
                continue  # hubs: do not traverse through
            if hg.degree_v(v):
                for e in hg.nbr_e_of_v(v):
                    nxt |= set(e)
        front = nxt - seen
        seen |= nxt
    return seen - {start}


def validate(hg):
    """Role validator: every hyperedge must carry each REQUIRED_ROLES role, and
    every member must have a role. Returns a list of problems."""
    problems = []
    for e in hg.all_e:
        d = hg.e(e)
        roles = d.get("roles", {})
        for req in REQUIRED_ROLES:
            if req not in roles.values():
                problems.append({"row": d.get("row"), "problem": f"missing role {req}"})
        for v in e:
            if v not in roles:
                problems.append({"row": d.get("row"), "problem": f"member without role {v}"})
    return problems


def gaps(hg, G):
    units = [v for v in hg.all_v if hg.v(v)["type"] == "unit"]
    edges = {u: [hg.e(e) for e in hg.nbr_e_of_v(u)] for u in units}
    placeholder = sorted(u for u in units if any(d["impl_placeholder"] for d in edges[u]))
    no_src = sorted(u for u in units if not any(d["has_source"] for d in edges[u]))
    no_impl = sorted(u for u in units if not any(d["has_impl"] for d in edges[u]))
    nodes = [v for v in hg.all_v if hg.v(v)["type"] == "graph_node"]
    unbound = sorted(v[5:] for v in nodes if v[5:] in G and not hg.v(v).get("bound_in_527"))
    untouched = sorted(v[5:] for v in nodes if not any(x.startswith("unit:") for x in nbr_vertices(hg, v)))
    return {
        "units": len(units), "sourced": len(units) - len(no_src), "no_source": no_src,
        "implemented": len(units) - len(no_impl), "no_impl": no_impl,
        "impl_placeholder_only": placeholder,
        "bound_units": sum(1 for u in units if hg.v(u)["bound"]),
        "graph_nodes": len(G), "unbound_nodes_per_527": unbound,
        "nodes_touched_by_no_unit": untouched,
        "role_problems": validate(hg),
    }


def shared_impl(hg, n=5):
    c = collections.Counter()
    for v in hg.all_v:
        if hg.v(v)["type"] == "implementation":
            c[v] = sum(1 for x in nbr_vertices(hg, v) if x.startswith("unit:"))
    return c.most_common(n)


def summary(g):
    return {k: (len(v) if isinstance(v, list) else v) for k, v in g.items()}


def write_outputs(hg, G, out=DATA, png=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / "math_hypergraph.json").write_text(json.dumps(
        {"note": "organization only; credit 0, OBL OPEN, no flags",
         "vertices": {k: hg.v(k) for k in sorted(hg.all_v)},
         "hyperedges": sorted((hg.e(e) for e in hg.all_e), key=lambda d: d["row"])},
        sort_keys=True, separators=(",", ":"), default=list) + "\n")
    g = gaps(hg, G)
    q = "node:math.d1-component.congruence-erratum"
    res = {"note": "organization only; credit 0, OBL OPEN, no flags",
           "stats": {"vertices": hg.num_v, "hyperedges": hg.num_e,
                     "by_type": dict(sorted(collections.Counter(hg.v(v)["type"] for v in hg.all_v).items()))},
           "gap_counts": summary(g), "gaps": g,
           "reach_example": {"start": q, "1hop": sorted(reach(hg, q, 1)), "2hop_count": len(reach(hg, q, 2))},
           "shared_impl_top5": shared_impl(hg)}
    (out / "query_results.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=list) + "\n")
    if png:
        render_png(hg, out / "bindings_hypergraph.png")
    return res


def render_png(hg, path):
    # library draw() only serves a web page; render with Pillow
    import math
    from PIL import Image, ImageDraw
    E = sorted((e for e in hg.all_e if any(v.startswith("node:") for v in e)), key=lambda e: hg.e(e)["row"])
    vs = sorted({v for e in E for v in e if not v.startswith(("lane:", "record:"))})
    W = 1400; img = Image.new("RGB", (W, W), "white"); d = ImageDraw.Draw(img); cx = cy = W / 2
    pos = {v: (cx + 560 * math.cos(2 * math.pi * i / len(vs)), cy + 560 * math.sin(2 * math.pi * i / len(vs))) for i, v in enumerate(vs)}
    col = {"unit": "#1f77b4", "graph_node": "#d62728", "source_file": "#2ca02c", "implementation": "#ff7f0e"}
    for e in E:
        m = [v for v in e if v in pos]
        hx = sum(pos[v][0] for v in m) / len(m) * 0.6 + cx * 0.4; hy = sum(pos[v][1] for v in m) / len(m) * 0.6 + cy * 0.4
        c = "#000000" if hg.e(e)["bound"] else "#999999"
        for v in m:
            d.line([pos[v], (hx, hy)], fill=c, width=1)
        d.rectangle([hx - 4, hy - 4, hx + 4, hy + 4], fill=c)
    for v, (x, y) in pos.items():
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col[hg.v(v)["type"]])
    d.text((20, 20), "Unit-to-graph-node hyperedges (square = 1 binding hyperedge; black bound, grey unbound)", fill="black")
    d.text((20, 40), "blue unit, red graph node, green source, orange impl. Organization only: credit 0, OBL OPEN", fill="black")
    img.convert("P", palette=Image.ADAPTIVE, colors=16).save(path, optimize=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build"); b.add_argument("--png", action="store_true"); b.add_argument("--out", type=Path, default=DATA)
    r = sp.add_parser("reach"); r.add_argument("vertex"); r.add_argument("depth", nargs="?", type=int, default=2)
    sp.add_parser("gaps")
    s = sp.add_parser("shared-impl"); s.add_argument("n", nargs="?", type=int, default=5)
    sp.add_parser("validate")
    a = ap.parse_args(argv)
    hg, G = build()
    if a.cmd == "build":
        print(json.dumps(write_outputs(hg, G, a.out, a.png)["gap_counts"], indent=1)); return 0
    if a.cmd == "reach":
        try:
            print("\n".join(sorted(reach(hg, a.vertex, a.depth))))
        except KeyError:
            print(f"unknown vertex: {a.vertex}", file=sys.stderr); return 2
        return 0
    if a.cmd == "gaps":
        print(json.dumps(summary(gaps(hg, G)), indent=1)); return 0
    if a.cmd == "shared-impl":
        for v, k in shared_impl(hg, a.n):
            print(f"{k}\t{v}")
        return 0
    p = validate(hg)
    print(json.dumps(p, indent=1)); return 1 if p else 0


if __name__ == "__main__":
    sys.exit(main())
