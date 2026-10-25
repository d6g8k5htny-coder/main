"""Build five deterministic site assets from one fully admitted native archive.

Declared build and native identities remain comparison context, with unknown
custody and no scientific authority. This pure unit performs no publication,
filesystem access, network access, or execution of captured source.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
import re

from architecture.site_graph_projection import project_site_graph


MIB = 1024 * 1024
CURRENT_PAYLOAD_LIMIT = 16 * MIB
PAGE_LIMIT = 4 * MIB
MANIFEST_LIMIT = 65536
BUNDLE_LIMIT = 32 * MIB
BUILD_REPOSITORY_LIMIT = 256
BUILD_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
BUILD_COMMIT = re.compile(r"[0-9a-f]{40}")
MANIFEST_PATH = "current-build-manifest.json"
PAGE_PATH = "proof-tree.html"
BUILD_SCOPE = "declared-build-context-and-admitted-native-archive"
AXES = ("source", "review", "kernel", "computation", "alignment")


# All CSS is authored here. No captured graph string enters a style context.
_STYLE = """
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #142332;
        background: #f5f7fa; line-height: 1.5; }
body { margin: 0; }
main { max-width: 100rem; margin: auto; padding: 1.5rem; }
a { color: #154e86; text-underline-offset: .16em; }
a:focus-visible { outline: 3px solid #c24900; outline-offset: 4px; }
h1, h2, h3 { line-height: 1.2; overflow-wrap: anywhere; }
section, article { margin-block: 1.5rem; }
article { background: white; border: 1px solid #b7c7d6; border-radius: .5rem;
          padding: 1rem; scroll-margin-top: 1rem; }
dl { display: grid; grid-template-columns: minmax(10rem, 1fr) minmax(0, 4fr);
     gap: .4rem 1rem; }
dt { font-weight: 650; }
dd { margin: 0; overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem;
      background: #edf2f7; border: 1px solid #ccd7e2; }
.overview { overflow: auto; border: 1px solid #b7c7d6; background: white; }
svg { display: block; min-width: 82rem; width: 100%; height: auto; }
svg a rect { fill: #e9f2fa; stroke: #406584; stroke-width: 1; }
svg a:focus-visible rect { stroke: #c24900; stroke-width: 4; }
svg text { fill: #142332; font-family: monospace; font-size: 11px; }
svg .required { stroke: #637b91; stroke-width: 1.5; }
svg .context { stroke: #8498aa; stroke-width: 1; stroke-dasharray: 5 4; }
.table-scroll { overflow: auto; }
table { border-collapse: collapse; width: 100%; background: white; }
caption { text-align: left; margin-bottom: .5rem; }
th, td { text-align: left; vertical-align: top; border: 1px solid #c4d1dd;
         padding: .6rem; overflow-wrap: anywhere; }
.axes { padding-left: 1.5rem; }
@media (max-width: 40rem) { main { padding: 1rem; } dl { display: block; }
                           dd { margin-bottom: .7rem; } }
"""


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _descriptor(path: str, raw: bytes) -> dict:
    return {"path": path, "bytes": len(raw), "sha256": _sha(raw)}


def _text(value: str) -> str:
    """Escape both text and quoted attributes; never derive markup from data."""
    return escape(value, quote=True)


def _fields(values: list[tuple[str, str]]) -> str:
    return "<dl>" + "".join(
        "<dt>" + _text(label) + "</dt><dd>" + _text(value) + "</dd>"
        for label, value in values
    ) + "</dl>"


def _overview(graph: dict, fragments: dict[str, str]) -> str:
    """A deterministic SVG with one keyboard anchor for every admitted node."""
    identities = sorted(graph["nodes"])
    # Coordinates and fragment IDs come only from enumeration, never graph text.
    positions = {identity: (20 + (index % 3) * 480, 20 + (index // 3) * 76)
                 for index, identity in enumerate(identities)}
    height = 40 + ((len(identities) + 2) // 3) * 76
    pieces = [
        '<div class="overview"><svg xmlns="http://www.w3.org/2000/svg" '
        'viewBox="0 0 1460 ' + str(height) + '" role="group" '
        'aria-labelledby="overview-title overview-description">',
        '<title id="overview-title">Admitted graph overview</title>',
        '<desc id="overview-description">Each labeled node links to its full '
        'record and five evidence dimensions. Arrows run from the source node '
        'to the target node. Solid lines are required dependencies; dashed '
        'lines are context. The ordered edge table supplies exact relations.</desc>',
        '<defs><marker id="edge-arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerUnits="userSpaceOnUse" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#637b91"></path>'
        '</marker></defs>',
    ]
    # Intersect each center ray with a rectangle expanded on all four sides.
    # The 6x6 user-space marker fits inside a radius below 6.709 about its tip;
    # nine units of clearance also leave room for the two-unit focus stroke.
    half_width, half_height, clearance = 220, 28, 9
    for edge in graph["edges"]:
        x1, y1 = positions[edge["from"]]
        x2, y2 = positions[edge["to"]]
        dx, dy = x2 - x1, y2 - y1
        # Distinct fixed nodes have nonzero dx in multiples of 480, or nonzero
        # dy in multiples of 76. Thus this denominator is >2, keeping start before
        # end even for vertically adjacent boxes with only a 20-unit gap.
        fraction = 1 / max(abs(dx) / (half_width + clearance),
                           abs(dy) / (half_height + clearance))
        start_x, start_y = x1 + half_width + dx * fraction, y1 + half_height + dy * fraction
        end_x, end_y = x2 + half_width - dx * fraction, y2 + half_height - dy * fraction
        kind = "required" if edge["required"] else "context"
        pieces.append(
            '<line class="' + kind + '" x1="' + str(start_x) + '" y1="'
            + str(start_y) + '" x2="' + str(end_x) + '" y2="'
            + str(end_y) + '" marker-end="url(#edge-arrow)"><title>'
            + _text(edge["from"]) + " to " + _text(edge["to"]) + ": "
            + _text(edge["relation"]) + " (" + kind + ")</title></line>"
        )
    for identity in identities:
        x, y = positions[identity]
        label = _text(identity)
        pieces.append(
            '<a href="#' + fragments[identity] + '" tabindex="0" aria-label="'
            + label + '"><title>' + label + '</title><rect x="' + str(x)
            + '" y="' + str(y) + '" width="440" height="56" rx="5"></rect>'
            + '<text x="' + str(x + 10) + '" y="' + str(y + 32) + '">'
            + label + "</text></a>"
        )
    pieces.append("</svg></div>")
    return "\n".join(pieces)


def _node_cards(graph: dict, fragments: dict[str, str]) -> str:
    pieces = []
    for identity in sorted(graph["nodes"]):
        # Preserve the complete original typed record, including every source
        # and review field. These literal paths/URLs are not guessed hyperlinks.
        record = json.dumps(graph["nodes"][identity], sort_keys=True,
                            ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        pieces.append('<article id="' + fragments[identity] + '" data-node-id="'
                      + _text(identity) + '"><h3>' + _text(identity) + "</h3>")
        pieces.append('<h4>Recorded evidence dimensions</h4><ul class="axes">')
        for axis in AXES:
            state = graph["dimensions"][identity][axis]
            pieces.append('<li data-axis="' + _text(axis) + '" data-state="'
                          + _text(state) + '">' + _text(axis) + ": "
                          + _text(state) + "</li>")
        pieces.append("</ul><h4>Complete original typed node record</h4>")
        pieces.append('<pre data-node-record="true">' + _text(record) + "</pre>")
        pieces.append('<p><a href="#graph-navigation">Return to graph navigation</a></p></article>')
    return "\n".join(pieces)


def _edge_table(graph: dict, fragments: dict[str, str]) -> str:
    pieces = [
        '<div class="table-scroll"><table><caption>Every original edge, in '
        'recorded order. Source to target preserves direction; required and '
        'context preserve the recorded requirement.</caption>',
        '<thead><tr><th scope="col">Source</th><th scope="col">Target</th>'
        '<th scope="col">Relation</th><th scope="col">Requirement</th></tr></thead><tbody>',
    ]
    for edge in graph["edges"]:
        source, target, relation = edge["from"], edge["to"], edge["relation"]
        required = "true" if edge["required"] else "false"
        requirement = "required" if edge["required"] else "context"
        pieces.append(
            '<tr data-edge-from="' + _text(source) + '" data-edge-to="'
            + _text(target) + '" data-edge-relation="' + _text(relation)
            + '" data-edge-required="' + required + '"><td><a data-edge-role="source" '
            + 'href="#' + fragments[source] + '">' + _text(source) + '</a></td>'
            + '<td><a data-edge-role="target" href="#' + fragments[target] + '">'
            + _text(target) + '</a></td><td data-edge-role="relation">'
            + _text(relation) + '</td><td data-edge-role="requirement">'
            + requirement + "</td></tr>"
        )
    pieces.append("</tbody></table></div>")
    return "\n".join(pieces)


def _page(graph: dict, binding: dict[str, str], assets: dict[str, bytes],
          build_repository: str, build_commit: str, current_payload: dict) -> bytes:
    fragments = {identity: "node-" + str(index).zfill(3)
                 for index, identity in enumerate(sorted(graph["nodes"]))}
    source = graph["source"]
    # The strict admission fixes these Math source values. Only its repository
    # tree is linked; unacquired per-node source paths remain literal metadata.
    math_tree = "https://github.com/" + source["repository"] + "/tree/" + source["commit"]
    graph_path = next(path for path in assets if path.endswith("/graph.json"))
    receipt_path = next(path for path in assets if path.endswith("/architecture-run-receipt.json"))
    projection_path = next(path for path in assets if path.endswith("/projection.json"))
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>Source-bound native dependency graph</title><style>' + _STYLE + '</style></head>',
        '<body><main><h1>Source-bound native dependency graph</h1>',
        '<p>This page presents the admitted historical graph and its recorded '
        'metadata. Declared build context and native pins are comparison context; '
        'custody is unknown. Scientific effect: NONE. Scientific status authority: false.</p>',
        '<p>Recorded classifications and review dispositions retain their exact '
        'source scope. Navigation, hashes, archive admission and engineering '
        'checks do not add kernel, computation, alignment or scientific evidence.</p>',
        '<nav aria-label="Graph sections"><a href="#graph-navigation">Graph overview</a> | '
        '<a href="#node-records">Node records</a> | <a href="#ordered-edges">Ordered edges</a> | '
        '<a href="#bundle-provenance">Bundle provenance</a></nav>',
        '<section id="bundle-provenance"><h2>Bundle provenance and exact assets</h2>',
        '<h3>Declared executing build context</h3>',
        _fields([("Build repository", build_repository), ("Build commit", build_commit),
                 ("Current payload path", current_payload["path"]),
                 ("Current payload bytes", str(current_payload["bytes"])),
                 ("Current payload SHA-256 (whole current.json bytes)", current_payload["sha256"]),
                 ("Build verification scope", BUILD_SCOPE)]),
        '<h3>Declared native producer and archive binding</h3>',
        _fields([(field, binding[field]) for field in sorted(binding)]),
        '<p>Native verification scope: declared-native-pins-and-archive-content. '
        'The artifact SHA-256 identifies the whole private native ZIP, not an '
        'individual node or a Git blob. The receipt SHA-256 identifies the exact '
        'original producer receipt bytes. The private archive and diagnostic '
        'bodies are not public assets.</p>',
        '<h3>Fixed dated Math graph source</h3>',
        _fields([(field, source[field]) for field in sorted(source)]),
        '<p><a href="' + _text(math_tree) + '">Fixed dated Math repository tree</a>. '
        'The graph_source graph_sha256 identifies the whole fixed source GRAPH.json; '
        'gate_sha256 identifies its pinned gate source. Neither is a per-node '
        'source-file hash or an Artifact Index / Lean mapping.</p>',
        '<h3>Exact admitted public buffers</h3><ul>',
    ]
    for path, label in ((graph_path, "Original native graph"),
                        (receipt_path, "Original producer receipt"),
                        (projection_path, "Complete admission-linked projection")):
        pieces.append('<li><a href="' + _text(path) + '">' + label + '</a>: '
                      + str(len(assets[path])) + ' bytes; whole-file SHA-256 '
                      + _sha(assets[path]) + "</li>")
    pieces.extend([
        '<li><a href="' + MANIFEST_PATH + '">Current build manifest</a> '
        '(hashes these three buffers and this page; excludes its own hash).</li></ul></section>',
        '<section id="graph-navigation"><h2>Graph overview</h2>',
        '<p>' + str(len(graph["nodes"])) + ' nodes and ' + str(len(graph["edges"]))
        + ' directed edges. Each overview node is a keyboard link to its complete '
        'record below. The edge table records every exact ordered tuple.</p>',
        _overview(graph, fragments), '</section>',
        '<section id="node-records"><h2>Complete node records</h2>',
        '<p>All five evidence dimensions are shown as recorded. not_recorded '
        'means no evidence is recorded for that dimension in this graph. Source '
        'and review metadata remain original literal JSON values, with their '
        'types and scope preserved.</p>',
        _node_cards(graph, fragments), '</section>',
        '<section id="ordered-edges"><h2>Ordered dependency edges</h2>',
        _edge_table(graph, fragments), '</section>',
        '</main></body></html>\n',
    ])
    return "\n".join(pieces).encode("utf-8")


def build_native_graph_bundle(
    raw_zip: bytes,
    expected: dict[str, str],
    source_graph_bytes: bytes,
    *,
    build_repository: str,
    build_commit: str,
    current_payload_bytes: bytes,
) -> dict[str, bytes]:
    """Admit one original raw archive and return a complete bounded asset map.

    The payload bytes are opaque declared context here; JSON admission belongs
    to the site caller. No output is returned before every encoded limit passes.
    Native expected.repository retains the existing reader's grammar and caps;
    the separate 256-character limit applies only to build_repository.
    """
    if (type(raw_zip) is not bytes or type(source_graph_bytes) is not bytes
            or type(expected) is not dict
            or any(type(key) is not str or type(value) is not str
                   for key, value in expected.items())
            or type(current_payload_bytes) is not bytes
            or len(current_payload_bytes) > CURRENT_PAYLOAD_LIMIT):
        raise ValueError("native graph bundle input type or payload capacity refused")
    if (type(build_repository) is not str
            or len(build_repository) > BUILD_REPOSITORY_LIMIT
            or BUILD_REPOSITORY.fullmatch(build_repository) is None
            or any(part in (".", "..") for part in build_repository.split("/"))
            or type(build_commit) is not str
            or BUILD_COMMIT.fullmatch(build_commit) is None):
        raise ValueError("native graph bundle declared build context refused")

    # Only the actual strict projector establishes admission. Copy caller pins
    # before that call; do not accept a dataclass or caller-authored projection.
    binding = dict(expected)
    admitted = project_site_graph(raw_zip, binding, source_graph_bytes)
    graph = json.loads(admitted.original_graph_bytes)
    prefix = "native-graph/" + binding["artifact_sha256"] + "/"
    outputs = {
        prefix + "graph.json": admitted.original_graph_bytes,
        prefix + "architecture-run-receipt.json": admitted.original_receipt_bytes,
        prefix + "projection.json": admitted.projection_bytes,
    }
    current_payload = _descriptor("current.json", current_payload_bytes)
    page = _page(graph, binding, outputs, build_repository, build_commit, current_payload)
    if len(page) > PAGE_LIMIT:
        raise ValueError("native graph bundle page capacity refused")
    outputs[PAGE_PATH] = page
    manifest = {
        "schema_version": 1,
        "scientific_effect": "NONE",
        "scientific_status_authority": False,
        "custody": "unknown",
        "verification_scope": BUILD_SCOPE,
        "build": {"repository": build_repository, "commit": build_commit,
                  "current_payload": current_payload},
        "native": {"binding": binding, "graph_source": graph["source"]},
        "files": [_descriptor(path, outputs[path]) for path in sorted(outputs)],
    }
    manifest_bytes = _canonical(manifest)
    if len(manifest_bytes) > MANIFEST_LIMIT:
        raise ValueError("native graph bundle manifest capacity refused")
    outputs[MANIFEST_PATH] = manifest_bytes
    if sum(len(raw) for raw in outputs.values()) > BUNDLE_LIMIT:
        raise ValueError("native graph bundle aggregate capacity refused")
    return outputs
