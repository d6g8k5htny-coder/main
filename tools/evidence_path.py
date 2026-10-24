#!/usr/bin/env python3
"""Emit the read-only evidence-to-artifact path for one graph node.

LANES-20261022 Lane 1 (main#475). Given a graph node id, report
node -> pinned source (url + sha) -> artifact (Artifact Index binding)
-> execution receipt -> review link, plus an explicit ``missing`` list.

The module treats every input as inert JSON. It performs no network access,
writes no files, executes nothing it reads, and asserts no scientific status:
every result carries ``scientific_effect: "NONE"``. A missing link is reported
as missing; it is never padded, inferred or invented. Unknown nodes are refused.
"""
from __future__ import annotations

import argparse
import json
from ipaddress import IPv6Address
from pathlib import Path
import re
import sys
from urllib.parse import quote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "docs" / "site" / "current.json"
DEFAULT_BINDINGS = ROOT / "architecture" / "public-artifact-index-bindings.json"
FILE_LIMIT = 16 * 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}")
SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")

# Known, documented gaps. These are reported verbatim and never filled in.
KNOWN_MISSING = {
    "math.d5-component.punctured-pin-proof": (
        "public projection not admitted (#327 exporter held by Claude017)",
    ),
}


class EvidencePathRefused(ValueError):
    """An opaque refusal whose message names the reason code only."""

    def __init__(self, code):
        super().__init__("EVIDENCE_PATH_REFUSED: " + code)
        self.code = code


def _require(condition, code):
    if not condition:
        raise EvidencePathRefused(code)


def _identifier(value, code):
    _require(type(value) is str and IDENTIFIER.fullmatch(value) is not None, code)
    return value


def graph_nodes(graph_data):
    """Return the node id set named by a graph ``data`` object (nodes and edges)."""
    _require(type(graph_data) is dict, "graph-shape")
    nodes = set()
    for node in graph_data.get("nodes", ()) or ():
        if type(node) is dict and "id" in node:
            nodes.add(_identifier(node["id"], "graph-node-id"))
        elif type(node) is str:
            nodes.add(_identifier(node, "graph-node-id"))
    for edge in graph_data.get("edges", ()) or ():
        _require(type(edge) is dict, "graph-edge-shape")
        for key in ("from", "to"):
            if key in edge:
                nodes.add(_identifier(edge[key], "graph-edge-id"))
    return nodes


def _pinned_source(identity):
    if type(identity) is not dict:
        return None
    repository, commit = identity.get("repository"), identity.get("commit")
    if type(repository) is not str or REPOSITORY.fullmatch(repository) is None:
        return None
    if type(commit) is not str or SHA.fullmatch(commit) is None:
        return None
    path = identity.get("path")
    # A pinned source names a file: a missing, empty, absolute or traversing
    # path is not a source, so the caller reports it as missing evidence.
    if type(path) is not str or not path or "\\" in path:
        return None
    if any(part in ("", ".", "..") for part in path.split("/")):
        return None
    # Quote filename data before URL parsing can interpret escapes or delimiters.
    url = ("https://github.com/" + repository + "/blob/" + commit + "/"
           + quote(path, safe="/"))
    return {"url": url, "sha": commit}


def _ipv4_number(part):
    """Parse one WHATWG IPv4 number; None means invalid numeric syntax."""
    if not part:
        return None
    radix = 10
    if part[:2].lower() == "0x":
        radix, part = 16, part[2:]
    elif len(part) >= 2 and part[0] == "0":
        radix, part = 8, part[1:]
    if any(c.lower() not in "0123456789abcdef"[:radix] for c in part):
        return None
    return int(part or "0", radix)


def _numeric_host_ok(host):
    """Apply special-URL numeric-host rules to an already admitted ASCII host.

    See https://url.spec.whatwg.org/#concept-ipv4-parser. Non-numeric-ending
    names retain the caller's existing policy; this is not DNS validation.
    """
    parts = host.split(".")
    if parts[-1] == "":
        parts.pop()
    last = parts[-1]
    if re.fullmatch(r"[0-9]+", last) is None and _ipv4_number(last) is None:
        return True
    if len(parts) > 4:
        return False
    numbers = [_ipv4_number(part) for part in parts]
    if any(number is None for number in numbers):
        return False
    return (all(number <= 255 for number in numbers[:-1])
            and numbers[-1] < 256 ** (5 - len(numbers)))


def _https_url(value):
    # A locator needs a usable authority; parsing alone is not validation.
    if type(value) is not str or not value.startswith("https://") or len(value) > 2048:
        return False
    # Refuse raw characters that parsers can strip or reinterpret. Encoded
    # path/query data remains inert and the original URL is never rewritten.
    if "\\" in value or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        return False
    try:
        parts = urlsplit(value)
        host = parts.hostname
        parts.port  # Access validates the port's numeric syntax and range.
        if parts.scheme != "https" or not host:
            return False
        authority = parts.netloc.rsplit("@", 1)[-1]
        if authority.startswith("["):
            if re.fullmatch(r"\[[0-9A-Fa-f:.]+\](?::[0-9]*)?", authority) is None:
                return False
            IPv6Address(host)
        else:
            ascii_host = host.encode("idna").decode("ascii")
            if (re.fullmatch(r"[A-Za-z0-9_.-]+", ascii_host) is None
                    or not _numeric_host_ok(ascii_host)):
                return False
        return True
    except (ValueError, UnicodeError):
        return False


def evidence_path(node_id, graph, bindings, receipts=None, reviews=None):
    """Return the evidence path for ``node_id`` or raise ``EvidencePathRefused``.

    ``graph`` is a ``{"data": ..., "identity": ...}`` object (docs/site/current.json
    ``graph``); ``bindings`` is the public Artifact Index bindings list;
    ``receipts`` and ``reviews`` map artifact ids to ``{"url", "sha"?}`` records.
    Inputs are not mutated.
    """
    _identifier(node_id, "node-id")
    _require(type(graph) is dict, "graph-shape")
    nodes = graph_nodes(graph.get("data"))
    _require(node_id in nodes, "unknown-node")
    _require(type(bindings) is list, "bindings-shape")
    receipts = receipts or {}
    reviews = reviews or {}
    _require(type(receipts) is dict and type(reviews) is dict, "evidence-shape")

    missing = []
    source = _pinned_source(graph.get("identity"))
    if source is None:
        missing.append("pinned source (url+sha) not recorded")

    artifacts = []
    for binding in bindings:
        _require(type(binding) is dict, "binding-shape")
        if binding.get("node_id") != node_id:
            continue
        artifact_id = _identifier(binding.get("artifact_id"), "binding-artifact-id")
        roles = binding.get("roles")
        _require(type(roles) is list and all(type(r) is str for r in roles), "binding-roles")
        artifacts.append({"artifact_id": artifact_id, "roles": sorted(roles)})
    artifacts.sort(key=lambda item: item["artifact_id"])

    known = KNOWN_MISSING.get(node_id, ())
    missing.extend(known)
    if not artifacts and not known:
        missing.append("artifact index binding not recorded")

    receipt_links, review_links = [], []
    for artifact in artifacts:
        artifact_id = artifact["artifact_id"]
        for table, links, label in ((receipts, receipt_links, "execution receipt"),
                                    (reviews, review_links, "review link")):
            record = table.get(artifact_id)
            if type(record) is dict and _https_url(record.get("url")):
                entry = {"artifact_id": artifact_id, "url": record["url"]}
                if type(record.get("sha")) is str and SHA.fullmatch(record["sha"]):
                    entry["sha"] = record["sha"]
                links.append(entry)
            else:
                missing.append(label + " not recorded for " + artifact_id)
    if not artifacts:
        missing.append("execution receipt not recorded")
        missing.append("review link not recorded")

    return {
        "schema_version": 0,
        "node_id": node_id,
        "source": source,
        "artifacts": artifacts,
        "receipts": receipt_links,
        "reviews": review_links,
        "missing": missing,
        "complete": not missing,
        "scientific_effect": "NONE",
    }


def _read_json(path):
    with open(path, "rb") as handle:
        raw = handle.read(FILE_LIMIT + 1)
    _require(len(raw) <= FILE_LIMIT, "file-limit")
    return json.loads(raw.decode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("node_id")
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH),
                        help="JSON file whose 'graph' object holds data + identity")
    parser.add_argument("--bindings", default=str(DEFAULT_BINDINGS))
    parser.add_argument("--evidence", help="optional JSON {receipts:{}, reviews:{}}")
    args = parser.parse_args(argv)
    try:
        document = _read_json(args.graph)
        graph = document.get("graph", document) if type(document) is dict else document
        evidence = _read_json(args.evidence) if args.evidence else {}
        _require(type(evidence) is dict, "evidence-shape")
        result = evidence_path(args.node_id, graph, _read_json(args.bindings),
                               evidence.get("receipts"), evidence.get("reviews"))
    except EvidencePathRefused as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print("EVIDENCE_PATH_REFUSED: unreadable-input", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
