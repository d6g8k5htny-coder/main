"""Admit a native graph archive and preserve the exact members for a site.

This standalone input schema needs an explicit site consumer. It does not
modify current.json, authenticate caller-supplied pins, or publish any files.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from zipfile import BadZipFile, ZipFile

from architecture.graph_artifact_readback import read_graph_archive


PROJECTION_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class SiteGraphProjection:
    """Original immutable members and the complete provenance-linked projection."""

    original_graph_bytes: bytes
    original_receipt_bytes: bytes
    projection_bytes: bytes


def project_site_graph(
    raw_zip: bytes, expected: dict[str, str], source_graph_bytes: bytes
) -> SiteGraphProjection:
    """Use one fully admitted archive; perform no filesystem or network I/O.

    The existing reader independently checks archive geometry, every member,
    the declared native binding, original receipt and fixed dated Math source.
    Its declared-native-pins scope and unknown custody are retained unchanged.
    A caller must preserve this unit at its own separate build/write boundary.
    """
    admitted = read_graph_archive(raw_zip, expected, source_graph_bytes)
    try:
        admission = admitted["receipt"]
        inventory = {item["path"]: item for item in admission["archive"]["members"]}
        # Admission has already finished on this exact immutable byte buffer.
        # The second parser only obtains two original bounded member buffers;
        # it never extracts a file and never substitutes for full admission.
        with ZipFile(BytesIO(raw_zip)) as archive:
            graph_raw = archive.read("generated/graph.json")
            receipt_raw = archive.read("architecture-run-receipt.json")
        for name, raw in (("generated/graph.json", graph_raw),
                          ("architecture-run-receipt.json", receipt_raw)):
            item = inventory[name]
            if (item["path"] != name or type(item["bytes"]) is not int
                    or item["bytes"] != len(raw)
                    or item["sha256"] != hashlib.sha256(raw).hexdigest()):
                raise ValueError("site graph member identity refused")
        if (admission["graph_member"] != inventory["generated/graph.json"]
                or admission["producer_receipt_sha256"] != hashlib.sha256(receipt_raw).hexdigest()):
            raise ValueError("site graph receipt linkage refused")
        document = {
            "schema_version": 1,
            "scientific_effect": "NONE",
            "scientific_status_authority": False,
            "custody": "unknown",
            "graph": {
                "identity": {
                    "kind": "native-artifact-member",
                    "binding": admission["binding"],
                    "member": admission["graph_member"],
                },
                "data": admitted["graph"],
            },
            "admission": admission,
        }
        output = (json.dumps(document, sort_keys=True, ensure_ascii=True,
                             separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if len(output) > PROJECTION_LIMIT:
            raise ValueError("site graph projection capacity refused")
    except (BadZipFile, KeyError, OSError, RuntimeError, TypeError, UnicodeError) as error:
        raise ValueError("site graph projection refused") from error
    return SiteGraphProjection(graph_raw, receipt_raw, output)
