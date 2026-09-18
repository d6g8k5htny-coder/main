#!/usr/bin/env python3
"""Replay the exact GP-DATA-214 coefficient-table payload.

This harness imports the frozen source carrier without modifying it, invokes
the same symbolic constructors used by its main routine, and serializes the
coefficient evidence with the source's canonical JSON options.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "GP-DATA-214-v1.0_r_uniform_exact_endpoint_normalizer.py"
OUTPUT = HERE / "GP-DATA-214-v1.0_coefficient_table.json"

EXPECTED_SOURCE_BYTES = 37_512
EXPECTED_SOURCE_SHA256 = (
    "63ef800a51969903b9c3d97c74e61f77ee037afadff73038047ef7cfce040fb9"
)
EXPECTED_TABLE_BYTES = 3_269
EXPECTED_TABLE_SHA256 = (
    "b2724c9379d9a3d7cb55998908d3480818ee4ac29e918a9b7a4192c350b2224e"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    if len(source_bytes) != EXPECTED_SOURCE_BYTES:
        raise SystemExit(
            f"FAIL source bytes: {len(source_bytes)} != {EXPECTED_SOURCE_BYTES}"
        )
    source_sha256 = sha256(source_bytes)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"FAIL source SHA-256: {source_sha256} != {EXPECTED_SOURCE_SHA256}"
        )

    spec = importlib.util.spec_from_file_location("gp_data_214", SOURCE)
    if spec is None or spec.loader is None:
        raise SystemExit("FAIL could not construct module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    r, g, c, s, _transform = module.symbolic_blocks()
    _g_coefficients, g_evidence = module.checked_series_coefficients(g, r, 6)
    _c_coefficients, c_evidence = module.checked_series_coefficients(c, r, 6)
    _s_coefficients, s_evidence = module.checked_series_coefficients(s, r, 6)
    evidence = {"G": g_evidence, "C": c_evidence, "S": s_evidence}
    table_bytes = json.dumps(
        evidence, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    table_sha256 = sha256(table_bytes)
    if len(table_bytes) != EXPECTED_TABLE_BYTES:
        raise SystemExit(
            f"FAIL table bytes: {len(table_bytes)} != {EXPECTED_TABLE_BYTES}"
        )
    if table_sha256 != EXPECTED_TABLE_SHA256:
        raise SystemExit(
            f"FAIL table SHA-256: {table_sha256} != {EXPECTED_TABLE_SHA256}"
        )

    OUTPUT.write_bytes(table_bytes)
    if OUTPUT.read_bytes() != table_bytes:
        raise SystemExit("FAIL written payload differs from generated payload")

    print(f"source={SOURCE.name}")
    print(f"source_bytes={len(source_bytes)}")
    print(f"source_sha256={source_sha256}")
    print(f"output={OUTPUT.name}")
    print(f"coefficient_table_bytes={len(table_bytes)}")
    print(f"coefficient_table_sha256={table_sha256}")
    print("ALL_ASSERTIONS_PASS")


if __name__ == "__main__":
    main()
