#!/usr/bin/env python3
"""Fail-closed identity and post-ratification scope verifier."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE = Path(__file__).with_name("GP-DER-118-v1.10.md")
BEGIN = "BEGIN_FROZEN_THEOREM_BODY"
END = "END_FROZEN_THEOREM_BODY"
EXPECTED_BYTES = 29293
EXPECTED_SHA256 = "9b7901e112a4857e3ea59942858684f72fc09a2825b1efa859360dd8fa55f014"


def check(condition: bool, label: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {label}")
    print(f"PASS: {label}")


raw = SOURCE.read_bytes()
check(b"\r" not in raw, "LF-only transport")
text = raw.decode("utf-8")
check(text.splitlines().count(BEGIN) == 1, "unique begin marker")
check(text.splitlines().count(END) == 1, "unique end marker")

body = text.split(BEGIN + "\n", 1)[1].split("\n" + END, 1)[0].encode("utf-8") + b"\n"
check(len(body) == EXPECTED_BYTES, "frozen-body byte count")
check(hashlib.sha256(body).hexdigest() == EXPECTED_SHA256, "frozen-body SHA-256")

required = (
    "5. EXACT ENDPOINT-HESSIAN NORMALIZER — RATIFIED CLOSED COMPONENT",
    "Phi(J)=(-1,-a/2,q,+1,+a/2,q)",
    "Delta_r=(A_M+1,B_M+a/2,C_M-q,A_S-1,B_S-a/2,C_S-q)",
    "CL-DER-204-v1.0",
    "AO48-DER-031",
    "CL-REC-242-v1.0",
    "KIMI-AUD-005",
    "AO48-OPR-042-v1.0",
    "Section 5 is CLOSED",
    "does not set RP-C, RP-S",
    "OVERALL P0.1 HOLD / NO PROMOTION",
)
for phrase in required:
    check(phrase in text, f"required phrase: {phrase}")

for stale in ("__BODY_BYTES__", "__SHA256__", "END GP-DER-118-v1.9"):
    check(stale not in text, f"stale token absent: {stale}")

print(f"BODY_BYTES={len(body)}")
print(f"BODY_SHA256={hashlib.sha256(body).hexdigest()}")
print(f"WHOLE_BYTES={len(raw)}")
print(f"WHOLE_SHA256={hashlib.sha256(raw).hexdigest()}")
print("ALL_ASSERTIONS_PASS")
