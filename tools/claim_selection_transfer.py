"""Opt-in exact selection/estimand transfer diagnostic, not a scientific gate.

Algebra is a deliberately narrow copy of an existing Math- source's cubic
probability-to-count interface. A successful arithmetic comparison says NOTHING
about whether the caller's declared premises or source pin have been verified.
"""
from __future__ import annotations

import json
import re
import sys
from fractions import Fraction

SCHEMA = "claim-selection-transfer.v1"
LIMIT = 16384
IDENTITY = ("law", "conditioning", "event", "count", "region", "selection")
SOURCE = {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "9f3fb6918b28e6378c034ebc19e76d432f5391cb",
    "note": "frontiers/three_fronts_20260924/RN_COUNT_INTERFACE.md",
    "note_blob": "371fd6d17920f2eb3b1c5ec30297acf6daf1d385",
    "program": "frontiers/three_fronts_20260924/frontier_math.py",
    "program_blob": "d64e119adad0dfb3da5ff033a81234129fccb253",
    "test_blob": "283a469ab831daae537c7f37b36be9523750d1cd",
    "results_blob": "f0cd87a8ca4f07164d73008667dfbe7e9aa3a016",
}


class Refused(ValueError):
    """A declaration has not met the small, fail-closed protocol."""


def fields(value, required):
    if type(value) is not dict or set(value) != set(required):
        raise Refused("invalid object fields")
    return value


def identity(value):
    fields(value, IDENTITY)
    for part in IDENTITY:
        token = value[part]
        if type(token) is not str or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,80}", token):
            raise Refused("invalid declared identity")
    return value


def rate(value):
    """Parse a bounded nonnegative rational string, never float/bool/coercion."""
    if type(value) is not str or not re.fullmatch(r"(?:0|[1-9][0-9]{0,4})(?:/[1-9][0-9]{0,4})?", value):
        raise Refused("invalid exact power")
    q = Fraction(value)
    if q > 64:
        raise Refused("power out of scope")
    return q


def fmt(q):
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def evaluate(raw):
    fields(raw, ("schema", "target", "event_bound", "count_information"))
    if raw["schema"] != SCHEMA:
        raise Refused("schema mismatch")

    target = fields(raw["target"], ("identity", "observable", "requested_power"))
    anchor = identity(target["identity"])
    if target["observable"] != "event_weighted_count_expectation":
        # Whole E[N], pair probability and unconditional count are *different*
        # estimands. An unverified event-support assertion cannot bridge them.
        raise Refused("unsupported observable")
    want = rate(target["requested_power"])

    event = fields(raw["event_bound"], ("identity", "rate"))
    if identity(event["identity"]) != anchor or event["rate"] != "r^3":
        raise Refused("unmatched cubic event premise")

    info = raw["count_information"]
    if type(info) is not dict or info.get("kind") not in (
        "none", "unconditional_p_moment", "uniform_conditional_mean"
    ):
        raise Refused("unsupported count premise")
    kind = info["kind"]
    if kind == "none":
        fields(info, ("kind", "identity"))
    elif kind == "unconditional_p_moment":
        fields(info, ("kind", "identity", "order", "beta"))
    else:
        fields(info, ("kind", "identity"))
    if identity(info["identity"]) != anchor:
        raise Refused("count premise identity mismatch")

    if kind == "unconditional_p_moment":
        p = info["order"]
        if type(p) is not int or not 2 <= p <= 64:
            raise Refused("moment order out of scope")
        beta = rate(info["beta"])
        # Existing source: Math- frontier_math.py:holder_exponent(p,beta)
        exponent = (Fraction(3) * (p - 1) - beta) / p
        route = "holder_from_pth_moment"
    elif kind == "uniform_conditional_mean":
        # Existing source RN_COUNT_INTERFACE.md identity (N3):
        # E[N 1_E] = P(E) E[N|E], with an actual *uniform* mean premise.
        exponent = Fraction(3)
        route = "explicit_uniform_conditional_mean"
    else:
        exponent = None
        route = "event_probability_only"

    return {
        "schema": SCHEMA,
        "route": route,
        "requested_power": fmt(want),
        "derived_power": fmt(exponent) if exponent is not None else None,
        "exponent_comparison": (
            "no_count_rate_implied" if exponent is None else
            "meets_declared_target" if exponent >= want else
            "insufficient_for_declared_target"
        ),
        "identity_match": True,
        "observable": "event_weighted_count_expectation",
        "premises_independently_verified": False,
        "source_identity_runtime_verified": False,
        "scientific_acceptance": False,
        "scientific_effect": "NONE",
        "graph_binding_created": False,
        "source": SOURCE,
    }


def strict_pairs(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise Refused("duplicate JSON key")
        obj[key] = value
    return obj


def refuse_float(_):
    raise Refused("non-exact JSON scalar")


def parse(data):
    if type(data) is not bytes or not data or len(data) > LIMIT:
        raise Refused("invalid input length")
    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=strict_pairs,
                          parse_float=refuse_float, parse_constant=refuse_float)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise Refused("invalid strict JSON") from exc


def main():
    try:
        # Bounded standard input: no source file execution, filesystem write,
        # fetch, network request or implicit Git checkout.
        data = sys.stdin.buffer.read(LIMIT + 1)
        result = evaluate(parse(data))
    except (Refused, OSError, OverflowError, TypeError, KeyError):
        sys.stderr.write("CLAIM_SELECTION_TRANSFER_REFUSED\n")
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
