"""Bounded exact-rational original spectral moment enclosure producer.

No scientific target is built in. Certificates quantify over every input L.
The standalone verify_certificate.py imports none of this module.
"""
from fractions import Fraction as Q
import argparse
import json
import os
import re
import time

SOURCES = {"original": "f38c085ab5b15368e6989a1fd6298490f3b671f5a6c3a01bcdc81b516022bc33",
           "method_v2": "8bdea4b42e9efd2d846cd1aa73691e8194fa26cd937ac02b79991334f88204f6"}
CAPS = {"max_work": 100000, "max_seconds": 3, "max_rational_bits": 8192,
        "max_squarings": 16}

class KernelHold(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

def text(q):
    return str(q.numerator) + "/" + str(q.denominator)

def rational(s):
    if not isinstance(s, str) or len(s) > 3000 or not re.fullmatch(r"-?\d+/[1-9]\d*", s):
        raise KernelHold("INVALID_RATIONAL_INPUT")
    try:
        q = Q(s)
    except (ValueError, ZeroDivisionError):
        raise KernelHold("INVALID_RATIONAL_INPUT") from None
    if max(q.numerator.bit_length(), q.denominator.bit_length()) > 8192:
        raise KernelHold("RATIONAL_SIZE_LIMIT")
    return q

class Arithmetic:
    def __init__(self, bits, limits=None):
        self.bits = bits
        self.den = 1 << bits
        self.limits = dict(CAPS)
        if limits is not None:
            if not isinstance(limits, dict) or set(limits) - set(CAPS):
                raise KernelHold("INVALID_LIMITS")
            for k, v in limits.items():
                if type(v) is not int or not 1 <= v <= CAPS[k]:
                    raise KernelHold("INVALID_LIMITS")
                self.limits[k] = v
        self.start = time.monotonic()
        self.work = 0

    def tick(self, *qs):
        self.work += 1
        if self.work > self.limits["max_work"]:
            raise KernelHold("WORK_LIMIT")
        if time.monotonic() - self.start > self.limits["max_seconds"]:
            raise KernelHold("TIME_LIMIT")
        for q in qs:
            if max(q.numerator.bit_length(), q.denominator.bit_length()) > self.limits["max_rational_bits"]:
                raise KernelHold("RATIONAL_SIZE_LIMIT")

    def rounded(self, lo, hi):
        self.tick(lo, hi)
        if lo > hi:
            raise KernelHold("INTERNAL_INTERVAL_ERROR")
        a = Q((lo.numerator * self.den) // lo.denominator, self.den)
        b = Q(-((-hi.numerator * self.den) // hi.denominator), self.den)
        self.tick(a, b)
        return a, b

    def add(self, a, b):
        return self.rounded(a[0] + b[0], a[1] + b[1])

    def mul(self, a, b):
        v = [x*y for x in a for y in b]
        return self.rounded(min(v), max(v))

    def div(self, a, b):
        if b[0] <= 0:
            raise KernelHold("DENOMINATOR_NOT_POSITIVE")
        v = [x/y for x in a for y in b]
        return self.rounded(min(v), max(v))

    def power(self, a, n):
        if a[0] < 0 or type(n) is not int or n < 0:
            raise KernelHold("INTERNAL_POWER_ERROR")
        return self.rounded(a[0]**n, a[1]**n)

    def up(self, x):
        return self.rounded(x, x)[1]

def pack(iv):
    return [text(iv[0]), text(iv[1])]

def atan_bracket(inv, terms, ar):
    x = Q(1, inv)
    total = Q(0)
    for j in range(terms):
        term = x**(2*j+1) / (2*j+1)
        total += (-1 if j % 2 else 1) * term
        ar.tick(total, term)
    next_term = x**(2*terms+1)/(2*terms+1)
    other = total + (-1 if terms % 2 else 1)*next_term
    ar.tick(other)
    return min(total, other), max(total, other)

def pi_witness(terms, ar):
    a, b = atan_bracket(5, terms, ar), atan_bracket(239, terms, ar)
    p = ar.rounded(16*a[0] - 4*b[1], 16*a[1] - 4*b[0])
    return {"terms": terms, "interval": pack(p)}, p

def _exp_negative(x, pairs, ar):
    if not isinstance(x, Q) or x < 0:
        raise KernelHold("INVALID_EXP_ARGUMENT")
    ar.tick(x)
    y, s = x, 0
    while y > 1:
        if s >= ar.limits["max_squarings"]:
            raise KernelHold("EXP_RANGE_LIMIT")
        y /= 2
        s += 1
        ar.tick(y)
    term, total = Q(1), Q(1)
    even = None
    for j in range(1, 2*pairs+2):
        term = -term*y/j
        total += term
        ar.tick(term, total)
        if j == 2*pairs:
            even = total
    seed = ar.rounded(max(Q(0), total), min(Q(1), even))
    levels = [pack(seed)]
    value = seed
    for _ in range(s):
        value = ar.rounded(value[0]**2, value[1]**2)
        levels.append(pack(value))
    return {"x": text(x), "y": text(y), "squarings": s,
            "pairs": pairs, "levels": levels, "result": pack(value)}

def exp_negative(x, bits=80, pairs=16, limits=None):
    if type(bits) is not int or not 16 <= bits <= 128 or type(pairs) is not int or not 1 <= pairs <= 24:
        raise KernelHold("SETTING_LIMIT")
    return _exp_negative(x, pairs, Arithmetic(bits, limits))

def _result(w):
    return tuple(rational(v) for v in w["result"])

def certify(request):
    if not isinstance(request, dict):
        raise KernelHold("INVALID_REQUEST")
    allowed = {"L_lo", "L_hi", "cutoff", "bits", "pi_terms", "exp_pairs", "width_goal", "input_role", "limits"}
    if set(request)-allowed:
        raise KernelHold("UNKNOWN_REQUEST_FIELD")
    m = request.get("cutoff")
    if type(m) is not int or m < 1:
        raise KernelHold("CUTOFF_NOT_POSITIVE_INTEGER")
    if m > 64:
        raise KernelHold("CUTOFF_WORK_LIMIT")
    lo, hi = rational(request.get("L_lo")), rational(request.get("L_hi"))
    if not 0 < lo <= hi:
        raise KernelHold("INVALID_L_INTERVAL")
    if any(max(q.numerator.bit_length(), q.denominator.bit_length()) > 256 for q in (lo, hi)):
        raise KernelHold("INPUT_SIZE_LIMIT")
    bits, pn, en = (request.get(k, v) for k, v in (("bits", 80), ("pi_terms", 20), ("exp_pairs", 16)))
    for v, lower, upper in [(bits, 16, 128), (pn, 1, 32), (en, 1, 24)]:
        if type(v) is not int or not lower <= v <= upper:
            raise KernelHold("SETTING_LIMIT")
    role = request.get("input_role")
    if role not in ("TEST_ONLY", "UNASSIGNED_PARAMETER_INPUT"):
        raise KernelHold("INPUT_ROLE_REQUIRED_NO_TARGET_ASSIGNED")
    goal = None if request.get("width_goal") is None else rational(request["width_goal"])
    if goal is not None and goal <= 0:
        raise KernelHold("INVALID_WIDTH_GOAL")
    ar = Arithmetic(bits, request.get("limits"))
    ar.tick(lo, hi)
    pi_record, pi = pi_witness(pn, ar)
    h = ar.div(ar.mul((Q(2), Q(2)), pi), (lo, hi))
    alpha = ar.mul(ar.power(h, 2), (Q(1, 2), Q(1, 2)))
    if alpha[0] <= 0 or alpha[0]*m*m < 3:
        raise KernelHold("CUTOFF_THRESHOLD_NOT_CERTIFIED")
    finite = {str(2*k): (Q(1), Q(1)) if k == 0 else (Q(0), Q(0)) for k in range(4)}
    modes = []
    for j in range(1, m+1):
        x = ar.mul(alpha, (Q(j*j), Q(j*j)))
        wlo = _exp_negative(x[1], en, ar)
        whi = _exp_negative(x[0], en, ar)
        weight = (_result(wlo)[0], _result(whi)[1])
        hm = ar.mul(h, (Q(j), Q(j)))
        contrib = {}
        for k in range(4):
            term = ar.mul(ar.mul(ar.power(hm, 2*k), weight), (Q(2), Q(2)))
            contrib[str(2*k)] = pack(term)
            finite[str(2*k)] = ar.add(finite[str(2*k)], term)
        modes.append({"m": j, "exp_at_hi": wlo, "exp_at_lo": whi,
                      "weight": pack(weight), "contribution": contrib})
    tail_exp = _exp_negative(alpha[0]*m*m, en, ar)
    eu = _result(tail_exp)[1]
    integrals = [ar.up(eu/(2*alpha[0]*m))]
    for k in range(1, 4):
        integrals.append(ar.up((Q(m**(2*k-1))*eu+(2*k-1)*integrals[-1])/(2*alpha[0])))
    tails = {str(2*k): ar.up(2*h[1]**(2*k)*integrals[k]) for k in range(4)}
    raw = {key: ar.add(v, (Q(0), tails[key])) for key, v in finite.items()}
    if raw["0"][0] < 1:
        raise KernelHold("ZERO_MODE_OR_NORMALIZER_ERROR")
    bounds = {"z": raw["0"]}
    bounds.update({"mu"+key: ar.div(raw[key], raw["0"]) for key in ("2", "4", "6")})
    spread_record, spread = None, Q(0)
    if lo != hi:
        two_pi2 = ar.mul(ar.power(pi, 2), (Q(2), Q(2)))
        al = ar.div(two_pi2, (lo*lo, lo*lo))
        ah = ar.div(two_pi2, (hi*hi, hi*hi))
        e_lhi = _exp_negative(ah[1], en, ar)
        e_llo = _exp_negative(al[0], en, ar)
        exact_lower = max(Q(0), 2*(_result(e_lhi)[0]-_result(e_llo)[1]))
        spread = ar.rounded(exact_lower, exact_lower)[0]
        spread_record = {"alpha_Llo": pack(al), "alpha_Lhi": pack(ah),
                         "exp_Lhi_at_alpha_hi": e_lhi, "exp_Llo_at_alpha_lo": e_llo}
    widths = {k: v[1]-v[0] for k, v in bounds.items()}
    outcome = "BOUNDS_ONLY" if goal is None else (
        "TOLERANCE_MET" if all(w <= goal for w in widths.values()) else
        "HOLD_INPUT_UNCERTAINTY" if spread > goal else "HOLD_TARGET_UNMET")
    canonical_request = {"L_lo": text(lo), "L_hi": text(hi), "cutoff": m,
                         "bits": bits, "pi_terms": pn, "exp_pairs": en,
                         "width_goal": None if goal is None else text(goal), "input_role": role,
                         "limits": ar.limits}
    return {"schema": "original-full-spectrum-moments-v1", "status": "ENCLOSED",
            "outcome": outcome, "request": canonical_request, "sources": dict(SOURCES),
            "semantics": "FOR_ALL_L_IN_CLOSED_INPUT_INTERVAL; NO_SCIENTIFIC_TARGET_ASSIGNED",
            "pi": pi_record, "h": pack(h), "alpha": pack(alpha), "modes": modes,
            "finite": {k: pack(v) for k, v in finite.items()}, "tail_exp": tail_exp,
            "tail_integral_upper": [text(v) for v in integrals],
            "tail_upper": {k: text(v) for k, v in tails.items()},
            "raw": {k: pack(v) for k, v in raw.items()},
            "bounds": {k: pack(v) for k, v in bounds.items()},
            "endpoint_spread": spread_record, "normalizer_input_spread_lower": text(spread),
            "error_accounting": {"producer_rounding_spec": "DYADIC_FLOOR_CEIL_EACH_LOCAL_CALL",
                                 "producer_round_quantum": text(Q(1, 1 << bits)),
                                 "checker_certifies": "FORWARD_CONTAINMENT_ONLY_NOT_MINIMAL_ROUNDING_OR_EXECUTION",
                                 "finite_widths": {k: text(v[1]-v[0]) for k, v in finite.items()},
                                 "finite_width_scope": "includes input uncertainty, pi, exp and arithmetic; not a global sum of local errors"}}

def _strict_json(data):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError("DUPLICATE_JSON_KEY")
            out[k] = v
        return out
    def constant(value):
        raise ValueError("NONFINITE_JSON_CONSTANT")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("request", help="explicit JSON request; no default target L")
    p.add_argument("--output")
    args = p.parse_args()
    try:
        with open(args.request, "rb") as f:
            data = f.read(65537)
        if len(data) > 65536:
            raise KernelHold("REQUEST_SIZE_LIMIT")
        try:
            request = _strict_json(data)
        except (ValueError, UnicodeError, RecursionError):
            raise KernelHold("INVALID_JSON_REQUEST") from None
        c = certify(request)
        s = json.dumps(c, indent=2, sort_keys=True, allow_nan=False)+"\n"
        if args.output:
            try:
                fd = os.open(args.output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            except FileExistsError:
                raise KernelHold("OUTPUT_ALREADY_EXISTS") from None
            with os.fdopen(fd, "w", encoding="utf-8") as f: f.write(s)
        else:
            print(s, end="")
        return 0
    except KernelHold as e:
        print(json.dumps({"status": "HOLD", "reason": e.reason}))
        return 2
    except OSError as e:
        print(json.dumps({"status": "HOLD", "reason": "IO_ERROR", "errno": e.errno}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
