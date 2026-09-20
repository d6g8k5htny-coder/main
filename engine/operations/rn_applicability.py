"""Source-bound RN algebra operations; no field-law or authority inference.

This is a new Git-side candidate interface, not a change to REGISTRY.json.
Caller assertions of provenance are checked for consistency, not authenticated.
"""
from dataclasses import dataclass, asdict
from fractions import Fraction as F
import hashlib
import json

from research.interval import Interval
from research.rn.gaussian_moments import ORDER, MomentEngine, moment_cap, rational

RN5_ID = "1LtnvNd0vAW-y3pzbyHLjgtF7Uw5sTph5"
RN5_SHA = "ac89f60b8206bfe011e6c2bc653e7acb39bc83a55c2a2c8fd1e17fd70c6c0383"
RETIRED = frozenset({"FOURTH_MOMENT_AS_SECOND", "CS_WITHOUT_SQRT_PROBABILITY",
                     "MIX_CONDITIONAL_COVARIANCE_MARGINAL_MEAN"})


@dataclass(frozen=True)
class Context:
    source_id: str
    source_sha256: str
    dimension: int
    order: tuple
    normalization: str
    conditioned_law: str
    domain: tuple
    evidence_tier: str = "EXACT_RATIONAL_ALGEBRA_CANDIDATE"

    def validate(self):
        if self.source_id != RN5_ID or self.source_sha256 != RN5_SHA:
            raise ValueError("stale or different RN5 source")
        if type(self.dimension) is not int or self.dimension != 2 or tuple(self.order) != ORDER:
            raise ValueError("RN5 requires dimension two and (xx, yy, xy)")
        if not isinstance(self.normalization, str) or not self.normalization.strip():
            raise ValueError("explicit normalization required")
        if not isinstance(self.conditioned_law, str) or not self.conditioned_law.strip():
            raise ValueError("explicit conditioned-law identity required")
        if self.evidence_tier != "EXACT_RATIONAL_ALGEBRA_CANDIDATE":
            raise ValueError("unsupported evidence tier; no field certification")
        if len(self.domain) != 2:
            raise ValueError("closed mark interval required")
        Interval(*(rational(x) for x in self.domain))


@dataclass(frozen=True)
class Law:
    context: Context
    intercept: tuple
    slope: tuple
    covariance: tuple
    mean_law: str
    covariance_law: str

    def fingerprint(self):
        """All numerical inputs participate; reusing a label cannot reuse a law."""
        payload = asdict(self)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str,
                                         separators=(",", ":")).encode()).hexdigest()


def applicable(law, expected, *, cost=None):
    law.context.validate()
    expected.validate()
    for key in asdict(expected):
        if cost is not None:
            cost.charge()
        if getattr(law.context, key) != getattr(expected, key):
            raise ValueError("applicability mismatch: " + key)
    if cost is not None:
        cost.charge(2)
    if law.mean_law != expected.conditioned_law or law.covariance_law != expected.conditioned_law:
        raise ValueError("mean and covariance do not identify the same conditioned law")


def apply_moment_bound(law, expected, *, degree, source_bytes, strategy="BERNSTEIN_RN5", depth=2):
    if strategy in RETIRED or strategy != "BERNSTEIN_RN5":
        raise ValueError("retired or unknown operation strategy")
    applicable(law, expected)
    if hashlib.sha256(source_bytes).hexdigest() != expected.source_sha256:
        raise ValueError("source bytes do not match pinned RN5 identity")
    polynomial = MomentEngine(law.intercept, law.slope, law.covariance,
                              order=law.context.order).determinant(degree)
    cap = moment_cap(polynomial, Interval(*expected.domain), depth=depth)
    return {"operation": "RN5_AFFINE_MOMENT_BOUND_V1", "law_sha256": law.fingerprint(),
            "source_sha256": expected.source_sha256, "context": asdict(expected),
            "degree": degree, "polynomial": polynomial, "upper": cap,
            "authority": "NONE", "independence_credit": 0,
            "field_certified": False, "original_prize_closed": False,
            "does_not_establish": "No identified RN field law, spatial cover, covariance-family enclosure, "
            "weighted Palm bound, novelty, research-wide utility or status change."}
