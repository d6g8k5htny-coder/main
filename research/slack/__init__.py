"""Bound-slack registry: how loose is each bound the corpus states, exactly.

The corpus records bound-slack failures in prose — the headline one being
``chi2_grad_bound`` returning ``~1.57e14`` where the true ``|grad chi^2|`` is
``~1.563e-5``, a ``~1e19`` slack — and nothing in the repository noticed. This
package makes each such case a tracked record with exactly-computed arithmetic.

THE HONESTY RULE, enforced in code and not merely stated: **a slack ratio is a
statement about a bound's UTILITY, never about its CORRECTNESS.** A bound loose
by ``1e19`` is still, as far as this package knows, a true bound. Unsoundness
lives in a separate field that can only be set by supplying a
``SoundnessWitness`` this package re-verifies by exact interval comparison, so a
merely loose bound cannot be marked unsound.

See ``registry.py``'s module docstring for the severity ladder and its
justification, and ``README.md`` for what this package does not establish.
"""
from .registry import (  # noqa: F401
    DOES_NOT_ESTABLISH, FORBIDDEN_CORRECTNESS_WORDS, HONESTY_RULE, REGISTRY,
    SEARCH_LOG, SEVERITY_LADDER, AttainedKind, BoundDirection, Provenance,
    RatioKind, Severity, SlackRecord, SoundnessWitness, audit_registry,
    forbidden_words_in_report, order_of_magnitude, render_report, report,
    rounding_interval, sci, severity_of, sorted_records, truncated_interval,
)

__all__ = [
    "AttainedKind", "BoundDirection", "Provenance", "RatioKind", "Severity",
    "SlackRecord", "SoundnessWitness",
    "REGISTRY", "SEARCH_LOG", "SEVERITY_LADDER", "HONESTY_RULE",
    "DOES_NOT_ESTABLISH", "FORBIDDEN_CORRECTNESS_WORDS",
    "audit_registry", "forbidden_words_in_report", "order_of_magnitude",
    "render_report", "report", "rounding_interval", "sci", "severity_of",
    "sorted_records", "truncated_interval",
]
