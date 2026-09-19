#!/usr/bin/env python3
"""Run a catalogued operation's displayed exact identity and record a trial.

The register tab ``reusable_operations`` (GP-REG-032-v1.2, sheet 42, 2026-09-18
export) catalogues fifteen operations OP01-OP15. ``engine/operations/REGISTRY.json``
transcribes every one of them cell-for-cell and adds, kept separate under
``git_side``, whether the ``Action / output`` cell displays an identity that
exact arithmetic can reproduce *in full from the cell alone*. For those four
(OP02, OP03, OP04, OP05) this module performs that arithmetic over
``fractions.Fraction`` -- exponents as Fractions, constants such as
``6^(2/3)/(3 kappa^(2/3))`` as (rational coefficient, exponent vector) pairs,
never as floats -- and writes one JSON record in the shape of the register's
own trial ledger (tab ``operation_trials``, sheet 43): its eighteen columns are
the record's top-level keys, plus ``does_not_establish`` and ``authority``.

    python3 engine/operations/trial.py --list
    python3 engine/operations/trial.py --op OP02 --out engine/operations/trials/
    python3 engine/operations/trial.py --all  --out engine/operations/trials/

Records are append-only: the writer creates files with ``open(path, "x")`` and
refuses a path that exists.

A record's ``Run evidence.utc`` is the time the runner ran, read from the
clock (``utc_now()``, microsecond precision). ``--utc`` exists for tests only
(they use it to make two runs collide and to prove the second is refused, and
to give sandbox records a known stamp); a committed record must never carry a
supplied stamp. The runner refuses a ``utc`` later than the clock, and
``tools/operations_check.py`` refuses a record whose ``utc`` is later than
the check's own clock or, for a record in ``git HEAD``, later than the commit
that added it. A record whose stated time is false is exactly the dishonest
record this repository exists to exclude.

The ``Verified result`` vocabulary is CLOSED: ``IDENTITY_REPRODUCED``,
``IDENTITY_NOT_REPRODUCED``, ``NOT_RUN``. A computation that raises is
recorded as ``IDENTITY_NOT_REPRODUCED`` with the failure named and a failure
charge of 1; it is never swallowed into success. An operation the registry
marks ``NOT_MACHINE_CHECKABLE_HERE`` can only ever be recorded ``NOT_RUN``.

WHAT THIS MODULE DOES NOT ESTABLISH
-----------------------------------
A trial is a record of a computation, not evidence. ``IDENTITY_REPRODUCED``
says that the algebra the cell displays was reproduced exactly from the cell;
it does not say the operation's premises hold anywhere, that any caller
satisfies its ``Required scope``, that the source the cell cites is correct,
or that the operation is useful or new -- Utility stays ``UNMEASURED`` and
Novelty ``NOT_ASSESSED``, which are the register's words and are transcribed,
not decided, here. Nothing here moves a claim, premise, obligation, gate or
grade. The five validity premises of Theorem D1 v2.2(2) are OPEN and
``D3-LEMMA-RN-UNIF`` remains open. No original prize problem is solved.

The status-word and usefulness-word scans below are word lists applied to the
repository-authored text of a record, not a reading of what a sentence means;
a status claimed in other words is not caught by them. They are a guard
against the obvious, and the does_not_establish text of every record is the
statement that carries the weight -- which is why that text is not free: the
checker requires every trial to carry ``GENERIC_DOES_NOT_ESTABLISH`` verbatim,
plus the operation's own sentence and, for NOT_RUN, ``NOT_RUN_DOES_NOT_ESTABLISH``,
and requires the registry to carry ``REGISTRY_DOES_NOT_ESTABLISH``,
``REGISTRY_AUTHORITY`` and ``REGISTRY_UTILITY_AND_NOVELTY`` verbatim. Extra
sentences may be added; the fixed ones may not be removed or replaced.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCHEMA = "q0.operation.trial/v1"
REGISTRY_SCHEMA = "q0.operations.registry/v1"
DEFAULT_REGISTRY = os.path.join(ROOT, "engine", "operations", "REGISTRY.json")
DEFAULT_REGISTER = os.path.join(ROOT, "registers", "json", "reusable_operations.json")
DEFAULT_TRIALS_TAB = os.path.join(ROOT, "registers", "json", "operation_trials.json")

#: The eighteen columns of the register's ``operation_trials`` tab, in order.
#: A trial record's top-level keys are exactly these plus the two repository
#: fields. ``tools/operations_check.py`` asserts the tab header still reads so.
TRIAL_COLUMNS: Tuple[str, ...] = (
    "Trial ID", "Problem ID / SHA-256", "Split", "Operation ID",
    "Library / catalog SHA-256", "Arm", "Budget / cost unit",
    "Seed / environment", "Verified result", "Search cost", "Retrieval cost",
    "Verification cost", "Acquisition cost", "Maintenance cost",
    "Timeout / failure charge", "Run evidence", "Scope / constraints", "Notes",
)
REPOSITORY_FIELDS: Tuple[str, ...] = ("does_not_establish", "authority")
COST_COLUMNS: Tuple[str, ...] = (
    "Search cost", "Retrieval cost", "Verification cost", "Acquisition cost",
    "Maintenance cost",
)
#: The register's twelve ``reusable_operations`` columns, in order.
REGISTER_COLUMNS: Tuple[str, ...] = (
    "Operation", "Use when", "Reuse state", "Required scope", "Action / output",
    "Evidence checked", "Exact source", "Current review / authority",
    "Do not infer", "Next useful step", "Utility", "Novelty",
)

VERIFIED_RESULTS = frozenset({"IDENTITY_REPRODUCED", "IDENTITY_NOT_REPRODUCED", "NOT_RUN"})
TRIAL_KINDS = frozenset({"EXACT_IDENTITY", "NOT_MACHINE_CHECKABLE_HERE"})
GIT_SIDE_KEYS = ("exact_identity_checkable", "trial_kind", "why", "research_module")
#: Closed vocabularies for two ledger columns this repository cannot honestly
#: fill any other way. The Drive guide calls its own checks "exposed
#: development tests, not held-out usefulness trials"; so are these. No
#: comparison arm (fixed library / lemma cache / scope-aware operations) is
#: run here, so ``Arm`` says so.
SPLITS = frozenset({"EXPOSED_DEVELOPMENT", "NOT_APPLICABLE"})
ARMS = frozenset({"NONE_SINGLE_EXACT_REPLAY", "NOT_APPLICABLE"})
NOT_MEASURED = "NOT_MEASURED"
COST_UNIT = "checked_equalities"
#: The only text the runner writes into these two fields; the checker requires
#: them verbatim so a hand-edited record cannot describe an arithmetic or a
#: budget the runner never had.
BUDGET = (f"unit={COST_UNIT}; budget=NONE (one deterministic exact replay, unbounded); "
          "failure rule: a computation that raises is IDENTITY_NOT_REPRODUCED and charged 1")
ARITHMETIC = ("exact rational (fractions.Fraction); constants as (rational coefficient, "
              "exponent vector) monomials; exact integer roots; no floats anywhere")
AUTHORITY = "NONE — a trial is a record of a computation, not evidence"
#: ``Seed / environment`` is exactly these six keys, in this order. ``seed``
#: and ``runner`` are fixed sentences; ``repository_commit_at_run`` is null or
#: a 40-hex commit; the checker refuses any other shape, so a record cannot
#: claim a seed, a runner or an interpreter the runner never had.
ENVIRONMENT_KEYS: Tuple[str, ...] = (
    "seed", "python", "implementation", "platform", "repository_commit_at_run", "runner",
)
SEED = "NONE (deterministic; no randomness anywhere in the computation)"
RUNNER_PATH = "engine/operations/trial.py"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
PYTHON_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+[0-9A-Za-z+.\-]*$")
OP_ID_RE = re.compile(r"^OP\d{2}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")

#: The registry's fixed top-level sentences. ``REGISTRY.json`` must carry them
#: verbatim (``tools/operations_check.py`` requires equality for ``authority``
#: and ``utility_and_novelty`` and containment of every sentence of
#: ``REGISTRY_DOES_NOT_ESTABLISH``): a registry whose authority read
#: "canonical and final" once passed a checker that only required the field
#: to be non-empty.
REGISTRY_AUTHORITY = (
    "NONE — the twelve cells of every entry are the register's words, transcribed, not "
    "decided, here; git_side is repository bookkeeping about what this repository can "
    "compute and carries no mathematical authority")
REGISTRY_UTILITY_AND_NOVELTY = (
    "Every entry's Utility and Novelty are the register's cells (UNMEASURED / NOT_ASSESSED). "
    "This file never says more, in a cell, in git_side or in these sentences; "
    "tools/operations_check.py fails if it does.")
REGISTRY_DOES_NOT_ESTABLISH: List[str] = [
    "Transcribing an operation here does not establish that it is correct, reviewed or "
    "admissible anywhere, and it leaves the register's Utility cell UNMEASURED and its "
    "Novelty cell NOT_ASSESSED; the register's Reuse state, Current review / authority and "
    "Do not infer cells govern and are carried verbatim.",
    "exact_identity_checkable = true means only that the Action / output cell displays "
    "algebra this repository can reproduce in exact arithmetic from the cell alone; it says "
    "nothing about the operation's premises, scope or source, and a trial that reproduces "
    "it is a record, not evidence.",
    "exact_identity_checkable = false is a statement about this repository's reach, not "
    "about the operation: it does not mean the operation is wrong, unverifiable or "
    "unverified elsewhere.",
    "Nothing here moves a claim, premise, obligation, gate or grade. The five validity "
    "premises of Theorem D1 v2.2(2) are OPEN and D3-LEMMA-RN-UNIF remains open. No original "
    "prize problem is solved.",
]
#: The registry's provenance blocks are pinned too: ``register.export`` and
#: ``register.row_sha256_rule`` are these sentences, and the checker compares
#: ``register.path`` / ``tab`` / ``sheet_index`` and ``trial_ledger.path`` /
#: ``tab`` / ``sheet_index`` / ``columns`` / ``record_schema`` / ``records_dir``
#: with the files it was actually pointed at.
REGISTER_EXPORT = "GP-REG-032-v1.2 xlsx export of 2026-09-18 (registers/source/)"
ROW_SHA256_RULE = ('sha256 of json.dumps(row, ensure_ascii=False, separators=(",",":")) over '
                   'the 12-cell row list')

GENERIC_DOES_NOT_ESTABLISH: List[str] = [
    "A trial is a record that an exact computation ran on a register cell. It is "
    "not evidence, not a review verdict and not a status, and it warrants nothing; "
    "it moves no claim, premise, obligation, gate or grade, and it earns no "
    "independence credit for anything.",
    "IDENTITY_REPRODUCED means only that the algebra the Action / output cell "
    "displays was reproduced in exact arithmetic from the cell alone. It does not "
    "establish that the operation's premises hold in any application, that any "
    "caller meets its Required scope, or that the source object the cell "
    "cites is correct.",
    "Utility remains UNMEASURED and Novelty remains NOT_ASSESSED, as the register "
    "says; this trial measures neither. It is an exposed development check on the "
    "register's own displayed cell, not a held-out trial; Utility stays UNMEASURED "
    "and Novelty stays NOT_ASSESSED, and no gain of either sign is assigned to the "
    "operation.",
    "The five validity premises of Theorem D1 v2.2(2) are OPEN and D3-LEMMA-RN-UNIF "
    "remains open. Nothing recorded here composes the 2D upper or lower tracks with "
    "the 3D lifetime track, and no original prize problem is solved.",
]

OP_DOES_NOT_ESTABLISH: Dict[str, str] = {
    "OP02": "The r^2 normalizer factor is multiplied as the Required scope names it "
            "and is not justified here; the constant 1/6 is a power ledger, not a "
            "contact coefficient and not a limiting mass.",
    "OP03": "The Jacobian identity is local algebra for one fixed kappa > 0 under the "
            "caller-supplied measure r dr. It is not an asymptotic; weighted mass, "
            "domination and tails, mark integration and the selection bridge are "
            "untouched by it.",
    "OP04": "What is reproduced is the exponent arithmetic only: 1/4 + 1/4 + 1/2 = 1, "
            "which the displayed Hölder application requires, and its agreement with the "
            "moment orders 4,4,2 of the Required scope cell. The inequality "
            "E[|ABC| I] <= (E A^4 E B^4)^(1/4) (E C^2)^(1/2) itself, the finiteness of any "
            "moment and any SIDE24 input are not checked here, and this trial decides "
            "nothing about them.",
    "OP05": "The witness A=B=1, C=1/4 is one deterministic point. It reproduces the "
            "register's refutation of the wrong universal bound at that point and "
            "says nothing about any field value, any RN3 number or any consumer "
            "beyond what the RN5 source says; RN5's own Gaussian witness is not "
            "replayed, and the corrected bound of OP04 is not evaluated at the point "
            "because the cell does not display that.",
}

#: The sentence every NOT_RUN record carries, verbatim, in addition to the
#: generic ones. The checker requires it.
NOT_RUN_DOES_NOT_ESTABLISH = (
    "NOT_RUN records that the operation was consulted and that this repository cannot "
    "reproduce its Action / output cell by exact arithmetic from the cell alone; it records "
    "no computation and says nothing for or against the operation.")


def required_does_not_establish(op_id: str, verdict: str) -> List[str]:
    """The sentences a trial for ``op_id`` with ``verdict`` must carry verbatim:
    the generic ones, the operation's own when one is defined, and the NOT_RUN
    sentence when the verdict is NOT_RUN. The writer emits exactly these; the
    checker requires each of them to be present (extra sentences are allowed and
    are scanned like any other authored text)."""
    out = list(GENERIC_DOES_NOT_ESTABLISH)
    if op_id in OP_DOES_NOT_ESTABLISH:
        out.insert(1, OP_DOES_NOT_ESTABLISH[op_id])
    if verdict == "NOT_RUN":
        out.insert(1, NOT_RUN_DOES_NOT_ESTABLISH)
    return out

# ---------------------------------------------------------------------------
# what repository-authored text may not say
# ---------------------------------------------------------------------------

#: Words that read as a status or a review verdict. None may appear, whole
#: word and in any case, in any text this repository authors about an
#: operation or a trial: the writer refuses to build such a record and
#: ``tools/operations_check.py`` refuses to pass one. The list carries, for
#: each family, the participle, the verb forms and the noun (PROVEN, PROVES,
#: PROOF, ...), because a round of adversarial mutation found the verb forms
#: (``proves``, ``certifies``, ``closes``, ``promotes``, ``discharges``,
#: ``passes``) and the nouns (``certificate``, ``proof``) slipping through a
#: participle-only list. It is a word list, not a reading: a sentence that
#: claims a status in other words is not caught by it, which is why every
#: record also says in its own does_not_establish that it moves nothing.
STATUS_WORDS: Tuple[str, ...] = (
    # the five words of the repository's first non-negotiable rule, in every form
    "PROVE", "PROVES", "PROVED", "PROVEN", "PROOF", "PROOFS",
    "CERTIFY", "CERTIFIES", "CERTIFIED", "CERTIFICATE", "CERTIFICATES", "CERTIFICATION",
    "CLOSE", "CLOSES", "CLOSED", "CLOSURE",
    "PROMOTE", "PROMOTES", "PROMOTED", "PROMOTION",
    "DISCHARGE", "DISCHARGES", "DISCHARGED",
    # review-verdict and gate words
    "INDEPENDENT",
    "RATIFY", "RATIFIES", "RATIFIED", "RATIFICATION",
    "VERIFY", "VERIFIES", "VERIFIED", "VERIFICATION",
    "ACCEPT", "ACCEPTS", "ACCEPTED", "ACCEPTANCE",
    "ADMIT", "ADMITS", "ADMITTED", "ADMISSION",
    "PASS", "PASSES", "PASSED", "PASS_TECHNICAL",
    "APPROVE", "APPROVES", "APPROVED", "APPROVAL",
    "SATISFY", "SATISFIES", "SATISFIED",
    "VALIDATE", "VALIDATES", "VALIDATED", "VALIDATION",
    "CONFIRM", "CONFIRMS", "CONFIRMED", "CONFIRMATION",
    "ESTABLISHED", "ESTABLISHES",
    # promotion and release vocabulary (canonical promotion and external release
    # are the operator's acts), found slipping through a third adversarial round
    "CANONICAL", "FINAL", "AUTHORITATIVE", "OFFICIAL", "RELEASED",
    # words that state an identity or claim as standing
    "SETTLED", "HOLDS", "RESOLVED",
)
STATUS_RE = re.compile(r"\b(" + "|".join(STATUS_WORDS) + r")\b", re.IGNORECASE)
#: Words that speak of an operation's usefulness, novelty or gain. A sentence
#: of repository-authored text that uses one must carry the register's own
#: words ``UNMEASURED`` or ``NOT_ASSESSED`` (that case, verbatim) and may not
#: pair it with a value word; the register's Utility and Novelty cells are the
#: only source of those two facts and they say UNMEASURED / NOT_ASSESSED.
#: The synonyms (valuable, beneficial, improves, important, ...) are listed
#: because an adversarial round found them slipping through. As with
#: ``STATUS_WORDS`` this is a word list, not a reading of the sentence.
USEFULNESS_WORDS: Tuple[str, ...] = (
    "utility", "utilities", "novelty", "novelties", "useful", "usefulness", "novel",
    "gain", "gains", "gained",
    "valuable", "beneficial", "benefit", "benefits", "benefited",
    "improve", "improves", "improved", "improvement", "improvements",
    "helpful", "important", "importance", "worthwhile", "effective", "effectiveness",
)
VALUE_WORDS: Tuple[str, ...] = (
    "high", "low", "medium", "moderate", "new", "known", "measured", "assessed",
    "positive", "substantial", "significant", "large", "small",
    # phrases that assert a value without naming one
    "beyond question", "beyond doubt", "beyond dispute", "unquestionable", "undeniable",
)
REGISTER_MARKERS: Tuple[str, ...] = ("UNMEASURED", "NOT_ASSESSED")
#: The one register header name that contains a usefulness word. Only this
#: exact, case-sensitive phrase is exempt from the scan; the bare words
#: ``Utility`` and ``Novelty`` are not, because they are what a claim would use.
EXEMPT_PHRASES: Tuple[str, ...] = ("Next useful step",)
#: Column names are the register's, not a claim; every other key is scanned.
EXEMPT_KEYS = frozenset(TRIAL_COLUMNS) | frozenset(REGISTER_COLUMNS)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+|\s*\|\s*|\n+")
_USEFULNESS_RE = re.compile(r"\b(" + "|".join(USEFULNESS_WORDS) + r")\b", re.IGNORECASE)
_VALUE_RE = re.compile(r"\b(" + "|".join(VALUE_WORDS) + r")\b", re.IGNORECASE)


def authored_strings(obj: Any) -> List[str]:
    """Every string in ``obj``, keys included, except a string that is exactly
    a register column name: that is the register's word, not the repository's."""
    if isinstance(obj, str):
        return [] if obj in EXEMPT_KEYS else [obj]
    if isinstance(obj, dict):
        out: List[str] = []
        for k, v in obj.items():
            out.extend(authored_strings(str(k)))
            out.extend(authored_strings(v))
        return out
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.extend(authored_strings(v))
        return out
    return []


def status_word_hits(obj: Any) -> List[str]:
    """The status words (upper-cased, sorted, unique) found anywhere in ``obj``."""
    hits = set()
    for s in authored_strings(obj):
        for m in STATUS_RE.finditer(s):
            hits.add(m.group(1).upper())
    return sorted(hits)


def usefulness_hits(text: str) -> List[str]:
    """Sentences of ``text`` that speak of usefulness, novelty or gain without
    the register's UNMEASURED / NOT_ASSESSED, or that pair such a word with a
    value word. Returns one short reason per offending sentence."""
    out: List[str] = []
    for phrase in EXEMPT_PHRASES:
        text = text.replace(phrase, " ")
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        m = _USEFULNESS_RE.search(sentence)
        if not m:
            continue
        v = _VALUE_RE.search(sentence)
        if v:
            out.append(f"{m.group(1)!r} paired with the value word {v.group(1)!r}")
        elif not any(marker in sentence for marker in REGISTER_MARKERS):
            out.append(f"{m.group(1)!r} without the register's UNMEASURED / NOT_ASSESSED")
    return out


def usefulness_hits_in(obj: Any) -> List[str]:
    out: List[str] = []
    for s in authored_strings(obj):
        out.extend(usefulness_hits(s))
    return out


def authored_fields(record: Dict[str, Any], cells: Dict[str, str]) -> Dict[str, Any]:
    """The parts of a trial record this repository wrote: everything except
    ``Scope / constraints`` (the Required scope cell) and, inside ``Notes``,
    the quoted ``Do not infer`` cell. The quoted Utility and Novelty cells stay
    in: they are the register's UNMEASURED / NOT_ASSESSED, which the scan
    requires, never a claim."""
    authored = {k: v for k, v in record.items() if k != "Scope / constraints"}
    notes = record.get("Notes")
    if isinstance(notes, str):
        cell = cells.get("Do not infer", "")
        authored["Notes"] = notes.replace(cell, " ") if cell else notes
    return authored


# ---------------------------------------------------------------------------
# exact monomial arithmetic: (rational coefficient, exponent vector)
# ---------------------------------------------------------------------------

def _factor(n: int) -> Dict[int, int]:
    """Prime factorisation of a positive integer by trial division."""
    if n <= 0:
        raise ValueError(f"cannot factor non-positive integer {n}")
    out: Dict[int, int] = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


class Monomial:
    """``coefficient * prod base^exponent`` with a rational coefficient and
    rational exponents. Integer bases are kept as primes ("2", "3") and folded
    into the coefficient whenever their exponent is an integer, so equality of
    canonical forms is exact equality of the expressions.
    """

    __slots__ = ("coefficient", "exponents")

    def __init__(self, coefficient: Any, exponents: Optional[Dict[str, Any]] = None):
        self.coefficient = Fraction(coefficient)
        exps: Dict[str, Fraction] = {}
        for base, e in (exponents or {}).items():
            e = Fraction(e)
            if e == 0:
                continue
            if isinstance(base, int) or str(base).isdigit():
                for p, m in _factor(int(base)).items():
                    exps[str(p)] = exps.get(str(p), Fraction(0)) + m * e
            else:
                if not re.match(r"^[A-Za-z_][A-Za-z_0-9]*$", str(base)):
                    raise ValueError(f"bad symbol {base!r}")
                exps[str(base)] = exps.get(str(base), Fraction(0)) + e
        self.exponents = self._fold(exps)

    def _fold(self, exps: Dict[str, Fraction]) -> Dict[str, Fraction]:
        out: Dict[str, Fraction] = {}
        for base, e in exps.items():
            if e == 0:
                continue
            if base.isdigit() and e.denominator == 1:
                self.coefficient *= Fraction(int(base)) ** int(e)
            else:
                out[base] = e
        return dict(sorted(out.items()))

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "Monomial":
        return cls(Fraction(obj["coefficient"]),
                   {k: Fraction(v) for k, v in obj.get("exponents", {}).items()})

    def to_json(self) -> Dict[str, Any]:
        return {"coefficient": str(self.coefficient),
                "exponents": {k: str(v) for k, v in self.exponents.items()}}

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, Monomial) and self.coefficient == other.coefficient
                and self.exponents == other.exponents)

    def __hash__(self) -> int:  # pragma: no cover - not used as a key
        return hash((self.coefficient, tuple(self.exponents.items())))

    def __mul__(self, other: "Monomial") -> "Monomial":
        exps = dict(self.exponents)
        for k, v in other.exponents.items():
            exps[k] = exps.get(k, Fraction(0)) + v
        return Monomial(self.coefficient * other.coefficient, exps)

    def __pow__(self, e: Any) -> "Monomial":
        e = Fraction(e)
        if self.coefficient == 0:
            if e <= 0:
                raise ValueError("0 to a non-positive power")
            return Monomial(0)
        if self.coefficient < 0 and e.denominator != 1:
            raise ValueError("fractional power of a negative coefficient is not a "
                             "real monomial")
        exps = {k: v * e for k, v in self.exponents.items()}
        sign = -1 if self.coefficient < 0 else 1
        num, den = abs(self.coefficient.numerator), self.coefficient.denominator
        for p, m in _factor(num).items():
            exps[str(p)] = exps.get(str(p), Fraction(0)) + m * e
        for p, m in _factor(den).items():
            exps[str(p)] = exps.get(str(p), Fraction(0)) - m * e
        return Monomial(sign, exps)

    def derivative(self, symbol: str) -> "Monomial":
        e = self.exponents.get(symbol, Fraction(0))
        if e == 0:
            return Monomial(0)
        exps = dict(self.exponents)
        exps[symbol] = e - 1
        return Monomial(self.coefficient * e, exps)

    def __repr__(self) -> str:
        parts = [str(self.coefficient)]
        parts += [f"{k}^({v})" for k, v in self.exponents.items()]
        return " * ".join(parts)


def exact_root(x: Fraction, n: int) -> Fraction:
    """The exact positive n-th root of a non-negative rational that is a
    perfect n-th power, or ``ValueError``. Never a float."""
    x = Fraction(x)
    if n <= 0:
        raise ValueError("root order must be positive")
    if x < 0:
        raise ValueError(f"no real {n}-th root of {x}")
    num, den = x.numerator, x.denominator
    rn, rd = _iroot(num, n), _iroot(den, n)
    if rn ** n != num or rd ** n != den:
        raise ValueError(f"{x} is not a perfect {n}-th power")
    return Fraction(rn, rd)


def _iroot(k: int, n: int) -> int:
    """floor(k ** (1/n)) by integer bisection."""
    if k < 2:
        return k
    lo, hi = 0, 1 << ((k.bit_length() + n - 1) // n + 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** n <= k:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# the four identities, as data bound to the register cells
# ---------------------------------------------------------------------------

#: Each spec names, in ``displayed``, the exact substring of the register's
#: ``Action / output`` cell it reproduces, and, where the arithmetic also takes
#: data from the ``Required scope`` cell, names that verbatim in
#: ``scope_displayed``; the writer and the checker refuse a spec whose
#: displayed text is not in its cell, so the arithmetic can never drift from
#: the register's words. Everything else in a spec is the cell's content
#: restated as exact data.
IDENTITIES: Dict[str, Dict[str, Any]] = {
    "OP02": {
        "displayed": "(r^3/6) r^-5 r^2 = 1/6",
        "scope_displayed": "gap_factor=r^3/6; density_factor=r^-5; normalizer_factor=r^2",
        "kind": "monomial_product",
        "factors": [
            {"name": "gap_factor r^3/6", "coefficient": "1/6", "exponents": {"r": "3"}},
            {"name": "density_factor r^-5", "coefficient": "1", "exponents": {"r": "-5"}},
            {"name": "normalizer_factor r^2", "coefficient": "1", "exponents": {"r": "2"}},
        ],
        "expected": {"coefficient": "1/6", "exponents": {}},
    },
    "OP03": {
        "displayed": "r dr = 6^(2/3)/(3 kappa^(2/3)) ell^(-1/3) d ell",
        "scope_displayed": "gap_map=ell=kappa*r^3/6",
        "kind": "pushforward_jacobian",
        "gap_map": {"coefficient": "1/6", "exponents": {"kappa": "1", "r": "3"}},
        "inverse_displayed": "r=(6 ell/kappa)^(1/3)",
        "inverse": {"coefficient": "1", "exponents": {"6": "1/3", "ell": "1/3", "kappa": "-1/3"}},
        "expected": {"coefficient": "1/3", "exponents": {"6": "2/3", "kappa": "-2/3", "ell": "-1/3"}},
        "expected_ell_exponent": "-1/3",
    },
    "OP04": {
        "displayed": "(E A^4 E B^4)^(1/4) (E C^2)^(1/2)",
        "scope_displayed": "moment_orders=4,4,2",
        "kind": "holder_exponents",
        "reproduces": "exponent arithmetic only: the displayed exponents 1/4, 1/4, 1/2 sum to "
                      "one and are the reciprocals of the Required scope moment orders 4,4,2; "
                      "the inequality E[|ABC| I] <= (E A^4 E B^4)^(1/4) (E C^2)^(1/2) is a "
                      "theorem about expectations and is not checked",
        "exponents": ["1/4", "1/4", "1/2"],
        "moment_orders": ["4", "4", "2"],
        "expected_sum": "1",
    },
    "OP05": {
        "displayed": "A=B=1, C=1/4 gives LHS=1/4 but wrong RHS=1/16",
        "kind": "rational_witness",
        "reproduces": "the displayed witness only: LHS = |ABC| = 1/4, the wrong RHS "
                      "(E A^4 E B^4)^(1/4) (E C^4)^(1/2) = 1/16 by exact integer roots, and "
                      "the strict comparison 1/4 > 1/16 the cell's 'but' states; the corrected "
                      "RHS (E C^2)^(1/2) of OP04 is not evaluated at the witness because the "
                      "cell does not display it",
        "A": "1", "B": "1", "C": "1/4",
        "expected_lhs": "1/4",
        "expected_wrong_rhs": "1/16",
    },
}


def _step(statement: str, lhs: Any, rhs: Any) -> Dict[str, Any]:
    def show(v: Any) -> Any:
        if isinstance(v, Monomial):
            return v.to_json()
        if isinstance(v, Fraction):
            return str(v)
        if isinstance(v, bool):
            return v
        if isinstance(v, (list, tuple)):
            return [show(x) for x in v]
        return str(v)
    equal = lhs == rhs
    return {"statement": statement, "lhs": show(lhs), "rhs": show(rhs), "equal": bool(equal)}


def _eval_monomial_product(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    factors = [Monomial.from_json(f) for f in spec["factors"]]
    total_exp = sum((Fraction(e) for f in spec["factors"] for e in f["exponents"].values()),
                    Fraction(0))
    steps.append(_step("sum of the r exponents of every supplied factor", total_exp, Fraction(0)))
    coeff = Fraction(1)
    for f in factors:
        coeff *= f.coefficient
    steps.append(_step("product of the coefficients", coeff,
                       Fraction(spec["expected"]["coefficient"])))
    prod = Monomial(1)
    for f in factors:
        prod = prod * f
    steps.append(_step("product of the factors as a monomial equals the displayed constant",
                       prod, Monomial.from_json(spec["expected"])))
    return steps


def _eval_pushforward(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    gap = Monomial.from_json(spec["gap_map"])            # ell = kappa r^3 / 6
    inv = Monomial.from_json(spec["inverse"])            # r = (6 ell / kappa)^(1/3)
    # substitute r = inv into the gap map: kappa * inv^3 / 6 must be ell
    back = Monomial(gap.coefficient, {k: v for k, v in gap.exponents.items() if k != "r"})
    back = back * (inv ** gap.exponents.get("r", Fraction(0)))
    steps.append(_step("gap map evaluated at the displayed inverse returns ell",
                       back, Monomial(1, {"ell": 1})))
    drdl = inv.derivative("ell")
    rdr = inv * drdl
    expected = Monomial.from_json(spec["expected"])
    steps.append(_step("r(ell) * dr/dell as a monomial equals the displayed constant times "
                       "ell^(-1/3)", rdr, expected))
    steps.append(_step("the ell exponent of r dr / d ell", rdr.exponents.get("ell", Fraction(0)),
                       Fraction(spec["expected_ell_exponent"])))
    steps.append(_step("the displayed constant 6^(2/3)/(3 kappa^(2/3)) in reduced monomial "
                       "form equals the computed constant",
                       Monomial(expected.coefficient, {k: v for k, v in expected.exponents.items()
                                                       if k != "ell"}),
                       Monomial(rdr.coefficient, {k: v for k, v in rdr.exponents.items()
                                                  if k != "ell"})))
    return steps


def _eval_holder(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    exps = [Fraction(e) for e in spec["exponents"]]
    steps.append(_step("the displayed Hölder exponents sum to one (conjugacy)",
                       sum(exps, Fraction(0)), Fraction(spec["expected_sum"])))
    recip = [Fraction(1, int(m)) for m in spec["moment_orders"]]
    steps.append(_step("the reciprocals of the Required scope moment orders are the "
                       "displayed exponents", recip, exps))
    return steps


def _eval_witness(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps = []
    A, B, C = (Fraction(spec[k]) for k in ("A", "B", "C"))
    lhs = abs(A * B * C)                                  # deterministic point, I = 1
    steps.append(_step("LHS = |ABC| at the deterministic witness", lhs,
                       Fraction(spec["expected_lhs"])))
    ea4, eb4, ec4 = A ** 4, B ** 4, C ** 4
    wrong = exact_root(ea4 * eb4, 4) * exact_root(ec4, 2)
    steps.append(_step("wrong RHS = (E A^4 E B^4)^(1/4) (E C^4)^(1/2) at the witness",
                       wrong, Fraction(spec["expected_wrong_rhs"])))
    steps.append(_step("the wrong bound fails at the witness: LHS > wrong RHS",
                       lhs > wrong, True))
    # The corrected RHS (E C^2)^(1/2) of OP04 is deliberately not evaluated:
    # the cell displays LHS, the wrong RHS and their comparison, nothing more,
    # and a trial reproduces displayed algebra only (spec["reproduces"]).
    return steps


_EVALUATORS = {
    "monomial_product": _eval_monomial_product,
    "pushforward_jacobian": _eval_pushforward,
    "holder_exponents": _eval_holder,
    "rational_witness": _eval_witness,
}


def evaluate(spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Run one identity spec. Returns ``(steps, failure)``. A raise is caught
    and named in ``failure``; it is never turned into a passing step."""
    try:
        fn = _EVALUATORS[spec["kind"]]
        steps = fn(spec)
        return steps, None
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return [], f"{type(exc).__name__}: {exc}"


def verdict_from(steps: List[Dict[str, Any]], failure: Optional[str]) -> str:
    if failure is not None or not steps:
        return "IDENTITY_NOT_REPRODUCED"
    return "IDENTITY_REPRODUCED" if all(s["equal"] is True for s in steps) \
        else "IDENTITY_NOT_REPRODUCED"


def spec_bound_to_cell(spec: Dict[str, Any], action_cell: str,
                       scope_cell: Optional[str] = None) -> bool:
    """The spec's ``displayed`` text must appear verbatim in the register's
    ``Action / output`` cell (and, when named, the inverse too), and its
    ``scope_displayed`` text, when named, verbatim in the ``Required scope``
    cell. A spec with no ``displayed`` string is bound to nothing."""
    displayed = spec.get("displayed")
    if not isinstance(displayed, str) or not displayed or displayed not in action_cell:
        return False
    inv = spec.get("inverse_displayed")
    if inv is not None and inv not in action_cell:
        return False
    scope = spec.get("scope_displayed")
    if scope is not None:
        if scope_cell is None or scope not in scope_cell:
            return False
    return True


def trial_id(op_id: str, row_sha: str, registry_sha: str, spec: Optional[Dict[str, Any]],
             utc: str) -> str:
    """``TRIAL-<op>-<YYYYMMDDTHHMMSSZ>-<16 hex>``: the hex is the head of the
    sha256 of the canonical JSON of the inputs, so the id is deterministic and
    ``tools/operations_check.py`` recomputes it from the record."""
    input_digest = sha256_text(canonical_json({
        "operation_id": op_id, "row_sha256": row_sha,
        "registry_sha256": registry_sha, "spec": spec}))
    stamp = re.sub(r"[-:]", "", utc.split(".")[0].rstrip("Z")) + "Z"
    return f"TRIAL-{op_id}-{stamp}-{input_digest[:16]}"


# ---------------------------------------------------------------------------
# registry access
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def row_sha256(row: List[str]) -> str:
    """The register row's canonical digest: sha256 of
    ``json.dumps(row, ensure_ascii=False, separators=(",", ":"))``."""
    return sha256_text(json.dumps(row, ensure_ascii=False, separators=(",", ":")))


def load_registry(path: str = DEFAULT_REGISTRY) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def registry_entries(registry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["operation_id"]: e for e in registry.get("entries", [])}


def utc_now() -> str:
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def parse_utc(utc: Any) -> Optional[datetime.datetime]:
    """The aware datetime of a ``UTC_RE`` stamp, or None when it is not one."""
    if not (isinstance(utc, str) and UTC_RE.match(utc)):
        return None
    base, _, frac = utc[:-1].partition(".")
    try:
        dt = datetime.datetime.strptime(base, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    dt = dt.replace(tzinfo=datetime.timezone.utc)
    if frac:
        dt += datetime.timedelta(microseconds=int(frac.ljust(6, "0")))
    return dt


def repository_commit(root: str = ROOT) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


# ---------------------------------------------------------------------------
# building a trial record
# ---------------------------------------------------------------------------

class TrialRefused(Exception):
    """The trial could not be recorded honestly (unknown op, spec not bound to
    the cell, destination exists or is governed)."""


def build_trial(op_id: str, registry: Dict[str, Any], registry_sha256: str,
                utc: Optional[str] = None, spec: Optional[Dict[str, Any]] = None,
                root: str = ROOT) -> Dict[str, Any]:
    """Assemble one ``q0.operation.trial/v1`` record for ``op_id``.

    ``spec`` overrides the built-in identity (tests use a deliberately wrong
    one to prove the result is IDENTITY_NOT_REPRODUCED and never success).
    """
    entries = registry_entries(registry)
    if op_id not in entries:
        raise TrialRefused(f"{op_id} is not in the registry")
    entry = entries[op_id]
    cells = entry["cells"]
    git_side = entry["git_side"]
    utc = utc or utc_now()
    when = parse_utc(utc)
    if when is None:
        raise TrialRefused(f"bad utc {utc!r}")
    now = datetime.datetime.now(datetime.timezone.utc)
    if when > now:
        raise TrialRefused(f"utc {utc} is later than the clock ({utc_now()}); a record's utc "
                           "is the time the runner ran, and a run in the future did not happen")

    checkable = git_side.get("trial_kind") == "EXACT_IDENTITY"
    if checkable:
        spec = copy.deepcopy(spec if spec is not None else IDENTITIES.get(op_id))
        if spec is None:
            raise TrialRefused(f"{op_id} is marked EXACT_IDENTITY but has no identity spec")
        if not spec_bound_to_cell(spec, cells["Action / output"], cells["Required scope"]):
            raise TrialRefused(
                f"{op_id}: the identity spec's displayed text is not in the register's "
                "Action / output cell (or its scope_displayed text not in Required scope); "
                "refusing to run an identity the register does not display")
        steps, failure = evaluate(spec)
        verdict = verdict_from(steps, failure)
        failure_charge = 0 if failure is None else 1
    else:
        spec, steps, failure = None, [], None
        verdict = "NOT_RUN"
        failure_charge = 0

    tid = trial_id(op_id, entry["row_sha256"], registry_sha256, spec, utc)

    run_evidence: Dict[str, Any] = {
        "record_schema": SCHEMA,
        "trial_kind": git_side.get("trial_kind"),
        "identity_displayed": (spec or {}).get("displayed"),
        "reproduces": (spec or {}).get("reproduces"),
        "identity_spec": spec,
        "arithmetic": ARITHMETIC,
        "steps": steps,
        "checked_equalities": len(steps),
        "all_equal": bool(steps) and all(s["equal"] is True for s in steps),
        "failure": failure,
        "utc": utc,
        "reason_not_run": None if checkable else git_side.get("why"),
    }
    notes = (
        "Do not infer (register, verbatim): " + cells["Do not infer"]
        + " | Trial kind: " + str(git_side.get("trial_kind"))
        + " | A trial is a record of a computation, not evidence; it moves no status."
        + " | Register cells, verbatim: Utility: " + cells["Utility"]
        + "; Novelty: " + cells["Novelty"] + "."
    )
    dne = required_does_not_establish(op_id, verdict)

    record: Dict[str, Any] = {
        "Trial ID": tid,
        "Problem ID / SHA-256": entry["row_sha256"],
        "Split": "EXPOSED_DEVELOPMENT" if checkable else "NOT_APPLICABLE",
        "Operation ID": op_id,
        "Library / catalog SHA-256": registry_sha256,
        "Arm": "NONE_SINGLE_EXACT_REPLAY" if checkable else "NOT_APPLICABLE",
        "Budget / cost unit": BUDGET,
        "Seed / environment": {
            "seed": SEED,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.system(),
            "repository_commit_at_run": repository_commit(root),
            "runner": RUNNER_PATH,
        },
        "Verified result": verdict,
        "Search cost": NOT_MEASURED,
        "Retrieval cost": NOT_MEASURED,
        "Verification cost": len(steps) if checkable else NOT_MEASURED,
        "Acquisition cost": NOT_MEASURED,
        "Maintenance cost": NOT_MEASURED,
        "Timeout / failure charge": failure_charge,
        "Run evidence": run_evidence,
        "Scope / constraints": cells["Required scope"],
        "Notes": notes,
        "does_not_establish": dne,
        "authority": AUTHORITY,
    }
    assert tuple(record.keys()) == TRIAL_COLUMNS + REPOSITORY_FIELDS
    assert tuple(record["Seed / environment"].keys()) == ENVIRONMENT_KEYS
    # The writer holds itself to the same rule as the checker: nothing this
    # repository authors about a trial may read as a status or a usefulness
    # claim. A record that would is refused, never written.
    authored = authored_fields(record, cells)
    status = status_word_hits(authored)
    if status:
        raise TrialRefused(f"{op_id}: the record's repository-authored text uses status "
                           f"words {status}; refusing to write a trial that claims a status")
    useful = usefulness_hits_in(authored)
    if useful:
        raise TrialRefused(f"{op_id}: the record's repository-authored text speaks of "
                           f"usefulness ({useful[0]}); Utility and Novelty are the register's "
                           "words")
    return record


GOVERNED_PREFIXES = ("registers", "claims", "drive", "governance", "reviews", "quarantine")


def write_trial(record: Dict[str, Any], out_dir: str, root: str = ROOT) -> str:
    """Append-only write. Refuses an existing path and a governed destination."""
    rel = os.path.relpath(os.path.abspath(out_dir), root)
    if not rel.startswith(".."):
        head = rel.split(os.sep)[0]
        if head in GOVERNED_PREFIXES:
            raise TrialRefused(f"refusing to write a trial under governed path {rel!r}")
    path = os.path.join(out_dir, record["Trial ID"] + ".json")
    if os.path.exists(path):
        raise TrialRefused(f"{path} already exists; trials are append-only and are never "
                           "rewritten")
    os.makedirs(out_dir, exist_ok=True)
    text = json.dumps(record, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(text)
    except FileExistsError as exc:
        raise TrialRefused(f"{path} already exists (concurrent writer)") from exc
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", default=DEFAULT_REGISTRY)
    ap.add_argument("--op", help="operation id, e.g. OP02")
    ap.add_argument("--all", action="store_true", help="every EXACT_IDENTITY operation")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", help="directory the trial record is appended to")
    ap.add_argument("--utc", help="TESTS ONLY: fix the timestamp (YYYY-MM-DDTHH:MM:SS[.ffffff]Z). "
                                  "A committed record's utc is the time the runner ran; a "
                                  "stamp later than the clock is refused")
    args = ap.parse_args(argv)

    try:
        registry = load_registry(args.registry)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"trial: cannot read registry {args.registry}: {exc}")
        return 2
    entries = registry_entries(registry)

    if args.list:
        for op_id, e in entries.items():
            g = e["git_side"]
            print(f"{op_id}  {g['trial_kind']:<28} {e['cells']['Operation']}")
        n = sum(1 for e in entries.values() if e["git_side"]["trial_kind"] == "EXACT_IDENTITY")
        print(f"registry={len(entries)} checkable={n}  (a trial is a record, not evidence)")
        return 0

    if not args.out or not (args.op or args.all):
        ap.error("--out and one of --op/--all are required (or --list)")
    if args.op and not OP_ID_RE.match(args.op):
        ap.error(f"bad operation id {args.op!r}")

    ops = [args.op] if args.op else [
        k for k, e in entries.items() if e["git_side"]["trial_kind"] == "EXACT_IDENTITY"]
    registry_sha = sha256_file(args.registry)
    worst = 0
    for op_id in ops:
        try:
            rec = build_trial(op_id, registry, registry_sha, utc=args.utc)
            path = write_trial(rec, args.out)
        except TrialRefused as exc:
            print(f"trial: REFUSED {op_id}: {exc}")
            worst = max(worst, 2)
            continue
        print(f"trial: {op_id} {rec['Verified result']} checked_equalities="
              f"{rec['Run evidence']['checked_equalities']} -> {os.path.relpath(path)}  "
              "(a record of a computation, not evidence)")
        if rec["Verified result"] == "IDENTITY_NOT_REPRODUCED":
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
