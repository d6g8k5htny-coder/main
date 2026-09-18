#!/usr/bin/env python3
"""Run a lane's registered computation and record what happened.

    python3 engine/run.py --list          # lanes, and which have a runnable entry
    python3 engine/run.py --lane A5       # run one lane, write one receipt
    python3 engine/run.py --all           # run every registered lane
    python3 engine/run.py --lane A1 --dry-run   # resolve and print; write nothing

A RUN RECORDS, IT DOES NOT DECIDE
---------------------------------
This program reads ``engine/lanes/*.json``, calls repository code, and writes a
receipt under ``engine/receipts/``. That is the whole of its authority.

* It **cannot** write to ``engine/lanes/`` or to ``claims/graph.json``, and
  that is structural rather than a convention. The only writer this module
  imports is :func:`engine.receipt.write_receipt`, which refuses any
  destination outside the receipts root and refuses outright to write under
  ``engine/lanes/``, ``claims/``, ``registers/`` or ``drive/`` (see
  ``engine.receipt.FORBIDDEN_WRITE_ROOTS``). No other write path exists here:
  this module opens no file for writing, creates no directory, deletes
  nothing, renames nothing and imports no writer for a governed path.
  ``tests/test_receipts.py`` asserts all of that against the parsed syntax
  tree, so adding one would fail the build.
* A receipt is **not evidence**. A green run is a run, not a proof. Nothing
  this program produces promotes, closes, discharges or reclassifies anything.
  ``OBL-H5-JETMOD``, ``OBL-H5-ZBAND`` (hi side), ``OBL-H5-REMOTE-THRESHOLD``,
  ``OBL-D1-PROMOTE`` and **both** Pieces of ``D3-LEMMA-RN-UNIF`` are OPEN, and
  they stay OPEN whatever any number below turns out to be. Only an operator
  decision under ``governance/`` can change a mathematical status.
* The registered entry points call **reference** machinery. Lane A1's kernel is
  a reference Gaussian, not the program's ``kplane``; lane A5's integrand is a
  reference Gaussian, not ``kappa_far``. Neither run touches a jet, a band
  endpoint, a rung or a cell of the program's own cover. The receipts say so
  in their ``does_not_establish`` field, which the writer will not let be
  empty.

THE REGISTRY
------------
:data:`TASKS` maps a lane key to the repository code run for it. It lives here,
not in ``engine/lanes/*.json``, for two reasons: the lane files are transcribed
status records owned elsewhere in this repository, and a runner that edited
them would be exactly the thing the paragraph above says it cannot be.

A lane with no entry in :data:`TASKS` is reported by ``--list`` as having no
runnable entry point **in this repository**. That is a statement about code,
like a lane's ``repo_state``. It is not a mathematical status and no verdict
may be read from it. Likewise, a registered module that fails to import -- a
sibling agent may still be landing it -- produces an ``UNAVAILABLE`` receipt
rather than a crash.

Standard library only. Python 3.11.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:  # allow `python3 engine/run.py` from anywhere
    sys.path.insert(0, ROOT)

from engine.receipt import (  # noqa: E402  (after the sys.path fix-up)
    OUTCOME_DRY_RUN, OUTCOME_FAILED, OUTCOME_RAN, OUTCOME_UNAVAILABLE,
    NumericResult, Receipt, ReceiptRejected, write_receipt,
)

#: Read-only inputs. This module opens these for reading and never for writing;
#: ``engine.receipt.write_receipt`` refuses to write under either of them even
#: if asked. Named here so the prohibition is visible at the top of the file.
LANES_DIR = os.path.join(ROOT, "engine", "lanes")
CLAIM_GRAPH = os.path.join(ROOT, "claims", "graph.json")

NO_AUTHORITY = [
    "A receipt is a record that a computation ran. It is NOT evidence, NOT a proof,",
    "NOT a certificate, NOT a review verdict and NOT a status. A green run is a run.",
    "This program promotes, closes, discharges and reclassifies nothing: OBL-H5-JETMOD,",
    "OBL-H5-ZBAND (hi side), OBL-H5-REMOTE-THRESHOLD, OBL-D1-PROMOTE and both Pieces of",
    "D3-LEMMA-RN-UNIF are OPEN and stay OPEN. Only an operator decision under governance/",
    "can change a mathematical status, and running this program is not one.",
]


class TaskUnavailable(Exception):
    """The registered code is not importable here (another agent may be landing it)."""


@dataclass(frozen=True)
class LaneTask:
    """One registered computation: what to run for a lane, and its caveat.

    ``call`` returns ``(results, notes)``. It raises :class:`TaskUnavailable`
    when the repository module it drives cannot be imported, which is recorded
    as an ``UNAVAILABLE`` receipt rather than treated as a failure.
    """

    lane: str
    module: str
    entry_point: str
    arguments: Dict[str, Any]
    summary: str
    does_not_establish: str
    call: Callable[[], Tuple[List[NumericResult], List[str]]]


# ---------------------------------------------------------------------------
# reading the lane files (read-only)
# ---------------------------------------------------------------------------

def read_json(path: str) -> Any:
    """Read one JSON file. Read-only: this module never opens a file to write."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_lanes(lanes_dir: str = LANES_DIR) -> Dict[str, Dict[str, Any]]:
    """Every lane file, keyed by lane key. Missing directory yields ``{}``."""
    lanes: Dict[str, Dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(lanes_dir, "*.json"))):
        try:
            lane = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue  # a sibling agent may be mid-write; skip, do not crash
        key = lane.get("key") or os.path.splitext(os.path.basename(path))[0]
        lanes[str(key)] = lane
    return lanes


def lane_does_not_establish(lane: Mapping[str, Any]) -> str:
    """The lane file's own ``does_not_establish``, transcribed, or ``""``."""
    v = lane.get("does_not_establish")
    return v.strip() if isinstance(v, str) else ""


# ---------------------------------------------------------------------------
# the registered computations
# ---------------------------------------------------------------------------

_A1_ARGS: Dict[str, Any] = {
    "kappa": "1/8",
    "prec": 40,
    "line": "live v3 clean",
    "reference_kernel": "gaussian-reference exp(-|z|^2/2)",
    "reference_band": ["1/40", "1/20"],
    "reference_power": 3,
    "n_trunc": 3,
}

_A1_DNE = (
    "This run does not discharge, reduce or reclassify OBL-H5-JETMOD, which "
    "stays OPEN (display only), and it bears on OBL-H5-ZBAND (hi side), "
    "OBL-H5-REMOTE-THRESHOLD and OBL-D1-PROMOTE (chart side) not at all. The "
    "band enclosure below is computed for a REFERENCE Gaussian kernel over a "
    "REFERENCE band with a PLACEHOLDER axial displacement map: the program's "
    "own kplane and its certified decay envelope, the 24 jets with their "
    "powers p_J, and the actual band endpoints r_k are none of them bound in "
    "this repository, so no jet of this program is enclosed here. The "
    "implied-modulus numbers are an observation under a stated reading of a "
    "DISPLAY (a dense certified sampling plus an explicit kappa = 1/8 fit); "
    "they refute no source's modulus statement. The three published I_hi/r^3 "
    "values remain POINT certifications and no arithmetic on them makes a "
    "band enclosure."
)


def _task_a1() -> Tuple[List[NumericResult], List[str]]:
    """Lane A1: the published-point arithmetic and a REFERENCE band enclosure.

    Two things run here, and neither is a jet of this program.

    1. ``research.bands.ladder`` on the three published ``I_hi/r^3`` point
       certifications: the same-``r`` spread between the frozen-v1 and live-v3
       lines, and the constant each adjacent band forces under the stated
       reading of the displayed ``kappa = 1/8`` fit. Exact rationals and
       certified enclosures throughout.
    2. ``research.bands.lattice`` executing the source's own proof step shape
       -- the periodized lattice sum evaluated with ``r`` as an interval over a
       band, truncated sum plus a proved uniform tail bound -- on a REFERENCE
       Gaussian kernel. Finite, per band, no fitted exponent, no sampling.
    """
    try:
        from research.bands import (  # type: ignore
            gaussian_reference, implied_modulus_ratio, implied_modulus_table,
            normalized_band_enclosure, same_r_version_spread,
        )
        from research.interval import Interval  # type: ignore
    except ImportError as exc:
        raise TaskUnavailable(f"research.bands / research.interval: {exc}") from exc

    results: List[NumericResult] = []
    notes: List[str] = []

    kappa = Fraction(_A1_ARGS["kappa"])
    prec = int(_A1_ARGS["prec"])

    spread = same_r_version_spread()
    results.append(NumericResult.from_fraction(
        "same_r_version_spread_at_r_0.05", spread["spread"],
        "frozen v1 minus live v3 clean, at ONE separation. Published point "
        "certifications disagree across engine versions at fixed r; this "
        "adjudicates between them in no way."))
    results.append(NumericResult.from_fraction(
        "same_r_version_spread_relative_percent", spread["relative_percent"],
        "the same spread as a percentage of the live v3 clean value."))

    table = implied_modulus_table(kappa=kappa, prec=prec)
    for m in table:
        results.append(NumericResult.from_interval(
            f"implied_modulus_constant[{m.band.name}]", m.constant,
            "certified enclosure of |Delta| / delta^kappa for this band, under "
            "the stated reading of the DISPLAYED kappa = 1/8 fit. A lower bound "
            "on any admissible C under that reading only."))
    if table:
        notes.append("implied-modulus assumption: " + table[0].assumption)

    ratio = implied_modulus_ratio(kappa=kappa, prec=prec)
    results.append(NumericResult.from_interval(
        "implied_modulus_ratio_wide_over_narrow", ratio["ratio"],
        "how far apart the constants forced by the two adjacent bands are. "
        "It shows why three point certifications cannot substitute for a band "
        "enclosure. It does not show the displayed modulus is wrong."))

    band = Interval(Fraction(_A1_ARGS["reference_band"][0]),
                    Fraction(_A1_ARGS["reference_band"][1]))
    normalized, breakdown = normalized_band_enclosure(
        gaussian_reference(), band, int(_A1_ARGS["reference_power"]),
        n_trunc=int(_A1_ARGS["n_trunc"]), prec=prec)
    results.append(NumericResult.from_interval(
        "REFERENCE_band_enclosure_S(B)", breakdown.total,
        "certified enclosure of the periodized REFERENCE Gaussian sum, valid "
        "for every r in the band at once. Not a jet of this program."))
    results.append(NumericResult.from_fraction(
        "REFERENCE_band_tail_bound", Fraction(breakdown.tail),
        "proved bound on the omitted image-lattice points, uniform over the "
        "displacement box. OBL-H5-JETMOD asks for exactly this uniformity, for "
        "the program's own lattice, which is not bound here."))
    results.append(NumericResult.from_interval(
        "REFERENCE_band_enclosure_S(B)_over_r^3", normalized,
        "the shape of the obligation's content line J(B)/r^{p_J}, for a "
        "REFERENCE kernel and an arbitrary power. No jet J and no power p_J of "
        "this program is bound here."))
    results.append(NumericResult.from_int(
        "REFERENCE_band_kernel_evaluations", breakdown.terms,
        "finite, per band: (2*n_trunc+1)^2 certified kernel evaluations plus "
        "one closed-form tail bound. Never a fitted exponent."))

    if not breakdown.certified:
        notes.append("NON-CERTIFYING: the plugged-in kernel is not certified.")
    notes.extend(breakdown.caveats)
    notes.append(
        "The displacement map used is research.bands.lattice.axial_displacement, "
        "explicitly a PLACEHOLDER for the program's own pin geometry.")
    return results, notes


_A5_ARGS: Dict[str, Any] = {
    "region": "rn5_annulus_polar(split='radius')",
    "region_domain": "0.1 <= |y| <= 5",
    "integrand": "REFERENCE:radial_gaussian",
    "tol": "1/10",
    "max_depth": 22,
    "prec": 40,
}

_A5_DNE = (
    "This run does not close Piece 1 or Piece 2 of D3-LEMMA-RN-UNIF; both stay "
    "OPEN, and the sources' own receipts still read lemma_closed: false. It "
    "does not supply the annulus Riemann-sum driver for the program's own "
    "integrand, does not certify a single cell of the program's cover, does "
    "not reassemble the remote budget, and turns neither RN5's ten boxes nor "
    "any smoke test into a coverage certificate. The integrand covered here is "
    "a REFERENCE Gaussian chosen because it can be certified end to end: it is "
    "not kappa_far, not the corrected RN5 envelope and not any quantity "
    "appearing in any claim. The RN5 annulus region and the T4 polar cover "
    "d in [5,17] are different regions serving different purposes and are not "
    "merged. The moment-envelope numbers reproduce a known defect; they "
    "re-prove nothing in RN3 and discharge no premise."
)


def _task_a5() -> Tuple[List[NumericResult], List[str]]:
    """Lane A5: the cover driver on the RN5 annulus, plus the RN5 counterexample.

    The driver returns a *ledger*, not a number; getting a number out of it
    means calling ``total()``, which refuses while any cell is PENDING. That
    ordering is the point, and this task reports the refusal as a result when
    it happens rather than reporting a partial cover as a total.
    """
    try:
        from research.cover import (  # type: ignore
            ACCEPTED, DriverConfig, PendingCellsError, RadialGaussianReference,
            radial_gaussian_closed_form, rn5_annulus_polar, run,
        )
        from research.rn.moment_envelope import (  # type: ignore
            counterexample, envelope_correct_pow4, envelope_defective_pow4,
        )
    except ImportError as exc:
        raise TaskUnavailable(f"research.cover / research.rn: {exc}") from exc

    results: List[NumericResult] = []
    notes: List[str] = []

    region = rn5_annulus_polar(split="radius")
    config = DriverConfig(tol=Fraction(_A5_ARGS["tol"]),
                          max_depth=int(_A5_ARGS["max_depth"]),
                          prec=int(_A5_ARGS["prec"]))
    ledger = run(region, RadialGaussianReference(), config)

    results.append(NumericResult.from_int(
        "cover_cells_total", len(ledger.records),
        "cells visited by the adaptive driver on the REFERENCE integrand."))
    results.append(NumericResult.from_int(
        "cover_cells_pending", len(ledger.pending()),
        "RN5's recipe requires verifying that no cell remains pending before "
        "any total is reported. A non-zero count here means the cover is "
        "incomplete and total() refuses."))
    results.append(NumericResult.from_int(
        "cover_cells_accepted", len(ledger.by_disposition(ACCEPTED)),
        "cells whose contribution was accepted."))
    results.append(NumericResult.from_int(
        "cover_cells_rejected", len(ledger.rejected()),
        "rejected cells are retained with their boundary-area bounds, as RN5's "
        "recipe requires."))

    try:
        total = ledger.total()
    except PendingCellsError as exc:
        notes.append(f"total() refused: {exc}")
        notes.append("A partial cover reported as a total is the exact failure "
                     "mode RN5 names. Nothing further is reported.")
        return results, notes

    results.append(NumericResult.from_interval(
        "REFERENCE_cover_total_enclosure", total.enclosure,
        "certified enclosure of the integral of the REFERENCE Gaussian over "
        "0.1 <= |y| <= 5. NOT kappa_far and NOT the corrected RN5 envelope."))
    results.append(NumericResult.from_interval(
        "REFERENCE_cover_area_accounted", total.area_accounted,
        "area the accepted cells account for; the region's exact area is the "
        "cross-check that the partition is a partition."))
    results.append(NumericResult.from_fraction(
        "REFERENCE_cover_area_rejected_bound", Fraction(total.area_rejected_bound),
        "upper bound on the area carried by rejected cells; exactly zero here "
        "because the annulus radii 1/10 and 5 are exact rationals."))
    closed = radial_gaussian_closed_form(region, 60)
    results.append(NumericResult.from_interval(
        "REFERENCE_cover_closed_form_crosscheck", closed,
        "an independent certified evaluation of the same REFERENCE integral. "
        "It must lie inside the driver's enclosure; it is a check on the code, "
        "not evidence for any claim."))
    notes.append(
        f"closed form inside the driver's enclosure: {closed in total.enclosure}")
    notes.append(f"total.certified={total.certified} "
                 f"covers_region={total.covers_region}")
    for c in total.caveats:
        notes.append(f"caveat: {c}")

    ce = counterexample()
    correct = envelope_correct_pow4(ce["EA4"], ce["EB4"], ce["EC2"])
    defective = envelope_defective_pow4(ce["EA4"], ce["EB4"], ce["EC4"])
    typed_lower = ce["typed_expectation_lower"]
    results.append(NumericResult.from_fraction(
        "rn5_typed_expectation_lower_bound", typed_lower,
        "exact lower bound on the typed expectation in the RN5 Gaussian "
        "counterexample."))
    results.append(NumericResult.from_fraction(
        "rn5_correct_hoelder_442_envelope_pow4", correct,
        "fourth power of the correct Hoelder(4,4,2) envelope."))
    results.append(NumericResult.from_fraction(
        "rn5_defective_envelope_pow4", defective,
        "fourth power of the defective expression, which substitutes a fourth "
        "determinant moment where a second is required."))
    results.append(NumericResult.from_int(
        "rn5_defective_envelope_is_violated", int(defective < typed_lower ** 4),
        "1 when the defective expression falls below the typed expectation it "
        "claims to bound, i.e. when it is not an upper bound at all. Exact "
        "rational comparison, no floats."))
    notes.append(
        "The RN5 counterexample is reproduced in exact rationals so the "
        "Hoelder(4,4,2) defect stays falsifiable in CI. The defective "
        "expression substitutes a fourth determinant moment where a second is "
        "required and can DECREASE the value, so it is not an upper bound at "
        "all. Recording that is not a repair of any frozen engine.")
    return results, notes


TASKS: Dict[str, LaneTask] = {
    "A1": LaneTask(
        lane="A1",
        module="research.bands",
        entry_point="engine.run._task_a1",
        arguments=_A1_ARGS,
        summary=("published-point arithmetic (exact) plus a REFERENCE-kernel "
                 "interval-r band enclosure with a uniform tail bound"),
        does_not_establish=_A1_DNE,
        call=_task_a1,
    ),
    "A5": LaneTask(
        lane="A5",
        module="research.cover",
        entry_point="engine.run._task_a5",
        arguments=_A5_ARGS,
        summary=("adaptive cover of the RN5 annulus 0.1 <= |y| <= 5 on a "
                 "REFERENCE integrand, with the accept/refine/reject ledger"),
        does_not_establish=_A5_DNE,
        call=_task_a5,
    ),
}


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def compose_does_not_establish(task: LaneTask, lane: Optional[Mapping[str, Any]]) -> str:
    """The task's caveat, preceded by the lane file's own transcribed one."""
    parts: List[str] = []
    if lane is not None:
        lane_dne = lane_does_not_establish(lane)
        if lane_dne:
            parts.append(f"From engine/lanes/{task.lane}.json: {lane_dne}")
    parts.append(f"From this run: {task.does_not_establish}")
    return " ".join(parts)


def run_lane(key: str, lanes: Mapping[str, Mapping[str, Any]],
             dry_run: bool = False) -> Receipt:
    """Run one lane's registered computation and return its receipt.

    Never raises for a failure inside the registered code: an exception becomes
    a ``FAILED`` receipt and an unimportable module becomes an ``UNAVAILABLE``
    one, because a record of a failed run is worth more than a traceback.
    """
    task = TASKS[key]
    lane = lanes.get(key)
    dne = compose_does_not_establish(task, lane)

    if dry_run:
        return Receipt.build(
            lane=task.lane, module=task.module, entry_point=task.entry_point,
            arguments=task.arguments, outcome=OUTCOME_DRY_RUN,
            runtime_seconds=0.0, does_not_establish=dne,
            notes=[f"DRY RUN: nothing was executed and no receipt was written. "
                   f"Would run {task.entry_point} -> {task.summary}."],
            root=ROOT)

    started = time.monotonic()
    outcome = OUTCOME_RAN
    error: Optional[str] = None
    results: List[NumericResult] = []
    notes: List[str] = []
    try:
        results, notes = task.call()
    except TaskUnavailable as exc:
        outcome = OUTCOME_UNAVAILABLE
        error = str(exc)
        notes = ["The registered module did not import in this working tree. "
                 "This is a statement about code present here, not a "
                 "mathematical status and not a verdict on the lane."]
    except Exception as exc:  # noqa: BLE001 - a failed run is still recorded
        outcome = OUTCOME_FAILED
        error = f"{type(exc).__name__}: {exc}"
        notes = [line.rstrip() for line in
                 traceback.format_exc(limit=12).splitlines()[-6:]]
    elapsed = time.monotonic() - started

    if outcome == OUTCOME_RAN and not results:
        outcome = OUTCOME_FAILED
        error = error or "the registered entry point produced no numeric results"

    return Receipt.build(
        lane=task.lane, module=task.module, entry_point=task.entry_point,
        arguments=task.arguments, outcome=outcome, runtime_seconds=elapsed,
        does_not_establish=dne, results=results, notes=notes, error=error,
        root=ROOT)


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

def format_receipt(receipt: Receipt, path: Optional[str]) -> str:
    """A short human summary of one receipt."""
    lines = [f"lane {receipt.lane}: {receipt.outcome}"
             f"  ({receipt.runtime_seconds}s, {len(receipt.results)} result(s))",
             f"  run      {receipt.module} :: {receipt.entry_point}",
             f"  commit   {receipt.commit[:12]}{' (dirty tree)' if receipt.dirty else ''}",
             f"  verdict  {receipt.verdict}"]
    if receipt.error:
        lines.append(f"  error    {receipt.error}")
    for r in receipt.results:
        if r.provenance == "certified_interval":
            shown = f"[{_short(r.lo)}, {_short(r.hi)}]"
        else:
            shown = _short(r.value)
        mark = "" if r.certifying else "  <-- NON-CERTIFYING, not a bound"
        lines.append(f"  {r.name}: {shown}  ({r.provenance}){mark}")
    lines.append(f"  receipt  {path if path else '(not written: dry run)'}")
    return "\n".join(lines)


def _short(value: Optional[str], places: int = 9) -> str:
    """A readable rendering of an exact rational. Display only."""
    if value is None:
        return "-"
    try:
        q = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return str(value)
    except OverflowError:  # pragma: no cover - a rational too large for float
        return str(value)
    if q.denominator == 1:
        return str(q.numerator)
    try:
        return f"{float(q):.{places}g}~"
    except OverflowError:  # pragma: no cover
        return str(q)


def list_lanes(lanes: Mapping[str, Mapping[str, Any]]) -> str:
    """Every lane, with whether this repository has a runnable entry for it."""
    rows = ["lane  registered  module / entry point",
            "----  ----------  ---------------------"]
    for key in sorted(lanes) or sorted(TASKS):
        t = TASKS.get(key)
        if t is None:
            rows.append(f"{key:<5} {'no':<11} (no runnable entry point in this repository)")
        else:
            rows.append(f"{key:<5} {'yes':<11} {t.module} :: {t.entry_point}")
    for key in sorted(set(TASKS) - set(lanes)):
        rows.append(f"{key:<5} {'orphan':<11} registered but engine/lanes/{key}.json is absent")
    rows.append("")
    rows.append("'registered' says whether code for the lane exists here. It is not a")
    rows.append("mathematical status and no verdict may be read from it.")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a lane's registered computation and write a receipt. "
                    "A run records; it does not decide.")
    ap.add_argument("--lane", metavar="KEY", help="run one lane, e.g. --lane A5")
    ap.add_argument("--all", action="store_true", help="run every registered lane")
    ap.add_argument("--list", action="store_true",
                    help="list the lanes and which have a runnable entry point")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print; execute nothing and write nothing")
    ap.add_argument("--receipts-dir", metavar="DIR", default=None,
                    help="where receipts are written (default engine/receipts/)")
    ap.add_argument("--json", action="store_true", help="emit receipts as JSON")
    args = ap.parse_args(argv)

    lanes = load_lanes()

    if args.list:
        print(list_lanes(lanes))
        return 0

    if args.lane and args.all:
        print("--lane and --all are mutually exclusive", file=sys.stderr)
        return 2
    if args.lane:
        if args.lane not in TASKS:
            known = ", ".join(sorted(TASKS)) or "(none)"
            print(f"lane {args.lane!r} has no registered entry point in this "
                  f"repository. Registered: {known}. Use --list.", file=sys.stderr)
            return 2
        keys = [args.lane]
    elif args.all:
        keys = sorted(TASKS)
    else:
        ap.print_help()
        return 2

    receipts: List[Tuple[Receipt, Optional[str]]] = []
    failures = 0
    for key in keys:
        receipt = run_lane(key, lanes, dry_run=args.dry_run)
        path: Optional[str] = None
        if not args.dry_run:
            try:
                path = write_receipt(receipt, args.receipts_dir)
            except ReceiptRejected as exc:
                print(f"lane {key}: the writer REFUSED the receipt: {exc}",
                      file=sys.stderr)
                failures += 1
                continue
        receipts.append((receipt, path))
        if receipt.outcome == OUTCOME_FAILED:
            failures += 1

    if args.json:
        print(json.dumps([r.to_dict() for r, _ in receipts], indent=2,
                         ensure_ascii=False, sort_keys=True))
    else:
        for receipt, path in receipts:
            print(format_receipt(receipt, path))
            print()
        for line in NO_AUTHORITY:
            print(line)

    unavailable = sum(1 for r, _ in receipts if r.outcome == OUTCOME_UNAVAILABLE)
    print(f"\nlanes={len(keys)} written={sum(1 for _, p in receipts if p)} "
          f"failed={failures} unavailable={unavailable}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
