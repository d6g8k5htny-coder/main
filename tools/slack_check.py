#!/usr/bin/env python3
"""CI entry for the bound-slack registry in ``research/slack``.

Re-runs every registry invariant over the full record set and refuses a report
that uses forbidden promotion or correctness language. Exit 0 when clean,
1 otherwise, in the style of the other checkers under ``tools/``.

What a pass means: the records are internally consistent — every value is a
certified interval, every ratio is computed exactly, every ``known_unsound``
rests on a witness the module re-computed, and severity was classified from the
ratio's lower endpoint. What a pass does NOT mean: that any bound is tight, or
loose, or correct. A slack ratio measures a bound's utility, never its
correctness, and this checker verifies the registry's bookkeeping, not any
mathematics.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.slack import registry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    problems = list(registry.audit_registry())
    problems += [
        f"forbidden word in report: {w!r}" for w in registry.forbidden_words_in_report()
    ]
    records = registry.sorted_records()
    for p in problems:
        print(f"  PROBLEM {p}")
    unsound = sum(1 for r in records if getattr(r, "known_unsound", False))
    print(
        f"slack_check: records={len(records)} known_unsound={unsound} "
        f"problems={len(problems)}"
    )
    print(
        "slack_check: bookkeeping only. A slack ratio measures utility, never "
        "correctness; nothing here repairs, refutes, promotes or closes anything."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
