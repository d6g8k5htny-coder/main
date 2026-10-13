#!/usr/bin/env bash
# Integration of the main-side deliverable: run after all checks pass. Usage: bash integrate_main.sh
set -euo pipefail
S=<SCRATCH>
export PATH="$HOME/.elan/bin:$PATH"
cd $S/lvs_main
git status --short
# gates
python3 tools/formal_gate_check.py
python3 -B -S -m unittest tests.test_formal_gate 2>&1 | tail -2
python3 -B -O -S -m unittest tests.test_formal_gate 2>&1 | tail -2
python3 -B -S -m unittest discover -s tests -p 'test_formal*.py' 2>&1 | tail -2
python3 -B -S -m unittest discover -s tests -p 'test_required*.py' 2>&1 | tail -2
git diff --quiet -- tools/ && echo "tools/ unchanged"
git add formal/UniversalLaw/P15 formal/sources/p15_v1 formal/manifest.json formal/ALIGNMENT.md formal/SCOPE.md formal/README.md formal/REVIEW_LANE.md formal/reviews/TEMPLATE_alignment_review.json formal/UniversalLaw.lean tests/test_formal_gate.py docs/LEAN_VERIFICATION_SPECIFICATION.md docs/FORMAL_VERIFICATION.md docs/RESEARCH_INDEX.md
git status --short
