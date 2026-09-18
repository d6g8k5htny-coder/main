## What this changes

<!-- One or two sentences. Name the exact object(s). -->

## Type of change

- [ ] Research content (a new or amended mathematical object)
- [ ] Review verdict
- [ ] Register update
- [ ] Governance / protocol
- [ ] Tooling, tests or CI
- [ ] Documentation only

## Status discipline

- [ ] No claim status was promoted without the exact predicate that licenses it
- [ ] Frozen bodies were not edited in place; any change is a numbered successor
- [ ] `claims/graph.json` updated if a claim or premise status moved, with `source` cited
- [ ] `registers/json/work_events.json` appended (never rewritten) if this is a claim or publication
- [ ] Nothing composes the 2D track with the 3D lifetime track
- [ ] Nothing moves the prize track into the q0 dependency graph

## If this is a review

- [ ] Exact object correctness verdict recorded
- [ ] Scope / dependency verdict recorded
- [ ] Reviewer authorship and source exposure disclosed
- [ ] Organizational independence recorded **separately** (same provider = zero credit)

## Verification

```
python3 tools/registers_import.py --check
python3 tools/registers_check.py
python3 tools/claims_check.py
python3 tools/verify_manifests.py
python3 -m pytest -q
```

- [ ] All of the above pass locally
