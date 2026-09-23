# Inventable JETMOD probes (REFUSED only)

Fail-closed inventable attempts against **named walls only** from the sibling
sweep CLOSED EMPTY. Do not invent new walls.

```bash
python3 docs/math_status_probes/inventable_jetmod_probes.py
python3 tools/math_status_check.py   # also validates these receipts
```

`discharges_OBL_H5_JETMOD` stays false. `lemma_closed` stays false.
OBL-H5-JETMOD stays OPEN. Green ≠ discharge.
