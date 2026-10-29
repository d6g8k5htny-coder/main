# CBZ tooling

Checkpoint registry, **not G11 final admission**. Exact final source/run evidence remains required.

| Tool | Usage/purpose | Actual verification/boundary | Owner/criterion |
|---|---|---|---|
| Math formal/gate.py + gate_helper.py | `LEAN_NUM_THREADS=2 python3.11 -B -S formal/gate.py --execute` |244 tests/mode; compiled/type/axiom/adversarial audits; complete final sequence still running | none;6,7 |
| Main tools/formal_gate_check.py | `python3.11 -B -S tools/formal_gate_check.py --run-lean` | Fresh build/recheck/axioms/controls passed; attempt-binding repair separate, old receipts never retroactively upgraded | none;6,7 |
| Main tools/cbz_inventory.py, mission addition | `generate --root ROOT --output EXTERNAL.json`; `verify --root ROOT --manifest EXTERNAL.json`; optional `--classifications FILE` |50/50 tests per mode, both source mutants killed; independent review repaired/re-reviewed; actual12-repo capture/verify13,654 entries. Complete byte/binding census only, no semantic/kernel authenticity certificate | none—self-contained;1,3,8,11 |
| Main tools/public_shop_check.py / museum_check.py | Validate pinned fixture bytes and execute Node fixture suites | Observed4/4 and45/45; broad prepared Node763/763 zero skips | none;5,11 |
| Math downstream run_validation.py | Existing documented validation campaign |71 methods and24 semantic mutations per mode observed, scoped software gate only | none;7 |
| Git bundle/SHA-lease cleanup | `git bundle verify PATH`; guarded atomic ref deletion | Ancestor checks, excluded PR/protected/workflow refs, actual absence readback; no mathematical certification | none;1,2 |

New census self-tests:

```bash
python3.11 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
```

Inventory output lives outside scanned roots. Immutable ancestor bindings retain exact Git blobs; current proof bytes must remain identical. Explicit historical quarantine bindings cannot discharge CLOSED proofs. Private evidence is not copied into public manifests. A named missing tool is not a finished tool.
