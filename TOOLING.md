# CBZ tooling

Checkpoint registry, **not G11 final admission**. Exact final source/run evidence remains required.

| Tool | Usage/purpose | Actual verification/boundary | Owner/criterion |
|---|---|---|---|
| Math formal/gate.py + gate_helper.py | `LEAN_NUM_THREADS=2 python3.11 -B -S formal/gate.py --execute` | 244 tests/mode at recorded cut; complete core gate passed at8151 with compiled/type/axiom/adversarial controls. No global final qualification | none;6,7 |
| Main tools/formal_gate_check.py | `python3.11 -B -S tools/formal_gate_check.py --run-lean` | Fresh build/recheck/axioms/controls passed; attempt-binding repair separate, old receipts never retroactively upgraded | none;6,7 |
| Main tools/cbz_inventory.py, mission addition | `generate --root ROOT --output EXTERNAL.json`; `verify --root ROOT --manifest EXTERNAL.json`; optional `--classifications FILE` | 75/75 tests and nine intended source-mutant kills per mode. RF312 exact effective-map/blob binding; explicit replay source/target/axiom/raw-artifact binding. Actual 94/207 component roundtrips passed. No semantic/kernel authenticity certificate | none—self-contained; 1,3,8,11 |
| Main tools/public_shop_check.py / museum_check.py | Validate pinned fixture bytes and execute Node fixture suites | Observed4/4 and45/45; broad prepared Node763/763 zero skips | none;5,11 |
| Math downstream run_validation.py | Existing documented validation campaign |71 methods and24 semantic mutations per mode observed, scoped software gate only | none;7 |
| Git bundle/SHA-lease cleanup | `git bundle verify PATH`; guarded atomic ref deletion | Ancestor checks, excluded PR/protected/workflow refs, actual absence readback; no mathematical certification | none;1,2 |
| Bounded proof-closure replay | [Exact tool commands](cbz/evidence/kernel/additional/replay-tool/README.md), [additional package replay](cbz/evidence/kernel/additional/HOW_TO_REPLAY.md), [determinant replay](cbz/evidence/kernel/determinant/HOW_TO_REPLAY.md) | 31 tests/three Python mutants per mode; actual Lean positive/six adverse/six auditor mutants in both modes. Official empty trust-zero kernel; supplied text does not authenticate execution | none;3,6,9,11 |
| PR/citation evidence | [Exact ordinary/optimized test and mutation commands](cbz/evidence/stale-pr-dispositions/README.md) | 35+9 tests and49 intended source/harness kills per mode. Actual network fetch and26-head historical regeneration. Advisory recorded facts, no current approval or mathematical certification | none;2,3,10,11,12 |

New census self-tests:

```bash
python3.11 -B -S -m unittest discover -s tests -p test_cbz_inventory.py -v
python3.11 -B -O -S -m unittest discover -s tests -p test_cbz_inventory.py -v
```

Inventory output lives outside scanned roots. Immutable ancestor bindings retain exact Git blobs; current proof bytes must remain identical. Explicit historical quarantine bindings cannot discharge CLOSED proofs. Private evidence is not copied into public manifests. A named missing tool is not a finished tool.
