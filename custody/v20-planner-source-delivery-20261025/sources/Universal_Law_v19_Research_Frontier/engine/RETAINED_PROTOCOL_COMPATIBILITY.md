# Retained diagnostic protocol compatibility

The Final Integrated engine runs the v18 `research-flow-worker-1` diagnostic
runtime. Its retained v17 change-cycle receipts bind `UL-worker-v17.1`, version
17.1, and exact original checker/source hashes. These are distinct protocols;
changing a receipt's version or weakening current worker identity checks would
invalidate its custody.

`verify_example.py` therefore uses `retained_worker_compat.py` for archival
verification only. Current-protocol receipts go to the current verifier. A v17
receipt goes to the original verifier from the bundled sealed archive at
`../history/sealed/Universal_Law_Mathematical_Conductor_v17.zip`, whose fixed
SHA256 is `0deaa4e91cd5cf011236592cc5d91237b52a830e9f03b193227d4bac95ab9814`.
The archive's bytes are checked before bounded safe extraction into a private
temporary directory. A fixed isolated Python child runs the original verifier,
which checks receipt, job, captured dataset, notes, source and artifact hashes;
when replay is requested it executes the original resource-bounded diagnostic
and compares its exact result. Recorded command strings are never executed.
Unknown protocols, altered archives/artifacts and certificate claims fail closed.
There is no fallback between protocols and no legacy scheduling path in the
live coordinator.

All old receipts, journals and canonical mathematics remain byte-identical.
The offline `conductor.html` is a derived view rebuilt from the current
Coordinator snapshot and current template. The derivative `manifest.json` and
`SHA256SUMS` describe the actual extended runtime and compatibility code; they
exclude two ephemeral historical lock files that the final archive omits.
Original manifest/lock identities remain inside the unchanged sealed v17 ZIP.
Twelve omitted original empty log/lock files were restored from that ZIP before
updating the derivative manifest; no receipt was regenerated.

`python -B verify.py --skip-tests` checks retained receipt integrity without
re-executing diagnostics. `python -B verify_example.py` replays all five original
diagnostics. Neither a passing old nor a passing current diagnostic certifies a
mathematical theorem or clears independent review obligations.
