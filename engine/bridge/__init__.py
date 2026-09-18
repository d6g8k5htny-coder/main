"""The repository-side half of a PROPOSED, NOT DEPLOYED execution contract.

``engine/bridge/`` holds two record shapes and their validators: the work
order (``work_order.py``, ``q0.bridge.work_order/v1``) that would record the
authorization of a bounded task, and the run receipt (``run_receipt.py``,
``q0.bridge.run_receipt/v1``) that would record what a run did. The contract
they implement is mirrored under
``drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/`` with the disposition
"PROPOSED EXECUTION CONTRACT / NOT DEPLOYED", and that disposition is not
changed by the existence of this code.

WHAT THIS PACKAGE DOES NOT ESTABLISH
------------------------------------
Nothing mathematical, and nothing operational either. A valid work order is a
well-formed authorization record, not an authorization: the authority lives
in the Drive record it names. A stored receipt is a record that something
ran, not evidence that it ran correctly and not a verdict on anything. No
function here enforces a branch protection, holds a credential, accepts a
result on the owner's behalf, or moves a claim, premise or obligation. The
five validity premises of Theorem D1 v2.2(2) are OPEN, ``D3-LEMMA-RN-UNIF`` is
not closed, and no record this package can validate or write says otherwise.
"""
