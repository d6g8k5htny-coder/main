# RN5 portable repair and review evidence

Start with round5/RN5_NEAR_MOMENT_REPAIR.md and round5/LM004_REVIEW_ERRATUM_v1.1.md.
SCOPED_HOLDS.json describes five scoped exclusions and the old report's reversible move.
These are author-side candidates, with zero organizational-independence credit.
Full near-annulus coverage, all-small-r extension, and independent acceptance remain open.

## Replay

Use Python 3.12 and the python-flint version pinned in round5/requirements.txt.
From this archive's root:

    python3 -O round5/near_moments.py
    python3 -O round5/near_boxes.py
    python3 -O round5/verify_round5.py

Expected: 65 point-law / whole-mark certificates, 10 accepted local boxes, 66 verification checks.
The original result files are in round5/output. Replay rewrites these local generated files;
elapsed time and run metadata may change. The archive itself remains the immutable baseline.
Source files are hash-pinned and copied at the paths expected by the verifier.
The complete payload inventory is DELIVERY_MANIFEST.json; it deliberately excludes itself.

## Review packet

R1. Reconstruct Holder (4,4,2) and the exact typed Gaussian counterexample.
R2. Reproduce source hashes, normalization, jet orders, six pins, marks, and H3 input scope.
R3. Audit image and spectral tail bounds in the imported rn_field.py dependency.
R4. Reconstruct conditional covariance, polynomial Wick moments, and the whole-mark cap.
R5. Verify the conditional L2 Taylor remainder and covariance remainder inequalities.
R6. Verify separate marginal whitening; no independence of Hessian blocks is assumed.
R7. Re-run positive-definiteness checks, all local boxes and the independent raw-law checks.
R8. Preserve the local-box limitation; require an actual complete cover before an integral bound.
R9. Reproduce LM004 frozen-body extraction and withdraw only the false duplicate-term claim.
R10. Disclose actual provider, authorship/coauthorship and exposure; do not claim blind review.

Current next task: construct a complete corrected near-region cover; optimize near-axis cells;
reassemble the remote budget only after all pending cells have valid bounds. Locate the exact
CL v5 archive/code before attempting executable crosswalk or accepting later CL coverage.
