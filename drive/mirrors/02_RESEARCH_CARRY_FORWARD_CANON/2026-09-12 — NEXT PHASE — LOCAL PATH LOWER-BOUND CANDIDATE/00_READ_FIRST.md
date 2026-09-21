# NEXT PHASE — consolidation and local-path lower-bound candidate

**Researcher:** Dylan Roy. **Date:** 2026-09-12. **Package ID:** NEXT-PHASE-20260912-v1.0.

**Status:** ADDITIVE RESEARCH PACKAGE. The ratified 3D SIDE24 result is preserved. P0.1 remains HOLD. The accepted 2D lower campaign remains OPEN. This package adds a complete author-side candidate proof for an existential cubic lower bound; it does not promote that proof or supersede the old controlling status records.

## The substantive advance

The new candidate replaces counted third-saddle reliability by a local superlevel path. The exact polynomial

    F(x,y)=x^3/3-x/4-1/12+2(x^2-1/4)y-5y^2

has the required pinned maximum and saddle. A path from (-1/2,0) to (-2,3/4) has minimum height -99/1280, strictly above the candidate saddle level -1/6, and endpoint height 9/16>0. A C2 perturbation tolerance 1/16 preserves the required level separation and Hessian types.

Inside the original six-pin conditional Gaussian law, the candidate restricts one transverse second derivative to an interval of width proportional to r and three cubic coefficients to fixed intervals. The resulting event has probability at least a positive constant times r. The determinant weight on it is at least 65 r^4. An independent normalizer estimate gives Z_r<=C_Z r^2. The proposed conclusion is therefore 1-q(r,6/5)>=c r^3 for every sufficiently small r, with some c>0.

The complete proof is in **02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md**. It supplies the conditional Gaussian and Taylor arguments, not only a power-counting heuristic. No numerical value of c or r0 is claimed. The mathematical analysis remains an author-side candidate pending independent review.

## Read in this order

1. **01_CURRENT_STATE.md** and **CURRENT_STATE.json** — current recorded state, exact source addresses, and the no-promotion boundary.
2. **02_LOCAL_PATH_LOWER_BOUND_CANDIDATE.md** — exact object, witness, robustness, Gaussian density, conditional remainder control, normalizer and composition.
3. **03_INDEPENDENT_REVIEW_PACKET.md** — six decisive review interfaces and required evidence.
4. **EXECUTION_RECEIPT.json** — actual runs, counts, hashes and limitations.
5. **INPUT_HASH_LEDGER.json** and **MANIFEST.sha256** — source retrieval and package identities.

**04_WEIGHTED_TRANSFER_BACKUP.md** preserves a safe Holder/uniform-integrability route for the older quantitative campaign. **RELATED_WORK_SCREENING.md** records limited primary-source screening and expressly withholds a novelty claim.

## What was executed

Five exact raw Drive sources were retrieved and hashed. GP-DER-118-v1.10's 30,933-byte whole-file identity and 29,293-byte frozen-body identity both match the recorded hashes. The 32,989-byte W10 source hash also matches.

The algebra companion passed **48 checks** in normal Python and Python -O with byte-identical outputs. **Eight mutation families, sixteen executions**, were rejected as intended. The run suite does not use bare assert statements as acceptance gates. These are author-side algebra/regression checks, not independent review, full Gaussian probability certification, or Lean compilation.

No pre-existing theorem, code file, manifest, control overlay or research register was overwritten. The new state view is a reference view, not a replacement control overlay. The old 1.2.1 q0 verifier was not installed or changed.

## Reproduce the included checks

    python -m pip install -r requirements.txt
    python verify_local_path.py
    python -O verify_local_path.py
    python run_reproduction.py

The checker needs only its own source, CURRENT_STATE.json, and the pinned SymPy dependency. The runner also records the proof and dependency file hashes. The runner regenerates the included test receipts; it does not alter Drive or evaluate scientific status.

## Main remaining action

Perform a genuinely independent mathematical review of the exact candidate version. Especially scrutinize the exact conditioning version, ten-jet covariance limit, conditional C4 norm bound inside the shrinking jet event, and the topological interpretation of the path. A reviewer may approve the candidate, identify an amendment, or refute it; no favorable result is presumed.

For stronger quantitative goals, the original WP/Lambda/Bonferroni and other loss obligations remain open. The candidate does not establish the measured 0.9144-class coefficient, an optimal constant, a numerical radius, a limiting coefficient, a full lifetime distribution, or a result in another dimension.

## Integrity convention

MANIFEST.sha256 hashes the final authored package files but excludes itself. A later Drive readback receipt is a separate envelope and is excluded from that manifest to avoid circular hashing. Original inputs are not duplicated into this Drive folder or the distribution ZIP; their source IDs and exact retrieved hashes are recorded in INPUT_HASH_LEDGER.json. This package is not a complete copy of the research Drive.
