# Verifier criteria v8 (2026-08-02, Palm-transfer round / V5)

Supersedes v7. Changes: twelve scripts (three new Part-G scripts added);
document checks for the Part-G closures and for the honestly retained
OPEN status of RP-C/RP-S.

1. **Re-execution.** All twelve scripts executed via subprocess in BOTH
   normal and `python3 -O` modes (24 runs). Pass = exit 0 plus
   ALL_ASSERTIONS_PASS in stdout.
2. **No bare asserts; ck helper everywhere** (AST scan + source check).
3. **Part-G document checks.** The supplement must contain Theorem G.7.1
   (RP-A/RP-L closed), the RP-F audit verdict ("RP-F is CLOSED"), and the
   G.9 retention of RP-C/RP-S as OPEN with the sharply localized residual
   step; the candidate theorem must still read HOLD and the selection
   estimate "conditional on exactly two premises".
4. **Numeric-corroboration honesty.** The transcripts of
   `verify_corrected_pin_floor.py` and `verify_collar_factorization.py`
   must carry explicit NUMERIC scope-limit lines in stdout.
5. **Symbolic spot check** (as v7): the (2.3) pin equivalence and the
   pinned contact-plane identity −det H_M = (κ²η²/s²)(u²−α).

Result accounting: 30 checks; PASS requires 30/30 in both modes.
Transcripts: `verify.out.txt`, `verify.O.out.txt`.

Scope limit: as before — these criteria verify internal algebraic,
arithmetic, document, and regression claims. They are organizational
evidence, not proof evidence. RP-C/RP-S remain open; the candidate
theorem remains on HOLD.
