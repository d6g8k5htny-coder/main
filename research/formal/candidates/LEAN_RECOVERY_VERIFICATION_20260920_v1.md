# Recovered Lean algebra verification — 2026-09-20 v1

Thirteen recovered theorem declarations compile with Lean v4.19.0 and Mathlib commit c44e0c8ee63ca166450922a373c7409c5d26b00b. All original theorem statements and proof tactics are unchanged. The original build failed because foldPotential used real-number division without a noncomputable declaration; the only Lean-source repair adds that declaration. The original source carrier and failing build log are preserved.

This advances the compilation subtask in OQ-013 / HB-FOR-192. It does not recover the lost GP-FOR-001 project, formally prove the surrounding probability or Gaussian-field premises, change governing statuses, or supply organizationally independent review. The dated Status.lean records remain source metadata; they are not assertions that today's registers have those statuses.

## Contents and verified scope

- original/GP_FOR_192.zip: exact 10644-byte Drive carrier 1hTeWNKxLcmXEB2i5enAdmzXFknUxwTL6; SHA256 a6440511fc259706457b18227734013966251105d538fdeabe633bb73f168631.
- recovered/: corrected source, exact Lean toolchain, Lake configuration and complete transitive dependency manifest. It contains six algebraic declarations (fold gap, Gram scalar expansion, two convex-combination identities, scalar cancellation, contact-power identity) and seven probability-algebra companions (cancellation, cross-multiplication, quotient, threshold and power ledgers).
- audit/: original build failure, tool/dependency setup logs, exact source comparison, and bootstrap metadata.
- verification/: successful build, direct source elaboration even with existing build outputs, every theorem's axiom report, two explicit counterexamples and two required compiler rejection controls.
- run_formal_verification.py: repeatable source/lock-bound execution and rejection checks; all output goes to a new directory outside the source project.

All 13 theorem axiom reports contain only propext, Classical.choice and Quot.sound; no sorryAx occurs. The wrong fold changes denominator 6 to 7; s=1 gives 4/3 != 8/7. The wrong quotient halves the coefficient; all scalar inputs equal to 1 satisfy the original hypotheses but would require 1 <= 1/2. Both mutations are rejected and both displayed counterexample inequalities elaborate.

## Replay

Use the official Lean installation procedure and the checked-in lean-toolchain. Mathlib's v4.19.0 release declares this same toolchain: https://raw.githubusercontent.com/leanprover-community/mathlib4/v4.19.0/lean-toolchain . Lake dependency/cache guidance: https://leanprover-community.github.io/install/project.html . No runtime or dependency cache is bundled.

With Lean v4.19.0 available, run lake update and lake exe cache get in recovered/. Then run the following from this extracted bundle, replacing the two absolute paths:

    python3 run_formal_verification.py --project recovered --lean-bin /absolute/path/to/lean-4.19.0/bin --output /absolute/path/to/new-verification-directory

The verifier requires the pinned toolchain and Mathlib revision, elaborates the actual sources, audits all 13 theorem declarations, and requires the negative controls to fail. Its report binds source bytes, compiler executable, command logs and exact outcomes. It is a local execution record, not an authenticated timestamp or proof of another provider's independent review. Standard Lean/Mathlib foundations and compiler correctness remain trusted.

The member manifest covers every file other than itself. The outer ZIP SHA and byte count are supplied separately by the delivery receipt.
