# GP-FOR-192 compiler-compatible successors

Scientific effect NONE. These changes concern this newly introduced execution project, not an accepted parent proof or a historical scientific register.

The actual first hosted run, 36351673121, on Math- commit 9cc0fff64b5e48b48b1ebc5a1bd6ef6b4ae754c7 installed Lean 4.34.1/mathlib d13f23b723b8a846827a245b89c10fc7d3f11612, passed 30 gate tests in each Python mode and verified source hashes, then FAILED to build. It did not reach the axiom audit or earn kernel-checked status. The failure artifact is 10941934636, archive SHA256 446ade8c0a869d19b93d742f43cbe2b4e784bb1ccb115aa975584fe3c1d9fc42. Its build log is retained here as evidence/initial-build-failure.log.

## Three diagnosed repairs

1. Algebra.lean:8 used `def foldPotential` over noncomputable Real division. The successor uses `noncomputable def foldPotential`. The mathematical expression is identical; no new axiom or hypothesis is introduced by this annotation.
2. Algebra.lean:49 had a `ring` command after `field_simp [hr]` had already closed the goal. Delete that surplus tactic only.
3. ProbabilityCompanions.lean:15 similarly had a surplus `ring` after `field_simp [hr, hcZ]`. Delete that surplus tactic only.

Original files are retained verbatim as originals/Algebra.lean.txt (SHA256 4c196820c4db8fafc288dd35642828ba577e3544d60aba7d1d6d24f14ad1e8ae) and originals/ProbabilityCompanions.lean.txt (SHA256 4ace6600a476859982c8851ae9097c89b082d3c96291b3c8330bd4cc00bae65d). The newly introduced, unmerged execution copies were replaced by explicitly named AlgebraV2.lean and ProbabilityCompanionsV2.lean; no original carrier was modified. Namespace and all thirteen theorem statements remain unchanged. Tests compare each successor to exactly the documented transformation of its original, including whitespace and all other proof text.

Unused-variable warnings for hcW/hq are retained, not suppressed or removed by broadening a statement. The successor needs its own actual successful build/audit/negative-control run; the existence of this note does not establish success. Statement preservation and kernel evidence remain distinct from independent alignment review.
