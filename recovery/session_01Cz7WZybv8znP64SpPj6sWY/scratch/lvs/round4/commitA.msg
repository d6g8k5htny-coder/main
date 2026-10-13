Proof packet round 4: theorems of 13 registered Side24/RN Props (Side24Analysis, Side24Interfaces, RNCluster)

Three new modules of specifications/lean_statement_proofs_20261011, each a
theorem whose type is exactly a registered Prop of the statement packet applied
to its own binders, with helpers under Aux:

SpecificationProofs/Side24Analysis.lean (9): thetaSummable (d), DerivativePairingBound,
DerivativeBoundAtLeastOne, DerivativeBoundAtZero, ReferenceOddAndPinBlocks,
ReferencePinDensityProduct, IntegrandRatioBounds, LatticeImageSumBound,
DensityComparison. Routes: operator-norm Leibniz bound plus the Hermite-type
recursion (no closed form for the iterated Frechet derivative), peeling of
iteratedFDeriv in frame directions, HurwitzKernelBounds one-dimensional
summability lifted to Fin d -> Z by a piFinset product bound, Loewner
inverse/determinant monotonicity through CFC.sqrt.
SpecificationProofs/Side24Interfaces.lean (3): coneMomentD1Identity (I),
traceMoments (I), rayleighSquareLaw (I), from the interface's law fields alone
(a.e.-measurability by the Dirac-image argument; polar-coordinate disk integral
for the Rayleigh law).
SpecificationProofs/RNCluster.lean (1): rareClusterPoissonObstructionO (mu a A).

Root SpecificationProofs.lean imports the three modules after the existing eight.
check.py --allow-dirty: SOURCE_CHECK_PASS, theorem_count 73 (was 60), 72 Props
covered (was 59); test_check.py 70/70 in both Python modes; lake build clean with
-DwarningAsError; every one of the 156 theorems of the new modules depends on
exactly {propext, Classical.choice, Quot.sound}; environment audit PASS. No
hosted receipt yet: the first lean-specification-proofs.yml run on this branch is
it. Statement packet untouched; no claim-level label moves; alignment
PENDING_INDEPENDENT_REVIEW; organizational-independence credit 0. Scientific
effect NONE.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
