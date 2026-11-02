import SourceCriticalEventTests
open SourceCriticalEventTests
example : (identityFixture (SourceCoordinates.E 1) (SourceCoordinates.E 1)).c <
    (identityFixture (SourceCoordinates.E 1) (SourceCoordinates.E 1)).f boundaryClosed.val := by
  change (0 : ℝ) < MorseCone.quadratic boundaryPoint
  rw [boundary_energy_zero]
  norm_num
