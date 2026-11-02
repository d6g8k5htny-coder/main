import SourceCriticalEvent
open SourceCoordinates SourceCriticalEvent
example : ‖2 * oneReal (EuclideanSpace.single 0 (1 : ℝ))‖^2 =
    ‖(EuclideanSpace.single 0 (1 : ℝ) : E 1)‖^2 := by
  norm_num [oneReal_apply, PiLp.norm_single]
