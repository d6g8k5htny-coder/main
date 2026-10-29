import ResearchFormalCoreR1
example : ResearchFormalCoreR1.foldPotential 1 1 - ResearchFormalCoreR1.foldPotential 1 (-1) = (2 : Real)^3 / 7 := by
  norm_num [ResearchFormalCoreR1.foldPotential]
