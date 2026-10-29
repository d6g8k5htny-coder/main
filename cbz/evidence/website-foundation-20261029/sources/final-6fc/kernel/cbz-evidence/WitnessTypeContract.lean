import CBZNonvacuity
example : (∃ q : Nat, q ∈ [0,1,2,3,4,5,6]) := CBZNonvacuity.pairing_domain_inhabited
example : ((0:Nat) ∈ [0,1,2,3,4,5,6] ∧ (1:Nat) ∈ [0,1,2,3,4,5,6] ∧ (2:Nat) ∈ [0,1,2,3,4,5,6] ∧ (3:Nat) ∈ [0,1,2,3,4,5,6] ∧ (4:Nat) ∈ [0,1,2,3,4,5,6] ∧ (5:Nat) ∈ [0,1,2,3,4,5,6] ∧ (6:Nat) ∈ [0,1,2,3,4,5,6]) := CBZNonvacuity.pairing_all_seven
example : (∃ d : Nat, d ∈ [2,3]) := CBZNonvacuity.dimensions_inhabited
example : ((2:Nat) ∈ [2,3] ∧ (3:Nat) ∈ [2,3]) := CBZNonvacuity.dimensions_both
