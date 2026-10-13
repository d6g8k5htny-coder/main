import Specifications.Side24

 
                                                                                       

                                                                                    

                                                                                          
                                                                                               
                                                                                            
                                                                                               
                                                                                             
                                                                                               
                                                            

                                                                                           
                                                                                             
                                                                                              
                                                                                         
                                                                       
                                                                                            
                                                                                             
                                                                                              
                                                                           
                                                                                                          
                                                   


namespace UniversalLaw.Proofs.Side24Analysis

open UniversalLaw.Spec.Side24

namespace Aux

                                                       

                 
theorem ir_epsilon_pos : (0 : ℝ) < epsilon := by
  unfold epsilon
  positivity

                                                                                    
theorem ir_epsilon_lt : epsilon < (1 / 100 : ℝ) := by
  unfold epsilon
  norm_num

                     
theorem ir_one_sub_epsilon_pos : (0 : ℝ) < 1 - epsilon := by
  have := ir_epsilon_lt
  linarith

                     
theorem ir_one_le_one_add_epsilon : (1 : ℝ) ≤ 1 + epsilon := by
  have := ir_epsilon_pos
  linarith

                                            

                                                                                            
                                                                                             
                                            
theorem ir_rpow_le_pow_five (b : ℝ) (hb : b ≤ 5) :
    (1 + epsilon) ^ b ≤ (1 + epsilon) ^ 5 := by
  have h := Real.rpow_le_rpow_of_exponent_le ir_one_le_one_add_epsilon hb
  rwa [show (5 : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast] at h

                                                                        
                                                                        
theorem ir_bernoulli (a : ℝ) (ha : 1 ≤ a) : 1 - a * epsilon ≤ (1 - epsilon) ^ a := by
  have hε := ir_epsilon_lt
  have h := one_add_mul_self_le_rpow_one_add (show (-1 : ℝ) ≤ -epsilon by linarith) ha
  rw [show (1 : ℝ) + -epsilon = 1 - epsilon by ring] at h
  linarith

                                                                         

                                                                                
theorem ir_lower_rational : (1 - 13 * epsilon) * (1 + epsilon) ^ 5 ≤ 1 - 5 * epsilon := by
  unfold epsilon
  norm_num

                                                                                  
theorem ir_upper_rational : (1 + epsilon) ^ 5 ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := by
  unfold epsilon
  norm_num

                                                               

                                               
                                                      
theorem ir_lower_bound (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b := by
  have hε := ir_epsilon_pos
  have hε1 := ir_epsilon_lt
  have hpb : 0 < (1 + epsilon) ^ b := Real.rpow_pos_of_pos (by linarith) _
  have hae : a * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right ha5 hε.le
  rw [le_div_iff₀ hpb]
  calc (1 - 13 * epsilon) * (1 + epsilon) ^ b
      ≤ (1 - 13 * epsilon) * (1 + epsilon) ^ 5 :=
        mul_le_mul_of_nonneg_left (ir_rpow_le_pow_five b hb5) (by linarith)
    _ ≤ 1 - 5 * epsilon := ir_lower_rational
    _ ≤ 1 - a * epsilon := by linarith
    _ ≤ (1 - epsilon) ^ a := ir_bernoulli a ha1

                                           
                                                      
theorem ir_upper_bound (a b : ℝ) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon := by
  have hε := ir_epsilon_pos
  have hqb : 0 < (1 - epsilon) ^ b := Real.rpow_pos_of_pos ir_one_sub_epsilon_pos _
  have hbe : b * epsilon ≤ 5 * epsilon := mul_le_mul_of_nonneg_right hb5 hε.le
  rw [div_le_iff₀ hqb]
  calc (1 + epsilon) ^ a ≤ (1 + epsilon) ^ 5 := ir_rpow_le_pow_five a ha5
    _ ≤ (1 + 28 * epsilon) * (1 - 5 * epsilon) := ir_upper_rational
    _ ≤ (1 + 28 * epsilon) * (1 - b * epsilon) :=
        mul_le_mul_of_nonneg_left (by linarith) (by linarith)
    _ ≤ (1 + 28 * epsilon) * (1 - epsilon) ^ b :=
        mul_le_mul_of_nonneg_left (ir_bernoulli b hb1) (by linarith)

                                                               
theorem ir_ratio_bounds (a b : ℝ) (ha1 : 1 ≤ a) (ha5 : a ≤ 5) (hb1 : 1 ≤ b) (hb5 : b ≤ 5) :
    1 - 13 * epsilon ≤ (1 - epsilon) ^ a / (1 + epsilon) ^ b ∧
      (1 + epsilon) ^ a / (1 - epsilon) ^ b ≤ 1 + 28 * epsilon :=
  ⟨ir_lower_bound a b ha1 ha5 hb5, ir_upper_bound a b ha5 hb1 hb5⟩

                                                    

                                                            
theorem ir_exponentA_two : exponentA 2 = 13 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

                                                     
theorem ir_exponentB_two : exponentB 2 = 5 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

                                                            
theorem ir_exponentA_three : exponentA 3 = 25 / 6 := by
  simp [exponentA, symCoordDim]
  norm_num

                                                     
theorem ir_exponentB_three : exponentB 3 = 9 / 2 := by
  simp [exponentB, symCoordDim]
  norm_num

end Aux

                                                                      

                                                                                       
                                                                                            
                                                                                              
                                                                          
                                                                                     
                                                                               
                                                                        
                                                                                     
                                                                  
                                                                         
                                                   
                                                                                            
                                                                                            
                                                                                                
                                                                                       
                                                                                         
                                                                                          
                                                                                            
                                                                                            
                                                                                            
                                                                                        
theorem integrandRatioBounds : UniversalLaw.Spec.Side24.IntegrandRatioBounds := by
  intro d hd
  rcases hd with rfl | rfl
  · rw [Aux.ir_exponentA_two, Aux.ir_exponentB_two]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  · rw [Aux.ir_exponentA_three, Aux.ir_exponentB_three]
    exact Aux.ir_ratio_bounds _ _ (by norm_num) (by norm_num) (by norm_num) (by norm_num)

end UniversalLaw.Proofs.Side24Analysis
