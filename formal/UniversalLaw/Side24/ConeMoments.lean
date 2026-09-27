/-!
# SIDE24 coefficient note — reference covariance and cone moments, arithmetic skeleton

Informal source (Layer 0, byte-pinned): `Math-` commit
`9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/PROOF.md`,
SHA-256 `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769`,
section 1.

Core Lean has no rational-number type, so every rational identity is stated
as the equivalent cross-multiplied integer identity, with the rational form in
the docstring. The kernel checks the integers; a reviewer checks that the
integer statement is the stated rational statement (the alignment review lane).

Not formalised here: the Gaussian conditioning that produces the covariance
formula `Cov(A_ij,A_kl | V=0) = (2/3)δ_ijδ_kl + δ_ikδ_jl + δ_ilδ_jk`, the
elementary integral `∫_0^a (a-z)^2 e^{-z/2} dz/2 = a^2-4a+8-8e^{-a/2}`, the
Gaussian moment identities `E s^4 = 3 (Var s)^2` and
`E exp(-s^2/2) = (1+Var s)^{-1/2}`, and the Rayleigh cone integration. Only the
arithmetic that combines those inputs is kernel-checked.
-/

namespace UniversalLaw.Side24

/-- Source: "Var(t)=15, Cov(t,G)=(-3,0,...,0)" and "tau^2=Var(t|G=0)=6":
the Gaussian conditional variance `15 - (-3)^2 / Var(G_1)` with `Var(G_1) = 1`. -/
theorem conditional_third_derivative_variance : (15 : Int) - (-3) * (-3) / 1 = 6 := by
  decide

/-- Source: "Cov(A_ij,A_kl | V=0) = (2/3)delta_ij delta_kl + delta_ik delta_jl
+ delta_il delta_jk". With `i=j=k=l` the three deltas give `2/3 + 1 + 1 = 8/3`
(source: "For m=1 the variance of A is8/3"); for `A_12` only `δ_ik δ_jl` survives
(variance `1`); for the pair `A_11, A_22` only `(2/3)δ_ij δ_kl` survives
(covariance `2/3`). All in thirds. -/
theorem transverse_covariance_entries_thirds :
    2 + 3 + 3 = 8 ∧ 0 + 3 + 0 = 3 ∧ 2 + 0 + 0 = 2 := by decide

/-- Source: "D_1=E[A^2 1{A<0}]=4/3" — half of the variance `8/3` by centred
symmetry. In thirds: `8 = 2 · 4`. -/
theorem cone_moment_m1_thirds : 8 = 2 * 4 := by decide

/-- Source: "For m=2 write A=[[s+x,y],[y,s-x]]. Then s,x,y are independent,
Var(s)=5/3 and Var(x)=Var(y)=1." With `s = (A_11+A_22)/2`, `x = (A_11-A_22)/2`:
`Var s = (8/3 + 8/3 + 2·2/3)/4 = 20/12 = 5/3`,
`Var x = (8/3 + 8/3 - 2·2/3)/4 = 12/12 = 1`,
`Cov(s,x) = (Var A_11 - Var A_22)/4 = 0`, `Var y = Var A_12 = 1`.
Stated in thirds over the common denominator 12. This is the "shared trace
variance5/3" that the note lists first among the items for nonauthor review. -/
theorem trace_and_traceless_variances :
    8 + 8 + 2 * 2 = 20 ∧ 20 * 3 = 5 * 12 ∧
    8 + 8 - 2 * 2 = 12 ∧ 12 = 1 * 12 ∧
    (8 : Int) - 8 = 0 := by decide

/-- Source: "E s^2=5/3, E s^4=25/3": for a centred Gaussian `E s^4 = 3 (E s^2)^2`,
so `E s^4 = 3 · (5/3)^2 = 75/9 = 25/3`. Cross-multiplied: `3 · 5^2 · 3 = 25 · 9`. -/
theorem fourth_moment_of_trace : 3 * 5 ^ 2 * 3 = 25 * 9 := by decide

/-- Source: "E exp(-s^2/2)=sqrt(3/8)": `E exp(-s^2/2) = (1 + Var s)^{-1/2}` with
`1 + 5/3 = 8/3`. In thirds: `3 + 5 = 8`. -/
theorem exponential_moment_of_trace : 3 + 5 = 8 := by decide

/-- Source: "D_2 = (1/2)[25/3-20/3+8-8sqrt(3/8)] = 29/6-sqrt(6)".
The bracket is the polynomial `a^2 - 4a + 8 - 8e^{-a/2}` averaged at
`a = s^2`, so `4 · E s^2 = 4 · 5/3 = 20/3`.
Rational part: `(25/3 - 20/3 + 8)/2 = (29/3)/2 = 29/6`; in thirds
`25 - 20 + 24 = 29`. Irrational part: `(1/2)·8·√(3/8) = 4√(3/8) = √(16·3/8) = √6`;
cross-multiplied `4^2 · 3 = 6 · 8`. -/
theorem cone_moment_m2_algebra :
    4 * 5 = 20 ∧ 25 - 20 + 24 = 29 ∧ 4 ^ 2 * 3 = 6 * 8 := by decide

/-- Source: "simplifying 6^(2/3)/24^(1/3)=(3/2)^(1/3)". Cubing both sides:
`6^2 / 24 = 3/2`, i.e. `6^2 · 2 = 3 · 24`. -/
theorem cube_root_simplification : 6 ^ 2 * 2 = 3 * 24 := by decide

/-- Source: "p_G(0) p_V(0) = (2pi)^(-d) / sqrt(3)": `det diag(3,1,...,1) = 3`, and
"Its dimension is at most10": the joint vector `(G, t, svec H)` has
`d + 1 + d(d+1)/2` coordinates, `6` for `d = 2` and `10` for `d = 3`. -/
theorem pin_determinant_and_joint_dimension :
    3 * 1 * 1 = 3 ∧ 2 + 1 + 2 * 3 / 2 = 6 ∧ 3 + 1 + 3 * 4 / 2 = 10 := by decide

/-- Source: "The reference Hessian block has eigenvalues d+2 on trace and2 on
traceless matrices": for `d = 2,3` these are `4, 5` and `2`; each exceeds `1/3`
(`1 < 3·2`), consistent with the later `C_ref >= I/3`. -/
theorem hessian_block_eigenvalues : 2 + 2 = 4 ∧ 3 + 2 = 5 ∧ 1 < 3 * 2 := by decide

end UniversalLaw.Side24
