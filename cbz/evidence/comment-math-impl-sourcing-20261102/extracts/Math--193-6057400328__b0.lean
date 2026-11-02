import Mathlib.LinearAlgebra.Matrix.Block
import Mathlib.LinearAlgebra.Matrix.ToLinearEquiv
import Mathlib.LinearAlgebra.Finsupp.VectorSpace

noncomputable section
universe u v
namespace StrictParentBasis

open Module

variable (K : Type u) [Field K]
variable {α : Type v} [Fintype α] [LinearOrder α]

/-- Strictly decreasing parents are a conditional algebraic input. -/
def Strict (p : α → Option α) : Prop :=
  ∀ i j, p i = some j → j < i

/-- Roots contribute a point vector; other vertices contribute a fixed parent difference. -/
def difference (p : α → Option α) (i : α) : α →₀ K := by
  classical
  exact Finsupp.single i (1 : K) - match p i with
    | none => 0
    | some j => Finsupp.single j (1 : K)

def parentMatrix (p : α → Option α) : Matrix α α K :=
  fun i j => difference K p j i

theorem matrix_det_one (p : α → Option α) (hp : Strict p) :
    (parentMatrix K p).det = 1 := by
  classical
  have ht : (parentMatrix K p).IsUpperTriangular := by
    intro i j hji
    change j < i at hji
    cases h : p j with
    | none =>
        simp [parentMatrix, difference, h, (ne_of_lt hji).symm]
    | some a =>
        have ha : a < j := hp j a h
        have hai : a < i := ha.trans hji
        simp [parentMatrix, difference, h, (ne_of_lt hji).symm, (ne_of_lt hai).symm]
  have hd : ∀ i, parentMatrix K p i i = 1 := by
    intro i
    cases h : p i with
    | none => simp [parentMatrix, difference, h]
    | some j =>
        have hji : j < i := hp i j h
        simp [parentMatrix, difference, h, (ne_of_lt hji).symm]
  rw [Matrix.det_of_isUpperTriangular ht]
  simp [hd]

def equiv (p : α → Option α) (hp : Strict p) : (α →₀ K) ≃ₗ[K] (α →₀ K) := by
  classical
  exact Matrix.toLinearEquiv
    (Finsupp.basisSingleOne : Basis α K (α →₀ K)) (parentMatrix K p)
    (by rw [matrix_det_one K p hp]; exact isUnit_one)

theorem equiv_single (p : α → Option α) (hp : Strict p) (i : α) :
    equiv K p hp (Finsupp.single i 1) = difference K p i := by
  classical
  let b : Basis α K (α →₀ K) := Finsupp.basisSingleOne
  have hb : b i = Finsupp.single i 1 := by simp [b]
  change Matrix.toLin b b (parentMatrix K p) (Finsupp.single i 1) = _
  rw [← hb, Matrix.toLin_self]
  ext j
  simp [b, parentMatrix]

def basis (p : α → Option α) (hp : Strict p) : Basis α K (α →₀ K) :=
  Finsupp.basisSingleOne.map (equiv K p hp)

theorem basis_apply (p : α → Option α) (hp : Strict p) (i : α) :
    basis K p hp i = difference K p i := by
  simp [basis, Basis.map_apply, Finsupp.coe_basisSingleOne, equiv_single]

theorem family_linearIndependent (p : α → Option α) (hp : Strict p) :
    LinearIndependent K (difference K p) := by
  have hf : (basis K p hp : α → α →₀ K) = difference K p :=
    funext (basis_apply K p hp)
  rw [← hf]
  exact (basis K p hp).linearIndependent

theorem family_span (p : α → Option α) (hp : Strict p) :
    Submodule.span K (Set.range (difference K p)) = ⊤ := by
  have hf : (basis K p hp : α → α →₀ K) = difference K p :=
    funext (basis_apply K p hp)
  rw [← hf]
  exact (basis K p hp).span_eq

end StrictParentBasis

