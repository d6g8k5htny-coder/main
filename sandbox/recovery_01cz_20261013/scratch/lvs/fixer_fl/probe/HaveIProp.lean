-- Core-Lean probe (no Mathlib, no lake): the exact shape planned for IsLifetimeDensityVersion:
-- a `def … : Prop :=` whose body opens with a term-level `haveI := G.mΩ` on its own line,
-- followed by a multi-line conjunction that needs the instance through (a) an inst-implicit
-- argument, (b) a FunLike-style class resolved by TC, (c) a plain implicit `{_ : MS α}`.
class MS (α : Type) where
structure Meas (α : Type) [MS α] where
class FL (F : Type) (α : outParam Type) where
instance instFL {α : Type} [MS α] : FL (Meas α) α := ⟨⟩
def aeFL {α F : Type} [FL F α] (μ : F) (p : α → Prop) : Prop := ∀ x, p x
def mapInst {α : Type} [MS α] (μ : Meas α) : Prop := True
def intImpl {α : Type} {_ : MS α} (μ : Meas α) (f : α → Nat) : Nat := 0
structure Foo where
  Ω : Type
  [mΩ : MS Ω]
  P : Meas Ω
  F : Ω → Nat → Nat

def IsVersion (G : Foo) (ν : Nat → Nat) : Prop :=
  haveI := G.mΩ
  (∀ ℓ : Nat, 0 ≤ ν ℓ) ∧
  (∀ A : Nat,
    aeFL G.P (fun ω => G.F ω A = 0) ∧
    mapInst G.P ∧
    intImpl G.P (fun ω => G.F ω A) = 0)

-- the same without haveI: (b) and (c) are expected to elaborate, (a) is expected to FAIL
def NoHaveI (G : Foo) : Prop :=
  aeFL G.P (fun ω => G.F ω 0 = 0) ∧ intImpl G.P (fun ω => G.F ω 0) = 0
def NoHaveIInst (G : Foo) : Prop := mapInst G.P

#print IsVersion
#print axioms IsVersion
