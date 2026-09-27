/-!
# SIDE24 published endpoints — consistency of the frozen artifact

Informal source (Layer 0, byte-pinned): `Math-` commit
`9d7b6802424fb4715b31999066aafca8ee2f3cca`, `coefficients/side24_v1/ENCLOSURE.json`,
1090 bytes, SHA-256 `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811`.

The published decimal endpoints are transcribed here as integers in units of
`10^-20`. The kernel checks that each interval is nonempty and exactly one
`10^-20` step wide, and that the two intervals are disjoint and ordered.

This module does **not** establish that the coefficient lies in either
interval: that claim needs `Real.Gamma`, `Real.pi`, `Real.exp` and the
Gaussian cone moments, none of which exist in core Lean. Its formalization
status is recorded as `none` in `formal/registry.json`.
-/

namespace UniversalLaw.Side24

/-- `0.07340691930603427103` and `0.07340691930603427104` in units of `10^-20`. -/
def d2Lower : Nat := 7340691930603427103
def d2Upper : Nat := 7340691930603427104

/-- `0.04177593184059834334` and `0.04177593184059834335` in units of `10^-20`. -/
def d3Lower : Nat := 4177593184059834334
def d3Upper : Nat := 4177593184059834335

/-- Source: "0.07340691930603427103 < c_2,24 < 0.07340691930603427104" — the
published endpoints are distinct and adjacent on the 20-digit grid. -/
theorem d2_endpoints_adjacent : d2Lower < d2Upper ∧ d2Upper = d2Lower + 1 := by decide

/-- Source: "0.04177593184059834334 < c_3,24 < 0.04177593184059834335". -/
theorem d3_endpoints_adjacent : d3Lower < d3Upper ∧ d3Upper = d3Lower + 1 := by decide

/-- The published `d = 3` interval lies strictly below the `d = 2` interval. -/
theorem d3_below_d2 : d3Upper < d2Lower := by decide

/-- Both published intervals lie in `(0, 1/10)`: `10^-20`-units below `10^19`. -/
theorem endpoints_in_unit_tenth : 0 < d3Lower ∧ d2Upper < 10 ^ 19 := by decide

end UniversalLaw.Side24
