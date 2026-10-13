#!/usr/bin/env python3
"""Round 7, third slice (RN successor pinned-law interface): DRAFT generator for the design read. Reads the landed RN.lean
(Math- 7f5f1c82), the landed REGISTER.json and the packet's pinned source copies; writes RN_append.lean (the block to
append before `end`), RN_draft.lean (landed module + block), register_rows_proposed.json (NOT applied) and scan_draft.txt
(the gate's own scan_module inventory of the draft plus check_register on a provisional register). Every anchor is sliced
from the pinned source text and asserted verbatim. Nothing touches any packet file. Scientific effect NONE.
usage: python3 -B -S gen_rn_append.py <scratch>"""
import copy, hashlib, importlib.util, json, re, sys
from collections import Counter
from pathlib import Path, PurePosixPath
S = Path(sys.argv[1]); H = S / 'lvs/r7/hold_rn'; PK = S / 'lvs/r7/wt_lt/specifications/lean_statements_20261010'
PACKAGE = 'specifications/lean_statements_20261010'; MODULE = 'Specifications/RN.lean'; NS = 'UniversalLaw.Spec.RN.'
landed = (H / 'RN_landed.lean').read_text(encoding='utf-8'); lines = landed.splitlines()
reg = json.loads((PK / 'REGISTER.json').read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('chk', PK / 'check.py'); check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
SRC = {s['id']: s for s in reg['sources']}
def src_line(sid): s = SRC[sid]; return f"Source: {s['repository']} commit {s['commit']} path {s['path']} sha256 {s['sha256']}"
def src_text(sid): return (PK / SRC[sid]['local_copy']).read_text(encoding='utf-8')
def anchor(sid, marker, start=None, end=None):
    """the unique source line containing `marker`, sliced [start:end] (substrings), asserted verbatim in the source"""
    t = src_text(sid); ls = [l for l in t.splitlines() if marker in l]; assert len(ls) == 1, (sid, marker, len(ls))
    l = ls[0]
    if start is not None: l = l[l.index(start):]
    if end is not None: l = l[:l.index(end) + len(end)]
    l = l.strip(); assert len(l) >= 16 and l in t and '"' not in l, (sid, l); return l
RW, LP, D5R, DL, FC, PR, RC, AB, FA = ('rn-fixed-remote-window', 'lifetime-parent', 'd5-reconciliation', 'd5-dimension-lift', 'c6-fourier-cutoff',
                                      'c6-palm-route', 'remote-collision', 'rn-annulus-bridge', 'rn-fixed-annulus-window')
A_REG = anchor(RW, 'Q_r denotes continuous Gaussian regression', end='observations.')
A_FJ = anchor(RW, 'F_j(H)=|det H| 1{H has j negative eigenvalues}')
A_W = anchor(RW, 'W_r=F_d(H_M) F_(d-1)(H_S),')
A_KR12 = anchor(RW, 'E[(W_r/r^2) F_j(H_x)|U_r=v_r,Y_x=(0,t)] dt dx. (12)')
A_DENS = anchor(RW, 'The conditional density p_(Y_x|U_r=v_r)(0,t) differs', start='The conditional density', end='p_(Y_x|U_r=v_r)(0,t)')
A_KR = anchor(RW, 'Apply Kac-Rice under Q_r to grad f(x) on the compact remote domain.', end='compact remote domain.')
A_POS = anchor(RW, 'their inverses, and the relevant Schur complements are uniformly bounded and positive', start='Uniform convergence shows', end='for r<=r_*.')
A_K = anchor(LP, 'K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)', start='K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)', end='exp(-|z+Ln|^2/2)')
A_INERTIA = anchor(LP, 'A maximum has d negative Hessian eigenvalues; the relevant saddle has d-1 negative and one positive eigenvalue.',
                   start='A maximum has d negative', end='one positive eigenvalue.')
A_Z55 = anchor(LP, '0<z_*<=Z_r/r^2<=z^*<infinity')
A_MORSE = anchor(LP, 'so their determinants are nonzero almost surely', start='Pinned Hessians have a full density', end='nonzero almost surely.')
assert A_K == 'K_L(z) = sum_{n in Z^d} exp(-|z+Ln|^2/2)'

# ----------------------------------------------------------------------------- predecessors: the 19 interface-blocked Props
preds = []
for i, l in enumerate(lines):
    m = re.match(r'^def (\w+)', l)
    if m and i >= 540 and m.group(1) != 'RareClusterPoissonObstructionO':
        j = i - 1; d = {}
        while j > 0 and not lines[j].startswith('/--'):
            for key in ('Source:', 'Anchor:', 'Interface:', 'Does not claim:'):
                if lines[j].startswith(key): d[key] = lines[j][len(key):].strip()
            j -= 1
        d['first'] = lines[j][len('/-- '):]
        k = i; hdr = []
        while True:
            hdr.append(lines[k])
            if ': Prop :=' in lines[k]: break
            k += 1
        d['name'] = m.group(1); d['hdr'] = hdr; d['line'] = i + 1
        d['sid'] = next(s['id'] for s in reg['sources'] if s['path'] == re.search(r'path (\S+)', d['Source:']).group(1))
        preds.append(d)
assert len(preds) == 19 and all('PinnedLawInterface' in ''.join(p['hdr']) for p in preds)
# successor anchors, one per predecessor, distinct from the predecessor's own anchor
SUCC_ANCHOR = {
 'FixedRemoteMeanMeasureA2': anchor(RW, '- integral_E Lambda_j(x;b,k,u) dx |'),
 'FixedRemoteCountBoundA3': anchor(RW, 'P_(Q_r^W){N_(r,j)(E)>=1} <= C k r^3 |E|.', end='|E|.'),
 'FixedAnnulusWindowCandidateA2': anchor(FA, '**Candidate theorem.** There are C,r_*>0, uniform in the frame, compact marks, E0'),
 'AnnulusBridgeAllHeight2': anchor(AB, '**Candidate theorem.** There are C,r_*>0, uniform in the stated marks, frames, Borel E'),
 'D5aPuncturedPinDisks': anchor(D5R, 'for Borel `E ⊂ {0 < p² + q² ≤ 1/16}`, and the same at `S`', start='for Borel `E ⊂ {0 < p²', end='the same at `S`'),
 'D5bMicrodisk': anchor(D5R, '**D5-b** (microdisk)', start='**D5-b** (microdisk)', end='`O(r⁵)`'),
 'D5cCollarC1': anchor(D5R, '**D5-c** (C1)', start='**D5-c** (C1)', end='fixed `R ≥ 1`'),
 'D5dSummedPinNeighbourhoodC2': anchor(D5R, 'the same on `{|(u,v)| ≤ R} ∖ {(±1/2, 0)}`', start='the same on `{|(u,v)|', end='{(±1/2, 0)}`'),
 'D5eIntermediateShellsI3I4': anchor(D5R, 'the dyadic shells and `{A_0 r ≤ |X| ≤ ρ}`', start='the dyadic shells and', end='|X| ≤ ρ}`'),
 'D5fGlobalWindowFirstMomentI5': anchor(D5R, 'planar `c r³ ≤ P(A_r) ≤ C r³`', start='planar `c r³', end='≤ C r³`'),
 'DimensionLiftPuncturedPinBallP': anchor(DL, '**Theorem P_d (all-height punctured pin ball).**', end='pin ball).**'),
 'DimensionLiftCompactCollarC': anchor(DL, '**Theorem C_d (compact collar).**', end='collar).**'),
 'DimensionLiftIntermediateShellsI': anchor(DL, '**Theorem I_d (intermediate height-window shells).**', end='shells).**'),
 'DimensionLiftGlobalWindowG': anchor(DL, '**Theorem G_d (global height-window first moment).**', end='moment).**'),
 'FourierCutoffPlanarSecondFactorialF4': anchor(FC, 'For d=2, G_2 is the reviewed planar I5 composition under the complete D5'),
 'PalmRouteFactorialMomentsQ': anchor(PR, '**Theorem Q (factorial moments, every fixed dimension).**', end='dimension).**'),
 'PalmRouteSecondFactorialTheta': anchor(PR, '**Corollary Theta.** In every fixed `d >= 2`,'),
 'RemoteCollisionSecondFactorialD': anchor(RC, 'Q_r^W{ N(E) >= 2 } <= C r^5 |E|.', end='|E|.'),
 'WitnessCollisionNearPinPairsTarget': anchor(D5R, '**D5-open (catalog C6): shrinking multiple-witness collision.**', start='**D5-open', end='collision.**'),
}
for p in preds:
    p['succ_anchor'] = SUCC_ANCHOR[p['name']]; assert p['succ_anchor'] != p['Anchor:'].strip('"'), p['name']
    assert p['succ_anchor'] in src_text(p['sid']), p['name']

# ----------------------------------------------------------------------------- the hand-written block: auxiliaries and the successor interface
def doc(summary, sid, anc, iface, dnc):
    return f"/-- {summary}\n\n{src_line(sid)}\nAnchor: \"{anc}\"\nInterface: {iface}\nDoes not claim: {dnc} -/\n"
BLOCK = []
BLOCK.append("""
/-! ## Round 7, third slice (DRAFT for the design read; not activated): the successor pinned-law interface

Everything above this line is byte-identical to the module at Math- `7f5f1c82` (unchanged since `8acc4856`). The
declarations below transcribe what `PinnedLawInterface` deliberately left informal (its docstring and the scope note before
it): the periodized covariance `K_L` of the field, the `2(d+1)` pin observations and the jet covariances of the stationary
field, the Gaussian regression mean and Schur-complement covariance, the typed determinant weight
`W_r = F_d(H_M) F_(d-1)(H_S)` of a sample path, and the Kac-Rice contact kernel `Λ'_j(x; b, k, R)(y)` DEFINED from the
regression law's Gaussian parameters with two Schur steps (the `(d+1)`-jet at `x` given the pins; the three Hessians at
`M`, `S`, `x` given the pins and that jet). The successor `PinnedLawInterface2 extends PinnedLawInterface` adds the fields
that tie the given objects to these definitions: `pinObsCov_posDef`, `Q_regression`, `W_det`, `Z_floor`, `critical_finite`,
`count_measurable`, `kacRice_window`, `kacRice_allHeight`. The nineteen successor Props restate the nineteen
interface-blocked Props of the Frontier section over it. Formal progress of every declaration here: `specified`. Nothing is
proved; the predecessor interface and Props are unchanged; no inhabitant is constructed. What the successor does NOT pin
(for the design read): the parent's free kernel `Λ` (Theorem A's limit object as `r → 0`; the mean measures themselves
become determined instead); the sign and scale conventions of the derivative covariances are hand-written
(`Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K(x - y)`); the Hessian coordinates are the raw upper-triangular
entries; `Z_floor` and the two Kac-Rice identities are theorems of the sources assumed as fields (Mathlib `d13f23b7` has no
Kac-Rice formula); measurability and integrability beyond the stated fields are not stated. -/
""")
BLOCK.append(doc("The unnormalised periodized Gaussian (theta) sum `sum_{n in Z^d} exp(-|z+Ln|^2/2)`, as `Lifetime.lean` has it (the packet's convention is duplication per module; this module had no covariance kernel because its interface left the regression informal).",
    LP, A_K, 'none.', 'summability (Mathlib\'s `tsum` is `0` for a non-summable family); anything about the field beyond its covariance.') +
"noncomputable def thetaSum (d : ℕ) (L : ℝ) (z : E d) : ℝ :=\n  ∑' n : Fin d → ℤ, Real.exp (-(‖z + latticeVec d L n‖ ^ 2) / 2)\n")
BLOCK.append(doc("The periodized Gaussian covariance `K_L(z) / K_L(0)` of the field, normalised to `1` at `0`.",
    LP, A_K, 'none.', 'positive definiteness of the kernel; that this normalisation is the sources\' (the regression formulas below are invariant under a constant factor).') +
"noncomputable def periodicCovariance (d : ℕ) (L : ℝ) (z : E d) : ℝ :=\n  thetaSum d L z / thetaSum d L 0\n")
BLOCK.append(doc("The iterated directional derivative `∂_{l_1} ⋯ ∂_{l_m} K_L (z)` of the covariance along a list of directions (`iteratedFDeriv`, which is `0` where the function is not differentiable enough).",
    RW, A_REG, 'none.', 'differentiability of `K_L`; convergence of the differentiated theta sum.') +
"noncomputable def dK (d : ℕ) (L : ℝ) (z : E d) (l : List (E d)) : ℝ :=\n  iteratedFDeriv ℝ l.length (periodicCovariance d L) z (fun i => l.get i)\n")
BLOCK.append(doc("The jet covariance of the stationary field, `Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K_L(x - y)` (hand-written convention: derivatives in the second argument of a stationary covariance pick up one sign each).",
    RW, A_REG, 'none; the sign and scale convention is this module\'s and is a reader item.', 'that this is the covariance of any field; symmetry of the resulting matrices.') +
"noncomputable def jetCovL (d : ℕ) (L : ℝ) (x : E d) (l1 : List (E d)) (y : E d) (l2 : List (E d)) : ℝ :=\n  (-1 : ℝ) ^ l2.length * dK d L (x - y) (l1 ++ l2)\n")
BLOCK.append("""/-- An observation of the field: a point and the list of differentiation directions (`[]` the value, `[v]` the derivative along `v`, `[v, w]` a second derivative).

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none.
Does not claim: anything; bookkeeping. -/
abbrev Obs (d : ℕ) : Type := E d × List (E d)

/-- The `2(d+1)` pin observations: the two values and the `2d` frame gradient components.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none.
Does not claim: anything; bookkeeping. -/
abbrev PinIdx (d : ℕ) : Type := Fin 2 ⊕ (Fin 2 × Fin d)

/-- The `(d+1)`-jet at a witness point: the value (`none`) and the `d` frame gradient components (`some i`). -/
abbrev JetIdx (d : ℕ) : Type := Option (Fin d)

/-- Upper-triangular index pairs, the coordinates of a symmetric `d × d` matrix. -/
abbrev SymIdx (d : ℕ) : Type := {{p : Fin d × Fin d // p.1 ≤ p.2}}

/-- Hessian coordinates at the three sites `M`, `S`, `x` (in that order). -/
abbrev HessIdx (d : ℕ) : Type := Fin 3 × SymIdx d

/-- The conditioning observations of the second Schur step: the pins and the witness jet. -/
abbrev CondIdx (d : ℕ) : Type := PinIdx d ⊕ JetIdx d

/-- The two pins `M = -(r/2) u`, `S = (r/2) u` as a `Fin 2`-indexed family. -/
noncomputable def pinSite (d : ℕ) (r : ℝ) (u : E d) : Fin 2 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u

/-- The three sites `M`, `S`, `x` of the Hessian coordinates. -/
noncomputable def hessSite (d : ℕ) (r : ℝ) (u x : E d) : Fin 3 → E d
  | 0 => pinM d r u
  | 1 => pinS d r u
  | 2 => x

/-- The pin observations as `Obs`: values at `M`, `S`; gradient components along the frame `R` at `M`, `S`.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none; the axial direction is `R 0`, which needs `[NeZero d]`.
Does not claim: anything; bookkeeping. -/
noncomputable def pinDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : PinIdx d → Obs d
  | Sum.inl a => (pinSite d r (R 0) a, [])
  | Sum.inr (a, i) => (pinSite d r (R 0) a, [R i])

/-- The witness `(d+1)`-jet at `x` as `Obs`. -/
noncomputable def jetDesc (d : ℕ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) : JetIdx d → Obs d
  | none => (x, [])
  | some i => (x, [R i])

/-- Pins and witness jet together. -/
noncomputable def condDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : CondIdx d → Obs d
  | Sum.inl a => pinDesc d r R a
  | Sum.inr a => jetDesc d x R a

/-- The Hessian coordinates at `M`, `S`, `x` as `Obs`: second derivatives along frame pairs. -/
noncomputable def hessDesc (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : HessIdx d → Obs d
  | (a, p) => (hessSite d r (R 0) x a, [R p.1.1, R p.1.2])

/-- The covariance matrix between two families of observations, by the stationary rule `jetCovL`.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none.
Does not claim: positive semidefiniteness; that it is the covariance of any random vector. -/
noncomputable def covOf (d : ℕ) (L : ℝ) {{ι κ : Type}} (f : ι → Obs d) (g : κ → Obs d) : Matrix ι κ ℝ :=
  Matrix.of fun a b => jetCovL d L (f a).1 (f a).2 (g b).1 (g b).2

/-- The pin targets: `f(M) = b`, `f(S) = b - k r^3`, gradients `0`.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none.
Does not claim: anything; the source's observations. -/
noncomputable def pinTarget (d : ℕ) (b k r : ℝ) : PinIdx d → ℝ
  | Sum.inl 0 => b
  | Sum.inl 1 => b - k * r ^ 3
  | Sum.inr _ => 0

/-- The witness-jet target `(y, 0)`: height `y`, gradient `0`. -/
noncomputable def jetTarget (d : ℕ) (y : ℝ) : JetIdx d → ℝ
  | none => y
  | some _ => 0

/-- Pin and witness-jet targets together. -/
noncomputable def condTarget (d : ℕ) (b k r y : ℝ) : CondIdx d → ℝ
  | Sum.inl a => pinTarget d b k r a
  | Sum.inr a => jetTarget d y a

/-- The observation covariance `Sigma_r` of the `2(d+1)` pin observations.

""" + src_line(RW) + f"""
Anchor: "{A_POS}"
Interface: none.
Does not claim: invertibility (a field of the successor interface, in the pin regime). -/
noncomputable def pinObsCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) : Matrix (PinIdx d) (PinIdx d) ℝ :=
  covOf d L (pinDesc d r R) (pinDesc d r R)

/-- The cross-covariance `v_r(x)` between the value at `x` and the pin observations. -/
noncomputable def pinCross (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : PinIdx d → ℝ :=
  fun a => jetCovL d L x [] (pinDesc d r R a).1 (pinDesc d r R a).2

/-- The Gaussian regression mean of `f(x)` given the pin observations, `v_r(x)^T Sigma_r^{{-1}} (targets)`, with the ordinary matrix inverse.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none; written with the ordinary inverse, the regression when `Sigma_r` is invertible (the field `pinObsCov_posDef`).
Does not claim: invertibility of `pinObsCov`; that this is the mean of any law. -/
noncomputable def regMean (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : ℝ :=
  dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinTarget d b k r))

/-- The Schur-complement covariance `K_L(x - y) - v_r(x)^T Sigma_r^{{-1}} v_r(y)` of the regression.

""" + src_line(RW) + f"""
Anchor: "{A_REG}"
Interface: none; written with the ordinary matrix inverse.
Does not claim: invertibility of `pinObsCov`; positive semidefiniteness of the resulting kernel. -/
noncomputable def regCov (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x y : E d) : ℝ :=
  periodicCovariance d L (x - y) -
    dotProduct (pinCross d L r R x) (Matrix.mulVec (pinObsCov d L r R)⁻¹ (pinCross d L r R y))

/-- The Hessian matrix of `g` at `x` in the frame `R`, from the module's bilinear `hessian`.

""" + src_line(RW) + f"""
Anchor: "{A_W}"
Interface: none.
Does not claim: symmetry (true for `C^2` functions, not stated). -/
noncomputable def hessMat (d : ℕ) (g : E d → ℝ) (x : E d) (R : OrthonormalBasis (Fin d) ℝ (E d)) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => hessian d g x (R i) (R j)

/-- The typed endpoint weight `W_r = F_d(H_M) F_(d-1)(H_S)` of a sample path: `|det H_M| |det H_S|` when `H_M` is negative
definite and `H_S` has index `d - 1` (through `HessianHasInertia`), `0` otherwise, hence `0` on singular matrices.

""" + src_line(RW) + f"""
Anchor: "{A_W}"
Interface: none.
Does not claim: that the frame-dependent determinants agree with the source's (they do for orthonormal frames; not stated). -/
noncomputable def typedWeight (d : ℕ) [NeZero d] (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (g : E d → ℝ) : ℝ :=
  Set.indicator {{g : E d → ℝ | HessianHasInertia d g (pinM d r (R 0)) d ∧ HessianHasInertia d g (pinS d r (R 0)) (d - 1)}}
    (fun g => |(hessMat d g (pinM d r (R 0)) R).det| * |(hessMat d g (pinS d r (R 0)) R).det|) g

/-- Gaussian density with covariance `C` and mean `m` at `v` on a finite index type, by the ordinary inverse and determinant (not a probability density where `C` is singular).

""" + src_line(RW) + f"""
Anchor: "{A_DENS}"
Interface: none.
Does not claim: integrability or normalisation; invertibility of `C`. -/
noncomputable def gaussDensityOn {{ι : Type}} [Fintype ι] [DecidableEq ι] (C : Matrix ι ι ℝ) (m v : ι → ℝ) : ℝ :=
  (2 * Real.pi) ^ (-(Fintype.card ι : ℝ) / 2) * C.det ^ (-(1 : ℝ) / 2) *
    Real.exp (-(dotProduct (v - m) (Matrix.mulVec C⁻¹ (v - m))) / 2)

/-- First Schur step: the covariance of the witness `(d+1)`-jet at `x` given the pins. -/
noncomputable def jetCovGivenPins (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (JetIdx d) (JetIdx d) ℝ :=
  covOf d L (jetDesc d x R) (jetDesc d x R) -
    covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹ *
      covOf d L (pinDesc d r R) (jetDesc d x R)

/-- First Schur step: the mean of the witness jet given the pin targets. -/
noncomputable def jetMeanGivenPins (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) : JetIdx d → ℝ :=
  Matrix.mulVec (covOf d L (jetDesc d x R) (pinDesc d r R) * (covOf d L (pinDesc d r R) (pinDesc d r R))⁻¹) (pinTarget d b k r)

/-- The conditional density `p_(Y_x | U_r = v_r)(0, y)` of the witness jet at `(y, 0)` given the pins.

""" + src_line(RW) + f"""
Anchor: "{A_DENS}"
Interface: none.
Does not claim: that it is a density of any law (it is the Gaussian formula at the regression parameters). -/
noncomputable def jetDensityAt (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) : ℝ :=
  gaussDensityOn (jetCovGivenPins d L r R x) (jetMeanGivenPins d L r b k R x) (jetTarget d y)

/-- Second Schur step: the covariance of the three Hessians (at `M`, `S`, `x`) given the pins and the witness jet. -/
noncomputable def hessCovGivenCond (d : ℕ) [NeZero d] (L r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) :
    Matrix (HessIdx d) (HessIdx d) ℝ :=
  covOf d L (hessDesc d r R x) (hessDesc d r R x) -
    covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹ *
      covOf d L (condDesc d r R x) (hessDesc d r R x)

/-- Second Schur step: the mean of the three Hessians given the pin and witness-jet targets. -/
noncomputable def hessMeanGivenCond (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (x : E d) (y : ℝ) : HessIdx d → ℝ :=
  Matrix.mulVec (covOf d L (hessDesc d r R x) (condDesc d r R x) * (covOf d L (condDesc d r R x) (condDesc d r R x))⁻¹)
    (condTarget d b k r y)

/-- The symmetric matrix at site `a` read off the Hessian coordinates (raw upper-triangular entries, no `sqrt 2` scaling). -/
noncomputable def symOfH (d : ℕ) (h : HessIdx d → ℝ) (a : Fin 3) : Matrix (Fin d) (Fin d) ℝ :=
  Matrix.of fun i j => if hij : i ≤ j then h (a, ⟨(i, j), hij⟩) else h (a, ⟨(j, i), le_of_not_ge hij⟩)

/-- `F_j(H) = |det H| 1{{H has j negative eigenvalues}}` for a matrix, the inertia read through a Sylvester decomposition
(a complementary pair of subspaces of dimensions `j`, `d - j` on which the form is negative, respectively positive, definite),
hence `0` on singular matrices.

""" + src_line(RW) + f"""
Anchor: "{A_FJ}"
Interface: none.
Does not claim: that the decomposition characterises the inertia of non-symmetric matrices. -/
noncomputable def typedF (d : ℕ) (H : Matrix (Fin d) (Fin d) ℝ) (j : ℕ) : ℝ :=
  Set.indicator {{H : Matrix (Fin d) (Fin d) ℝ | ∃ V W : Submodule ℝ (Fin d → ℝ), IsCompl V W ∧
      Module.finrank ℝ V = j ∧ Module.finrank ℝ W = d - j ∧
      (∀ v ∈ V, v ≠ 0 → dotProduct v (H.mulVec v) < 0) ∧ (∀ w ∈ W, w ≠ 0 → 0 < dotProduct w (H.mulVec w))}}
    (fun H => |H.det|) H

/-- The Kac-Rice integrand `W_r F_j(H_x) = F_d(H_M) F_(d-1)(H_S) F_j(H_x)` on the Hessian coordinates.

""" + src_line(RW) + f"""
Anchor: "{A_KR12}"
Interface: none.
Does not claim: anything; the source's integrand on coordinates. -/
noncomputable def kacRiceWeight (d : ℕ) (j : ℕ) (h : HessIdx d → ℝ) : ℝ :=
  typedF d (symOfH d h 0) d * typedF d (symOfH d h 1) (d - 1) * typedF d (symOfH d h 2) j

/-- The contact kernel `Λ'_j(x; b, k, R)(y)` for normalizer `Z`: `Z^{{-1}}` times the conditional jet density at `(y, 0)` times
the expectation of the Kac-Rice weight under the Gaussian law (`multivariateGaussian`, Mathlib) of the three Hessians given pins
and jet. This is the integrand of the source's weighted Kac-Rice identity (12), DEFINED here; the parent's free `Λ` (13) is its
`r → 0` limit object and is not tied to it.

""" + src_line(RW) + f"""
Anchor: "{A_KR12}"
Interface: none; `multivariateGaussian` is the Dirac mass at `0` for a covariance that is not positive semidefinite and a degenerate Gaussian on a singular one.
Does not claim: integrability; that it equals the source's kernel for any field; positivity or continuity. -/
noncomputable def contactKernel (d : ℕ) [NeZero d] (L r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (j : ℕ) (x : E d) (y : ℝ) (Z : ℝ) : ℝ :=
  Z⁻¹ * jetDensityAt d L r b k R x y *
    ∫ h, kacRiceWeight d j (fun a => h a)
      ∂(ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (hessMeanGivenCond d L r b k R x y)) (hessCovGivenCond d L r R x))

/-- Successor of `PinnedLawInterface` (numbered, the predecessor unchanged): the given law `Q_r` is characterised as the
Gaussian regression of the `K_L` field on the pin observations, the observation covariance is positive definite in the pin
regime, the given weight `W_r` IS the typed determinant weight of the sample path, the full normalizer has the source's
`z_* r^2` floor on compact windows, the critical set is almost surely finite and the counts are measurable, and the mean window
and all-height counts of a Borel region are given by the weighted Kac-Rice identity with the DEFINED contact kernel. With
these fields the laws, the weight and the mean counts are determined, so the round-4 refuting inhabitants (Dirac laws at a
pinned path with extra critical points) are excluded.

""" + src_line(RW) + f"""
Anchor: "{A_KR}"
Interface: every field is a given object or a hypothesis; nothing is derived. `Q_regression` characterises `Q` through finite vectors of point evaluations only (not sample-path jets), so `Q_pinned` stays a separate field; `pinObsCov_posDef` (source Section 4), `Z_floor` (lifetime-parent (5.5)), `critical_finite` (lifetime-parent Section 9) and the two Kac-Rice identities (source (12); d5-dimension-lift Section 4.7) are the sources' theorems assumed as fields, Mathlib `d13f23b7` having no Kac-Rice formula; the parent's kernel `Λ` stays free; the axial direction is `R 0`, which needs `[NeZero d]`.
Does not claim: existence of an object satisfying this interface; that `Q` is the regression law of any field or that `Q^W` is the sources' typed law (no field is constructed); the sign and scale conventions of `jetCovL`; integrability of the weight or of the counts; anything about the predecessor interface, which is unchanged. -/
structure PinnedLawInterface2 (d : ℕ) [NeZero d] (L : ℝ) (Ω : Type) [MeasurableSpace Ω]
    extends PinnedLawInterface d L Ω where
  /-- Section 4 clause "the covariances of (U_r,Y_x), their inverses, and the relevant Schur complements are uniformly bounded
  and positive": the observation covariance is positive definite in the pin regime, so the ordinary-inverse formulas are the
  regression. -/
  pinObsCov_posDef : ∀ (r : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L → (pinObsCov d L r R).PosDef
  /-- Section 1 clause "Q_r denotes continuous Gaussian regression on these 2(d+1) observations": under `Q`, every finite
  vector of point evaluations of the field is the multivariate Gaussian with the regression mean and the Schur-complement
  covariance. -/
  Q_regression : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (m : ℕ) (x : Fin m → E d),
      Measure.map (fun ω => WithLp.toLp 2 (fun i => F ω (x i))) (Q r b k R) =
        ProbabilityTheory.multivariateGaussian (WithLp.toLp 2 (fun i => regMean d L r b k R (x i)))
          (Matrix.of fun i j => regCov d L r R (x i) (x j))
  /-- Section 1 clause `W_r=F_d(H_M) F_(d-1)(H_S)`: the given weight is the typed determinant weight of the sample path. -/
  W_det : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)) (ω : Ω), W r b k R ω = typedWeight d r R (F ω)
  /-- lifetime-parent (5.5) `0 < z_* <= Z_r / r^2`: the full normalizer has a `z_* r^2` floor on compact windows `b ∈ B`,
  `k ∈ K` with `k > 0`, for `0 < r ≤ r_*` (a theorem of the source, assumed as a field). -/
  Z_floor : ∀ bLo bHi kLo kHi : ℝ, 0 < kLo → kLo ≤ kHi → bLo ≤ bHi →
    ∃ zStar : ℝ, 0 < zStar ∧ ∃ rStar : ℝ, 0 < rStar ∧ rStar < L ∧
      ∀ r : ℝ, 0 < r → r ≤ rStar → ∀ b ∈ Set.Icc bLo bHi, ∀ k ∈ Set.Icc kLo kHi,
        ∀ R : OrthonormalBasis (Fin d) ℝ (E d), zStar * r ^ 2 ≤ ∫ ω, W r b k R ω ∂(Q r b k R)
  /-- lifetime-parent Section 9 "Q is almost surely Morse": the critical set of a sample path in the fundamental domain is
  finite, `Q^W`-almost surely, in the pin regime. -/
  critical_finite : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ᵐ ω ∂(QW r b k R), {{x : E d | x ∈ FundDomain d L ∧ IsCriticalPoint d (F ω) x}}.Finite
  /-- The window and all-height counts are measurable functions of the sample for every Borel region. -/
  count_measurable : ∀ (b k r : ℝ) (j : ℕ) (A : Set (E d)), MeasurableSet A →
    Measurable (fun ω => (windowCount d (F ω) b k r j A : ℝ)) ∧ Measurable (fun ω => (allHeightCount d (F ω) j A : ℝ))
  /-- The weighted Kac-Rice identity (12): the tilted mean index-`j` window count of a Borel region of the fundamental domain
  is the integral of the contact kernel over the region and the height window (a theorem of the sources, assumed as a field). -/
  kacRice_window : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (windowCount d (F ω) b k r j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y in Set.Ioo (b - k * r ^ 3) b, contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))
  /-- The all-height analogue (d5-dimension-lift Section 4.7): the height integral runs over all of `ℝ`. -/
  kacRice_allHeight : ∀ (r b k : ℝ) (R : OrthonormalBasis (Fin d) ℝ (E d)), 0 < r → r < L →
    ∀ (j : ℕ) (A : Set (E d)), MeasurableSet A → A ⊆ FundDomain d L →
      ∫ ω, (allHeightCount d (F ω) j A : ℝ) ∂(QW r b k R) =
        ∫ x in A, ∫ y, contactKernel d L r b k R j x y (∫ ω, W r b k R ω ∂(Q r b k R))
""")

# ----------------------------------------------------------------------------- the 19 successor Props, generated from the predecessors' headers
def binders(hdr_text):
    """explicit binder names in order (instance binders skipped)"""
    out = []
    for grp in re.findall(r'\(([^()]*?) : [^()]*?(?:\([^()]*\)[^()]*)*\)', hdr_text):
        out.extend(grp.split())
    return out
succ_rows = []
for p in preds:
    name = p['name']; sname = name + 'B'
    hdr = '\n'.join(p['hdr'])
    args = binders(hdr); assert 'I' in args, (name, args)
    app = ' '.join(a if a != 'I' else 'I.toPinnedLawInterface' for a in args)
    new_hdr = hdr.replace(f'def {name} ', f'def {sname} ', 1).replace('(I : PinnedLawInterface ', '(I : PinnedLawInterface2 ', 1)
    assert new_hdr.count('PinnedLawInterface2') == 1 and 'def ' + sname in new_hdr, name
    if name == 'WitnessCollisionNearPinPairsTarget':
        summary = ('Successor of `WitnessCollisionNearPinPairsTarget` over the successor interface: the OPEN target (stated as a target and not as a result) over any object '
                   'satisfying `PinnedLawInterface2`. The source marks this target OPEN; no field set makes it provable, and the successor is registered `specified` with that said.')
    else:
        first = p['first']; summ = first.split(':', 1)[0].rstrip(','); summ = summ[0].lower() + summ[1:]
        summary = (f'Successor of `{name}` over the successor interface: {summ}, for any object satisfying `PinnedLawInterface2`, whose fields determine the laws, the '
                   'weight and the mean counts and exclude the round-4 refuting inhabitant.')
    iface = (f'`I : PinnedLawInterface2 …` is given; the Prop is `{name}` applied to the projection `I.toPinnedLawInterface`, so every object of the predecessor\'s reading is the '
             f'successor\'s (predecessor: {p["Interface:"]})')
    dnc = f'the result (statement only); existence of an inhabitant; anything about `{name}`, which is unchanged; the predecessor\'s own exclusions: {p["Does not claim:"].rstrip()}'
    dnc = dnc[:-3].rstrip() if dnc.endswith(' -/') else dnc
    if name == 'WitnessCollisionNearPinPairsTarget':
        dnc = ('that this target holds, is attainable, or is implied by any reviewed source or by any field of the successor interface (the source marks it OPEN); ' + dnc)
    docs = doc(summary, p['sid'], p['succ_anchor'], iface, dnc)
    BLOCK.append('\n' + docs + new_hdr + '\n  ' + name + ' ' + app + '\n')
    succ_rows.append({'pred': name, 'declaration': NS + sname, 'kind': 'def_prop', 'anchor': p['succ_anchor'], 'does_not_claim': dnc})

app_text = ''.join(BLOCK)
assert app_text.count('<<') == 0
(H / 'RN_append.lean').write_text(app_text, encoding='utf-8')
i = landed.rfind('\nend '); assert landed[i + 1:] == 'end UniversalLaw.Spec.RN\n'
draft = landed[:i] + app_text + '\n' + landed[i + 1:]
(H / 'RN_draft.lean').write_text(draft, encoding='utf-8')

# ----------------------------------------------------------------------------- gate scan + provisional register
sources, _ = check.load_sources(PK, None, reg)
MODULES = list(check.MODULES)
scans = {m: check.scan_module((draft if m == MODULE else (PK / m).read_text(encoding='utf-8')), PurePosixPath(m).stem, sources) for m in MODULES}
rn = scans[MODULE]; old_rn = check.scan_module(landed, 'RN', sources)
assert [ (r['name'], r['kind']) for r in rn[:len(old_rn)] ] == [ (r['name'], r['kind']) for r in old_rn ], 'committed prefix of the inventory changed'
new = rn[len(old_rn):]
struct_row = next(r for r in new if r['kind'] == 'structure'); assert struct_row['name'] == NS + 'PinnedLawInterface2'
assert struct_row['doc_anchor'] == A_KR
_sdoc = app_text[:app_text.index('structure PinnedLawInterface2 (d : ℕ)')]; _sdoc = _sdoc[_sdoc.rindex('/-- '):]
STRUCT_DNC = next(l for l in _sdoc.splitlines() if l.startswith('Does not claim: '))[len('Does not claim: '):]
assert STRUCT_DNC.endswith(' -/'); STRUCT_DNC = STRUCT_DNC[:-3].rstrip()
# proposed rows: structure after PinnedLawInterface in every claim row registering it; each successor after its predecessor
by_decl_anchor = {r['name']: r for r in new if r['kind']}
reg2 = copy.deepcopy(reg); claim_rows = {}
for c in reg2['claims']:
    sts = c['statements']; out = []
    for s in sts:
        out.append(s)
        if s['package'] == PACKAGE and s['module'] == MODULE:
            short = s['declaration'].split('.')[-1]
            if short == 'PinnedLawInterface':
                sc = struct_row
                row = {'package': PACKAGE, 'module': MODULE, 'declaration': sc['name'], 'kind': 'structure', 'formal_progress': 'specified', 'anchor': sc['doc_anchor'], 'does_not_claim': STRUCT_DNC}
                out.append(row); claim_rows.setdefault(c['claim_id'], []).append(row)
            for sr in succ_rows:
                if sr['pred'] == short:
                    sc = by_decl_anchor[sr['declaration']]
                    assert sc['doc_anchor'] == sr['anchor'], sr['declaration']
                    row = {'package': PACKAGE, 'module': MODULE, 'declaration': sr['declaration'], 'kind': 'def_prop', 'formal_progress': 'specified', 'anchor': sr['anchor'], 'does_not_claim': sr['does_not_claim']}
                    out.append(row); claim_rows.setdefault(c['claim_id'], []).append(row)
    c['statements'] = out
stats = check.check_register(reg2, scans, sources)
n_new = sum(len(c['statements']) for c in reg2['claims']) - sum(len(c['statements']) for c in reg['claims'])
inv = Counter(r['kind'] for r in rn)
out_lines = [f'statement gate scan_module (check.py of Math- 7f5f1c82, sources loaded from the packet) on RN_draft.lean',
             f'committed RN.lean sha256 {hashlib.sha256(landed.encode()).hexdigest()} bytes {len(landed.encode())}',
             f'draft RN_draft.lean sha256 {hashlib.sha256(draft.encode()).hexdigest()} bytes {len(draft.encode())}',
             f'committed decls {len(old_rn)} {dict(Counter(r["kind"] for r in old_rn))}', f'draft decls {len(rn)} {dict(inv)}',
             f'committed prefix of the inventory identical: True', f'check_register on the provisional register (+{n_new} statement entries): {stats}']
for r in new: out_lines.append(f"line {r['line']:5d} {('structure' if r['kind']=='structure' else 'def_prop' if r['kind']=='def_prop' else 'aux'):9s} {r['name']}  source={r.get('source_id')}  anchor={r.get('doc_anchor')!r}")
(H / 'scan_draft.txt').write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
json.dump({'status': 'PROPOSED, NOT APPLIED: design input for the RN slice (third round-7 slice); nothing is written to any packet before routing/activation.',
           'base': 'Math- main 7f5f1c82 (RN.lean unchanged since 8acc4856)', 'claim_rows': claim_rows,
           'claim_row_text_edits_proposed': 'each affected row\'s not_established gains one sentence naming the successor interface, the fields that are the sources\' theorems assumed (pinObsCov_posDef, Z_floor, critical_finite, kacRice_window, kacRice_allHeight), the point-evaluation-only scope of Q_regression, the hand-written derivative-covariance convention, and that no inhabitant is constructed; next_step names the successor Props as the row\'s targets (BLOCKED_MISSING_MATHLIB, weeks each; the Witness target stays OPEN).',
           'scientific_effect': 'NONE'}, open(H / 'register_rows_proposed.json', 'w'), indent=1, ensure_ascii=False)
print('\n'.join(out_lines[:7])); print('new registered-kind decls:', sum(1 for r in new if r['kind']), 'new aux:', sum(1 for r in new if not r['kind']), 'rows proposed:', n_new, 'claim rows touched:', len(claim_rows))
