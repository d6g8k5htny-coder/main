# Round 7, third slice (RN successor pinned-law interface): design-review packet, DRAFT, not proposed yet

Status: prepared while the second slice (Lifetime, main#307 6137849133) waits on the coordinator's routing; NOT proposed and
NOT activated. Nothing here is a packet change: Math- is untouched for this slice. This directory is the exact object a design
read would take. Author: Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (author of the statement packet).
Organizational-independence credit 0. Scientific effect NONE. Every declaration would enter as `specified`; no theorem, no
inhabitant constructed, no label moved. Base: `Specifications/RN.lean` at Math- main `7f5f1c82` (unchanged since `8acc4856`).

## Files

| file | what it is |
|---|---|
| `gen_rn_append.py` | the generator: assembles the block from hand-written Lean (auxiliaries, successor interface) and the 19 successor Props generated from the predecessors' headers; every anchor is sliced from the pinned source text and asserted verbatim; runs the gate's `scan_module` and `check_register` on a provisional register |
| `RN_append.lean` (sha256 `e93e648e6665f101259fff058a6643dc490ead7e5953b860910ae608e1bd0361`) | the block to be appended to `Specifications/RN.lean` before its `end` line: 36 auxiliaries, `structure PinnedLawInterface2 … extends PinnedLawInterface` (8 new fields), 19 `def … : Prop` successors (suffix `B`), four-label docstrings, verbatim anchors distinct from the predecessors' |
| `RN_draft.lean` (sha256 `2c40be8579427492ab7e595e1120b8007514667cb9f2de3b1d86f8852afc87a2`) | the full draft module (landed `RN.lean` minus its `end` line + the block + `end`); prefix byte-identical |
| `elab_draft.log` | `lake env lean RN_draft.lean` at the pin (Lean 4.34.1 `5045d005`, Mathlib `d13f23b7`, `LEAN_NUM_THREADS=2`): exit 0, no output, real 0m19.309s |
| `scan_draft.txt` | the gate's own `scan_module` on the draft: 108 declarations = the committed 52 (27 Props, 1 structure, 24 auxiliaries; identical prefix) + 56 new (36 auxiliary, 1 structure, 19 `def_prop`); `check_register` on the provisional register passes (407 statement entries, 177 registered local statements) |
| `register_rows_proposed.json` | the 41 statement rows that would be added across 20 claim rows (the structure after `PinnedLawInterface` wherever it is registered; each successor after its predecessor wherever the predecessor is registered), NOT applied |
| `RN_landed.lean` | the landed module, for the byte comparison |

## What the successor pins (fields of `PinnedLawInterface2`, all given objects or hypotheses, nothing derived)

| field | source clause | formal shape | source's object or added assumption? |
|---|---|---|---|
| `pinObsCov_posDef` | rn-fixed-remote-window §4 "the covariances of (U_r,Y_x), their inverses, and the relevant Schur complements are uniformly bounded and positive" | `(pinObsCov d L r R).PosDef` in the pin regime | the source's theorem assumed; makes the ordinary-inverse formulas the regression |
| `Q_regression` | §1 "Q_r denotes continuous Gaussian regression on these 2(d+1) observations" | `Measure.map (point evaluations) Q = multivariateGaussian (regMean) (regCov)` for every finite family of points | the definition of `Q`, through point evaluations only (so `Q_pinned` stays a field) |
| `W_det` | §1 `W_r=F_d(H_M) F_(d-1)(H_S)` | `W r b k R ω = typedWeight d r R (F ω)` | the definition of `W` (replaces the free `W` + `W_typed` by a tie to the defined weight; the originals stay) |
| `Z_floor` | lifetime-parent (5.5) `0<z_*<=Z_r/r^2` | `∃ z_* > 0, ∃ r_* ∈ (0, L), ∀ r ≤ r_*, b ∈ B, k ∈ K, R: z_* r^2 ≤ ∫ W dQ` on compact windows | a theorem of the source assumed as a field |
| `critical_finite` | lifetime-parent §9 "Q is almost surely Morse" | the critical set in the fundamental domain is finite `Q^W`-a.s. in the pin regime | a theorem of the source assumed |
| `count_measurable` | none (a formal necessity) | the window and all-height counts are measurable for every Borel region | added assumption |
| `kacRice_window` | rn-fixed-remote-window (12) | `∫ windowCount dQ^W = ∫_A ∫_{(b-kr^3,b)} contactKernel … (∫ W dQ)` for Borel `A ⊆ FundDomain` | the weighted Kac-Rice identity, a theorem of the sources assumed (Mathlib `d13f23b7` has no Kac-Rice formula), with the kernel DEFINED |
| `kacRice_allHeight` | d5-dimension-lift §4.7 | the same with the height integral over `ℝ` | likewise |

The contact kernel `contactKernel d L r b k R j x y Z = Z⁻¹ · p_(jet at x | pins)(y, 0) · E[F_d(H_M) F_(d-1)(H_S) F_j(H_x) | pins, jet]`
is defined from the regression law's Gaussian parameters with two Schur steps (`jetCovGivenPins`/`jetMeanGivenPins`;
`hessCovGivenCond`/`hessMeanGivenCond`), a Gaussian density on a finite index type (`gaussDensityOn`), the Sylvester-inertia
indicator `typedF` on matrices and Mathlib's `multivariateGaussian`. The parent's free kernel `Λ` (Theorem A's limit object
as `r → 0`) is NOT tied to it; the mean measures themselves become determined instead.

## The 19 successor Props

Each `XB` is `X` applied to the projection `I.toPinnedLawInterface` of an `I : PinnedLawInterface2`, with the predecessor's
parameters unchanged (generated from the predecessor's header). Anchors are verbatim lines of the same pinned source,
distinct as strings from the predecessors' anchors (the renderer matches proof-packet theorems to Props by anchor equality).
`WitnessCollisionNearPinPairsTargetB` is the successor of the OPEN target: its docstring and row say that the source marks it
OPEN and that no field set makes it provable.

Why the successor removes the known blocker: the round-4 triage refuted each predecessor's universal closure with a Dirac law
at a pinned path carrying extra critical points (the Frontier rows). Under `PinnedLawInterface2`, `Q` is the Gaussian
regression law (`Q_regression`), `W` the typed weight (`W_det`) and the mean counts are the Kac-Rice integrals
(`kacRice_window`, `kacRice_allHeight`), so those inhabitants no longer exist. This says nothing about the truth of the displays.

## Caveats (each is in the docstrings)

1. The sign and scale convention `Cov(∂_{l1} f(x), ∂_{l2} f(y)) = (-1)^{|l2|} ∂_{l1 ++ l2} K(x - y)` of `jetCovL` is
   hand-written; a slip there passes every gate. Reader item.
2. The Hessian coordinates are the raw upper-triangular entries (no `sqrt 2` scaling), consistent with `symOfH`.
3. `Q_regression` characterises `Q` through finite vectors of point evaluations only, not through sample-path jets; so
   `Q_pinned` stays a field.
4. `Z_floor`, `critical_finite` and the two Kac-Rice identities are theorems of the sources assumed as fields; `does_not_claim`
   names them. `pinObsCov_posDef` likewise.
5. `multivariateGaussian` is the Dirac mass at `0` for a covariance that is not positive semidefinite and a degenerate Gaussian
   on a singular one; the kernel's second Schur step has no PosDef field (its covariance is a given formula), which is why
   the kernel is a definition and the identities are the fields.
6. Integrability of the weight and of the counts, and measurability of the critical set, are not stated beyond
   `count_measurable` and `critical_finite` (Mathlib's measure and integral are total).
7. `typedF` reads the inertia through a Sylvester decomposition (complementary subspaces of dimensions `j`, `d - j`); for a
   non-symmetric matrix this is not the eigenvalue count. The Hessian coordinates are symmetric by construction of `symOfH`.
8. `periodicCovariance` duplicates `Lifetime.lean`'s definition per the packet's convention (RN.lean had no kernel).

## Verification performed (author, local, not a receipt)

- Generator assertions: every anchor verbatim in its pinned source and distinct from the predecessor's; the committed
  inventory prefix identical; every successor header carries exactly one `PinnedLawInterface2` binder.
- Statement gate `scan_module` on the full draft with the packet's pinned sources and `check_register` on the provisional
  register (landed register + the proposed rows): pass; see `scan_draft.txt`.
- Elaboration at the pin: `lake env lean RN_draft.lean` exit 0, no output (no error, no warning, no `sorry`), real 0m19.309s
  (`elab_draft.log`). A single-file elaboration of the draft module, not a build receipt.
- Not run (needs the packet mutation no routing has authorised): `check.py` source mode with the rows applied, the
  docs re-render, `test_check.py`, `--execute`, hosted workflows.

## Scope if activated (four files, append only; serial after the Lifetime slice or rebased onto whatever lands first)

`Specifications/RN.lean` (append), `REGISTER.json` (41 rows across 20 claim rows and one sentence each in those rows'
`not_established` / `next_step`), `README.md`, `SCOPE.md` (re-rendered: +19 Props, +1 structure, +36 auxiliaries; frontier
+19 open-target rows with the 19 RN interface-blocked rows pointing at their successors). A binder in the style of
`bind_lifetime.py` would be written against the then-current base. What it does not establish: nothing mathematical; every
successor stays `specified` and Mathlib-blocked (weeks each: first-moment bounds need the sources' Kac-Rice estimates over the
defined kernel, second moments the marked Kac-Rice for pairs); the Witness target stays OPEN.
