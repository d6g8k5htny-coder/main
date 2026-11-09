# Formal package: SIDE24 arithmetic skeleton (Lean 4, core only)

Scientific effect: **NONE**. This directory is the second package in the
repository's formal-verification lane. The lane's contract is the
[formal-verification guide](../docs/FORMAL_VERIFICATION.md); the lane is
coordinated in [work item #95](https://github.com/d6g8k5htny-coder/main/issues/95);
the first package is the
[Math- pilot](https://github.com/d6g8k5htny-coder/Math-/tree/cc2989c1280f4f227d0c6aa30c8841d6ba01e46e/formal)
(13 GP-FOR-192 scalar companions, Lean 4 + Mathlib). This package follows the same
manifest, status, gate and alignment contract and adds nothing to the
scientific-status vocabulary.

What it covers: the exact integer and rational identities and inequalities that
sections 1–4 of the [SIDE24 coefficient note](sources/side24_v1/PROOF.md)
(`Math-@9d7b6802…`, SHA-256 `c06dacc…`) rely on — 31 theorems, stated in core
Lean as cross-multiplied integer facts and proved by `decide`. What it does not
cover is listed target by target in [SCOPE.md](SCOPE.md): not the coefficient
bound `0.07340691930603427103 < c_{2,24} < 0.07340691930603427104`, not the
definition of the coefficient, not any Gaussian, Kac–Rice, cone-integral,
Stirling or interval-arithmetic step, not `coefficient.py`. Relative to the
guide's work-offer table this is the **first step only** of the "Numerical
formalizer" offer; the exact coefficient definition and the proof-producing
enclosure pipeline are not supplied.

## Layout

| Path | Role |
|---|---|
| `lean-toolchain`, `lakefile.toml`, `lake-manifest.json` | Pinned toolchain `leanprover/lean4:v4.34.1`; Lake package `UniversalLaw` with **no** dependencies (`dependency_revisions` is empty and the gate checks the lock agrees) |
| `UniversalLaw.lean` | Root module; imports exactly the three registered modules |
| `UniversalLaw/Side24/Ledger.lean` | 17 targets: pairing count, image constant, `60E < 10^-108`, Taylor partial sum of `exp(288/125)`, odd-block minors, section-4 exponent bounds, density-comparison order of magnitude |
| `UniversalLaw/Side24/ConeMoments.lean` | 10 targets: conditional variances, transverse covariance thirds, `D_1 = 4/3`, `D_2 = 29/6 − √6` algebra, cube-root simplification, Hessian block eigenvalues |
| `UniversalLaw/Side24/Endpoints.lean` | 4 targets: the published 20-digit endpoints as integers — adjacency, ordering, magnitude. **Not an enclosure of the coefficient.** |
| `sources/side24_v1/` | Byte copies of the two pinned Layer 0 sources (`PROOF.md`, `ENCLOSURE.json`) |
| `manifest.json` | Evidence sidecar: package identity, toolchain, allowed axioms, SHA-256 of every bound file, pinned sources, 31 targets with verbatim informal anchors and `does_not_claim`, executable negative controls |
| `SCOPE.md` | Exact coverage and "not established" per target; the review contract |
| `ALIGNMENT.md` | Generated from the manifest (`--write-alignment`); never hand-edited |
| `GLOSSARY.md` | Descriptive mapping of project terms to standard objects and Lean availability |
| `REVIEW_LANE.md`, `reviews/` | Independent statement-alignment review procedure, JSON record template, validator |

## Status vocabulary (same as the Math- lane)

- `formalization_status` in `manifest.json` is **`proved`**: proof text is
  supplied and bound by hash. The manifest cannot say `kernel-checked`; the gate
  refuses a manifest that self-awards execution.
- **`kernel-checked`** appears only in the receipt written by a trusted
  `--run-lean` execution (`formal/.lake/formal-evidence/receipt.json`), which
  records the checked Git commit, Lean version, manifest digest, per-target
  transitive axioms (all `[]` here), log hashes, and negative-control outcomes.
  A receipt supplied by a caller is untrusted until matched to the actual
  workflow run and commit.
- `alignment_status` is **`PENDING_INDEPENDENT_REVIEW`**. The author is
  Anthropic / Claude via a Cursor cloud agent (2026-09-27); an alignment record
  from the same provider, family or agent is refused by the validator. The
  manifest cannot say `ACCEPTED`.
- In the ladder of [#95](https://github.com/d6g8k5htny-coder/main/issues/95)
  (L0 prose … L5 proof-assistant-checked, `verification_level` separate from
  scientific status) a target with a trusted `kernel-checked` receipt is L5
  evidence for exactly that target.
- None of these labels is a Layer 0 ACCEPT/AMEND verdict or moves one.

## Run it

Toolchain (once, ~200 MB, no Mathlib cache needed):

```sh
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
```

From the repository root:

```sh
python3 tools/formal_gate_check.py               # source-only: hashes, inventory, anchors, no Lean run
python3 tools/formal_gate_check.py --run-lean    # fresh build, leanchecker, axiom audit, controls, receipt
python3 -B -S -m unittest tests.test_formal_gate -v
```

The source-only gate prints `SOURCE_IDENTITY_PASS (not a Lean build or
scientific acceptance): <manifest digest>`. `--run-lean` removes `.lake/build`,
runs `lake build`, `leanchecker` on every module, a transitive `#print axioms`
audit restricted to `propext` / `Classical.choice` / `Quot.sound`, an
elaborated-type capture, and the executable negative controls: seven
tightened-or-wrong statements written into a scratch copy of the Lean text
(each must be rejected by Lean), plus `sorry`, an indirectly imported custom
axiom, and `native_decide` (each must be rejected by the axiom gate). It then
re-runs the source check to confirm the manifest digest did not move, and only
then writes the receipt. Locally the whole run takes about five seconds.

Maintenance (these write): `--refresh-hashes` after editing any bound file,
then `--write-alignment` after editing targets, then the gate again.

The gate fails closed on: hash mismatch; an unregistered `.lean` file or any
unbound file inside `formal/`; a bound file that does not exist; a control or
scope file missing from the bound set; toolchain or dependency-lock drift;
`sorry`, `native_decide` or an `axiom` declaration outside comments; a root
module that imports anything but the registered modules; a target inventory
that differs from the theorem names actually declared (in module order); a
pinned source whose local bytes changed or whose repository is not one of the
project's public repositories; an anchor that is not a verbatim substring; a
missing `does_not_claim`; a stale `ALIGNMENT.md`; a manifest with duplicate JSON
keys.

## Why core Lean and not Mathlib here

Everything the arithmetic skeleton needs is `Nat`/`Int` and `decide`, which the
kernel evaluates with GMP-backed literals in well under a second. That keeps
this package's CI cost at seconds with no multi-gigabyte cache. The Math- pilot
already carries the Mathlib dependency (revision
`d13f23b723b8a846827a245b89c10fc7d3f11612`, same Lean `v4.34.1`); any statement
here that needs `ℝ`, `Real.Gamma`, `Real.exp`, `Real.pi`, Gaussian measures or
matrices — the coefficient definition `c_{d,ref}`, the enclosure itself, the
P15 constant `ρ*` — belongs in that Mathlib lane, not in a second dependency
tree. See the rollout record for the coordination items.

## Coordination

- [docs/FORMAL_VERIFICATION.md](../docs/FORMAL_VERIFICATION.md) — lane contract
  and work offers.
- [docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md](../docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md)
  — what this package adds, how it converged onto the Math- contract, what is
  asked of other repositories and agents, and this session's capability limits.
- [Work item #95](https://github.com/d6g8k5htny-coder/main/issues/95) — record
  actual pickup of alignment review or of the remaining "Numerical formalizer"
  steps there.

## What this package does not do

It does not change `STATUS.md`, `PROOF_INDEX.md`, `LANDING_CLAIMS` or any
ACCEPT/AMEND outcome. It does not turn a hash match, a green build or a
kernel-checked lemma into acceptance of the theorem built on it. It does not
verify `coefficient.py`. It does not make AI-authored Lean text independent;
the author label stays until a distinct reviewer's record validates. It is not
a second scientific-status database and adds no status vocabulary beyond the
lane's.
