# Formal verification layer (Layer 1)

This directory adds machine-checked logic to the existing provenance-and-review
system without replacing any of it. The existing system — SHA-256 and Git-object
provenance, explicit scope and "does not claim" statements, fail-closed CI, and
source-bound nonauthor review — is **Layer 0**. This directory is **Layer 1**: Lean 4
statements and proofs that the Lean kernel checks, bound by hash to the exact
Layer 0 bytes they formalise.

Adopted under the [current owner instruction](../governance/OP-AUTONOMY-20260923-v2.1.md);
the rollout record is [docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md](../docs/FORMAL_VERIFICATION_ROLLOUT_20260927.md).

## What is in this directory

| Path | Role |
|---|---|
| `lean-toolchain`, `lakefile.toml`, `lake-manifest.json` | Pinned Lean 4 toolchain (`v4.34.1`) and a dependency-free Lake project |
| `UniversalLaw.lean`, `UniversalLaw/**/*.lean` | The library. Every theorem is kernel-checked by `lake build` |
| `UniversalLaw/Audit.lean` | `#print axioms` for every registered declaration |
| `registry.json` | Per-claim **formalization status**, Lean file hashes, byte-pinned informal anchors, axiom allow/deny lists, alignment-review lane, AI-prover cross-check lane |
| `ALIGNMENT.md` | Generated informal–formal pairing table (from the registry; never hand-edited) |
| `sources/` | Byte copies of the exact Layer 0 sources the Lean text is attached to, pinned by commit, size and SHA-256 |
| `GLOSSARY.md` | Project terminology mapped to standard mathematical objects and (where they exist) Lean names |
| `REVIEW_LANE.md`, `reviews/` | The formalization (statement-alignment) review lane and its records |

## Formalization status levels

Each claim in `registry.json` carries exactly one of:

| Status | Meaning |
|---|---|
| `none` | No Lean object. The claim lives only in Layer 0 prose. |
| `specified` | A Lean statement compiles; no proof is offered. Its hypotheses are the exact claimed scope. |
| `proved` | A Lean proof compiles but depends on project-declared axioms (an unformalised parent taken as an explicit `axiom` with a scope note) or on axioms outside the allow-list. |
| `kernel-checked` | A Lean proof compiles and `#print axioms` reports only `propext`, `Classical.choice`, `Quot.sound` — or nothing. No `sorry`, no `native_decide`, no project axioms. |

Separately, each Lean statement carries an **alignment review** status: `open`
(author-side; nobody but the author has checked that the Lean statement says
what the informal text says) or `reviewed` (a distinct reviewer recorded that
check under `reviews/`). The kernel guarantees the proof; the alignment review
guarantees the statement is the intended one. Both are needed; neither changes
a Layer 0 review outcome.

## Current pilot: SIDE24 arithmetic skeleton

Pilot object: the [SIDE24 coefficient note](sources/side24_v1/PROOF.md)
(`Math-@9d7b6802…`, SHA-256 `c06dacc…`), the D3 row of [STATUS.md](../STATUS.md).

**Kernel-checked (31 declarations, zero axioms):** every exact integer or rational
identity and inequality that sections 1–4 of the note rely on, stated as
cross-multiplied integer facts because core Lean has no rational type. Examples:
the pairing count `76`; the image constant `1458·(76·24^6+15) = 21175738586478`;
`60E < 10^-108` as `60·21175738586478 < 10^17`; the degree-20 Taylor partial sum of
`exp(288/125)` exceeding `10`; the odd-block minors `2/3`, `7/9` and
`8 − √58 > 1/3` as `23² > 58·9`; the section-4 exponent bounds `a+2b<14`, `2a+b<13`
for `d = 2, 3` (the `d = 3` case is `77 < 78`, so this is tight); the shared trace
variance `5/3`, `E s⁴ = 25/3`, the `D_2 = 29/6 − √6` algebra, and the cube-root
simplification `6^(2/3)/24^(1/3) = (3/2)^(1/3)`; the adjacency of the published
20-digit endpoints.

**Not formalised (status `none`):** the coefficient bound itself,
`0.07340691930603427103 < c_{2,24} < 0.07340691930603427104`, and every analytic
step — Gaussian conditioning, the cone integral, the derivative bound on
`exp(−|x|²/2)`, the positive-semidefinite covariance comparison, the Stirling
remainder for `Γ(7/6)`, and the `π`, `log`, `exp` enclosures in `coefficient.py`.
Those remain Layer 0 prose under nonauthor review. A kernel-checked arithmetic
skeleton is **not** acceptance of the argument built on it.

Every kernel-checked declaration is paired in `registry.json` with a verbatim
quote (`informal_anchor`) from the pinned source bytes; the gate fails if the
quote is not a substring of those bytes. This is the alignment mechanism until
a Lean Blueprint build is adopted (see below).

## Run it

Toolchain (once; ~200 MB, no Mathlib):

```sh
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
export PATH="$HOME/.elan/bin:$PATH"
```

Build, audit, gate, controls (from the repository root):

```sh
(cd formal && lake build && lake env lean UniversalLaw/Audit.lean > ../audit.txt)
python3 tools/formal_gate_check.py --axioms-output audit.txt
python3 -B -S -m unittest tests.test_formal_gate -v
```

`python3 tools/formal_gate_check.py --run-lean` does the build and audit itself.
`--static-only` skips the kernel lane and says so in its output. Without one of
the three the gate refuses to run: the axiom lane is required by default.

The gate fails closed on: a Lean file whose SHA-256 differs from the registry; a
Lean file present but not registered; `sorry`, `native_decide`, or an `axiom` not
listed in `project_axioms` with a scope note (comments are stripped first — the
kernel's `sorryAx` check is the authoritative guard); a pinned source whose local
copy differs; an informal anchor that is not a verbatim substring; a declaration
missing from its file; a `kernel-checked` declaration missing from the audit or
depending on a non-allowed axiom; any declaration depending on `sorryAx` or
`Lean.ofReduceBool`; an alignment table that drifted from the registry; a
`reviewed` alignment without a reviewer and an existing record file.

Maintenance commands (these write): `--refresh-hashes` after editing Lean files;
`--write-alignment` after editing the registry.

## Why core Lean and not Mathlib yet

The arithmetic skeleton needs only `Nat`/`Int` and `decide`; the kernel evaluates
these with GMP-backed literals in well under a second, so the whole formal lane
adds seconds, not minutes, to CI and needs no multi-gigabyte cache. Mathlib is
required the moment a statement mentions `ℝ`, `Real.Gamma`, `Real.exp`, `Real.pi`,
Gaussian measures or matrices; that is the next expansion and it changes the CI
cost profile (cache download, longer timeouts), so it is a separate, reviewable
step rather than something bundled into the pilot.

## Expansion path

1. **Mathlib lane.** Add `mathlib` to `lakefile.toml` at a pinned revision, use
   `lake exe cache get` in CI, and move the toolchain to Mathlib's. First targets
   that become *statable*: the reference coefficient `c_{d,ref}` (display (1) of
   the note); the P15 constant `ρ* = 1/(3 − log(3e − 2))` with its published
   20-digit enclosure and `ρ* < 6/7` (Theorem F, D6 — finite combinatorics plus one
   `Real.log`, so the most tractable full-theorem target); the interval-arithmetic
   enclosures of `coefficient.py` re-derived with `norm_num`/`interval_cases`.
2. **`specified` statements before proofs.** State D6 Theorem F and the SIDE24
   coefficient bound as Lean `def … : Prop` with hypotheses that are exactly the
   Layer 0 scope. Anything not in the hypotheses is not claimed.
3. **Parent dependencies as explicit axioms.** Where a proof needs an unreviewed
   parent (for example the D1 selection chain), state it as an `axiom` listed in
   `project_axioms` with a scope note. The result is then `proved`, never
   `kernel-checked`, until the parent is itself formalised.
4. **Lean Blueprint.** Replace the anchor-substring mechanism with a
   `leanblueprint` document (LaTeX prose + `\lean{}` references, dependency graph,
   `\leanok` flags). The registry's `informal_anchor` fields are the seed text.
5. **AI-prover cross-check lane.** Run Goedel-Prover / AlphaProof-style systems
   on `specified` statements. A success is recorded in `registry.json` under
   `cross_checks` with prover, version, date, declaration and record path. It is
   still checked by the same kernel, is still author-side until aligned, and does
   not replace the alignment review.
6. **External validation.** Papers written in the standard vocabulary of
   [GLOSSARY.md](GLOSSARY.md), with the Lean files as supplementary material.

## What this layer does not do

It does not change `STATUS.md`, `PROOF_INDEX.md`, `LANDING_CLAIMS`, or any
ACCEPT/AMEND outcome. It does not turn a hash match, a green build, or a
kernel-checked lemma into mathematical acceptance of a theorem. It does not
verify the Python enclosure code. It does not make AI-authored Lean text
independent: the author label stays until a distinct reviewer records the
alignment check. It cannot verify the Mathlib-dependent statements it lists as
`none`; it names them so they are visible targets, not silent gaps.
