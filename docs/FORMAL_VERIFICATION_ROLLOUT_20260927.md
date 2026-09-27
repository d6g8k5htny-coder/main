# Formal verification rollout record — 2026-09-27 (main-side package)

Scientific effect: **NONE**. Coordination record for all participating agents,
current and future, across `main`, `Math-`, `query-`, `meta-framework` and
`Universal-Law-Workspace`. It records what the main-side formal package adds,
how it fits the lane that already exists, what did not change, what each
repository is asked to do next, and what this session could not do. Read it
with the [formal-verification guide](FORMAL_VERIFICATION.md) before touching
`formal/`, `STATUS.md`, `PROOF_INDEX.md` or any acceptance workflow.

## Owner instruction

Dylan Roy, 2026-09-27, in the request that started this work (recorded verbatim
in the pull request that landed this file):

> As we get closer to retrieving all the information for the github and
> following the intent behind what we're aiming to achieve I think it's time we
> incorporate the below so we further our goals. […] Read it and then understand
> how it will better our project and then implement anything and everything. Be
> sure to coordinate with all other agents and future agents for all
> repositories so they will be aware of the changes and why, I Dylan Roy pre
> approve any decision, request, or action needed to achieve this goal

The "below" was a roadmap produced by another AI model: keep the existing
provenance / scope / review / hard-gate / museum system as **Layer 0**; add a
**Layer 1** of Lean 4 statements and proofs bound by hash to the Layer 0 bytes;
carry a per-proof formalization status (`none` / `specified` / `proved` /
`kernel-checked`); build a glossary mapping project terms to standard
mathematics; open a statement-alignment review lane; extend the hard gate so
nothing is called "verified" without compilation plus alignment; treat parent
dependencies as explicit hypotheses; translate numerical bounds into exact
rational inequalities; keep AI-authorship labels author-side until a distinct
reviewer checks alignment; optionally add Coq/Metamath, an AI-prover cross-check
lane and external peer review. Roadmap step 1 was a SIDE24 pilot. This is
consistent with the [current owner instruction](../governance/OP-AUTONOMY-20260923-v2.1.md)
and [AGENTS.md](../AGENTS.md): it extends the evidence stack and creates no new
permission queue.

## Sequence of events (so nobody reconstructs it wrongly)

1. The same owner directive reached more than one agent. An OpenAI agent landed
   the lane's contract first: [Math- PR92](https://github.com/d6g8k5htny-coder/Math-/pull/92)
   (exact successor commit `cc2989c1280f4f227d0c6aa30c8841d6ba01e46e`: Lean 4 +
   Mathlib package, 13 GP-FOR-192 scalar companions, `formal/gate.py`,
   `manifest.json`, `SCOPE.md`, alignment-record validator, receipt) and, in this
   repository, [#177](https://github.com/d6g8k5htny-coder/main/pull/177)
   ([guide](FORMAL_VERIFICATION.md), [reconnaissance memo](reconnaissance/2026-09-27-formal-verification.md),
   museum route, AGENTS.md section pointing at work item
   [#95](https://github.com/d6g8k5htny-coder/main/issues/95)).
2. In parallel, this Anthropic/Claude cloud agent built a **core-Lean** package
   for the SIDE24 arithmetic skeleton with its own `registry.json` and status
   levels, and opened [#178](https://github.com/d6g8k5htny-coder/main/pull/178).
   A third agent replayed it (build, audit, gate, 38 controls, 165 repository
   tests), resolved the `AGENTS.md` conflict with #177 to name both pilots and
   forbid a third registry, and merged it (`eaf1264`); the
   [queue reconciliation page](QUEUE_RECONCILIATION_20260927.md) records that
   replay and notes "the two pilots have no registry crosswalk yet".
3. [#181](https://github.com/d6g8k5htny-coder/main/pull/181) is that crosswalk. The `main` package is **converged onto
   the Math- contract** instead of kept as a second vocabulary: `registry.json`
   and the custom `Audit.lean` are removed; the package now carries a
   `manifest.json` sidecar with the same labels (`proved` at source,
   `kernel-checked` only in a run receipt, alignment as a separate pending
   dimension), the same alignment-record contract and validator, the same
   self-award refusals, and executable negative controls. The project-wide
   `none` rows the first version carried for unformalised claims are dropped:
   the lane records per-package targets, and "not yet formalised" is described
   in prose (`SCOPE.md`, the guide's work-offer table), not as register rows.
   The 31 Lean statements are byte-for-byte unchanged apart from two module
   docstrings that referred to the removed files.

Two packages therefore exist in one lane, on one contract:

| Package | Repo | Backend | Targets | Source-level status | Author |
|---|---|---|---|---|---|
| GP-FOR-192 scalar companions | Math- `formal/` @ `cc2989c1…` | Lean 4.34.1 + Mathlib `d13f23b7…` | 13 | `proved`; kernel-checked in [run 36352398376](https://github.com/d6g8k5htny-coder/Math-/actions/runs/36352398376) | OpenAI |
| SIDE24 arithmetic skeleton | main `formal/` (#178, converged in #181) | Lean 4.34.1, core only | 31 | `proved`; kernel-checked in the `formal-verification` workflow receipt | Anthropic / Claude |

Neither author can review the other's alignment for independence credit of its
own package; each **can** review the other's, and that cross-review is the
cheapest independent review available to the lane.

In the vocabulary of [#95](https://github.com/d6g8k5htny-coder/main/issues/95)
(section C, the L0–L5 ladder with `verification_level` recorded separately from
scientific status), a target with a trusted `kernel-checked` receipt is **L5
evidence for exactly that target**. Both packages together satisfy the issue's
"at least one small formally/checker-certified pilot lemma" acceptance item;
neither populates the schema graph or the promotion engine the issue also asks
for, and the issue's rule "do not create a second competing status database"
is the reason the two packages now share one sidecar contract.

## What the `main` package contains after convergence

(#178 landed the first version; this follow-up replaces its registry with the
lane contract. Rows describe the converged head.)

| Item | Where | Verified how |
|---|---|---|
| Lean 4 package, toolchain `v4.34.1`, no dependencies | [`formal/`](../formal/README.md) | `lake build` < 1 s |
| 31 targets: exact integer/rational facts of SIDE24 §1–4 (Ledger 17, ConeMoments 10, Endpoints 4) | `formal/UniversalLaw/Side24/*.lean` | `leanchecker` passes; every target's transitive axioms `[]` |
| Evidence sidecar: `proved`, `PENDING_INDEPENDENT_REVIEW`, hashes of 17 bound files, 2 pinned sources, verbatim anchors, `does_not_claim`, 7 statement-level controls | [`formal/manifest.json`](../formal/manifest.json) | source gate `SOURCE_IDENTITY_PASS` |
| Exact coverage / not-established table and review contract | [`formal/SCOPE.md`](../formal/SCOPE.md) | bound by hash; digest in every alignment record |
| Byte copies of `Math-@9d7b6802…` `coefficients/side24_v1/PROOF.md`, `ENCLOSURE.json` | `formal/sources/side24_v1/` | sizes and SHA-256 checked |
| Generated informal–formal table | [`formal/ALIGNMENT.md`](../formal/ALIGNMENT.md) | gate refuses drift |
| Glossary (SIDE24 / RN / P15 vocabulary, descriptive, "proposed not adopted") | [`formal/GLOSSARY.md`](../formal/GLOSSARY.md) | prose |
| Review lane, JSON template, validator | [`formal/REVIEW_LANE.md`](../formal/REVIEW_LANE.md), `formal/reviews/`, `--alignment` | 6 validator controls |
| Fail-closed gate + receipt | [`tools/formal_gate_check.py`](../tools/formal_gate_check.py) | 43 tests (`tests/test_formal_gate.py`), incl. one real `--run-lean` |
| CI lane | `.github/workflows/formal-verification.yml` | pinned elan script digest; `--run-lean`; uploads `formal/.lake/formal-evidence` |
| Front-door wiring | `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `docs/WORKSPACE.md`, `docs/RESEARCH_INDEX.md`, landing/navigation manifests | landing and navigation checks |

Locally on the converged head: `--run-lean` completed in about five seconds with
build, `leanchecker`, axiom audit and elaborated-type capture all exit 0; the
seven tightened/wrong statements were each rejected by Lean
(`REJECTED_BY_LEAN`); `sorry`, an indirectly imported custom axiom and
`native_decide` were each rejected by the axiom gate (`REJECTED_BY_AXIOM_GATE`);
the receipt reports `kernel-checked` with `checked_commit` and all-empty axiom
lists. The 43 gate tests pass in normal and `-O` mode. The workflow's first
hosted run is the PR's own check; a hosted result, not this file, determines
the hosted outcome.

## What did not change

- No row of `STATUS.md` (a byte-pinned snapshot read by the museum and shop
  tests), no line of `PROOF_INDEX.md`, no `LANDING_CLAIMS`, `claims/`, prize or
  `lemma_closed` value. The manifest declares `scientific_status_authority:
  false` and the gate refuses one that says otherwise.
- No Layer 0 verdict. The SIDE24 coefficient bound remains "ACCEPT — scoped"
  by source-bound review; the kernel checked its arithmetic skeleton, not its
  analysis, and not the bound.
- No parent theorem became accepted; no AMEND item closed.
- The Math- pilot's files, run, status and scope are untouched.
- AI authorship: the Lean text is author-side (Anthropic / Claude, Cursor cloud
  agent) until a distinct reviewer's record validates.

## Design choices, briefly

- **One contract, two packages.** A second status vocabulary or register would
  have been exactly the "competing registry" #177 warns against. Converging
  cost a rewrite of the gate and manifest; it bought one review procedure, one
  receipt shape and one set of refusals across both repositories.
- **Core Lean for arithmetic, Mathlib where the pilot already pays for it.**
  The arithmetic skeleton needs only `decide` on integers; building it costs
  seconds and no cache. Real-number statements (the coefficient definition,
  the enclosure, `ρ*`) belong in the existing Mathlib package rather than in a
  second dependency tree.
- **Kernel audit, not build success, is the criterion.** A green `lake build`
  tolerates `sorry` warnings and `native_decide`. The receipt is written only
  after the transitive `#print axioms` audit passes for every target.
- **Executable controls, not described ones.** Every negative control in the
  manifest is actually compiled and must actually fail; the gate refuses to
  write a receipt otherwise.
- **Anchors now, Blueprint later.** Verbatim-substring anchors into pinned
  bytes give a checkable informal–formal link today. Blueprint `checkdecls`
  checks that declarations exist, not that they mean the same thing (memo), so
  the independent alignment review stays regardless.

## Requests to each repository and to future agents

Coordination under the same owner instruction; no further permission needed.
Claim work in [#95](https://github.com/d6g8k5htny-coder/main/issues/95) to
avoid duplicate writers, as the guide asks.

**`main` (this repository)**
- **Alignment review of the 31 targets** by a non-Anthropic reviewer, per
  [`formal/REVIEW_LANE.md`](../formal/REVIEW_LANE.md); record the JSON and
  Markdown under `formal/reviews/`, validate with `--alignment`.
- **Gate unification.** `tools/formal_gate_check.py` and Math- `formal/gate.py`
  implement the same contract twice. Either package the Math- gate for reuse or
  fold this one into it; until then keep the vocabularies identical and add any
  new refusal to both.
- **Package transfer.** If the lane decides all formal packages should live in
  Math- beside the sources they formalise, move `formal/` there as a separate
  Lake package (or as a `lean_lib` inside the pilot's Lake project) with an
  explicit successor note; the sidecar, anchors and controls transfer as-is.
  The token available to this session could not push to Math-.
- When editing bound files: `--refresh-hashes`, then `--write-alignment`,
  then the gate; a red hosted run is preserved, not rewritten.
- Museum: if a formalization view is added, it reads the manifests and receipts
  and displays `scientific_status_authority: false`; the two status vocabularies
  are never merged.

**`Math-`**
- **Numerical formalizer, remaining steps** (in the Mathlib package): the exact
  definition of `c_{d,ref}` (display (1) of the SIDE24 note) as a Lean `def`
  over `ℝ`; a proof-producing enclosure pipeline so that the decimal endpoints
  become a theorem rather than data. The main-side package supplies the exact
  arithmetic those steps will consume and states precisely what it does not
  enclose.
- Under the D3 entry of `PROOF_INDEX.md`, one descriptive line may point to the
  main-side sidecar and receipt and state that the arithmetic skeleton has
  kernel evidence while the analytic content does not. Do not change the D3
  verdict; do not add a status column.
- New proof notes: display exact decimal or rational constants so anchors can
  quote them. Frozen bodies stay frozen; a formalisation attaches by hash and
  never edits them. A defect found by formalisation is an erratum successor.
- Next full-theorem target the guide's table does not list: D6 Theorem F's
  constant `ρ* = 1/(3 − log(3e − 2))`, its 20-digit enclosure and `ρ* < 6/7` —
  finite combinatorics plus one `Real.log`, so the most tractable statement to
  move from `specified` to `proved`.

**`meta-framework`**
- Optionally a descriptive `formal_evidence` field per source entry pointing to
  the package manifest and the run receipt that covers it. Descriptive only; the
  manifests and receipts stay the evidence, and nothing in the catalog becomes a
  status authority.

**`query-`**
- No change required. If Lean file identities are exposed later, reuse the
  `repository + path + commit + blob + bytes + sha256` row shape.

**`Universal-Law-Workspace`**
- Navigation only: link the guide and both package READMEs.

**All agents**
- `kernel-checked` is a receipt-level label for exact targets. It is not
  "verified" in the Layer 0 sense and must not be written up as if the
  surrounding theorem were.
- A target listed under "Not established" in a `SCOPE.md` is a visible boundary,
  not a defect and not a claim.
- Before citing a formal result, quote the run (workflow run id or local
  `--run-lean` output and `checked_commit`), not the manifest alone.
- Reviewing the other provider's package is the cheapest independent review
  in the lane; record pickup in #95.

## Capability limits of this session (reported, not permission requests)

- The Git token available here can push only to `main`. The Math-,
  meta-framework, query- and Universal-Law-Workspace items above are recorded
  here and in [#181](https://github.com/d6g8k5htny-coder/main/pull/181) for an agent with access.
- The GitHub CLI is read-only here and its token was intermittently invalid
  (HTTP 401) during the session, so issue #95 could not be posted to. An agent
  with write access should link this record there.
- The `formal-verification` workflow's first hosted run on the converged gate
  is the follow-up PR's own check; the merged first version was replayed
  locally by the merging agent, not on GitHub Actions.

## Uncertain or not verified

- Hosted cost is no longer uncertain for the first version: the pre-convergence
  workflow completed on `main` in 34 seconds end to end
  ([run 36353990289](https://github.com/d6g8k5htny-coder/main/actions/runs/36353990289),
  all steps green, including the elan download). The converged workflow does
  the same work through `--run-lean` plus the executable controls; its first
  hosted run is #181's check.
- Whether two of the four `Endpoints.lean` anchors — which quote
  `ENCLOSURE.json` metadata rather than prose, because the JSON has no prose —
  are acceptable pairings. An alignment reviewer should say.
- The glossary rows for D1 (matrix cap, marked cylinder, elder selection) and
  the D5 region names were not extracted from their sources in this session and
  say "read the source".
- Whether the lane will keep two gates or unify them (see the `main` requests).
