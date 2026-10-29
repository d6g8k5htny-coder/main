# Draft quarantine entries for research PRs in this slice

These are drafts for the canonical `PROOF_INDEX.md`, which Root writes under governance B-003. They use the quarantine
fields of `GOVERNANCE.md` at `main` `81acc060`. Root assigns the IDs; `Q-DRAFT-<PR>` is a placeholder.

The boundaries below restate each PR's own claims and limits, and its recorded reviews, at the exact head named. They
are not a new mathematical review. "Paper-reviewed" means prose review, which by governance cannot establish kernel
closure. None of the three directories involved exists on `main` at `81acc060`, and no file on `main` imports from
them: the only mentions are three Atlas contender lists under `docs/reports/`.

```yaml
id:             Q-DRAFT-336
title:          Stationary quartic H0 law with explicit sampling weights
location:       main PR #336, head 3ab12c6542c88554ad0d42e0e19aff2461dc3c0d (refs/pull/336/head);
                experiments/universality/quartic_cusp_field/{PROOF,REVIEW}.md and a README.md row
layer_a_status: no formal code, so A1/A2 do not arise; no executable certification, so A3 does not arise;
                A7 requires it to be listed as a paper proof, not as closed
exact_boundary: Claimed and paper-reviewed, for one explicit smooth stationary torus field with a matched
                quartic surgery, protected annulus and fixed high-level enclosure: exactly one globally
                selected finite superlevel H0 bar; the younger maximum dies at the middle saddle, with lifetime
                exactly C*lambda^2; parameter density gamma*lambda^(gamma-1) on the declared interval gives
                lifetime density beta*ell0^(-beta)*ell^(beta-1) with beta = gamma/2; the independent-whole-copy
                marked Poisson limit holds under t_n -> 0 and n*t_n^beta -> Lambda.
                Not claimed: a generic two-control cusp, a Gaussian contact law or a covariance match. The
                control path is C1 but not C2 at zero, and the two-site observation law is singular.
                Not established: any kernel proof or certified numerics.
blocking_reason: no formalization exists, so it cannot be CLOSED; the PR is stacked on #334 and #331, both
                closed on 2026-10-29, so it cannot integrate as it stands
attempted:      a paper proof with two original full nonauthor mathematical PASS reviews and one fresh full
                nonauthor PASS at this head (all OpenAI/Codex, source-exposed, organizational independence 0)
resumption_cost: a Lean formalization of the surgery and enclosure argument (large), or a recompose onto
                main to hold it in-tree as quarantined
isolation:      not on main; nothing on main depends on it
owner:          none
maintenance:    zero
```

```yaml
id:             Q-DRAFT-340
title:          Smooth cusp population bars and fold-edge dominance
location:       main PR #340, head f02758555b4bce4554e34e8913947f69578d0816 (refs/pull/340/head), stacked
                on #336; experiments/universality/two_parameter_cusp_population/{PROOF,REVIEW}.md and a
                README.md row
layer_a_status: as Q-DRAFT-336; it imports Q-DRAFT-336's surgery, so it is quarantined with it
exact_boundary: Claimed and paper-reviewed, for one declared stationary non-Gaussian two-control cusp
                population: almost every field has zero or one finite H0 bar, with total probability at most
                1/5; short bars near the fold edges have leading density C_edge*ell^(-1/3), with cumulative
                count (3/2)*C_edge*t^(2/3); controls away from the edges give the exact tip density
                C_tip*ell^(1/4) below the stated cutoff, with cumulative count (4/5)*C_tip*t^(5/4); and
                independent whole copies with t_n -> 0 and n*t_n^(2/3) -> lambda give the stated Poisson
                random measure with explicit latent marks.
                Not claimed, per the PR: a general contact-stratum classification, Gaussian contact
                admission, a higher-homology theorem, a same-field spatial limit, numerical coupling,
                a compiler result or a blind experiment.
                Not established: any kernel proof.
blocking_reason: no formalization; it depends on Q-DRAFT-336; its stack base is closed
attempted:      two original full nonauthor mathematical PASS reviews of the 696-line preparation and a fresh
                full PASS of the 782-line proof at this head, plus bounded metadata reviews (all OpenAI/Codex)
resumption_cost: as Q-DRAFT-336, plus formal versions of the uniform root, annular and height guards
isolation:      not on main; nothing on main depends on it
owner:          none
maintenance:    zero
```

```yaml
id:             Q-DRAFT-429
title:          Fold isolation, near-diagonal counts and covariance obstruction (planar finite Fourier fields)
location:       main PR #429, head 16764aab8ce7b74327276ac4be619f2d3e4ed4af (refs/pull/429/head);
                83 new files under experiments/universality/fold_isolation_diagonal/. If #429 is integrated,
                the location becomes that path on main.
layer_a_status: no formal code. Its added Python controls have no float constructs on their added lines
                (lexical signal; see dispositions.json). A7 requires it to be listed as a paper proof.
exact_boundary: Claimed and paper-reviewed, for planar finite Fourier Gaussian fields under the displayed
                normalization, physical twelfth moment, mark and radius assumptions:
                (1) arbitrary-angle fold isolation;
                (2) an almost-sure finite critical-point count;
                (3) for a growing family, ordered distinct short maximum-saddle candidates are bounded by
                    "C16^4 K^8(1+L)^2 exp(-c/eta)" (verbatim from the PR body; read the bound and its
                    constant from PROOF.md at this head);
                (4) these upper moments plus a uniform one-contact floor do not give a nonzero Poisson limit:
                    epsilon = N^-12 gives an empty candidate limit, and positive-mean preservation needs a
                    separate, possibly smaller existence diagonal.
                Not established: any kernel proof; the specialist-human review the PR requested
                (review 5635594683); anything beyond the displayed hypotheses.
blocking_reason: no formalization exists, so it cannot be CLOSED
attempted:      four ACCEPT_SCOPED mathematical reviews at exact byte and hypothesis identities
                (OpenAI/Codex), and five PASS_SCOPED different-family slices (Anthropic Claude, including this
                unit's own session, which is therefore not independent of this draft)
resumption_cost: Lean formalization of the four arguments (large); integration as an in-tree quarantined
                package is small
isolation:      not on main; nothing on main depends on it
owner:          none
maintenance:    zero
```

## PRs in this slice that need no new proof-index entry

- **#389**, the SIDE24 V3 manuscript, adds no new obligation. Its mathematics is SIDE24's, which `STATUS.md` already lists.
- **#358**, the Lean verification specification, is a specification, not a proof.
- **#344, #346, #349 and #385** belong to the cusp-population stack. The stack owner, active on #343 and #350, should write their entries with the stack's disposition. For #349, any entry must record that its added code makes inexact `math.` calls in a path titled "certify global H0 bars", so it is not an A3-admissible certificate unless a later review shows otherwise.
