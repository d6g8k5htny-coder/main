# Full-spectrum method to fixed d2/L24 code: a source chain

This appendix records one cross-unit source relationship for
[GB-8w / main#555](https://github.com/d6g8k5htny-coder/main/pull/555).
It supports the existing retro inventory without changing its candidate files,
GRAPH, v2 bindings, Artifact Index, scientific status or ownership.

The source-declared chain is:

1. `full_spectrum_enclosure_method_20261018/PROOF.md` is the durable METHOD_V2.
2. The fixed `finite_l_d2_coefficient_20261018` proof and source map explicitly
   reference that method by its complete byte count and SHA256.
3. That coefficient package identifies the six producer/checker/helper/test
   carriers below. Those files are already inventoried in
   [GB-8e](GB-8e.agent5.node-candidates.json).

The useful addition is this relationship. It is not another complete inventory,
a new authoritative graph edge, or a finding that the general method has an
implementation for every dimension and side length.

## Why the two mathematical source units are related

All source links in this appendix use the coherent historical Math- commit
**9f3fb6918b28e6378c034ebc19e76d432f5391cb**, whose tree is
`dd42defa2440f5a15777399fce2881867e004d2b`.

The [coefficient proof, exact-object paragraph][coefficient-proof] states:

> This package evaluates only the explicit original collision expression
> C_alg(2,24), with physical derivatives, the complete infinite lattice
> spectrum and the common full normalizer including the zero mode.

Its next source-reference paragraph identifies METHOD_V2 as **32593 bytes**,
SHA256 **8bdea4b42e9efd2d846cd1aa73691e8194fa26cd937ac02b79991334f88204f6**,
and points to [the original public source comment][method-comment].
The [source map][source-map] repeats these exact values in `method.bytes`,
`method.sha256` and `sources.method_v2`.

Those values match the complete durable [method proof][method-proof] below.
This supplies positive evidence for a **method referenced by a specialized
coefficient package** relationship. It does not rely on similar filenames,
matching model confidence, or a workflow reference alone.

The dated `NO IMPLEMENTATION FOUND` entry in
[RETRO-MATH-BIND-01](RETRO-MATH-BIND-01.md) remains a bounded search observation.
The external specialization found here does not turn that observation into a
claim of generic backend coverage, and does not require rewriting it as a
historical success. The method unit remains unbound unless the existing
reviewed node/binding process establishes otherwise.

## Exact source and carrier identities

All paths are relative to `d6g8k5htny-coder/Math-` at the commit above.
The two package prefixes are:

- Method: `frontiers/full_spectrum_enclosure_method_20261018/`.
- Coefficient: `frontiers/finite_l_d2_coefficient_20261018/`.

| Package / file | Bytes | Git blob | SHA256 |
| --- | ---: | --- | --- |
| Method / [PROOF.md][method-proof] | 32593 | `74746107f9ed006bbe03b0f609e23db005b260ce` | `8bdea4b42e9efd2d846cd1aa73691e8194fa26cd937ac02b79991334f88204f6` |
| Coefficient / [PROOF.md][coefficient-proof] | 7891 | `790b99815f05440e8b47d50fb794b1f50edc9bfb` | `53f56c9b47ce1b4e8781f4d6f9a23e3b7bd4c0807809a85c1c6359ea74462c62` |
| Coefficient / [SOURCE_MAP.json][source-map] | 1792 | `dd39cc67a528480f5e5097342bde6f55f21a02c0` | `3ab15c99174924cf8f172cf2ec035bcb2d41ab64ca59ee910aed4f6a79dba53a` |
| Coefficient / [coefficient.py][producer] | 5398 | `9e0d8b5cf8271285c9a32afc96e055ad4dd4c2be` | `6fbb99eb9a9212c8c7c27ad34a362e81cf6ceb5662526072a7fc7f22322df7a6` |
| Coefficient / [verify_coefficient.py][checker] | 6914 | `49f3b805f4a92e11156be5afa9a6f7ea0fb91b25` | `434254a62602e16fddcde5804e0a8d74359cc23b54c7eb02f2f9b4d7590ed181` |
| Coefficient / [moment_kernel.py][kernel] | 12063 | `145921d6b24e5e8e76dbe6c19372cec13020c6ef` | `68c3caf28190e8be58464cabfc39ab565449d9bb64716ab139ac0dd0a181193e` |
| Coefficient / [verify_certificate.py][moment-checker] | 10829 | `e5329fdce27ad3b57c01394714355cc1a3e8d2c0` | `a0afbfdf9ca323db38e06171c138f683669c42fe99f6892d6233ab03802d4715` |
| Coefficient / [contract.py][contract] | 3170 | `acd5926bc87823d4ce3cd1f35bdd07528189fbec` | `7dbd6de6671fe5919151987c2ef62fb165409601e3d4e38d60ae60d8c9513678` |
| Coefficient / [test_coefficient.py][tests] | 6907 | `12b7df5ab5f28741d3cd4bc5009f96d17ee5220b` | `0dca94d76c93aa612188e197b831f13ae069926ff6e9237412f2ff089c80fde2` |

The [coefficient proof's trusted-computation section][roles] declares the
producer/checker/helper roles below. The test-file row is the appendix author's
description from static inspection, not a role declared in that proof.

| Carrier | Role, provenance and boundary |
| --- | --- |
| `coefficient.py` | Producer for the fixed collision expression; uses the moment kernel and common contract. |
| `verify_coefficient.py` | Separate checker reconstructing coefficient witnesses; uses `verify_certificate.py` and the common contract, and does not import or execute the producer. |
| `moment_kernel.py` | Reused full-spectrum moment/exponential-witness producer. Its unassigned-parameter input semantics are retained. |
| `verify_certificate.py` | Reused moment/exponential-witness checker, consumed by the coefficient checker. |
| `contract.py` | Shared trusted parsing, fixed configuration, source/custody and resource logic. The proof explicitly excludes scalar/Gamma/root enclosure calculations from this helper. |
| `test_coefficient.py` | Appendix-author description from static inspection: existing source tests and tamper controls. This is not a source-declared proof role or a fresh execution receipt. |

These are documentary roles, not a new audit of code correctness. Software
separation with a shared contract and specification is not organizational
independence.

## Scope that this chain does not supply

The source identifies a fixed d=2, L=24 expression evaluation, not a complete
implementation or verification of the general METHOD_V2. The coefficient proof's
acceptance boundary retains the ordinary-bar identification, G1–G4, M31 effective
remainder/window, count-tail, sampler/input-law, confidence/power and resource
admission questions. This appendix discharges none of them.

In particular, these programs are not an implementation of
`far_sector_lifetime_refinement_20261022`'s paper-only density/bin/window
argument. A source reuse relationship must not be substituted for that distinct
implementation claim. Graph/node adoption, execution receipts, scientific
review, formal alignment and acceptance retain their separate meanings.

## Verification and attribution

On 23 October 2026 the appendix author checked the nine complete public source
files against their commit-tree paths, regular-file modes, byte counts and Git
blob identities, and recomputed their SHA256 values. The source map's method
identity and the coefficient proof's explicit reference match the durable method.
This was inert byte/provenance verification. No mathematical code, parser,
producer, checker, tests, sampler or workflow was imported or executed.

Actual appendix author: OpenAI/Codex `/root`, under
[pickup main#555/6376049516][pickup]. Read-only discovery and contention inputs
came from `oct23_retro_dependency_gap` and
`oct23_cross_family_support`. Source/history exposed; same provider and shared
principal; organizational, blind, human and scientific independence credit 0.
The original method/code authors and their published reviews are not renamed.
The original publication's Git author field records publishing custody, not
actual appendix authorship; the author attribution above remains controlling.

Existing GB-8w/#555 author and integration roles remain with their actual
holders. This appendix has its own review/integration stage; earlier reviews of
other bytes are not adopted as approval of this file.

[Owner mandate](../../governance/OWNER_MANDATE_20261015.md) and
[standing delegation](../../governance/OP-AUTONOMY-20260923-v2.1.md) apply.
NOT A RESULT; Scientific effect NONE; eng ≠ discharge; no flags; OBL OPEN.

[method-proof]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/full_spectrum_enclosure_method_20261018/PROOF.md
[coefficient-proof]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/PROOF.md#L11
[source-map]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/SOURCE_MAP.json
[roles]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/PROOF.md#L86-L107
[producer]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/coefficient.py
[checker]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/verify_coefficient.py
[kernel]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/moment_kernel.py
[moment-checker]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/verify_certificate.py
[contract]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/contract.py
[tests]: https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/frontiers/finite_l_d2_coefficient_20261018/test_coefficient.py
[method-comment]: https://github.com/d6g8k5htny-coder/Math-/issues/193#issuecomment-6243041893
[pickup]: https://github.com/d6g8k5htny-coder/main/pull/555#issuecomment-6376049516
