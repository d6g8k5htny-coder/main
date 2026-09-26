# Universal Law

**Gaussian random fields, persistent homology, and reproducible multi-model mathematical research.**

[![CI](https://github.com/d6g8k5htny-coder/main/actions/workflows/ci.yml/badge.svg)](https://github.com/d6g8k5htny-coder/main/actions/workflows/ci.yml)
[![Navigation](https://github.com/d6g8k5htny-coder/main/actions/workflows/navigation.yml/badge.svg)](https://github.com/d6g8k5htny-coder/main/actions/workflows/navigation.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This is the **single public front door** for the Universal Law research program. The project studies persistence lifetimes and critical-point geometry of Gaussian random fields, together with exact/reproducible computational methods and a fail-closed review workflow. Proofs, code, reviews, and source identities live in several focused repositories, but readers should start here.

## Current status

| Class | Current objects |
|---|---|
| **ACCEPT — scoped** | D2 lifetime remainder; D3 SIDE24 coefficient calculation; D4 fixed-remote RN count theorem; D6/P15 full-price theorem at its stated realized-family scope |
| **AMEND / open** | D1 parent quantitative Theorem-A chain; D5 pin-neighborhood / microdisk bound; SARD-G A1/A6 |
| **Engineering only** | `query-` package, Universal-Law-Workspace federation map, CI/reproducibility infrastructure |

**ACCEPT — scoped** means a source-bound technical review accepted the stated object at its declared hypotheses. It does **not** silently accept imported parents, broaden scope, close prize problems, or turn CI into mathematical evidence.

Read the exact scope, sources, and review links in **[STATUS.md](STATUS.md)**.

## Run one thing

A clean public checkout of the read-only query package needs no private Drive access and no dependency install for its test suite:

```sh
git clone https://github.com/d6g8k5htny-coder/query-.git
cd query-
python -B -S -m unittest discover -s tests -p 'test_*.py' -v
```

That command is also the canonical package-control step in the `query-` GitHub Actions workflow. Passing it verifies the package controls; it does not verify a theorem.

## Where things live

| Repository | Purpose |
|---|---|
| **[main](https://github.com/d6g8k5htny-coder/main)** | This front door: status, public reading path, review discussions, integration |
| **[Math-](https://github.com/d6g8k5htny-coder/Math-)** | Proofs, calculations, mathematical tests, proof index |
| **[query-](https://github.com/d6g8k5htny-coder/query-)** | Read-only exact-source lookup and local verification |
| **[meta-framework](https://github.com/d6g8k5htny-coder/meta-framework)** | Machine-readable source identities and repository routing |
| **[Universal-Law-Workspace](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace)** | Supporting federation/navigation map; not a second research home or status authority |

For full proof texts, use **[Public mathematics](docs/PUBLIC_MATHEMATICS.md)**. For detailed reading order and open work, use the **[Research guide](docs/RESEARCH_INDEX.md)**.

## Evidence rules

- Exact source identities are pinned by Git object and/or SHA-256 where relevant.
- Numerical or floating-point work is labeled non-certifying when it is not a proof.
- Independent review is recorded separately from same-author checks.
- A merge, green CI run, hash match, publication, or model agreement does not promote mathematical status.

## Research still in progress

The parent D1 quantitative selection chain, pin-neighborhood/microdisk work, SARD-G execution details, and other explicitly open regional obligations remain visible rather than being flattened into a solved-program claim. See [STATUS.md](STATUS.md) and the [public proof catalog](docs/PUBLIC_MATHEMATICS.md).

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). New work should attach to an existing mathematical or engineering lane instead of creating a new repository or duplicate status surface.

## License and citation

Released under the [MIT License](LICENSE). Citation metadata is in [CITATION.cff](CITATION.cff).
