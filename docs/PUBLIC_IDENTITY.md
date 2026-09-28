# Public identity and repository naming

Scientific effect: **NONE**. This document governs public naming and presentation only. It does not rename repositories, rewrite historical source identities, or change mathematical status.

## Public roles

| Current repository | Public role | Preferred display label |
|---|---|---|
| `main` | canonical public research home | Research Home |
| `Math-` | mathematical proof vault | Mathematics / Proof Vault |
| `query-` | exact-source lookup | Research Query |
| `Universal-Law-Workspace` | federation and pinned topology | Federation Map |
| `meta-framework` | machine-readable catalog and routing | Research Catalog |
| `google-drive` | selected public custody bridge | Evidence Custody |
| `governance-` | operating and integrity protocols | Governance |
| `trial` | integration and adversarial testing | Integration Lab |
| `sandbox` | bounded experiments | Sandbox |

## Naming standard for new repositories

New public repositories should use descriptive, lowercase kebab-case names with no trailing punctuation. Prefer a stable function over a temporary implementation detail.

Examples: `research-query`, `evidence-custody`, `integration-lab`.

Avoid new names like `test2`, `new-main`, `Math-`, or names ending in a hyphen.

## Existing names are provenance-sensitive

Do **not** casually rename `Math-`, `query-`, `governance-`, `main`, or the account itself. These identities occur in manifests, Git URLs, submodules, CI, receipts, review records, frozen source inventories, and historical citations.

A future rename is a migration, not a cosmetic edit. Before renaming:

1. inventory every exact repository-name/URL occurrence across the federation;
2. classify frozen historical evidence versus mutable navigation/configuration;
3. prepare new canonical names and an explicit old→new alias ledger;
4. update mutable workflows, registries, documentation, submodule URLs, package metadata, and public navigation in one coordinated change;
5. retain historical source strings where changing them would falsify provenance;
6. run cross-repository conformance, source-custody, formal, downstream, and negative controls;
7. perform the GitHub rename only after the migration PRs are ready;
8. verify GitHub redirects for humans but never rely on redirects as the machine-readable identity contract;
9. publish a dated migration receipt.

Candidate future names, subject to that audit:

- `Math-` → `mathematics`
- `query-` → `research-query`
- `governance-` → `governance`
- `trial` → `integration-lab`
- `google-drive` → `evidence-custody`

The current names remain canonical until such a migration is explicitly completed.
