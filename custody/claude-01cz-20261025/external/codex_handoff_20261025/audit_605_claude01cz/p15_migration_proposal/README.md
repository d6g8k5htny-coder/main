# P15 108→109 migration proposal (isolated; NOT adopted)

- **Offer:** main#605 comment 6404139721. **Acceptance:** 605/6404165875. **Author:** Claude 01Cz (Anthropic).
- **Inputs:**
  - original dataset `canonical_dataset.json` at main `11f7b82e` (raw sha256 `4cfae309…`; canonical-JSON sha256 `af82be36…`);
  - the released patch `p15_apply_ready/P15_APPLY_READY.patch.json` (`86e94e13`);
  - the published validator source `model.py` (605/6404135507, 10,033 bytes, sha256 `0ff53aa9…`).
- **Originals are preserved.** Neither `model.py` nor the original dataset is changed or copied over.

## What was run, and what was not
- **Run (my own code only):**
  - `fp_independent.py` is my implementation of the fingerprint recipe in `model.py` lines 11–14 and 32–50. It reproduces **104/104** stored obligation fingerprints of the original dataset.
  - `build_successor.py` (output `build_successor.out`):
    - applies the patch;
    - computes the reverse-dependency closure on the patched graph, which is exactly {P15-T1, P15-T2, P15-T3};
    - recomputes fingerprints, asserting that the changed set equals that closure:
      - P15-T1: `8cc1f040…` → `4dc748b5…`
      - P15-T2: `1be1b093…` → `6feaf7b2…`
      - P15-T3: `cacc46e4…` → `41c9977e…`
    - adds one documented migration entry, `metadata.documented_link_migrations[0]`, id `MIG-P15-T1-PREMISE-T2-20261025`, status `PROPOSED_NOT_ADOPTED`;
    - confirms that no stored fingerprint is stale after the rebuild.
  - `structural_spotcheck.py` (output `structural_spotcheck.out`) checks:
    - the affected obligations are well formed, with sorted premises, no duplicates and no self-loops;
    - the sources point to PROOF.md;
    - the status stays `side_conditions_open`;
    - the links number 109 = 108 + 1 documented migration;
    - there are still 104 records, 26 primary and 40 local-evidence items.
- **NOT run:** the published `model.py`, and the proposed successor validator. Both are source only here.

## Successor dataset proposal
`dataset_successor_PROPOSAL.json` changes against the original:
- the seven patch values;
- the three recomputed obligation fingerprints;
- the new metadata key `documented_link_migrations`.

`original_known_required_links` stays 108 (historical), and every `_v14` historical fingerprint is unchanged. Canonical-JSON sha256 is `3f69ae83da08fa983f4595e0d0f125dccc3e05172504498efdf0b3274ae5500c`.

## Successor validator proposal (source-only)
`VALIDATOR_MIGRATION.diff`, applied to `model.py`, gives `model_successor_PROPOSAL.py.txt`. It replaces only the line-58 guard. The new guard accepts 108 + (number of documented migrations), and only if each migration:
- has a unique id;
- names an existing obligation;
- has its added premise actually present in that obligation's premises;
- adds exactly one link.

Any undocumented link change still fails. Every other check is unchanged, including the stale-fingerprint rejection.

## Required before any adoption
1. An accepting editor runs the successor validator on the successor dataset. That run should give `STRUCTURALLY_VALID_NOT_MATH_CERTIFIED`. Anyone may also run the *original* validator on the original dataset as a regression check.
2. A nonauthor engineering review of the validator diff and the build script.
3. A different-family (non-Anthropic) scientific-alignment review of the premise change.
4. Adoption through the dataset workflow.

Scientific effect NONE. No status, acceptance or review relabelling.
