# Embedded q0 recovery — 27 September 2026

**Source accessibility and custody only. Scientific effect: NONE.**
This is an additive companion to the signature-based census, not a new mathematical status register. Existing source files, hypotheses, verdicts, and the frozen public-text inventory are unchanged.

## What the continuation recovered

The saved Drive census, rather than an interrupted screenshot's interim count, supplied the baseline: **260 unresolved R2 rows / 256 expected hashes**. Four freshly acquired consolidated Dropbox files and 77 selected loose originals account for **249 of those rows / 245 hashes**. **Eleven rows / eleven expected hashes remain unmatched.** This is a bounded historical-inventory reconciliation, not a census of every Dropbox object or Git branch.

The embedded extractor produced **387 occurrences / 386 distinct manifest-matching hashes**: 160 marked text blocks, 48 scripts decoded as data, and 179 structured-data/text/image objects. Every accepted original-manifest payload has a full SHA-256 and size match; short printed hash prefixes are insufficient. Of 60 JSON reserialization misses, 49 needed R2 originals were located as loose Dropbox files and recovered exactly. Those 49 parse to the same JSON values as the consolidated versions, but their original formatting matters to byte identity.

The 81 whole Dropbox downloads total **12,022,714 bytes**. They include the four consolidated files, not just newly missing sources. The four carriers also match fresh Drive downloads. Their calculated Git blob identities equal the objects returned for their historical public paths at `main@7caac254cbba5f513b2dc0afb56b78a598bc0c93`. GitHub was checked by returned blob metadata, not a second complete download. Most of the material was therefore already held in consolidated form; the improvement is recoverable individual sources and explicit locators, not 386 newly discovered mathematical results.

## Reproduce without executing the archived programs

With Python 3.11 or newer, place the four acquired carriers together, and run this tool, NOT `q0_verify.py`:

```sh
python -B -S tools/source_census_20260927/unpack_sources.py \
  --master /path/to/Q0_MASTER.md \
  --ledger /path/to/Q0_LEDGER.md \
  --machine /path/to/q0_machine.json \
  --scripts /path/to/q0_verify.py \
  --remaining /path/to/baseline-R2_REMAINING.csv \
  --out /path/to/new-extraction-directory
```

`--remaining` is optional. The output directory must not already exist. Original names and container identities are recorded in `EXTRACTIONS.json`; files are stored under full-hash names in `payloads/`. Literal marked slices record byte offsets. JSON encodings are accepted only when both the complete expected hash and byte count match. Python source is parsed as AST and selected literal base64 values are decoded; the carrier and recovered scripts are never imported or run. No pickle deserialization, shell evaluation, notebook execution, or macro execution occurs.

Inputs are limited to 32 MiB each and must be regular files, not final-path symlinks. This is a format-specific offline extraction utility, not a security sandbox or a general decoder for every Python program. Its manifest checks establish declared byte identity, not the truth of a source's internal claims.

## Explicitly excluded from recovery counts

Twenty sections marked as original-PDF text extracts remain reading copies, not reconstructed PDF bytes. The unmanifested script shim with only a hash prefix is rejected. Six currently available image files have different bytes from the expected R2 versions; they remain labelled variants, even where their titles match. A semantically equivalent JSON encoding never substitutes for a missing exact hash.

The remaining eleven targets comprise seven image identities (six with available but nonmatching variants), the CLAUDIT reconciliation addendum, the July 27 Drive research-audit white paper, and two pasted-text carriers. A title-search miss is not proof of global absence. `TRANSVERSE_CONTACT_ASYMPTOTIC.md` is a separate unrecovered project source; it is not silently folded into this eleven-row denominator.

## Verification and review scope

Fresh local checks: **40 tests in normal Python and 40 in optimized Python** (20 existing census tests plus 20 new extraction tests). Four intentionally weakened guards were rejected in both modes: full-hash equality, size equality, safe source names, and final-symlink rejection. The eight mutation runs fail at behavior assertions rather than import errors. Initial marker extraction, external full-hash support, and derived-PDF reporting have recorded test-first failures.

A second extraction from the four fresh Drive downloads reproduces the complete Dropbox extraction receipt, and all 387 output identities verify. The baseline ledger partitions exactly into 249 resolved and 11 unresolved rows without overlap. The complete pre-existing repository suite was not run locally for this successor; hosted results and any nonauthor review must be read separately. The Claude engineering review of PR173 covers the earlier scanner, not this new unpacker or the fresh raw-source reconciliation.

Script SHA-256: `60bddb2540a436d2467377b8dc2bf924ecfe1aeab098a81723cc3e8bcbc07b9f`.
Test SHA-256: `a7f0a8ddc5bb74c541c60052e7fa4d84e8c1ef646430a9a6af505655beca6e6a`.
Both published Git blob identities were read back and equal the locally tested files.

## Detailed source map and evidence

[Drive extraction and raw-recovery destination](https://drive.google.com/drive/folders/1jW4sMLZjG4TUGZggER0cwGbaI-cyODNi) holds the report, exact source index, full expected-hash queue, raw-source comparisons and reproducible evidence package. Existing canonical Drive sources are linked rather than overwritten. The old source map's individual-file locators are marked metadata-only unless freshly retrieved; fresh carrier equality is reported separately. Drive access controls are unchanged.

[Coordination](https://github.com/d6g8k5htny-coder/main/issues/86#issuecomment-5859441962) keeps this extraction disjoint from the active proof and prior-scanner lanes. No research computation or old verification suite ran, no Vault99 original was opened, and no mathematical acceptance or independence credit was assigned.
