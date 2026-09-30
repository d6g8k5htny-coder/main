# Reproduce and cite implementation plan

> For agentic workers: use superpowers:executing-plans for implementation and
> a separate nonauthor review before integration.

**Goal:** Make one calculation and its immutable source easy to reproduce and cite.
**Architecture:** Extend the existing static reader site; retain current primary
navigation and technical library. A pure ES module builds allowlisted immutable
source links without retrieving or executing their content.
**Tech Stack:** HTML, existing CSS, JavaScript modules, Python stdlib and Git.
**Spec:** [Reader requirements](../specs/2026-09-30-reproduce-cite.md).

## Global constraints

No new dependencies, scientific-source changes, settings changes or status
promotion. Existing navy/blue/gold palette and primary navigation remain.
Reference syntax validation never claims source existence or review acceptance.

## Review focus

- Invalid input after success clears the old reference.
- Encoded path separators, dot segments and control characters cannot change URL routing.
- JavaScript unavailable leaves useful static reading and citation instructions.
- Displayed reproduction commands execute the named immutable commit.
- Long commits/paths and secondary navigation remain readable on narrow screens.

## Tasks

- [ ] Add failing pure-module tests for reference validation, URL encoding and
  DOM stale-output behavior in `tests/test_source_reference.mjs`.
- [ ] Implement `buildReference(repository, commit, path)` in
  `docs/site/source-reference.mjs`; implement local form wiring in
  `docs/site/cite.mjs`; test before proceeding.
- [ ] Add Cite/Reproduce pages and shared secondary links, update README and
  reproduction entry, CFF metadata and route tests. Verify the displayed recipe.
- [ ] Record source-backed audit in `docs/REPOSITORY_STANDARDS.md`; retain rejected
  and deferred recommendations with their concrete reasons.
- [ ] Run affected checks and browser review, independent engineering review,
  then complete applicable stable-tree checks once. Publish PR, verify required
  hosted checks, guard integration by exact head, verify deployment and register
  one immutable delivery.
