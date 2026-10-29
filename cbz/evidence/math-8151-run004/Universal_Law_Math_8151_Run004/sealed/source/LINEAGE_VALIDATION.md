# Declared-author lineage validation

Scientific effect: **NONE**. This is an engineering successor to the validator,
not an alignment acceptance, new proof, or scientific-status authority.
Implementer: OpenAI / GPT-6 Astra Pro, ChatGPT session
`github-meaningful-advance-20261005T1610Z`, acting under Dylan Roy's delegation.
Pickup: main#229 comment 5998762623.

## Observed gap and bounded repair

The previous `check_alignment` compared only `author` and `reviewer`. It ignored
`proposal_authors` even when a record expressly named an unknown or same-provider
proposer. It also treated literal UNKNOWN identities as ordinary nonempty strings.
Thus an ACCEPTED disposition could pass the structural validator despite declared
lineage contradicting [SCOPE's review contract](SCOPE.md#review-contract).
This is not evidence that a controller accepted a false review or that a theorem
passed on an invalid proof. Reviewers withheld the relevant candidate correctly.

The successor validates `provider`, `family`, and `agent` for the main author,
reviewer, and **every declared proposer**. Values must be nonempty strings.
Whitespace is collapsed and comparisons are case-insensitive. The following exact
normalized placeholders are rejected: `unknown`, `unverified`, `unspecified`,
`tbd`, `n/a`, `none`, `null`, `?`, `not known`, and `not available`.
Every author's three normalized fields must differ from the reviewer's respective
fields, preserving the existing single-author distinctness rule.

An absent `proposal_authors` retains the historical single-author record shape.
An explicit empty list is valid. A present collection must be a list of identity
objects. Each proposer's optional `targets` must be a nonempty, duplicate-free
list of names in the reviewed target inventory. Omitting that field does not
exempt the proposer from the package's identity checks. Contributor `pr` labels
are descriptive metadata, not authentication.

## Boundaries that remain external

This validator cannot discover undeclared proposers, identify model aliases,
authenticate a claimed provider/session, or infer a provider from a role or app
name. It does not prove that all contributors were listed. A controller must still
read the immutable source, native reviewer evidence and contributor history.
Unknown lineage must not be replaced by an invented provider to pass the check.
This change neither authorizes a waiver nor creates a composite-review rule.

Manifest/scope digests, full target coverage, ACCEPTED disposition and immutable
evidence references remain required. No existing withheld record is edited by
this change. Old accepted records remain records of their old exact manifests;
the new manifest must not be substituted into them without a fresh review.

## Source preservation and regression checks

No Lean declaration, target, toolchain, dependency revision, historical proof,
workflow, axiom audit, build/recheck command or executable negative control is
changed. `source_check` additionally requires this note and
`tests/test_alignment_lineage.py` to be hash-bound. The two older Python tests
that pin the whole gate now pin this separately scoped successor; their proof
and toolchain pins and all other test logic are unchanged.

The new suite has 23 test methods, including role/field/placeholder subcases,
malformed identities, matching proposer identities, valid legacy and declared
proposer records, retained digest/coverage checks and omitted source bindings.
Against the old gate it reported 132 failing subcases and 8 malformed-party
errors in each Python mode. After repair all 23 methods passed in both modes.
The fixtures are synthetic, not fabricated review evidence.

```sh
python3 -B -S -m unittest discover -s formal/tests -p test_alignment_lineage.py -v
python3 -B -O -S -m unittest discover -s formal/tests -p test_alignment_lineage.py -v
python3 -m unittest discover -s formal/tests -v
python3 -O -m unittest discover -s formal/tests -v
python3 formal/gate.py
python3 formal/gate.py --execute
```

Only the focused suite was run in the implementer's partial staging checkout at
publication. Full-suite, exact-source and Lean execution claims require their
actual complete-checkout results; the complete suite size is not a protocol constant; use the actual current-run count.
Local focused tests are not a Lean kernel replay or independent review.
