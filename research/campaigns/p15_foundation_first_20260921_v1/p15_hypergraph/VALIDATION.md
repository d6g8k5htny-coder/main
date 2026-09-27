# P15 finite-rank candidate validation

Claim: `CLAIM-P15-FINITE-RANK-20260921-ROOT01`.
Environment: CPython 3.11.16, standard library only, 2026-09-21.

The analytic proof is `PROOF.md`, SHA-256
`f7994948d7f3b81d73dd700296cb93184c6f5a21b8664d1dd96ace59eed80f1c`.
It is unchanged by the finite companion. The nonauthor same-provider review
in `review/FOUNDATION_REVIEW.md` checked that exact proof and the listed
foundations, finding no substantive gap. Its organizational independence
credit is zero; its finding is not external acceptance or formal verification.

The final source/dependency manifest SHA-256 is
`c2c74d00f3c8a79789c8587c2e6347adffedaaa5ee24cb70c339834b8a44b091`.
All nine copied and repository source bodies passed exact identity guards,
their inventory identities matched, and no selected source ID appeared in the
pinned exclusion metadata. Metadata and source bytes were checked again after
each replay. The manifest explicitly includes the P11-D zero-Q amendment and
transitive P11-A/P11-B foundation edges. Hash matching proves identity only.

## Observed finite results

- `python -B -m unittest -v test_finite.py`: **22 tests passed**.
- `python -B -O -m unittest -v test_finite.py`: **22 tests passed**.
- `python -B check.py --repo … --output outputs/normal.json`: **13 scenarios passed**.
- The same replay with `-O`: **13 scenarios passed**.
- Normal and optimized receipts are byte-identical: 9,463 bytes, SHA-256
  `2d88a4352e7aba16f7b9338c3f8909de85b6a2e4d0d43884830ab6b344b969ff`.

Representative exact controls:

- The complete 3-uniform hypergraph on five singleton blocks has a genuine
  two-color obstruction. A zero-priced generator still covers this set when
  all coordinate probabilities are zero; deleting it is rejected.
- A case with global failure `7/16` and macro failure zero retains its eight
  zero-priced lifted transversal witnesses. This tests the intermediate
  endpoint, not just global `Q=0`.
- A singleton-forbidden block disappears with every incident macro edge;
  shrinking the deleted triple to a pair changes the actual good family.
- Incomplete lifting gives actual failure `1/1000` despite macro occupancy
  failure one. The exact complete-family validator rejects this input.
- In a shared-coordinate example, the actual good probability is `19/64`;
  multiplying local-good and macro-good probabilities gives the incorrect
  `999/4096`.
- Two two-color local blocks coupled by a macro edge create the complete
  graph on four original coordinates. Four colors work; replacing the product
  palette by its maximum of two fails.
- Occupancy covers can contain the empty generator, at price one. An empty
  generator family instead costs zero and covers nothing. Refinement after
  this lifting cannot halve the empty generator's price.
- Changing a source copy or manifest causes refusal. Missing local/macro
  covers, overlapping blocks, invalid scalar endpoints and incomplete lifting
  are rejected in normal and optimized Python.

The finite scenarios intentionally test supplied small coloring/cover
interfaces; their covers need not satisfy the analytic theorem's low-Q
half-budgets. They verify the composition mechanism and catch specific invalid
shortcuts. The analytic half-budgets and arbitrary-size existence conclusion
come from the written source proofs and successor proof, not these cases.

No source archive was executed, no shared repository file was edited by this
subtask, and no canonical claim status changed. Root handles the separate
claimed GitHub review delivery. Foundation defects discovered later block
downstream acceptance irrespective of these passing finite checks.
