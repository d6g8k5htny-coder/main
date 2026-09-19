# P11 execution limitations and corrected development defects

1. The first rank-three antichain test compared a size-ordered tuple to a lexicographically normalized tuple. This wrongly omitted53 valid antichains and tested113 instead of166. The production enumeration now compares sets, certifies exactly166 nonempty-edge rank<=3 antichains on4 coordinates, and preserves the first result as development history. This was a test-coverage defect, not a counterexample to a written proof.

2. The first fault fixture for dropping the graph second-moment factor2 used a star. That fixture returned UNDETECTED: the weakened bound still happened to hold there. It was NOT counted as a rejected mutation. The final fixture is a10-clique at p=1/10, where exact moments refute the altered bound. An intentional survivor remains in the production harness, with required UNDETECTED/exit0 classification.

3. Finite test weights are rational lower approximations to phi(p), sometimes strictly above p. The library computes finite certificates for supplied weights; the theorem's w<=phi(p) admission is mathematical and not replaced with a floating-point logarithm test. Exact exponential upper enclosures certify returned costs in nontrivial examples.

4. The large rank-three palette makes many small-instance obstruction families empty. Such tests validate formulas and coverage mechanics but do not constitute numerical evidence of a universal theorem. The written proof supplies the universal quantifiers. The65-clique and130-leaf graph-hierarchy fixtures have genuinely nonempty64-piece obstructions.

5. Source/author exposure is disclosed. External reviewer count0; no theorem-prover compilation; no agent independence inferred from two implementations. No q0 edits or external submissions.

6. Unknown total token billing, agent-hours and full compute resource use remain null. Subprocess durations and captured outputs are recorded. No claim that a successful finite test suite measures truth probability.
