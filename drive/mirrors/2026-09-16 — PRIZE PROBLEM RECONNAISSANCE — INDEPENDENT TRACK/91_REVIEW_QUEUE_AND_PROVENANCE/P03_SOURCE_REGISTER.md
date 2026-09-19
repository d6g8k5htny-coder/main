# Sources and priority ledger — checked 2026-09-16

Only original research, first-person mathematical arguments, official proof-library documentation, and clearly labeled first-party reports are used mathematically. Search-result absence is not a novelty proof.

## S01 — Alexander Walker, arXiv:2203.06045v2 (2025)

https://arxiv.org/pdf/2203.06045v2
Title: Integer Sets of Large Harmonic Sum Which Avoid Long Arithmetic Progressions.
Read the primary PDF and visually inspected page3 (zero-index page2). The modular digit construction, canonical alphabets, Walker55 benchmark, and digital approximation theorem are prior work. Remark1.3's concrete base7 {0,2,5} example is contradicted by the exact integer witness in PR-AP-008. This does not refute Theorem1.2, Theorem2.1, or the mod-free base55 construction. No claim about whether someone previously noticed the error.

## S02 — Classical Behrend construction / modern primary exposition

Kevin O'Bryant, Sets of integers that do not contain long arithmetic progressions, arXiv:0811.3057.
https://arxiv.org/abs/0811.3057
Official mathlib development:
https://leanprover-community.github.io/mathlib4_docs/Mathlib/Combinatorics/Additive/AP/Three/Behrend.html
The sphere construction is classical; the needed version is fully rederived in PR-AP-009. No local Lean compilation was performed. The official documentation is a reference, not a compilation receipt for this package.

## S03 — Bloom and Sisask

Breaking the logarithmic barrier in Roth's theorem on arithmetic progressions, arXiv:2007.03528v2.
https://arxiv.org/abs/2007.03528
Consumes only the published r_3(N)<<N/(log N)^(1+c), c>0, to establish summability/attainment for k3. No claim that this is the latest or sharpest r3 bound; no independent proof of the source theorem.

## S04 — Classical finite van der Waerden theorem

Bruce M. Landman and Aaron Robertson, On Generalized Van der Waerden Triples, arXiv:math/9910092.
https://arxiv.org/abs/math/9910092
Its stated classical van der Waerden theorem is the input to the automata live/dead argument. We use existence of W(k,s), not any numerical value or computational estimate.

## S05 — Will Sawin, MathOverflow comments, 14 March 2025

https://mathoverflow.net/questions/489375/why-is-erd%C5%91s-conjecture-on-arithmetic-progressions-not-discussed-much-and-is-t/489376
First-person original mathematical explanation: separate finite extremizers in base4-scale blocks, compare their reciprocal sum to sum r_k(4^j)/4^j, and use monotonicity to obtain a necessary logarithmic cardinality bound. This was found after the local base3 derivation. The substance of PR-AP-012's separated-profile idea is therefore PRIOR WORK, not claimed new. The factor6 presentation and downstream consequences are locally proved but their historical novelty is unestablished.

## S06 — Joseph L. Gerver, Proc. AMS 62 (1977), 211--214

The sum of the reciprocals of a set of integers with no arithmetic progression of k terms.
Primary paper text reproduced at:
https://studylib.net/doc/18211266/the-sum-of-the-reciprocals-of-a-set-of-integers-with
Some formulas in the mirror are poorly rendered; it is used for priority/context, not as a substitute for our written algebra. It already discusses the uniform harmonic supremum and head/density distinctions. An attempted original publisher PDF fetch failed; no unseen publisher bytes are treated as received.

## S07 — Recent first-party head/block report — NOT RECONSTRUCTED

https://computoergosum.com/en/principia/erdos-169-certs/index.html
The accessed report describes a head-plus-Behrend improvement above Walker55, with a claimed finite lower sum >4.439753474215620 and formalization links. The report is relevant prior art. The linked raw formalization files could not be retrieved/rebuilt in this run (web cache/download failures and unavailable container DNS). Therefore this is a first-party claim, not an independently confirmed result here. We do NOT claim our much smaller relative improvements beat it.

## S08 — Earlier efficient reciprocal summation

Walker S01 cites Baillie--Schmelzer (2008) and an existing implementation for digit-restricted reciprocal sums. Phase02's SOURCE_REGISTER is preserved inside the unchanged baseline ZIP. PR-AP-011 extends this package's exact enclosure machinery to transient DFAs; it does not claim to invent efficient digit-series summation or automata decidability.

## Disposition summary

- Correctness of the explicit (2,133,264) counterexample is elementary and executable; historical novelty of noticing it is unknown.
- The carry-state methods are standard in principle; the explicit theorem/certifier implementation is new to this project, not asserted historically new.
- Sparse-tail replacement and finite-state nonextremality: complete local proofs with classical ingredients; external review and broader priority search needed.
- Extremal-profile bridge: core strategy matched to Sawin2025 and must be credited. No prize progress by renaming a known reduction.
- Complete bounded alphabet classification and exact code are reproducible artifacts, not an unrestricted harmonic record.
- No new substantive RH, Collatz, Sidon, Talagrand, or r_k asymptotic theorem was obtained in Phase03.
