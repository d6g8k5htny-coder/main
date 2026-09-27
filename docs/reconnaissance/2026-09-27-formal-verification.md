# Reconnaissance memo — formal-verification integration

Date: 2026-09-27. Author: ChatGPT / OpenAI. Scientific effect NONE. This memo implements the existing external-reconnaissance practice; it does not create a new authority hierarchy.

## Findings and decisions

Lean's [official axiom reference](https://lean-lang.org/doc/reference/latest/Axioms/) explains that new axioms are assumptions, `sorryAx` admits arbitrary statements, and `#print axioms` reports dependencies transitively. Native-evaluation proofs introduce additional compiler-trust axioms. Decision: compilation alone is insufficient; the pilot allows only propext, Classical.choice and Quot.sound, rejects all others, records actual axiom reports and tests the rejection path.

The [official Blueprint README](https://github.com/PatrickMassot/leanblueprint/blob/master/README.md) describes `checkdecls` as checking that named declarations exist. It does not establish natural-language equivalence. Decision: keep Blueprint links and a separate independent statement/alignment review. The pilot's link-only check is not represented as a full Blueprint website build.

The [mathlib 4.34.1 source](https://github.com/leanprover-community/mathlib4/tree/d13f23b723b8a846827a245b89c10fc7d3f11612) specifies Lean 4.34.1 and its dependency manifest. Decision: bind the toolchain, complete dependency lock, installed dependency commits, exact source manifest and checked Git commit. First hosted installation verified that these pinned dependencies were retrievable. The current run, not this memo, determines compilation success.

Consensus retrieved [Aria, Xie et al., arXiv:2510.04520](https://arxiv.org/abs/2510.04520), a 2025 preprint about dependency-graph-based autoformalization. Its reported compilation and correctness rates differ. Treat this as contextual research, not as a peer-reviewed certification of our system or a benchmark for this pilot. Decision: grounded definitions and independent semantic review remain necessary even when generated Lean compiles.

## Source-specific judgment

Recovered GP-FOR-192 supplies thirteen narrow algebraic/conditional statements suitable for a first executable package. Its historical source label is not proof of successful compilation: the first real run failed and is retained. Compiler-compatible successors must preserve statements and disclose their changes.

The proposed glossary equivalences were not adopted. Elder selection requires its own point-process/pairing/conditioning definitions; a lifetime asymptotic coefficient is not automatically a Hermite-expansion coefficient. A missing formal translation is an obligation to resolve, not proof of novelty or ill-definition.

The proposed coefficient pilot was narrowed rather than mislabeled. Rational arithmetic on two endpoints cannot establish that an analytically defined coefficient lies between them. The next numerical lane must supply the definition and all rigorous enclosure/remainder steps. Likewise GP214's upper bound cannot satisfy the formal transfer theorem's lower-normalizer premise.

## Deferred, not represented as delivered

Coq/Rocq, Metamath, additional independent Lean checkers and AI proof generators are optional interoperability lanes, not duplicated now. No claim is made that AlphaProof is a generally callable CI service, that an external prover ran, that a paper was submitted, or that an independent reviewer participated. Their future use must produce identifiable execution/review artifacts and preserve the same scope and provenance contracts.

## Revisit conditions

Reassess on toolchain changes, new upstream numerical/measure-theoretic libraries, a completed alignment review, an axiom-audit false negative, or an actual prover integration. Do not use a moving documentation URL as an execution pin: the implementation's exact Git commits and source manifest are the replay identity.
