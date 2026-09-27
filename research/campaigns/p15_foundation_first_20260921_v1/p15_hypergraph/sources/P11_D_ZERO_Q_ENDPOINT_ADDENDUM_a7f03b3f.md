# Nonblocking endpoint amendment for the attached P11-D

Target: archive ee4a16e1bb068bb28782391042c238626185e4acc9e6ad51a424bde2f8a64b57; relative path proofs/04_ALL_FINITE_PRIMITIVE_RANKS_HAZARD_CLOSURE.md; file SHA256 0ce1352b67899d3cd805def6ea25cc7cee8a33ac6dfbb5e73bd897565d60e1ea.

## Exact issue

The induction is stated for Q <= e_r, including Q=0. Its intermediate display (D2) ends in the strict inequality sum_C p_v < 3Q. Taken literally at Q=0, even an empty extracted core has 0 < 0, which is false. This is an endpoint wording issue, not a counterexample to the stated non-strict cover theorem.

## Complete repair

Before the positive-Q induction, insert:

> If Q=0, every original edge e has product probability zero. Since 0 <= z_v <= A p_v with finite A, its price product is also zero. Color every vertex with one color and use the original edge family as the generator family. This is a coloring certificate of cost zero with generator sizes in {2,...,r}. The claimed bound holds. Henceforth assume Q>0.

Now all stated strict positive-Q estimates can be read literally. Alternatively replace the strict zero-sensitive displays by non-strict inequalities with a positive-Q qualifier.

The original source has NOT been overwritten. The amended proof is not externally reviewed. The exact counterexample and repaired zero-cost certificate are exercised by the new checker.
