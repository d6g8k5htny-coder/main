# P14-C — Unboundedly many individually necessary constraints with bounded cover width

Author-side exact construction and proof. No novelty or unrestricted prize closure is claimed. Large row families are defined by a formula, NOT stored/enumerated in their entirety.

## 1. Construction

Let s>=5. Partition X into H, of size817, and T, of size2s. Fix one s-subset S0 of T. Let F contain every s-subset of T EXCEPT S0 and its complement. For each S in F impose the strict inequality

    (3/5)|U intersect H| + (1/s)|U intersect S| <1.       (1)

The number of rows is d=binom(2s,s)-2. Every row is individually necessary: U=S violates row S with equality and satisfies every other row, since two different s-sets intersect in at most s-1 points. Therefore deleting duplicate/dominated/redundant input rows does not remove the large row count.

There are no singleton-forbidden coordinates. The column envelope a is3/5 on H and1/s on T.

## 2. Exact fractional cover width tau_*=2

Choose any complementary pair S1,T\S1 in F and give both rows weight1. Their combined normalized coefficient profiles cover each tail coordinate once and each heavy coordinate twice. This is a primal certificate of mass2.

Give each tail coordinate dual weight1/s and each heavy coordinate dual weight0. Every row then has dual load1 and the objective is2. Weak duality proves tau_*=2 exactly, without enumerating F.

Hence P14-B supplies K=816 and no dilution, uniformly in s, d, all product probabilities, and original witness rank.

## 3. Not a single weighted threshold in disguise

With no heavy vertices selected, S0 and T\S0 are both good. S1 and T\S1 are both bad. For ANY putative linear budget w(U)<c, summing the two good inequalities gives w(T)<2c, whereas summing the two bad inequalities gives w(T)>=2c. Contradiction.

This excludes a representation by one strict linear inequality, even allowing arbitrary real coefficients. It does not exclude every other structured representation or claim to be outside all earlier compatible-gate classes.

## 4. Nonempty actual obstruction

Every good set contains at most one heavy coordinate, because two contribute6/5 in every row. Thus H itself cannot be covered by816 good sets:817 heavy coordinates would require at least817 parts. The theorem's 816-piece obstruction is nonempty.

The largest minimal forbidden set has size s. Included tail s-subsets give witnesses of size s. The other minimal witnesses are heavy pairs and one heavy coordinate with ceil(2s/5) tails. All smaller tail sets can be extended to a row in F; for sizes< s there is more than one possible extension, and at most two rows were removed. Thus this witness description is exact for s>=5.

The witness-overlap graph is connected: heavy pairs connect H, and every tail coordinate lies in a one-heavy mixed minimal witness. No disjoint-component reduction is being used to create the example.

## 5. High activation with a high good-event probability

Let p_h=2^(-17) on H and p_t=1/4 on T. Failure is contained in the union of 'some heavy coordinate selected' and 'at least s tail coordinates selected'. Hence

    Q<=817/131072 + 3/(2s+3) <1/4    (s>=5).          (2)

For the second bound, Y~Bin(2s,1/4) has mean s/2 and variance3s/8. The one-sided variance bound at distance s/2 gives P(Y>=s)<=3/(2s+3). Formula(2) is a proof for every s, not a sampled claim.

The empty-activation probability is

    pi0=(1-2^(-17))^817 (3/4)^(2s).

This tends to zero. At s=20, even816*pi0<1 follows from816*3^40<4^40. Since minimum witness size is2, the older exact-cell condition816*pi0>=1 fails. Its componentwise version also fails because the witness system is connected. Failure of that sufficient criterion is NOT a counterexample to its theorem.

## 6. Concrete s=20 instance

There are857 coordinates and137846528818 individually necessary resource rows. The prior raw-row palette408d is56241383757744; the generic fractional certificate yields816.

The exact good probability can be evaluated without expanding the rows. Write Z~Bin(40,1/4), let u=2^(-17), and let h=817. Then

    P(D)=(1-u)^h [P(Z<20)+2(1/4)^20(3/4)^20]
          +h u(1-u)^(h-1) P(Z<8).                    (3)

The bracket's extra term accounts for exactly the TWO permitted tail sets of size20. Two or more heavy selections are never good. For a single heavy coordinate, strict capacity requires fewer than8 tails.

The companion computes(3) as a rational number and computes a rational upper cover cost at inflated prices2p for the full filtered family by binomial coefficient grouping. No enumeration of137 billion rows or2^857 outcomes is claimed.

## 7. Extra scalar-sandwich sharpening on this particular family

The exact good sets above all have a(U)<=1. They include S0 with a(S0)=1, so kappa=1 would violate the required STRICT upper sandwich. But kappa=409/408>1 is valid, yielding K=409 from P14-A. The obstruction remains nonempty. The companion checks both generic fractional K816 and this tailored K409 certificate.

This last sharpening uses a full description of the good sets, not merely the two-row fractional certificate. It should not be inferred for arbitrary systems with tau=2.
