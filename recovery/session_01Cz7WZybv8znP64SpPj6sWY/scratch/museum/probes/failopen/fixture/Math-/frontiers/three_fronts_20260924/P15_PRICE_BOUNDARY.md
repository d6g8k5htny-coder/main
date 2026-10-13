# Why the realized-cover theorem cannot simply use every transformed price

**Object:** P15-PRICE-BOUNDARY-20260924-v1. Author: OpenAI / ChatGPT.
This is an exact counterexample to one proposed extension, not to P15-B or to
an unrestricted large-palette theorem.

The realized-cover theorem deliberately assumes c_v<=p_v. Consider its smallest
one-block member: a_1=d_1=1, X={x,y}, no macro edges. Its good sets are the empty
set and singletons, its palette optimum is K_H(d)=1, and its one-piece obstruction
is exactly {X}. The prescribed cover is {{x,y}}.

Take independent original probabilities p_x=p_y=1/2 and prices c_x=c_y=2/3.
These are admissible TRANSFORMED prices: phi(1/2)=log 2>2/3. For example,

    log 2=2 integral_0^(1/3) 1/(1-t^2) dt > 2/3.

Nevertheless mu_p(D)=3/4, and

    min(1,-log mu_p(D))=log(4/3)<1/3,

using log(1+x)<x for x>0. Every cover of the obstruction {X} must contain a
generator g subset X. The possible generator costs are1,2/3,2/3,4/9 for the empty
set, the two singletons, and X respectively. Additional nonnegative-price
generators cannot reduce the minimum. Therefore the EXACT optimal cover price is

    covercost_c(O_1(D))=4/9 > 1/3 > log(4/3).

Thus replacing c<=p by c<=phi(p) in the realized-family theorem while keeping its
same palette K_H(d) is FALSE in general, even if arbitrary alternative covers
rather than the prescribed cover are permitted. This is stronger than merely
saying that the existing proof does not establish the extension.

It does not contradict the original P15-B amalgamation: that source separately
requires each local cover to meet its local phi(q_i) budget, which fails here.
It also does not refute a theorem allowing more colors: with two colors this
particular whole ground set is properly colorable, the obstruction is empty,
and the empty family has price zero. No conclusion about the original408-scale
or unrestricted prize follows from a one-color counterexample.

The accompanying three finite tests enumerate all generator families on the
actual two-coordinate ground set, compute the product-measure probability exactly,
and check the rational comparisons used above. The two elementary logarithm
inequalities are proved in this note, not replaced by floating-point evaluations.
