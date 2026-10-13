# Full transformed-price budgets without a probability cutoff

**Object:** P15-FULL-TRANSFORMED-PRICE-20260924-v1.  
**Author:** OpenAI / ChatGPT. **Disposition:** author-side mathematical derivation; nonauthor review open.

[Research home](https://github.com/d6g8k5htny-coder/main) · [Math index](../../README.md) · [Original realization](../three_fronts_20260924/P15_REALIZED_COVERS.md) · [Demand-one counterexample](../three_fronts_20260924/P15_PRICE_BOUNDARY.md)

## 1. Exact scope and result

Use the original realized family, not arbitrary P15 inputs. H is a finite clutter of block supports, every edge has size at least two (empty H is allowed). The original coordinate blocks are disjoint, with

    |X_i|=a_i d_i+1,     a_i>=1, d_i>=2 integers.

The decreasing family D consists of U with |U intersect X_i|<=a_i in each block and with occupied-block support containing no H-edge. Its complete minimal original forbidders and its actual local restrictions are proved in P15-REALIZED-COVERS-20260924-v1, SHA256 `c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`.

Write O_K(D) for sets not partitionable into at most K members of D. Let K_H(d) be the least palette size with |P_i|>=d_i and empty intersection over every H-edge. For K>=K_H(d), the specified family of full-block generators {X_i} covers O_K(D), regardless of probabilities or prices. A generator g costs product_(v in g)c_v. The empty generator costs one; an empty family costs zero.

Put phi(p)=min(1,-log(1-p)), with phi(1)=1. Define

    p_star=1-exp(-1),
    h_star=3-log(3e-2),
    rho_star=1/h_star.                                   (F1)

**Theorem F.** For EVERY independent original probability vector p in [0,1]^X, every price vector 0<=c_v<=phi(p_v), and the SAME palette K>=K_H(d),

    covercost_c(O_K(D)) <= min(1,rho_star[-log mu_p(D)]).  (F2)

Before capping, the union of full-block generators has price at most rho_star times the global hazard. The constant rho_star is optimal uniformly over this entire realized class, even when alternative generator covers are allowed. Explicitly,

    0.84547981724898672067 < rho_star
                            < 0.84547981724898672068 < 6/7. (F3)

Thus 6/7 is a simpler valid rational factor. The probability ceilings 1/4 and 3/7 are no longer required. Demand d_i>=2 is still essential to this uniform family guarantee. This is not an unrestricted downset/prize theorem; palette size remains K_H(d), which can depend on H and d.

The prior probability<=1/4 theorem, SHA `3b79d2de60d77df9dd0d81cea60935d3dbceeb26fcf2d38d62667a425180a535`, gives the stronger factor16/27 on its smaller domain and should still be used there. The local-only3/7 note is a historical intermediate result, not a prerequisite here. No prior proof or counterexample is overwritten.

## 2. A general hazard interpolation lemma

Let A be any proper decreasing family on n>=1 coordinates containing the empty set. Put

    p_i(t)=1-exp(-t_i),     t in [0,1]^n,
    F_A(t)=-log mu_(p(t))(A),
    H_A=F_A(1,...,1)>0.

The probability is positive on this compact cube because the empty set is admitted and every p_i(t)<1. In one coordinate, after averaging over all others, its good-event probability is exactly

    mu(A)=A0+B0 exp(-t_i),     A0>=0, B0>=0.             (F4)

Here A0 is the probability of membership when that coordinate is present, and A0+B0 is the probability when it is absent. The inequality B0>=0 follows from decreasingness. Their sum is positive. Direct differentiation yields

    partial_i F_A = B0 exp(-t_i)/(A0+B0 exp(-t_i)),
    partial_i^2 F_A = -A0 B0 exp(-t_i)
                           /(A0+B0 exp(-t_i))^2 <= 0.    (F5)

Only SEPARATE coordinatewise concavity is asserted; joint concavity is not needed or claimed. Also F_A>=0. The chord inequality in coordinate i therefore gives

    F_A(t) >= t_i F_A(t with t_i=1).

Iterating across coordinates, keeping all factors nonnegative, proves

    F_A(t) >= (product_i t_i) H_A.                       (F6)

Now take arbitrary original probabilities p_i in [0,1], set t_i=phi(p_i), and let p'_i=1-exp(-t_i). Then p'_i<=p_i, including p_i=1. Decreasingness gives mu_p(A)<=mu_(p')(A), so

    -log mu_p(A) >= H_A product_i phi(p_i).              (F7)

This holds as an extended inequality if the left side is infinite. At a zero probability the product is zero; no division by it is performed. For any transformed prices,

    product_i c_i <= min(1, [-log mu_p(A)]/H_A).          (F8)

This is a PRICE estimate for the full-ground generator. It does not assert that this generator alone covers an arbitrary obstruction. That setwise property is supplied separately for the capacity construction below.

The constant1/H_A in the uncapped hazard comparison is exact: at every p_i=p_star and every c_i=1, the two sides of (F7) are equal. In particular a full-ground generator universally meets the capped hazard budget precisely when H_A>=1. The necessity uses that same reference vector; it does not infer coverage of arbitrary A.

## 3. Capacity families and the sharp worst case

For a capacity restriction A={S:|S|<=a} on n coordinates,

    H_(n,a)=-log P(Bin(n,p_star)<=a).                    (F9)

Our n=ad+1 and d>=2 imply n>=2a+1. Adding independent Bernoulli trials without raising the threshold can only decrease the good probability. It remains to compare odd majority tails. Let S~Bin(2a+1,p), with p>1/2. Directly adjoining two trials gives

    P(Bin(2a+3,p)<=a+1)-P(S<=a)
       =(1-p)^2 P(S=a+1)-p^2 P(S=a)
       =p(1-2p)P(S=a)<0.                                (F10)

The ratio P(S=a+1)/P(S=a)=p/(1-p) proves the second equality. Since a>=1, the maximal odd-tail probability occurs at a=1,n=3. Consequently

    H_(n,a) >= H_(3,1)
       =-log[3exp(-2)-2exp(-3)]
       =3-log(3e-2)=h_star>1.                           (F11)

For the last inequality, e>2 and

    e^2-(3e-2)=(e-1)(e-2)>0,

so 3exp(-2)-2exp(-3)<exp(-1). The argument includes all capacities and demands in the stated class; it is not a finite-grid extrapolation.

Equations (F8)-(F11) give, for each actual local restriction and its full-block generator,

    price({X_i}) <= rho_star [-log mu_p(D_i)],            (F12)

and the price is at most one. Thus it also meets the original P15-B local phi(q_i) premise. The inequality price<=q_i is NOT required and is in general stronger than what is proved here; at p_star in a three-coordinate block it would be false.

## 4. Global assembly on the original coordinates

First assume all local good probabilities are positive. They are independent across disjoint coordinate blocks. Global goodness implies every local restriction is good, without independence of crossing failures, so

    mu_p(D)<=product_i mu_p(D_i),
    sum_i [-log mu_p(D_i)] <= -log mu_p(D).               (F13)

Sum (F12). The unchanged setwise palette argument gives coverage at K>=K_H(d); (F13) gives its budget. If its union price exceeds one, use the empty generator. This proves (F2). If a local good probability is zero, global goodness has probability zero and the price-one cap proves the assertion. If global goodness is zero for crossing reasons alone, the same cap applies. Zero-price generators are kept for setwise completeness.

For the established six blocks of409 coordinates (a_i=1,d_i=408, all15 four-block supports), the SAME816-label cover now works for every independent p_v in [0,1] and every c_v<=phi(p_v). Its whole-ground chromatic number remains818. No coordinates are cloned, and the full crossing family is retained. At sufficiently low probabilities the earlier16/27 budget is better; on the whole cube (F2) is valid.

## 5. Sharpness and the exact demand boundary

Take one block, a=1,d=2,n=3, no macro edges and K=2. Every proper subset is two-decomposable and the full triple is not. At p_i=p_star and c_i=1, every possible generator has price one, including the empty generator. Any cover of the nonempty obstruction therefore costs at least one, and one generator attains it. The available hazard is h_star. Replacing rho_star by any smaller constant in (F2) would make its right side strictly less than one. This proves sharpness even among alternative covers, not only for our chosen full-block cover.

If d=1, n=a+1 and K=1, the sole obstruction is again the full block. At p_i=p_star,c_i=1 every cover costs at least one, but

    mu_p(D)=1-p_star^(a+1)>1-p_star=exp(-1),

hence its hazard is less than one. The universal same-palette transformed-price guarantee is false in this demand-one family. This preserves and generalizes the earlier two-coordinate counterexample; it does not retract original P15-B, whose local-budget premise is missing in that case.

Thus d>=2 is sufficient for the realized capacity family on the full probability cube, while a uniform statement admitting all d=1 examples would be false. This is not a necessity claim for every multi-block instance with some demand one; other constraints may change its budget.

## 6. Exact numerical certificate and validation limits

The proof of the simpler bound rho_star<6/7 can be made with short rational certificates. The exponential series and its positive geometric tail give

    e < 31967/11760 < 87/32,
    87/32-31967/11760=11/23520.

The positive Taylor polynomial through degree5 satisfies

    sum_(j=0)^5 (11/6)^j/j! -197/32=26081/933120>0.

Therefore 3e-2<197/32<exp(11/6), giving h_star>7/6 and rho_star<6/7.

For (F3), full_price.py bounds e by its positive series plus a geometric tail. It encloses logarithms using binary reduction and

    log x = 2 sum_(j>=0) t^(2j+1)/(2j+1), t=(x-1)/(x+1),

on 1<=x<=2, with remainder bounded by2 t^(2m+1)/[(2m+1)(1-t^2)]. Exact Fraction endpoints are propagated monotonically through the binomial tail and logarithm, then reciprocated and rounded outward to20 decimal places. No floating-point value determines a certified endpoint.

The finite suite checks the recurrence against subset enumeration, the odd-majority identity, the coordinatewise derivative sign, exact constants and endpoint conventions, heterogeneous probability grids, and preservation of the demand-one counterexample. These checks support algebra and implementation; the continuum hazard inequality is the written proof, not a conclusion from the grid. No independent reviewer, formal checker, Gaussian/RN closure or unrestricted prize acceptance is claimed.

## 7. Source and reconnaissance

Project inputs are the unchanged realized-cover and price-boundary sources linked above, original P15-B (Drive19D-eHQAIXMGGy2ThZUfZ0GGjIKWm5C2j; SHA9b18b6e9abc90d18deef06ab12e3aa7794dad40e1618d99daa88e369c300e8c3), and the published restricted-price note (Math-PR4). This proof derives (F4)-(F13) directly rather than importing a theorem from an abstract.

Primary-source reconnaissance on24September2026 read the arXiv abstracts of Gunby–He–Narayanan, Down-set thresholds,2112.08525v2, and Warnke, Note on down-set thresholds,2310.11662v2. Their downset/expectation-threshold results provide neighboring context, not the hazard-transfer theorem above, and do not license arbitrary-downset closure. The bounded search did not establish novelty or historical priority. Attribution to classical separate concavity, Bernoulli conditioning, and binomial identities is not replaced by a project-specific invention claim.
