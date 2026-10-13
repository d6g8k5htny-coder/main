# Nonauthor analytic review — P15 full transformed-price theorem

Scientific effect: **NONE**. This file does not change `lemma_closed`, prizes, premises, or `claims/LANDING_CLAIMS.json`. It does not accept an unrestricted downset theorem or an unrestricted P15 prize.

## Pickup

Math- issue 57 is in scope. The immutable subject is present at the pinned blob, byte length, and SHA256. This run is the single bounded review. No child agent was launched.

## Claim

| Field | Value |
|---|---|
| Object | `P15-FULL-TRANSFORMED-PRICE-20260924-v1`, Theorem F |
| Read at | `53320cbc8a4d15b9f675885a1d7760656bcc3b4f` |
| Path | `frontiers/full_price_20260924/PROOF.md` |
| Blob | `582180e41dca0ad815ad0f18574df42040912149` |
| Size | 11352 bytes |
| SHA256 | `87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9` |
| Author of the object | OpenAI / ChatGPT |
| Scope | Realized disjoint capacity/clutter family; `a_i>=1`, `d_i>=2`; every independent `p` in `[0,1]^X`; `0<=c_v<=phi(p_v)`; same palette `K>=K_H(d)`; uniform factor `rho_star=1/h_star` |
| Excluded | Arbitrary downsets, demand one as a uniform extension, unrestricted prize closure, and any flip of the landing disposition |
| State | Delivery record on publication of this file |

`frontiers/full_price_20260924/PROOF.md` was not edited.

## Reviewer provenance

| Field | Value |
|---|---|
| Provider | xAI |
| Model | Grok 4.7 (`grok-4.7-high-fast`) |
| Agent | Cursor cloud run `bc-3902d28e-89f5-40ac-9c3a-1afccff355ed` |
| Run URL | https://cursor.com/agents/bc-3902d28e-89f5-40ac-9c3a-1afccff355ed |
| Account wrapper | Cursor application, owning user Electric_Universe_Theory |

Organizational independence is not awarded. Provider separation is narrower: the proof's author is OpenAI, and this reviewer is xAI Grok, so the dispositions are a nonauthor technical review.

## Source exposure

Read in full: the immutable proof above.

Read as the cited setwise dependency, and checked for the local restriction and the full-block cover: `frontiers/three_fronts_20260924/P15_REALIZED_COVERS.md`, 11467 bytes, SHA256 `c0dbb821fb57b685a20cc321e4074456f39a9697f3104afd1f728e4732179bb9`. That is the digest named in the proof. The check uses (P1), (P2), and Theorem P2 there. It does not accept that note's `c_v<=p_v` price theorem, and it does not widen the note beyond the realized family.

Read as the two-coordinate predecessor of the demand-one boundary: `frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md`, 2266 bytes, SHA256 `498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b`.

Author unit tests were not used as evidence. No SMT transcript is in this tree; the issue states that the existing SMT pilot checks seven algebraic substeps and is not theorem acceptance. This review does not treat that pilot as acceptance.

## Method

Sections 1–6 of the proof were checked as written arguments. Separately, `algebra_check.py` (standard library only) checks the separate-derivative numerator, the odd-binomial form of (F10) and its sign on and off `(1/2,1)`, fixed-threshold augmentation, the rational `6/7` certificate, an independent enclosure of (F3), and the demand-one comparison `e>1`.

Command:

```sh
python3 -B -S reviews/p15_full_price_nonauthor_20260926/algebra_check.py
```

Script SHA256: `68c66cbeaab7941cafa7cf2d6f93d2e98543b5a040b7eede1433ffa2127cf4e9`. The run printed `all certificates passed`.

## Dispositions

| Slice | Disposition |
|---|---|
| 1. Coordinatewise hazard transform, separate concavity, chord, `p=0,1` | **ACCEPT** |
| 2. Capacity local event, binomial ratio, odd-majority worst case | **ACCEPT** |
| 3. Global hazard direction and disjoint-block independence | **ACCEPT** |
| 4. Full-block cover, same palette, realized family | **ACCEPT** |
| 5. Sharpness at capacity one, demand two, alternative covers | **ACCEPT** |
| 6. Demand-one boundary and exact theorem scope | **ACCEPT** |

Overall exact-scope disposition: **ACCEPT** Theorem F on the realized family stated in section 1 of the proof.

## 1. Coordinatewise hazard — ACCEPT

Let `A` be a proper decreasing family on `n>=1` coordinates that contains the empty set. On `t in [0,1]^n`, `p_i(t)=1-exp(-t_i)<1`, so the product measure of the empty set is positive and `F_A=-log mu_{p(t)}(A)` is finite. Properness at the all-ones vector gives `H_A=F_A(1,...,1)>0`, because every nonempty cylinder still has positive probability and some set lies outside `A`.

Fix coordinate `i` and average the others. Decreasingness splits the average into sets that remain admissible when `i` is present, with mass `A0`, and sets that are admissible only when `i` is absent, with mass `B0`. Thus

`mu(A)=A0+B0 exp(-t_i)`, `A0>=0`, `B0>=0`, `A0+B0>0`.

`B0>=0` is the decreasing constraint. Differentiating in `t_i` produces (F5). The numerator of the second derivative is `-A0 B0 exp(-t_i)`, so the second pure partial is `<=0`. That is separate concavity in each coordinate. Joint concavity is not used.

`F_A>=0` because `mu<=1`. A univariate concave function therefore lies above the chord from `0` to `1`, and the value at `0` drops out:

`F_A(t) >= t_i F_A(t with t_i=1)`.

Iterating in the remaining coordinates, with every `t_j>=0`, gives (F6).

For `p_i in [0,1]`, `phi(p)=min(1,-log(1-p))` with the stated value `phi(1)=1`. The comparison probabilities `p'_i=1-exp(-phi(p_i))` satisfy `p'<=p`. At `p=0`, `phi(0)=0`. On `(0,1-e^{-1}]`, `phi(p)=-log(1-p)` and `p'=p`. On `(1-e^{-1},1]`, `phi(p)=1` and `p'=1-e^{-1}`. No coordinate of `p'` equals `1`, so `F_A(phi(p))` never evaluates `log 0`. Coupling `p'` as a subset of `p` and using decreasingness gives `mu_p(A)<=mu_{p'}(A)`. If `mu_p(A)=0`, (F7) is the extended inequality `infinity >= H_A product phi(p_i)`. A zero factor `phi(p_i)=0` makes the product zero and is not used as a divisor.

Every admissible price satisfies `0<=c_i<=phi(p_i)<=1`, so the full-ground product is at most `1` and, by (F7), at most `[-log mu_p(A)]/H_A`. That is (F8), again in the extended sense when the hazard is infinite. Equality holds in (F7) at `p_i=1-e^{-1}` and `c_i=1`. The same reference vector is where the uncapped factor `1` meets `min(1,-log mu)` for every transformed price vector if and only if `H_A>=1`: sufficiency is `1/H_A<=1`, and necessity is the product `1` against hazard `H_A`. (F8) itself remains a price bound for that one generator. Coverage of an obstruction is a separate setwise fact.

## 2. Capacity worst case — ACCEPT

On one block the realized local restriction is exactly `{S:|S|<=a}`, by (P2) of the pinned realized-cover note. With `n=a d+1` and `d>=2`,

`n-(2a+1)=a(d-2)>=0`,

so `n>=2a+1`. For a fixed threshold, one added Bernoulli trial changes the good probability by `-p P(Bin(n)=a)`. At `p in (0,1)` this is strictly negative. Raising `n` above `2a+1` therefore strictly raises the hazard at `p_star`. The minimum of `H_(n,a)` occurs at `n=2a+1`.

For `S~Bin(2a+1,p)`, adjoining two trials gives

`P(Bin(2a+3,p)<=a+1)-P(S<=a)=(1-p)^2 P(S=a+1)-p^2 P(S=a)`.

The central binomial identity `C(2a+1,a)=C(2a+1,a+1)` makes the ratio of those two point masses `p/(1-p)`. Substitution then produces the second equality of (F10),

`p(1-2p) P(S=a)`.

The algebraic reduction is valid for every `p in [0,1]`. The factor `P(S=a)` is positive on `(0,1)` and zero at `p=1`. The factor `1-2p` is negative only for `p>1/2`. Strict negativity is therefore exactly the open interval `1/2<p<1`. Both endpoints give difference zero: at `p=1/2` by the linear factor, and at `p=1` because `P(S=a)=0`. The proof's sentence "with `p>1/2`" followed by "`<0`" is one endpoint wider than this interval. The only probability at which the worst-case comparison is used is `p_star=1-e^{-1}`. The rational enclosure of `e` gives `1/2<1-1/e_lo<p_star<1-1/e_hi<1`, so `p_star` is interior and (F10) applies strictly there. Theorem F and the reduction to `H_(3,1)` use that interior point, so the endpoint `p=1` does not require an amendment. The odd-tail good probability is then strictly decreasing in `a>=1`, and its maximum is at `a=1`, `n=3`.

The resulting value is

`P(Bin(3,p_star)<=1)=3e^{-2}-2e^{-3}`,

`H_(3,1)=3-log(3e-2)=h_star`.

Since `e>2`, `(e-1)(e-2)>0` rearranges to `3e^{-2}-2e^{-3}<e^{-1}`, hence `h_star>1`. Every larger capacity or demand in the stated class has hazard at least `h_star`. This comparison is an identity at `p_star`, not a grid extrapolation.

Equations (F8) and `H_(n,a)>=h_star` give (F12): the full-block price is at most `rho_star[-log mu_p(D_i)]` and at most `1`. Because `rho_star<1`, that price is also at most `min(1,-log mu_p(D_i))=phi(q_i)`. The stronger comparison `price<=q_i` fails when every `c_v=1` and `mu_p(D_i)>0`, which is the three-coordinate reference point: the generator price is `1` and `q_i<1`.

## 3. Global hazard direction — ACCEPT

Assume every local good probability is positive. The blocks are disjoint and the coordinates are independent, so the local events `{|U intersect X_i|<=a_i}` are independent and

`mu_p(intersection_i D_i)=product_i mu_p(D_i)`.

Global goodness is that intersection cut by the further requirement that the occupied-block support contains no edge of `H`. The inclusion `D subset intersection_i D_i` does not use independence of crossing failures. It supplies

`mu_p(D)<=product_i mu_p(D_i)`,

and the logarithm reverses the inequality to (F13). Summing (F12) bounds the union of the full-block generators by `rho_star[-log mu_p(D)]`.

Independence is used only for that product of local events. A crossing constraint can only make `mu_p(D)` smaller. If the union price exceeds one, the empty generator has price one. The switch is valid: the union price is already at most `rho_star` times the global hazard, so that product exceeds one and the capped right-hand side of (F2) equals one. If some local good probability is zero, then `mu_p(D)=0` and the same price-one cap gives (F2). If global goodness vanishes only because of a crossing event, the local sum remains a lower bound in the extended reals and the cap again gives (F2).

## 4. Full-block cover and realized family — ACCEPT

The price bound in sections 2–3 does not create the cover. For this `D`, the pinned note's Theorem P2 states that `G={X_1,...,X_b}` covers `O_K(D)` if and only if `K>=K_H(d)`.

Sufficiency: a set that meets no generator omits at least one vertex in every block, so each block has size at most `a_i d_i` and splits into at most `d_i` pieces of size at most `a_i`. Labels taken from palettes `P_i` with empty intersection along every edge of `H` put each color class inside `D`. Necessity: the set that omits one vertex in every block avoids every `X_i`, and any decomposition into members of `D` uses at least `d_i` colors in block `i` whose supports are independent sets of `H`. Below `K_H(d)` that set is an uncovered obstruction.

Both directions use the capacity-and-clutter description (P1). An arbitrary decreasing family need not have the full blocks as a cover at palette `K_H(d)`. Section 2 of the proof already separates the full-ground price estimate from obstruction coverage. The same-palette hypothesis of Theorem F is exactly `K>=K_H(d)` from (P7). Probabilities and prices do not enter that setwise statement. The six-block numerical illustration, with `K_H=816` and whole-ground chromatic number `818`, is an instance of this same cover; it is not a second theorem.

The realized-cover price theorem under `c_v<=p_v` is a different hypothesis. Theorem F replaces that price ceiling by `c_v<=phi(p_v)` and pays for the wider ceiling with the factor `rho_star` on the `d_i>=2` family.

## 5. Sharpness — ACCEPT

Take one block, `a=1`, `d=2`, `n=3`, empty `H`, and `K=2`. Then `K_H(d)=2`, `D` is the sets of size at most one, and `O_2(D)` is exactly the full triple. Every proper subset splits into two sets of size at most one.

At `p_i=p_star` and `c_i=1`, `phi(p_star)=1`, so every generator price is `1`, including the empty generator. A generator cover of the triple must contain at least one generator, because the empty family leaves the triple uncovered. Every alternative cover therefore costs at least one, and the full block attains one.

The hazard of this block is `h_star`, so the right-hand side of (F2) equals `min(1,rho_star h_star)=1`. Replacing `rho_star` by any smaller positive constant makes that right-hand side strictly smaller than one while every cover still costs at least one. The witness lies inside the realized class, so the uniform constant is sharp for the class. The argument counts every generator cover of this obstruction, not only `{X}`.

## 6. Demand one and exact scope — ACCEPT

If `d=1`, then `n=a+1` and, with empty `H`, `K_H(d)=1`. The only obstruction is the full block. At `p_i=p_star` and `c_i=1`, every cover still costs at least one. The good probability is `1-p_star^{a+1}`. For `a>=1` and `p_star in (0,1)`,

`p_star^{a+1}<p_star`,

so `mu_p(D)>1-p_star=e^{-1}` and the hazard is strictly less than one. Then

`min(1, rho_star[-log mu_p(D)]) < 1 <= covercost`.

The same-palette transformed-price bound (F2) is false on this demand-one family. For `a=1` the same comparison is `e>1`, which is the two-coordinate member `n=2`. That preserves the earlier boundary, where `p=1/2` and `c=2/3` already make every cover cost `4/9>log(4/3)`. The `p_star` witness is the uniform-factor form of that boundary. Original P15-B asks for a local `phi(q_i)` budget on each local cover; the demand-one reference point has full-block price `1` and local hazard below one, so that local premise is absent.

The theorem's positive statement remains the realized family with every `d_i>=2`. A single multi-block example that happens to have some demand equal to one can still have a large crossing penalty. Section 5 does not claim that every such mixed instance fails. It claims that no uniform statement with this palette and this transformed-price ceiling survives once every demand-one example is admitted, while `d_i>=2` is sufficient for the realized capacity family on the full probability cube.

The displayed scope in section 1 matches these slices: disjoint blocks, edges of size at least two, empty `H` allowed, `a_i>=1`, `d_i>=2`, independent `p`, prices through `phi`, palette `K_H(d)`, and uniform factor `rho_star`, with `6/7` as a coarser rational upper bound. The older factor `16/27` on probabilities at most `1/4` is a different theorem and was not re-proved here.

## Numerical certificate

The proof's rational bound was recomputed from the exponential series. The degree-7 tail gives

`e < 31967/11760 < 87/32`, `87/32-31967/11760=11/23520`,

and the degree-5 Taylor gap at `11/6` is exactly `26081/933120`. Thus `3e-2<197/32<exp(11/6)`, `h_star>7/6`, and `rho_star<6/7`.

An independent series enclosure of `e`, binary reduction of `3e-2`, and the odd logarithm remainder on `[1,2]` places `rho_star` in the open interval stated in (F3):

`0.84547981724898672067 < rho_star < 0.84547981724898672068 < 6/7`.

## Overall

**ACCEPT** the exact statement of Theorem F. The six slices close on that statement: the hazard transform and its boundary conventions, the reduction of every `d_i>=2` capacity hazard to `h_star` at an interior `p_star`, the one-sided use of disjoint-block independence, the realized full-block cover at `K_H(d)`, sharpness against every generator cover of the `a=1`, `d=2` triple, and a demand-one counterexample to dropping `d_i>=2`.

This acceptance is the scoped nonauthor reading of the pinned proof. It leaves the landing disposition untouched.
