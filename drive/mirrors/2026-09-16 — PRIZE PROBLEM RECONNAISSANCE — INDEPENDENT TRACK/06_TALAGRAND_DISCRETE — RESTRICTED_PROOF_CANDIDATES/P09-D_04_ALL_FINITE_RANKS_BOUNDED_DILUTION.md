# P09-D — All finite ranks with bounded dilution and explicit rank-dependent pieces

**Complete author-side proof candidate.** No external independent mathematical review or theorem-prover run. Novelty is not established. Bounded-rank qualitative existence already follows from the established Kahn--Kalai theorem; this note gives a direct explicit construction with a different parameter tradeoff.

## 1. The statement with all quantifiers exposed

For every integer r>=3 define

\[
\epsilon_r=\frac{r+1}{4r},\qquad
L_r=\frac{16r}{3(r+1)},\qquad
K_r=1600\,4^{r-3}\left(\frac{r!}{6}\right)^5,
\]

and

\[
C_r=C_3+\sum_{j=4}^r2^{-j-4},\qquad
C_3=135054337/282419200.
\]

**Theorem D.** Let D be any decreasing family on a finite ground set, with every inclusion-minimal forbidden set of size at most r. For any vector of independent coordinate probabilities p_v in (0,1), if

\[
\mu_{\mathbf p}(D)\ge1-\epsilon_r,
\]

then D^(K_r) has a cover at probabilities p_v/L_r of cost strictly less than C_r. Uniformly in r,

\[
L_r<16/3<6,
\qquad
C_r<C_3+1/128<12/25+1/128=1561/3200<1/2.
\]

In particular, because epsilon_r>1/4,

\[
\boxed{\mu_{\mathbf p}(D)\ge3/4
\quad\Longrightarrow\quad
D^{(K_r)}\text{ is }(\mathbf p/6)\text{-small}.}
\tag{1}
\]

This theorem has the form 'FOR EVERY r, there exists K_r'. It does not supply one finite K for every r. The unrestricted discrete convexity problem remains open here.

## 2. Induction base

At r=3, epsilon_3=1/3, L_3=4, and K_3=1600. This is exactly P09-A. No previous approximate value or unchecked witness count is consumed.

## 3. A summable sequence of sprinkling losses

Suppose r>=4 and the result holds at r-1. Put

\[
a_r=1-r^{-2},\qquad
t_v=p_v/r^2,\qquad
q_v=\frac{p_v-t_v}{1-t_v}
=\frac{a_rp_v}{1-p_v/r^2}.
\tag{2}
\]

All coordinates lie in (0,1), q_v<=p_v, and q_v>=a_rp_v. Independent q- and t-samples have union parameter

\[
q_v+t_v-q_vt_v=p_v.
\]

The precise identities that preserve the induction are

\[
\epsilon_r/a_r=\epsilon_{r-1},
\qquad
L_r=L_{r-1}/a_r.
\tag{3}
\]

The first provides sufficient good-event probability after extracting cores. The second ensures that a previous cover at q/L_(r-1) remains valid at the target p/L_r.

The losses are summable: product_{j=4}^r(1-j^(-2))=3(r+1)/(4r). This is the reason L_r stays bounded. It does not imply that the color counts stay bounded.

## 4. High-core reduction

Let H be the minimal-forbidden antichain and H_r its r-element edges. Build the core family K from:

- every original edge of size less than r;
- every nonempty proper subset A of an H_r edge whose t-link-completion probability is greater than a_r.

Retain just the inclusion-minimal cores. Let F be the r-uniform family of H_r edges containing no retained core.

If the first sample X_q contains a K-core, choose its first contained core deterministically from X_q alone. The second sample completes an original edge with conditional probability greater than a_r (or probability one for an original smaller edge). The sprinkling lemma gives

\[
P_q(X_q\text{ hits K})\le\epsilon_r/a_r=\epsilon_{r-1}.
\]

The induction hypothesis covers K^(K_(r-1)) at q/L_(r-1) for cost <C_(r-1). Since q_v/L_(r-1)>=p_v/L_r, the same generators cost no more at p/L_r.

## 5. The residual is controlled through its proper links

Every nonempty proper residual link F(A) has t-hit probability at most a_r. Otherwise A would have qualified as a high core in H_r, and any residual edge containing A would contain a retained minimal core, a contradiction.

Since F is a subfamily of H and t_v<=p_v, its full t-hit probability is at most epsilon_r. Set

\[
\lambda_r=a_r/(1-a_r)=r^2-1.
\]

Applying P09-B,

\[
M_{\mathbf t}(F)
\le\frac{\epsilon_r}{1-\epsilon_r}
 r!(1+\lambda_r)^{r-1}
\le r!r^{2r-2}.
\tag{4}
\]

The final inequality uses epsilon_r<1/2. This is a weighted EXPECTED EDGE COUNT, not a hit probability.

## 6. Change parameter explicitly, then color

Let z_v=p_v/L_r. Since each residual edge has exactly r elements and z_v/t_v=r^2/L_r for every coordinate,

\[
M_{\mathbf z}(F)
=\left(\frac{r^2}{L_r}\right)^r M_{\mathbf t}(F)
\le\frac{r!r^{4r-2}}{4^r},
\tag{5}
\]

because L_r>=4. Notice that z can be larger than t. The factor in (5) cannot be dropped or bounded by one.

Choose

\[
h_r=4r^5.
\]

A uniform h_r-coloring has expected monochromatic residual-edge cost M_z(F)/h_r^(r-1). Hence some fixed coloring supplies a generator cover of F^(h_r) of cost at most

\[
\frac{r!r^{4r-2}}{4^r(4r^5)^{r-1}}
=\frac{r!r^{3-r}}{4^{2r-1}}
\le\frac{r^3}{2^{4r-2}}
\le2^{-r-4}.
\tag{6}
\]

Here r!<=r^r. The last inequality is r^3<=2^(3r-6). It holds at r=4 with equality, and its ratio step follows from ((r+1)/r)^3<8 for r>=4. This is an all-rank proof, not extrapolation from a table of ranks.

## 7. Compose the colorings and sum the budget

If a set is K_(r-1)-colorable in K and h_r-colorable in F, the ordered-pair refinement has K_(r-1)h_r colors and avoids every original edge: any removed edge contains a K-core, and every remaining edge is in F.

Therefore

\[
D^{(K_{r-1}h_r)}
\subseteq K^{(K_{r-1})}\cup F^{(h_r)}.
\]

Combining the covers yields total cost <C_(r-1)+2^(-r-4)=C_r. The piece recurrence is

\[
K_r=K_{r-1}\,4r^5,
\]

which unrolls to the stated factorial expression. The dilution recurrence in (3) unrolls to L_r=16r/[3(r+1)]. This completes the induction.

## 8. What is and is not uniform

Uniform in ground-set size: yes.
Uniform in numbers of witnesses, degrees, codegrees, and coordinate probabilities: yes, within the stated product domain.
Uniform in maximal witness rank for the dilution factor and total cost cap: yes.
Uniform in rank for the number of pieces: NO.

The theorem is a constructive finite procedure in principle: list cores, compute their link probabilities, recurse, and find a coloring by conditional expectation or finite search. It is not a polynomial-time algorithm, and exact comparisons with arbitrary real inputs are mathematical predicates, not a numerical implementation promise.

For orientation, K_4=6,553,600 and K_5=81,920,000,000. The separately optimized P09-C gives a much smaller 6,400-piece rank-four result by allowing dilution 8. Neither result dominates the other in both parameters.

The growth of THIS recurrence is not an impossibility theorem about better recurrences or the original conjecture. A new shared-color or compression argument could change the tradeoff. Such an argument is not supplied here.
