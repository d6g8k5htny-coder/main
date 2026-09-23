# P10-B — A scalar inequality for threshold-gate cover composition

Author-side proof; no external review; novelty unestablished. Standard second moments, Chebyshev, and elementary factorial inequalities are used. This is not a new proof of an external concentration theorem.

Define h(q)=-log(1-q) for 0<=q<1, and

\[
\phi(q)=\min\{1,h(q)\},\qquad \phi(1)=1.
\]

Then q<=phi(q)<=2q for 0<=q<=1. The upper bound follows from h(q)<=q/(1-q)<=2q when q<=1/2 and phi(q)<=1<=2q otherwise.

Let Z_i be independent Bernoulli(q_i), S=sum Z_i, lambda=sum q_i, and Q=P(S>=s), for an integer s>=1. Put m=K(s-1)+1 with K>=64. Write e_m for the elementary symmetric polynomial, zero if m exceeds the number of variables.

## Theorem

\[
\boxed{\min\{1,e_m(\phi(q_1),...,\phi(q_n))\}\le\phi(Q).} \tag{1}
\]

For s=1, Q=1-product(1-q_i). If all q_i<1, sum phi(q_i)<=sum h(q_i)=h(Q), so capping at one proves (1). If a q_i=1 then Q=1 and the claim is immediate.

Suppose s>=2. If Q>2/3, then phi(Q)=1 because e<3; (1) is trivial. If Q=0 then fewer than s positive q_i exist and e_m=0. The case m>n is also immediate. It remains to take 0<Q<=2/3 and m<=n.

### A. Control the mean

If lambda>=4s, Chebyshev and Var(S)<=lambda give

\[
P(S\le s-1)\le\frac{\lambda}{(\lambda-s+1)^2}
\le\frac{4s}{(3s+1)^2}\le\frac{4}{9s}\le\frac29<\frac13.
\]

The middle function decreases for lambda>s-1. This contradicts P(S<=s-1)>=1/3. Therefore lambda<4s.

### B. Compare a factorial moment with the tail probability

Let Y=binom(S,s), whose mean is mu=e_s(q). In the expansion of E[Y^2], disjoint ordered pairs contribute at most mu^2. For a fixed selected s-set, another s-set sharing s-j coordinates can be chosen by removing j of its s coordinates and adding j others. Its extra inclusion weight sums to at most binom(s,j)lambda^j/j!. Thus

\[
E[Y^2]\le\mu^2+\mu C_s,
\quad C_s=\sum_{j=0}^{s-1}\binom sj\frac{\lambda^j}{j!}.
\]

Cauchy--Schwarz on Y1_(Y>0) gives Q>=mu/(mu+C_s). Hence

\[
\mu\le\frac{Q}{1-Q}C_s\le3Q C_s.
\]

Also

\[
C_s\le\sum_{j\ge0}\frac{(s\lambda)^j}{(j!)^2}
\le e^{2\sqrt{s\lambda}}\le e^{4s}.
\]

The exponential comparison follows by retaining the diagonal terms in e^x e^x, x=sqrt(s lambda). Therefore mu<=3Q e^(4s).

### C. Amplify from s to m

Counting disjoint products gives binom(m,s)e_m(q)<=e_s(q)e_(m-s)(q), and e_(m-s)(q)<=lambda^(m-s)/(m-s)!. Consequently

\[
e_m(q)\le e_s(q)\frac{s!\lambda^{m-s}}{m!}.
\]

Use phi(q_i)<=2q_i, lambda<4s, s!<=s^s, and m!>=(m/e)^m:

\[
e_m(\phi(q))\le
3Q\left(\frac{e^4}{4}\right)^s
\left(\frac{8es}{m}\right)^m.
\]

For s>=2 and K>=64, m>=32s. Using e<3, the last expression is at most

\[
3Q\left[\frac{81}{4}\left(\frac34\right)^{32}\right]^s.
\]

Let theta=(81/4)(3/4)^32. The exact integer inequality 81*3^32<4^32 proves theta<1/4. Therefore

\[
\boxed{e_m(\phi(q))\le3Q\theta^s<\frac3{16}Q\le\phi(Q).} \tag{2}
\]

This proves (1) in every case. All unbounded n,s are handled analytically; the computer only checks finite arithmetic and example instances. The constant 64 is deliberately conservative.

## Exact statements checked numerically

The finite companion checks the ordered-pair second moment and the extension inequality separately. It tests the stronger rational implication e_m(2q)<=3Q/16 whenever Q<=2/3 at large arities with m<=n. It does not use floating-point logarithms as theorem inputs.
