# P09-B — From proper-link hit probabilities to weighted moments

Complete author-side proof candidate. The hypergraph in this note is R-UNIFORM. A separate lower-rank core construction is what permits its use for general rank-at-most-R antichains. Do not omit that reduction.

For a nonempty proper subset A of an edge of a simple R-uniform hypergraph F, define

\[
F(A)=\{e\setminus A:e\in F,\ A\subset e\}.
\]

This is a simple (R-|A|)-uniform hypergraph on X\A. Let t be an independent coordinate-probability vector. Suppose every such link has hit probability at most a<1.

Write lambda=a/(1-a) and recursively define, for j>=1,

\[
u_j=\lambda\left(1+\sum_{i=1}^{j-1}\binom ji u_{j-i}\right).
\tag{1}
\]

Here and below the letter u_j in (1) is the link-mass bound, not a random vector.

## 1. Link mass bounds

**Lemma B1.** Every proper link F(A) of rank j has total t-weight at most u_j.

For j=1, its distinct singleton events are independent, but independence is unnecessary: the hypergraph second-moment lemma has c=0 and gives mass <=a/(1-a)=u_1.

For j>1 use induction. Fix an edge e of F(A). If another link edge f meets e in S=e intersect f, then S is a nonempty proper subset of e. Its contribution to the directed dependency of e is product_{v in f\S}t_v. Summing over all f with that exact S is bounded by the entire weight of F(A union S). This is a proper link of rank j-|S|, whose weight is bounded by induction.

Consequently the dependency load is at most

\[
\sum_{i=1}^{j-1}\binom ji u_{j-i}.
\]

Its hit probability is at most a by hypothesis, so the hypergraph second-moment lemma yields (1). This proves the induction. Different f are classified by their EXACT intersection S; bounding each class by a larger link only overcounts in the upper-bound direction.

## 2. Full hypergraph mass

Suppose additionally that F itself has hit probability at most epsilon<1. Applying the same dependency grouping to an edge of F gives

\[
\sum_{e\in F}\prod_{v\in e}t_v
\le \frac{\epsilon}{1-\epsilon}
 \left(1+\sum_{j=1}^{R-1}\binom Rj u_j\right).
\tag{2}
\]

Empty F is immediate. These bounds concern actual link-hit probabilities, not sampled estimates or raw expected counts.

## 3. Closed form and a simple factorial bound

The recurrence has the exact solution

\[
u_j=\sum_{m=1}^j m!\,\left\{\!\begin{matrix}j\\m\end{matrix}\!\right\}\lambda^m,
\tag{3}
\]

where the brace is a Stirling number of the second kind. To verify it, enumerate ordered partitions of j labeled elements according to the first block: either it is all j elements, giving lambda, or it has a nonempty proper size i, giving lambda binom(j,i) u_(j-i). This is exactly (1).

Every ordered partition into m blocks is represented by at least one permutation of the j elements with m-1 cuts between consecutive positions. Therefore

\[
m!\,\left\{\!\begin{matrix}j\\m\end{matrix}\!\right\}
\le j!\binom{j-1}{m-1}.
\]

Summing (3),

\[
\boxed{u_j\le j!\lambda(1+\lambda)^{j-1}.}       \tag{4}
\]

Moreover, (1) at j=R gives the useful identity

\[
1+\sum_{j=1}^{R-1}\binom Rj u_j=u_R/\lambda
\]

when lambda>0. Thus (2) and (4) imply

\[
\boxed{M_t(F)\le
\frac{\epsilon}{1-\epsilon}R!(1+\lambda)^{R-1}.}\tag{5}
\]

Here M_t(F)=sum_e w_t(e) denotes the EXPECTED EDGE COUNT, not the probability measure of a family. The same notation is used in the subsequent notes.

At a=0 every link contains no edge of positive weight, so under strictly positive coordinates F is empty; handle this trivial case separately rather than dividing by lambda=0.

## 4. Rank-four constants

For a=3/4, lambda=3:

\[
u_1=3,\qquad u_2=21,\qquad u_3=219.
\]

For R=4 and epsilon=1/4, (2) gives

\[
M_t(F)\le\frac13(1+4\cdot219+6\cdot21+4\cdot3)=1015/3.
\]

These constants are independently recomputed by the recurrence and ordered-partition formula in the exact checker. Neither formula is asserted as a new combinatorial identity; its role here is to quantify the proof's rank dependence.
