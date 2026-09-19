# PR-SID-001 — Safe Sidon splicing and a quantified all-scales trap

Date: 2026-09-16. Complete author-side elementary derivation. Not a solution of Erdős #39; no novelty claimed. Dense subsequences of infinite Sidon sets are classical; the current target requires near-square-root density at every sufficiently large scale.

A Sidon set here has all unordered pair sums distinct INCLUDING diagonal pairs a+a.

## 1. Safe extension lemma

Let F subset [1,M] be finite Sidon, q=M+2, t=q-1, and G any finite positive-integer Sidon set. Then F union C, C=qG+t, is Sidon.

Proof. Old-old and new-new equalities are handled by F and G. An equality between two mixed sums implies a-a'=q(g'-g); |a-a'|<q, so the pairs agree. Old-old sums are at most 2M, whereas each sum involving a new element exceeds 2M. Finally mixed versus new-new equality modulo q would require an old a to be t modulo q, impossible for a<=M<q-1. This exhausts every split of four summands and includes repeated summands. QED.

## 2. An explicit finite carrier

For each odd prime p, set

G_p={2p t+(t^2 mod p)+1:0<=t<p}.

It has p elements and lies in [1,2p^2]. If g_i+g_j=g_k+g_l, the correction from the four residues has absolute value strictly less than 2p; hence i+j=k+l. Reducing modulo p gives i^2+j^2=k^2+l^2, so ij=kl modulo p because 2 is invertible. The two unordered root pairs of the corresponding quadratic agree modulo p and therefore as indices in {0,...,p-1}. Thus G_p is Sidon. Its first two elements are 1 and 2p+2.

## 3. Good block ends do not imply good global density

Starting from ANY finite Sidon prefix, iterate the splice. At stage j let m_j=|F_(j-1)|, M_j=max F_(j-1), q_j=M_j+2. Choose primes p_j tending to infinity so fast that

p_j>q_j^j and p_j>(m_j+1)^(j^2).

Such primes exist by unboundedness of the primes. Put F_j=F_(j-1) union (q_jG_(p_j)+q_j-1), and A=union_j F_j.

Every finite stage and hence A is Sidon. At the block endpoint N_j=max F_j, we have N_j<=3q_jp_j^2 and A(N_j)>=p_j. Thus

limsup_(N->infinity) log A(N)/log N >=1/2.

The reverse inequality follows from distinct positive differences: A(N)(A(N)-1)/2<=N-1. Therefore the limsup is exactly 1/2.

But just before the second element of block j,

N'_j=q_j(2p_j+2)+(q_j-1)-1,

only its first new element has appeared. All later blocks start beyond max F_j, so A(N'_j)=m_j+1. Since N'_j>=p_j>(m_j+1)^(j^2),

log A(N'_j)/log N'_j <1/j^2.

Hence

liminf_(N->infinity)log A(N)/log N=0.

## Conclusion

This produces a Sidon extension with excellent exponent along a subsequence and catastrophic intervening sparsity. It is a rigorous warning against promoting block-end diagnostics to Erdős #39's all-large-N requirement. It does NOT prove every block construction fails, and does NOT improve the best known all-scales density exponent.

The simplistic independent-thinning estimate M^3 p^4 versus Mp limits one first-moment deletion guarantee to M^(1/3); it is not an impossibility theorem for every probabilistic or algebraic method. The present splice avoids mixed collisions but pays scale gaps. The unresolved research target is a dense overlap-compatible construction without those gaps.
