# P12-B — Strict-capacity packing and a probability-to-mean bound

Status: elementary author-side proofs; next-fit packing and one-sided second moments are standard, not claimed novel.

## B1. Packing with strict capacity

Let finite item weights satisfy 0<=a_i<1. Good sets have total weight **strictly below** 1. If U cannot be partitioned into K good sets, then

\[
 \boxed{a(U)>K/2.} \tag{B1}
\]
Proof: run next-fit in any fixed item order, opening a new bin whenever insertion would make its load >=1. Each positive item fits alone. When a new bin is opened, the preceding bin load plus the first item of the new bin is >=1; consequently each adjacent pair of completed bin loads has sum >=1. If K+1 bins are ever opened, use their first K+1 loads. For odd K, pairing them gives total >=(K+1)/2>K/2. For even K, pair the first K bins and include the positive first item in bin K+1, giving total >K/2. Thus a(U)<=K/2 guarantees <=K bins, proving the contrapositive. Zero-weight items may be put in any bin; the empty set uses empty parts.

This is a sufficient packing criterion, not an equivalence: total>K/2 need not imply nonpackability. Capacity equality is forbidden throughout.

## B2. A mean cap from large probability of satisfying a constraint

Let W=sum_i a_i X_i with independent Bernoulli(p_i), 0<=a_i<1. If P(W>=1)<=1/4, then

\[
 \boxed{\mathbb EW<2.} \tag{B2}
\]
Indeed m=EW and Var(W)=sum a_i^2 p_i(1-p_i)<=m. For m>=2, the one-sided Chebyshev/Cantelli bound gives

\[
 P(W<1)\le\frac{m}{m+(m-1)^2}\le\frac23,
\]
contradicting P(W<1)>=3/4. The rational function decreases for m>1. Degenerate variance zero is handled directly; if m>=2 then W=m almost surely and failure has probability one.

For completeness, Cantelli follows from Markov applied to (W-m-t)^2: for x>0 and t>=0,
P(W-m<=-x)<= (Var(W)+t^2)/(x+t)^2; choosing t=Var(W)/x gives Var/(Var+x^2). No normality assumption is used.

## B3. Multiple resource coordinates

For d constraints a_{ji}>=0 with capacities one, set J={i:max_j a_{ji}>=1} and work outside J. If every constraint is satisfied simultaneously with probability >=3/4, then each restricted row has mean <2 by B2. With a_i=max_j a_{ji},

\[
 M_p:=\sum_{i\notin J}p_i a_i
 \le\sum_{j=1}^d\sum_{i\notin J}p_i a_{ji}<2d. \tag{B3}
\]
Any scalar bin with sum a_i<1 satisfies EVERY original resource constraint. Therefore (B1) for the max-coefficient scalar load is a sufficient simultaneous decomposition criterion. This does not pretend that the constraints are independent; only the original coordinates are independent.
