# PR-TAL-003 — Zero-scope extraction and a five-copy laminar theorem

2026-09-16. Complete author-side derivation, NOT independently reviewed. Historical novelty UNESTABLISHED. This is a restricted subclass of the discrete Convexity Conjecture, not a prize solution. The previous PR-TAL-001 remains immutable; the following strengthens its eight-copy conclusion using a sharper argument.

## 1. Convention

For a finite ground set X, scopes B and capacities r_B>=0, write D={S:|S intersect B|<=r_B for every B}. D_(k) consists of unions of k D-members; D^(k)=2^X minus D_(k). A cover is a family of generators whose upward closures contain D^(k). At independent coordinate probabilities p_i its cost is sum_G product_(i in G)p_i. A cost at most1/2 is smallness. Equal scopes are identified, retaining minimum capacity. All constraints and ground sets are finite, with no uniform size restriction.

## 2. Extract all rank-zero scopes FIRST

Let Z be the union of the scopes having capacity zero. Every D-member avoids Z. Delete Z from all remaining scopes, retaining their positive capacities, then remove vacuous or redundant constraints. Let D_R denote the residual downset on X minus Z. Coordinate independence gives EXACTLY

    mu_p(D)=P(no selected coordinate in Z)*mu_p(D_R)=z*u.

For uniform or nonuniform p_i, the singleton cover of any bad set meeting Z costs sum_(i in Z)p_i <= -log z (provided z>0). The residual obstruction is covered separately. It is not necessary to charge a common forbidden coordinate repeatedly to every original rank-zero scope.

If mu_p(D)>=1-epsilon>0, both z,u>=1-epsilon; put l0=-log z, l1=-log u. Then

    l0+l1=-log mu_p(D)<=-log(1-epsilon)<=epsilon/(1-epsilon).

This factoring is exact, not conditioning on nested positive-capacity events.

## 3. Positive-capacity single-block estimate

Assume 0<=p_i<=1/2. For a block of positive capacity r>=1 let Y be its number of selected coordinates, lambda=E Y, q=P(Y>=r+1). For e_j the jth elementary symmetric polynomial in p_i,

    q >= e_(r+1)(p)*exp(-2lambda),
    e_a(p)e_b(p)>=binom(a+b,a)e_(a+b)(p),
    e_b(p)<=lambda^b/b!.

The first bound follows by restricting q to the exact atom r+1 and using product(1-p_i)>=exp(-2lambda). The next two follow by expanding products and retaining disjoint index choices. Hence for any integer t>=2,

    e_(tr+1)(p) <= q exp(2lambda)lambda^((t-1)r)(r+1)!/(tr+1)! .  (1)

If q<=1/5, Cantelli and Var(Y)<=lambda give 4(lambda-r)^2<=lambda whenever lambda>r. When lambda<=r all upper bounds below hold automatically. For r1,2,3,4 take respectively

    u=(33/20,57/20,4,21/4).

Each satisfies u>=r and 4(u-r)^2>=u, so lambda<=u. The rational Taylor enclosures certify

    exp(33/10)<28, exp(57/10)<300, exp(8)<3000, exp(21/2)<37000.

Set

    a_r=E_r*u_r^(4r)*(r+1)!/(5r+1)!       (r=1,...,4),
    a_r=(3/5)^r                          (r>=5).

The first four a_r are approximately0.57649,0.19628,0.05774,0.02895. Decimals are diagnostic; Fractions are authority.

For r>=5 the Cantelli condition implies lambda<=5r/4. The factorial product obeys

    log[(5r+1)!/(r+1)!]>=integral_r^(5r) log x dx
                               =4r log r+5r log5-4r.

Substitute in (1), with t=5, to get

    e_(5r+1)(p) <=q*[exp(13/2)/1280]^r <q*(3/5)^r.

The last inequality follows from e<68/25 and (68/25)^13<768^2. There is no finite-rank extrapolation here; the displayed argument covers every r>=5.

The exact rational checker proves

    sup_(r>=1) a_r <3/5,
    A5:=sum_(r>=1)a_r=sum_(r=1)^4a_r+243/1250 <16/15.   (2)

The complete infinite series after rank4 is summed geometrically. For q=0, all relevant e_j are zero, and no division by q occurs.

For reference, the positive-rank two-copy dilution coefficients of PR-TAL-001 satisfy uniformly (q<=1/4)

    4^(-(2r+1))*e_(2r+1)(p) <=b_r*q, sup_(r>=1)b_r<3/5. (3)

PR-TAL-001 gives the explicit six rational coefficients for r1..6 and the all-r>=7 bound (1/4)(3/4)^r. Those inputs are copied with original hashes and independently re-evaluated here; this is not a novelty claim for the Bernoulli inequalities.

## 4. Laminar theorem with five pieces and no dilution

Suppose scopes are laminar: any pair is disjoint or nested. For UNIFORM p in[0,1],

    mu_p(D)>=4/5  =>  D^(5) has a p-cover of cost <=4/15<1/2.   (4)

Proof for p<=1/2, and also for unequal p_i<=1/2: after zero-scope extraction and redundancy removal, scopes with any fixed positive capacity r are disjoint. Their constraint events are independent. Since D_R implies all of them,

    u <= product_(r_B=r)(1-q_B).

Consequently sum_(r_B=r)q_B<=-log u=l1. [The direction here is important: an intersection has at most the probability of a subcollection.] The standard simultaneous balanced-coloring lemma for laminar scopes gives

    (D_R)_(5)={S: |S intersect B|<=5r_B for all retained B}.

Therefore all(5r_B+1)-subsets of B cover the residual obstruction. By(1),(2) their cost is at most A5*l1. Adding the zero-singleton cover gives

    cost<=l0+A5*l1<=(16/15)(l0+l1)
                    <=(16/15)*(1/4)=4/15.

The balanced-coloring argument is recalled explicitly: induct up the laminar forest; each child has color counts equal up to one; permute its high-count colors onto currently least populated colors. This preserves all internal child balances and makes the parent balanced. An element outside children is a one-element child. Thus counts<=5r imply every color count<=r. Conversely a union of five D-members necessarily has every count<=5r.

For uniform p>=1/2, mu_(1/2)(D)>=mu_p(D)>1/2. A random two-color partition of X has each part distributed as mu_(1/2); the union bound gives a positive probability that both belong to D. Thus X, and every subset by downward closure, belongs to D_(2), making D^(5) empty. This high-p argument is not extended to unequal probabilities without justification.

For arbitrary nonuniform p_i, cap p'_i=min(p_i,1/2). Since mu_p'(D)>=mu_p(D) and p_i/2<=p'_i, (4) holds with a cover evaluated at p_i/2. At p_i<=1/2 it holds without dilution.

## 5. Meaning and limits

Compared with the earlier eight-piece theorem, this changes both the number of pieces (8 to5) and sufficient good-event probability(7/8 to4/5). It is a stronger local theorem, not a new universal result. Standard laminar-matroid coloring and elementary concentration are credited. No unrestricted discrete, continuous, or fractional conjecture is closed. No historical originality of these exact constants is claimed.
