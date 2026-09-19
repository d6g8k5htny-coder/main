# P11-A — Arbitrary graph gates admit a composable 64-color hazard bound

Grade: complete author-side proof; external independent review and novelty unestablished. This is not the unrestricted discrete-convexity conjecture. The gate has arbitrary finite arity, but minimal obstructions of size at most two. Classical second moments and conditional expectation are used, not claimed new.

## 1. Objects and target

For a downset D, D_(K) is its K-fold union closure and D^(K) its complement. A generator family G covers an increasing family I when each U in I contains some g in G. At coordinate cost vector w, its cost is sum_(g in G) product_(v in g) w_v. The empty generator has cost one; the empty family has cost zero.

Let p_v in [0,1] be independent selection probabilities and let

    phi(p)=min(1,-log(1-p)), phi(1)=1.

Assume 0<=w_v<=phi(p_v). We prove that when every minimal obstruction has size <=2, D^(64) has a generator cover of w-cost <=phi(Q), Q=1-mu_p(D).

First take D=Ind(G), with G a finite simple graph and no singleton obstructions. Put Z=mu_p(Ind(G)), H=-log Z. If Z=0 or H>=1, the empty generator proves the assertion. If H=0, every graph edge has product p-weight zero, hence product w-weight zero; all edges give a zero-cost cover. It remains to take 0<H<1. Then Q=1-exp(-H)<2/3, since e<3.

## 2. Elementary graph second moment

For any induced residual graph, let mu=sum_(uv in E)p_u p_v and c=max_v sum_(u adjacent v)p_u. Let Y count selected edges. Ordered disjoint edge pairs are bounded by their contribution to mu^2. The ordered intersecting-pair sum is at most sum_v p_v(sum_(u adjacent v)p_u)^2 <=2c mu. Thus

    E Y^2 <=mu^2+(1+2c)mu.

Cauchy--Schwarz gives P(Y>0)>=mu/(mu+1+2c). If P(Y>0)<=Q, then mu<=(1+2c)Q/(1-Q). Zero mu needs no division.

## 3. Deterministic core extraction

Repeatedly choose, by a fixed input-dependent order, a vertex with current weighted neighborhood sum >2, delete that vertex and all incident edges, and record it in C. This selection is NOT adaptive to the random sample. Let N_i be its neighborhood after previous core vertices have been removed.

Let E_i be the event that core vertex v_i is selected, all earlier core vertices are absent, and at least one vertex of N_i is selected. These events are disjoint, imply non-independence, and have probability

    p_(v_i) product_(j<i)(1-p_(v_j)) [1-product_(u in N_i)(1-p_u)].

The bracket is >1-exp(-2)>6/7. Summing gives

    product_(v in C)(1-p_v) >=1-(7/6)Q.

For Q<2/3,

    1-(7/6)Q >=(1-Q)^2,

because their difference is Q(5/6-Q)>=0. Therefore

    S:=sum_(v in C)-log(1-p_v) <=2H.

In particular no extracted vertex has p_v=1 in this regime, and sum_(v in C)w_v<=S<=2H.

## 4. Color the core rather than paying for it

Use eight colors on C, and a DISJOINT set of 56 colors on R=V\C. Every crossing edge automatically has distinct colors.

The total w-mass of edges internal to C is at most

    (sum_C w_v)^2/2 <=2H^2.

Independent random eight-coloring, followed if desired by deterministic conditional expectation, supplies a core coloring with monochromatic edge cost <=H^2/4.

On R, weighted neighborhood sums are <=2. Its bad probability is <=Q, so Section2 gives mu_R<=5Q/(1-Q)=5(exp(H)-1). Since 0<=H<=1, convexity and e<3 imply exp(H)-1<2H. Also phi(p)<=2p for all p. Thus residual edge w-mass is <4*10H=40H. Some 56-coloring has monochromatic edge cost <(5/7)H.

The union of the two internal monochromatic-edge families has cost

    <=H^2/4+(5/7)H <=(27/28)H <H.

Any U avoiding those generators is properly 64-colored by the fixed combined coloring. Hence the generators cover Ind(G)^(64). This proves the gate inequality at all Q, including the trivial cap when H>=1.

The important improvement over a singleton-core cover is that core color cost is quadratic in H. We reserve disjoint colors for the core and residual; we do NOT remove crossing edges without providing those separate palettes.

## 5. Singleton obstructions and zero coordinates

For general minimal rank<=2, let J be singleton forbidden coordinates. By antichain minimality all remaining graph edges avoid J. Good probability factors as product_(j in J)(1-p_j) times the graph's good probability. Thus total hazard is H_J+H_G.

Include singleton generators J, whose w-cost is <=H_J, and the graph cover of cost <=H_G; cap by the empty generator if necessary. This gives cost <=min(1,H_J+H_G)=phi(Q). Zero or unit probabilities are covered by the preceding endpoint cases and this factorization.

## 6. Composition consequence

P10-A's exact common-palette substitution and product-cost multiplication apply at w_i equal to child cover costs, because those costs satisfy w_i<=phi(q_i). P11-A therefore adds arbitrary-arity graph/minimal-rank-two gates to P10's threshold/small-gate hierarchy WITHOUT increasing K=64 and WITHOUT probability dilution or depth losses. Disjoint original supports remain essential.
