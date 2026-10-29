# D1 Section 9 repair — Borel elder mark in weighted pair Kac–Rice

**Object:** D1-SECTION9-BOREL-REPAIR-20260925-v1.1  
**Author lane:** OpenAI / ChatGPT  
**Parent:** UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1, Drive file 1foDgiDi4XIKOfbrZEE8dWKV_BkZU8LIb, 40261 B, SHA256 9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7.  
**Disposition:** AUTHOR-SIDE AMENDMENT to Section 9 only; exact-source nonauthor re-review required.  
**Scientific effect:** NONE. The parent file is unchanged. Theorem A/C, the cap theorem, residual independence, normalizer, and coefficient (15.2) are not accepted by this amendment.

## 1. Review finding consumed — and one citation correction

The nonauthor review on main PR112 correctly identified the load-bearing write-up gap: Section 9 names a monotone-class passage from continuous cylinder weights to the global Borel elder mark but does not explicitly construct the regular conditional law on the whole C^2 field space or state the closure hypotheses of the determining class.

The review's theorem-number statement is not correct for arXiv:2304.07424v3. Primary-source inspection of the exact v3 PDF/HTML gives:

- Theorem 2.2: Gaussian unweighted Kac–Rice under C^1 paths and positive-definite point covariance; compact domains have finite expected level-set measure.
- Theorem 6.1: Crofton's formula for rectifiable sets.
- Theorem 7.1: Expected integral on the level set, i.e. the weighted formula for a nonnegative weight g(t,Z(.)), under lower-semicontinuity in the location and mark and continuity of the conditional mark law. Its displayed formula is (7.2).

Therefore the parent was right to point to Theorem 7.1 for the continuous-cylinder base case. It was also right not to apply Theorem 7.1 directly to the elder indicator: the elder equality/path mark is Borel but need not be lower semicontinuous. The repair below makes the author's separate finite-measure/monotone-class extension explicit.

Primary source inspected: Armentano–Azaïs–León, *On a general Kac–Rice formula for the measure of a level set*, arXiv:2304.07424v3 (5 Dec 2023), especially Theorems 2.2 and 7.1. No priority claim.

## 2. Exact replacement for parent Section 9

### 9. A source-bound weighted Kac–Rice interface for the global elder mark

Fix a compact ordered-pair domain D contained in the off-diagonal set of X x X, with positive distance from the diagonal. Write

    G(x,y) = (grad f(x), grad f(y)) in R^(2d).

By the positive Fourier spectrum and distinct-site distribution argument of parent Section 2, Cov G(x,y) is positive definite at every pair, uniformly on D. The sample field is smooth. The derivative of G with respect to (x,y) is block diagonal, so its normal Jacobian is

    Delta(x,y) = |det H_x det H_y|.

Armentano–Azaïs–León v3, Theorem 2.2, therefore gives the unweighted zero-count formula for G=0 on D, and finiteness on compact D. We use their Theorem 7.1 only for bounded nonnegative continuous cylinder weights, for which its lower-semicontinuity hypotheses hold automatically. The Borel elder mark is reached by the independent finite-measure argument below, not by claiming it is lower semicontinuous.

#### 9.1 A regular conditional law on the whole C^2 field

Let E = C^2(X), equipped with its usual separable Banach topology on the compact torus. For t=(x,y) in D and z in R^(2d), finite-dimensional Gaussian regression gives a smooth conditional mean m_(t,z) and a centered residual Gaussian field R_t such that, as a random element of E,

    law(f | G(t)=z) = law(m_(t,z) + R_t).

The residual can be constructed from the full Gaussian field by subtracting

    Cov(f,G(t)) Cov(G(t))^(-1) G(t).

Uniform nonsingularity on D and smooth covariance imply that this subtraction is continuous in (t,z) in every finite C^q mean norm needed here. In particular,

    K(t,z,A) = P{m_(t,z) + R_t in A},  A in Borel(E),

is a regular conditional probability kernel on E, and for every bounded continuous cylinder functional F, the map

    (t,z) -> integral F(phi) K(t,z,dphi)

is continuous. This is the whole-field version required below; it is not merely a family of conditional laws for a finite list of cylinder coordinates.

#### 9.2 Two finite measures on D x E

On the almost-sure zero set of G, define

    mu(A) = E sum_{t in D : G(t)=0} 1_A(t,f),

for A in Borel(D x E). Theorem 2.2 makes mu finite.

Define a second finite Borel measure

    nu(A) = integral_D p_(G(t))(0)
            E[ Delta(t) 1_A(t,f) | G(t)=0 ] dt.

The conditional expectation is defined by the kernel K(t,0,.). Its total mass equals the finite unweighted Kac–Rice integral, so nu is finite.

For every bounded nonnegative continuous cylinder function h(t,phi), Theorem 7.1 (or equivalently the ordinary Gaussian regression proof of its continuous case) gives

    integral h dmu = integral h dnu.                        (9.1)

#### 9.3 Cylinder functions generate the full Borel sigma-algebra

Choose a countable dense set Q in X. The maps

    phi -> partial^alpha phi(q),  q in Q, |alpha| <= 2,

together with a countable coordinate basis on D, separate points of D x E. Because the C^2 norm on the compact torus is recovered from suprema over the dense set for derivatives through order two, these evaluations generate Borel(D x E).

Let H be the bounded real Borel functions h for which (9.1) holds. Since mu and nu are finite, H

1. is a vector space,
2. contains the constants,
3. is closed under bounded pointwise convergence by dominated convergence.

It contains the algebra of bounded continuous cylinder functions. The functional monotone-class theorem therefore implies that H contains every bounded Borel function on D x E.

This is the missing closure. It does not extend Theorem 7.1's lower-semicontinuous hypothesis by assertion; it first obtains equality of two finite measures on a generating algebra and then identifies those measures on their full Borel sigma-algebra.

#### 9.4 Apply the extension to the elder/type mark

Parent Section 8 represents the maximin connection event using countably many strict rational polygonal-path tests in finitely many torus charts. Hence

    (t,phi) -> 1{ d_phi(x) = phi(y) }

is Borel on the Morse distinct-critical-value locus, with the essential-maximum convention preserved. Hessian index indicators are Borel functions of the 2-jet. Their product with the elder mark is therefore a bounded Borel weight on D x E, so (9.1) applies.

No local cap event is substituted for the global mark. The cap theorem is used later only to estimate the probability that this global elder mark fails.

Disintegrate the two endpoint heights using their nondegenerate joint Gaussian density conditional on G(t)=0. At prescribed heights

    f(M)=b,  f(S)=b-k r^3,

the Jacobian Delta = |det H_M det H_S| and the type indicators give exactly the parent endpoint weight W_r. Thus candidate pair intensity contains the full-pin density times E_Q W_r; the selected intensity contains the same factor times the Q^W global elder probability p_r. There is no second determinant factor.

Finally exhaust D toward the diagonal. The parent Section 10 bound is locally r dr, hence integrable at zero; nonnegative monotone convergence gives the near-diagonal formula. This last step imports the parent Sections 3/5/10 estimates and does not re-prove or accept them here.

## 3. What this repair does and does not close

Repaired if a nonauthor reviewer accepts this successor:
- exact theorem attribution for the continuous-cylinder base;
- regular conditional law on C^2;
- finite-measure formulation;
- functional monotone-class hypotheses;
- Borel product-space status of the elder/type mark.

Still separate:
- contact covariance;
- residual-field independence in parent Section 4;
- typed boundary integration;
- full normalizer;
- Theorem A;
- off-diagonal/unbounded-mark domination;
- coefficient (15.2);
- numerical constants;
- RN/24-jet certificates;
- every authoritative scientific-status gate.

## 4. Review request

Review this exact file, not the parent prose by memory. Return:
- C1: primary-source theorem attribution;
- C2: C^2-valued regular conditional Gaussian kernel;
- C3: finiteness and equality of mu,nu on continuous cylinders;
- C4: Borel generation / functional monotone class;
- C5: elder/type mark measurability and determinant accounting;
- C6: scope boundary.

Disposition per interface: ACCEPT / AMEND / COUNTEREXAMPLE / BLOCKED. The reviewer should explicitly state whether the original PR112 theorem-number objection is withdrawn, retained with a primary-source citation, or replaced by a narrower hypothesis objection.

## 5. Source-integrity note

The immediately preceding v1 commit on this branch was created through a string-serialization path that stripped several backslashes from LaTeX notation. It is preserved in Git provenance but is not a valid review target. This v1.1 file deliberately uses plain-text equations to remove that serialization ambiguity. Review only the exact current head and digest.
