# Quantitative two-atom floors for the scalar D2 expressions

Dylan Roy — delegated AI work. Actual author: OpenAI / GPT-6 Astra Pro,
`continuation-d2-proof-execution-audit-20261006`; main#229 pickup6008048138.
Scientific effect NONE; author-exposed; organizational-independence credit0.
This is an ordinary mathematical derivation with finite rational controls,
NOT a Lean proof or a new numerical certificate for a periodic field.

## 1. Scope and exact interfaces

The consumed algebra is `formal/ResearchFormalCoreR1/D2Schur.lean`,
at Math- commit2f8b6f372be383d752e9dd30d38234faa977243c, Git blob
b95460c0d263a32ea274b347079cca6aaab3d2e9. In particular the ratio, determinant
interpolation, and tau endpoint identities are the existing scalar interfaces.
The current publication base preserves that blob. No primary formal source,
manifest, gate, dependency, workflow or scientific register is modified.

The actual-integral identities also appear in the separately reviewed companion
`companions/d2_moment_bridge_20261006/D2MomentBridge.lean`,
commitb122462634460a8bb9e88406072cb7a356957407, blob
b3184107e290189b91e3e18fd2aa1cb631bb9bde. Its source-and-execution review is
5422935330, with hosted readback6007916533. That review does NOT cover this note.
The short integral arguments below are supplied explicitly rather than treating
a pending integration, historical receipt, or a digest as a new proof premise.

Weighted square completion is elementary algebra, not a claim of discovering
a new general inequality. The useful consequence here is a transparent
atom-mass-to-scalar-floor interface. It does not identify these scalar
expressions with any actual Gaussian/Palm conditional covariance.

## 2. Hypotheses and quantities

Let mu be a probability measure, X a measurable real random variable, and
assume X^2, X^4 and X^6 are integrable. Write

    s = E[X^2], m4 = E[X^4], m6 = E[X^6],
    g = m4 - s^2, Delta = m6 - m4^2/s.

Choose a,b != 0 with a^2 != b^2, and p,q>0 such that

    mu{X=a} >= p, mu{X=b} >= q, p+q <= 1.

These are lower bounds on actual atom masses; equality is not required.
The atom events are disjoint and measurable. Symmetry, independence and
Gaussianity are not assumed. Define

    u=a^2, v=b^2, A=p*u, B=q*v,
    s0=A+B,
    g0=p*q*(u-v)^2/(p+q),
    delta0=A*B*(u-v)^2/(A+B).

All three floors are strictly positive under the stated assumptions.

## 3. Two-square completion and moment floors

For A,B>0 and real u,v,c, direct expansion gives

    A*(u-c)^2 + B*(v-c)^2
      = A*B*(u-v)^2/(A+B)
        + (A+B)*(c-(A*u+B*v)/(A+B))^2.                 (3.1)

First, retaining the two atom contributions to the nonnegative X^2 integral
gives s>=s0>0. Thus all divisions below are legitimate.

Put c=m4/s. Expanding the square and using the stated integrability yields

    E[(X^3-c*X)^2] = m6 - 2*c*m4 + c^2*s = Delta.     (3.2)

Keeping the disjoint atom contributions in this nonnegative integral gives

    Delta >= p*a^2*(a^2-c)^2 + q*b^2*(b^2-c)^2
          >= delta0,                                (3.3)

where the second inequality is (3.1) with weights A,B. Replacing actual
atom masses by the lower bounds p,q is legitimate before completion because
each squared summand is nonnegative. No independence or cancellation of
uncontrolled terms is used.

Probability normalization separately gives

    E[(X^2-s)^2] = m4 - 2*s*E[X^2] + s^2 = g.

Keeping its two atom contributions and using (3.1) with weights p,q and
center c=s gives g>=g0. Consequently

    s >= s0 > 0,   g >= g0 > 0,   Delta >= delta0 > 0. (3.4)

For an arbitrary finite measure of mass M, the centered identity instead
uses center E[X^2]/M and gap E[X^4]-E[X^2]^2/M.
The unnormalized gap m4-s^2 is NOT generally nonnegative. The residual
identity (3.2) does not require total mass one, but the combined result here
does: do not export the probability-only gap to an arbitrary measure.

## 4. Sharpness and excluded boundary cases

For fixed admissible p,q,a,b, the law

    p*delta_a + q*delta_b + (1-p-q)*delta_0

attains s=s0 and Delta=delta0: its residual vanishes at zero and its center
is (A*u+B*v)/(A+B). Thus the Delta floor cannot be increased using only
these four inputs over this class of probability laws.

The gap floor is separately attained by putting the remaining mass at any
real point z with

    z^2=(p*u+q*v)/(p+q).

Then s=z^2, the residual gap square vanishes on the added point, and (3.1)
attains equality. Such a z exists since u,v>0. These are separate sharpness
claims, not a claim that both floors are simultaneously sharp for one law.

A zero atom together with one nonzero absolute radius is not a strict-Delta
case: P(0)=1/2, P(1)=P(-1)=1/4 gives g=1/4 but Delta=0.
Atoms a and -a have the same squared radius and also need not give strictness.
These are the distinctions recorded by AUD-309-REVIEW-STRICTNESS-01.
The floors can vanish in a varying family as an atom mass tends to zero or
the squared radii coalesce. No family-uniform floor follows without a uniform
positive bound on the displayed expressions. Two atoms are sufficient,
not necessary; continuous laws may have strictly positive gaps.

## 5. All-direction scalar floors without moment upper bounds

Let t be any real number in [0,1/4], including both endpoints. Put x=s^2.
The exact source formulas, expressed in s,g,Delta, are

    D0=x*(g+x), D1=g*(g+4*x)/4,
    D(t)=(1-4*t)*D0+4*t*D1,
    N=x*g*(g+2*x), S(t)=N/D(t),
    T(t)=(1-4*t)*Delta+4*t*(Delta+9*s*g)/4.           (5.1)

D is the scalar pin determinant, S is the scalar Schur expression, and T
is the source's `d2Tau` variance expression, NOT its square root.
Their probabilistic identification remains outside this note.

Both determinant endpoints are positive. Monotonicity of their positive
polynomials in s and g, followed by convex interpolation, gives

    D(t) >= D_floor
       := min(s0^2*(g0+s0^2), g0*(g0+4*s0^2)/4) > 0. (5.2)

For the Schur expression, it would be wrong to put lower bounds separately
into an uncontrolled quotient denominator. Instead use N>0 and the affine
D to obtain S(t)>=min(N/D0,N/D1). Direct simplification gives

    N/D0 = g*(g+2*x)/(g+x) >= g,
    N/D1 = 4*x*(g+2*x)/(g+4*x) >= 2*x.

The inequalities follow, respectively, from gx/(g+x)>=0 and
2*x*g/(g+4*x)>=0. Hence

    S(t) >= S_floor := min(g0,2*s0^2) > 0.           (5.3)

Finally both tau endpoints are monotone in the positive inputs, so

    T(t) >= T_floor
       := min(delta0,(delta0+9*s0*g0)/4) > 0.         (5.4)

These are convenient lower bounds, not optimal determinant/Schur/tau
constants. They require no upper bound on the remaining moments. Their
homogeneous degrees under X -> cX are 8,4,6 respectively; the moment
floors have degrees 2,4,6. Negative c is allowed with c!=0.

## 6. Exact example

For p=q=1/4,a=1,b=2 and remaining mass1/2 at zero,

    s=5/4, m4=17/4, m6=65/4, g=43/16, Delta=9/5,
    s0=5/4, g0=9/8, delta0=9/5.

Equations (5.2)-(5.4) give, for every t in [0,1/4],

    D(t) >= 531/256,  S(t) >= 9/8,  T(t) >= 9/5.

This is an illustrative discrete probability law, not the periodic spectral
law and not a substitute for one.

## 7. Reproduction and interpretation

Run the twelve finite-model methods directly:

    python -B -S reviews/d2_two_atom_floors_20261006/test_two_atom_bounds.py
    python -B -O -S reviews/d2_two_atom_floors_20261006/test_two_atom_bounds.py

Run the source-bound wrapper with the four actual code mutations:

    python -B -S tests/test_d2_two_atom_floors.py
    python -B -O -S tests/test_d2_two_atom_floors.py

The wrapper authenticates five payload files and the consumed D2Schur blob,
executes the twelve methods, checks fixed finite-case counts, and injects
four mutations in isolated temporary directories. Each must exit1 with its
specified failed test names, twelve collected methods and zero harness errors.
Missing code, crashes, timeouts and generic nonzero exit codes do not count as
successful mathematical controls. The manifest is an identity inventory, not
an independent authenticity root or a proof verdict.

The original twelve-method test was run before its helper existed and failed;
the wrapper was similarly observed failing before the packet manifest existed.
Finite coverage includes216 square-completion cases,64 probability laws,
384 directional evaluations,16 Delta and3 gap attainment cases,2 mass-bound
cases,18 scaling checks and explicit countermodels. These counts overlap
and must not be summed into a count of independent theorems.

The dedicated read-only workflow executes the wrapper in normal/optimized
Python, retains original output plus tested commit/run/attempt identity, and
does not modify existing required checks. Configured CI is not a passed run;
actual run identities and conclusions belong in the PR discussion.

A future formalization should first prove (3.1) and the atom-integral lower
bound with explicit measurable disjoint events, then derive (3.3)-(5.4).
No new Lean targets, kernel receipt, actual periodic-law construction,
summability theorem, Gaussian/Palm identification, interval-quadrature result,
parent lifetime theorem, independent alignment or scientific acceptance is
asserted by this packet. Existing mathematical and execution failures remain
historical evidence and are not overwritten by these additive files.
