# A sharp stationary-variance successor to the RN density majorant

This author-side theorem refines the factor512 in the original density-weighted
moment lemma. It proves an analytic upper envelope. It does not independently
verify the spatial determinant/Mahalanobis premises, create a new spatial cover,
reprove the imported H3 normalization, or change any scientific status. The
original512-based theorem, its wedge result, and all frozen bytes remain intact.

## Exact field and imported premises

The field has normalized covariance K(x,y)=k(x)k(y), where
k(t)=sum[j in Z] exp(-(t+24j)^2/2) / sum[j in Z] exp(-(24j)^2/2).
It is stationary, centered before conditioning, and its derivative covariance
matrices are positive semidefinite. Write m2=-k''(0), m4=k''''(0).
Consequently the unconditional variances of original Hessian coordinates
(xx,yy,xy) at every site are exactly (m4,m4,m2^2).

Fix r=1/20,b=6/5, pins M=(-1/40,0), S=(1/40,0), pin values
(b,0,0,b-1/48000,0,0), and the full height window [b-1/48000,b].
The existing, separately proved six-pin energy is at most
1266466/160083<8. The imported fixed-r typed H3 floor is
Z_lo=15518583475065571/2000000000000000000. Neither imported proof is rerun.
The archive, relevant members, original repository dependencies and current
eligibility are checked through the existing non-executing source binder.

For the same six-pin conditional three-jet law Y=(f,fx,fy), assume its actual
covariance Sigma has det(Sigma)>=D>0 and its Mahalanobis energy at (v,0,0)
has q(v,y)>=Q>=0 throughout the proposed spatial region and full mark window.
An independently chosen entrywise covariance hull is not a replacement law.
Positive determinant with actual covariance PSD makes Sigma positive definite.

## General stationary-variance coefficient

Gaussian projection onto the six-pin span followed by the residual Y span is
orthogonal. The nine-observation energy is E6+q<=T=8+q. For any original Hessian
coordinate X, let s=Var(X) and a be its explained variance under this joint
projection. Then 0<=a<=s, conditional variance=s-a and conditional mean squared
is at most a*T. If s=0, X is identically zero. Otherwise lambda=a/s lies in[0,1].
The sixth noncentral Gaussian moment, divided by s^3, is at most

15(1-lambda)^3 +45T lambda(1-lambda)^2
 +15T^2 lambda^2(1-lambda)+T^3 lambda^3.

Its Bernstein controls are (15,15T,5T^2,T^3), each<=T^3 for T>=8.
Thus E|X|^6<=s^3*T^3 and ||X^2||_3<=s*T. This proof retains the coupling
between explained and residual variance; separate maximal mean and variance
bounds cannot be combined as if simultaneously attained.

For H=((A,C),(C,B)), |det H|<=|AB|+C^2<=(A^2+B^2)/2+C^2.
Minkowski in L3 therefore gives

||det H||_3 <= (Var(A)/2+Var(B)/2+Var(C))*T
              = (m4+m2^2)*T.

Hölder(3,3,3), without independence between any Hessians or coordinates, yields

E[|det H_M det H_S det H_y| | all nine pins]
 <= (m4+m2^2)^3*(8+q)^3.

Singular conditional Hessian covariance and deterministic conditional coordinates
are allowed. Type/event indicators in[0,1] can be discarded for this upper bound.

## Certified normalized moments

The repository's kernel_derivative(2,0) and kernel_derivative(4,0) are evaluated
with precision110 and outward rounding at256bits. That implementation includes
the j=-1,0,1 images, a uniform omitted-image bound and a positive interval for
the full normalizer. The derivative sign in m2 is retained. In particular the
calculation is not a planar approximation, m4=3 is not assumed, and image tails
are not replaced by zero. Exact rational intervals certify

0<m2<=1, 0<m4<3000000001/1000000000,
(m4+m2^2)^3 <64000001/1000000 <65.

The last inequality is checked on the full rational interval for the expression,
not inferred from the looser displayed separate moment caps. Every endpoint is
recorded in the report. Coefficient64 is refused by this certificate.

## Density, full mark and conditional wedge application

For C=64000001/1000000, multiply the moment and density bounds before taking
the lower bound q>=Q:

I(y) <= ell*C*(8+Q)^3*exp(-Q/2)
        /(Z_lo*(2*pi)^(3/2)*sqrt(D)),  ell=1/48000.

The function (8+q)^3 exp(-q/2) has derivative
-(q+8)^2*(q+2)*exp(-q/2)/2<0 for q>=0. Lowering Q to min(Q,2048)
in the entire function is a safe resource limit, not underflow or a proof of
the supplied Q. The height window length appears once. A transformed density
uses its matching absolute determinant Jacobian once, distinct from spatial area.
The lower endpoint of the majorant expression is not an integrand lower bound:
the typed integrand enclosure is [0,majorant_upper].

Conditionally on the existing wedge premises D=8/10^25,Q=103, the same fixed pins
and full height interval, the checker calculates a sharper integrand upper bound.
It also checks the conservative coefficient65 envelope satisfies

I(y)<13/10240000.

If that existing spatial certificate applies throughout rho in[1/10,11/100]
and signed turns in[-1/1024,1/1024], the region has area21*pi/5120000.
Multiplying this one area by the nonnegative uniform cap gives

integral_W I(y)dy <429/26214400000000.

These conservative fractions also follow by multiplying the original clean
integrand/integral caps by65/512. The new checker directly encloses the expression
instead of relying on that numerical relationship. No rectangle areas are added,
and no claim is made to replay the twelve strips or establish new coverage.

Direct exact evaluation with C=64000001/1000000 additionally proves the stronger
simple headlines

I(y)<1/1000000, integral_W I(y)dy<33/2560000000000.

The integral headline uses the same single area and pi<22/7. The exact interval
upper is checked strictly against each headline; a diagnostic decimal is not used.

## Evidence and retained limits

The portable CLI verifies exact dependency bytes before importing repository
modules and again after calculation. Current source eligibility is checked twice.
The report pins its own checker, proof and dependency manifest, has no absolute
runtime paths, and is deterministic across normal and optimized Python. An
existing output path, source drift or unverifiable premise contract is refused.
All archived programs remain data in this calculation.

The full annulus, all radii, all pin orientations, weighted-Palm/event interfaces,
RN/q0 closure and required independent acceptance remain unresolved by this
theorem. Technical verification has zero organizational-independence credit.
