# Exact BF identity: transverse curvature under the six pins

Scientific effect: NONE on STATUS. This is not A3.

## Theorem (Bargmann-Fock, any r>0)

Let C(u)=exp(-|u|^2/2). Pins at M=(-r/2,0), S=(r/2,0):

    f(M)=b, f(S)=b-k r^3, grad f(M)=grad f(S)=0.

Then, under the Gaussian regression on those six values,

    f_ss(M) ~ N(-b, 2)
    f_ss(S) ~ N(-(b-k r^3), 2)

exactly, for every r>0 and every (b,k).

## Proof

g(X)=f_ss(X)+f(X). For this kernel, d_ss C(u)+C(u)=0 along the t-axis. The six cross-covariances of g(M) and of g(S) with the pins vanish, and Var(g)=2. Hence g is independent of the pin sigma-algebra.

Checked at r in {0.17, 0.3, 1.0, 2.5}.

## What this is not

Parent (5.4) still needs type-convergence of W_r/r^2, including the soft eigenvalue alpha_M=f_tt(M)/r and mixed beta. Pure cubic: alpha_M=-6k exactly. Quartic remainder is (5.2). That remainder plus UI is A3 and remains AMEND.
