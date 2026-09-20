# LPW author clarification A1/A2 — additive, exact-target bound

**Target:** LPW-CAND-20260912 v1.0, whole-file SHA-256 cf58f72eb0399c626160c143b50f674ff3bd5c449225211edd6fd378a374e1a5.

**Status:** Author clarification; no edit to approved bytes; no independent approval of this new appendix claimed. Kimi's reported approval still binds only to its stated original target. This appendix does not use any new numerical covariance estimate.

## A1. Bound the nonstationary residual by its representation

Let V_r=(U_r,J), Gamma_r=Cov(V_r), C_r(z)=Cov(f(z),V_r), and a_r(z)=C_r(z) Gamma_r^{-1}. Write

    R_r(z)=f(z)-a_r(z)V_r,
    f_{r,v}(z)=R_r(z)+a_r(z)v.

The joint Gaussian construction makes R_r independent of V_r. It need not be stationary. No stationary-field supremum theorem is applied to R_r.

For k a fixed derivative order, define

    ||g||_{C^k}=max_{|alpha|<=k} sup_z |partial^alpha g(z)|.

For each coordinate i write a_{r,i}(z) for the corresponding coefficient function. Pathwise triangle inequality gives

    ||R_r||_{C^k}
    <= ||f||_{C^k} + sum_i ||a_{r,i}||_{C^k} |V_{r,i}|.

Consequently, for p>=1, Minkowski's inequality gives

    (E ||R_r||_{C^k}^p)^(1/p)
    <= (E ||f||_{C^k}^p)^(1/p)
       + sum_i ||a_{r,i}||_{C^k} (E |V_{r,i}|^p)^(1/p).

The original field f has finite C^k moments by its absolutely summable Gaussian Fourier majorant. Covariance convergence bounds the moments of V_r. Uniform invertibility of Gamma_r, together with the original candidate's covariance Cauchy-Schwarz bound for each derivative of C_r, bounds every ||a_{r,i}||_{C^k}. Thus the right side is uniformly finite. Adding the deterministic mean yields

    (E ||f_{r,v}||_{C^k}^p)^(1/p)
    <= (E ||R_r||_{C^k}^p)^(1/p)
       + sum_i ||a_{r,i}||_{C^k}|v_i|.

For v=(u_r,j) in the prescribed compact parameter set this is uniform. The statement is an everywhere-defined continuous Gaussian regression version, not an arbitrary version altered at particular zero-probability observations.

For Section 7 use k=4,p=1 to obtain B4. For Section 9 use the six-coordinate version with k=3,p=4 to obtain B3. The field norm is bounded by sums of known objects; nonstationarity of the residual introduces no missing stationarity premise.

The Markov estimate remains pointwise in j:

    Q_r(M4<=K | J=j)>=1/2,  K=max(1,2B4).

Integrate this inequality over the thin jet set E_r. Do not subtract an unconditional fixed tail probability from its order-r mass.

## A2. Separate top-order seminorm from full norm

Throughout the original proof, set

    M4_top(f)=max_{|alpha|=4} sup_z |partial^alpha f(z)|,
    N4(f)=||f||_{C^4}=max_{|alpha|<=4} sup_z |partial^alpha f(z)|.

Then M4_top<=N4. The Taylor remainder uses M4_top. The regression estimate may bound N4, hence also M4_top. One valid choice is

    B4 = sup_{r,j} E[N4(f) | U_r=u_r,J=j],
    K=max(1,2B4).

This potentially enlarges a constant but changes no assertion or exponent. For the denominator retain

    M3(f)=max(1,||f||_{C^3}),
    B3=sup_r E_{Q_r}[M3^4], C_Z=4B3.

## Reviewer conversion request

Confirm that these explicit inequalities are already licensed by the original proof and that the clarified norm convention preserves every downstream estimate. No favorable conversion is presumed. The approved original file is left intact.
