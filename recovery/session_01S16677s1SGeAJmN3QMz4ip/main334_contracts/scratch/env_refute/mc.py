import numpy as np
rng=np.random.default_rng(1)
for d in (1,2):
    n=2*d+2; alv=1.5; av=np.sqrt((alv+2)/alv)
    Av = av*rng.random(4_000_000)**(1/alv)
    v=rng.normal(size=n)*0.8
    phi=lambda x: np.exp(-np.sum(x**2,axis=-1)/2)/(2*np.pi)**(n/2)
    # weight homogeneous of degree 2d in the observation: W = |U1|^(2d) (deterministic given U), toy for Delta ~ s^(2d)
    Wv = abs(v[0])**(2*d)
    piF = np.mean(Av**-n * phi(v[None,:]/Av[:,None]))
    lhs = piF*Wv
    rhs = np.mean(Av**-2 * phi(v[None,:]/Av[:,None]) * abs(v[0]/Av)**(2*d))
    print(f"d={d}: lhs {lhs:.6e} rhs {rhs:.6e} ratio {lhs/rhs:.5f}")
