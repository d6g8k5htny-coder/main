import math
G=math.gamma
pi=math.pi
# (37) evaluated in closed form from the Euclidean limit covariance of J=(F,F_u,F_uu,F_uuu,F_v,F_uv,F_vv):
# odd block (F_u,F_v,T): cov [[1,0,-3],[0,1,0],[-3,0,15]], det 6, (C^-1)_33=1/6 -> density at (0,0,T) = (2pi)^-1.5 6^-0.5 exp(-T^2/12)
# even block: F_uv indep var 1; (F_uu,F_vv) cov [[3,1],[1,3]] (after integrating b out), det 8, quad form at (0,A) = 3A^2/8
I_k = 0.5*12**(-7/6)*G(7/6)                       # ∫_0^∞ k^{4/3} e^{-12k^2} dk
p_odd_const = (2*pi)**-1.5*6**-0.5
I_A = math.sqrt(pi)/(4*(3/16)**1.5)                # ∫_{A<0} A^2 e^{-3A^2/16} dA
p_uuvv_const = 1/(2*pi*math.sqrt(8))
p_uv0 = (2*pi)**-0.5
cG_37 = 144*(2*pi)*(p_odd_const*I_k)*p_uv0*(p_uuvv_const*I_A)
cG_39 = 2*G(7/6)*(1.5)**(1/3)/(3*math.sqrt(3)*pi**1.5)
print("c_G from (37) closed-form integration:", repr(cG_37))
print("c_G from (39) stated closed form     :", repr(cG_39))
print("relative difference                  :", abs(cG_37-cG_39)/cG_39)
# brute-force numeric quadrature of (37) as an independent cross-check (midpoint rule, no closed forms for the k and A integrals)
def quad(f,a,b,n):
    h=(b-a)/n; return h*sum(f(a+(i+0.5)*h) for i in range(n))
Ik_num = quad(lambda k: k**(4/3)*math.exp(-12*k*k), 0.0, 3.0, 200000)
IA_num = quad(lambda A: A*A*math.exp(-3*A*A/16), -40.0, 0.0, 200000)
cG_num = 144*(2*pi)*(p_odd_const*Ik_num)*p_uv0*(p_uuvv_const*IA_num)
print("c_G from (37) midpoint quadrature    :", repr(cG_num), " rel diff vs (39):", abs(cG_num-cG_39)/cG_39)
# source-formula inputs stated in §8: p_G(0)=1/(2pi), p_Vu(0)=1/(2pi sqrt3), tau_u^2=6, s_u^2=8/3
cG_src = G(7/6)/(24**(1/3)*math.sqrt(pi))*(2*pi)*(1/(2*pi))*(1/(2*pi*math.sqrt(3)))*6**(2/3)*((8/3)/2)
print("c_G from §8 source-coefficient form  :", repr(cG_src), " rel diff vs (39):", abs(cG_src-cG_39)/cG_39)
# normalization check in §8: (1/3)(3/2)^(2/3) == 12^(-1/3) and 144*12^(-7/3) == 12^(-1/3)
print("(1/3)(3/2)^(2/3) =",(1/3)*1.5**(2/3)," 12^(-1/3) =",12**(-1/3), " 144*12^(-7/3) =",144*12**(-7/3))
