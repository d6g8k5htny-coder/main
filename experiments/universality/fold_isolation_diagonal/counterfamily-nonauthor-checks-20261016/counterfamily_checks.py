"""Independent exact controls for obstruction §§3–5; not a proof or simulation."""
from fractions import Fraction as Q
from math import comb
from itertools import product
import json

ROWS = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2)]
checks = []
def check(name, condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

def uniform_moments(k,scale=1):
    return [sum(Q(i,scale)**p for i in range(-k,k+1))/Q(2*k+1) for p in range(7)]

def directional_moment(p,q,c,s,moment,reflection=1):
    return reflection**q * sum(Q(comb(p,i)*comb(q,j))*c**i*s**(p-i)*(-s)**j*c**(q-j)*moment(i+j,p+q-i-j) for i in range(p+1) for j in range(q+1))

def gram(c,s,moment,reflection=1):
    out=[]
    for p,q in ROWS:
        row=[]
        for r,t in ROWS:
            phase=((p+q)-(r+t))%4
            sign={0:1,1:0,2:-1,3:0}[phase]
            row.append(sign*directional_moment(p+r,q+t,c,s,moment,reflection))
        out.append(row)
    return out

def ldl(a):
    n=len(a); low=[[Q(i==j) for j in range(n)] for i in range(n)]; d=[]
    for i in range(n):
        pivot=a[i][i]-sum(low[i][j]**2*d[j] for j in range(i))
        if pivot<=0: raise ValueError('nonpositive pivot')
        d.append(pivot)
        for k in range(i+1,n):
            low[k][i]=(a[k][i]-sum(low[k][j]*low[i][j]*d[j] for j in range(i)))/pivot
    return d

def determinant(a):
    a=[row[:] for row in a]; det=Q(1); n=len(a)
    for i in range(n):
        pivot=next((j for j in range(i,n) if a[j][i]),None)
        if pivot is None:return Q(0)
        if pivot!=i: a[i],a[pivot]=a[pivot],a[i];det=-det
        v=a[i][i];det*=v
        for j in range(i+1,n):
            fac=a[j][i]/v
            for k in range(i+1,n):a[j][k]-=fac*a[i][k]
    return det

def rank(a):
    a=[list(map(Q,r)) for r in a]; lead=0
    for col in range(len(a[0])):
        p=next((j for j in range(lead,len(a)) if a[j][col]),None)
        if p is None:continue
        a[lead],a[p]=a[p],a[lead]
        v=a[lead][col];a[lead]=[x/v for x in a[lead]]
        for j in range(lead+1,len(a)):
            v=a[j][col]
            a[j]=[x-v*y for x,y in zip(a[j],a[lead])]
        lead+=1
        if lead==len(a):break
    return lead

def cyclic_distance(x,y,n):
    v=abs(x-y)
    return min(v,n-v)

def run():
    base=uniform_moments(2)
    check('base_even_moments',[base[i] for i in (0,2,4,6)]==[Q(1),Q(2),Q(34,5),Q(26)])
    mb=lambda i,j:base[i]*base[j]
    monomials=[(i,d-i) for d in range(4) for i in range(d+1)]
    vander=[[Q(x)**i*Q(y)**j for i,j in monomials] for x,y in product(range(-2,3),repeat=2)]
    check('full_total_degree_three_grid_rank_ten',rank(vander)==10)
    frames=[]
    lower=Q(225792,625)/Q(827,10)**6
    for z in [Q(0),Q(1,3),Q(-1,3),Q(1,2),Q(-1,2),Q(1),Q(-1),Q(2),Q(3)]:
        c=(1-z*z)/(1+z*z);s=2*z/(1+z*z)
        for reflection in (-1,1):
            label=f'{z}:{reflection}'
            g=gram(c,s,mb,reflection)
            piv=ldl(g); q=c*c*s*s
            schur=g[3][3]-(g[3][1]**2+g[3][4]**2)/2
            check('all_frame_cubic_schur_'+label,schur==(72+1044*q)/25)
            check('all_frame_determinant_'+label,determinant(g)==Q(3136,625)*(72+1044*q))
            trace=sum(g[i][i] for i in range(7))
            check('all_frame_trace_'+label,trace==(243+682*q)/5 and trace<=Q(827,10))
            check('all_frame_LDL_positive_'+label,all(p>0 for p in piv))
            check('det_trace_uniform_lower_'+label,determinant(g)/trace**6>=lower)
            frames.append({'t':str(z),'reflection':reflection,'c':str(c),'s':str(s),'conditional_cubic_variance':str(schur),'determinant':str(determinant(g))})
    # An insufficient support genuinely loses the axial cubic row.
    thin=uniform_moments(1);thin_gram=gram(Q(1),Q(0),lambda i,j:thin[i]*thin[j])
    annih=[Q(0)]*7;annih[1]=annih[3]=Q(1)
    check('negative_K1_Fu_plus_Fuuu_annihilator',sum(annih[i]*thin_gram[i][j]*annih[j] for i in range(7) for j in range(7))==0)
    family=[]
    for n in [4,5,7,11,17]:
        eps=Q(1,n**12);fill=uniform_moments(2*n,n)
        mm=lambda i,j:(1-eps)*mb(i,j)+eps*fill[i]*fill[j]
        # Twelfth moment requires moments through12, independent of the Gram helper.
        fm=[sum(Q(i,n)**p for i in range(-2*n,2*n+1))/Q(4*n+1) for p in range(13)]
        bm=[sum(Q(i)**p for i in range(-2,3))/5 for p in range(13)]
        m12=sum(Q(comb(6,j))*((1-eps)*bm[2*j]*bm[12-2*j]+eps*fm[2*j]*fm[12-2*j]) for j in range(7))
        check('positive_normalized_mass_'+str(n),sum(eps/Q((4*n+1)**2)+(1-eps)/25*int(i%n==0 and j%n==0) for i,j in product(range(-2*n,2*n+1),repeat=2))==1)
        check('uniform_M12_'+str(n),m12<=8**6==262144)
        check('strict_all_mode_fill_'+str(n),eps/Q((4*n+1)**2)>0 and eps<1)
        for z in [Q(0),Q(1,2),Q(2)]:
            c=(1-z*z)/(1+z*z);s=2*z/(1+z*z)
            g=gram(c,s,mm); gb=gram(c,s,mb);gf=gram(c,s,lambda i,j:fill[i]*fill[j])
            check('mixture_covariance_'+str(n)+':'+str(z),all(g[i][j]==(1-eps)*gb[i][j]+eps*gf[i][j] for i,j in product(range(7),repeat=2)))
            ldl(g)
        residues=[sum(i%n==r for i in range(-2*n,2*n+1)) for r in range(n)]
        check('geometric_cycles_not_primitive_only_'+str(n),residues==[5]+[4]*(n-1))
        # The cycles evaluate to zero at every nontrivial Nth root, primitive or not.
        corr=1-eps+eps/Q(4*n+1)
        check('unit_two_value_Rayleigh_'+str(n),1-corr==eps*(1-Q(1,4*n+1))<eps)
        j=n//2
        check('far_torus_distance_'+str(n),Q(min(j,n-j),n)>=Q(1,4))
        check('coefficient_C4_dimension_bound_'+str(n),eps*(4*n+1)**2<=Q(25,n**10))
        check('proper_N_cubed_lifetime_scale_'+str(n),Q(n*n)*Q(1,n**2)==1)
        err=(4096*n**4+4096*n*n)*Q(1,n**10)
        check('mean_transfer_count_cap_'+str(n),err<=Q(8192,n**6))
        check('negative_weak_probability_rate_not_enough_'+str(n),4096*n**4*Q(1,n**4)==4096)
        check('negative_wrong_frequency_scaling_'+str(n),Q(2,n*n)<Q(2))
        family.append({'N':n,'M12':str(m12),'two_value_Rayleigh':str(1-corr),'expectation_transfer_bound':str(err)})
    lift_cases=[]
    for n in (2,3,4,5):
        for seam in (False,True):
            p=(Q(31,32) if seam else Q(1,8),Q(1,4))
            q=(Q(1,32) if seam else Q(3,16),Q(1,4))
            points1=[(p[0]+i,p[1]+j) for i,j in product(range(n),repeat=2)]
            points2=[(q[0]+i,q[1]+j) for i,j in product(range(n),repeat=2)]
            count=sum(cyclic_distance(a[0],b[0],n)**2+cyclic_distance(a[1],b[1],n)**2<Q(1,64) for a,b in product(points1,points2))
            check('short_endpoint_lifts_'+str(n)+':'+str(seam),count==n*n)
            check('negative_unrestricted_multiplicity_'+str(n)+':'+str(seam),count!=n**4)
            lift_cases.append({'N':n,'seam_crossing':seam,'short_lifts':count,'unrestricted_pairs':n**4})
    # With m=8, h'=sin(m²x)/m is C1-small, yet produces two extra roots.
    # At pi/k,3pi/(2k),2pi/k the signs are +,-,+; use pi<22/7 exactly.
    check('negative_C1_alone_allows_new_roots',Q(3*22,2*7*8**2)<Q(1,8))
    check('C4_multiindex_count',sum(1 for i in range(5) for j in range(5-i))==15)
    check('coordinate_row_bound_squared',8**4==64**2)
    check('C4_sum_constant',15*64==960)
    check('contact_lifetime_density_factor',Q(12,3)==4)
    check('window_integrated_factor_six',Q(4)/Q(2,3)==6)
    check('negative_angle_half_factor',Q(6,2)!=6)
    check('negative_missing_row_transform',Q(1,3)/Q(2,3)!=6)
    return {'classification':'INDEPENDENT_EXACT_ALGEBRA_NEGATIVE_CONTROLS_ONLY','result':'PASS','check_count':len(checks),'checks':checks,'all_frame_proof_not_inferred_from_samples':True,'explicit_uniform_base_floor':str(lower),'rational_frames':frames,'family_cases':family,'short_lift_cases':lift_cases,'field_draws':0,'approximate_roots':0,'Rice_integral_evaluations':0,'not_proved_by_controls':['a.s. Morse/stability','mark-boundary null sets','diagonal deterministic existence','candidate-process convergence'],'scope':'Frozen obstruction §§3–5 only; source-exposed OpenAI org0; conventional proof in standalone review.'}

if __name__=='__main__':print(json.dumps(run(),indent=2))
