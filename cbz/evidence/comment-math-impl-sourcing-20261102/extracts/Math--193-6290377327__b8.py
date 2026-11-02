#!/usr/bin/env python3
"""NEW single-use mathematical rational bound producer; no imports from project code."""
import hashlib,json,sys,math
from fractions import Fraction as F
from pathlib import Path
CAP=200000
PEAK=0
STAGE='initialization'
class Hold(Exception): pass
class Q:
    def __init__(self,n=0,d=1):
        global PEAK
        self.f=n.f if isinstance(n,Q) else F(n,d)
        bits=max(abs(self.f.numerator).bit_length(),self.f.denominator.bit_length())
        PEAK=max(PEAK,bits)
        if bits>CAP: raise Hold('rational component exceeds 200000 bits')
    @staticmethod
    def cv(x): return x if isinstance(x,Q) else Q(x)
    @staticmethod
    def make(n,d):
        if d<0:n,d=-n,-d
        g=math.gcd(n,d);n//=g;d//=g
        if max(abs(n).bit_length(),d.bit_length())>CAP: raise Hold('preconstruction rational component cap')
        return Q(n,d)
    def __add__(self,x):
        y=self.cv(x).f;a,b=self.f.numerator,self.f.denominator;c,d=y.numerator,y.denominator
        g=math.gcd(b,d)
        return self.make(a*(d//g)+c*(b//g),(b//g)*d)
    __radd__=__add__
    def __sub__(self,x): return self+(-self.cv(x))
    def __rsub__(self,x): return self.cv(x)-self
    def __mul__(self,x):
        y=self.cv(x).f;a,b=self.f.numerator,self.f.denominator;c,d=y.numerator,y.denominator
        g=math.gcd(abs(a),d);h=math.gcd(abs(c),b)
        return self.make((a//g)*(c//h),(b//h)*(d//g))
    __rmul__=__mul__
    def __truediv__(self,x):
        y=self.cv(x).f
        if y.numerator==0:raise ZeroDivisionError('zero rational divisor')
        return self*self.make(y.denominator,y.numerator)
    def __rtruediv__(self,x): return self.cv(x)/self
    def __pow__(self,n):
        if type(n) is not int or not 0<=n<=15: raise Hold('power contract')
        ans=Q(1)
        for _ in range(n): ans=ans*self
        return ans
    def __lt__(self,x): return self.f<self.cv(x).f
    def __le__(self,x): return self.f<=self.cv(x).f
    def __eq__(self,x): return self.f==self.cv(x).f
    def __gt__(self,x): return self.cv(x)<self
    def __ge__(self,x): return self.cv(x)<=self
    def __neg__(self): return Q(-self.f)
def p2(k):
    if type(k) is not int or abs(k)+1>CAP: raise Hold('dyadic denominator/numerator cap')
    return Q(1<<k) if k>=0 else Q(1,1<<(-k))
def clog(x):
    x=Q.cv(x)
    if x<=0: raise Hold('ceil_log2 requires positive rational')
    k=x.f.numerator.bit_length()-x.f.denominator.bit_length()
    if p2(k)<x: k+=1
    if not p2(k-1)<x or not x<=p2(k): raise Hold('ceil_log2 invariant')
    return k
def ceildiv(a,b): return -((-a)//b)
def sha(b): return hashlib.sha256(b).hexdigest()
def fp(q):
    q=Q.cv(q); n=str(q.f.numerator); d=str(q.f.denominator)
    return {'sha256_n_slash_d':sha((n+'/'+d).encode()),'numerator_bits':abs(q.f.numerator).bit_length(),'denominator_bits':q.f.denominator.bit_length(),'sign':(q.f>0)-(q.f<0)}
def must(v,label):
    if not v: raise Hold('failed inequality/control: '+label)
def reject(fn,label):
    try: fn()
    except (Hold,ZeroDivisionError): return label
    raise Hold('adverse control accepted: '+label)
def hashcheck(b,h):
    if sha(b)!=h: raise Hold('source SHA mismatch')
def validate(t):
    e=t['epsC'];q=t['q'];B=t['B'];M0=t['M0'];eta=t['eta'];R=t['R']
    checks={
      'q_range':0<q<=1,'B_range':1<=B,'M0_range':1<=M0,'eta_range':0<eta<=1,'R_range':0<R<=1,
      'small':16*t['Hstar']*q<=e/4,
      'large':Q(16,3)*t['Hstar']*t['Jc']/B**15<=e/4,
      'C4_tail':8*B*12*t['P']*t['Tail']<=e/8,
      'transverse':8*B*12*t['P']*(t['Z']**2*eta**2/2)<=e/8,
      'varying_far':8*B*(t['Delta1']*R+t['Delta2']*R**2)+t['Cfar']*q*R<=e/4,
      'R_rpin':R<=t['rpin'],'R_eta':R<=eta/(2*t['EA']),
      'R_zeta':R<=t['zeta']/(2*t['D']),
      'R_cap':R<=t['hprime']**2*t['zetaprime']/(16*t['M']),
      'R_rbar':R<=t['rbar'],'R_Vcoef':R<=e/(4*t['Vcoef']),
      'ellstar_identity':t['ellstar']==(q*R)**3,
      'ellstar_domain':0<t['ellstar']<=1,
      'ellstar_radius':t['ellstar']<=(q*t['rbar'])**3,
    }
    for k,v in checks.items(): must(v,k)
    return checks

def main():
    global STAGE
    here=Path(__file__).resolve().parent
    inpbytes=(here/'INPUT_V1.json').read_bytes(); inp=json.loads(inpbytes)
    must(inp['epsilon']=={'numerator':1,'denominator':8},'fixed epsilon')
    must(inp['caps']=={'wall_seconds':30,'rational_component_bits':CAP,'evidence_bytes':5242880,'producer_invocations':1},'fixed caps')
    runtime=inp['runtime']
    must(sys.executable==runtime['sys_executable'],'absolute interpreter path')
    must(sys.version==runtime['version'] and list(sys.version_info[:3])==runtime['version_info'],'actual interpreter version')
    must(hasattr(sys,'set_int_max_str_digits')==runtime['has_set_int_max_str_digits'],'actual optional API capability')
    must(str(Path(sys.executable).resolve())==runtime['resolved_executable'],'resolved interpreter path')
    hashcheck(Path(sys.executable).resolve().read_bytes(),runtime['interpreter_sha256'])
    if hasattr(sys,'set_int_max_str_digits'): sys.set_int_max_str_digits(100000)
    root=Path(inp['repository_root']); bindings={}
    for p,h in inp['source_sha256'].items():
        bb=(root/p).read_bytes();hashcheck(bb,h);bindings[p]={'sha256':h,'bytes':len(bb)}
    freeze_bytes=(here/'PRE_EXECUTION_FREEZE_V1.json').read_bytes();freeze=json.loads(freeze_bytes)
    for p,h in freeze['members_sha256'].items(): hashcheck((here/p).read_bytes(),h)
    STAGE='meaningful controls'
    controls=[]
    for x,k in [(Q(1),0),(Q(2),1),(Q(1,2),-1),(Q(3,2),1),(Q(3,4),0),(Q(5,8),0),(Q(1,8),-3),(Q(9,8),1)]:
        must(clog(x)==k,'ceil_log2 boundary');controls.append('ceil_log2 '+str(x.f))
    controls += [reject(lambda:clog(0),'zero log rejected'),reject(lambda:clog(-1),'negative log rejected'),reject(lambda:p2(True),'boolean exponent rejected'),reject(lambda:p2(-CAP),'overcap dyadic rejected'),reject(lambda:hashcheck(b'x','0'*64),'wrong source SHA rejected')]
    for cc,ss in [(Q(1),Q(1)),(Q(2,3),Q(3,2)),(Q(7,5),Q(2))]:
        lhs=Q(24)/(cc**4*ss**8)*(ss**3/(2*cc)+3*ss/(4*cc**2)+Q(3)/(8*cc**3*ss))
        rhs=12/(cc**5*ss**5)+18/(cc**6*ss**7)+9/(cc**7*ss**9)
        must(lhs==rhs,'tail algebra');controls.append('exact tail algebra')
    STAGE='source constants'
    t={}; t['T']=1+2*sum((Q(1+k)**4 for k in range(1,128)),Q(0))+8*Q(129)**4/p2(128)
    t['A']=2*t['T']**2;t['N']=t['A']**2
    a0=Q(1,1056);b0=Q(1,2230272);v0=Q(3,901120)
    t['lambda']=min(Q(1),b0,min(a0,v0)/(2*(1+t['N']/a0)))
    t['rpin']=min(Q(1,4),t['lambda']/(6*t['N']))
    t['K0']=2*t['N']/t['lambda']+12*t['N']**2/t['lambda']**2
    t['K']=3*t['K0']+4*t['N']/t['lambda'];t['L']=1+12*t['N']/t['lambda']
    t['P']=(2/t['lambda'])**3
    t['Hstar']=96*t['P']*t['L']**4*(3*t['A']**4+12**4)
    t['c']=24/t['N'];t['Jc']=12/t['c']**5+18/t['c']**6+9/t['c']**7
    t['mu']=(t['rpin']/12)**6/2787840000
    floor=json.loads((root/inp['floor_path']).read_bytes())['floor']
    for name in ['rpin','mu']:
        must(t[name]==Q(int(floor[name]['numerator']),int(floor[name]['denominator'])),'adopted floor '+name)
    controls.append('actual adopted rpin and mu exact match')
    t['Lfar']=1+12*t['N']/t['mu']
    t['Cfar']=576/t['mu']**3*8*t['Lfar']**4*(3*t['A']**4+1)
    t['Cminus']=Q(2904569377252513601516770161,39614081257132168796771975168)
    t['epsilon']=Q(1,8);t['epsC']=t['epsilon']*t['Cminus']
    STAGE='direct dyadic selection'
    kq=max(0,clog(64*t['Hstar']/t['epsC'])); t['q']=p2(-kq)
    kB=max(0,ceildiv(clog(Q(64,3)*t['Hstar']*t['Jc']/t['epsC']),15));t['B']=p2(kB)
    t['Z']=12*t['B']**3;t['zeta']=12*t['q']**3
    t['Tcoef']=16*t['L']**4*(15*t['A']**6+t['Z']**4*t['A']**2)
    kM=max(0,ceildiv(clog(768*t['B']*t['P']*t['Tcoef']/t['epsC']),2));t['M0']=p2(kM)
    t['Tail']=t['Tcoef']/t['M0']**2
    keta=max(0,ceildiv(clog(384*t['B']*t['P']*t['Z']**2/t['epsC']),2));t['eta']=p2(-keta)
    t['M']=t['L']*(t['M0']+t['Z']);t['D']=t['K']*(t['M0']+t['Z'])
    t['Ea']=t['M']/6+t['D']/2;t['EA']=t['M']/2+t['D']
    t['Eg']=t['Ea']*t['M']+t['Z']/2*t['EA']+t['M']**2/4
    t['etaprime']=t['eta']/2;t['zetaprime']=t['zeta']/2
    hc=[Q(1,4),t['etaprime']/(48*t['M']),t['zetaprime']*t['etaprime']/(52*t['M']**2)]
    t['hprime']=min(hc)
    rc=[t['rpin'],t['eta']/(2*t['EA']),t['zeta']/(2*t['D']),t['hprime']**2*t['zetaprime']/(16*t['M'])]
    t['rbar']=min(rc)
    t['Dp']=3*t['N']*t['P']*(5/t['lambda']+2*t['Z']**2/t['lambda']**2)
    t['Delta1']=12*(t['P']*(2*t['Eg']*t['M']**2+t['Z']*t['eta']*t['Eg'])+t['Dp']*t['Z']**2*t['N']/4)
    t['Delta2']=12*t['P']*t['Eg']**2
    t['Vcoef']=8*t['B']*(t['Delta1']+t['Delta2'])+t['Cfar']*t['q']
    kR=max(0,clog(1/t['rbar']),clog(4*t['Vcoef']/t['epsC']));t['R']=p2(-kR)
    kell=3*(kq+kR);t['ellstar']=p2(-kell)
    checks=validate(t);controls.append('ordinary actual full witness accepted')
    bad=dict(t);bad['q']=Q(1);controls.append(reject(lambda:validate(bad),'tampered small sector rejected'))
    bad=dict(t);bad['R']=Q(1);controls.append(reject(lambda:validate(bad),'tampered radius rejected'))
    bad=dict(t);bad['ellstar']=t['ellstar']*2;controls.append(reject(lambda:validate(bad),'tampered ellstar rejected'))
    must(all(0<t[k] for k in t),'all derived constants positive')
    t['small_budget']=16*t['Hstar']*t['q'];t['large_budget']=Q(16,3)*t['Hstar']*t['Jc']/t['B']**15
    t['tail_budget']=8*t['B']*12*t['P']*t['Tail'];t['transverse_budget']=8*t['B']*12*t['P']*(t['Z']**2*t['eta']**2/2)
    t['varying_far_budget']=8*t['B']*(t['Delta1']*t['R']+t['Delta2']*t['R']**2)+t['Cfar']*t['q']*t['R']
    t['total_budget']=sum((t[k] for k in ['small_budget','large_budget','tail_budget','transverse_budget','varying_far_budget']),Q(0))
    must(t['total_budget']<=t['epsC'],'total allocation')
    t['cumulative_bound']=t['epsC']*Q(3,2)*(t['q']*t['R'])**2
    t['illustrative_bin_bound']=t['epsC']*Q(9,8)*(t['q']*t['R'])**2
    t['illustrative_lower_endpoint']=t['ellstar']/8
    for k,limit in [('small_budget',t['epsC']/4),('large_budget',t['epsC']/4),('tail_budget',t['epsC']/8),('transverse_budget',t['epsC']/8),('varying_far_budget',t['epsC']/4)]: t[k+'_margin']=limit-t[k]
    out={'status':'PASS_CONDITIONAL_MATHEMATICAL_INPUT_ONLY','schema_version':1,
      'actor':'/root/p5_effective_window_author_1020','claim':'58f65037-c222-4e94-b69b-50fb28ffe2f1',
      'bindings':{'producer_sha256':sha(Path(__file__).read_bytes()),'input_sha256':sha(inpbytes),'pre_execution_freeze_sha256':sha(freeze_bytes),'sources':bindings},
      'epsilon':{'numerator':1,'denominator':8,'scope':'mathematical validation only; not a study tolerance'},
      'dyadic_exponents':{'kq':kq,'kB':kB,'kM0':kM,'keta':keta,'kR':kR,'kellstar':kell},
      'branches':{'hprime_min_indices':[i for i,v in enumerate(hc) if v==t['hprime']],'rbar_min_indices':[i for i,v in enumerate(rc) if v==t['rbar']]},
      'rational_fingerprints':{k:fp(v) for k,v in sorted(t.items())},'checks':checks,
      'endpoint_interface':{'ellstar':{'numerator':1,'denominator_power_of_two':kell},
        'cumulative_outward_interval':{'lower_equals_upper':True,'factor_numerator':str((t['epsC']*Q(3,2)).f.numerator),'factor_denominator':str((t['epsC']*Q(3,2)).f.denominator),'multiply_by_two_to_power':-2*(kq+kR)},
        'illustrative_half_open_bin':{'lower_dyadic_exponent':kell+3,'upper_dyadic_exponent':kell,'weight_factor':'9/8','error_factor_numerator':str((t['epsC']*Q(9,8)).f.numerator),'error_factor_denominator':str((t['epsC']*Q(9,8)).f.denominator),'multiply_by_two_to_power':-2*(kq+kR),'lower_equals_upper':True,'scope':'mathematical validation only; no empirical bin'},
        'general_half_open_R':'epsilon*Cminus*(3/2)*(b^(2/3)-a^(2/3)); mathematical interface, no general root evaluation primitive implemented'},
      'controls':controls,'resource':{'peak_rational_component_bits':PEAK,'component_bit_cap':CAP,'wall_cap_seconds':30,'combined_evidence_cap_bytes':5242880,'producer_invocations':1},
      'scientific_execution':'HOLD_NOT_ADMITTED','imports':'Accepted P5 V1/V2 and Q14, actual GAD/classical marked-density premises; exact Python integer/Fraction semantics; no kernel certification'}
    raw=(json.dumps(out,sort_keys=True,indent=2)+'\n').encode()
    if len(raw)>4*1024*1024: raise Hold('producer output exceeds reserved 4MiB')
    sys.stdout.buffer.write(raw)
if __name__=='__main__':
    try: main()
    except Exception as e:
        sys.stdout.write(json.dumps({'status':'HOLD','stage':STAGE,'error_type':type(e).__name__,'reason':str(e),'peak_rational_component_bits':PEAK,'no_unchanged_retry':True})+'\n')
        sys.exit(2)
