import re,importlib.util,sys
sys.argv=['x']
T='<SCRATCH>/repos/math_c264ctl/companions/torus_filtration_20261011/test_torus_filtration.py'
src=open(T,encoding='utf-8').read()
ns={}
m=re.search(r'^DISCRIMINATOR_LOGS = \{.*?^\}', src, re.M|re.S)
exec(m.group(0), ns)
D=ns['DISCRIMINATOR_LOGS']
base='<SCRATCH>/lvs/c264_controls_integ/precommit/lean/'
for k,v in D.items():
    real=open(base+'discrim-'+k+'.log',encoding='utf-8').read()
    real2=re.sub(r'^\S*/(Reject\w+\.lean)', r'/tmp/evidence/\1', real, flags=re.M)
    print(k, real2==v)
