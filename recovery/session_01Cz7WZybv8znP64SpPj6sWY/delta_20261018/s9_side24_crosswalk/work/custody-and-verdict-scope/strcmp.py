import re,sys
D=sys.argv[1]
def txt(n):
    r=open(D+'/'+n+'.txt','rb').read().decode('utf-8-sig').replace('\r\n','\n')
    return r
a=txt('CL-AUD-LSDER062-20260728-01'); b=txt('LS-DER-063-v1.0'); c=txt('LS-DER-061-v1.0')
nums=lambda t: re.findall(r'-?\d\.\d{40,}e-\d+', t)
A=nums(a); B=nums(b); C=nums(c)
print('CL-AUD-062 long numbers:', [ (x[:12], len(x.split('e')[0])-3) for x in A])
print('LS-DER-063 long numbers:', [ (x[:12], len(x.split('e')[0])-3) for x in B])
print('LS-DER-061 long numbers:', [ (x[:12], len(x.split('e')[0])-3) for x in C])
r2=A[5]; d1=B[0]
print('CL-AUD-062 R2 == LS-DER-063 (1):', r2==d1)
r1_12=A[2]; print('R1(12) is prefix of 063(1):', d1.split('e')[0].startswith(r1_12.split('e')[0]), r1_12.split('e')[1]==d1.split('e')[1])
r1_13=A[3]; d2=B[1]; print('R1(13) is prefix of 063(2):', d2.split('e')[0].startswith(r1_13.split('e')[0]))
r1_17=A[4]; d3=B[2]; print('R1(17) is prefix of 063(3):', d3.split('e')[0].startswith(r1_17.split('e')[0]), r1_17[:20], d3[:20])
# where 061 (12) diverges from 063 (1)
x=C[0].split('e')[0]; y=d1.split('e')[0]
k=next(i for i in range(min(len(x),len(y))) if x[i]!=y[i]); print('061(12) vs 063(1) first differing char index', k, '-> significant digit', k-2)
