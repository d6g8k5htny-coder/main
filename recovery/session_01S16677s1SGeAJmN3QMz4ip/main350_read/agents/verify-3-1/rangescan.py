import hashlib,sys
p="/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/wt350/experiments/universality/two_parameter_cusp_population/PROOF.md"
raw=open(p,"rb").read(); lines=raw.split(b"\n")
assert raw.endswith(b"\n"); lines=lines[:-1]
L=len(lines); print("lines",L,"bytes",len(raw))
sizes=[len(l)+1 for l in lines]
pref=[0]
for s in sizes: pref.append(pref[-1]+s)
hits696=[]; hits32809=[]; hitsd4=[]; hits001=[]
for i in range(L):
    for j in range(i+1,L+1):
        b=pref[j]-pref[i]
        if b>32809: break
        if b==32809: hits32809.append((i+1,j,j-i))
        if j-i==696: hits696.append((i+1,j,b))
        if b==28073 and j-i==620:
            h=hashlib.sha256(b"\n".join(lines[i:j])+b"\n").hexdigest()
            hitsd4.append((i+1,j,h[:12]))
print("ranges with 32809 bytes:",hits32809)
print("ranges with 696 lines:",hits696)
print("620-line/28073-byte ranges:",hitsd4)
# Section 12 body arithmetic
s12=b"\n".join(lines[728:782])+b"\n"  # lines 729-782
print("Section 12 body (729-782): lines",782-728,"bytes",len(s12))
print("core+ledger body: lines",620+(782-728),"bytes",28073+len(s12),"=> residual vs 696/32809:",696-620-(782-728),"lines",32809-28073-len(s12),"bytes")
