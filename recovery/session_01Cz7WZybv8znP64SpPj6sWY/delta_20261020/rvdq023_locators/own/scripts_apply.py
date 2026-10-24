import sys, hashlib
# Same rule as declared in each export's FREEZE POLICY / OP-PROT-017 N1-N8 (re-implemented, not imported)
def extract(raw, begin, end):
    t = raw.decode("utf-8-sig").replace("\r\n","\n").replace("\r","\n")
    L = t.split("\n")
    b=[i for i,l in enumerate(L) if l==begin]; e=[i for i,l in enumerate(L) if l==end]
    assert len(b)==1 and len(e)==1 and e[0]>b[0], (b,e)
    return ("\n".join(L[b[0]+1:e[0]]).rstrip("\n")+"\n").encode("utf-8"), b[0]+1, e[0]+1
src, begin, end, out = sys.argv[1:5]
raw=open(src,"rb").read()
body,bl,el=extract(raw,begin,end)
open(out,"wb").write(body)
print("export_bytes",len(raw),"export_sha256",hashlib.sha256(raw).hexdigest())
print("begin_line",bl,"end_line",el)
print("body_bytes",len(body),"body_sha256",hashlib.sha256(body).hexdigest())
print("body_first_line",body.split(b"\n")[0].decode())
print("body_lines(LF count)",body.count(b"\n"))
