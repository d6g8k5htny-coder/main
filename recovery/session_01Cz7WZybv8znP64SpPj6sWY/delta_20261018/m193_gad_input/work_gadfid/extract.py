import hashlib,sys
base='<SCRATCH>/m193/'
for fn,off,ln,h,out in [('gad/c6242006932.md',1326,40231,'8d992e194f0a898e563c4c3d05b8079cde292f9e87d7ff51030ad12ef5faf319','gad_src.txt'),('gad/c6242020976.md',651,38035,'6fedc43ba34408f23f8934166b3898e2cd19175c57e61e37227c85f3ca5d4c70','gad_rev.txt')]:
    b=open(base+fn,'rb').read()[off:off+ln]
    d=hashlib.sha256(b).hexdigest()
    print(fn,len(b),d,d==h)
    open(base+'work_gadfid/'+out,'wb').write(b)
