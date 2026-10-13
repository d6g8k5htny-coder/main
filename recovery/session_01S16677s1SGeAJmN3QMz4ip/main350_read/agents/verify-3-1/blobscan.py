import subprocess, hashlib, sys
GIT="/home/user/main/.git"
targets={"001c4f609e167fcf139b5f46611bacdbf7203e4191060953d34485f78e2fc7c6":"outer696",
         "24e6f4149c0a2df2f8df1dbfa8d66da3e01a7d12bee4b803bac093cfea36bf3f":"quartic695",
         "d4c2f0bc923d95a8a1905f4745dd4c2cad6025c1f1dcda99441e9f5285f5a213":"core620",
         "33ba8362069346058ba33bbd8c1bd1feed62b749a96b6cbbc8836ddcbc6e4f2e":"quarticPROOF751",
         "3d952a906553d578876456539a8f0dcd64aedd45099e8394cbe9db99866ec531":"stPROOF804"}
out=subprocess.run(["git","--git-dir",GIT,"cat-file","--batch-all-objects","--batch-check","--unordered"],capture_output=True,text=True,check=True).stdout
blobs=[l.split()[0] for l in out.splitlines() if l.split()[1]=="blob"]
print("total objects",len(out.splitlines()),"blobs",len(blobs))
p=subprocess.Popen(["git","--git-dir",GIT,"cat-file","--batch"],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
p.stdin.write(("\n".join(blobs)+"\n").encode()); p.stdin.close()
found=0
for oid in blobs:
    hdr=p.stdout.readline().decode().split()
    n=int(hdr[2]); data=p.stdout.read(n); p.stdout.read(1)
    h=hashlib.sha256(data).hexdigest()
    if h in targets:
        found+=1; print("FOUND",targets[h],oid,"bytes",n,"lines",data.count(b"\n"))
    elif n in (32809,) :
        print("size-32809 blob (other sha)",oid,h[:16])
print("matches found:",found)
