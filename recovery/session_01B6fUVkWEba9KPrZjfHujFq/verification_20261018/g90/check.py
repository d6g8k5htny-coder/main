import json, sys, zipfile, hashlib, io
work = sys.argv[1]
d = json.load(open(work + "/objects.json"))
cid = "1e11b2462c69ea4fe5c73b04abd36d90dd2ee7b5181bf7579c67f50482e9374e"
ch = d["chunks"][cid]
print("chunk record keys:", sorted(ch.keys()) if isinstance(ch, dict) else type(ch))
pk = zipfile.ZipFile(work + "/pack_55e00c96.zip")
names = [n for n in pk.namelist() if cid in n]
print("pack members matching chunk:", names)
raw = pk.read(names[0])
print("chunk bytes", len(raw), "sha256", hashlib.sha256(raw).hexdigest())
z = zipfile.ZipFile(io.BytesIO(raw))
infos = z.infolist()
print("custody zip members:", len(infos))
target = "a90833637b7180094a63b03740424dcf10551182b83da82411a7480cc5b2863c"
for i, inf in enumerate(infos):
    b = z.read(inf)
    if hashlib.sha256(b).hexdigest() == target:
        print("MATCH index0=%d index1=%d size=%d name_tail=%s" % (i, i + 1, len(b), inf.filename.split("/")[-1]))
        print("full member path:", inf.filename)
