import json, sys, zipfile, hashlib, io
work = sys.argv[1]
pk = zipfile.ZipFile(work + "/pack_55e00c96.zip")
raw = pk.read("1e11b2462c69ea4fe5c73b04abd36d90dd2ee7b5181bf7579c67f50482e9374e")
z = zipfile.ZipFile(io.BytesIO(raw))
infos = z.infolist()
files = [i for i in infos if not i.is_dir()]
t = "Prize_Research_P14_Fractional_20260917/proofs/04_LIMITATIONS_AND_HIERARCHY.md"
print("dirs", len(infos) - len(files), "files", len(files))
print("index among files (0-based):", [i for i, x in enumerate(files) if x.filename == t])
srt = sorted(x.filename for x in infos)
print("index in sorted names (0-based):", srt.index(t))
srtf = sorted(x.filename for x in files)
print("index in sorted files (0-based):", srtf.index(t))
inf = z.getinfo(t); print("crc %08x compress_size %d file_size %d" % (inf.CRC, inf.compress_size, inf.file_size))
