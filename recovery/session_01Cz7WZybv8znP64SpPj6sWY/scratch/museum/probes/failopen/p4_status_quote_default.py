"""F: museum_data.project() resolves the STATUS ACCEPT-row quote for D2/D3/D4/D6 with
next((...), None): a missing row silently yields status_quote None (the card stays class ACCEPT-scoped, the
class being taken from PROOF_INDEX bullet membership), and a duplicated row silently yields the first.
The browser validator demands exactly one row and refuses. Here a successor STATUS (D4 ACCEPT row withdrawn, or
D4 row duplicated with a narrower scope placed first) is pinned in a scratch import; the pin is updated
exactly as a maintainer repinning SOURCES would. No network; cd32303 bytes are read, never edited.
usage: python3 -I p4_status_quote_default.py <cd32303 tree> <fixture dir>"""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
CD, FIX = Path(sys.argv[1]), Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("museum_data", CD / "tools/museum_data.py")
md = importlib.util.module_from_spec(spec); spec.loader.exec_module(md)
index = (FIX / "Math-/PROOF_INDEX.md").read_bytes(); status = (FIX / "main/STATUS.md").read_bytes()
lines = status.decode().split("\n"); d4 = next(l for l in lines if l.startswith("| **D4 —"))
variants = {
  "control (pinned STATUS)": status,
  "D4 ACCEPT row withdrawn": "\n".join(l for l in lines if l != d4).encode(),
  "D4 row duplicated, narrower first": "\n".join((l.replace("Fixed positive spatial separation", "WITHDRAWN PENDING RE-REVIEW; fixed positive spatial separation") + "\n" + l) if l == d4 else l for l in lines).encode(),
}
here = Path(__file__).resolve().parent
for name, raw in variants.items():
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    md.STATUS = dict(md.SOURCES["STATUS.md"], bytes=len(raw), blob=blob, sha256=hashlib.sha256(raw).hexdigest())
    try:
        claims = md.project(index, raw); d4c = claims[2]
        py = f"project() OK -> {d4c['id']} class={d4c['class']} status_quote={json.dumps((d4c['status_quote'] or None) and d4c['status_quote'][:60])}"
        # browser verdict on the same projection, through validateBoundManifest with matching descriptors
        man = json.loads((CD / "docs/site/museum.json").read_text())
        man["claims"] = claims; man["status_source"] = dict(md.descriptor("STATUS.md"), **{k: md.STATUS[k] for k in ("bytes", "blob", "sha256")})
        for c in man["claims"]:
            if c["id"] == "sard-g-a1-a6-open": c["proof"].update({k: md.STATUS[k] for k in ("bytes", "blob", "sha256")})
            if c["review"] and c["review"].get("pointer_only") and c["review"]["path"] == "STATUS.md": c["review"].update({k: md.STATUS[k] for k in ("bytes", "blob", "sha256")})
        for p in man["packets"]: pass
        (here / "work").mkdir(exist_ok=True); (here / "work/p4_manifest.json").write_text(json.dumps(man)); (here / "work/p4_index.md").write_bytes(index); (here / "work/p4_status.md").write_bytes(raw)
        js = ("import fs from 'node:fs';import {pathToFileURL} from 'node:url';const m=await import(pathToFileURL(process.argv[1]+'/docs/site/museum.mjs'));"
              "try{m.validateBoundManifest(JSON.parse(fs.readFileSync(process.argv[2],'utf8')),fs.readFileSync(process.argv[3],'utf8'),fs.readFileSync(process.argv[4],'utf8'));console.log('PASS')}catch(e){console.log('REFUSED: '+e.message)}")
        r = subprocess.run(["/opt/node22/bin/node", "--input-type=module", "-e", js, str(CD), str(here / "work/p4_manifest.json"), str(here / "work/p4_index.md"), str(here / "work/p4_status.md")], capture_output=True, text=True)
        br = r.stdout.strip() or r.stderr.strip()[-200:]
    except Exception as e:
        py, br = f"project() RAISED {type(e).__name__}: {e}", "-"
    print(f"== {name}\n   python : {py}\n   browser validateBoundManifest: {br}")
