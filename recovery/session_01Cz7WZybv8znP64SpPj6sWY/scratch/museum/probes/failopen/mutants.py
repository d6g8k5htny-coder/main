"""Guard-deletion / inversion mutants against the cd32303 bytes. Each mutant is applied to a fresh scratch copy
of the extracted tree (the extracted bytes themselves are never edited), then the museum CI steps of
.github/workflows/public-shop.yml are run offline:
  S1 python3 -B -m unittest discover -s tests -p test_museum_data.py
  S2 python3 -B tools/museum_check.py --math-root <byte-verified fixture>/Math-   (CI fetches pinned raw URLs; offline here)
  S3 node --test tests/test_museum_geometry.mjs
  S4 node --test tests/test_public_shop_frontend.mjs  (same workflow; only matters for core.mjs)
SURVIVED = every step exits 0 with the guard removed.  usage: python3 -I mutants.py [ID ...]"""
import os, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SRC = (HERE / "../../cd32303").resolve()
FIX = HERE / "fixture/Math-"
WORK = HERE / "work/mut"
M = "docs/site/museum.mjs"; G = "docs/site/geometry.mjs"; D = "tools/museum_data.py"; C = "tools/museum_check.py"; T = "tests/test_museum_frontend.mjs"
MUTANTS = [
 ("M01", M, "if(claim.title!==title)throw Error('Claim title differs from source projection');", "", "delete claim-title binding"),
 ("M02", M, "}else if(!sourceContainsPath(quote,claim.review))throw Error('Review file differs from the source quote');", "}", "delete byte-frozen review-file binding (cross-claim review substitution guard)"),
 ("M03", M, "if(isOpen&&claim.status_quote!==statusCells(quote)[1])throw Error('AMEND reason differs from source quote');", "", "delete AMEND reason binding"),
 ("M04", M, "if(!replay||!(replay.command===null||nonempty(replay.command))||!(replay.url===null||/^https:\\/\\/github\\.com\\/d6g8k5htny-coder\\/(main|Math-|query-)\\/blob\\/[0-9a-f]{40}\\//.test(replay.url))||!nonempty(replay.notice))throw Error('Invalid replay declaration');", "", "delete replay declaration validation"),
 ("M05", M, "if(replay.url!==null&&replay.url!==replay.source.html_url)throw Error('Replay source URL differs from its identity');", "", "delete replay URL/identity binding"),
 ("M06", M, "if(replay.command&&(!replay.source||replay.source.repository!==claim.proof.repository||replay.source.commit!==claim.proof.commit||replay.checkout!==claim.proof.commit))throw Error('Replay command lacks a matching pinned checkout/source');", "", "delete replay checkout binding"),
 ("M07", M, "!Array.isArray(manifest.packets)||manifest.packets.length>1", "!Array.isArray(manifest.packets)", "delete packet-count cap"),
 ("M08", M, "if(!firstSourceMatches(quote,claim.proof)||claim.proof.repository!==manifest.index_source.repository||claim.proof.commit!==manifest.index_source.commit)throw", "if(!firstSourceMatches(quote,claim.proof))throw", "delete proof repository/commit = index snapshot binding"),
 ("M09", M, "if(!firstSourceMatches(parent,claim.proof)||claim.proof.repository!==manifest.index_source.repository||claim.proof.commit!==manifest.index_source.commit)throw Error('Open D1 proof differs from its exact index pointer');", "", "delete D1 parent proof binding"),
 ("M10", M, "}else if(!sameIdentity(claim.proof,claim.id==='d5-pin-neighborhoods-open'?manifest.index_source:manifest.status_source))throw Error('Open claim source pointer identity differs');", "}", "delete open-claim source identity binding"),
 ("M11", M, "||claim.review.sha256!==pointerSource.sha256||claim.review.url!==pointerSource.url", "", "delete pointer-source identity binding"),
 ("M12", M, "||(claim.id==='d5-pin-neighborhoods-open'&&claim.review.review_url!==D5_OPEN_REVIEW)", "", "delete D5 open-review pin in validateBoundManifest"),
 ("M13", M, "if(source.pointer_only===true&&((!/^https:\\/\\/github\\.com\\/d6g8k5htny-coder\\/(main|Math-)\\/issues\\/\\d+(?:#issuecomment-\\d+)?$/.test(source.review_url)&&source.review_url!==D5_OPEN_REVIEW)||!nonempty(source.review_notice)))throw Error('Invalid review pointer');", "", "delete review-pointer URL shape + notice guard"),
 ("M14", M, "if(claim.id!==expectedIds[i])throw Error('Claim projection order or object identity mismatch');", "", "delete claim id/order binding"),
 ("M15", M, "if(reviewed.length!==11||open.length!==3)throw Error('Pinned source projection count mismatch');", "", "delete pinned-source row-count guard"),
 ("M16", M, "if(claim.review){validateDescriptor(claim.review);await verifiedBytes(claim.review,fetcher);}", "", "verifyClaim: skip review byte verification"),
 ("M17", M, "if(claim.replay.source){\n    validateDescriptor", "if(false){\n    validateDescriptor", "verifyClaim: skip replay source verification (control)"),
 ("M18", M, "if(!/Scientific effect:\\s*\\*\\*NONE\\*\\*/.test(text)||!/Review status:\\s*\\*\\*REVIEW_REQUIRED\\*\\*/.test(text))throw Error('Packet source declarations differ');", "", "renderPacket: delete source-declaration check"),
 ("M19", M, "if(!nonempty(quote)||!sourceText.replaceAll('\\r\\n','\\n').includes(quote))throw Error('Exhibit quote differs from the pinned source');", "", "exhibit view: delete quote-in-source check"),
 ("M20", M, "containers['museum-state'].textContent=failed?", "containers['museum-state'].textContent=false?", "banner always says 'displayed source bytes verified' even with refused views"),
 ("M21", M, "placeholders[i].replaceChildren(refusal(document,claim.title,error));return false;", "placeholders[i].replaceChildren(renderClaim(document,claim));return true;", "claim catch swallows verification failure and renders the card as verified"),
 ("M22", M, "await verifiedBytes(validateDescriptor(source),fetcher);", "validateDescriptor(source);", "lifetime card rendered without byte verification"),
 ("M23", M, "const text=decoder.decode(await verifiedBytes(packet.result,fetcher));", "const text='Scientific effect: **NONE** Review status: **REVIEW_REQUIRED**';", "packet card rendered without fetching/verifying RESULT.md"),
 ("M24", M, "const proofText=decoder.decode(await verifiedBytes(claim.proof,fetcher));", "const proofText='';", "verifyClaim: skip proof byte verification (control)"),
 ("M25", M, "claim.proof.availability?'Open pinned source contract ↗':'Open pinned full proof ↗'", "'Open pinned full proof ↗'", "open obligations labelled 'Open pinned full proof'"),
 ("M26", M, "if(claim.class==='AMEND/open')scope.append(element(document,'p','OPEN / NOT LANDED — the claimed closure remains open.'));", "", "drop OPEN / NOT LANDED line from AMEND cards"),
 ("M27", M, "if(claim.review.pointer_only)source.append(element(document,'p',claim.review.review_notice));", "", "drop 'not byte-frozen' pointer notice"),
 ("M28", M, "manifest.scientific_status_authority!==false", "false", "accept scientific_status_authority:true (control)"),
 ("M31", M, "validateBoundManifest(manifest,decoder.decode(index),decoder.decode(status));\n", "\n", "startMuseum never calls validateBoundManifest"),
 ("M32", M, "const [index,status]=await Promise.all([verifiedBytes(manifest.index_source,cachedFetch),verifiedBytes(manifest.status_source,cachedFetch)]);", "const [index,status]=await Promise.all([manifest.index_source,manifest.status_source].map(s=>cachedFetch(s.url).then(r=>r.arrayBuffer()).then(b=>new Uint8Array(b))));", "startMuseum uses index/status bytes without digest verification"),
 ("M33", M, "await verifyClaim(claim,cachedFetch);placeholders", "placeholders", "startMuseum renders claim cards without verifyClaim"),
 ("G02", G, " || new Set(selection).size !== selection.length", "", "p15Model: delete duplicate-selection guard"),
 ("G06", G, "if (globalThis.THREE?.REVISION !== THREE_CDN.version) reject(new Error('Unexpected Three.js revision'));\n      else resolve(globalThis.THREE);", "resolve(globalThis.THREE);", "loadThree: delete revision check"),
 ("G01", G, "if (!source || source.repository !== 'd6g8k5htny-coder/Math-' || source.path !== SOURCE_PATHS[kind] ||", "if (false &&", "mountGeometry: delete source identity check (control)"),
 ("P01", D, "        if \"](\" + proof_path + \")\" not in bullet:\n            raise ValueError(\"proof pointer differs from audited projection\")\n", "", "project: delete proof-pointer binding"),
 ("P02", D, "        if \"](\" + review_path + \")\" not in bullet:\n            raise ValueError(\"review pointer differs from audited projection\")\n", "", "project: delete review-pointer binding (control)"),
 ("P03", D, "    if len(bullets) != 11:\n        raise ValueError(\"reviewed projection must have eleven bullets\")\n", "", "project: delete eleven-bullet guard"),
 ("P04", D, "    if len(open_rows) != 3:\n        raise ValueError(\"open projection must have three rows\")\n", "", "project: delete three-AMEND-row guard"),
 ("P05", D, "    if (identity.get(\"scientific_effect\") != \"NONE\" or identity.get(\"review_status\") != \"REVIEW_REQUIRED\"\n            or output.get(\"scientific_acceptance\") is not False):\n        raise ValueError(\"landed packet authority fields changed\")\n", "", "build: delete landed-packet authority check"),
 ("P06", D, "    if text.count(start) != 1 or end not in text.split(start, 1)[1]:\n        raise ValueError(\"source excerpt boundary changed\")\n", "", "excerpt: delete boundary check"),
 ("P07", D, "            or expected.get(\"repository\") not in (\"d6g8k5htny-coder/main\", \"d6g8k5htny-coder/Math-\")\n", "", "verify_bytes: delete repository allow-list"),
 ("P08", D, "            or expected[\"path\"].startswith(\"/\") or \"..\" in expected[\"path\"].split(\"/\")):", "):", "verify_bytes: delete path traversal guard"),
 ("P09", D, "\n            or hashlib.sha256(raw).hexdigest() != expected[\"sha256\"]):", "):", "verify_bytes: delete SHA-256 comparison (git blob SHA-1 still compared)"),
 ("P10", D, "    if text.count(marker) != 1:\n", "    if False:\n", "section: delete missing/duplicate heading guard"),
 ("P11", D, "    if not Path(path).exists() or Path(path).read_bytes() != dump(expected):", "    if False:", "check_export: never differs (control)"),
 ("C01", C, "        museum_data.check_export(ROOT / \"docs/site/museum.json\", expected)\n", "", "museum_check: drop the export comparison"),
 ("C02", C, "cwd=ROOT, env=env, check=True)", "cwd=ROOT, env=env, check=False)", "museum_check: ignore node test failures"),
 ("C03", C, "env = dict(os.environ, MUSEUM_FIXTURE=str(manifest))", "env = dict(os.environ, MUSEUM_FIXTURE_PATH=str(manifest))", "museum_check: env var drift silently skips the pinned integration test"),
]
STEPS = [("S1", ["python3", "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_museum_data.py"]),
         ("S2", ["python3", "-B", "tools/museum_check.py", "--math-root", str(FIX)]),
         ("S3", ["node", "--test", "tests/test_museum_geometry.mjs"]),
         ("S4", ["node", "--test", "tests/test_public_shop_frontend.mjs"])]
def run(tree):
    env = dict(os.environ, PATH="/opt/node22/bin:" + os.environ["PATH"])
    res = {}
    for name, cmd in STEPS:
        p = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True, timeout=300)
        out = p.stdout + p.stderr
        skipped = [l for l in out.splitlines() if l.startswith("# skipped")]
        res[name] = (p.returncode, skipped[0] if skipped else "")
    return res
def main(ids):
    for mid, rel, old, new, desc in MUTANTS:
        if ids and mid not in ids: continue
        if WORK.exists(): shutil.rmtree(WORK)
        shutil.copytree(SRC, WORK, ignore=shutil.ignore_patterns("__pycache__"))
        f = WORK / rel; text = f.read_text(encoding="utf-8")
        n = text.count(old)
        if n != 1: print(f"{mid} BAD-MUTANT occurrences={n}: {desc}"); continue
        f.write_text(text.replace(old, new), encoding="utf-8")
        res = run(WORK)
        survived = all(rc == 0 for rc, _ in res.values())
        detail = " ".join(f"{k}={'ok' if rc == 0 else 'FAIL'}{(' [' + sk.strip() + ']') if sk and not sk.endswith(' 0') else ''}" for k, (rc, sk) in res.items())
        print(f"{mid} {'SURVIVED' if survived else 'killed  '} {detail} | {rel}: {desc}", flush=True)
    if WORK.exists(): shutil.rmtree(WORK)
if __name__ == "__main__":
    main(set(sys.argv[1:]))
