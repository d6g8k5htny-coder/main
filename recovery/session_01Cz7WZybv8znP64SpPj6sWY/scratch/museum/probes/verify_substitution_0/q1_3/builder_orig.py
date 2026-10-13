"""Build the museum display projection from immutable public source bytes.

This is a selected view of PROOF_INDEX and STATUS, not a scientific register or
another source inventory. --check is read-only. Math input defaults to fixed raw
commit URLs; --math-root uses pinned Git objects or verified plain file fixtures.
The historical public_shop_data.py exporter is independent and unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

MAIN_COMMIT = "71400b94f6cb354a8cf7aba73ffede2138a64efa"
MATH_COMMIT = "d6628da09384728992dcbe6e921cc28ba85aebb0"
CANVAS_NOTICE = "This canvas explains the pinned source. It is not a proof and does not change status."
# Selected source identities audited through the public GitHub connector.
# The 2,138-artifact inventory remains docs/public-math/sources.json.
SOURCES = json.loads(r'''{
  "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",
    "blob": "247b3ecf80bfbe896948d5d489b2d5842a81c481",
    "bytes": 17734,
    "sha256": "380b7d0abdb0fe2de5a6564565af9560d3f1b1ce22fd5538927db0c52f4f3a4a"
  },
  "coefficients/side24_v1/PROOF.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "coefficients/side24_v1/PROOF.md",
    "blob": "44b66f04f89fcd87383b3603fa69f1feb64cdddd",
    "bytes": 10272,
    "sha256": "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769"
  },
  "frontiers/remote_window_20260924/PROOF.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/remote_window_20260924/PROOF.md",
    "blob": "b383bfcc88ec4ad497dff01fb6640e429ba24a84",
    "bytes": 18355,
    "sha256": "a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7"
  },
  "frontiers/rn_annulus_bridge_20260925/PROOF.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/rn_annulus_bridge_20260925/PROOF.md",
    "blob": "6f317515b3d417661f86e2fed09bc7d950899c2b",
    "bytes": 16948,
    "sha256": "d55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05"
  },
  "frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md",
    "blob": "081abc13c5e66342c13df2af2bd6b114f320486f",
    "bytes": 17646,
    "sha256": "1fd9fe7141e464fd1c09ebf0318d8a701729c9c61ea24f73f7e20a9cf10c552b"
  },
  "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "blob": "89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73",
    "bytes": 15902,
    "sha256": "079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d"
  },
  "frontiers/axial_density_20260925/PROOF.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/axial_density_20260925/PROOF.md",
    "blob": "a72418d78fd3c1ec96260dbf19f9a2f3f4ca35ba",
    "bytes": 10858,
    "sha256": "8b9376a69fda9f0f231d9502be839ae45d8006d1891ff3e700fa521cc96f6cb4"
  },
  "reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md",
    "blob": "024d927779f79fadaf932541ea6474f2995f8c50",
    "bytes": 9902,
    "sha256": "f64c954245f17bcbc64a5ccf58a60689cf2a2fd85362f8321c0f64ff66584e36"
  },
  "reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md",
    "blob": "044ac5fdaf403a38e33983e31f0ad69f8e76d6d5",
    "bytes": 3272,
    "sha256": "83f653393dc6245980f6848e2bfc65ac9ad4c7d8304b028b88764356fd3825b6"
  },
  "frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md",
    "blob": "1d6946658e00e2446d68e8a4884425d4bcec62fb",
    "bytes": 2266,
    "sha256": "498af650ef7139c39d2630561dbfd0822c495ce6308a6cac8b767aefe0c2a40b"
  },
  "frontiers/full_price_20260924/PROOF.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "frontiers/full_price_20260924/PROOF.md",
    "blob": "582180e41dca0ad815ad0f18574df42040912149",
    "bytes": 11352,
    "sha256": "87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9"
  },
  "PROOF_INDEX.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "PROOF_INDEX.md",
    "blob": "e4cb1d943eaeb2049d080fcc53a57e042aebc6e8",
    "bytes": 12708,
    "sha256": "f9c4db77aeab1c61d2a0c02f88a964cbfc6be63d18619acf0e047c884e0c0249"
  },
  "claims/LANDING_CLAIMS.json": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "claims/LANDING_CLAIMS.json",
    "blob": "1e178c2d948bab4de696c6b66724dd7a37fb34f1",
    "bytes": 16417,
    "sha256": "c6fb2fa2bda1fdae9eeaaaf13008497921537b77aa4d4537aec30c9b961b989e"
  },
  "imports/hardening_ebedb780/EC-014/proof.md.export.txt": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "imports/hardening_ebedb780/EC-014/proof.md.export.txt",
    "blob": "7189b4d1538866594c6d8c94f09e247a2cd65abe",
    "bytes": 11331,
    "sha256": "c5e64b9ba3a906535f1ae58a161d594a427371599c102ae4a68849b087a2f62c"
  },
  "coefficients/side24_v1/ENCLOSURE.json": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "coefficients/side24_v1/ENCLOSURE.json",
    "blob": "57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7",
    "bytes": 1090,
    "sha256": "72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811"
  },
  "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md",
    "blob": "559d72242fdb7e4e3ac4a10d10641ae1da84f40e",
    "bytes": 15925,
    "sha256": "69ea7a479a9e8a5150cbe07b373772ab22413a1d81194d5a6a78adf6ada5f03f"
  },
  "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md",
    "blob": "1f153966dae48d10574db0d272f75d08ca3a8b23",
    "bytes": 12741,
    "sha256": "29a0c6d00ce739baf1d29a3163977d8e707e2079e216fba4fee3669cebec127d"
  },
  "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",
    "blob": "8988251631cab8e65eb2b01a08f7fc593b018570",
    "bytes": 10073,
    "sha256": "04cf4c986f29ac7ac33b3cc87feac23f3cf494e0c1b3bf4476e4e6666de46574"
  },
  "reviews/replacement_20260925_pr19_pr21/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/replacement_20260925_pr19_pr21/REVIEW.md",
    "blob": "f1868bd92b600d92901146af1caf7baf960dba9c",
    "bytes": 15615,
    "sha256": "7ff9e4a028f24afc731200b14bc5b993c0a874906cf6ea25e8c890afed8cdc53"
  },
  "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md",
    "blob": "4003ff13cf696389b9530fd445ee46daf3b4c646",
    "bytes": 12510,
    "sha256": "20d6451412f304a20f70e0bc6aabb87bdd0bb9761b17834469d60867407ebd9c"
  },
  "reviews/d2_cumulative_correction_20260925/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/d2_cumulative_correction_20260925/REVIEW.md",
    "blob": "3ab51b0112bbcd29febc433881f459240645056d",
    "bytes": 3723,
    "sha256": "910d371ebb4a310a7e2248348ba8cd09d6f3229740525d74ec7d06d8d1470717"
  },
  "reviews/p15_full_price_nonauthor_20260926/REVIEW.md": {
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "path": "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",
    "blob": "07db19f826763b8faa7414ba2e403d9c45bc0b6e",
    "bytes": 15569,
    "sha256": "691ea0db5c0f073326c9f7208c9804c6f5d0daf92ba7467529c9d57e08b15eb7"
  },
  "STATUS.md": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "STATUS.md",
    "blob": "52688102260e916dec1a38143186198b089a0261",
    "bytes": 6881,
    "sha256": "9c3e21144423591a3dbff926858048c0363b9a32d6fe258a1de5088d38096141"
  },
  "reviews/sard_g_successor_a1_a6_20260926/REVIEW.md": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "reviews/sard_g_successor_a1_a6_20260926/REVIEW.md",
    "blob": "9fc61b47c2b4c4bcc3898a6c61a6dcf9cbe1d136",
    "bytes": 25057,
    "sha256": "894d27c0de6dd4312ea5109ac2480a68cb1b5ded0b4a60749d4edee2288fb537"
  },
  "incoming/side24-identity-replay-20260926/IDENTITY.json": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "incoming/side24-identity-replay-20260926/IDENTITY.json",
    "blob": "a1c211e58f72af9c5206ed6731b6771a7366e301",
    "bytes": 673,
    "sha256": "6087bfe1ccc12bd52ebb169d00975ce44e846d611f0628333a0dbd71a4448176"
  },
  "incoming/side24-identity-replay-20260926/output.json": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "incoming/side24-identity-replay-20260926/output.json",
    "blob": "f1dd2c6ba5a74d9f52fe91d66dd3d136e7d2166e",
    "bytes": 362,
    "sha256": "813bcf5d81e1d5d52d37c112df23716701809f03e7b55939a2afb2e8e57741be"
  },
  "docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb",
    "blob": "c7fcde5461305ff249384e873476d34a4f470976",
    "bytes": 61961,
    "sha256": "e348db8e9d667caff2a8da1d85ffd5ad599db34b34e542d9662d0bc1edc84024"
  },
  "incoming/side24-identity-replay-20260926/RESULT.md": {
    "repository": "d6g8k5htny-coder/main",
    "commit": "71400b94f6cb354a8cf7aba73ffede2138a64efa",
    "path": "incoming/side24-identity-replay-20260926/RESULT.md",
    "blob": "132cc35d978f46133f28a669ed6e35d1ddf2e7bb",
    "bytes": 1118,
    "sha256": "fa40a6b8afc413646081145d6d5d7b01cc4b5b4ddec32f4c6e82caa399e38cd8"
  },
  "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md": {
    "path": "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md",
    "blob": "dfed3b8d318a3ab1950957f393307733a4bef3f2",
    "bytes": 40261,
    "repository": "d6g8k5htny-coder/Math-",
    "commit": "d6628da09384728992dcbe6e921cc28ba85aebb0",
    "sha256": "9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7"
  }
}''')
SOURCES["README.md"] = {"repository":"d6g8k5htny-coder/Math-","commit":"d6628da09384728992dcbe6e921cc28ba85aebb0","path":"README.md","blob":"abe16871da7740fa28ae4dbb28fb31853a5868bd","bytes":6023,"sha256":"89d4c618ad4560373f8fba3c787522cea8f0191fd8e176f3e0741c6230390730"}
INDEX = SOURCES["PROOF_INDEX.md"]
STATUS = SOURCES["STATUS.md"]
NOTEBOOK_PATH = "docs/notebooks/SIDE24_PUBLIC_COEFFICIENTS.ipynb"
CLAIM_SPECS = json.loads(r'''[
  [
    "d2-lifetime-remainder",
    "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",
    "https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206",
    "D2",
    "python -B -S -m unittest discover -s frontiers/three_fronts_20260924 -p 'test_*.py' -v"
  ],
  [
    "d3-side24-coefficient",
    "coefficients/side24_v1/PROOF.md",
    "https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490",
    "D3",
    null
  ],
  [
    "d4-fixed-remote-rn",
    "frontiers/remote_window_20260924/PROOF.md",
    "https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172",
    "D4",
    "python -B -S -m unittest discover -s frontiers/remote_window_20260924 -p 'test_*.py' -v"
  ],
  [
    "d5-all-height-annulus",
    "frontiers/rn_annulus_bridge_20260925/PROOF.md",
    "reviews/pr28_annulus_bridge_nonauthor_20260925/REVIEW.md",
    null,
    "python3 -B -S reviews/pr28_annulus_bridge_nonauthor_20260925/algebra_check.py -v"
  ],
  [
    "d5-height-window-annulus",
    "frontiers/rn_thin_tube_20260925/FIXED_ANNULUS_CANDIDATE.md",
    "reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md",
    null,
    "python3 -B -S reviews/pr22_fixed_annulus_nonauthor_20260925/algebra_check.py"
  ],
  [
    "d5-two-scale",
    "frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md",
    "reviews/replacement_20260925_pr19_pr21/TWO_SCALE_REVIEW.md",
    null,
    null
  ],
  [
    "d5-inner-belt-density",
    "frontiers/axial_density_20260925/PROOF.md",
    "reviews/replacement_20260925_pr19_pr21/REVIEW.md",
    null,
    null
  ],
  [
    "d5-fixed-transverse",
    "reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md",
    "reviews/pr16_fixed_transverse_nonauthor_20260925/REVIEW.md",
    null,
    "python3 -B -S reviews/pr16_fixed_transverse_nonauthor_20260925/algebra_check.py"
  ],
  [
    "cumulative-transfer-correction",
    "reviews/collision_mechanism_20260925/CUMULATIVE_TRANSFER_CORRECTION.md",
    "reviews/d2_cumulative_correction_20260925/REVIEW.md",
    null,
    null
  ],
  [
    "p15-demand-one-counterexample",
    "frontiers/three_fronts_20260924/P15_PRICE_BOUNDARY.md",
    "https://github.com/d6g8k5htny-coder/main/issues/67",
    null,
    null
  ],
  [
    "d6-p15-full-price",
    "frontiers/full_price_20260924/PROOF.md",
    "reviews/p15_full_price_nonauthor_20260926/REVIEW.md",
    "D6",
    "python -B -S frontiers/full_price_20260924/full_price.py"
  ]
]''')


def verify_bytes(raw, expected):
    """Reject invalid descriptors and changed bytes before any parsing."""
    if (not isinstance(expected, dict)
            or expected.get("repository") not in ("d6g8k5htny-coder/main", "d6g8k5htny-coder/Math-")
            or not re.fullmatch(r"[0-9a-f]{40}", expected.get("commit", ""))
            or not re.fullmatch(r"[0-9a-f]{40}", expected.get("blob", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", ""))
            or type(expected.get("bytes")) is not int or expected["bytes"] <= 0
            or not isinstance(expected.get("path"), str)
            or expected["path"].startswith("/") or ".." in expected["path"].split("/")):
        raise ValueError("invalid source identity")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if (len(raw) != expected["bytes"] or blob != expected["blob"]
            or hashlib.sha256(raw).hexdigest() != expected["sha256"]):
        raise ValueError("source identity mismatch: " + expected["path"])
    return raw


def descriptor(path, **extra):
    identity = dict(SOURCES[path])
    base = identity["repository"] + "/" + identity["commit"] + "/" + quote(path, safe="/")
    identity["url"] = "https://raw.githubusercontent.com/" + base
    identity["html_url"] = ("https://github.com/" + identity["repository"]
                            + "/blob/" + identity["commit"] + "/" + quote(path, safe="/"))
    return dict(identity, **extra)


def pointer(path, target):
    return descriptor(path, pointer_only=True, review_url=target,
        review_notice="The pinned source binds this review pointer; the linked review is not byte-frozen by this descriptor.")


def section(text, heading):
    marker = "## " + heading + "\n"
    if text.count(marker) != 1:
        raise ValueError("missing or duplicated projection heading: " + heading)
    return text.split(marker, 1)[1].split("\n## ", 1)[0]


def status_rows(text, heading):
    rows = [line for line in section(text, heading).splitlines() if line.startswith("|")]
    return [(line, [cell.strip() for cell in line[1:-1].split("|")])
            for line in rows[2:]]


def project(index_raw, status_raw):
    """Return the exact eleven reviewed bullets and three open table rows."""
    index = verify_bytes(index_raw, INDEX).decode("utf-8")
    status = verify_bytes(status_raw, STATUS).decode("utf-8")
    bullets = [line for line in section(index, "Reviewed scoped results").splitlines()
               if line.startswith("- ")]
    if len(bullets) != 11:
        raise ValueError("reviewed projection must have eleven bullets")
    accepted_rows = status_rows(status, "ACCEPT — scoped")
    claims = []
    for bullet, (claim_id, proof_path, review_path, status_id, command) in zip(bullets, CLAIM_SPECS):
        title = bullet[2:].split(":", 1)[0]
        if "](" + proof_path + ")" not in bullet:
            raise ValueError("proof pointer differs from audited projection")
        if "](" + review_path + ")" not in bullet:
            raise ValueError("review pointer differs from audited projection")
        review = (pointer("PROOF_INDEX.md", review_path) if review_path.startswith("https://")
                  else descriptor(review_path))
        counterexample = claim_id == "p15-demand-one-counterexample"
        status_quote = next((cells[1] for _, cells in accepted_rows
                             if status_id and cells[0].startswith("**" + status_id + " —")), None)
        replay = {"command": command, "url": None,
                  "notice": ("Finite algebra or implementation replay only; it is not theorem acceptance."
                             if command else "No replay command is recorded in the pinned proof/review.")}
        if command:
            replay_source = "README.md" if status_id in ("D2", "D4", "D6") else review_path
            replay.update(source=descriptor(replay_source), checkout=MATH_COMMIT,
                          working_directory="Math- repository root",
                          notice="From a Math- checkout at " + MATH_COMMIT
                          + ", run at the repository root. Finite algebra or implementation replay only; it is not theorem acceptance.")
        if claim_id == "d3-side24-coefficient":
            replay = {"command": None, "url": descriptor(NOTEBOOK_PATH)["html_url"],
                      "source": descriptor(NOTEBOOK_PATH),
                      "notice": "The notebook verifies and displays published coefficient bytes; no random field or lifetime solver runs."}
        claims.append(dict(id=claim_id, title=title, scope_quote=bullet,
            source_label="EXACT_COUNTEREXAMPLE" if counterexample else "ACCEPT",
            **{"class": "engineering-only" if counterexample else "ACCEPT-scoped"},
            proof=descriptor(proof_path), review=review, replay=replay, status_quote=status_quote))
    open_rows = status_rows(status, "AMEND / open")
    if len(open_rows) != 3:
        raise ValueError("open projection must have three rows")
    open_specs = [
        ("d1-parent-selection-open", "imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md",
         "https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743"),
        ("d5-pin-neighborhoods-open", "PROOF_INDEX.md",
         "https://github.com/d6g8k5htny-coder/Math-/blob/4e188e25b1e1ef560f3eeb75c0d354d2ccf0ea22/reviews/d5_pin_neighborhood_20260926/REVIEW.md"),
        ("sard-g-a1-a6-open", "STATUS.md", "reviews/sard_g_successor_a1_a6_20260926/REVIEW.md")
    ]
    for (row, cells), (claim_id, proof_path, review_target) in zip(open_rows, open_specs):
        proof = descriptor(proof_path)
        if claim_id == "d5-pin-neighborhoods-open":
            proof["availability"] = "NO COMPLETE PROOF YET"
        elif claim_id == "sard-g-a1-a6-open":
            proof.update(availability="OPEN / NOT LANDED",
                         proof_url="https://github.com/d6g8k5htny-coder/main/pull/122")
        review = (pointer("STATUS.md" if claim_id.startswith("d1-") else "PROOF_INDEX.md", review_target)
                  if review_target.startswith("https://") else descriptor(review_target))
        claims.append(dict(id=claim_id, title=cells[0].replace("**", ""), scope_quote=row,
            source_label="AMEND / open", **{"class": "AMEND/open"},
            proof=proof, review=review,
            replay={"command": None, "url": None, "notice": "OPEN / NOT LANDED. No replay is presented as acceptance."},
            status_quote=cells[1]))
    return claims


def packet_descriptors():
    """Only the packet already present on the pinned main tree is selected."""
    prefix = "incoming/side24-identity-replay-20260926/"
    return [dict(id="side24-identity-replay-20260926", issue=None,
        result=descriptor(prefix + "RESULT.md"), identity=descriptor(prefix + "IDENTITY.json"),
        output=descriptor(prefix + "output.json"),
        scientific_effect="NONE", review_status="REVIEW_REQUIRED")]


def read_source(identity, main_root, math_root):
    root = main_root if identity["repository"].endswith("/main") else math_root
    if root is not None:
        # Regenerate historical displays from the exact object even when the
        # repository's working file has since changed. Plain fixture folders
        # have no Git object and remain subject to the same byte checks below.
        result = subprocess.run(["git", "show", identity["commit"] + ":" + identity["path"]],
            cwd=root, capture_output=True)
        if result.returncode == 0:
            raw = result.stdout
        else:
            raw = (Path(root) / identity["path"]).read_bytes()
    else:
        # No branch alias, latest-release endpoint, registry discovery, or code execution.
        with urlopen(descriptor(identity["path"])["url"], timeout=30) as response:
            raw = response.read(identity["bytes"] + 1)
    return verify_bytes(raw, identity)


def excerpt(text, start, end):
    if text.count(start) != 1 or end not in text.split(start, 1)[1]:
        raise ValueError("source excerpt boundary changed")
    return start + text.split(start, 1)[1].split(end, 1)[0]


def build(main_root, math_root=None):
    """Verify all selected sources and generate the read-only display payload."""
    identities = list(SOURCES.values())
    with ThreadPoolExecutor(max_workers=8) as pool:
        raw_files = dict(zip(SOURCES, pool.map(
            lambda identity: read_source(identity, main_root, math_root), identities)))
    text = {path: raw.decode("utf-8") for path, raw in raw_files.items()}
    claims = project(raw_files["PROOF_INDEX.md"], raw_files["STATUS.md"])
    exhibit_paths = {
        "ec014": "imports/hardening_ebedb780/EC-014/proof.md.export.txt",
        "remote": "frontiers/remote_window_20260924/PROOF.md",
        "annulus": "frontiers/rn_annulus_bridge_20260925/PROOF.md",
        "p15": "frontiers/full_price_20260924/PROOF.md",
        "lifetime": "frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md",
    }
    exhibits = {key: descriptor(path, id=key, **{"class": "illustration"},
                               notice=CANVAS_NOTICE) for key, path in exhibit_paths.items()}
    exhibits["lifetime"].update(
        **{"class": "engineering-only"}, availability="NO COMMITTED SAMPLES",
        sample_available=False,
        unavailable_notice="No committed persistence/lifetime sample JSON was found on the pinned main or Math default trees. The theorem source is linked for scope; no synthetic data is substituted.")
    exhibits["annulus"]["scope_boundary"] = (
        "The reviewed fixed annulus and the AMEND pin-neighborhood row are distinct scopes.")
    normalized_ec = text[exhibit_paths["ec014"]].replace("\r\n", "\n")
    quotes = {
        "ec014": excerpt(normalized_ec, "2. Exact object proposed for closure", "3. Evidence package"),
        "remote": excerpt(text[exhibit_paths["remote"]], "## 1. The count and the new scope", "## 2."),
        "annulus": excerpt(text[exhibit_paths["annulus"]], "## 1. Statement and exact scope", "## 2."),
        "p15": excerpt(text[exhibit_paths["p15"]], "## 5. Sharpness and the exact demand boundary", "## 6."),
    }
    identity = json.loads(raw_files["incoming/side24-identity-replay-20260926/IDENTITY.json"])
    output = json.loads(raw_files["incoming/side24-identity-replay-20260926/output.json"])
    if (identity.get("scientific_effect") != "NONE" or identity.get("review_status") != "REVIEW_REQUIRED"
            or output.get("scientific_acceptance") is not False):
        raise ValueError("landed packet authority fields changed")
    return dict(schema_version=1, scientific_status_authority=False,
        index_source=descriptor("PROOF_INDEX.md"), status_source=descriptor("STATUS.md"),
        claims=claims, exhibits=exhibits, exhibit_quotes=quotes, packets=packet_descriptors(),
        inventory_url="../public-math/sources.json",
        meaning="Selected display projection only; the existing 2,138-row public source catalog remains the inventory.",
        canvas_notice=CANVAS_NOTICE)


def dump(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def check_export(path, expected):
    if not Path(path).exists() or Path(path).read_bytes() != dump(expected):
        raise ValueError("generated museum export differs: " + str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--math-root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    data = build(args.main_root, args.math_root)
    destination = args.main_root / "docs/site/museum.json"
    if args.check:
        check_export(destination, data)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(dump(data))
    print("Museum projection verified: 11 reviewed bullets, 3 AMEND rows, 1 landed packet; scientific effect NONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
