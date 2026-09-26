"""Reproduce read-only public-shop exports from exact public source bytes.

Run with --main-root for the pinned main checkout and optionally --math-root
for an offline Math checkout. Otherwise only the three pinned Math resources
are fetched. --check compares exports without writing. No scientific register
is changed; snapshot table cells remain verbatim Markdown.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

MAIN_COMMIT = "a26f744be7597e3f0c0543c34ead23049bbac657"
MATH_COMMIT = "9d7b6802424fb4715b31999066aafca8ee2f3cca"
OWNER = "d6g8k5htny-coder"


def source(repo, commit, path, size, blob, digest):
    return dict(repository=OWNER + "/" + repo, commit=commit, path=path,
                bytes=size, blob=blob, sha256=digest)


STATUS = source("main", MAIN_COMMIT, "STATUS.md", 6881,
    "52688102260e916dec1a38143186198b089a0261",
    "9c3e21144423591a3dbff926858048c0363b9a32d6fe258a1de5088d38096141")
COEFFICIENT = source("Math-", MATH_COMMIT, "coefficients/side24_v1/ENCLOSURE.json", 1090,
    "57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7",
    "72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811")
PROOF = source("Math-", MATH_COMMIT, "coefficients/side24_v1/PROOF.md", 10272,
    "44b66f04f89fcd87383b3603fa69f1feb64cdddd",
    "c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769")
INVENTORY = source("main", MAIN_COMMIT, "docs/public-math/sources.json", 1863,
    "edf9a7088a574134fa982946c9e779476b617e3d",
    "78ac59ed59579a1ee7cd0912f52ad702249b6978e836e5bccd90577a02591a52")
IMPORTS = source("Math-", MATH_COMMIT, "imports/hardening_ebedb780/MANIFEST.json", 18262,
    "4cc679d65d8af2162fa9915ad7edf35fda16a576",
    "fa7f33839924d340cc1123fe05276f681e6aa12193c68b2d1aa6ab489150d8e3")
SHARDS = [
  [
    "docs/public-math/sources-01.json",
    "550519fe77b73927169e2a1efdc0276bb0f6141f",
    58276
  ],
  [
    "docs/public-math/sources-02.json",
    "0037f84f28f3e9dada4d921c903ee0271723d544",
    86078
  ],
  [
    "docs/public-math/sources-03.json",
    "9b1eb04eb09fc3227d394110e1913e85306392ce",
    96034
  ],
  [
    "docs/public-math/sources-04.json",
    "b4c66773898213e7b58483b018fd15063911af93",
    87348
  ],
  [
    "docs/public-math/sources-05.json",
    "5247d6927cee10e49d9b9d8f378797ebae29c926",
    68164
  ],
  [
    "docs/public-math/sources-06.json",
    "eda5411fe6e3035e9e981a3f614aac46882828b9",
    81840
  ],
  [
    "docs/public-math/sources-07.json",
    "0cd78768bb71eef7a84ea9fe35867a5b6d103ea3",
    95161
  ],
  [
    "docs/public-math/sources-08.json",
    "882219d447d249c3221a526ea0f59dd2d69a108f",
    87801
  ],
  [
    "docs/public-math/sources-09.json",
    "a59e56e0351a77b624058be9efad3f30d495d1e1",
    76228
  ],
  [
    "docs/public-math/sources-10.json",
    "addd0228282e1ec31228dca1476f2d770f206181",
    76144
  ],
  [
    "docs/public-math/sources-11.json",
    "b50b28a293503b35fa0851be9f8854f5fe4aed55",
    86339
  ],
  [
    "docs/public-math/sources-12.json",
    "950783d361a676c61e96f9299a845d9e2b9a2c66",
    63991
  ],
  [
    "docs/public-math/sources-13.json",
    "9d4ab75e0423e456294ee90399a3c22648153fa5",
    69414
  ],
  [
    "docs/public-math/sources-14.json",
    "f9df9ee169376c01f7f3a7b1f794e79ded81915b",
    20295
  ]
]
COEFFICIENT_EXPECTED = json.loads(r'''{
  "cone_moments": {
    "2": "4/3",
    "3": "29/6-sqrt(6)"
  },
  "dimensions": {
    "2": {
      "lower": "0.07340691930603427103",
      "upper": "0.07340691930603427104"
    },
    "3": {
      "lower": "0.04177593184059834334",
      "upper": "0.04177593184059834335"
    }
  },
  "image_ledger": {
    "covariance_relative_bound": "31763607879717/2500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "density_comparison_below_reported": true,
    "derivative_bound": "10587869293239/50000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
    "exp_lower_gt_10": true,
    "image_constant": 21175738586478,
    "relative_below_eps": true
  },
  "method": "outward rational arithmetic; analytic image and Stirling remainder bounds",
  "object": "SIDE24-COEFFICIENT-D23-20260924-v1",
  "relative_periodization_bound": "1e-106",
  "scientific_acceptance": false,
  "scope": "coefficient of issue63 Eq15.2; parent theorem unreviewed"
}
''')
TABLES = {
    "ACCEPT — scoped": ("accept", ["Object", "Accepted scope", "Source and review", "Explicit limits"]),
    "AMEND / open": ("amend", ["Object", "Current reason", "Source"]),
    "Engineering only": ("engineering", ["Surface", "What it provides"]),
}
OTHER_HEADINGS = {"Where to read — public source custody", "Reading rule"}
EXPECTED_COUNTS = {"accept": 4, "amend": 3, "engineering": 3}
IMPORT_IDS = ["EC-014", "EC-015", "EC-021", "P02-LM-001", "P02-LM-002",
              "P02-LM-005", "P02-LM-007", "P02-LM-008", "P15-B"]


def blob_sha(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def identity_for(raw, template):
    return dict(template, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                blob=blob_sha(raw))


def verify_bytes(raw, expected):
    if any(identity_for(raw, expected)[key] != expected[key]
           for key in ("bytes", "sha256", "blob")):
        raise ValueError("source identity mismatch: " + expected["path"])
    return raw


def with_url(identity):
    result = dict(identity)
    result["url"] = ("https://raw.githubusercontent.com/" + identity["repository"]
                     + "/" + identity["commit"] + "/" + quote(identity["path"], safe="/"))
    return result


def parse_status(raw, identity=STATUS):
    verify_bytes(raw, identity)
    text = raw.decode("utf-8")
    date = re.search(r"^\*\*Snapshot date:\*\* (\d{4}-\d{2}-\d{2})\s*$", text, re.M)
    if not date:
        raise ValueError("missing snapshot date")
    sections = []
    current = None
    stage = None
    seen = set()
    for line in text.splitlines():
        if line.startswith("## "):
            title = line[3:]
            if title in seen:
                raise ValueError("duplicate heading: " + title)
            seen.add(title)
            if title not in TABLES and title not in OTHER_HEADINGS:
                raise ValueError("unknown heading: " + title)
            if title in TABLES:
                key, headers = TABLES[title]
                current = dict(key=key, title=title, headers=headers, rows=[])
                sections.append(current)
                stage = "header"
            else:
                current, stage = None, None
        elif line.startswith("### ") and line != "### Default-branch boundary":
            raise ValueError("unknown heading: " + line[4:])
        elif line.startswith("|"):
            if current is None:
                raise ValueError("table outside classified section")
            if not line.endswith("|"):
                raise ValueError("malformed table row")
            cells = [cell.strip() for cell in line[1:-1].split("|")]
            if stage == "header":
                if cells != current["headers"]:
                    raise ValueError("table header schema changed")
                stage = "separator"
            elif stage == "separator":
                if len(cells) != len(current["headers"]) or not all(
                        re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
                    raise ValueError("table separator schema changed")
                stage = "rows"
            else:
                if len(cells) != len(current["headers"]) or not all(cells):
                    raise ValueError("table row width/content changed")
                current["rows"].append(cells)
    if seen != set(TABLES) | OTHER_HEADINGS:
        raise ValueError("missing required heading")
    counts = {section["key"]: len(section["rows"]) for section in sections}
    if counts != EXPECTED_COUNTS:
        raise ValueError("snapshot row counts changed; explicit source review required")
    for section in sections:
        section["count"] = counts[section["key"]]
    boundary = text.split("### Default-branch boundary\n", 1)[1].split("\n## Reading rule", 1)[0].strip()
    return dict(schema_version=1, scientific_status_authority=False,
        source=with_url(identity), snapshot_date=date.group(1),
        meaning="Counts of selected rows in the pinned STATUS snapshot, not totals of all theorems or an independent acceptance register.",
        sections=sections, counts=counts,
        snapshot_context=dict(default_branch_boundary=boundary,
            notice="This historical snapshot's custody/query boundary may be stale. Current observations are separate; scientific rows are not inferred from merges."))


def parse_coefficient(raw, identity=COEFFICIENT):
    verify_bytes(raw, identity)
    data = json.loads(raw)
    if data != COEFFICIENT_EXPECTED or data.get("scientific_acceptance") is not False:
        raise ValueError("coefficient schema or values changed")
    for bounds in data["dimensions"].values():
        if any(type(value) is not str for value in bounds.values()) or not (
                Decimal(bounds["lower"]) < Decimal(bounds["upper"])):
            raise ValueError("coefficient schema requires exact decimal strings")
    return data


def inventory_config(read_main):
    raw = read_main(INVENTORY["path"])
    verify_bytes(raw, INVENTORY)
    data = json.loads(raw)
    if data["source_count"] != 2138 or len(data["pages"]) != len(SHARDS):
        raise ValueError("inventory schema/count changed")
    pages = []
    for (path, blob, size), expected in zip(SHARDS, data["pages"]):
        if "docs/public-math/" + expected["path"] != path:
            raise ValueError("inventory shard order/path changed")
        raw_page = read_main(path)
        if len(raw_page) != size or blob_sha(raw_page) != blob:
            raise ValueError("shard identity mismatch: " + path)
        page = json.loads(raw_page)
        if set(page) != {"sources"} or len(page["sources"]) != expected["count"]:
            raise ValueError("inventory shard schema/count changed")
        for row in page["sources"]:
            if set(row) != {"repository", "path", "commit", "blob", "bytes", "sha256"}:
                raise ValueError("inventory row schema changed")
            if row["repository"] not in {"main", "Math-"} or not all(
                    re.fullmatch("[0-9a-f]{" + str(n) + "}", row[key])
                    for key, n in (("commit", 40), ("blob", 40), ("sha256", 64))):
                raise ValueError("inventory row identity changed")
        identity = identity_for(raw_page, dict(INVENTORY, path=path))
        pages.append(dict(with_url(identity), count=expected["count"]))
    if sum(page["count"] for page in pages) != data["source_count"]:
        raise ValueError("inventory total mismatch")
    return with_url(INVENTORY), pages


def build(read_main, read_math, observations=None):
    status = parse_status(read_main(STATUS["path"]))
    parse_coefficient(read_math(COEFFICIENT["path"]))
    verify_bytes(read_math(PROOF["path"]), PROOF)
    manifest = json.loads(verify_bytes(read_math(IMPORTS["path"]), IMPORTS))
    copies = manifest["byte_copies"]
    if (manifest["scientific_effect"] != "NONE"
            or [row["id"] for row in copies] != IMPORT_IDS
            or any(row["kind"] != "BYTE_COPY" or row["source_label_adopted"] is not False for row in copies)):
        raise ValueError("import custody schema changed")
    inventory, pages = inventory_config(read_main)
    inventory.update(source_count=2138, pages=pages,
        relative_url="../public-math/sources.json",
        meaning="Existing public-text inventory; loaded in place, never copied into a second catalog.")
    imports = dict(with_url(IMPORTS), count=len(copies), byte_copy_ids=IMPORT_IDS,
        manifest_url=with_url(IMPORTS)["url"], scientific_effect="NONE",
        transcription_count=len(manifest["transcriptions"]),
        meaning="Nine byte-custody imports, separately one transcription; not accepted-theorem counts.")
    config = dict(schema_version=1, scientific_status_authority=False,
        coefficient=with_url(COEFFICIENT), proof=with_url(PROOF),
        status=with_url(STATUS), inventory=inventory, imports=imports)
    status_bytes = dump(status)
    config["status_json"] = dict(url="status.json", bytes=len(status_bytes),
        sha256=hashlib.sha256(status_bytes).hexdigest())
    if observations:
        if (observations.get("scientific_status_authority") is not False
                or observations["open_math_prs"] != len(observations["pull_requests"])
                or not re.fullmatch("[0-9a-f]{40}", observations["math_tip"])):
            raise ValueError("observation schema/count changed")
        config["observations"] = {key: observations[key]
            for key in ("observed_at", "math_tip", "open_math_prs")}
        config["observations"]["path"] = "observations.json"
    return status, config


def dump(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--math-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = args.output or args.main_root / "docs/site"
    math_sources = {item["path"]: item for item in (COEFFICIENT, PROOF, IMPORTS)}
    def read_main(path):
        # Use the pinned Git object when present, so a later unrelated branch
        # movement cannot silently substitute different public source bytes.
        result = subprocess.run(["git", "show", MAIN_COMMIT + ":" + path],
            cwd=args.main_root, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        return (args.main_root / path).read_bytes()
    def read_math(path):
        if args.math_root:
            return (args.math_root / path).read_bytes()
        identity = math_sources[path]
        with urlopen(with_url(identity)["url"], timeout=30) as response:
            return response.read(identity["bytes"] + 1)
    observation_path = output / "observations.json"
    observations = json.loads(observation_path.read_bytes()) if observation_path.exists() else None
    status, config = build(read_main, read_math, observations)
    # Query pin refresh is maintained independently. Preserve only an explicit
    # well-formed optional identity block supplied by that reviewed integration.
    existing_config = output / "config.json"
    if existing_config.exists():
        query = json.loads(existing_config.read_bytes()).get("query")
        if query is not None:
            if not all(re.fullmatch("[0-9a-f]{40}", query.get(k, "")) for k in ("commit", "math_pin")):
                raise ValueError("invalid query identity block")
            config["query"] = query
    for name, data in (("status.json", status), ("config.json", config)):
        destination = output / name
        raw = dump(data)
        if args.check:
            if not destination.exists() or destination.read_bytes() != raw:
                raise ValueError("generated export differs: " + str(destination))
        else:
            output.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
    print("Public shop exports verified; scientific effect NONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
