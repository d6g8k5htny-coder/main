"""Check the retained original Pages artifact against the exact prepared receipt.

This is a bounded offline observation, not a public-readback or recovery check.
"""
import hashlib
import json
import pathlib
import shutil
import sys
import tarfile
import tempfile
import zipfile


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main(receipt_path, pages_path, metadata_path):
    receipt_raw = pathlib.Path(receipt_path).read_bytes()
    git_sha = hashlib.sha1(
        b"blob " + str(len(receipt_raw)).encode() + b"\0" + receipt_raw
    ).hexdigest()
    require(git_sha == "8f30c3b91f0f55a356ab0c05ee4a96f151c7f420",
            "prepared receipt Git blob differs")
    receipt = json.loads(receipt_raw)
    require(receipt["phase"] == "prepared", "receipt is not prepared")
    require(receipt["source"]["commit"] == "6d928a3216233fb75dc717ee0b2d2e4bb08db266",
            "prepared source differs")
    require(receipt["run"] == {"id": 42737966287, "attempt": 1},
            "prepared run differs")

    pages_raw = pathlib.Path(pages_path).read_bytes()
    metadata_raw = pathlib.Path(metadata_path).read_bytes()
    require(len(pages_raw) == 36126909 and sha(pages_raw) ==
            "3f5daa7b15454230f70dac32e23bef9f9c7a4d311bf2ecca04209328806b0e67",
            "original Pages ZIP differs")
    require(len(metadata_raw) == 723214 and sha(metadata_raw) ==
            "3676ea54b9b946d69ee561f43ef4818fca7a4dc0734eb6182c85dc43ea28af0f",
            "original metadata ZIP differs")

    with zipfile.ZipFile(pages_path) as pages, zipfile.ZipFile(metadata_path) as meta:
        require(pages.namelist() == ["artifact.tar"] and pages.testzip() is None,
                "Pages ZIP population or CRC differs")
        expected = {"candidate.json", "current-meta.json", "current.json",
                    "snapshots/index.json"}
        names = meta.namelist()
        require(len(names) == len(set(names)) == len(expected) and set(names) == expected
                and meta.testzip() is None, "metadata ZIP population or CRC differs")
        candidate = json.loads(meta.read("candidate.json"))
        require(all(receipt.get(k) == v for k, v in candidate.items())
                and set(candidate) == set(receipt) -
                {"phase", "previous_receipt_commit", "recorded_at", "recorded_by"},
                "candidate differs from prepared receipt")
        controls = {"metadata": ("current-meta.json", "site/current-meta.json"),
                    "payload": ("current.json", "site/current.json"),
                    "manifest": ("snapshots/index.json", "site/snapshots/index.json")}
        for field, (member, _) in controls.items():
            data = meta.read(member)
            require(len(data) == receipt[field]["bytes"]
                    and sha(data) == receipt[field]["sha256"],
                    "metadata member differs from receipt: " + field)

        with tempfile.TemporaryFile() as tmp:
            with pages.open("artifact.tar") as src:
                shutil.copyfileobj(src, tmp, 1024 * 1024)
            tmp.seek(0)
            with tarfile.open(fileobj=tmp, mode="r:") as tar:
                files, seen = [], set()
                for member in tar.getmembers():
                    require(member.isfile() or member.isdir(), "nonregular tar entry")
                    raw = member.name
                    name = raw[2:] if raw.startswith("./") else raw
                    if name in ("", "."):
                        continue
                    path = pathlib.PurePosixPath(name)
                    require(not raw.startswith("/") and all(x not in ("", "..")
                            for x in path.parts) and str(path) == name,
                            "unsafe tar path")
                    require(name not in seen, "duplicate tar path")
                    seen.add(name)
                    if member.isfile():
                        files.append((name, member))
                files.sort(key=lambda x: x[0])
                digest = hashlib.sha256(b"current-site-generated-browser-input-v1\0")
                total, observed_controls = 0, {}
                target_names = {path for _, path in controls.values()}
                for name, member in files:
                    name_bytes = name.encode()
                    total += member.size
                    digest.update(len(name_bytes).to_bytes(8, "big"))
                    digest.update(name_bytes)
                    digest.update(member.size.to_bytes(8, "big"))
                    source = tar.extractfile(member)
                    require(source is not None, "unreadable tar file")
                    count, content = 0, bytearray() if name in target_names else None
                    while chunk := source.read(1024 * 1024):
                        count += len(chunk)
                        digest.update(chunk)
                        if content is not None:
                            content.extend(chunk)
                    require(count == member.size, "tar file length differs")
                    if content is not None:
                        observed_controls[name] = bytes(content)
                result = {"sha256": digest.hexdigest(), "files": len(files), "bytes": total}
                require(result == receipt["artifact"], "prepared fingerprint differs")
                for _, (member, path) in controls.items():
                    require(observed_controls.get(path) == meta.read(member),
                            "tar and metadata control bytes differ")
    return {"prepared_head": "6e01d1f5f83d320172e29391eea336e3256aeea2",
            "receipt_git_blob": git_sha, "pages_artifact": 13728153243,
            "metadata_artifact": 13728342519, "artifact": result,
            "scope": "offline original bytes; no public-readback or recovery claim"}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: verify_prepared_artifact.py RECEIPT PAGES_ZIP METADATA_ZIP")
    print(json.dumps(main(*sys.argv[1:]), sort_keys=True))
