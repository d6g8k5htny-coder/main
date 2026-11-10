"""Independently verify and retain the G11 live-provenance artifact."""
import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile

artifact = 13738108831
expected = "dd7a9ea89dd4762e8c978324af0cece67bce2f48ae4fe9d6cdbcf52eefd81a0a"

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

url = "https://api.github.com/repos/d6g8k5htny-coder/main/actions/artifacts/%d/zip" % artifact
request = urllib.request.Request(url, headers={
    "Authorization": "Bearer " + os.environ["GH_TOKEN"],
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "cbz-g11-provenance-verification",
})
try:
    urllib.request.build_opener(NoRedirect).open(request, timeout=30)
    raise RuntimeError("expected artifact redirect")
except urllib.error.HTTPError as exc:
    if exc.code not in (301, 302, 303, 307, 308):
        raise
    location = exc.headers.get("Location")
parsed = urllib.parse.urlparse(location or "")
host = parsed.hostname or ""
if parsed.scheme != "https" or not (host.endswith(".blob.core.windows.net") or
                                    host.endswith(".actions.githubusercontent.com") or
                                    host.endswith(".githubusercontent.com")):
    raise ValueError("unexpected artifact redirect host")
with urllib.request.urlopen(location, timeout=60) as response:
    raw = response.read(2000000)
if hashlib.sha256(raw).hexdigest() != expected or len(raw) != 67548:
    raise ValueError("artifact digest or length mismatch")
summary, order = [], []
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    wanted = ["g11-mutations-normal.json", "g11-mutations-optimized.json"]
    if len(names) != 2 or len(set(names)) != 2 or sorted(names) != sorted(wanted):
        raise ValueError("unexpected artifact member population")
    if archive.testzip() is not None:
        raise ValueError("artifact CRC failure")
    for name in wanted:
        data = archive.read(name)
        if len(data) > 5000000 or any(bad in data for bad in
                (b"github_pat_", b"ghp_", b"Authorization: Bearer")):
            raise ValueError("unexpected report size or credential marker")
        report = json.loads(data)
        base = report.get("baseline", {})
        records = report.get("mutations", [])
        if (report.get("status") != "PASS" or report.get("executed") != 328 or
                report.get("total") != 328 or report.get("killed") != 328 or
                len(records) != 328 or base.get("status") != "OK" or
                base.get("ran") != 263 or base.get("exit") != 0 or
                base.get("failures") != 0 or base.get("errors") != 0 or
                base.get("skipped") != 0):
            raise ValueError("incomplete baseline or mutation report")
        labels = []
        for record in records:
            event = record.get("origin_provenance", {})
            frames = event.get("frames", [])
            selected = [i for i, frame in enumerate(frames) if frame.get("selected") is True]
            if (record.get("killed") is not True or record.get("exit") != 1 or
                    record.get("failures") != 1 or record.get("errors") != 0 or
                    record.get("skipped") != 0 or record.get("traceback_line") != record.get("expected_line") or
                    record.get("test_id") != record.get("expected_test") or
                    record.get("origin_authorized") is not True or event.get("authorized") is not True or
                    event.get("entered") != 1 or event.get("phase") not in ("body", "body-exited") or
                    len(selected) != 1 or any(frame.get("oracle") is not True for frame in frames[selected[0] + 1:])):
                raise ValueError("invalid selected-body provenance")
            labels.append(record["mutation"])
        if len(set(labels)) != 328:
            raise ValueError("duplicate mutation labels")
        order.append(labels)
        summary.append({"member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                        "controls": 263, "mutations": 328})
if order[0] != order[1]:
    raise ValueError("mode mutation inventory mismatch")
print("PROVENANCE_ZIP_VERIFIED " + json.dumps({"artifact": artifact, "sha256": expected,
      "bytes": len(raw), "reports": summary, "validated_origin_records": 656}, sort_keys=True))
encoded = base64.b64encode(raw).decode("ascii")
chunks = [encoded[index:index + 6000] for index in range(0, len(encoded), 6000)]
for index, chunk in enumerate(chunks):
    print("PROVENANCE_ZIP_CHUNK %03d/%03d %s" % (index, len(chunks), chunk))
