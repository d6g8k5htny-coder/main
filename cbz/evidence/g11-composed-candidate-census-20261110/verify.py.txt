"""Verify the retained original composed-tree G11 census artifact."""
import base64
import hashlib
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import zipfile

artifact = 13739226619
expected_zip = "d62c42e871159133e789f7693f1f04dcf0cfc787e8284cad21b334208ef542ae"
expected_report = "32982e3eaaf007f05263ae01032065189b42408a972212d8d2e0338d722d5ace"
expected_commit = "ef2903b4028ca369223ff2128cc5f81ca6749ee0"
expected_tools = []

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

url = "https://api.github.com/repos/d6g8k5htny-coder/main/actions/artifacts/%d/zip" % artifact
request = urllib.request.Request(url, headers={
    "Authorization": "Bearer " + os.environ["GH_TOKEN"],
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "cbz-g11-artifact-verification",
})
try:
    urllib.request.build_opener(NoRedirect).open(request, timeout=30)
    raise RuntimeError("expected an artifact redirect")
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
if hashlib.sha256(raw).hexdigest() != expected_zip or len(raw) != 10256:
    raise ValueError("artifact bytes disagree with observed digest or length")
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    wanted = ["g11-census-normal.json", "g11-census-optimized.json"]
    if len(names) != 2 or sorted(names) != sorted(wanted) or len(set(names)) != 2:
        raise ValueError("unexpected artifact member population")
    if archive.testzip() is not None:
        raise ValueError("artifact CRC failure")
    for name in wanted:
        data = archive.read(name)
        if len(data) != 48658 or hashlib.sha256(data).hexdigest() != expected_report:
            raise ValueError("report hash mismatch")
        report = json.loads(data)
        if (report.get("commit") != expected_commit or report.get("status") != "PASS" or
                report.get("failing") != [] or report.get("failing_tools") != expected_tools or
                report.get("files") != 1491 or report.get("tracked") != 5970 or
                report.get("skipped") != []):
            raise ValueError("report content mismatch")
        if any(bad in data for bad in (b"github_pat_", b"ghp_", b"Authorization: Bearer")):
            raise ValueError("unexpected credential marker")
print("CENSUS_ZIP_VERIFIED " + json.dumps({"artifact": artifact, "zip_sha256": expected_zip,
      "zip_bytes": len(raw), "reports": 2, "report_sha256": expected_report,
      "commit": expected_commit, "status": "PASS", "failing_tools": expected_tools}, sort_keys=True))
print("CENSUS_ZIP_BASE64=" + base64.b64encode(raw).decode("ascii"))
