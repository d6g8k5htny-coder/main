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

artifact = 13738689802
expected_zip = "dbb1db8d458c349bfd00d68ca3994ee344023700f6eb6aa438e3aaa300b4060d"
expected_report = "aec7cdf79dc79000b8a5fcf5e638f0334fc51f661ed802fa82836d77e3f0ff60"
expected_commit = "709975c6ff42f0bca020abd4164a45479e29a0df"
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
                report.get("files") != 1505 or report.get("tracked") != 6026 or
                report.get("skipped") != []):
            raise ValueError("report content mismatch")
        if any(bad in data for bad in (b"github_pat_", b"ghp_", b"Authorization: Bearer")):
            raise ValueError("unexpected credential marker")
print("CENSUS_ZIP_VERIFIED " + json.dumps({"artifact": artifact, "zip_sha256": expected_zip,
      "zip_bytes": len(raw), "reports": 2, "report_sha256": expected_report,
      "commit": expected_commit, "status": "PASS", "failing_tools": expected_tools}, sort_keys=True))
print("CENSUS_ZIP_BASE64=" + base64.b64encode(raw).decode("ascii"))
