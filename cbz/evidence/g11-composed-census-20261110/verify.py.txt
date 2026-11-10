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

artifact = 13737684031
expected_zip = "99ab43607660da2132055acd3ee1f8694c0184bce10444722397a49f90a31cdf"
expected_report = "6a48e545d508299a158a99a5826e7ecda5d45e7ac4e54102b9c4a9f33106d3ce"
expected_commit = "0f7412c5933a97dfa15d84667d176bcc60a130e6"
expected_tools = ["tools/cbz/g11_origin_mutation_selftest.py",
                  "tools/cbz_site_publisher_cas_mutations.py",
                  "tools/cbz_site_recovery_mutations.py"]

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
if hashlib.sha256(raw).hexdigest() != expected_zip or len(raw) != 10188:
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
        if len(data) != 48104 or hashlib.sha256(data).hexdigest() != expected_report:
            raise ValueError("report hash mismatch")
        report = json.loads(data)
        if (report.get("commit") != expected_commit or report.get("status") != "FAIL" or
                report.get("failing") != [] or report.get("failing_tools") != expected_tools or
                report.get("files") != 1473 or report.get("tracked") != 5912 or
                report.get("skipped") != []):
            raise ValueError("report content mismatch")
        if any(bad in data for bad in (b"github_pat_", b"ghp_", b"Authorization: Bearer")):
            raise ValueError("unexpected credential marker")
print("CENSUS_ZIP_VERIFIED " + json.dumps({"artifact": artifact, "zip_sha256": expected_zip,
      "zip_bytes": len(raw), "reports": 2, "report_sha256": expected_report,
      "commit": expected_commit, "status": "FAIL", "failing_tools": expected_tools}, sort_keys=True))
print("CENSUS_ZIP_BASE64=" + base64.b64encode(raw).decode("ascii"))
