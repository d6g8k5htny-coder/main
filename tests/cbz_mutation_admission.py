"""Exact, source-bound admission of the fixed CBZ auditor mutation controls.

Profiles are reviewed expectations for the preserved mutants, not learned at
test time. Matching the complete report preserves every test/subtest, assertion
site, mismatch, skip and suite count. Formatting/source drift fails closed.
"""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "cbz/evidence/a8-admission-repair-20261101/profiles.json"


def normalize_report(text, tool_path=None):
    """Erase only execution locations and terminal elapsed time."""
    if tool_path is not None:
        text = text.replace(str(tool_path), "<MUTANT>")
    text = text.replace(str(ROOT), "<WORKSPACE>")
    aliases = {}

    def temporary(match):
        path = match.group(0)
        if path not in aliases:
            aliases[path] = "<TMP%d>" % len(aliases)
        return aliases[path]

    text = re.sub(r"/tmp/tmp[A-Za-z0-9_-]+", temporary, text)
    return re.sub(r"^Ran ([1-9][0-9]*) (tests?) in [0-9]+(?:\.[0-9]+)?s$",
                  r"Ran \1 \2 in <ELAPSED>s", text, flags=re.M)


def _valid_profile(profile, required_returncode):
    if not isinstance(profile, dict):
        return False
    text = profile.get("stderr")
    if not isinstance(text, str):
        return False
    runs = re.findall(r"^Ran ([1-9][0-9]*) tests? in <ELAPSED>s$", text, re.M)
    if len(runs) != 1 or int(runs[0]) != profile.get("tests"):
        return False
    if re.search(r"^ERROR:|^ERROR$|^Traceback.*ImportError", text, re.M):
        return False
    headers = re.findall(r"^FAIL: ", text, re.M)
    errors = re.findall(r"^AssertionError(?::|$)", text, re.M)
    failures = profile.get("failures")
    skipped = profile.get("skipped")
    if type(failures) is not int or type(skipped) is not int or skipped < 0:
        return False
    if len(headers) != failures or len(errors) != failures:
        return False
    if len(re.findall(r"^Traceback \(most recent call last\):$", text, re.M)) != failures:
        return False
    suffix = ", skipped=%d" % skipped if skipped else ""
    if required_returncode == 1:
        if failures < 1:
            return False
        summary = "FAILED (failures=%d%s)\n" % (failures, suffix)
    elif required_returncode == 0:
        if failures != 0:
            return False
        summary = "OK%s\n" % (" (skipped=%d)" % skipped if skipped else "")
    else:
        return False
    summaries = re.findall(r"^(?:FAILED|OK).*$", text, re.M)
    return len(summaries) == 1 and text.endswith("\n" + summary) and summaries[0] + "\n" == summary


def matches_report(result, profile, *, required_returncode, tool_path=None):
    """No partial/name-only/error outcomes count as a behavioral kill."""
    if not _valid_profile(profile, required_returncode):
        return False
    if result.returncode != required_returncode:
        return False
    if result.stdout != "" or not isinstance(result.stderr, str):
        return False
    return normalize_report(result.stderr, tool_path) == profile["stderr"]


def profile_for(label):
    """Load only frozen expectations, checking every consumed source blob."""
    data = json.loads(PROFILES.read_text(encoding="utf-8"))
    if data.get("schema") != "cbz-a8-admission-profiles-v1":
        raise ValueError("unsupported A8 profile schema")
    for path, expected in data["source_blobs"].items():
        raw = (ROOT / path).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if actual != expected:
            raise ValueError("A8 profile source mismatch: " + path)
    return data["profiles"][label]
