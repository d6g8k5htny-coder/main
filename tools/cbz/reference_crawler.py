#!/usr/bin/env python3
"""Federation reference crawler for CBZ-0: cross-repository reference identity and interface drift.

The crawler discovers the workspace topology at run time and checks every
cross-repository reference it can find. It never relies on a hardcoded list of
repositories: renaming, adding or pruning a repository changes what it crawls,
and the crawl continues.

Topology, read from the workspace repository at one revision:
  * pins.json (schema ulw.pins.v1): each pinned repository's name, url and pinned_sha;
  * workspace/repositories.json: registered repositories and canonical_inputs;
  * the repository registry that canonical_inputs.repository_registry names
    (registry.json in meta-framework today): its "repositories" entries.
The workspace itself is a node at the crawled revision. A repository that is
registered but not pinned is crawled at its default-branch tip, cloned from the URL its
registered owner/name gives. --at NAME=REV crawls one repository at a candidate
revision, such as a pull request head, in place of its pin, while the rest of the
federation stays at its pins. That is the merge gate: exit 1 means the candidate
records a broken reference or breaks a live reference into itself.

References, extracted from every file of every node at its crawled commit that is
valid UTF-8 text, whatever its extension:
  * JSON (.json and .ipynb documents, and each line of .jsonl and .ndjson files; JSON
    with // or /* */ comments and trailing commas is read too; an unterminated
    comment makes the file UNPARSEABLE): any object with a
    repository key (repository, repo or full_name) and a string "path" is a record.
    Its identities are a commit (COMMIT_KEYS: 7 to 40 hex digits, so an abbreviated
    commit resolves through Git), a Git blob id (BLOB_KEYS), a SHA-256 (SHA256_KEYS)
    and a byte count (BYTES_KEYS). Every recorded value is checked, not only the
    first, and each distinct commit is checked separately. A record with an identity
    but no commit is live, and so is one whose "ref" names a branch: its path must
    exist at the target's crawled commit. A record with no identity key at all (only a
    repository and a path, say a card that names the repository it concerns) makes no
    claim about a file and is not a reference. Repository aliases
    that disagree, and identity values that are not valid (a null counts as absent),
    are blocking findings. A byte_identical_at_<name> key with a 40-hex value claims
    the same identity at that further commit, and is checked there too. Such a claim
    needs a recorded blob, sha256 or byte count to compare, and an invalid value is a
    MALFORMED_IDENTITY;
  * text: GitHub blob, tree or raw URLs of a topology owner with a 40-hex commit.
    The URL path may contain balanced parentheses and is percent-decoded. In JSON,
    URLs are matched in the decoded string values, so JSON escapes such as \n or \/
    never become part of a path.
Owner and repository names match case-insensitively, as on GitHub. A binary file (one
that is not valid UTF-8) is counted per repository, not read. A file over
MAX_TEXT_BYTES is an UNREAD finding. Nothing is skipped without being counted.
An object with integer line_start and line_end is an excerpt reference: its sha256,
bytes and text, where present, are checked against exactly those lines of the file
(splitlines with line ends kept, as the site generators record them), not the whole file.
Files under a tests, test, testdata, fixtures or examples directory, and files whose
name starts with TEMPLATE, are synthetic data. They are skipped unless
--include-fixtures is given. The report lists every skipped and every binary path.

Findings. BLOCKING findings make the exit status 1:
  UNKNOWN_REPOSITORY    the reference names a repository outside the topology
  MISSING_COMMIT        the referenced commit cannot be found or fetched
  MISSING_PATH          the path does not exist at the referenced commit
  IDENTITY_MISMATCH     the recorded blob, sha256 or byte count differs
  LIVE_DRIFT            a reference without a commit no longer matches the target's
                        crawled commit
  UNREACHABLE_REPOSITORY  a topology repository has no mirror and cannot be cloned
  MISSING_PIN           a pinned commit, or an --at revision, cannot be found or fetched
  TARGET_UNREACHABLE    a reference into a topology repository that is unreachable
  ALIAS_CONFLICT        a record's repository fields name different repositories
  MALFORMED_IDENTITY    a record's commit, blob, sha256 or bytes value is not a valid value
  UNPARSEABLE           a JSON file, or a JSON Lines line, that does not parse, even as JSON
                        with comments: its records are unchecked, so the census is incomplete.
                        URLs are still matched in its raw text
  UNREAD                a file over MAX_TEXT_BYTES, which is not read, so the census is incomplete
  TOPOLOGY_INCOMPLETE   the workspace registry, its canonical repository_registry input, or the
                        registry document is missing, unreadable or of the wrong shape, so some
                        repositories may not have been discovered
The crawl never halts on one repository: an unreachable node is a finding, and the
rest of the federation is still crawled.
INFO findings are reported and do not block, unless a flag says otherwise:
  FROZEN_CHANGED        a commit-pinned reference whose path has since changed. For .lean
                        files it lists theorem-statement drift, for .py files AST signature
                        drift, and for .json files type-shape drift. --strict blocks on any
                        of those interface changes.
  FROZEN_REMOVED        a commit-pinned reference whose path no longer exists at the pin
  PIN_SKEW              a pinned commit differs from its default-branch tip; blocks only with
                        --require-current-pins
  TOPOLOGY_UNPINNED     registered but not pinned
  TOPOLOGY_UNREGISTERED pinned but not registered
  FOREIGN_REPOSITORY    a JSON reference to a repository whose owner is outside the topology
                        (a Lean dependency, for example). It is outside the federation and is
                        counted, not checked. A foreign repository that shares a short name with
                        a topology repository is never confused with it.
  JSON_WITH_COMMENTS    a JSON file that parses only once comments and trailing commas are
                        removed; it is read that way
  CANDIDATE             a repository crawled at an --at revision instead of its pin
Exit status 0 therefore means a complete census of the readable files with nothing
broken. Exit status 2 means the crawl itself failed: the workspace's pins are missing,
unreadable or of the wrong schema.

Domain (governance B-006). The checks are on Git objects and file text. Lean
statement drift compares comment-stripped, whitespace-normalized declaration text,
which is not elaboration. AST signatures cover top-level functions, classes and their
methods. JSON type shapes are compared to depth 6. An identity nested in a sub-object
(for example {"path": ..., "identity": {"blob": ...}}) is not attached to its parent.
A reference without a commit (a live reference) is judged at the target's crawled
commit. Free-text fields such as "meaning" or "adoption_state" are not interpreted, so
a record of a blob seen only on an unmerged branch must carry that branch's commit to
verify. None of this is a mathematical judgement.

The key sets were taken from a census of the federation's reference-shaped objects
(29-30 October 2026). Every Git blob key's resolvable values equalled the Git blob
id. Keys that pin another repository (such as math_pin) are not commit keys.

Usage:
  reference_crawler.py --workspace DIR [--workspace-rev REV] [--cache DIR] [--repo NAME=PATH]...
                   [--offline] [--json FILE] [--markdown FILE] [--strict] [--require-current-pins]
                   [--include-fixtures] [--at NAME=REV]...
"""

import argparse
import ast
import hashlib
import json
import posixpath
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

BLOCKING = {"UNKNOWN_REPOSITORY", "MISSING_COMMIT", "MISSING_PATH", "IDENTITY_MISMATCH", "LIVE_DRIFT",
            "UNREACHABLE_REPOSITORY", "MISSING_PIN", "TARGET_UNREACHABLE", "ALIAS_CONFLICT",
            "MALFORMED_IDENTITY", "UNPARSEABLE", "UNREAD", "TOPOLOGY_INCOMPLETE"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMITISH = re.compile(r"^[0-9a-f]{7,40}$")
REF_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
REPO_KEYS = ("repository", "repo", "full_name")
COMMIT_KEYS = ("commit", "ref", "source_commit", "cut", "head", "at_commit")
BLOB_KEYS = ("blob", "git_blob", "observed_blob", "blob_sha", "blob_sha1", "git_blob_sha", "git_blob_sha1")
SHA256_KEYS = ("sha256", "raw_sha256")
BYTES_KEYS = ("bytes", "byte_length")
ALSO_AT_PREFIX = "byte_identical_at_"
JSON_SUFFIXES = (".json", ".ipynb")
JSON_LINES_SUFFIXES = (".jsonl", ".ndjson")
URL_PATH_CHAR = r"[^\s()\]>\"'`#?|,\\]"  # a backslash is never part of a URL path
URL_RE = re.compile(
    r"https://(?:github\.com/(?P<o1>[\w.-]+)/(?P<r1>[\w.-]+)/(?:blob|tree|raw)/(?P<c1>[0-9a-f]{40})/"
    r"|raw\.githubusercontent\.com/(?P<o2>[\w.-]+)/(?P<r2>[\w.-]+)/(?P<c2>[0-9a-f]{40})/)"
    rf"(?P<path>(?:{URL_PATH_CHAR}|\({URL_PATH_CHAR}*\))+)")
SHAPE_DEPTH = 6
MAX_TEXT_BYTES = 64_000_000
# Synthetic test data is not federation reference data: files under these directory
# names are skipped unless --include-fixtures is given, and the skipped count is reported.
FIXTURE_SEGMENTS = {"tests", "test", "testdata", "fixtures", "examples"}
TEMPLATE_PREFIX = "TEMPLATE"


class CrawlError(Exception):
    """The crawl cannot proceed: topology unreadable or a repository unreachable."""


# ---------------------------------------------------------------- Git access

def git(path, *args, check=True, binary=False, inp=None):
    proc = subprocess.run(["git", "-C", str(path), *args], input=inp,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=not binary)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, proc.stdout, proc.stderr)
    return proc


class Mirror:
    """One repository's objects, with commit lookups, cached trees and on-demand fetch."""

    def __init__(self, name, path, offline=False):
        self.name, self.path, self.offline = name, Path(path), offline
        self._trees = {}
        self._sizes = {}
        self._fetched = set()

    def has_commit(self, sha):
        return git(self.path, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0

    def ensure_commit(self, sha):
        if self.has_commit(sha):
            return True
        if self.offline or sha in self._fetched:
            return False
        self._fetched.add(sha)
        git(self.path, "fetch", "--quiet", "origin", sha, check=False)
        return self.has_commit(sha)

    def tree(self, sha):
        """Map of path -> (type, object id) for every entry at sha, directories included."""
        if sha not in self._trees:
            out = git(self.path, "ls-tree", "-r", "-t", "-l", "-z", sha).stdout
            entries, sizes = {}, {}
            for item in out.split("\0"):
                if item:
                    meta, path = item.split("\t", 1)
                    _mode, kind, oid, size = meta.split()
                    entries[path] = (kind, oid)
                    sizes[path] = int(size) if size.isdigit() else 0
            self._trees[sha], self._sizes[sha] = entries, sizes
        return self._trees[sha]

    def size(self, sha, path):
        """The byte size of the blob at path in sha's tree."""
        self.tree(sha)
        return self._sizes[sha][path]

    def blob(self, oid):
        return git(self.path, "cat-file", "blob", oid, binary=True).stdout

    def default_tip(self):
        proc = git(self.path, "rev-parse", "--verify", "--quiet", "HEAD^{commit}", check=False)
        return proc.stdout.strip() or None


# ---------------------------------------------------------------- topology

@dataclass
class Node:
    name: str
    full_name: str
    url: str
    commit: str
    pinned: bool
    registered: bool
    mirror: Mirror = field(repr=False, default=None)
    available: bool = True
    candidate: bool = False


def owner_of(full_name_or_url):
    m = re.search(r"github\.com[/:]([\w.-]+)/", full_name_or_url or "")
    if m:
        return m.group(1)
    if "/" in (full_name_or_url or "") and "://" not in full_name_or_url:
        return full_name_or_url.split("/")[0]
    return None


def name_of(value):
    """Short repository name from a short name, owner/name, or a URL."""
    value = (value or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.rsplit("/", 1)[-1] if "/" in value else value


def read_json(mirror, sha, path):
    entry = mirror.tree(sha).get(path)
    if not entry or entry[0] != "blob":
        return None
    return json.loads(mirror.blob(entry[1]).decode("utf-8"))


def resolve_revision(mirror, rev):
    """The commit rev names in mirror, fetching a full SHA if needed, or None."""
    if HEX40.match(rev):
        mirror.ensure_commit(rev)
    return git(mirror.path, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}", check=False).stdout.strip() or None


def unpinned_node(name, full):
    """A registered repository without a pin, to be cloned from the URL its owner/name gives."""
    url = f"https://github.com/{full}.git" if "/" in str(full) else ""
    return Node(name, full if "/" in str(full) else name, url, "", False, True)


def discover(workspace, rev, mirror_for, findings, at=None):
    """Build the topology from the workspace's pins and registries at rev; --at revisions replace pins."""
    at = at or {}
    ws = Mirror("workspace", workspace)
    rev = git(workspace, "rev-parse", "--verify", f"{rev}^{{commit}}").stdout.strip()
    try:
        pins = read_json(ws, rev, "pins.json")
    except (ValueError, UnicodeDecodeError) as exc:
        raise CrawlError(f"workspace pins.json does not parse: {exc}") from None
    if not pins or pins.get("schema") != "ulw.pins.v1" or not isinstance(pins.get("repos"), list):
        raise CrawlError("workspace pins.json is missing or is not schema ulw.pins.v1")
    def incomplete(repo, path, detail):
        findings.append(finding("TOPOLOGY_INCOMPLETE", repo, path, detail))

    ws_registry = "workspace/repositories.json"
    try:
        registry = read_json(ws, rev, ws_registry)
    except (ValueError, UnicodeDecodeError):
        registry = None
        findings.append(finding("UNPARSEABLE", "workspace", ws_registry, "the workspace registry does not parse"))
    else:
        if registry is None:
            incomplete("workspace", ws_registry, "the workspace registry is missing")
        elif not isinstance(registry, dict):
            incomplete("workspace", ws_registry, "the workspace registry is not a JSON object")
            registry = None
    registered = {}
    if registry is not None:
        entries = registry.get("repositories")
        if not isinstance(entries, list):
            incomplete("workspace", ws_registry, "the workspace registry has no repositories list")
            entries = []
        for i, r in enumerate(entries):
            if isinstance(r, dict) and isinstance(r.get("full_name"), str) and r["full_name"]:
                registered[name_of(r["full_name"])] = r["full_name"]
            else:
                incomplete("workspace", ws_registry, f"repositories[{i}] has no full_name")
    registry = registry or {}
    ws_full = registry.get("workspace") or "workspace"
    nodes = {name_of(ws_full): Node(name_of(ws_full), ws_full, "", rev, False, True, ws)}
    for entry in pins["repos"]:
        name = entry.get("name")
        sha = entry.get("pinned_sha", "")
        if not name or not HEX40.match(sha or ""):
            raise CrawlError(f"pins.json entry {entry!r} lacks a name or a 40-hex pinned_sha")
        full = registered.get(name) or (f"{owner_of(entry.get('url'))}/{name}" if owner_of(entry.get("url")) else name)
        nodes[name] = Node(name, full, entry.get("url", ""), sha, True, name in registered)
    for name, full in registered.items():
        if name not in nodes:
            nodes[name] = unpinned_node(name, full)
    # The registry that canonical_inputs names contributes registered repositories too.
    inputs = registry.get("canonical_inputs")
    reg_input = inputs.get("repository_registry") if isinstance(inputs, dict) else None
    extra = {}
    if registry and not (isinstance(reg_input, dict) and isinstance(reg_input.get("repository"), str)
                         and isinstance(reg_input.get("path"), str) and reg_input["repository"] and reg_input["path"]):
        incomplete("workspace", ws_registry, "canonical_inputs.repository_registry does not name a repository and a path")
    elif registry:
        host = nodes.get(name_of(reg_input["repository"]))
        doc = None
        if host is None:
            incomplete("workspace", ws_registry, f"the repository registry's host {reg_input['repository']!r} is not in the topology")
        else:
            try:
                host.mirror = host.mirror or mirror_for(host)
                commit = host.commit or host.mirror.default_tip()
                if not commit or not host.mirror.ensure_commit(commit):
                    incomplete(host.name, reg_input["path"], f"the registry host's commit {commit or '(none)'} is not available")
                else:
                    doc = read_json(host.mirror, commit, reg_input["path"])
                    if doc is None:
                        incomplete(host.name, reg_input["path"], f"the repository registry is missing at {commit}")
            except CrawlError:
                incomplete(host.name, reg_input["path"], f"the registry host {host.name} is unreachable")
            except (ValueError, UnicodeDecodeError):
                findings.append(finding("UNPARSEABLE", host.name, reg_input["path"],
                                        "the repository registry does not parse"))
        if doc is not None:
            repos = doc.get("repositories") if isinstance(doc, dict) else None
            if not isinstance(repos, dict):
                incomplete(host.name, reg_input["path"], "the repository registry has no repositories map")
            else:
                for key in repos:
                    extra[name_of(key)] = key
    for name in at:
        if name not in nodes or nodes[name].mirror is ws:
            raise CrawlError(f"--at names {name!r}, which is not a crawled repository (use --workspace-rev for the workspace)")
    for name, full in extra.items():
        if name not in nodes:
            nodes[name] = unpinned_node(name, full)
    for node in nodes.values():
        if node.mirror is None:
            try:
                node.mirror = mirror_for(node)
            except CrawlError as exc:
                # One unreachable repository is a finding, not a halt.
                findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", str(exc)))
                node.available = False
                continue
        if node.name in at:
            sha = resolve_revision(node.mirror, at[node.name])
            if sha is None:
                findings.append(finding("MISSING_PIN", node.name, "", f"candidate revision {at[node.name]!r} not found"))
                node.available = False
                continue
            findings.append(finding("CANDIDATE", node.name, "", f"crawled at candidate {sha}; recorded {node.commit or 'none'}"))
            node.commit, node.candidate = sha, True
        elif not node.commit:
            node.commit = node.mirror.default_tip()
            if not node.commit:
                findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", "no reachable default tip"))
                node.available = False
                continue
        elif not node.mirror.ensure_commit(node.commit):
            findings.append(finding("MISSING_PIN", node.name, "", f"pinned commit {node.commit} not found"))
            node.available = False
            continue
        if node.pinned and not node.registered:
            findings.append(finding("TOPOLOGY_UNREGISTERED", node.name, "", "pinned but not in the registry"))
        if node.registered and not node.pinned and node.mirror is not ws:
            findings.append(finding("TOPOLOGY_UNPINNED", node.name, "", "registered but not pinned"))
        if node.pinned and not node.candidate:
            tip = node.mirror.default_tip()
            if tip and tip != node.commit:
                findings.append(finding("PIN_SKEW", node.name, "", f"pinned {node.commit}, tip {tip}"))
    return nodes


# ---------------------------------------------------------------- references

@dataclass
class Reference:
    source: str
    source_path: str
    repo: str
    path: str
    commit: str = None
    blobs: tuple = ()
    sha256s: tuple = ()
    sizes: tuple = ()
    kind: str = "json"
    lines: tuple = None
    text: str = None


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def distinct(obj, keys, valid):
    """The distinct valid values recorded under any of keys, in key order."""
    out = []
    for k in keys:
        if valid(obj.get(k)) and obj[k] not in out:
            out.append(obj[k])
    return tuple(out)


def hex40(value):
    return isinstance(value, str) and bool(HEX40.match(value))


def record_references(obj, repos, path, source, source_path, out, findings):
    """The references one JSON record makes. A record that contradicts itself is reported."""
    def report(cls, detail):
        if findings is not None:
            findings.append(finding(cls, name_of(repos[0]), path, detail, Reference(source, source_path, repos[0], path)))

    owners = {owner_of(r).lower() for r in repos if "/" in r and owner_of(r)}
    if len({name_of(r).lower() for r in repos}) > 1 or len(owners) > 1:
        report("ALIAS_CONFLICT", f"the repository fields disagree: {', '.join(repos)}")
        return
    repo = max(repos, key=lambda r: "/" in r)  # prefer the owner-qualified alias
    commits, named, malformed = [], [], []
    for k in COMMIT_KEYS:
        value = obj.get(k)
        if value is None:
            continue
        if isinstance(value, str) and COMMITISH.match(value):
            if value not in commits:
                commits.append(value)
        elif k == "ref" and isinstance(value, str) and REF_NAME.match(value):
            named.append(value)  # a "ref" that names a branch: a live reference
        else:
            malformed.append(k)
    checks = ((BLOB_KEYS, hex40), (SHA256_KEYS, lambda v: isinstance(v, str) and bool(HEX64.match(v))),
              (BYTES_KEYS, is_int))
    for keys, valid in checks:
        malformed += [k for k in keys if obj.get(k) is not None and not valid(obj[k])]
    also = sorted(k for k in obj if k.startswith(ALSO_AT_PREFIX))
    malformed += [k for k in also if obj[k] is not None and not hex40(obj[k])]
    for k in malformed:
        report("MALFORMED_IDENTITY", f"{k} = {str(obj[k])[:60]!r} is not a valid value")
    blobs = distinct(obj, BLOB_KEYS, hex40)
    sha256s = distinct(obj, SHA256_KEYS, checks[1][1])
    sizes = distinct(obj, BYTES_KEYS, is_int)
    lines = (obj["line_start"], obj["line_end"]) if is_int(obj.get("line_start")) and is_int(obj.get("line_end")) else None
    text = obj.get("text") if lines and isinstance(obj.get("text"), str) else None
    also = [k for k in also if hex40(obj[k])]
    if also and not (blobs or sha256s or sizes):
        report("MALFORMED_IDENTITY", f"{also[0]} claims byte identity, but there is no blob, sha256 or bytes to compare")
        also = []
    if not (commits or named or blobs or sha256s or sizes):
        return  # no identity is claimed, so there is nothing to check
    # Each distinct commit is its own claim; a record without one is a live reference.
    for commit in commits or [None]:
        out.append(Reference(source, source_path, repo, path, commit, blobs, sha256s, sizes, "json", lines, text))
    for key in also:
        # A claim that the same identity also holds at a further commit.
        out.append(Reference(source, source_path, repo, path, obj[key], blobs, sha256s, sizes, "json", lines, text))


def json_references(obj, source, source_path, out, findings=None):
    if isinstance(obj, dict):
        repos = distinct(obj, REPO_KEYS, lambda v: isinstance(v, str) and bool(v.strip()))
        path = obj.get("path")
        if repos and isinstance(path, str) and path:
            record_references(obj, repos, path, source, source_path, out, findings)
        for value in obj.values():
            json_references(value, source, source_path, out, findings)
    elif isinstance(obj, list):
        for value in obj:
            json_references(value, source, source_path, out, findings)


def text_references(text, source, source_path, owners, out):
    for m in URL_RE.finditer(text):
        owner = m.group("o1") or m.group("o2")
        if owner.lower() not in owners:
            continue
        repo = f"{owner}/{m.group('r1') or m.group('r2')}"
        commit = m.group("c1") or m.group("c2")
        path = urllib.parse.unquote(m.group("path").rstrip(".:;"))
        out.append(Reference(source, source_path, repo, path, commit, kind="url"))


def json_strings(obj):
    """Every string in a parsed JSON document, keys included."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from json_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from json_strings(value)


def strip_json_comments(text):
    """JSON with // and /* */ comments and trailing commas, reduced to JSON. Strings are kept intact."""
    pieces, i, n = [], 0, len(text)  # (is_string, text) pieces outside comments
    while i < n:
        if text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            pieces.append((True, text[i:j + 1]))
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated block comment")
            i = end + 2
        else:
            pieces.append((False, text[i]))
            i += 1
    out = []
    for k, (is_string, piece) in enumerate(pieces):
        if not is_string and piece == ",":
            j = k + 1
            while j < len(pieces) and not pieces[j][0] and pieces[j][1].isspace():
                j += 1
            if j < len(pieces) and not pieces[j][0] and pieces[j][1] in "}]":
                continue  # a trailing comma
        out.append(piece)
    return "".join(out)


def json_documents(text, path):
    """The parsed JSON documents of a file, the text that did not parse, and whether comments were removed."""
    if path.endswith(JSON_LINES_SUFFIXES):
        docs, rest = [], []
        for line in text.splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except ValueError:
                    rest.append(line)
        return docs, "\n".join(rest), False
    try:
        return [json.loads(text)], "", False
    except ValueError:
        pass
    try:
        return [json.loads(strip_json_comments(text))], "", True
    except ValueError:
        return [], text, False


def file_references(text, path, source, owners, refs, findings):
    """References in one text file. In JSON, URLs are matched in the decoded strings."""
    if not path.endswith(JSON_SUFFIXES + JSON_LINES_SUFFIXES):
        text_references(text, source, path, owners, refs)
        return
    docs, unparsed, with_comments = json_documents(text, path)
    if with_comments:
        findings.append(finding("JSON_WITH_COMMENTS", source, path, "read after removing comments and trailing commas"))
    for doc in docs:
        json_references(doc, source, path, refs, findings)
        for string in json_strings(doc):
            text_references(string, source, path, owners, refs)
    if unparsed:
        findings.append(finding("UNPARSEABLE", source, path,
                                "JSON does not parse: its records are unchecked; URLs are matched in the raw text"))
        text_references(unparsed, source, path, owners, refs)


def is_fixture(path):
    parts = path.split("/")
    return any(part in FIXTURE_SEGMENTS for part in parts[:-1]) or parts[-1].startswith(TEMPLATE_PREFIX)


def collect(node, owners, findings, include_fixtures=False, skipped=None, binary=None):
    """References in every file of node at its crawled commit. Nothing is skipped uncounted."""
    refs = []
    for path, (kind, oid) in sorted(node.mirror.tree(node.commit).items()):
        if kind != "blob":
            continue
        if not include_fixtures and is_fixture(path):
            if skipped is not None:
                skipped.setdefault(node.name, []).append(path)
            continue
        size = node.mirror.size(node.commit, path)
        if size > MAX_TEXT_BYTES:
            findings.append(finding("UNREAD", node.name, path, f"{size} bytes, over the {MAX_TEXT_BYTES}-byte read limit"))
            continue
        try:
            text = node.mirror.blob(oid).decode("utf-8")
        except UnicodeDecodeError:
            if binary is not None:
                binary.setdefault(node.name, []).append(path)
            continue
        file_references(text, path, node.name, owners, refs, findings)
    return refs


# ---------------------------------------------------------------- interface extractors

def strip_lean_comments(text):
    """Remove Lean line and (nested) block comments, keeping string literals and line count."""
    out, i, depth, n = [], 0, 0, len(text)
    while i < n:
        if depth:
            if text.startswith("/-", i):
                depth, i = depth + 1, i + 2
            elif text.startswith("-/", i):
                depth, i = depth - 1, i + 2
            else:
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
        elif text.startswith("/-", i):
            depth, i = 1, i + 2
        elif text.startswith("--", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


LEAN_DECL = re.compile(
    r"^(?:@\[[^\]]*\]\s*)*(?:(?:private|protected|noncomputable|partial|unsafe|nonrec|scoped|local)\s+)*"
    r"(?P<kw>theorem|lemma|def|abbrev|axiom|instance|structure|inductive|class|opaque|example)\b(?P<rest>.*)$")
LEAN_SCOPE = re.compile(r"^(?P<kw>namespace|section|end)\b\s*(?P<name>[\w.']*)")


def lean_statements(text):
    """Map of declaration name -> normalized statement text (up to its top-level :=)."""
    lines = strip_lean_comments(text).splitlines()
    decls, scopes, anon = {}, [], 0
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        scope = LEAN_SCOPE.match(line)
        if scope:
            kw, name = scope.group("kw"), scope.group("name")
            if kw in ("namespace", "section"):
                scopes.append((kw, name))
            elif scopes:
                scopes.pop()
            i += 1
            continue
        m = LEAN_DECL.match(line)
        if not m:
            i += 1
            continue
        body = [m.group("rest")]
        j = i + 1
        while not re.search(r":=|\bwhere\b", " ".join(body)) and j < len(lines) and not LEAN_DECL.match(lines[j].strip()):
            body.append(lines[j].strip())
            j += 1
        joined = " ".join(body)
        stmt = re.split(r":=|\bwhere\b", joined, maxsplit=1)[0]
        tokens = stmt.split()
        if m.group("kw") == "example" or not tokens or tokens[0].startswith((":", "(", "{", "[")):
            anon += 1
            name, signature = f"<{m.group('kw')}#{anon}>", " ".join(tokens)
        else:
            name, signature = tokens[0], " ".join(tokens[1:])
        prefix = ".".join(n for kw, n in scopes if kw == "namespace" and n)
        decls[f"{prefix}.{name}" if prefix else name] = f"{m.group('kw')} {signature}".strip()
        i = j if j > i + 1 else i + 1
    return decls


def python_signatures(text):
    """Map of qualified name -> signature for top-level functions, classes and methods."""
    tree = ast.parse(text)
    sigs = {}

    def sig(fn):
        ret = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
        prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {fn.name}({ast.unparse(fn.args)}){ret}"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sigs[node.name] = sig(node)
        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(ast.unparse(b) for b in node.bases)
            sigs[node.name] = f"class {node.name}({bases})"
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sigs[f"{node.name}.{item.name}"] = sig(item)
    return sigs


def json_shape(value, depth=0):
    """Flattened map of JSON path -> type, to depth SHAPE_DEPTH; every list item is merged."""
    shape = {}

    def walk(v, path, d):
        kind = ("object" if isinstance(v, dict) else "array" if isinstance(v, list)
                else "boolean" if isinstance(v, bool) else "number" if isinstance(v, (int, float))
                else "null" if v is None else "string")
        shape.setdefault(path or "$", set()).add(kind)
        if d >= SHAPE_DEPTH:
            return
        if isinstance(v, dict):
            for k, child in v.items():
                walk(child, f"{path}.{k}" if path else k, d + 1)
        elif isinstance(v, list):
            for child in v:
                walk(child, f"{path}[]", d + 1)

    walk(value, "", depth)
    return {k: "|".join(sorted(v)) for k, v in shape.items()}


def diff_maps(old, new):
    removed = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    return {"removed": removed, "added": added, "changed": changed}


def interface_drift(path, old_bytes, new_bytes):
    """Interface-level change between two versions of one file, or None if not applicable."""
    try:
        old, new = old_bytes.decode("utf-8"), new_bytes.decode("utf-8")
        if path.endswith(".lean"):
            return {"kind": "lean_statements", **diff_maps(lean_statements(old), lean_statements(new))}
        if path.endswith(".py"):
            return {"kind": "python_signatures", **diff_maps(python_signatures(old), python_signatures(new))}
        if path.endswith(".json"):
            return {"kind": "json_shape", **diff_maps(json_shape(json.loads(old)), json_shape(json.loads(new)))}
    except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
        return {"kind": "unparsed", "error": str(exc)[:200], "removed": [], "added": [], "changed": []}
    return None


# ---------------------------------------------------------------- checks

def finding(cls, repo, path, detail, ref=None, extra=None):
    row = {"class": cls, "blocking": cls in BLOCKING, "repo": repo, "path": path, "detail": detail}
    if ref is not None:
        row["source"] = f"{ref.source}:{ref.source_path}"
        row["reference_kind"] = ref.kind
        row["commit"] = ref.commit
    if extra:
        row.update(extra)
    return row


def identity_problems(data, ref, scope=""):
    """The recorded SHA-256 and byte-count claims that data does not satisfy."""
    digest = hashlib.sha256(data).hexdigest()
    problems = [f"sha256{scope} differs" for d in ref.sha256s if d != digest]
    problems += [f"bytes {n} recorded, {len(data)} found{scope}" for n in ref.sizes if n != len(data)]
    return problems


def resolve(ref, nodes, owners):
    """The topology node a reference names, or None; and whether its owner is outside the federation."""
    name = name_of(ref.repo)
    target = nodes.get(name) or next((n for k, n in nodes.items() if k.lower() == name.lower()), None)
    owner = (owner_of(ref.repo) or "").lower() if "/" in ref.repo else ""
    if target is not None and owner and (owner_of(target.full_name) or owner).lower() != owner:
        target = None  # another owner's repository with the same short name
    return target, bool(owner) and owner not in {o.lower() for o in owners}


def check_reference(ref, nodes, owners, findings):
    target, foreign = resolve(ref, nodes, owners)
    if target is None:
        if foreign:
            findings.append(finding("FOREIGN_REPOSITORY", ref.repo, ref.path,
                                    "the owner is outside the federation, so this reference is not checked", ref))
        else:
            findings.append(finding("UNKNOWN_REPOSITORY", name_of(ref.repo), ref.path,
                                    f"repository {ref.repo!r} is not in the workspace topology", ref))
        return
    if not target.available:
        findings.append(finding("TARGET_UNREACHABLE", target.name, ref.path,
                                f"{target.name} is unreachable, so this reference is unchecked", ref))
        return
    mirror = target.mirror
    live = ref.commit is None
    commit = target.commit if live else ref.commit
    if not live and not mirror.ensure_commit(commit):
        findings.append(finding("MISSING_COMMIT", target.name, ref.path, f"commit {commit} not found", ref))
        return
    path = posixpath.normpath(ref.path).lstrip("/")
    entry = mirror.tree(commit).get(path)
    if entry is None:
        cls = "LIVE_DRIFT" if live else "MISSING_PATH"
        findings.append(finding(cls, target.name, path, f"path absent at {commit}", ref))
        return
    kind, oid = entry
    problems = [f"blob {b} recorded, {oid} found" for b in ref.blobs if b != oid]
    if ref.lines and kind == "blob":
        # An excerpt: the identity covers exactly lines line_start..line_end of the file.
        start, end = ref.lines
        file_lines = mirror.blob(oid).decode("utf-8", "replace").splitlines(keepends=True)
        if not 1 <= start <= end <= len(file_lines):
            problems.append(f"lines {start}-{end} outside 1-{len(file_lines)}")
        else:
            excerpt = "".join(file_lines[start - 1:end]).encode("utf-8")
            problems += identity_problems(excerpt, ref, f" in lines {start}-{end}")
            if ref.text is not None and ref.text.encode("utf-8") != excerpt:
                problems.append(f"text of lines {start}-{end} differs")
    elif (ref.sha256s or ref.sizes) and kind == "blob":
        problems += identity_problems(mirror.blob(oid), ref)
    elif ref.sha256s or ref.sizes or ref.lines:
        problems.append(f"a file identity is recorded, but the path is a {kind}")
    if problems:
        cls = "LIVE_DRIFT" if live else "IDENTITY_MISMATCH"
        findings.append(finding(cls, target.name, path, "; ".join(problems), ref))
        return
    if live or kind != "blob":
        return
    current = mirror.tree(target.commit).get(path)
    if current is None:
        findings.append(finding("FROZEN_REMOVED", target.name, path, f"removed at {target.commit}", ref))
    elif current[1] != oid:
        drift = interface_drift(path, mirror.blob(oid), mirror.blob(current[1]))
        findings.append(finding("FROZEN_CHANGED", target.name, path,
                                f"changed between {commit} and {target.commit}", ref,
                                {"interface": drift} if drift else None))


def interface_changed(row):
    d = row.get("interface")
    return bool(d and (d["removed"] or d["added"] or d["changed"] or d["kind"] == "unparsed"))


def crawl(workspace, rev="HEAD", cache=None, overrides=None, offline=False, include_fixtures=False, skipped=None,
          binary=None, at=None):
    findings = []
    overrides = overrides or {}

    def mirror_for(node):
        if node.name in overrides:
            return Mirror(node.name, overrides[node.name], offline)
        if cache is None or not node.url:
            raise CrawlError(f"no local mirror for {node.name}, and no cache or url to clone from")
        path = Path(cache) / f"{node.name}.git"
        if not path.exists():
            if offline:
                raise CrawlError(f"{node.name} is not mirrored and --offline forbids cloning")
            proc = subprocess.run(["git", "clone", "--quiet", "--bare", node.url, str(path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode != 0:
                raise CrawlError(f"cannot clone {node.name} from {node.url}: {proc.stderr.strip()[:200]}")
        return Mirror(node.name, path, offline)

    nodes = discover(workspace, rev, mirror_for, findings, at)
    owners = {o.lower() for n in nodes.values() for o in (owner_of(n.full_name), owner_of(n.url)) if o}
    references = []
    for node in nodes.values():
        if node.available:
            references.extend(collect(node, owners, findings, include_fixtures, skipped, binary))
    for ref in references:
        check_reference(ref, nodes, owners, findings)
    return nodes, references, findings


def summarize(nodes, references, findings, strict=False, require_current_pins=False, skipped=None, binary=None):
    def blocks(row):
        return (row["blocking"] or (strict and row["class"] == "FROZEN_CHANGED" and interface_changed(row))
                or (require_current_pins and row["class"] == "PIN_SKEW"))
    counts = {}
    for row in findings:
        counts[row["class"]] = counts.get(row["class"], 0) + 1
    blocking = [row for row in findings if blocks(row)]
    return {
        "schema": "cbz.references.v1",
        "topology": [{"name": n.name, "full_name": n.full_name, "commit": n.commit, "pinned": n.pinned,
                      "registered": n.registered, "candidate": n.candidate} for n in nodes.values()],
        "references": len(references),
        "counts": dict(sorted(counts.items())),
        "blocking": len(blocking),
        "strict": strict, "require_current_pins": require_current_pins,
        "skipped_fixture_files": {k: len(v) for k, v in sorted((skipped or {}).items())},
        "binary_files_not_read": {k: len(v) for k, v in sorted((binary or {}).items())},
        "skipped_fixture_paths": dict(sorted((skipped or {}).items())),
        "binary_paths": dict(sorted((binary or {}).items())),
        "findings": findings,
    }


def render(report):
    skipped = sum(report["skipped_fixture_files"].values())
    binary = sum(report["binary_files_not_read"].values())
    lines = ["# Federation reference report", "",
             f"{report['references']} references checked across {len(report['topology'])} repositories; "
             f"**{report['blocking']} blocking**. Fixture files skipped: {skipped} "
             f"(use --include-fixtures to check them). Binary files not read: {binary}.", "",
             "| Repository | Commit | Pinned | Registered |", "|---|---|---|---|"]
    for n in report["topology"]:
        lines.append(f"| {n['name']} | `{n['commit'][:12]}` | {n['pinned']} | {n['registered']} |")
    lines += ["", "| Finding | Count | Blocking by default |", "|---|---|---|"]
    for cls, count in report["counts"].items():
        lines.append(f"| {cls} | {count} | {cls in BLOCKING} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--workspace-rev", default="HEAD")
    p.add_argument("--cache", type=Path)
    p.add_argument("--repo", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--json", type=Path)
    p.add_argument("--markdown", type=Path)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--require-current-pins", action="store_true")
    p.add_argument("--include-fixtures", action="store_true",
                   help="also check references inside tests/, fixtures/ and examples/ directories")
    p.add_argument("--at", action="append", default=[], metavar="NAME=REV",
                   help="crawl repository NAME at REV, a candidate such as a PR head, instead of its pin")
    args = p.parse_args(argv)
    overrides = {}
    for item in args.repo:
        name, sep, path = item.partition("=")
        if not sep:
            p.error(f"--repo expects NAME=PATH, got {item!r}")
        overrides[name] = path
    at = {}
    for item in args.at:
        name, sep, rev = item.partition("=")
        if not sep or not rev:
            p.error(f"--at expects NAME=REV, got {item!r}")
        at[name] = rev
    skipped, binary = {}, {}
    try:
        nodes, references, findings = crawl(args.workspace, args.workspace_rev, args.cache, overrides, args.offline,
                                            args.include_fixtures, skipped, binary, at)
    except (CrawlError, subprocess.CalledProcessError) as exc:  # workspace-level failure
        print(f"REFERENCE CRAWL ERROR: {exc}", file=sys.stderr)
        return 2
    report = summarize(nodes, references, findings, args.strict, args.require_current_pins, skipped, binary)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.write_text(render(report))
    print(f"reference crawl: {report['references']} references, {len(report['topology'])} repositories, "
          f"{report['blocking']} blocking; counts {report['counts']}")
    return 1 if report["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main())
