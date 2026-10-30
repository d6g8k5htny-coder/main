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
registered but not pinned is crawled at its default-branch tip.

References, extracted from every file of every node at its crawled commit that is
valid UTF-8 text, whatever its extension:
  * JSON (.json and .ipynb documents, and each line of .jsonl and .ndjson files): any
    object with a repository key (repository, repo or full_name), a
    string "path", and at least one identity: a commit (COMMIT_KEYS), a Git blob id
    (BLOB_KEYS), a SHA-256 (SHA256_KEYS) or a byte count (BYTES_KEYS). Every
    recorded value is checked, not only the first. A byte_identical_at_<name> key
    with a 40-hex value claims the same identity at that further commit, and is
    checked there too;
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
Files under a tests, test, testdata, fixtures or examples directory are synthetic data.
They are skipped unless --include-fixtures is given, and the report counts them.

Findings. BLOCKING findings make the exit status 1:
  UNKNOWN_REPOSITORY    the reference names a repository outside the topology
  MISSING_COMMIT        the referenced commit cannot be found or fetched
  MISSING_PATH          the path does not exist at the referenced commit
  IDENTITY_MISMATCH     the recorded blob, sha256 or byte count differs
  LIVE_DRIFT            a reference without a commit no longer matches the target's
                        crawled commit
  UNREACHABLE_REPOSITORY  a topology repository has no mirror and cannot be cloned
  MISSING_PIN           a pinned commit cannot be found or fetched
  TARGET_UNREACHABLE    a reference into a topology repository that is unreachable
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
  UNPARSEABLE           a JSON file, or a JSON Lines line, that does not parse: its records are
                        unchecked, and URLs are still matched in its raw text
  UNREAD                a file over MAX_TEXT_BYTES, which is not read
Exit status 2 means the crawl itself failed: the workspace's pins are missing,
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
                   [--include-fixtures]
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
            "UNREACHABLE_REPOSITORY", "MISSING_PIN", "TARGET_UNREACHABLE"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
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


def discover(workspace, rev, mirror_for, findings):
    """Build the topology from the workspace's pins and registries at rev."""
    ws = Mirror("workspace", workspace)
    rev = git(workspace, "rev-parse", "--verify", f"{rev}^{{commit}}").stdout.strip()
    try:
        pins = read_json(ws, rev, "pins.json")
    except (ValueError, UnicodeDecodeError) as exc:
        raise CrawlError(f"workspace pins.json does not parse: {exc}") from None
    if not pins or pins.get("schema") != "ulw.pins.v1" or not isinstance(pins.get("repos"), list):
        raise CrawlError("workspace pins.json is missing or is not schema ulw.pins.v1")
    try:
        registry = read_json(ws, rev, "workspace/repositories.json") or {}
    except (ValueError, UnicodeDecodeError):
        registry = {}
        findings.append(finding("UNPARSEABLE", "workspace", "workspace/repositories.json",
                                "the workspace registry does not parse"))
    registered = {name_of(r.get("full_name")): r.get("full_name")
                  for r in registry.get("repositories", []) if isinstance(r, dict) and r.get("full_name")}
    ws_full = registry.get("workspace") or "workspace"
    nodes = {name_of(ws_full): Node(name_of(ws_full), ws_full, "", rev, False, True, ws)}
    for entry in pins["repos"]:
        name = entry.get("name")
        sha = entry.get("pinned_sha", "")
        if not name or not HEX40.match(sha or ""):
            raise CrawlError(f"pins.json entry {entry!r} lacks a name or a 40-hex pinned_sha")
        full = registered.get(name) or (f"{owner_of(entry.get('url'))}/{name}" if owner_of(entry.get("url")) else name)
        nodes[name] = Node(name, full, entry.get("url", ""), sha, True, name in registered)
    # The registry that canonical_inputs names contributes registered repositories too.
    reg_input = (registry.get("canonical_inputs") or {}).get("repository_registry") or {}
    extra = {}
    if reg_input.get("repository") and reg_input.get("path"):
        host = nodes.get(name_of(reg_input["repository"]))
        if host is not None:
            try:
                host.mirror = host.mirror or mirror_for(host)
                doc = read_json(host.mirror, host.commit, reg_input["path"]) or {}
            except CrawlError:
                doc = {}  # the host's own UNREACHABLE_REPOSITORY finding is recorded below
            except (ValueError, UnicodeDecodeError):
                doc = {}
                findings.append(finding("UNPARSEABLE", host.name, reg_input["path"],
                                        "the repository registry does not parse"))
            repos = doc.get("repositories", {})
            for key in (repos if isinstance(repos, dict) else []):
                extra[name_of(key)] = key
    for name, full in list(registered.items()) + list(extra.items()):
        if name not in nodes:
            nodes[name] = Node(name, full if "/" in str(full) else name, "", "", False, True)
    for node in nodes.values():
        if node.mirror is None:
            try:
                node.mirror = mirror_for(node)
            except CrawlError as exc:
                # One unreachable repository is a finding, not a halt.
                findings.append(finding("UNREACHABLE_REPOSITORY", node.name, "", str(exc)))
                node.available = False
                continue
        if not node.commit:
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
        if node.pinned:
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


def json_references(obj, source, source_path, out):
    if isinstance(obj, dict):
        repo = next((obj[k] for k in REPO_KEYS if isinstance(obj.get(k), str)), None)
        path = obj.get("path")
        if repo and isinstance(path, str) and path:
            hex40 = lambda v: isinstance(v, str) and bool(HEX40.match(v))  # noqa: E731
            commit = next((obj[k] for k in COMMIT_KEYS if hex40(obj.get(k))), None)
            blobs = distinct(obj, BLOB_KEYS, hex40)
            sha256s = distinct(obj, SHA256_KEYS, lambda v: isinstance(v, str) and bool(HEX64.match(v)))
            sizes = distinct(obj, BYTES_KEYS, is_int)
            lines = (obj["line_start"], obj["line_end"]) if is_int(obj.get("line_start")) and is_int(obj.get("line_end")) else None
            text = obj.get("text") if lines and isinstance(obj.get("text"), str) else None
            if commit or blobs or sha256s or sizes:
                out.append(Reference(source, source_path, repo, path, commit, blobs, sha256s, sizes, "json", lines, text))
                for key in sorted(k for k in obj if k.startswith(ALSO_AT_PREFIX) and hex40(obj[k])):
                    # A claim that the same identity also holds at a further commit.
                    out.append(Reference(source, source_path, repo, path, obj[key], blobs, sha256s, sizes,
                                         "json", lines, text))
        for value in obj.values():
            json_references(value, source, source_path, out)
    elif isinstance(obj, list):
        for value in obj:
            json_references(value, source, source_path, out)


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


def json_documents(text, path):
    """The parsed JSON documents of a file, and the text that did not parse."""
    if path.endswith(JSON_LINES_SUFFIXES):
        docs, rest = [], []
        for line in text.splitlines():
            if line.strip():
                try:
                    docs.append(json.loads(line))
                except ValueError:
                    rest.append(line)
        return docs, "\n".join(rest)
    try:
        return [json.loads(text)], ""
    except ValueError:
        return [], text


def file_references(text, path, source, owners, refs, findings):
    """References in one text file. In JSON, URLs are matched in the decoded strings."""
    if not path.endswith(JSON_SUFFIXES + JSON_LINES_SUFFIXES):
        text_references(text, source, path, owners, refs)
        return
    docs, unparsed = json_documents(text, path)
    for doc in docs:
        json_references(doc, source, path, refs)
        for string in json_strings(doc):
            text_references(string, source, path, owners, refs)
    if unparsed:
        findings.append(finding("UNPARSEABLE", source, path,
                                "JSON does not parse: its records are unchecked; URLs are matched in the raw text"))
        text_references(unparsed, source, path, owners, refs)


def is_fixture(path):
    return any(part in FIXTURE_SEGMENTS for part in path.split("/")[:-1])


def collect(node, owners, findings, include_fixtures=False, skipped=None, binary=None):
    """References in every file of node at its crawled commit. Nothing is skipped uncounted."""
    refs = []
    for path, (kind, oid) in sorted(node.mirror.tree(node.commit).items()):
        if kind != "blob":
            continue
        if not include_fixtures and is_fixture(path):
            if skipped is not None:
                skipped[node.name] = skipped.get(node.name, 0) + 1
            continue
        size = node.mirror.size(node.commit, path)
        if size > MAX_TEXT_BYTES:
            findings.append(finding("UNREAD", node.name, path, f"{size} bytes, over the {MAX_TEXT_BYTES}-byte read limit"))
            continue
        try:
            text = node.mirror.blob(oid).decode("utf-8")
        except UnicodeDecodeError:
            if binary is not None:
                binary[node.name] = binary.get(node.name, 0) + 1
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
    """Flattened map of JSON path -> type, to depth SHAPE_DEPTH; list items are merged."""
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
            for child in v[:50]:
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
          binary=None):
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

    nodes = discover(workspace, rev, mirror_for, findings)
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
        "topology": [{"name": n.name, "full_name": n.full_name, "commit": n.commit,
                      "pinned": n.pinned, "registered": n.registered} for n in nodes.values()],
        "references": len(references),
        "counts": dict(sorted(counts.items())),
        "blocking": len(blocking),
        "strict": strict, "require_current_pins": require_current_pins,
        "skipped_fixture_files": dict(sorted((skipped or {}).items())),
        "binary_files_not_read": dict(sorted((binary or {}).items())),
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
    args = p.parse_args(argv)
    overrides = {}
    for item in args.repo:
        name, sep, path = item.partition("=")
        if not sep:
            p.error(f"--repo expects NAME=PATH, got {item!r}")
        overrides[name] = path
    skipped, binary = {}, {}
    try:
        nodes, references, findings = crawl(args.workspace, args.workspace_rev, args.cache, overrides, args.offline,
                                            args.include_fixtures, skipped, binary)
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
