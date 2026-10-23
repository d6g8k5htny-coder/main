#!/usr/bin/env python3
"""Bounded, reproducible lexical retrieval over a fixed set of public Git blobs.

No learned model, remote service, working-tree ingestion, or scientific inference.
Run from the public repository checkout; see docs/PUBLIC_VECTOR_INDEX.md.
"""
import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import selectors
import stat
import subprocess
import sys
import time

REPOSITORY = "d6g8k5htny-coder/main"
ALLOWLIST = (
    "AGENTS.md",
    "architecture/node-status.schema.json",
    "docs/WORKSPACE.md",
    "docs/site/node-status.json",
    "governance/OP-WORKFLOW-20260930.md",
)
MAX_SOURCE_BYTES = 32768
MAX_TOTAL_SOURCE_BYTES = 65536
CHUNK_CHARS = 2048
MAX_CHUNKS = 40
MAX_INDEX_BYTES = 131072
MAX_LOG_BYTES = 262144
FRAME_BYTES = 768
GIT_TIMEOUT_SECONDS = 15
MARKER = "PUBLIC_VECTOR_INDEX_V1"
ALGORITHM = {"name": "lexical-feature-hashing", "version": 1,
             "dimensions": 256, "tokenizer": "ascii-alnum-lower"}
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ")


def _git(repo, args, limit):
    """Read local plumbing only; prohibit lazy fetch, replacements and shell use."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    command = ["git", "--no-replace-objects", "-c", "protocol.allow=never", "-C", str(repo)] + args
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          env=env, shell=False) as process:
        deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        raw = bytearray()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise subprocess.TimeoutExpired(command, GIT_TIMEOUT_SECONDS)
                piece = os.read(process.stdout.fileno(), min(65536, limit + 1 - len(raw)))
                if not piece:
                    break
                raw.extend(piece)
                if len(raw) > limit:
                    raise ValueError("Git object or response exceeds byte limit")
            code = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise ValueError("local Git operation timed out") from exc
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        if code:
            raise ValueError("required local Git object or repository configuration is unavailable")
    return bytes(raw)


def _validate_commit(commit):
    if not isinstance(commit, str) or not SHA1.fullmatch(commit):
        raise ValueError("commit must be a full lowercase 40-character Git SHA1")


def _repository_root(repo):
    root = Path(_git(repo, ["rev-parse", "--show-toplevel"], 4096).decode().strip()).resolve()
    origins = _git(repo, ["config", "--local", "--get-all", "remote.origin.url"], 4096).decode().splitlines()
    permitted = {"https://github.com/" + REPOSITORY, "https://github.com/" + REPOSITORY + ".git",
                 "git@github.com:" + REPOSITORY, "git@github.com:" + REPOSITORY + ".git"}
    if len(origins) != 1 or origins[0] not in permitted:
        raise ValueError("origin must identify the fixed public source repository")
    return root


def _object(repo, oid, kind, limit):
    if not SHA1.fullmatch(oid):
        raise ValueError("invalid Git object identity")
    actual_kind = _git(repo, ["cat-file", "-t", oid], 32).decode().strip()
    if actual_kind != kind:
        raise ValueError("Git object type does not match source identity")
    size_raw = _git(repo, ["cat-file", "-s", oid], 32).decode().strip()
    if not re.fullmatch(r"0|[1-9][0-9]*", size_raw) or int(size_raw) > limit:
        raise ValueError("Git source object exceeds byte limit")
    raw = _git(repo, ["cat-file", kind, oid], limit)
    header = (kind + " " + str(len(raw)) + "\0").encode("ascii")
    if len(raw) != int(size_raw) or hashlib.sha1(header + raw).hexdigest() != oid:
        raise ValueError("Git object bytes do not match their source SHA1")
    return raw


def _tree(repo, oid, cache):
    if oid not in cache:
        raw = _object(repo, oid, "tree", 2097152)
        entries = {}
        offset = 0
        while offset < len(raw):
            space = raw.find(b" ", offset)
            nul = raw.find(b"\0", space + 1)
            if space < offset or nul < space or nul + 21 > len(raw):
                raise ValueError("malformed Git source tree")
            mode, name = raw[offset:space], raw[space + 1:nul]
            if not name or b"/" in name or name in entries or name in (b".", b".."):
                raise ValueError("ambiguous Git source tree entry")
            entries[name] = (mode, raw[nul + 1:nul + 21].hex())
            offset = nul + 21
        cache[oid] = entries
    return cache[oid]


def _source_blob(repo, root_tree, path, cache):
    tree_oid = root_tree
    parts = path.encode("ascii").split(b"/")
    for position, part in enumerate(parts):
        entry = _tree(repo, tree_oid, cache).get(part)
        if entry is None:
            return None
        mode, oid = entry
        if position < len(parts) - 1:
            if mode != b"40000":
                raise ValueError("allowlisted parent must have regular Git tree mode")
            tree_oid = oid
        else:
            if mode != b"100644":
                raise ValueError("allowlisted source must have regular Git mode 100644")
            return oid, _object(repo, oid, "blob", MAX_SOURCE_BYTES)


def _vector(text):
    counts = Counter()
    for token in re.findall(r"[a-z0-9]+", text.lower(), flags=re.ASCII):
        dimension = int.from_bytes(hashlib.sha256(token.encode("ascii")).digest()[:4], "big") % 256
        counts[dimension] += 1
    return [[dimension, counts[dimension]] for dimension in sorted(counts)]


def _json_bytes(data):
    return (json.dumps(data, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("ascii")


def build_index(repo, commit):
    """Fully rebuild from verified commit/tree/blob bytes; absent paths disappear."""
    _validate_commit(commit)
    repo = _repository_root(repo)
    commit_raw = _object(repo, commit, "commit", 65536)
    first_line = commit_raw.split(b"\n", 1)[0]
    if not re.fullmatch(rb"tree [0-9a-f]{40}", first_line):
        raise ValueError("commit has no valid source tree")
    root_tree = first_line[5:].decode("ascii")
    documents, chunks, cache = [], [], {}
    total = 0
    for path in ALLOWLIST:
        source = _source_blob(repo, root_tree, path, cache)
        if source is None:
            continue
        blob, raw = source
        total += len(raw)
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("combined source total exceeds byte limit")
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("allowlisted source must be UTF-8") from exc
        documents.append({"path": path, "blob_sha1": blob,
                          "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
        for start in range(0, len(text), CHUNK_CHARS):
            end = min(start + CHUNK_CHARS, len(text))
            excerpt = text[start:end]
            start_line = text.count("\n", 0, start) + 1
            end_line = text.count("\n", 0, end - 1) + 1
            identity = "\0".join((REPOSITORY, path, blob, str(start), str(end)))
            chunks.append({"id": hashlib.sha256(identity.encode("ascii")).hexdigest(),
                           "path": path, "start_char": start, "end_char": end,
                           "start_line": start_line, "end_line": end_line, "text": excerpt,
                           "source_url": "https://github.com/" + REPOSITORY + "/blob/" + commit + "/" +
                                         path + "#L" + str(start_line) + "-L" + str(end_line),
                           "vector": _vector(excerpt)})
            if len(chunks) > MAX_CHUNKS:
                raise ValueError("index chunk count exceeds limit")
    raw = _json_bytes({"schema_version": 1, "source": {"repository": REPOSITORY, "commit": commit},
                       "algorithm": ALGORITHM, "documents": documents, "chunks": chunks})
    if len(raw) > MAX_INDEX_BYTES:
        raise ValueError("serialized index payload exceeds byte limit")
    return raw


def check_index(repo, raw, commit):
    """Require exact current-source reconstruction, including every text/vector field."""
    _validate_commit(commit)
    if not isinstance(raw, bytes) or len(raw) > MAX_INDEX_BYTES:
        raise ValueError("index payload exceeds byte limit or is not bytes")
    if raw != build_index(repo, commit):
        raise ValueError("index is stale, noncanonical or does not match the expected source commit")
    return json.loads(raw)


def search_index(index, query, limit=5):
    """Cosine lexical retrieval; callers must first check_index at an expected commit."""
    if not isinstance(query, str) or len(query) > 4096:
        raise ValueError("query exceeds 4096-character limit")
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("search result limit must be 1..20")
    query_vector = dict(_vector(query))
    query_norm = math.sqrt(sum(v * v for v in query_vector.values()))
    if not query_norm:
        return []
    results = []
    for chunk in index["chunks"]:
        vector = dict(chunk["vector"])
        dot = sum(query_vector.get(dimension, 0) * count for dimension, count in vector.items())
        if dot:
            score = dot / (query_norm * math.sqrt(sum(v * v for v in vector.values())))
            results.append({"id": chunk["id"], "path": chunk["path"], "start_char": chunk["start_char"],
                            "end_char": chunk["end_char"], "source_url": chunk["source_url"],
                            "score": score, "text": chunk["text"]})
    results.sort(key=lambda result: (-result["score"], result["path"], result["start_char"], result["id"]))
    return results[:limit]


def _open_parent(path):
    """Open each directory with O_NOFOLLOW, using directory descriptors throughout."""
    path = Path(os.path.abspath(path))
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parent.parts[1:]:
            next_descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                      dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return path, descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ValueError("output parent must be existing directories without symlinks") from exc


def write_output(repo, output, raw):
    """Atomic external-file replacement; never follow a destination symlink."""
    root = _repository_root(repo)
    path = Path(os.path.abspath(output))
    resolved = path.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError("index output must be outside the source repository")
    if len(raw) > MAX_INDEX_BYTES:
        raise ValueError("index payload exceeds byte limit")
    path, directory = _open_parent(path)
    temporary = ".public-vector-index-" + secrets.token_hex(12)
    created = False
    try:
        try:
            existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("output must be a regular file, never a symlink")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=directory)
        created = True
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
    finally:
        if created:
            try:
                os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError:
                pass
        os.close(directory)


def frame_index(raw):
    """Frame checked index bytes for temporary logs; this protocol is not authentication."""
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_INDEX_BYTES:
        raise ValueError("framed payload must be 1..131072 bytes")
    digest = hashlib.sha256(raw).hexdigest()
    count = (len(raw) + FRAME_BYTES - 1) // FRAME_BYTES
    lines = ["{} BEGIN {} {} {}".format(MARKER, len(raw), digest, count)]
    for number, offset in enumerate(range(0, len(raw), FRAME_BYTES), 1):
        lines.append("{} CHUNK {} {}".format(MARKER, number,
                     base64.b64encode(raw[offset:offset + FRAME_BYTES]).decode("ascii")))
    lines.append("{} END {}".format(MARKER, digest))
    return "\n".join(lines) + "\n"


def reconstruct_index(log):
    """Reject incomplete, duplicated, reordered, corrupt or oversized log framing."""
    if not isinstance(log, str) or len(log.encode("utf-8")) > MAX_LOG_BYTES:
        raise ValueError("log exceeds byte limit")
    lines = []
    for line in log.splitlines():
        line = TIMESTAMP.sub("", line, count=1) if TIMESTAMP.match(line) else line
        if line.startswith(MARKER):
            lines.append(line)
    if len(lines) < 3:
        raise ValueError("missing complete index log frame")
    begin = re.fullmatch(MARKER + r" BEGIN ([1-9][0-9]{0,5}) ([0-9a-f]{64}) ([1-9][0-9]{0,2})", lines[0])
    if begin is None:
        raise ValueError("invalid or missing log frame header")
    size, digest, count = int(begin[1]), begin[2], int(begin[3])
    if size > MAX_INDEX_BYTES or count != (size + FRAME_BYTES - 1) // FRAME_BYTES:
        raise ValueError("declared log payload exceeds limit or has inconsistent chunk count")
    if len(lines) != count + 2 or lines[-1] != MARKER + " END " + digest:
        raise ValueError("incomplete, duplicate or missing terminal log frame")
    payload = bytearray()
    for number, line in enumerate(lines[1:-1], 1):
        match = re.fullmatch(MARKER + r" CHUNK ([1-9][0-9]{0,2}) ([A-Za-z0-9+/=]{1,1024})", line)
        if match is None or int(match[1]) != number:
            raise ValueError("missing, duplicated or reordered log chunk")
        try:
            piece = base64.b64decode(match[2], validate=True)
        except ValueError as exc:
            raise ValueError("invalid base64 log chunk") from exc
        expected = FRAME_BYTES if number < count else size - FRAME_BYTES * (count - 1)
        if len(piece) != expected or base64.b64encode(piece).decode("ascii") != match[2]:
            raise ValueError("invalid log chunk length or encoding")
        payload.extend(piece)
    raw = bytes(payload)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("reconstructed log payload digest or byte count mismatch")
    return raw


def _read_bounded(path, limit):
    with open(path, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("input file exceeds byte limit")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="rebuild only allowlisted committed public blobs")
    build.add_argument("--commit", required=True)
    build.add_argument("--output", required=True)
    for command in ("check", "search", "log"):
        item = commands.add_parser(command)
        item.add_argument("--index", required=True)
        if command != "log":
            item.add_argument("--commit", required=True)
        if command == "search":
            item.add_argument("--query", required=True)
            item.add_argument("--limit", type=int, default=5)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("--log", required=True)
    reconstruct.add_argument("--commit", required=True)
    reconstruct.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    repo = Path.cwd()
    try:
        if args.command == "build":
            raw = build_index(repo, args.commit)
            write_output(repo, args.output, raw)
            print("public lexical index: {} bytes; SHA256 {}".format(len(raw), hashlib.sha256(raw).hexdigest()))
        elif args.command == "reconstruct":
            raw = reconstruct_index(_read_bounded(args.log, MAX_LOG_BYTES).decode("utf-8"))
            check_index(repo, raw, args.commit)
            write_output(repo, args.output, raw)
            print("public lexical index reconstructed and source-checked")
        else:
            raw = _read_bounded(args.index, MAX_INDEX_BYTES)
            if args.command == "log":
                data = json.loads(raw)
                commit = data["source"]["commit"]
                check_index(repo, raw, commit)
                sys.stdout.write(frame_index(raw))
            else:
                data = check_index(repo, raw, args.commit)
                if args.command == "check":
                    print("public lexical index matches expected source commit " + args.commit)
                else:
                    sys.stdout.buffer.write(_json_bytes(search_index(data, args.query, args.limit)))
    except (ValueError, OSError, KeyError, TypeError, UnicodeError) as exc:
        print("public lexical index refused: " + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
