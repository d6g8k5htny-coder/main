#!/usr/bin/env python3
"""Bounded source inventory and lexical review, never a Layer A certifier.

Only a manifest-pinned, local, POSIX byte snapshot is examined. No Lean parser,
elaborator, kernel, tactic evaluator, dependency closure, or semantic checker is
run. Every report leaves G1 and CBZ-0 NOT_ESTABLISHED, including a scan without
lexical findings. Exit 1 means review findings; exit 2 means not established.
"""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata


MAX_SOURCE_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_TREE_ENTRIES = 200000
MAX_DIRECTORY_DEPTH = 128
MAX_FINDINGS = 10000

# A missing entry is a loss of advertised detection, covered by scanner mutants.
# These are review triggers, not claims that every occurrence is unsound.
REVIEW_TOKENS = {
    "sorry": ("PROHIBITED_TOKEN", "Logical placeholder requires review."),
    "admit": ("PROHIBITED_TOKEN", "Logical placeholder requires review."),
    "axiom": ("PROHIBITED_TOKEN", "Axiom declaration requires provenance and verification."),
    "native_decide": ("UNCERTIFIED_EXECUTION", "Native evaluation is not certified by this scanner."),
    "unsafe": ("UNSAFE_DECLARATION", "Unsafe declaration or use requires review."),
    "Float": ("FLOAT_USE", "Floating point cannot certify a bound in this audit."),
    "Float32": ("FLOAT_USE", "Floating point cannot certify a bound in this audit."),
    "run_tac": ("UNCERTIFIED_EXECUTION", "Metaprogram execution requires separate evidence."),
    "run_elab": ("UNCERTIFIED_EXECUTION", "Elaborator execution requires separate evidence."),
    "eval_expr": ("UNCERTIFIED_EXECUTION", "Expression evaluation requires separate evidence."),
    "implemented_by": ("UNCERTIFIED_EXECUTION", "Implementation replacement requires separate evidence."),
    "extern": ("UNCERTIFIED_EXECUTION", "External execution requires separate evidence."),
    "macro": ("EXTENSIBLE_SYNTAX", "Macro expansion is outside the lexical contract."),
    "macro_rules": ("EXTENSIBLE_SYNTAX", "Macro expansion is outside the lexical contract."),
    "syntax": ("EXTENSIBLE_SYNTAX", "Syntax extension is outside the lexical contract."),
    "elab": ("EXTENSIBLE_SYNTAX", "Custom elaboration is outside the lexical contract."),
    "elab_rules": ("EXTENSIBLE_SYNTAX", "Custom elaboration is outside the lexical contract."),
    "declare_syntax_cat": ("EXTENSIBLE_SYNTAX", "Custom syntax categories require separate analysis."),
    "initialize": ("EXTENSIBLE_SYNTAX", "Initialization may change the elaboration environment."),
    "builtin_initialize": ("EXTENSIBLE_SYNTAX", "Initialization may change the elaboration environment."),
}

BLOCKING_COVERAGE = (
    ("A1", "Kernel-checked closure without placeholders or stubbed obligations.",
     "Pinned Lean dependency closure, elaboration and kernel evidence, plus obligation review."),
    ("A2", "Absence of newly introduced unverified axioms.",
     "Pinned declaration/dependency axiom analysis and approved baseline comparison."),
    ("A3", "Certified exact bounds and their mathematical semantics.",
     "Exact mathematical specifications and kernel-checked bound certificates; no float certification."),
    ("A4", "Absence of weakened claims, bounds or tests.",
     "Trusted prior baseline, semantic statement/test comparison and independently justified tests."),
    ("A5", "Absence of improperly deleted obligations or research.",
     "Trusted prior inventory and preserved-history/deletion comparison across the federation."),
    ("A6", "Nonvacuity of hypotheses, witnesses and remaining obligations.",
     "Independent satisfiability/witness and obligation-completeness evidence."),
    ("A7", "Truth of status, gate and completion assertions.",
     "Current pinned gate receipts and a status-to-evidence reconciliation."),
)


class AuditFailure(Exception):
    def __init__(self, code, path, detail, **evidence):
        super().__init__(detail)
        self.error = {"code": code, "path": str(path), "detail": detail, **evidence}


def io_failure(error, path):
    if error.errno in (errno.ELOOP,):
        return AuditFailure("SYMLINK", path, "Symbolic link traversal is prohibited.")
    if error.errno in (errno.ENOENT,):
        return AuditFailure("MISSING_FILE", path, "Required filesystem entry is missing.")
    if error.errno in (errno.EACCES, errno.EPERM):
        return AuditFailure("UNREADABLE", path, "Filesystem access was denied.")
    return AuditFailure("IO_ERROR", path, f"Filesystem operation failed (errno {error.errno}).")


def check_entry(st, path, directory=False):
    if stat.S_ISLNK(st.st_mode):
        raise AuditFailure("SYMLINK", path, "Symbolic links are prohibited, including ancestors.")
    if directory and not stat.S_ISDIR(st.st_mode):
        raise AuditFailure("NOT_DIRECTORY", path, "Expected a real directory.")
    if not directory and not stat.S_ISREG(st.st_mode):
        raise AuditFailure("NONREGULAR_FILE", path, "Expected a regular source/manifest file.")
    mode = stat.S_IMODE(st.st_mode)
    if not mode & 0o444 or (directory and not mode & 0o111):
        raise AuditFailure("UNREADABLE", path, "Entry has no read permission bits or directory traversal bits.")


def absolute_input_path(value):
    path = Path(value)
    if ".." in path.parts:
        raise AuditFailure("INVALID_PATH", value, "Parent traversal in input paths is prohibited.")
    return path if path.is_absolute() else Path.cwd() / path


def open_directory(path):
    """Open each ancestor with no-follow semantics; return an owned descriptor."""
    path = absolute_input_path(path)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise AuditFailure("UNSUPPORTED_PLATFORM", path, "POSIX no-follow directory descriptors are required.")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        check_entry(os.fstat(fd), "/", directory=True)
        for index, part in enumerate(path.parts[1:], 1):
            display = str(Path(*path.parts[:index + 1]))
            st = os.stat(part, dir_fd=fd, follow_symlinks=False)
            check_entry(st, display, directory=True)
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
            check_entry(os.fstat(fd), display, directory=True)
        return fd
    except OSError as error:
        os.close(fd)
        raise io_failure(error, path) from error
    except BaseException:
        os.close(fd)
        raise


def read_child(parent_fd, name, display, maximum):
    """Read one regular file once, rejecting links, mode denial and observed races."""
    fd = None
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        check_entry(before, display)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
        opened = os.fstat(fd)
        check_entry(opened, display)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise AuditFailure("UNSTABLE_INPUT", display, "File identity changed while opening.")
        if opened.st_size > maximum:
            raise AuditFailure("RESOURCE_LIMIT", display, f"File exceeds the {maximum}-byte limit.")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise AuditFailure("RESOURCE_LIMIT", display, f"Read exceeds the {maximum}-byte limit.")
        after = os.fstat(fd)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(opened, key) != getattr(after, key) for key in keys) or total != after.st_size:
            raise AuditFailure("UNSTABLE_INPUT", display, "File changed during the read.")
        return b"".join(chunks)
    except OSError as error:
        raise io_failure(error, display) from error
    finally:
        if fd is not None:
            os.close(fd)


def read_absolute(path, maximum):
    path = absolute_input_path(path)
    fd = open_directory(path.parent)
    try:
        return read_child(fd, path.name, path, maximum)
    finally:
        os.close(fd)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def reject_noninteger(value):
    raise ValueError("Non-integer JSON numbers/constants are unsupported.")


def parse_manifest(data, manifest_path):
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AuditFailure("INVALID_UTF8", manifest_path, "Manifest is not strict UTF-8.",
                           byte_offset=error.start) from error
    try:
        document = json.loads(text, object_pairs_hook=unique_object,
                              parse_float=reject_noninteger, parse_constant=reject_noninteger)
        if not isinstance(document, dict) or set(document) != {"version", "root", "files"}:
            raise ValueError("Expected exactly version, root and files keys.")
        if type(document["version"]) is not int or document["version"] != 1:
            raise ValueError("Only integer manifest version 1 is supported.")
        root = document["root"]
        if not isinstance(root, str) or not root or "\x00" in root or ".." in Path(root).parts:
            raise ValueError("Root must be a nonempty path without parent traversal.")
        root.encode("utf-8", errors="strict")
        entries = document["files"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("An explicit nonempty file list is required.")
        paths = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError("Each file requires exactly path and sha256 keys.")
            name = entry["path"]
            if (not isinstance(name, str) or not name or name.startswith("/") or "\\" in name
                    or "\x00" in name or any(part in ("", ".", "..") for part in name.split("/"))
                    or not name.endswith(".lean")):
                raise ValueError("Every file must be a canonical relative .lean path.")
            name.encode("utf-8", errors="strict")
            if name in paths:
                raise ValueError("Duplicate file path.")
            paths.add(name)
            digest = entry["sha256"]
            if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
                raise ValueError("Each file must pin a lowercase 64-digit SHA-256.")
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = manifest_path.parent / root_path
        return absolute_input_path(root_path), sorted(entries, key=lambda entry: entry["path"])
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise AuditFailure("INVALID_MANIFEST", manifest_path, str(error)) from error


def discover_sources(root_fd):
    """Reconcile the full local tree, never following any symbolic link."""
    discovered = set()
    errors = []
    count = 0

    def visit(fd, prefix, depth):
        nonlocal count
        if depth > MAX_DIRECTORY_DEPTH:
            raise AuditFailure("RESOURCE_LIMIT", prefix, "Directory depth limit exceeded.")
        try:
            with os.scandir(fd) as iterator:
                names = []
                for entry in iterator:
                    count += 1
                    if count > MAX_TREE_ENTRIES:
                        raise AuditFailure("RESOURCE_LIMIT", prefix,
                                           "Directory entry limit exceeded during enumeration.")
                    names.append(entry.name)
                names.sort()
            for name in names:
                display = prefix + name
                try:
                    name.encode("utf-8", errors="strict")
                    entry = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    if stat.S_ISLNK(entry.st_mode):
                        raise AuditFailure("SYMLINK", display, "Tree contains a symbolic link; coverage is incomplete.")
                    if stat.S_ISDIR(entry.st_mode):
                        check_entry(entry, display, directory=True)
                        child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        try:
                            visit(child, display + "/", depth + 1)
                        finally:
                            os.close(child)
                    elif name.endswith(".lean"):
                        discovered.add(display)
                        check_entry(entry, display)
                    elif not stat.S_ISREG(entry.st_mode):
                        raise AuditFailure("NONREGULAR_FILE", display, "Tree contains a nonregular entry.")
                except AuditFailure as failure:
                    if failure.error["code"] == "RESOURCE_LIMIT":
                        raise
                    errors.append(failure.error)
                except UnicodeError:
                    errors.append(AuditFailure("INVALID_UTF8", repr(display), "Filename is not valid UTF-8.").error)
                except OSError as error:
                    errors.append(io_failure(error, display).error)
        except OSError as error:
            raise io_failure(error, prefix) from error

    visit(root_fd, "", 0)
    return discovered, errors


def read_relative(root_fd, name):
    parts = name.split("/")
    parent_fd = os.dup(root_fd)
    try:
        for index, part in enumerate(parts[:-1], 1):
            display = "/".join(parts[:index])
            entry = os.stat(part, dir_fd=parent_fd, follow_symlinks=False)
            check_entry(entry, display, directory=True)
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child
        return read_child(parent_fd, parts[-1], name, MAX_SOURCE_BYTES)
    except OSError as error:
        raise io_failure(error, name) from error
    finally:
        os.close(parent_fd)


def identifier_start(char):
    return char == "_" or unicodedata.category(char).startswith("L") or unicodedata.category(char) == "Nl"


def identifier_rest(char):
    return identifier_start(char) or char.isdigit() or unicodedata.category(char) in ("Mn", "Mc", "No") or char in "'!?"


class Lexer:
    """Lexical masks for a bounded subset; unsupported constructs cause review."""

    def __init__(self, text, path):
        self.text, self.path = text, path
        self.i, self.line, self.column, self.byte_offset = 0, 1, 1, 0
        self.lines = text.split("\n")
        self.findings = []

    def advance(self, end):
        for char in self.text[self.i:end]:
            self.byte_offset += len(char.encode("utf-8"))
            if char == "\n":
                self.line, self.column = self.line + 1, 1
            else:
                self.column += 1
        self.i = end

    def evidence(self, token):
        line = self.lines[self.line - 1].removesuffix("\r")
        start = max(0, self.column - 1 - 256) if len(line) > 1024 else 0
        return {
            "path": self.path, "line": self.line, "column": self.column,
            "byte_offset": self.byte_offset, "byte_length": len(token.encode("utf-8")),
            "token": token, "source_line": line[start:start + 1024],
            "source_line_start_column": start + 1, "source_line_truncated": len(line) > 1024,
        }

    def finding(self, code, token, detail):
        if len(self.findings) >= MAX_FINDINGS:
            self.fail("Finding count limit exceeded.", code="RESOURCE_LIMIT")
        self.findings.append({"code": code, "detail": detail, **self.evidence(token)})

    def fail(self, detail, code="LEXICAL_ERROR"):
        raise AuditFailure(code, self.path, detail, line=self.line,
                           column=self.column, byte_offset=self.byte_offset)

    def block_comment(self):
        depth = 1
        self.advance(self.i + 2)
        while self.i < len(self.text):
            if self.text.startswith("/-", self.i):
                depth += 1
                self.advance(self.i + 2)
            elif self.text.startswith("-/", self.i):
                depth -= 1
                self.advance(self.i + 2)
                if depth == 0:
                    return
            else:
                self.advance(self.i + 1)
        self.fail("Unterminated nested block comment.")

    def escape(self):
        if self.i + 1 >= len(self.text):
            self.fail("Unterminated string/character escape.")
        char = self.text[self.i + 1]
        end = self.i + 2
        recognized = char in "\\\"'nrt0"
        if char in "xu":
            size = 2 if char == "x" else 4
            digits = self.text[end:end + size]
            if len(digits) == size and all(c in "0123456789abcdefABCDEF" for c in digits):
                recognized, end = True, end + size
            elif char == "u" and self.text.startswith("{", end):
                close = self.text.find("}", end + 1, end + 9)
                digits = self.text[end + 1:close] if close != -1 else ""
                if (1 <= len(digits) <= 6 and all(c in "0123456789abcdefABCDEF" for c in digits)
                        and int(digits, 16) <= 0x10ffff and not 0xd800 <= int(digits, 16) <= 0xdfff):
                    recognized, end = True, close + 1
        if not recognized:
            self.finding("UNSUPPORTED_ESCAPE", self.text[self.i:end],
                         "Escape form is outside the bounded literal mask.")
        self.advance(end)

    def string(self):
        self.advance(self.i + 1)
        while self.i < len(self.text):
            if self.text[self.i] == '"':
                self.advance(self.i + 1)
                return
            if self.text[self.i] == "\\":
                self.escape()
            else:
                self.advance(self.i + 1)
        self.fail("Unterminated string literal.")

    def raw_string(self):
        end = self.i + 1
        while end < len(self.text) and self.text[end] == "#":
            end += 1
        if end >= len(self.text) or self.text[end] != '"':
            return False
        delimiter = '"' + "#" * (end - self.i - 1)
        close = self.text.find(delimiter, end + 1)
        if close == -1:
            self.fail("Unterminated raw string literal.")
        self.advance(close + len(delimiter))
        return True

    def character(self):
        self.advance(self.i + 1)
        if self.i >= len(self.text) or self.text[self.i] in "'\r\n":
            self.fail("Malformed character literal.")
        if self.text[self.i] == "\\":
            self.escape()
        else:
            self.advance(self.i + 1)
        if self.i >= len(self.text) or self.text[self.i] != "'":
            self.fail("Character literal must contain exactly one character or escape.")
        self.advance(self.i + 1)

    def scan(self):
        while self.i < len(self.text):
            char = self.text[self.i]
            if char in " \t\r\n":
                self.advance(self.i + 1)
            elif self.text.startswith("--", self.i):
                end = self.text.find("\n", self.i + 2)
                self.advance(len(self.text) if end == -1 else end)
            elif self.text.startswith("/-", self.i):
                self.block_comment()
            elif char == '"':
                self.string()
            elif char == "r" and self.raw_string():
                continue
            elif char == "'":
                self.character()
            elif char == "«":
                end = self.text.find("»", self.i + 1)
                if end == -1:
                    self.fail("Unterminated quoted identifier.")
                self.finding("QUOTED_IDENTIFIER", self.text[self.i:end + 1],
                             "Quoted identifier is preserved and requires separate name resolution.")
                self.advance(end + 1)
            elif identifier_start(char):
                end = self.i + 1
                while end < len(self.text) and identifier_rest(self.text[end]):
                    end += 1
                token = self.text[self.i:end]
                if token.endswith("!") and end < len(self.text) and self.text[end] == '"':
                    self.finding("STRING_INTERPOLATION", token,
                                 "String macro/interpolation expansion is outside lexical coverage.")
                if token in REVIEW_TOKENS:
                    code, detail = REVIEW_TOKENS[token]
                    self.finding(code, token, detail)
                if not token.isascii():
                    self.finding("UNSUPPORTED_SYNTAX", token,
                                 "Unicode identifier syntax requires Lean parsing; token is preserved.")
                self.advance(end)
            elif char.isascii() and (char.isdigit() or char in "()[]{}.,:;=+-*/<>|&!%^~?@"):
                self.advance(self.i + 1)
            else:
                self.finding("UNSUPPORTED_SYNTAX", char,
                             "Unknown, quotation, antiquotation or command syntax requires review.")
                self.advance(self.i + 1)
        return self.findings


def empty_report():
    return {
        "tool": "cbz_source_audit", "schema_version": 1,
        "scope": "Pinned local source inventory and bounded lexical review only; no Layer A certification.",
        "aggregate_status": "NOT_ESTABLISHED",
        "gate_claims": {"G1": "NOT_ESTABLISHED", "CBZ-0": "NOT_ESTABLISHED"},
        "input_integrity": {"status": "NOT_ESTABLISHED", "expected_files": 0,
                            "scanned_files": 0, "discovered_files": [], "unlisted_files": []},
        "coverage": {"lean_grammar_and_elaboration": "NOT_ESTABLISHED",
                     "kernel_checking": "NOT_ESTABLISHED", "dependency_closure": "NOT_ESTABLISHED",
                     "trusted_baseline_comparison": "NOT_ESTABLISHED",
                     "concurrent_tree_snapshot_atomicity": "NOT_ESTABLISHED"},
        "blocking_coverage": [{"invariant": invariant, "status": "NOT_ESTABLISHED",
                               "claim": claim, "required_evidence": evidence}
                              for invariant, claim, evidence in BLOCKING_COVERAGE],
        "files": [], "findings": [], "errors": [],
    }


def audit_manifest(manifest):
    report = empty_report()
    root_fd = None
    try:
        manifest_path = absolute_input_path(manifest)
        data = read_absolute(manifest_path, MAX_MANIFEST_BYTES)
        report["manifest_sha256"] = hashlib.sha256(data).hexdigest()
        root, entries = parse_manifest(data, manifest_path)
        report["root"] = str(root)
        report["input_integrity"]["expected_files"] = len(entries)
        report["files"] = [{"path": entry["path"], "expected_sha256": entry["sha256"], "status": "ERROR"}
                           for entry in entries]
        root_fd = open_directory(root)
        discovered, errors = discover_sources(root_fd)
        report["errors"].extend(errors)
        report["input_integrity"]["discovered_files"] = sorted(discovered)
        expected = {entry["path"] for entry in entries}
        unlisted = sorted(discovered - expected)
        report["input_integrity"]["unlisted_files"] = unlisted
        for path in unlisted:
            report["errors"].append(AuditFailure("UNLISTED_FILE", path,
                                                "Discovered .lean source is absent from the pinned manifest.").error)
        for entry, record in zip(entries, report["files"]):
            path = entry["path"]
            if path not in discovered:
                report["errors"].append(AuditFailure("MISSING_FILE", path,
                                                    "Pinned file was not discovered; it is not skipped.").error)
                continue
            try:
                source = read_relative(root_fd, path)
                record["sha256"] = hashlib.sha256(source).hexdigest()
                record["bytes"] = len(source)
                if record["sha256"] != entry["sha256"]:
                    raise AuditFailure("HASH_MISMATCH", path, "Source bytes differ from the pinned manifest.")
                try:
                    text = source.decode("utf-8", errors="strict")
                except UnicodeDecodeError as error:
                    raise AuditFailure("INVALID_UTF8", path, "Source is not strict UTF-8.",
                                       byte_offset=error.start) from error
                lexer = Lexer(text, path)
                try:
                    lexer.scan()
                finally:
                    report["findings"].extend(lexer.findings)
                record["status"] = "SCANNED"
                report["input_integrity"]["scanned_files"] += 1
            except AuditFailure as failure:
                report["errors"].append(failure.error)
    except AuditFailure as failure:
        report["errors"].append(failure.error)
    except (OSError, ValueError, RecursionError) as error:
        report["errors"].append(AuditFailure("AUDIT_ERROR", str(manifest),
                                            f"Audit could not complete ({type(error).__name__}).").error)
    finally:
        if root_fd is not None:
            os.close(root_fd)
    report["errors"].sort(key=lambda item: (item["path"], item["code"], item["detail"]))
    report["findings"].sort(key=lambda item: (item["path"], item["byte_offset"], item["code"]))
    if not report["errors"]:
        report["input_integrity"]["status"] = "CHECKED"
        if report["findings"]:
            report["aggregate_status"] = "REVIEW_REQUIRED"
    return report


def render_report(report):
    return json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n"


def write_output(path, report):
    """Create new evidence outside the source tree; never overwrite an input."""
    path = absolute_input_path(path)
    if report["root"] is not None and path.is_relative_to(Path(report["root"])):
        raise AuditFailure("OUTPUT_ERROR", path, "Report output must be outside the audited source root.")
    parent_fd = open_directory(path.parent)
    fd = None
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent_fd)
        check_entry(os.fstat(fd), path)
        output = render_report(report).encode("utf-8")
        with os.fdopen(fd, "wb") as target:
            fd = None
            target.write(output)
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path,
                        help="Create a new report outside the source root; existing paths are never overwritten.")
    args = parser.parse_args(argv)
    report = audit_manifest(args.manifest)
    if args.output is not None:
        try:
            write_output(args.output, report)
        except (OSError, AuditFailure) as error:
            report["aggregate_status"] = "NOT_ESTABLISHED"
            report["errors"].append(AuditFailure("OUTPUT_ERROR", args.output,
                                                f"Could not retain a new report ({type(error).__name__}).").error)
    sys.stdout.write(render_report(report))
    return 1 if report["aggregate_status"] == "REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    sys.exit(main())
