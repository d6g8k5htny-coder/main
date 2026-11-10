"""Private, bounded syntax diagnostics for the fixed formal trace vocabulary.

This module only consumes bytes. It does not read paths, execute trace text,
authenticate input, decide admission, or alter any original R1 outcome.
"""
from __future__ import annotations

import hashlib
import json
import re

_TRACE_LIMIT = 128 * 1024 * 1024
_REPORT_LIMIT = 4 * 1024 * 1024
_RECORD_LIMIT = 4096
_LINE_LIMIT = 16384
# Exact TRACE_CALLS vocabulary from original worker6199d4f8081dd37fb0e8d32a5f07e6277cc29147.
_CALLS = frozenset((
    "open", "openat", "openat2", "creat", "rename", "renameat", "renameat2",
    "unlink", "unlinkat", "mkdir", "mkdirat", "rmdir", "link", "linkat",
    "symlink", "symlinkat", "truncate", "ftruncate", "chmod", "fchmod",
    "fchmodat", "chown", "fchown", "fchownat", "lchown", "utime", "utimes",
    "utimensat", "chdir", "fchdir",
))
_OPENS = frozenset(("open", "openat", "openat2"))
_CONTEXT = frozenset(("chdir", "fchdir"))
_MUTATORS = _CALLS - _OPENS - _CONTEXT
_WRITE_FLAGS = frozenset(("O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND"))
_FLAGS = _WRITE_FLAGS | frozenset((
    "O_RDONLY", "O_CLOEXEC", "O_LARGEFILE", "O_NOFOLLOW", "O_NONBLOCK",
    "O_NDELAY", "O_DIRECTORY", "O_NOCTTY", "O_SYNC", "O_DSYNC", "O_RSYNC",
    "O_DIRECT", "O_NOATIME", "O_ASYNC", "O_PATH", "O_EXCL",
))
_HEADER = re.compile(r"^(?:\[pid\s+)?([0-9]+)\]?\s+[0-9]+\.[0-9]+\s+(.*)$")
_ENTER = re.compile(r"^([a-z][a-z0-9_]*)\(")
_RESUME = re.compile(r"^<\.\.\. ([a-z][a-z0-9_]*) resumed>(.*)$")
_UNKNOWN_CWD = object()


class _Problem(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _integer(text):
    try:
        return int(text)
    except (ValueError, OverflowError):
        raise _Problem("UNSUPPORTED_SYNTAX") from None


def _line(raw, number, offset):
    return {"line": number, "offset": offset, "bytes": len(raw),
            "sha256": _sha(raw), "raw_utf8": raw.decode("utf-8")}


def _cstring(token):
    if len(token) < 2 or token[0] != '"' or token[-1] != '"':
        raise _Problem("UNSUPPORTED_SYNTAX")
    escapes = {"a": 7, "b": 8, "f": 12, "n": 10, "r": 13,
               "t": 9, "v": 11, "\\": 92, '"': 34}
    output, i = bytearray(), 1
    while i < len(token) - 1:
        char = token[i]
        if char == '"':
            raise _Problem("UNSUPPORTED_SYNTAX")
        if char != "\\":
            output.extend(char.encode("utf-8"))
            i += 1
            continue
        i += 1
        if i >= len(token) - 1:
            raise _Problem("UNSUPPORTED_ESCAPE")
        char = token[i]
        if char in escapes:
            output.append(escapes[char])
            i += 1
        elif char in "01234567":
            end = i
            while end < min(i + 3, len(token) - 1) and token[end] in "01234567":
                end += 1
            value = int(token[i:end], 8)
            if value > 255:
                raise _Problem("UNSUPPORTED_ESCAPE")
            output.append(value)
            i = end
        else:
            raise _Problem("UNSUPPORTED_ESCAPE")
    if 0 in output:
        raise _Problem("UNSUPPORTED_ESCAPE")
    try:
        return bytes(output).decode("utf-8")
    except UnicodeError:
        raise _Problem("UNSUPPORTED_ESCAPE") from None


def _scan(text, initial=()):
    """Yield top-level separators; FD-angle contents are not argument syntax."""
    stack, angle, quoted, escaped = list(initial), 0, False, False
    pairs = {")": "(", "]": "[", "}": "{"}
    for position, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if angle:
            if char == "<":
                angle += 1
            elif char == ">":
                angle -= 1
            continue
        if char == '"':
            quoted = True
        elif char == "<":
            angle = 1
        elif char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != pairs[char]:
                raise _Problem("UNSUPPORTED_SYNTAX")
            stack.pop()
            if initial and not stack:
                yield position, "close"
                return
        elif char == "," and not stack:
            yield position, "comma"
    if quoted or escaped or angle or stack:
        raise _Problem("UNSUPPORTED_SYNTAX")


def _arguments(text):
    if not text.strip():
        return []
    commas = [position for position, kind in _scan(text) if kind == "comma"]
    answer, start = [], 0
    for end in commas + [len(text)]:
        token = text[start:end].strip()
        if not token:
            raise _Problem("UNSUPPORTED_SYNTAX")
        answer.append(token)
        start = end + 1
    return answer


def _call(payload):
    entered = _ENTER.match(payload)
    if entered is None:
        raise _Problem("UNSUPPORTED_SYNTAX")
    start = entered.end()
    closed = next(_scan(payload[start:], ("(",)), None)
    if closed is None:
        raise _Problem("UNSUPPORTED_SYNTAX")
    end = start + closed[0]
    return entered[1], _arguments(payload[start:end]), payload[end + 1:].strip()


def _angle_end(text, start):
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "<":
            depth += 1
        elif text[i] == ">":
            depth -= 1
            if depth == 0:
                return i + 1
    raise _Problem("AMBIGUOUS_FD_CONTEXT")


def _fd(token):
    match = re.fullmatch(r"(AT_FDCWD|-?[0-9]+)(.*)", token)
    if match is None:
        raise _Problem("AMBIGUOUS_FD_CONTEXT")
    number = match[1] if match[1] == "AT_FDCWD" else _integer(match[1])
    tail = match[2]
    path, device = None, None
    if tail:
        if not tail.startswith("<") or _angle_end(tail, 0) != len(tail):
            raise _Problem("AMBIGUOUS_FD_CONTEXT")
        inner = tail[1:-1]
        if "\\" in inner or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in inner):
            # Unquoted FD decoration has no admitted escape-decoding grammar.
            raise _Problem("AMBIGUOUS_FD_CONTEXT")
        if inner.endswith(" (deleted)"):
            raise _Problem("AMBIGUOUS_FD_CONTEXT")
        typed = re.fullmatch(r"(/[^<>]+)<(char|block) ([0-9]+):([0-9]+)>", inner)
        if typed:
            path = typed[1]
            device = {"kind": typed[2], "major": _integer(typed[3]),
                      "minor": _integer(typed[4])}
        elif inner.startswith("/") and "<" not in inner and ">" not in inner:
            path = inner
        else:
            raise _Problem("AMBIGUOUS_FD_CONTEXT")
    return {"fd": number, "path": path, "device": device, "raw": token}


def _result(text):
    match = re.match(r"^=\s*(-?[0-9]+)", text)
    if match is None:
        raise _Problem("UNSUPPORTED_SYNTAX")
    number, tail, fd_token = _integer(match[1]), text[match.end():], None
    if tail.startswith("<"):
        end = _angle_end(tail, 0)
        fd_token, tail = str(number) + tail[:end], tail[end:]
    errno = None
    if number < 0:
        error = re.match(r"^\s+([A-Z][A-Z0-9_]*)\b", tail)
        if error is None:
            raise _Problem("UNSUPPORTED_SYNTAX")
        errno, tail = error[1], tail[error.end():]
        tail = tail.strip()
        if tail.startswith("("):
            # strerror decoration is not a path/flags operand.
            end = tail.rfind(")")
            if end < 0:
                raise _Problem("UNSUPPORTED_SYNTAX")
            tail = tail[end + 1:]
    if not re.fullmatch(r"\s*(?:<[0-9]+\.[0-9]+>)?\s*", tail):
        raise _Problem("UNSUPPORTED_SYNTAX")
    return {"value": number, "errno": errno}, fd_token


def _result_suffix(payload):
    """Recover a native result only outside quoted strings/FD decoration."""
    quoted, escaped, angle, found = False, False, 0, None
    for i, char in enumerate(payload):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif angle:
            if char == "<":
                angle += 1
            elif char == ">":
                angle -= 1
        elif char == '"':
            quoted = True
        elif char == "<":
            angle = 1
        elif char == "=" and payload[:i].rstrip().endswith(")"):
            found = i
    if found is None:
        raise _Problem("UNSUPPORTED_SYNTAX")
    return _result(payload[found:])


def _flags(token):
    names = [name.strip() for name in token.split("|")]
    if not names or any(name not in _FLAGS for name in names):
        raise _Problem("UNSUPPORTED_FLAGS")
    if len(set(names)) != len(names):
        raise _Problem("UNSUPPORTED_FLAGS")
    access = set(names) & {"O_RDONLY", "O_WRONLY", "O_RDWR"}
    if len(access) > 1:
        raise _Problem("UNSUPPORTED_FLAGS")
    return names


def _how_flags(token):
    if not token.startswith("{") or not token.endswith("}"):
        raise _Problem("UNSUPPORTED_SYNTAX")
    fields = {}
    for field in _arguments(token[1:-1]):
        match = re.fullmatch(r"([a-z][a-z0-9_]*)\s*=\s*(.+)", field)
        if match is None or match[1] in fields or match[1] not in {"flags", "mode", "resolve"}:
            raise _Problem("UNSUPPORTED_SYNTAX")
        fields[match[1]] = match[2]
    if "flags" not in fields:
        raise _Problem("UNSUPPORTED_FLAGS")
    return _flags(fields["flags"])


def _intent(operation):
    if operation in _MUTATORS:
        return True
    if operation in _CONTEXT:
        return False
    return None


def _event(pid, operation, source_lines):
    return {"pid": pid, "syscall": operation, "state": "UNRESOLVED",
            "write_intent": _intent(operation), "targets": [], "anchors": [],
            "returned_fd": None, "symlink_content": None, "flags": None,
            "result": {"value": None, "errno": None}, "source_lines": source_lines}


def _decode(pid, operation, payload, source_lines, cwd):
    event, reasons, update = _event(pid, operation, source_lines), [], None
    completed = False
    try:
        event["result"], fd_token = _result_suffix(payload)
    except _Problem as error:
        reasons.append(error.reason)
        fd_token = None
    try:
        name, args, tail = _call(payload)
        if name != operation or operation not in _CALLS:
            raise _Problem("UNSUPPORTED_SYNTAX")
        event["result"], fd_token = _result(tail)
        completed = True

        def arity(*sizes):
            if len(args) not in sizes:
                raise _Problem("UNSUPPORTED_SYNTAX")

        anchors = {}
        def at(index):
            if index not in anchors:
                decoded = _fd(args[index])
                base = decoded["path"]
                if decoded["device"] is not None:
                    raise _Problem("AMBIGUOUS_FD_CONTEXT")
                if decoded["fd"] == "AT_FDCWD" and base is None:
                    base = cwd.get(pid)
                anchor = {"role": "cwd" if decoded["fd"] == "AT_FDCWD" else "dirfd",
                          "argument": index, "fd": decoded["fd"],
                          "raw": args[index], "path": base}
                anchors[index] = anchor
                event["anchors"].append(anchor)
            return anchors[index]["path"]

        def path(index, role="path", dirfd=None):
            base = at(dirfd) if dirfd is not None else cwd.get(pid)
            value = _cstring(args[index])
            anchor = None
            if not value.startswith("/"):
                anchor = base
                if not value or anchor is None:
                    reasons.append("MISSING_CONTEXT")
            target = {"role": role, "argument": index, "raw": args[index],
                      "path": value, "anchor": anchor}
            event["targets"].append(target)
            return value, anchor

        if operation in _OPENS:
            if operation == "open":
                arity(2, 3)
                flag_list = _flags(args[1])
                path_index, dirfd_index = 0, None
            elif operation == "openat":
                arity(3, 4)
                flag_list = _flags(args[2])
                path_index, dirfd_index = 1, 0
            else:
                arity(4)
                flag_list = _how_flags(args[2])
                path_index, dirfd_index = 1, 0
            event["flags"] = flag_list
            event["write_intent"] = bool(set(flag_list) & _WRITE_FLAGS)
            path(path_index, dirfd=dirfd_index)
        elif operation == "creat":
            arity(2)
            path(0)
        elif operation in {"rename", "link"}:
            arity(2)
            path(0, "source")
            path(1, "destination")
        elif operation in {"renameat", "renameat2", "linkat"}:
            arity(4 if operation == "renameat" else 5)
            path(1, "source", 0)
            path(3, "destination", 2)
        elif operation in {"symlink", "symlinkat"}:
            arity(2 if operation == "symlink" else 3)
            event["symlink_content"] = _cstring(args[0])
            path(1 if operation == "symlink" else 2, "destination",
                 None if operation == "symlink" else 1)
        elif operation in {"unlink", "rmdir"}:
            arity(1)
            path(0)
        elif operation in {"mkdir", "truncate", "chmod", "utime", "utimes"}:
            arity(2)
            path(0)
        elif operation in {"chown", "lchown"}:
            arity(3)
            path(0)
        elif operation in {"unlinkat", "mkdirat", "fchmodat", "fchownat", "utimensat"}:
            arity(*( (3, 4) if operation == "fchmodat" else
                     (5,) if operation == "fchownat" else
                     (4,) if operation == "utimensat" else (3,) ))
            path(1, dirfd=0)
        elif operation in {"ftruncate", "fchmod", "fchown"}:
            arity(3 if operation == "fchown" else 2)
            decoded = _fd(args[0])
            if decoded["path"] is None:
                reasons.append("MISSING_CONTEXT")
            if decoded["device"] is not None:
                reasons.append("AMBIGUOUS_FD_CONTEXT")
            event["targets"].append({"role": "fd", "argument": 0, "raw": args[0],
                                     "path": decoded["path"], "anchor": None})
        elif operation in _CONTEXT:
            arity(1)
            if operation == "chdir":
                value = _cstring(args[0])
                base = None if value.startswith("/") else cwd.get(pid)
                if not value.startswith("/"):
                    # Do not perform filesystem/canonical or symlink resolution.
                    reasons.append("MISSING_CONTEXT")
                event["anchors"].append({"role": "cwd_change", "argument": 0,
                                         "raw": args[0], "path": value, "anchor": base})
            else:
                decoded = _fd(args[0])
                value = decoded["path"]
                if value is None:
                    reasons.append("MISSING_CONTEXT")
                if decoded["device"] is not None:
                    reasons.append("AMBIGUOUS_FD_CONTEXT")
                event["anchors"].append({"role": "cwd_change", "argument": 0,
                                         "raw": args[0], "path": value})
            if event["result"]["value"] == 0 and not reasons:
                update = value
        if operation in _OPENS or operation == "creat":
            number = event["result"]["value"]
            if number is not None and number >= 0:
                event["returned_fd"] = _fd(fd_token or str(number))
                returned = event["returned_fd"]
                if returned["device"] is not None and (
                        not event["targets"] or event["targets"][0]["path"] != returned["path"]):
                    reasons.append("AMBIGUOUS_FD_CONTEXT")
        event["state"] = "COMPLETE" if not reasons else "UNRESOLVED"
    except _Problem as error:
        reasons.append(error.reason)
    value = event["result"]["value"]
    if value is not None and value < 0 and (
            event["write_intent"] is not False or operation in _CONTEXT):
        reasons.append("NATIVE_FAILURE")
    if any(not line["raw_utf8"].endswith("\n") for line in source_lines):
        reasons.append("INCOMPLETE_LINE")
        event["state"] = "UNRESOLVED"
        update = None
    if event["state"] != "COMPLETE":
        # A successful but unresolved transition changed the real cwd. Keeping
        # the former anchor would manufacture context for later relative calls.
        update = (_UNKNOWN_CWD if operation in _CONTEXT and value == 0 else None)
    return event, sorted(set(reasons)), update, completed


class _Index:
    """Bounded live records. Retired read-only pending slots keep no payload."""
    def __init__(self, raw, line_count):
        self.input = {"bytes": len(raw), "sha256": _sha(raw), "line_count": line_count}
        self.entries, self.reasons, self.sizes, self.reason_sizes = [], [], [], []
        self.counts = {"complete_calls": 0, "read_only_calls": 0,
                       "write_intent_calls": 0, "context_calls": 0}
        self.prefix, self.partial = 0, None
        self.live = self.concerns = self.event_bytes = self.concern_bytes = self.writes = 0
        # A fixed upper envelope for the small top-level record. No event or
        # pending data is removed to obtain this reserve or a COMPLETE result.
        maximum = dict(self.counts)
        maximum.update({name: line_count for name in maximum})
        header = {"scope": "TRACE_SYNTAX_DIAGNOSTIC_ONLY",
                  "status": "COMPLETE_BOUNDED_DIAGNOSTIC", "input": self.input,
                  "indexed_prefix_bytes": len(raw), "all_lines_considered": False,
                  "partial_reason": "SERIALIZED_LIMIT", "events": [], "concerns": [],
                  "counts": maximum, "manual_review_required": True}
        self.header_bound = len(_json(header)) + 1

    def put(self, changes, counts, prefix):
        """Atomically replace/append/remove records, including all provenance."""
        staged, assigned, seen = [], [], set()
        live, concerns = self.live, self.concerns
        event_bytes, concern_bytes, writes = self.event_bytes, self.concern_bytes, self.writes
        next_slot = len(self.entries)
        for slot, event, reasons in changes:
            if slot is None:
                slot = next_slot
                next_slot += 1
                assigned.append(slot)
                old, old_reasons, old_size, old_reason_size = None, (), 0, 0
            else:
                if slot in seen or not 0 <= slot < len(self.entries):
                    raise RuntimeError("invalid internal record transaction")
                old, old_reasons = self.entries[slot], self.reasons[slot]
                old_size, old_reason_size = self.sizes[slot], self.reason_sizes[slot]
            seen.add(slot)
            why = tuple(sorted(set(reasons))) if event is not None else ()
            size = len(_json(event)) if event is not None else 0
            # Public indices are <=4095; reserving four digits permits retiring
            # a resolved read-only pending slot without rewriting later indices.
            reason_size = sum(len(_json({"event_index": 4095, "reason": reason}))
                              for reason in why)
            live += int(event is not None) - int(old is not None)
            concerns += len(why) - len(old_reasons)
            event_bytes += size - old_size
            concern_bytes += reason_size - old_reason_size
            writes += int(event is not None and event["write_intent"] is True)
            writes -= int(old is not None and old["write_intent"] is True)
            staged.append((slot, event, why, size, reason_size))
        if live > _RECORD_LIMIT:
            return False, "RECORD_LIMIT", ()
        bound = (self.header_bound + event_bytes + concern_bytes
                 + 2 * max(0, live - 1) + 2 * max(0, concerns - 1))
        if bound > _REPORT_LIMIT:
            return False, "SERIALIZED_LIMIT", ()
        for slot, event, why, size, reason_size in staged:
            if slot == len(self.entries):
                self.entries.append(event)
                self.reasons.append(why)
                self.sizes.append(size)
                self.reason_sizes.append(reason_size)
            else:
                self.entries[slot], self.reasons[slot] = event, why
                self.sizes[slot], self.reason_sizes[slot] = size, reason_size
        self.live, self.concerns = live, concerns
        self.event_bytes, self.concern_bytes, self.writes = event_bytes, concern_bytes, writes
        self.counts = dict(counts)
        self.counts["write_intent_calls"] = writes
        self.prefix = prefix
        return True, None, tuple(assigned)

    def report(self):
        events, concerns = [], []
        for slot, event in enumerate(self.entries):
            if event is None:
                continue
            index = len(events)
            events.append(event)
            concerns.extend({"event_index": index, "reason": reason}
                            for reason in self.reasons[slot])
        status = ("PARTIAL_HOLD" if self.partial else
                  "HOLD" if concerns else "COMPLETE_BOUNDED_DIAGNOSTIC")
        result = {"scope": "TRACE_SYNTAX_DIAGNOSTIC_ONLY", "status": status,
                  "input": self.input, "indexed_prefix_bytes": self.prefix,
                  "all_lines_considered": self.partial is None and self.prefix == self.input["bytes"],
                  "partial_reason": self.partial, "events": events, "concerns": concerns,
                  "counts": self.counts, "manual_review_required": True}
        if len(_json(result)) + 1 > _REPORT_LIMIT or len(events) > _RECORD_LIMIT:
            raise RuntimeError("internal bounded report invariant failed")
        return result


def _effect(counts, event, completed):
    answer = dict(counts)
    # Native completion is distinct from resolving operand/FD/CWD context.
    if completed and event["syscall"] in _CALLS:
        answer["complete_calls"] += 1
        if event["syscall"] in _OPENS and event["write_intent"] is False:
            answer["read_only_calls"] += 1
        if event["syscall"] in _CONTEXT:
            answer["context_calls"] += 1
    return answer


def _retain(event, reasons):
    return bool(reasons or event["write_intent"] is not False
                or event["syscall"] in _CONTEXT)


def _unfinished(pid, operation, line):
    event = _event(pid, operation, [line])
    reasons = ["UNFINISHED_ENTRY"]
    if operation not in _CALLS:
        reasons.append("UNSUPPORTED_SYNTAX")
    if not line["raw_utf8"].endswith("\n"):
        reasons.append("INCOMPLETE_LINE")
    return event, reasons


def _orphan(pid, operation, payload, line):
    event = _event(pid, operation, [line])
    reasons = ["ORPHAN_RESUME"]
    if operation not in _CALLS:
        reasons.append("UNSUPPORTED_SYNTAX")
    try:
        event["result"], _ = _result_suffix(payload)
    except _Problem as error:
        reasons.append(error.reason)
    value = event["result"]["value"]
    if value is not None and value < 0 and (
            event["write_intent"] is not False or operation in _CONTEXT):
        reasons.append("NATIVE_FAILURE")
    if not line["raw_utf8"].endswith("\n"):
        reasons.append("INCOMPLETE_LINE")
    return event, reasons


def _analyze_trace(raw: bytes) -> dict:
    """Return bounded syntax evidence only, with input-local byte provenance."""
    if type(raw) is not bytes:
        raise ValueError("INVALID_INPUT")
    if len(raw) > _TRACE_LIMIT:
        raise ValueError("TRACE_LIMIT")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError("INVALID_UTF8") from None
    lines = raw.splitlines(keepends=True)
    index = _Index(raw, len(lines))
    pending, ambiguous, cwd = {}, set(), {}
    offset = 0

    def conflict_changes(pid):
        changes = []
        for item in pending.get(pid, ()):
            slot = item["slot"]
            reasons = index.reasons[slot]
            if "AMBIGUOUS_PAIR" not in reasons:
                event = dict(index.entries[slot])
                event["state"] = "UNRESOLVED"
                changes.append((slot, event, tuple(reasons) + ("AMBIGUOUS_PAIR",)))
        return changes

    for number, raw_line in enumerate(lines, 1):
        if len(raw_line) > _LINE_LIMIT:
            index.partial = "LINE_LIMIT"
            break
        line = _line(raw_line, number, offset)
        next_offset = offset + len(raw_line)
        text = line["raw_utf8"].rstrip("\r\n")
        header = _HEADER.match(text)
        if header is None:
            event = _event(None, None, [line])
            reasons = ["UNSUPPORTED_SYNTAX"]
            if not raw_line.endswith(b"\n"):
                reasons.append("INCOMPLETE_LINE")
            ok, reason, _ = index.put([(None, event, reasons)], index.counts, next_offset)
            if not ok:
                index.partial = reason
                break
            offset = next_offset
            continue
        try:
            pid = _integer(header[1])
            if pid <= 0:
                raise _Problem("UNSUPPORTED_SYNTAX")
        except _Problem:
            event = _event(None, None, [line])
            ok, reason, _ = index.put([(None, event, ["UNSUPPORTED_SYNTAX"])],
                                      index.counts, next_offset)
            if not ok:
                index.partial = reason
                break
            offset = next_offset
            continue
        payload = header[2]
        if (re.fullmatch(r"--- SIG[A-Z0-9]+(?: \{.*\})? ---", payload)
                or re.fullmatch(r"\+\+\+ (?:exited with [0-9]+|killed by SIG[A-Z0-9]+) \+\+\+", payload)):
            # Known trace metadata cannot clear pending calls or invent lineage.
            changes = []
            if not raw_line.endswith(b"\n"):
                changes.append((None, _event(pid, None, [line]), ["INCOMPLETE_LINE"]))
            ok, reason, _ = index.put(changes, index.counts, next_offset)
            if not ok:
                index.partial = reason
                break
            offset = next_offset
            continue
        resumed = _RESUME.match(payload)
        entered = _ENTER.match(payload)
        if resumed is not None:
            operation, ending = resumed[1], resumed[2]
            waiting = pending.get(pid, ())
            if (len(waiting) == 1 and pid not in ambiguous
                    and waiting[0]["operation"] == operation):
                item = waiting[0]
                slot = item["slot"]
                sources = index.entries[slot]["source_lines"] + [line]
                combined = item["payload"] + ending
                event, reasons, update, completed = _decode(pid, operation, combined, sources, cwd)
                counts = _effect(index.counts, event, completed)
                kept = event if _retain(event, reasons) else None
                ok, reason, _ = index.put([(slot, kept, reasons)], counts, next_offset)
                if not ok:
                    index.partial = reason
                    break
                del pending[pid]
                if update is _UNKNOWN_CWD:
                    cwd.pop(pid, None)
                elif update is not None:
                    cwd[pid] = update
            else:
                event, reasons = _orphan(pid, operation, ending, line)
                changes = conflict_changes(pid)
                if waiting or pid in ambiguous:
                    reasons.append("AMBIGUOUS_PAIR")
                changes.append((None, event, reasons))
                ok, reason, _ = index.put(changes, index.counts, next_offset)
                if not ok:
                    index.partial = reason
                    break
                if waiting:
                    ambiguous.add(pid)
                if operation in _CONTEXT and event["result"]["value"] == 0:
                    # An orphan successful context change has no known new cwd.
                    cwd.pop(pid, None)
            offset = next_offset
            continue
        operation = entered[1] if entered is not None else None
        unfinished = entered is not None and payload.endswith(" <unfinished ...>")
        if unfinished:
            event, reasons = _unfinished(pid, operation, line)
            waiting = pending.get(pid, ())
            changes = conflict_changes(pid)
            if waiting or pid in ambiguous:
                reasons.append("AMBIGUOUS_PAIR")
            changes.append((None, event, reasons))
            ok, reason, assigned = index.put(changes, index.counts, next_offset)
            if not ok:
                index.partial = reason
                break
            if waiting:
                ambiguous.add(pid)
            pending.setdefault(pid, []).append({
                "slot": assigned[-1], "operation": operation,
                "payload": payload[:-len("<unfinished ...>")],
            })
            offset = next_offset
            continue
        event, reasons, update, completed = _decode(pid, operation, payload, [line], cwd)
        waiting = pending.get(pid, ())
        changes = conflict_changes(pid)
        if waiting or pid in ambiguous:
            reasons.append("AMBIGUOUS_PAIR")
            event["state"] = "UNRESOLVED"
            update = (_UNKNOWN_CWD if operation in _CONTEXT
                      and event["result"]["value"] == 0 else None)
        counts = _effect(index.counts, event, completed)
        if _retain(event, reasons):
            changes.append((None, event, reasons))
        ok, reason, _ = index.put(changes, counts, next_offset)
        if not ok:
            index.partial = reason
            break
        if waiting:
            ambiguous.add(pid)
        if update is _UNKNOWN_CWD:
            cwd.pop(pid, None)
        elif update is not None:
            cwd[pid] = update
        offset = next_offset
    return index.report()
