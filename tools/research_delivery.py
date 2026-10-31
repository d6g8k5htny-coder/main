"""Build and verify bounded, deterministic research-delivery ZIPs (stdlib only).

Spec: {"version":1,"handoff":"HANDOFF.json","files":[
  {"path":"HANDOFF.json","bytes":123,"sha256":"<64 lowercase hex>"}, ...]}.
Every input, including the handoff, must be listed. DELIVERY_MANIFEST.json is
generated with filename/size/sha256 identities and deliberately has no self-hash.
Control JSON permits strings, integers, booleans, null, arrays and objects, not
floats or duplicate keys. The handoff payload digest uses sorted-key, compact,
UTF-8 JSON (ensure_ascii=False). Supplied payload bytes are otherwise opaque.

This tool reads only explicit inputs, never extracts or executes archive members,
and never accesses a network. Verification accepts only the builder's canonical
ZIP32/STORED format, with fixed metadata and no extra fields or archive comments.
Success establishes byte custody and envelope
structure, not source authenticity, review acceptance or scientific promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import struct
import zipfile
import zlib


MANIFEST = 'DELIVERY_MANIFEST.json'
MAX_FILES = 1000
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + 2 * MAX_JSON_BYTES
REQUIRED_FIELDS = (
    'protocol_version', 'checkpoint_utc', 'task_id', 'claim_id',
    'stable_target_key', 'source_identities', 'output_head',
    'verified_boundary', 'next_action',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def control_json(raw):
    require(len(raw) <= MAX_JSON_BYTES, 'control JSON exceeds size limit')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def bad_number(_):
        raise ValueError('control JSON forbids floating-point and nonfinite numbers')

    try:
        value = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                           parse_float=bad_number, parse_constant=bad_number)
        canonical(value)  # Reject invalid Unicode and excessive nesting too.
        return value
    except (UnicodeError, RecursionError) as exc:
        raise ValueError('invalid control JSON encoding or nesting') from exc


def safe_path(name):
    require(isinstance(name, str) and 0 < len(name) <= 1024, 'unsafe member path')
    require(not name.startswith('/') and '\\' not in name and ':' not in name
            and all(ord(c) >= 32 and ord(c) != 127 for c in name), 'unsafe member path')
    parts = name.split('/')
    require(all(p not in ('', '.', '..') for p in parts), 'unsafe member path')
    try:
        name.encode('utf-8')
    except UnicodeError as exc:
        raise ValueError('invalid member path encoding') from exc
    return parts


def identity_rows(document, manifest=False):
    require(type(document) is dict and set(document) == {'version', 'handoff', 'files'},
            'expected version, handoff and files fields')
    require(type(document['version']) is int and document['version'] == 1,
            'unsupported delivery version')
    safe_path(document['handoff'])
    rows = document['files']
    require(type(rows) is list and 1 <= len(rows) <= MAX_FILES, 'invalid file count')
    path_key, size_key = ('filename', 'size') if manifest else ('path', 'bytes')
    identities = {}
    for row in rows:
        require(type(row) is dict and set(row) == {path_key, size_key, 'sha256'},
                'invalid file identity fields')
        name, size, digest = row[path_key], row[size_key], row['sha256']
        safe_path(name)
        require(name != MANIFEST and name not in identities, 'reserved or duplicate member')
        require(type(size) is int and 0 <= size <= MAX_FILE_BYTES, 'invalid member size')
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest),
                'invalid SHA-256 identity')
        identities[name] = (size, digest)
    # Every member is a regular file, including the generated manifest. No file
    # can simultaneously be a directory prefix, regardless of identity-row order.
    files = set(identities) | {MANIFEST}
    for name in identities:
        parts = name.split('/')
        require(all('/'.join(parts[:end]) not in files
                    for end in range(1, len(parts))),
                'file/directory member conflict: ' + name)
    require(document['handoff'] in identities, 'handoff is not explicitly listed')
    require(sum(size for size, _ in identities.values()) <= MAX_TOTAL_BYTES,
            'payload exceeds total size limit')
    return identities


def handoff_payload(raw):
    envelope = control_json(raw)
    require(type(envelope) is dict and set(envelope) == {'payload', 'payload_sha256'},
            'invalid handoff envelope')
    payload = envelope['payload']
    require(type(payload) is dict and all(key in payload for key in REQUIRED_FIELDS),
            'missing required R17 handoff fields')
    for field in REQUIRED_FIELDS:
        if field == 'source_identities':
            require(type(payload[field]) in (list, dict), 'invalid source identities')
        elif field in ('verified_boundary', 'output_head') and type(payload[field]) is dict:
            require(bool(payload[field]), 'empty R17 field: ' + field)
        else:
            require(isinstance(payload[field], str) and payload[field].strip(),
                    'empty or invalid R17 field: ' + field)
    stamp = payload['checkpoint_utc']
    try:
        checkpoint = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('invalid checkpoint_utc') from exc
    require('T' in stamp and checkpoint.utcoffset() == timedelta(0),
            'checkpoint_utc must be a UTC timestamp')
    digest = sha256(canonical(payload))
    require(envelope['payload_sha256'] == digest, 'handoff payload digest mismatch')
    return digest


def read_regular(path, limit):
    """Explicit control/archive input; reject symlink or special-file leaves."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode), 'input is not a regular file')
        require(info.st_size <= limit, 'input exceeds size limit')
        raw = stream.read(limit + 1)
        require(len(raw) <= limit, 'input exceeds size limit')
        return raw


def read_payload(root_fd, name):
    """Open beneath the root descriptor without following any member symlink."""
    parts = safe_path(name)
    directory = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=directory)
            os.close(directory)
            directory = following
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=directory)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode), 'payload is not a regular file')
            require(info.st_size <= MAX_FILE_BYTES, 'payload exceeds size limit')
            raw = stream.read(MAX_FILE_BYTES + 1)
            require(len(raw) <= MAX_FILE_BYTES, 'payload exceeds size limit')
            return raw
    finally:
        os.close(directory)


def check_identity(name, raw, identity):
    require((len(raw), sha256(raw)) == identity, 'file identity mismatch: ' + name)


def result(raw, manifest_raw, count, payload_digest):
    return {'ok': True, 'files': count, 'archive_bytes': len(raw),
            'archive_sha256': sha256(raw), 'manifest_sha256': sha256(manifest_raw),
            'payload_sha256': payload_digest, 'scientific_effect': 'NONE'}


def build(root: Path, spec: Path, output: Path):
    """Validate explicit inputs and create a new archive; never overwrite output."""
    document = control_json(read_regular(spec, MAX_JSON_BYTES))
    identities = identity_rows(document)
    members = {}
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in sorted(identities):
            raw = read_payload(root_fd, name)
            check_identity(name, raw, identities[name])
            members[name] = raw
    finally:
        os.close(root_fd)
    payload_digest = handoff_payload(members[document['handoff']])
    manifest_raw = canonical({'version': 1, 'handoff': document['handoff'], 'files': [
        {'filename': name, 'size': size, 'sha256': digest}
        for name, (size, digest) in sorted(identities.items())]})
    require(len(manifest_raw) <= MAX_JSON_BYTES, 'manifest exceeds size limit')
    members[MANIFEST] = manifest_raw
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw)
    raw = buffer.getvalue()
    require(len(raw) <= MAX_ARCHIVE_BYTES, 'archive exceeds size limit')
    with Path(output).open('xb') as stream:
        stream.write(raw)
    return result(raw, manifest_raw, len(identities), payload_digest)


def stored_members(raw):
    """Parse only the builder's ZIP32/STORED format, without ZipInfo allocation.

    Check the actual bounded directory records, then matching local headers and
    contiguous byte ranges. EOCD counts alone are untrusted. Reject extensions,
    descriptors, compression, gaps and trailing bytes rather than interpreting
    additional ZIP dialects. No decompressor can silently truncate a member.
    """
    require(len(raw) >= 22, 'truncated ZIP end record')
    end = len(raw) - 22
    signature, disk, start_disk, disk_count, count, length, start, comment = struct.unpack_from(
        '<4s4H2LH', raw, end)
    require(signature == b'PK\x05\x06' and disk == start_disk == comment == 0,
            'unsupported ZIP end record')
    require(disk_count == count and 2 <= count <= MAX_FILES + 1, 'invalid ZIP member count')
    require(start + length == end, 'invalid ZIP directory bounds')
    position, next_local, total = start, 0, 0
    ranges = {}
    while position < end:
        require(len(ranges) < MAX_FILES + 1, 'too many actual ZIP directory records')
        require(position + 46 <= end, 'truncated ZIP directory header')
        (signature, creator, version, flags, method, time, date, crc, compressed, size,
         name_length, extra, note, disk, internal, external, offset) = struct.unpack_from(
             '<4s6H3L5H2L', raw, position)
        require(signature == b'PK\x01\x02' and creator == 788 and version == 20
                and method == 0 and time == 0 and date == 33
                and extra == note == disk == internal == 0
                and external == (stat.S_IFREG | 0o644) << 16,
                'noncanonical ZIP directory header (stored regular files only)')
        require(0 < name_length <= 4096 and position + 46 + name_length <= end,
                'invalid ZIP filename bounds')
        name_raw = raw[position + 46:position + 46 + name_length]
        try:
            name = name_raw.decode('utf-8')
        except UnicodeError as exc:
            raise ValueError('invalid ZIP filename encoding') from exc
        safe_path(name)
        require(flags == (0x800 if not name.isascii() else 0), 'unsupported ZIP flags')
        require(name not in ranges, 'duplicate ZIP member')
        limit = MAX_JSON_BYTES if name == MANIFEST else MAX_FILE_BYTES
        require(compressed == size and size <= limit, 'invalid stored ZIP member size')
        total += size
        require(total <= MAX_TOTAL_BYTES + MAX_JSON_BYTES, 'ZIP exceeds total size limit')
        require(offset == next_local and offset + 30 <= start, 'invalid local ZIP header bounds')
        local = struct.unpack_from('<4s5H3L2H', raw, offset)
        require(local == (b'PK\x03\x04', version, flags, method, time, date,
                          crc, size, size, name_length, 0), 'ZIP headers disagree')
        data_start = offset + 30 + name_length
        next_local = data_start + size
        require(next_local <= start and raw[offset + 30:data_start] == name_raw,
                'invalid local ZIP name or payload bounds')
        ranges[name] = (data_start, next_local, crc)
        position += 46 + name_length
    require(position == end and len(ranges) == count and next_local == start,
            'ZIP directory count mismatch or unaccounted bytes')
    require(list(ranges) == sorted(ranges), 'ZIP member ordering is not canonical')
    members = {}
    for name, (begin, finish, crc) in ranges.items():
        member = raw[begin:finish]
        require(zlib.crc32(member) == crc, 'ZIP member CRC mismatch')
        members[name] = member
    return members


def verify(archive: Path):
    """Verify the canonical stored delivery format; never extract or execute."""
    raw = read_regular(archive, MAX_ARCHIVE_BYTES)
    members = stored_members(raw)
    require(MANIFEST in members, 'delivery manifest missing')
    manifest_raw = members[MANIFEST]
    document = control_json(manifest_raw)
    identities = identity_rows(document, manifest=True)
    require(set(members) == set(identities) | {MANIFEST}, 'unexpected or missing ZIP members')
    for name, identity in identities.items():
        check_identity(name, members[name], identity)
    payload_digest = handoff_payload(members[document['handoff']])
    return result(raw, manifest_raw, len(identities), payload_digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build', help='build an explicit, identity-checked delivery')
    builder.add_argument('--root', type=Path, required=True)
    builder.add_argument('--spec', type=Path, required=True)
    builder.add_argument('--output', type=Path, required=True)
    verifier = commands.add_parser('verify', help='verify canonical ZIP32/STORED delivery only')
    verifier.add_argument('archive', type=Path)
    args = parser.parse_args()
    try:
        outcome = (build(args.root, args.spec, args.output) if args.command == 'build'
                   else verify(args.archive))
    except (ValueError, OSError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, separators=(',', ':')))
        return 1
    print(json.dumps(outcome, sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
