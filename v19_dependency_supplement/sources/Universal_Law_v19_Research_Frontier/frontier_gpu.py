"""Bounded ULP1 geometry frames and a read-only same-origin WebSocket.

Geometry is presentation metadata, never mathematical evidence. Only stdlib is used.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import ipaddress
import math
import select
import ssl
import struct
from urllib.parse import urlsplit

MAX_NODES = 200_000
HEADER_BYTES = 64
STRIDE = 32
MAX_FRAME_BYTES = HEADER_BYTES + MAX_NODES * STRIDE
_HEADER = struct.Struct('<4sHHIIIIII32s')
_NODE = struct.Struct('<8f')


def _uint32(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffff:
        raise ValueError(f'{name} must be a uint32')
    return value


def encode_positions(nodes, sequence=0, graph_sha256=None, synthetic=False) -> bytes:
    """Encode finite float32 records; graph_sha256 binds the represented graph.

    Each record is a mapping with x/y/z and optional size/r/g/b/a, or eight numbers.
    The all-zero hash explicitly means that no graph fingerprint was supplied.
    """
    _uint32(sequence, 'sequence')
    if not isinstance(synthetic, bool):
        raise ValueError('synthetic must be boolean')
    if graph_sha256 is None:
        digest = bytes(32)
    else:
        if not isinstance(graph_sha256, str) or len(graph_sha256) != 64:
            raise ValueError('graph_sha256 must be 64 hex characters')
        try:
            digest = bytes.fromhex(graph_sha256)
        except ValueError as exc:
            raise ValueError('graph_sha256 must be 64 hex characters') from exc
        if len(digest) != 32:
            raise ValueError('graph_sha256 must be 64 hex characters')
    if not isinstance(nodes, (list, tuple)) or len(nodes) > MAX_NODES:
        raise ValueError(f'nodes must be a bounded list/tuple of at most {MAX_NODES}')
    payload = bytearray(len(nodes) * STRIDE)
    for i, node in enumerate(nodes):
        if isinstance(node, dict):
            try:
                values = [node['x'], node['y'], node['z'], node.get('size', .035),
                          node.get('r', .3), node.get('g', .7), node.get('b', 1.), node.get('a', 1.)]
            except KeyError as exc:
                raise ValueError('node requires x, y, z') from exc
        elif isinstance(node, (list, tuple)) and len(node) == 8:
            values = node
        else:
            raise ValueError('node must have exactly eight components or be a position mapping')
        try:
            finite = all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('node components must be finite numbers')
        try:
            _NODE.pack_into(payload, i * STRIDE, *values)
        except (OverflowError, struct.error) as exc:
            raise ValueError('node components must fit finite float32') from exc
    return _HEADER.pack(b'ULP1', 1, HEADER_BYTES, len(nodes), STRIDE, sequence,
                        int(synthetic), len(payload), 0, digest) + payload


def decode_positions(raw) -> dict:
    """Validate the entire frame before returning any geometry."""
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError('frame must be bytes')
    length = raw.nbytes if isinstance(raw, memoryview) else len(raw)
    if not HEADER_BYTES <= length <= MAX_FRAME_BYTES:
        raise ValueError('frame length out of bounds')
    raw = bytes(raw)
    magic, version, header, count, stride, sequence, flags, size, reserved, digest = _HEADER.unpack_from(raw)
    if magic != b'ULP1' or version != 1 or header != HEADER_BYTES:
        raise ValueError('unsupported position header')
    if count > MAX_NODES or stride != STRIDE or flags & ~1 or reserved:
        raise ValueError('invalid position metadata')
    if size != count * STRIDE or len(raw) != HEADER_BYTES + size:
        raise ValueError('truncated or trailing position data')
    nodes = []
    for values in _NODE.iter_unpack(memoryview(raw)[HEADER_BYTES:]):
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite position data')
        nodes.append(list(values))
    return {'version': version, 'count': count, 'stride': stride, 'sequence': sequence,
            'synthetic': bool(flags & 1), 'graph_sha256': digest.hex(), 'nodes': nodes}


def _authority(value, scheme='http'):
    try:
        parsed = urlsplit(f'{scheme}://{value}')
        if (not parsed.hostname or parsed.username or parsed.password or parsed.path or
                parsed.query or parsed.fragment or ',' in value or any(c.isspace() for c in value)):
            raise ValueError()
        return parsed.hostname.lower(), parsed.port or (443 if scheme == 'https' else 80)
    except ValueError as exc:
        raise ValueError('invalid host authority') from exc


def validate_websocket_request(handler):
    """Return the RFC6455 accept value, or raise ValueError before upgrading."""
    if handler.command != 'GET' or handler.request_version != 'HTTP/1.1':
        raise ValueError('WebSocket requires GET HTTP/1.1')
    headers = handler.headers
    for name in ('Host', 'Origin', 'Upgrade', 'Connection', 'Sec-WebSocket-Key', 'Sec-WebSocket-Version'):
        if len(headers.get_all(name, [])) != 1:
            raise ValueError(f'missing or duplicated {name}')
    if headers['Upgrade'].lower() != 'websocket' or 'upgrade' not in [x.strip().lower() for x in headers['Connection'].split(',')]:
        raise ValueError('WebSocket upgrade required')
    if headers['Sec-WebSocket-Version'] != '13':
        raise ValueError('WebSocket version 13 required')
    origin = urlsplit(headers['Origin'])
    expected_scheme = 'https' if isinstance(handler.connection, ssl.SSLSocket) else 'http'
    if (origin.scheme != expected_scheme or not origin.netloc or origin.path or origin.query or
            origin.fragment or origin.username or origin.password):
        raise ValueError('same origin required')
    host, port = _authority(headers['Host'], origin.scheme)
    if (host, port) != _authority(origin.netloc, origin.scheme):
        raise ValueError('same origin required')
    bound_host, bound_port = handler.server.server_address[:2]
    if port != bound_port:
        raise ValueError('Host port does not match listener')
    # The workspace is loopback by default. Do not let a matching attacker-chosen
    # Origin/Host authorize DNS rebinding against the loopback listener.
    try:
        bound_loopback = ipaddress.ip_address(bound_host).is_loopback
    except ValueError:
        bound_loopback = bound_host.lower() == 'localhost'
    if bound_loopback and host not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Host does not match loopback listener')
    if not bound_loopback and bound_host not in ('', '0.0.0.0', '::') and host != bound_host:
        raise ValueError('Host does not match listener')
    try:
        key = headers['Sec-WebSocket-Key']
        decoded = base64.b64decode(key, validate=True)
        if len(decoded) != 16:
            raise ValueError()
    except (ValueError, binascii.Error) as exc:
        raise ValueError('invalid WebSocket key') from exc
    return base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode('ascii')).digest()).decode('ascii')


def _send_frame(connection, opcode, payload=b''):
    size = len(payload)
    header = bytes([0x80 | opcode])
    if size < 126:
        header += bytes([size])
    elif size <= 65535:
        header += b'\x7e' + struct.pack('!H', size)
    else:
        header += b'\x7f' + struct.pack('!Q', size)
    connection.sendall(header + payload)


def _receive_control(connection, pending):
    """Incremental bounded client parser; returns False when connection must end."""
    chunk = connection.recv(4096)
    if not chunk:
        return False
    pending.extend(chunk)
    while len(pending) >= 2:
        first, second = pending[:2]
        opcode, size = first & 15, second & 127
        if first & 0x70 or not first & 0x80 or not second & 0x80 or size > 125:
            _send_frame(connection, 8, struct.pack('!H', 1002))
            return False
        if opcode not in (8, 9, 10):
            _send_frame(connection, 8, struct.pack('!H', 1008))
            return False
        if len(pending) < 6 + size:
            return True
        key = pending[2:6]
        payload = bytes(pending[6 + i] ^ key[i % 4] for i in range(size))
        del pending[:6 + size]
        if opcode == 8 and size == 1:
            _send_frame(connection, 8, struct.pack('!H', 1002))
            return False
        if opcode == 8:
            _send_frame(connection, 8, struct.pack('!H', 1000))
            return False
        if opcode == 9:
            _send_frame(connection, 10, payload)
    return True


def _snapshot(frame_factory):
    raw = frame_factory()
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError('snapshot must be bytes')
    size = raw.nbytes if isinstance(raw, memoryview) else len(raw)
    if not HEADER_BYTES <= size <= MAX_FRAME_BYTES:
        raise ValueError('snapshot length out of bounds')
    raw = bytes(raw)
    decode_positions(raw)
    return raw


def stream_positions(handler, frame_factory, stop_event):
    """Upgrade a same-origin request and send bounded binary snapshots at <=10 Hz.

    No client application messages are accepted. Ping/pong and close are the only
    client operations; authorization to mutate workspace state is never granted.
    """
    try:
        accept = validate_websocket_request(handler)
    except (ValueError, UnicodeError) as exc:
        handler.send_error(403, str(exc))
        return
    try:
        first = _snapshot(frame_factory)
    except (ValueError, TypeError):
        handler.send_error(500, 'invalid position snapshot')
        return
    handler.protocol_version = 'HTTP/1.1'
    handler.send_response(101)
    handler.send_header('Upgrade', 'websocket')
    handler.send_header('Connection', 'Upgrade')
    handler.send_header('Sec-WebSocket-Accept', accept)
    handler.end_headers()
    handler.close_connection = True
    connection = handler.connection
    connection.settimeout(2)
    pending = bytearray()
    try:
        _send_frame(connection, 2, first)
        while not stop_event.wait(.1):
            ready, _, _ = select.select([connection], [], [], 0)
            if ready and not _receive_control(connection, pending):
                break
            raw = _snapshot(frame_factory)
            _send_frame(connection, 2, raw)
    except (OSError, ValueError, TypeError):
        pass
