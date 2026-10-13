"""Q4 limits lens -- FINDING probe (minor): museum_data.read_source's network path
(tools/museum_data.py:477-478)

    with urlopen(descriptor(identity["path"])["url"], timeout=30) as response:
        raw = response.read(identity["bytes"] + 1)

caps the size correctly (bytes+1) but `timeout=30` is a per-socket-operation timeout, not a deadline.
A response that delivers one byte every <30 s is never timed out; the read only ends when bytes+1
are received or the peer closes. Upper bound per source ~ (bytes+1) * 30 s.

No network: museum_data.urlopen is replaced by a stub that returns a *real* http.client.HTTPResponse
over a local socketpair, with the client socket given exactly the timeout read_source passes
(this is what HTTPConnection.connect does: socket.create_connection(..., timeout)). The reading code
under test (HTTPResponse.read(amt) -> BufferedReader.read) is the stdlib path urlopen returns.
Usage: python3 -I -B p4_urlopen_timeout_is_per_recv.py <path-to-cd32303-tree>
"""
import http.client
import socket
import sys
import threading
import time
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root / "tools"))
import museum_data  # noqa: E402  (read-only import; -B writes no bytecode)

PATH = "incoming/side24-identity-replay-20260926/output.json"
identity = museum_data.SOURCES[PATH]
body = (root / PATH).read_bytes()          # the cd32303 tree carries the same pinned bytes
museum_data.verify_bytes(body, identity)    # sanity: these are the pinned bytes
GAP = 20.0                                  # seconds between trickled bytes (< the 30 s timeout)
seen = {}


def stub_urlopen(url, timeout):
    seen["url"], seen["timeout"] = url, timeout
    server, client = socket.socketpair()
    client.settimeout(timeout)

    def trickle():
        server.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body[:-2])
        for byte in (body[-2:-1], body[-1:]):
            time.sleep(GAP)
            server.sendall(byte)
        time.sleep(GAP)
        server.close()

    threading.Thread(target=trickle, daemon=True).start()
    response = http.client.HTTPResponse(client, method="GET")
    response.begin()
    return response


museum_data.urlopen = stub_urlopen
t0 = time.monotonic()
try:
    raw = museum_data.read_source(identity, None, None)
    outcome = f"ACCEPTED {len(raw)} verified bytes"
except Exception as exc:  # noqa: BLE001
    outcome = f"raised {type(exc).__name__}: {exc}"
elapsed = time.monotonic() - t0
print(f"url={seen['url']}")
print(f"timeout passed by read_source={seen['timeout']}s; byte gap={GAP}s")
print(f"read_source: {outcome} after {elapsed:.1f}s (> the {seen['timeout']}s timeout, no error)")
largest = max(s["bytes"] for s in museum_data.SOURCES.values())
print(f"worst-case per source at one byte per 29.9 s: largest pin {largest} B -> {(largest + 1) * 29.9 / 86400:.1f} days")
