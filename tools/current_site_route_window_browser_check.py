#!/usr/bin/env python3
"""Exercise the required-route render budget with an admitted 10,000-node chain."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 150
SIZE = 10_000
TOTAL_OCCURRENCES = sum(range(2, SIZE + 1))
PAGES = math.ceil(TOTAL_OCCURRENCES / LIMIT)
DIGEST = re.compile(r'data-current-digest="[0-9a-f]{64}"')
BYTE_COUNT = re.compile(r'data-current-bytes="[0-9]+"')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@contextmanager
def serve():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(ROOT / "docs")))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/site/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def synthetic_payload() -> tuple[bytes, str, int]:
    source = ROOT / "docs/site/current.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    ids = [f"route399.{i:05d}" for i in range(SIZE)]
    payload["graph"]["data"] = {
        "nodes": {
            node_id: {"kind": "candidate", "classification": "OPEN_ACTIVE", "layer": "D1"}
            for node_id in ids
        },
        "edges": [
            {
                "from": ids[i - 1],
                "to": ids[i],
                "required": True,
                "relation": "synthetic_required_chain",
                "ordinal": i,
            }
            for i in range(1, SIZE)
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return raw, sha256(raw).hexdigest(), len(source.read_bytes())


def admitted_html(digest: str, size: int) -> str:
    source = (ROOT / "docs/site/dependencies.html").read_text(encoding="utf-8")
    rendered, digest_count = DIGEST.subn(f'data-current-digest="{digest}"', source)
    rendered, byte_count = BYTE_COUNT.subn(f'data-current-bytes="{size}"', rendered)
    require(digest_count == 1, f"expected one current digest marker, found {digest_count}")
    require(byte_count == 1, f"expected one current byte marker, found {byte_count}")
    return rendered


def route_measurement(page) -> dict[str, object]:
    nodes = page.locator("#atlas-route-list [data-route-node]")
    edges = page.locator("#atlas-route-list [data-route-edge]")
    count = nodes.count()
    require(0 < count <= LIMIT, f"route-node budget exceeded: {count}")
    require(edges.count() <= LIMIT, f"route-edge budget exceeded: {edges.count()}")
    return {
        "page": int(page.locator("#atlas-route-page").input_value()),
        "rendered_route_nodes": count,
        "rendered_route_edges": edges.count(),
        "first_node": nodes.first.inner_text(),
        "last_node": nodes.last.inner_text(),
        "count_text": page.locator("#atlas-route-count").inner_text(),
        "url": page.url,
    }


def run(browser_path: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    fixture, fixture_digest, _source_bytes = synthetic_payload()
    html = admitted_html(fixture_digest, len(fixture))
    report: dict[str, object] = {
        "scientific_effect": "NONE",
        "scope": "Synthetic display fixture only; not research-corpus evidence.",
        "fixture_nodes": SIZE,
        "fixture_edges": SIZE - 1,
        "fixture_sha256": fixture_digest,
        "fixture_bytes": len(fixture),
        "route_occurrences": TOTAL_OCCURRENCES,
        "route_pages": PAGES,
        "route_limit": LIMIT,
        "measurements": [],
        "page_errors": [],
        "passed": False,
    }
    try:
        report["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        report["source_commit"] = None

    with serve() as origin, sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path=str(browser_path), headless=True, chromium_sandbox=True
        )
        report["browser_version"] = browser.version
        context = browser.new_context(viewport={"width": 1280, "height": 900}, reduced_motion="reduce")
        page = context.new_page()
        page.set_default_timeout(30_000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        page.route(
            "**/dependencies.html?*",
            lambda route: route.fulfill(status=200, content_type="text/html", body=html),
        )
        page.route(
            "**/current.json",
            lambda route: route.fulfill(status=200, content_type="application/json", body=fixture),
        )
        started = time.perf_counter()
        try:
            page.goto(origin + "dependencies.html?atlas-node=route399.00000&atlas-view=list#current-atlas")
            expect(page.locator("#atlas-inspector h3")).to_have_text("route399.00000")
            expect(page.locator("#atlas-route-count")).to_contain_text("9999 unresolved targets")
            expect(page.locator("#atlas-route-count")).to_contain_text(
                f"{TOTAL_OCCURRENCES} route-node occurrences"
            )
            details = page.locator("#atlas-route-details")
            details.locator(":scope > summary").click()
            first = route_measurement(page)
            require(first["page"] == 1, f"unexpected first page: {first}")
            require("atlas-route-page" not in page.url, "default route page polluted the URL")
            report["measurements"].append({"step": "first", **first})

            middle = PAGES // 2
            page.locator("#atlas-route-page").fill(str(middle))
            page.locator("#atlas-route-page").press("Enter")
            expect(page.locator("#atlas-route-page")).to_have_value(str(middle))
            mid = route_measurement(page)
            require(f"atlas-route-page={middle}" in page.url, "middle route page missing from URL")
            require(mid["first_node"] != first["first_node"], "middle page reused first page")
            report["measurements"].append({"step": "middle", **mid})

            page.locator("#atlas-route-last").click()
            expect(page.locator("#atlas-route-page")).to_have_value(str(PAGES))
            last = route_measurement(page)
            require(last["rendered_route_nodes"] == TOTAL_OCCURRENCES % LIMIT, f"wrong final page: {last}")
            require("route399.09999" in str(last["last_node"]), "last route page omitted final target")
            require(f"atlas-route-page={PAGES}" in page.url, "last route page missing from URL")
            report["measurements"].append({"step": "last", **last})

            saved = page.url
            page.reload()
            expect(page.locator("#atlas-route-page")).to_have_value(str(PAGES))
            restored = route_measurement(page)
            require(page.url == saved, "reload rewrote the exact saved route page")
            report["measurements"].append({"step": "reloaded-last", **restored})

            page.locator("#atlas-route-details > summary").click()
            page.locator("#atlas-route-list [data-route-node]").last.click()
            expect(page.locator("#atlas-inspector h3")).to_have_text("route399.09999")
            require("atlas-route-page" not in page.url, "node selection did not reset route page")
            page.go_back()
            expect(page.locator("#atlas-inspector h3")).to_have_text("route399.00000")
            expect(page.locator("#atlas-route-page")).to_have_value(str(PAGES))
            require(page.url == saved, "Back did not restore exact root route page")

            require(not report["page_errors"], f"page errors: {report['page_errors']}")
            screenshot = output / "route-window-last-page.png"
            page.locator("#atlas-inspector").screenshot(path=str(screenshot))
            report["screenshot"] = {
                "path": screenshot.name,
                "sha256": sha256(screenshot.read_bytes()).hexdigest(),
            }
            report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            report["passed"] = True
        except Exception:
            report["error"] = traceback.format_exc()
        finally:
            context.close()
            browser.close()

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    candidates = [Path("/opt/google/chrome/chrome"), Path("/usr/bin/chromium")]
    default = next((path for path in candidates if path.exists()), candidates[0])
    parser.add_argument("--browser", type=Path, default=default)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.browser.is_file():
        raise SystemExit(f"browser not found: {args.browser}")
    report = run(args.browser, args.output)
    if not report["passed"]:
        print(json.dumps(report, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
