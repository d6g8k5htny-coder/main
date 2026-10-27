#!/usr/bin/env python3
"""Check current-source list continuity using real admitted publication bytes.

This is bounded engineering evidence: no source authentication, mathematical
review, spoken-AT, Safari or physical-device qualification is inferred.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from urllib.parse import parse_qs, quote, urlencode, urlsplit

from research_desk_browser_check import ROOT, serve


QUERY = "proof"
PAGE_SIZE = 50
PRODUCT_FILES = ["workspace.html", "current.mjs", "current-model.mjs",
                 "research-desk.mjs", "library-source-search.mjs", "current.json"]


def expected_urls(data, query):
    """Derive source identities independently of the production renderer."""
    words = query.lower().strip().split()
    sources = [source for source in data["sources"]
               if all(word in (source["repository"] + " " + source["path"]).lower()
                      for word in words)]
    return ["https://github.com/" + source["repository"] + "/blob/"
            + source["commit"] + "/"
            + "/".join(quote(part, safe="~()*!.'")
                       for part in source["path"].split("/"))
            for source in sources]


def run(origin, browser, output, report, local):
    from playwright.sync_api import expect

    def visit(page, row, query=QUERY):
        target = origin + "workspace.html?" + urlencode(
            {"keep": "literal#note", "source-q": query}) + "#library-current"
        with page.expect_response(
                lambda response: urlsplit(response.url).path.endswith("/current.json")) as pending:
            page.goto(target)
        response = pending.value
        assert response.status == 200, "Setup: current payload was not available"
        raw = response.body()
        binding = page.locator("#current-research").evaluate(
            "e=>({sha256:e.dataset.currentDigest,bytes:Number(e.dataset.currentBytes)})")
        digest = sha256(raw).hexdigest()
        assert digest == binding["sha256"] and len(raw) == binding["bytes"], (
            "Setup: actual payload bytes do not match the displayed publication binding")
        expect(page.locator(".current-search")).to_be_enabled()
        expect(page.locator(".current-state")).to_contain_text("Verified source observation:")
        expect(page.locator(".desk-save-source").first).to_be_visible()
        data = json.loads(raw)
        row["observation"] = {"sha256": digest, "bytes": len(raw),
                              "observed_at": data["observed_at"],
                              "sources": len(data["sources"])}
        capture = output / ("admitted-current-" + digest[:16] + ".json")
        if not capture.exists():
            capture.write_bytes(raw)
        urls = expected_urls(data, query)
        assert len(urls) > PAGE_SIZE * 2, "Setup: query needs more than two result batches"
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE)
        return data, urls

    def visible_urls(page):
        return page.locator(".current-results > li > a").evaluate_all(
            "links=>links.map(link=>link.href)")

    def check_window(page, urls, count):
        actual = visible_urls(page)
        assert actual == urls[:count], "Result identities/order differ from admitted payload"
        assert len(actual) == len(set(actual)), "Duplicate source URL in displayed result window"

    def case(name, width, scheme, operation):
        row = {"case": name, "viewport": {"width": width, "height": 900},
               "color_scheme": scheme}
        report["cases"].append(row)
        context = browser.new_context(viewport=row["viewport"],
                                      color_scheme=scheme, reduced_motion="reduce")
        if local:
            # Keep remote dated-catalog acquisition outside this current-list check.
            context.route("https://**/*", lambda route: route.abort())
        page = context.new_page()
        page.set_default_timeout(5000)
        page.on("pageerror", lambda error: report["page_errors"].append(str(error)))
        try:
            operation(page, context, row)
            row["passed"] = True
        except Exception:
            row["passed"] = False
            row["error"] = traceback.format_exc()
        finally:
            try:
                path = output / (name + ".png")
                page.screenshot(path=str(path))
                row["screenshot"] = path.name
            except Exception as error:
                row["screenshot_error"] = str(error)
            context.unroute_all(behavior="wait")
            context.close()

    # Replacing the first fifty rows must fail these real DOM/state checks.
    def retain_rows(page, _context, row):
        _data, urls = visit(page, row)
        disclosure = page.locator(".current-results > li details").first
        assert disclosure.count(), "Setup: admitted matches need an acquisition disclosure"
        disclosure.locator("summary").click()
        page.locator(".current-results > li .desk-save-source").first.click()
        expect(page.locator(".current-results > li .desk-save-source").first).to_have_text("Saved")
        page.evaluate("""() => {
            window.continuityRefs={
                row:document.querySelector('.current-results > li'),
                control:document.querySelector('.current-results > li .desk-save-source'),
                disclosure:document.querySelector('.current-results details[open]')};
        }""")
        before = output / (row["case"] + "-before.png")
        page.screenshot(path=str(before))
        row["before_screenshot"] = before.name
        page.locator(".current-more").evaluate("button=>button.click()")
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE * 2)
        check_window(page, urls, PAGE_SIZE * 2)
        row["retained"] = page.evaluate("""() => ({
            firstRow:continuityRefs.row===document.querySelector('.current-results > li'),
            saveControl:continuityRefs.control===document.querySelector('.current-results > li .desk-save-source'),
            savedState:continuityRefs.control.isConnected&&continuityRefs.control.disabled&&continuityRefs.control.textContent==='Saved',
            disclosure:continuityRefs.disclosure.isConnected,
            disclosureOpen:continuityRefs.disclosure.isConnected&&continuityRefs.disclosure.open
        })""")
        assert all(row["retained"].values()), row["retained"]

    for width, scheme in [(390, "light"), (390, "dark"),
                          (1440, "light"), (320, "light")]:
        case(f"retained-rows-disclosure-control-{width}-{scheme}", width, scheme, retain_rows)

    # A missing focus handoff must fail every More batch, including the final one.
    def keyboard_batches(page, _context, row):
        _data, urls = visit(page, row)
        row["matches"] = len(urls)
        row["batches"] = []
        failures = []
        old_count = PAGE_SIZE
        check_window(page, urls, old_count)
        while old_count < len(urls):
            page.locator(".current-more").focus()
            page.keyboard.press("Enter")
            count = min(old_count + PAGE_SIZE, len(urls))
            expect(page.locator(".current-results > li")).to_have_count(count)
            check_window(page, urls, count)
            expected_link = page.locator(".current-results > li > a").nth(old_count)
            focus = page.evaluate("""() => ({tag:document.activeElement.tagName,
                href:document.activeElement.href||null,
                visible:(()=>{const r=document.activeElement.getBoundingClientRect();
                    return r.width>0&&r.height>0&&r.top>=0&&r.bottom<=innerHeight})()})""")
            observation = {"from": old_count, "to": count, "expected_focus": urls[old_count],
                           "focus": focus, "more_hidden": page.locator(".current-more").is_hidden()}
            row["batches"].append(observation)
            if focus["tag"] != "A" or focus["href"] != urls[old_count] or not focus["visible"]:
                failures.append(f"{old_count}→{count}: first newly revealed source link is not visibly focused")
            if count == len(urls):
                assert page.locator(".current-more").is_hidden(), "Final batch must hide More"
                if focus["tag"] == "BODY":
                    failures.append("Final batch abandoned keyboard focus to BODY")
            else:
                expect(page.locator(".current-more")).to_be_visible()
            assert expected_link.get_attribute("href") == urls[old_count]
            old_count = count
        assert not failures, "; ".join(failures)

    case("keyboard-batches-ordered-new-source-focus", 390, "light", keyboard_batches)

    # A programmatic update must not steal a reader's newer unrelated focus.
    def outside_focus(page, _context, row):
        _data, urls = visit(page, row)
        outside = page.locator("#reading-list > summary")
        outside.focus()
        page.locator(".current-more").evaluate("button=>button.click()")
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE * 2)
        check_window(page, urls, PAGE_SIZE * 2)
        expect(outside).to_be_focused()
        row["active_element"] = page.evaluate("document.activeElement.outerHTML")

    case("programmatic-more-preserves-newer-outside-focus", 390, "dark", outside_focus)

    # Appending after a query/history change would retain stale identities.
    def resets(page, _context, row):
        data, urls = visit(page, row)
        row["resets"] = []

        def capture_old():
            page.evaluate("window.oldResultRow=document.querySelector('.current-results > li')")

        def reset_window(query, operation, label):
            capture_old()
            operation()
            wanted = expected_urls(data, query)
            count = min(PAGE_SIZE, len(wanted))
            expect(page.locator(".current-results > li")).to_have_count(count)
            check_window(page, wanted, count)
            assert page.evaluate("!oldResultRow.isConnected"), label + " retained a stale row"
            row["resets"].append({"stage": label, "rows": count, "query": query,
                                  "url": page.evaluate("location.href")})

        page.locator(".current-more").evaluate("button=>button.click()")
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE * 2)
        reset_window("main", lambda: page.locator(".current-search").fill("main"), "input")
        page.locator(".current-more").evaluate("button=>button.click()")
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE * 2)
        reset_window("", lambda: page.locator("#library-source-clear").click(), "clear")
        expect(page.locator(".current-search")).to_be_focused()
        params = parse_qs(urlsplit(page.evaluate("location.href")).query)
        assert params["keep"] == ["literal#note"] and "source-q" not in params
        page.locator(".current-search").fill(QUERY)
        first_url = page.evaluate("location.href")
        page.evaluate("location.hash='reading-list'")
        page.locator(".current-search").fill("main")
        second_url = page.evaluate("location.href")
        page.locator(".current-more").evaluate("button=>button.click()")
        reset_window(QUERY, page.go_back, "back")
        expect(page).to_have_url(first_url)
        page.locator(".current-more").evaluate("button=>button.click()")
        reset_window("main", page.go_forward, "forward")
        expect(page).to_have_url(second_url)
        assert page.evaluate("document.documentElement.scrollWidth<=innerWidth")

    case("query-clear-history-replace-and-reset-window", 320, "light", resets)

    # A value assignment without an input event must not append a mixed-query
    # prefix or spend the previous query's result limit on the new query.
    def programmatic_query(page, _context, row):
        data, urls = visit(page, row)
        page.locator(".current-more").evaluate("button=>button.click()")
        expect(page.locator(".current-results > li")).to_have_count(PAGE_SIZE * 2)
        check_window(page, urls, PAGE_SIZE * 2)
        outside = page.locator("#reading-list > summary")
        outside.focus()
        page.evaluate("""() => {
            window.oldResultRow=document.querySelector('.current-results > li');
            document.querySelector('.current-search').value='main';
            document.querySelector('.current-more').click();
        }""")
        wanted = expected_urls(data, "main")
        actual = visible_urls(page)
        row["programmatic_query"] = {
            "query": "main", "rows_before": PAGE_SIZE * 2,
            "rows_after": len(actual), "expected_rows": min(PAGE_SIZE, len(wanted)),
            "exact_reset_window": actual == wanted[:PAGE_SIZE],
            "old_row_disconnected": page.evaluate("!oldResultRow.isConnected"),
            "outside_focus_preserved": outside.evaluate("e=>e===document.activeElement")}
        assert row["programmatic_query"]["exact_reset_window"], row["programmatic_query"]
        assert row["programmatic_query"]["old_row_disconnected"], "New query retained an old row"
        expect(outside).to_be_focused()

    case("programmatic-query-change-replaces-resets-before-more", 390, "light", programmatic_query)

    # Failure must never enable controls or manufacture verified result rows.
    def refusal(page, context, row):
        context.route("**/current.json", lambda route: route.fulfill(status=503, body="Unavailable"))
        page.goto(origin + "workspace.html?source-q=proof#library-current")
        expect(page.locator(".current-state")).to_contain_text("Unavailable:")
        expect(page.locator(".current-search")).to_be_disabled()
        expect(page.locator("#desk-save-current")).to_be_disabled()
        expect(page.locator(".current-more")).to_be_hidden()
        assert page.locator(".current-results .desk-save-source").count() == 0
        assert page.locator("#library-source-tools").count() == 0
        row["message"] = page.locator(".current-state").inner_text()

    case("refused-source-keeps-controls-disabled", 390, "light", refusal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", default=shutil.which("chromium") or shutil.which("google-chrome"))
    parser.add_argument("--origin", help="Served /site/ origin; use its exact admitted payload")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    short_tmp = Path("/tmp/ulslc")
    short_tmp.mkdir(mode=0o700, exist_ok=True)
    os.environ["TMPDIR"] = str(short_tmp)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "cases": [],
              "page_errors": [], "invocation": sys.argv,
              "scope": "Local Chromium DOM/state/focus and exact admitted source identities; engineering only. No source authentication, mathematical, spoken-AT, Safari or physical-device claim.",
              "actor": {"provider": "OpenAI", "family": "GPT-6", "role": "engineering test author", "organizational_independence_credit": 0, "scientific_effect": "NONE"},
              "mode": "served origin" if args.origin else "included local snapshot",
              "checkout_identity_scope": "Local checkout only, including with --origin; checked_commit and source_sha256 do not identify served HTML or JavaScript.",
              "served_code_identity": "Not acquired by this driver; bind served HTML and modules independently.",
              "chromium_sandbox": True,
              "source_sha256": {name: sha256((ROOT / "docs/site" / name).read_bytes()).hexdigest()
                                for name in PRODUCT_FILES}}
    report["harness_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    report["checked_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()

    def execute(origin):
        from playwright.sync_api import sync_playwright
        report["origin"] = origin
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True)
            report["browser"] = browser.version
            try:
                run(origin, browser, args.output, report, not args.origin)
            finally:
                browser.close()

    try:
        if args.origin:
            execute(args.origin.rstrip("/") + "/")
        else:
            with serve() as origin:
                execute(origin)
    except Exception:
        report["runner_error"] = traceback.format_exc()
    report["passed"] = sum(row.get("passed", False) for row in report["cases"])
    report["failed"] = len(report["cases"]) - report["passed"]
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in
                     ["browser", "checked_commit", "passed", "failed", "page_errors", "runner_error"]}))
    return int(bool(report["failed"] or report["page_errors"] or report.get("runner_error")))


if __name__ == "__main__":
    raise SystemExit(main())
