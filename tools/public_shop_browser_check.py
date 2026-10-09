"""Run scoped headless Chromium Library and reader source-card checks against docs served on loopback.

Requires tests/browser-requirements.txt and the runner’s packaged Google Chrome.
Screenshots are evidence for inspection, not automatic visual certification.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256, file_digest
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import base64
import importlib.metadata
import json
import os
import mimetypes
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

ROOT=Path(__file__).resolve().parents[1]

@contextmanager
def local_docs(warm_release=False):
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_GET(self):
            requests.append(self.path)
            url=urlsplit(self.path)
            path=Path(self.translate_path(url.path))
            if warm_release and url.path == '/site/predecessor.html':
                data=re.sub(r'[?&]site-release=[0-9a-f]{64}', '', (ROOT/'docs/site/dependencies.html').read_text()).encode()
                self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            if warm_release and not url.query and path.suffix in ('.mjs','.js','.css') and path.is_file():
                # Controlled predecessor reproduces the actual former 30-result cap.
                # No routing interception: Chromium's HTTP cache remains enabled.
                text=re.sub(r'[?&]site-release=[0-9a-f]{64}', '', path.read_text())
                if path.name == 'dependencies.mjs':
                    text=text.replace('searchResults.replaceChildren(...results.map(', 'searchResults.replaceChildren(...results.slice(0,30).map(')
                data=text.encode();self.send_response(200)
                self.send_header('Content-Type',mimetypes.guess_type(path.name)[0] or 'application/javascript')
                self.send_header('Cache-Control','public, max-age=31536000, immutable')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            super().do_GET()
    server=ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler,directory=str(ROOT/"docs")))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        origin=f"http://127.0.0.1:{server.server_port}/site/"
        yield (origin,requests) if warm_release else origin
    finally:
        server.shutdown();server.server_close();thread.join()


def require(condition, message):
    if not condition:
        raise ValueError(message)

def check_warm_asset_release(browser, expect, result, output):
    with local_docs(warm_release=True) as (origin, requests):
        context=browser.new_context(viewport={"width":1200,"height":900},color_scheme="light")
        page=context.new_page();page.set_default_timeout(45000);errors=[]
        page.on('pageerror', lambda error:errors.append(str(error)))
        try:
            for _ in range(2):
                page.goto(origin+'predecessor.html')
                page.locator('#dependency-search').fill('math')
                expect(page.locator('#search-results > li')).to_have_count(30)
            require(requests.count('/site/dependencies.mjs')==1,'Predecessor was not reused from Chromium HTTP cache')
            page.goto(origin+'dependencies.html')
            page.locator('#dependency-search').fill('math')
            expect(page.locator('#search-results > li')).to_have_count(39)
            for name in ('dependencies.mjs','dependency-model.mjs','dependencies.css','brand.css'):
                require(any(urlsplit(url).path=='/site/'+name and re.fullmatch('[0-9a-f]{64}',parse_qs(urlsplit(url).query).get('site-release',[''])[0]) for url in requests), 'Missing versioned request: '+name)
            require(not errors, 'Warm-release page errors: '+str(errors))
            result['page_errors']=errors
            result['steps'].extend(['controlled predecessor shows 30 entries with HTTP cache enabled', 'second predecessor visit reuses cached module', 'new entry, transitive module and styles select release URLs', 'same browser then renders all 39 entries'])
            shot=output/'warm-asset-release.png';page.screenshot(path=str(shot))
            result['screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
        finally: context.close()


def check_flow(page, origin, expect, result):
    def ready():
        expect(page.locator("#search")).to_be_enabled(timeout=45000)
        expect(page.locator("#counts > article")).to_have_count(3)
        expect(page.locator("#coefficient-state")).to_contain_text("SHA-256 and byte count verified")
        expect(page.locator("#query-note")).to_contain_text("performs no live tip check")

    page.goto(origin+"workspace.html?q=PROOF.md&repository=Math-&path=coefficients%2Fside24_v1%2F&from=research#inventory")
    ready()
    expect(page.locator("#search")).to_have_value("PROOF.md")
    expect(page.locator("#repository-filter")).to_have_value("Math-")
    expect(page.locator("#path-filter")).to_have_value("coefficients/side24_v1/")
    expect(page.locator("#catalog tr")).to_have_count(1)
    expect(page.locator("#more")).to_be_hidden()
    expect(page.locator("#catalog a")).to_contain_text("coefficients/side24_v1/PROOF.md")
    link=page.locator("#catalog-link").get_attribute("href")
    require(set(parse_qs(urlsplit(link).query))=={"q","repository","path"},"Search link has unexpected parameters")
    require(urlsplit(link).fragment=="inventory","Search link lost inventory fragment")
    result["steps"].append("combined bookmark restores exact source; share link omits unrelated parameters")

    page.locator("#search").fill("nothing-matches-this-example")
    expect(page.locator("#catalog tr")).to_have_count(0)
    expect(page.locator("#more")).to_be_hidden()
    expect(page.locator("#inventory-state")).to_contain_text("clear filters")
    page.locator("#search").press("Tab")
    expect(page.locator("#catalog-clear")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator("#catalog-link")).to_be_focused()
    page.keyboard.press("Shift+Tab")
    page.keyboard.press("Enter")
    expect(page.locator("#search")).to_be_focused()
    expect(page.locator("#catalog tr")).to_have_count(50)
    expect(page.locator("#more")).to_be_visible()
    page.locator("#more").click()
    expect(page.locator("#catalog tr")).to_have_count(100)
    expect(page.locator("#repository-filter")).to_have_value("")
    expect(page.locator("#path-filter")).to_have_value("")
    result["steps"].append("zero-result recovery, Tab order and keyboard Clear focus")

    page.locator("#search").fill("SIDE24")
    expect(page.locator("#inventory-state")).to_contain_text("96 matches")
    page.locator("#catalog-link").click()
    ready()
    require(parse_qs(urlsplit(page.url).query)=={"q":["SIDE24"]},"Search link did not restore SIDE24 only")
    page.reload();ready()
    expect(page.locator("#search")).to_have_value("SIDE24")
    expect(page.locator("#inventory-state")).to_contain_text("96 matches")
    result["steps"].append("explicit search link and refresh retain results")

    # Real fragment navigation creates a history entry; typing replaces it.
    page.get_by_role("link",name="Recorded status",exact=True).click()
    page.locator("#search").fill("EC-014")
    require(urlsplit(page.url).fragment=="board","Board navigation was overwritten")
    page.go_back()
    expect(page.locator("#search")).to_have_value("SIDE24")
    require(urlsplit(page.url).fragment=="inventory","Back navigation lost inventory fragment")
    page.go_forward()
    expect(page.locator("#search")).to_have_value("EC-014")
    require(urlsplit(page.url).fragment=="board","Board navigation was overwritten")
    result["steps"].append("Back/Forward restore filters and preserve section navigation")

    # A compact, meaningful final view keeps the controls and source row legible.
    page.locator("#repository-filter").select_option("Math-")
    page.locator("#path-filter").fill("coefficients/side24_v1/")
    page.locator("#search").fill("PROOF.md")
    expect(page.locator("#catalog tr")).to_have_count(1)
    expect(page.locator("#more")).to_be_hidden()
    page.locator("#inventory").scroll_into_view_if_needed()
    result["layout"]=page.evaluate("""() => ({viewport: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
      tableWidth: document.querySelector('.table-wrap').scrollWidth,
      tableViewport: document.querySelector('.table-wrap').clientWidth})""")
    require(result["layout"]["documentWidth"]<=result["layout"]["viewport"],f"Document overflow: {result['layout']}")
    result["steps"].append("no document overflow; source table measured separately")



def check_latest_work_flow(page, origin, expect, result, output):
    # Static navigation must remain usable if remote source fetching is unavailable.
    remote_requests=[]
    def refuse_remote(route):
        remote_requests.append(route.request.url)
        route.abort()
    page.route("https://raw.githubusercontent.com/**",refuse_remote)
    try:
        page.goto(origin+"index.html")
        entry=page.get_by_role("link",name="Read the latest public work →",exact=True)
        entry.focus();page.keyboard.press("Enter")
        expect(page.locator("#reading-cut-20261009")).to_be_focused()
        expect(page.locator("#reading-cut-20261009 time")).to_have_attribute("datetime","2026-10-09T00:05:00Z")
        expect(page.locator("#reading-cut-20261009 .latest-card")).to_have_count(1)
        expect(page.locator("#reading-cut-20261009 .card-status dt")).to_have_count(3)
        expect(page.locator("#finite-h0-lifetimes")).to_contain_text("Organizational-independence credit is zero")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"9 October cut entry overflow")
        newest_shot=output/f'{result["case"]}-newest-cut.png'
        page.locator("#reading-cut-20261009").screenshot(path=str(newest_shot))
        result["newest_cut_screenshot"]={"path":newest_shot.name,"sha256":sha256(newest_shot.read_bytes()).hexdigest()}
        october_entry=page.get_by_role("link",name="7 October 18:37 UTC reading cut below",exact=True)
        october_entry.focus();page.keyboard.press("Enter")
        expect(page.locator("#reading-cut-20261007")).to_be_focused()
        expect(page.locator("#reading-cut-20261007 time")).to_have_attribute("datetime","2026-10-07T18:37:14Z")
        expect(page.locator("#reading-cut-20261007 .latest-card")).to_have_count(3)
        expect(page.locator("#two-thirds-term")).to_contain_text("does not present them as landed")
        sampling_entry=page.get_by_role("link",name="06:20 UTC grid-sampling reading cut below",exact=True)
        sampling_entry.focus();page.keyboard.press("Enter")
        expect(page.locator("#shrinking-bin-sampling")).to_be_focused()
        previous=page.get_by_role("link",name="00:18 UTC designated-pair reading cut below",exact=True)
        previous.focus();page.keyboard.press("Enter")
        expect(page.locator("#pair-endpoint-rate")).to_be_focused()
        earlier=page.get_by_role("link",name="20:01 UTC occurrence, density and pair-failure cut below",exact=True)
        earlier.focus();page.keyboard.press("Enter")
        expect(page.locator("#reading-addendum")).to_be_focused()
        historical=page.get_by_role("link",name="18:00 UTC reading cut below",exact=True)
        historical.focus();page.keyboard.press("Enter")
        expect(page.locator("#latest-heading")).to_have_text("Latest public work")
        expect(page.locator("#latest-work")).to_be_focused()
        def fragment_focus_indicator(selector):
            style=page.locator(selector).evaluate("el => { const s=getComputedStyle(el); return {width:s.outlineWidth, style:s.outlineStyle, color:s.outlineColor, offset:s.outlineOffset, visible:el.matches(':focus-visible')}; }")
            expected_color="rgb(16, 85, 141)" if result["color_scheme"]=="light" else "rgb(233, 198, 117)"
            require(style=={"width":"3px","style":"solid","color":expected_color,"offset":"4px","visible":True},f"Missing authored fragment focus: {selector}: {style}")
            return style
        result["latest_focus_indicator"]=fragment_focus_indicator("#latest-work")
        expect(page.locator("#latest-work time")).to_have_attribute("datetime","2026-10-03T18:00:00Z")
        expect(page.locator(".latest-chain > li")).to_have_count(3)
        expect(page.locator("#actual-bars")).to_contain_text("P, CAP, E1, E2, REC, CUB and ELDER")
        expect(page.locator("#strict-bar-coefficient")).to_contain_text("integration in progress at this cut")
        expect(page.locator("#actual-bars a[data-source-kind=pinned]")).to_have_count(3)
        expect(page.locator("#strict-bar-coefficient a[data-source-kind=pinned]")).to_have_count(2)
        expect(page.locator("#latest-work .latest-boundary")).to_contain_text("Conjecture 7 remains open")
        expect(page.locator("#latest-work .latest-card").first).to_contain_text("Landed at this cut")
        expect(page.locator("#latest-work .latest-card").first).to_contain_text("I1–I4")
        issue_card=page.locator("#latest-work .latest-card").nth(1)
        expect(issue_card).to_contain_text("C81–C82 concern the soft model")
        expect(issue_card).to_contain_text("C83–C84 control actual/contact field quantities")
        expect(issue_card.get_by_role("link",name="C84: coupled transverse-sector transfer",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5960630207")
        expect(issue_card).to_contain_text("same weighted observable")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Latest work entry overflow")
        shot=output/f'{result["case"]}-latest-entry.png'
        page.screenshot(path=str(shot))
        result["latest_entry_screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
        full=output/f'{result["case"]}-latest-full.png'
        page.locator("#latest-work").screenshot(path=str(full))
        result["latest_full_screenshot"]={"path":full.name,"sha256":sha256(full.read_bytes()).hexdigest()}
        summary=page.locator("#latest-work .latest-identities summary")
        summary.focus();page.keyboard.press("Enter")
        expect(page.locator("#latest-work .latest-identities")).to_have_attribute("open","")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded source identities overflow")
        page.keyboard.press("Enter");page.keyboard.press("Enter");page.keyboard.press("Enter")
        expect(summary).to_be_focused()
        require(page.locator("#latest-work .latest-identities").get_attribute("open") is None,"Identity disclosure did not close")
        newer=page.get_by_role("link",name="upstream routes below",exact=True)
        newer.focus();page.keyboard.press("Enter")
        expect(page.locator("#newer-work")).to_be_focused()
        result["upstream_focus_indicator"]=fragment_focus_indicator("#newer-work")
        expect(page.locator("#newer-work")).to_contain_text("Mutable upstream navigation")
        page.go_back()
        require(urlsplit(page.url).fragment=="latest-work","Back lost latest-work anchor")
        page.go_forward()
        require(urlsplit(page.url).fragment=="newer-work","Forward lost upstream anchor")
        require(not remote_requests,"Static latest-work reading route unexpectedly fetched a remote source")
        result["latest-source-requests"]=remote_requests
        result["steps"].extend(["Home keyboard route reaches the 9 October cut (screenshotted), then the 7 October cut, then C131/C132, then C124 and both dated historical reading anchors with focus", "landed source packets and issue-only scopes remain visible", "source identities repeatedly open/close without overflow", "upstream navigation and Back/Forward preserve anchors", "static reading path remains usable with remote source requests refused"])
    finally:
        page.unroute("https://raw.githubusercontent.com/**",refuse_remote)


def check_reading_addendum_flow(page, origin, expect, result, output):
    remote_requests=[]
    def refuse_remote(route):
        remote_requests.append(route.request.url)
        route.abort()
    page.route("https://**/*",refuse_remote)
    try:
        page.goto(origin+"research.html")
        jump=page.get_by_role("link",name="Latest public work",exact=True)
        jump.focus();page.keyboard.press("Enter")
        newest=page.locator("#reading-cut-20261009")
        expect(newest).to_be_focused()
        require(newest.evaluate('el => parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),"9 October fragment lacks visible focus")
        newest_details=page.locator("#reading-cut-20261009-details");newest_summary=newest_details.locator("summary")
        require(newest_details.get_attribute("open") is None,"9 October detail must start collapsed")
        newest_summary.focus();page.keyboard.press("Enter")
        expect(newest_details).to_have_attribute("open","")
        expect(newest_details.locator("a[data-source-kind=pinned]")).to_have_count(4)
        expect(newest_details.locator("a[data-source-kind=pinned]").first).to_be_visible()
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded 9 October reading overflow")
        newest_sources=output/f'{result["case"]}-newest-cut-sources.png'
        newest_details.screenshot(path=str(newest_sources))
        result["newest_cut_sources_screenshot"]={"path":newest_sources.name,"sha256":sha256(newest_sources.read_bytes()).hexdigest()}
        newest_summary.focus();page.keyboard.press("Enter")
        require(newest_details.get_attribute("open") is None,"9 October detail did not close")
        status_fonts=page.locator("#finite-h0-lifetimes .card-status dd").evaluate_all("els => els.map(el => getComputedStyle(el).fontFamily)")
        card_text_font=page.locator("#finite-h0-lifetimes > p:not(.card-kicker)").first.evaluate("el => getComputedStyle(el).fontFamily")
        require(len(status_fonts)==3 and not any("monospace" in font for font in status_fonts),f"Card-status rows must not be monospace: {status_fonts}")
        require(all(font==card_text_font for font in status_fonts),f"Card-status rows must use the card text face {card_text_font!r}: {status_fonts}")
        status_widths=page.locator("#finite-h0-lifetimes .card-status dd").evaluate_all("els => els.map(el => el.getBoundingClientRect().width)")
        text_width=page.locator("#finite-h0-lifetimes > p:not(.card-kicker)").first.evaluate("el => el.getBoundingClientRect().width")
        require(all(width<=text_width+1 for width in status_widths),f"Card-status rows exceed the card text measure {text_width}: {status_widths}")
        status_sizes=page.locator("#finite-h0-lifetimes .card-status dd").evaluate_all("els => els.map(el => getComputedStyle(el).fontSize)")
        card_text_size=page.locator("#finite-h0-lifetimes > p:not(.card-kicker)").first.evaluate("el => getComputedStyle(el).fontSize")
        require(all(size==card_text_size for size in status_sizes),f"Card-status rows must use the card text size {card_text_size!r}: {status_sizes}")
        overflowing=page.locator("#finite-h0-lifetimes p").evaluate_all("els => els.filter(el => el.scrollWidth > el.clientWidth + 1).map(el => el.textContent.slice(0, 40))")
        require(not overflowing,f"9 October card paragraphs overflow their box: {overflowing}")
        result["card_status_font_family"]=status_fonts
        october_entry=page.get_by_role("link",name="7 October 18:37 UTC reading cut below",exact=True)
        october_entry.focus();page.keyboard.press("Enter")
        october=page.locator("#reading-cut-20261007")
        expect(october).to_be_focused()
        require(october.evaluate('el => parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),"7 October fragment lacks visible focus")
        october_details=page.locator("#reading-cut-20261007-details");october_summary=october_details.locator("summary")
        require(october_details.get_attribute("open") is None,"7 October detail must start collapsed")
        october_summary.focus();page.keyboard.press("Enter")
        expect(october_details).to_have_attribute("open","")
        expect(october_details.locator("a[data-source-kind=pinned]")).to_have_count(10)
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded 7 October reading overflow")
        october_summary.focus();page.keyboard.press("Enter")
        require(october_details.get_attribute("open") is None,"7 October detail did not close")
        sampling_entry=page.get_by_role("link",name="06:20 UTC grid-sampling reading cut below",exact=True)
        sampling_entry.focus();page.keyboard.press("Enter")
        sampling=page.locator("#shrinking-bin-sampling")
        expect(sampling).to_be_focused()
        require(sampling.evaluate('el => parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),"Sampling fragment lacks visible focus")
        expect(sampling.locator("time")).to_have_attribute("datetime","2026-10-04T06:20:32Z")
        expect(sampling.locator(".latest-card")).to_have_count(2)
        expect(sampling).to_contain_text("C131 · EXACT VERTEX SAMPLES")
        expect(sampling).to_contain_text("C132 · CONDITIONAL SPECTRAL EXTENSION")
        expect(sampling).to_contain_text("h²√log(1/h) = o(τ_h)")
        expect(sampling).to_contain_text("small failure probability alone is insufficient")
        expect(sampling).to_contain_text("not a certificate for the current numerical sampler")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Collapsed sampling reading overflow")
        shot=output/f'{result["case"]}-sampling.png'
        sampling.screenshot(path=str(shot))
        result["sampling_screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
        sampling_details=page.locator("#sampling-details");sampling_summary=sampling_details.locator("summary")
        require(sampling_details.get_attribute("open") is None,"Sampling detail must start collapsed")
        sampling_summary.focus();page.keyboard.press("Enter")
        expect(sampling_details).to_have_attribute("open","")
        sampling_sources=[
            ("https://github.com/d6g8k5htny-coder/main/blob/5a7a41daac2e04ea513b97d5fae10a7334f422da/experiments/periodic_h0/EXACT_SAMPLE_SHRINKING_BIN.md","3d938d2223d99f7a8dbaa005b9eb3769813a6f8099a7780200ef048049637c82"),
            ("https://github.com/d6g8k5htny-coder/main/blob/ad7ea9156138911f44b91bcc0e9497a4ba53f181/experiments/periodic_h0/spectral_shrinking_bins/PROOF.md","fa6d443fd5eeeb21c0d311de4d11eb5b406758f3c5ff09a222d02ba54bd0752c"),
            ("https://github.com/d6g8k5htny-coder/main/blob/ad7ea9156138911f44b91bcc0e9497a4ba53f181/experiments/periodic_h0/spectral_shrinking_bins/SOURCES.json","783319599514159bd8e4ca6a4dd02021063d6c547e2c80cb3a8ae1ec1212ccb3"),
        ]
        sampling_pins=sampling_details.locator("a[data-source-kind=pinned]")
        expect(sampling_pins).to_have_count(3)
        for index,(source_url,digest) in enumerate(sampling_sources):
            expect(sampling_pins.nth(index)).to_have_attribute("href",source_url)
            expect(sampling_pins.nth(index)).to_be_visible()
            expect(sampling_pins.nth(index).locator("..")).to_contain_text(digest)
        expect(sampling_details).to_contain_text("d_n = o(τ_n)")
        expect(sampling_details).to_contain_text("n²b_n + √b_n = o(τ_n^(2/3))")
        expect(sampling_details).to_contain_text("C131 does not consume C6")
        expect(sampling_details).to_contain_text("Organizational-independence credit is zero")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded sampling reading overflow")
        expanded=output/f'{result["case"]}-sampling-sources.png'
        sampling_details.screenshot(path=str(expanded))
        result["sampling_sources_screenshot"]={"path":expanded.name,"sha256":sha256(expanded.read_bytes()).hexdigest()}
        sampling_summary.focus();page.keyboard.press("Enter")
        require(sampling_details.get_attribute("open") is None,"Sampling detail did not close")
        expect(sampling_summary).to_be_focused()
        previous=page.get_by_role("link",name="00:18 UTC designated-pair reading cut below",exact=True)
        previous.focus();page.keyboard.press("Enter")
        endpoint=page.locator("#pair-endpoint-rate")
        expect(endpoint).to_be_focused()
        require(endpoint.evaluate('el => parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),"C124 fragment lacks visible focus")
        expect(endpoint.locator("time")).to_have_attribute("datetime","2026-10-04T00:18:14Z")
        expect(endpoint).to_contain_text("physical k = 1")
        expect(endpoint).to_contain_text("1 − p_r = r³(α₁ + α₂) + O(r^(11/3) log(1/r)^(8/3))")
        expect(endpoint).to_contain_text("does not establish a once-counted replacement-bar measure or a lifetime-density rate")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Collapsed C124 reading overflow")
        shot=output/f'{result["case"]}-pair-endpoint.png'
        endpoint.screenshot(path=str(shot))
        result["pair_endpoint_screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
        details=page.locator("#pair-endpoint-details");summary=details.locator("summary")
        require(details.get_attribute("open") is None,"C124 detail must start collapsed")
        summary.focus();page.keyboard.press("Enter")
        expect(details).to_have_attribute("open","")
        pinned=details.locator("a[data-source-kind=pinned]")
        expect(pinned).to_have_count(5)
        prefix="https://github.com/d6g8k5htny-coder/Math-/blob/bbe85e270f2c8b747f2d5d9477c86e86e323fe15/frontiers/planar_soft_layer_chain_20261003/"
        identities=[
            ("C124/PROOF.md","90148657397dcce31e8039afa9015c022f74b2ed0d41e75f2f2339074a8a520f"),
            ("C124/REVIEW.md","19c405c68a93c3f5506629f0ebf1ffa78d5c25ad92f17fc4fd04ef0066deb6cf"),
            ("C124/REVIEW_CLAUDE.md","1ebe3e0faa30d1e2f81a4a6f292bf953811ca9ea8efa4b12f2a9ac6f4a2d5f2b"),
            ("C124/SOURCE_IDENTITIES.json","a0371bd1985358820250e0d75936d6ec5112f165824df7a9f6e29e25e3f46f6d"),
            ("SOURCES.json",None),
        ]
        for index,(path,digest) in enumerate(identities):
            expect(pinned.nth(index)).to_have_attribute("href",prefix+path)
            expect(pinned.nth(index)).to_be_visible()
            if digest: expect(pinned.nth(index).locator("..")).to_contain_text(digest)
        expect(details).to_contain_text("not total variation of transported real height/location marks")
        expect(details).to_contain_text("Organizational-independence credit is zero")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded C124 reading overflow")
        expanded=output/f'{result["case"]}-pair-endpoint-sources.png'
        details.screenshot(path=str(expanded))
        result["pair_endpoint_sources_screenshot"]={"path":expanded.name,"sha256":sha256(expanded.read_bytes()).hexdigest()}
        summary.focus();page.keyboard.press("Enter")
        require(details.get_attribute("open") is None,"C124 detail did not close")
        expect(summary).to_be_focused()
        earlier=page.get_by_role("link",name="20:01 UTC occurrence, density and pair-failure cut below",exact=True)
        earlier.focus();page.keyboard.press("Enter")
        expect(page.locator("#reading-addendum")).to_be_focused()
        require(page.locator("#reading-addendum").evaluate('el => parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),"Addendum fragment lacks visible focus")
        expect(page.locator("#reading-addendum time")).to_have_attribute("datetime","2026-10-03T20:01:00Z")
        expect(page.locator("#latest-work time")).to_have_attribute("datetime","2026-10-03T18:00:00Z")
        expect(page.locator("#reading-addendum .latest-card")).to_have_count(2)
        expect(page.locator("#reading-addendum a[data-source-kind=pinned]")).to_have_count(5)
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Collapsed addendum overflow")
        shot=output/f'{result["case"]}-reading-addendum.png'
        page.locator("#reading-addendum").screenshot(path=str(shot))
        result["reading_addendum_screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
        for identity in ("bar-occurrence-density-details","designated-pair-failure-details"):
            details=page.locator("#"+identity);summary=details.locator("summary")
            require(details.get_attribute("open") is None,"Addendum detail must start collapsed")
            summary.focus();page.keyboard.press("Enter")
            expect(details).to_have_attribute("open","")
            require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Expanded addendum overflow")
            expect(details.locator("a").first).to_be_visible()
            page.keyboard.press("Enter")
            require(details.get_attribute("open") is None,"Addendum detail did not close")
            expect(summary).to_be_focused()
        historical=page.get_by_role("link",name="18:00 UTC reading cut below",exact=True)
        historical.focus();page.keyboard.press("Enter")
        expect(page.locator("#latest-work")).to_be_focused()
        later=page.get_by_role("link",name="Read the later occurrence, density and pair-failure addendum ↑",exact=True)
        later.focus();page.keyboard.press("Enter")
        expect(page.locator("#reading-addendum")).to_be_focused()
        require(not remote_requests,"Static addendum unexpectedly requested a remote source")
        result["addendum_source_requests"]=remote_requests
        static=page.context.browser.new_context(viewport=result["viewport"],color_scheme=result["color_scheme"],java_script_enabled=False)
        try:
            fallback=static.new_page();fallback.goto(origin+"research.html#shrinking-bin-sampling")
            fallback_sampling=fallback.locator("#sampling-details")
            fallback_sampling.locator("summary").press("Enter")
            expect(fallback_sampling).to_have_attribute("open","")
            fallback_pins=fallback_sampling.locator("a[data-source-kind=pinned]")
            expect(fallback_pins).to_have_count(3)
            for index,(source_url,digest) in enumerate(sampling_sources):
                expect(fallback_pins.nth(index)).to_have_attribute("href",source_url)
                expect(fallback_pins.nth(index)).to_be_visible()
                expect(fallback_pins.nth(index).locator("..")).to_contain_text(digest)
            expect(fallback_sampling).to_contain_text("n²b_n + √b_n = o(τ_n^(2/3))")
            endpoint_details=fallback.locator("#pair-endpoint-details")
            endpoint_details.locator("summary").press("Enter")
            expect(endpoint_details).to_have_attribute("open","")
            expect(endpoint_details.locator("a[data-source-kind=pinned]")).to_have_count(5)
            expect(endpoint_details.locator("a[data-source-kind=pinned]").first).to_be_visible()
            for identity in ("bar-occurrence-density-details","designated-pair-failure-details"):
                details=fallback.locator("#"+identity)
                details.locator("summary").press("Enter")
                expect(details).to_have_attribute("open","")
                expect(details.locator("a").first).to_be_visible()
            require(fallback.evaluate("document.documentElement.scrollWidth <= innerWidth"),"JavaScript-off addendum overflow")
        finally:
            static.close()
        result["steps"].append("9 October cut: focus, collapsed four-pin disclosure opened (screenshotted) and closed by keyboard, card-status rows in the card text face, size and measure, card paragraphs without overflow; 7 October cut reached by its link; C131/C132 sampling scope, expectation/error/count conditions, three exact sources/bound hashes, keyboard disclosure and JavaScript-off access checked; C124 and both historical cuts preserved")
    finally:
        page.unroute("https://**/*",refuse_remote)


def check_curvature_export_flow(page, origin, expect, result, output):
    page.goto(origin+'explore.html?s=-1&R=1&unrelated=do-not-cite#peaks')
    download=page.get_by_role('button',name='Download curvature SVG',exact=True)
    expect(download).to_be_enabled()
    started=datetime.now(timezone.utc)
    download.focus()
    with page.expect_download() as pending:
        page.keyboard.press('Enter')
    exported=output/f'{result["case"]}-curvature.svg'
    pending.value.save_as(str(exported))
    data=exported.read_bytes()
    require(b'<!DOCTYPE' not in data and b'<!ENTITY' not in data,'SVG includes an external/entity declaration')
    root=ElementTree.fromstring(data)
    ns={'s':'http://www.w3.org/2000/svg'}
    metadata=json.loads(root.find('s:metadata',ns).text)
    require(metadata['schema']=='universal-law/curvature-teaching-figure/v1','Wrong figure metadata schema')
    require(metadata['params']=={'s':-1,'R':1} and metadata['eigenvalues']==[-2,0] and metadata['classification']=='flat','SVG metadata differs from displayed boundary state')
    require(metadata['mode']=='teaching_model' and metadata['verification']=='not_performed','Figure implies research verification')
    source=metadata['source']
    require(source['repository']=='d6g8k5htny-coder/Math-' and source['commit']=='9d7b6802424fb4715b31999066aafca8ee2f3cca' and source['path']=='coefficients/side24_v1/PROOF.md','Figure lost pinned source identity')
    require(source['blob']=='44b66f04f89fcd87383b3603fa69f1feb64cdddd' and source['bytes']==10272 and source['sha256']=='c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769','Figure changed recorded source bytes')
    require(source['url'].startswith('https://github.com/d6g8k5htny-coder/Math-/blob/9d7b6802424fb4715b31999066aafca8ee2f3cca/coefficients/side24_v1/PROOF.md#'),'Figure source URL is mutable or omitted section')
    permalink=urlsplit(metadata['permalink'])
    require(permalink.scheme=='https' and permalink.netloc=='d6g8k5htny-coder.github.io' and permalink.path=='/main/site/explore.html' and parse_qs(permalink.query)=={'s':['-1'],'R':['1']} and permalink.fragment=='peaks','Figure citation copied host or unrelated URL fields')
    generated=datetime.fromisoformat(metadata['generated_at'].replace('Z','+00:00'))
    require(started.timestamp()-2 <= generated.timestamp() <= datetime.now(timezone.utc).timestamp()+2,'Figure generation time is stale or invalid')
    allowed={'svg','title','desc','metadata','rect','polygon','line','text','circle','g'}
    for node in root.iter():
        require(node.tag.startswith('{http://www.w3.org/2000/svg}') and node.tag.rsplit('}',1)[-1] in allowed,'SVG includes unsupported content')
        for key,value in node.attrib.items():
            require(not key.lower().startswith('on') and key.rsplit('}',1)[-1] not in ('href','style','class') and not re.search(r'(?:url|var)\s*\(',value,re.I),'SVG carries executable, external or unresolved presentation')
    markers=[node for node in root.findall('.//s:circle',ns) if float(node.get('r','0'))==9]
    require(len(markers)==1 and float(markers[0].get('cx'))==241 and float(markers[0].get('cy'))==205,'Exported marker differs from the actual boundary geometry')
    require(root.find('s:title',ns) is not None and root.find('s:desc',ns) is not None and root.get('aria-labelledby'),'SVG lost accessible title/description')
    text=' '.join(root.itertext())
    require('teaching' in text.lower() and 'non-certifying' in text and 'Gaussian' in text,'Standalone figure omits teaching limits')
    standalone=page.context.new_page()
    try:
        standalone.set_viewport_size({'width':750,'height':650})
        standalone.goto(exported.resolve().as_uri())
        expect(standalone.locator('svg')).to_be_visible()
        require(standalone.locator('circle').count()==len(markers),'Standalone SVG lost marker')
        clipped=standalone.locator('svg text').evaluate_all("nodes=>nodes.map(node=>({text:node.textContent,box:node.getBBox()})).filter(({box})=>box.x < 0 || box.y < 0 || box.x+box.width > 600 || box.y+box.height > 535).map(({text})=>text)")
        require(not clipped,f'Standalone SVG clips visible labels or provenance: {clipped}')
        shot=output/f'{result["case"]}-curvature-export.png'
        standalone.screenshot(path=str(shot))
        result['curvature_export_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
    finally: standalone.close()
    result['curvature_svg']={'path':exported.name,'sha256':sha256(data).hexdigest(),'metadata':metadata}
    page.context.grant_permissions(['clipboard-read','clipboard-write'],origin=origin.rstrip('/'))
    copy=page.get_by_role('button',name='Copy figure citation',exact=True)
    copy.focus();page.keyboard.press('Enter')
    expect(page.locator('#curvature-export-status')).to_contain_text('Copied figure citation')
    require(page.evaluate('navigator.clipboard.readText()')==page.locator('#curvature-citation-text').input_value(),'Figure citation clipboard differs from displayed text')
    require('unrelated' not in page.locator('#curvature-citation-text').input_value(),'Citation leaked unrelated URL fields')
    page.add_init_script("const mode=sessionStorage.getItem('curvature-clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:mode==='defer'?{writeText:text=>new Promise(resolve=>{window.curvaturePending={text,resolve}})}:undefined})")
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','defer')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks')
    copy.click();expect(copy).to_be_disabled()
    center=page.locator('#curvature-center');center.focus();page.keyboard.press('Home')
    expect(center).to_have_value('-3')
    expect(copy).to_be_disabled()
    page.evaluate('window.curvaturePending.resolve()')
    expect(copy).to_be_enabled()
    require('Copied figure citation' not in page.locator('#curvature-export-status').text_content(),'Old copy claimed success for newly displayed settings')
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','deny')")
    page.goto(origin+'explore.html?s=-1&R=1#peaks');copy.click()
    expect(page.locator('#curvature-export-status')).to_contain_text('manually')
    manual=page.locator('#curvature-figure-citation summary')
    if page.locator('#curvature-figure-citation').get_attribute('open') is None:
        manual.press('Enter')
    expect(page.locator('#curvature-citation-text')).to_be_visible()
    page.evaluate("sessionStorage.setItem('curvature-clipboard-test','absent')")
    page.reload()
    require(page.evaluate('typeof navigator.clipboard')=='undefined','Clipboard absence simulation did not run before the page module')
    expect(copy).to_be_disabled();expect(download).to_be_enabled()
    require(bool(page.locator('#curvature-citation-text').input_value()),'Manual citation missing without clipboard API')
    page.evaluate("sessionStorage.removeItem('curvature-clipboard-test')")
    page.reload()
    require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Figure actions overflowed document')
    page.locator('#curvature-figure-citation summary').press('Enter')
    expect(page.locator('#curvature-citation-text')).to_be_visible()
    shot=output/f'{result["case"]}-curvature-export-controls.png'
    page.locator('#peaks').screenshot(path=str(shot))
    result['curvature_controls_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
    center=page.locator('#curvature-center');spread=page.locator('#curvature-spread')
    center.press('Home');center.press('ArrowRight')
    spread.press('Home');spread.press('ArrowRight');spread.press('ArrowRight')
    expect(center).to_have_value('-2.9');expect(spread).to_have_value('0.2')
    with page.expect_download() as pending:
        download.press('Enter')
    decimal=output/f'{result["case"]}-curvature-decimal.svg'
    pending.value.save_as(str(decimal))
    decimal_data=decimal.read_bytes()
    decimal_metadata=json.loads(ElementTree.fromstring(decimal_data).find('s:metadata',ns).text)
    require(decimal_metadata['params']=={'s':-2.9,'R':0.2} and decimal_metadata['eigenvalues']==[-3.1,-2.7],'Decimal export differs from the slider grid')
    require('-2.6999999999999997' not in decimal_data.decode() and '-2.6999999999999997' not in page.locator('#curvature-citation-text').input_value(),'Floating-point noise leaked into the figure citation or SVG')
    result['curvature_decimal_svg']={'path':decimal.name,'sha256':sha256(decimal_data).hexdigest(),'metadata':decimal_metadata}
    require(bool(page.locator('#curvature-citation-text').input_value()),'Valid current citation missing before capture refusal')
    page.locator('#curvature-diagram circle').evaluate("node=>node.setAttribute('class','unsupported-test-class')")
    download.press('Enter')
    expect(download).to_be_disabled();expect(copy).to_be_disabled()
    expect(page.locator('#curvature-citation-text')).to_have_value('')
    expect(page.locator('#curvature-export-status')).to_contain_text('unavailable')
    center.press('ArrowRight')
    expect(download).to_be_enabled();expect(copy).to_be_enabled()
    require('s = -2.8' in page.locator('#curvature-citation-text').input_value(),'Native redraw did not recover the current citation')
    result['steps'].append('Decimal grid export agrees with the page; refused capture clears manual citation and disables actions; native slider redraw recovers current output')
    result['steps'].append('Actual keyboard SVG download parsed/reopened with exact displayed geometry, concrete presentation and teaching/source metadata; real citation clipboard, denial/absence and pending settings change checked')


def check_other_teaching_exports(page, origin, expect, result, output):
    ns = {'s': 'http://www.w3.org/2000/svg'}
    geometry = {'x', 'y', 'cx', 'cy', 'r', 'rx', 'ry', 'width', 'height',
                'x1', 'y1', 'x2', 'y2', 'points', 'd', 'fill-rule'}
    allowed = {'svg', 'title', 'desc', 'metadata', 'rect', 'polygon', 'line',
               'text', 'circle', 'g'}
    result['other_teaching_svgs'] = []
    result['other_teaching_screenshots'] = []
    result['teaching_css_geometry_observations'] = []

    def css_fixture(rule):
        # Exercise stylesheet overrides through an already allowed same-origin
        # sheet. Inline <style> fixtures would be blocked by the real page CSP.
        return page.evaluate("""rule=>{
            const sheet=Array.from(document.styleSheets).find(s=>s.href && new URL(s.href).pathname.endsWith('/explore.css'));
            if(!sheet)throw new Error('Allowed Explore stylesheet unavailable');
            const index=sheet.insertRule(rule,sheet.cssRules.length);
            return {href:sheet.href,index};
        }""", rule)

    def remove_css_fixture(fixture):
        page.evaluate("""({href,index})=>{
            const sheet=Array.from(document.styleSheets).find(s=>s.href===href);
            if(!sheet)throw new Error('Explore fixture stylesheet unavailable');
            sheet.deleteRule(index);
        }""", fixture)

    def native_button(button):
        # Tab/Shift+Tab makes keyboard focus observable even after an earlier click.
        button.focus(); page.keyboard.press('Tab'); page.keyboard.press('Shift+Tab')
        expect(button).to_be_focused()
        require(button.evaluate("el=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=='none'"),
                'Teaching export button lacks visible keyboard focus')
        page.keyboard.press('Enter')

    def signature(tag, attrs, text):
        return tag, {key: attrs[key] for key in geometry if key in attrs}, text

    def read_svg(kind, params, fragment, label):
        button = page.locator(f'#{kind}-export-svg')
        expect(button).to_be_enabled()
        displayed = page.locator(f'#{kind}-diagram').evaluate("""svg=>({background:getComputedStyle(svg).backgroundColor,
            nodes:Array.from(svg.children).map(n=>({tag:n.localName,
            attrs:Object.fromEntries(Array.from(n.attributes,a=>[a.name,a.value])),
            style:['title','desc'].includes(n.localName)?{}:Object.fromEntries(
                ['fill','stroke','stroke-width','font-family','font-size','text-anchor'].map(k=>[k,getComputedStyle(n).getPropertyValue(k)])),
            text:n.localName==='text'||n.localName==='title'||n.localName==='desc'?n.textContent:null}))})""")
        inline = displayed['nodes']
        started = datetime.now(timezone.utc)
        with page.expect_download() as pending:
            native_button(button)
        path = output / f'{result["case"]}-{kind}-{label}.svg'
        pending.value.save_as(str(path)); data = path.read_bytes()
        require(b'<!DOCTYPE' not in data and b'<!ENTITY' not in data and b'<?xml-stylesheet' not in data,
                'Teaching SVG has external/entity content')
        root = ElementTree.fromstring(data)
        require(root.tag == '{http://www.w3.org/2000/svg}svg', 'Wrong SVG namespace')
        require(root.find('s:title', ns) is not None and root.find('s:desc', ns) is not None
                and root.get('aria-labelledby'), 'Teaching SVG lost accessible title/description')
        for identity in root.get('aria-labelledby').split():
            require(any(node.get('id') == identity for node in root), 'SVG accessibility reference has no target')
        require(any(node.get('x') == '0' and node.get('y') == '0' and node.get('width') == '600'
                    and node.get('height') == root.get('height') and node.get('fill') == displayed['background']
                    for node in root.findall('s:rect', ns)), 'Export background differs from the current theme')
        for node in root.iter():
            require(node.tag.startswith('{http://www.w3.org/2000/svg}')
                    and node.tag.rsplit('}', 1)[-1] in (allowed | ({'path'} if kind == 'pin' else set())), 'Unsupported teaching SVG node')
            for key, value in node.attrib.items():
                require(not key.lower().startswith('on')
                        and key.rsplit('}', 1)[-1] not in ('href', 'style', 'class')
                        and not re.search(r'(?:url|var)\s*\(', value, re.I),
                        'Teaching SVG carries executable, external or unresolved presentation')
        metadata_nodes = root.findall('s:metadata', ns)
        require(len(metadata_nodes) == 1, 'Teaching SVG does not have exactly one metadata record')
        metadata = json.loads(metadata_nodes[0].text)
        require(metadata['schema'] == f'universal-law/{kind}-teaching-figure/v1'
                and metadata['mode'] == 'teaching_model'
                and metadata['verification'] == 'not_performed', 'Wrong teaching/verification boundary')
        require(metadata['params'] == params, 'Exported parameters differ from current native controls')
        permalink = urlsplit(metadata['permalink'])
        require(permalink.scheme == 'https' and permalink.netloc == 'd6g8k5htny-coder.github.io'
                and permalink.path == '/main/site/explore.html' and permalink.fragment == fragment
                and parse_qs(permalink.query, keep_blank_values=True) == {
                    key: [','.join(map(str, value)) if isinstance(value, list) else str(value)]
                    for key, value in params.items()}, 'Teaching permalink leaks host or other figure settings')
        generated = datetime.fromisoformat(metadata['generated_at'].replace('Z', '+00:00'))
        require(started.timestamp() - 2 <= generated.timestamp() <= datetime.now(timezone.utc).timestamp() + 2,
                'Teaching generation timestamp is stale')
        text = ' '.join(root.itertext())
        require('teaching' in text.lower() and 'non-certifying' in text
                and 'not mathematical result acceptance' in text, 'Standalone teaching limits are missing')
        require('unrelated' not in metadata['permalink'] and 'unrelated' not in page.locator(f'#{kind}-citation-text').input_value(),
                'Teaching citation leaked unrelated fields')
        # Each displayed node must survive with exactly the actual geometry and text.
        # Export-only metadata/background/footer may occur between those nodes.
        exported_nodes = list(root.iter())
        candidates = [signature(node.tag.rsplit('}', 1)[-1], node.attrib,
                      node.text if node.tag.rsplit('}', 1)[-1] in ('title', 'desc', 'text') else None)
                      for node in exported_nodes]
        cursor = 0
        for item in inline:
            target = signature(item['tag'], item['attrs'], item['text'])
            while cursor < len(candidates) and candidates[cursor] != target:
                cursor += 1
            require(cursor < len(candidates), f'Export lost displayed {kind} geometry/label: {target}')
            node = exported_nodes[cursor]
            if item['style']:
                require(node.get('fill') == item['style']['fill'] and node.get('stroke') == item['style']['stroke'],
                        'Export colors differ from the current native theme')
                require(float(node.get('stroke-width')) == float(item['style']['stroke-width'].removesuffix('px')),
                        'Export stroke width differs from the displayed primitive')
                if item['tag'] == 'text':
                    require(node.get('font-family') == item['style']['font-family']
                            and float(node.get('font-size')) == float(item['style']['font-size'].removesuffix('px'))
                            and node.get('text-anchor') == item['style']['text-anchor'],
                            'Export omitted concrete current text presentation')
            cursor += 1
        if kind == 'palette':
            objects = params['objects']; valid = len(objects) <= 2
            require(metadata['capacity'] == 1 and metadata['groups'] == 2
                    and metadata['decomposable'] is valid, 'Palette capacity/classification differs')
            expected_parts = [[objects[0]] if objects else [], [objects[1]] if len(objects) == 2 else []] if valid else None
            require(metadata['assignments'] == expected_parts, 'Palette exported a wrong valid partition')
            require(metadata['attempted_placement'] == (None if valid else {
                    'groups': [[1], [2]], 'unplaced': [3], 'valid_partition': False}),
                    'Palette failed triple falsely claims a partition')
            diagram_text = root.findall('.//s:text', ns)
            labels = {(float(n.get('x')), float(n.get('y'))): n.text for n in diagram_text
                      if float(n.get('y', '0')) <= 300}
            group_labels = tuple(labels.get((x, 209)) for x in (125, 333))
            require(group_labels == (('1', '2') if not valid else ('1', '3') if objects else ('empty', 'empty')),
                    'Palette diagram group labels differ from the chosen assignment')
            extra = [(float(n.get('cx')), float(n.get('cy')), float(n.get('r')))
                     for n in root.findall('.//s:circle', ns) if float(n.get('r', '0')) == 26]
            require(extra == ([] if valid else [(510, 203, 26)])
                    and (labels.get((510, 279)) == 'no place') is (not valid),
                    'Palette diagram lost the failed triple or marked a valid pair as failed')
            source = metadata['source']
            require(source == {
                'repository': 'd6g8k5htny-coder/Math-', 'commit': 'd6628da09384728992dcbe6e921cc28ba85aebb0',
                'path': 'frontiers/full_price_20260924/PROOF.md', 'blob': '582180e41dca0ad815ad0f18574df42040912149',
                'bytes': 11352, 'sha256': '87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9',
                'url': 'https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/frontiers/full_price_20260924/PROOF.md#5-sharpness-and-the-exact-demand-boundary'},
                'Palette changed recorded pinned source')
        else:
            r = params['r']
            require(metadata['teaching_constants'] == {'A': 2, 'B': 4, 'rho': 3, 'L': 24, 'b': 1, 'k': 1}
                    and metadata['pins'] == [[-r / 2, 0], [r / 2, 0]]
                    and metadata['height_gap'] == r ** 3 and metadata['height_window'] == [1 - r ** 3, 1]
                    and metadata['annulus'] == [2 * r, 4 * r], 'Pin recorded scales differ from current teaching choices')
            expected_sources = []
            for source_path, blob, size, digest in [
                    ('frontiers/remote_window_20260924/PROOF.md', 'b383bfcc88ec4ad497dff01fb6640e429ba24a84', 18355, 'a332bae9bdc0106ce17047f7e0409cc3d94eb610a0c7b74ba5ba2d01a1620cb7'),
                    ('frontiers/rn_annulus_bridge_20260925/PROOF.md', '6f317515b3d417661f86e2fed09bc7d950899c2b', 16948, 'd55e2c03bb17e7977ff94130cc1ff21e54840cd1e20dc4e41ea9ad52228beb05')]:
                expected_sources.append({'repository': 'd6g8k5htny-coder/Math-',
                    'commit': 'd6628da09384728992dcbe6e921cc28ba85aebb0', 'path': source_path, 'blob': blob,
                    'bytes': size, 'sha256': digest,
                    'url': 'https://github.com/d6g8k5htny-coder/Math-/blob/d6628da09384728992dcbe6e921cc28ba85aebb0/' + source_path})
            require(metadata['sources'] == expected_sources, 'Pin changed recorded pinned sources or their order')
        standalone = page.context.new_page()
        try:
            standalone.set_viewport_size({'width': 750, 'height': 700})
            standalone.goto(path.resolve().as_uri())
            expect(standalone.locator('svg')).to_be_visible()
            clipped = standalone.locator('svg text').evaluate_all("""nodes=>nodes.filter(n=>{
                const b=n.getBBox(),v=n.ownerSVGElement.viewBox.baseVal;
                return b.x<v.x-1||b.y<v.y-1||b.x+b.width>v.x+v.width+1||b.y+b.height>v.y+v.height+1;
                }).map(n=>n.textContent)""")
            require(not clipped, f'Standalone teaching SVG clips labels/provenance: {clipped}')
            if label in ('0.5-annulus', 'triple'):
                shot = output / f'{result["case"]}-{kind}-export.png'; standalone.screenshot(path=str(shot))
                result['other_teaching_screenshots'].append({'path': shot.name, 'sha256': sha256(shot.read_bytes()).hexdigest()})
        finally:
            standalone.close()
        result['other_teaching_svgs'].append({'path': path.name, 'sha256': sha256(data).hexdigest(), 'metadata': metadata})
        return root, metadata

    page.context.grant_permissions(['clipboard-read', 'clipboard-write'], origin=origin.rstrip('/'))
    page.add_init_script("""const mode=sessionStorage.getItem('teaching-clipboard-test');if(mode)
        Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{
            writeText:async()=>{throw new Error('denied')}}:mode==='defer'?{
            writeText:text=>new Promise(resolve=>{window.teachingPending={text,resolve}})}:undefined});""")

    for kind, fragment in [('pin', 'distance'), ('palette', 'groups')]:
        start = origin + 'explore.html?s=-1&R=1&r=0.5&region=annulus&objects=1,2,3&unrelated=do-not-cite#' + fragment
        page.goto(start)
        export = page.locator(f'#{kind}-export-svg'); copy = page.locator(f'#{kind}-copy-citation')
        status = page.locator(f'#{kind}-export-status'); citation = page.locator(f'#{kind}-citation-text')
        details = page.locator(f'#{kind}-figure-citation')
        expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        native_button(copy)
        expect(status).to_contain_text('Copied figure citation')
        require(page.evaluate('navigator.clipboard.readText()') == citation.input_value(), 'Teaching clipboard differs from visible citation')
        if kind == 'pin':
            for r, region in [(0.5, 'annulus'), (0.25, 'annulus'), (0.25, 'remote'), (0.5, 'remote')]:
                if page.locator('#pin-distance').input_value() != str(r):
                    native_button(page.locator('#pin-half'))
                if not page.locator(f'input[name="region"][value="{region}"]').is_checked():
                    page.locator('input[name="region"]:checked').focus(); page.keyboard.press('ArrowRight')
                expect(page.locator('#pin-distance')).to_have_value(str(r))
                expect(page.locator(f'input[name="region"][value="{region}"]')).to_be_checked()
                expect(page.locator('#pin-gap')).to_have_text('0.125000' if r == 0.5 else '0.015625')
                root, metadata = read_svg(kind, {'r': r, 'region': region}, fragment, f'{r}-{region}')
                points = [(float(n.get('cx')), float(n.get('cy')), float(n.get('r')))
                          for n in root.findall('.//s:circle', ns) if float(n.get('r', '0')) == 5]
                require(points == [(202 - r * 19.5, 172, 5), (202 + r * 19.5, 172, 5)], 'Pin markers differ from current scales')
                require(any(n.get('cx') == '202' and n.get('cy') == '172' and n.get('r') == '117'
                            for n in root.findall('.//s:circle', ns)), 'Fixed remote radius moved')
                require(any(float(n.get('x', '-1')) == 448 and float(n.get('y', '-1')) == 78
                            and float(n.get('width', '-1')) == 113 and float(n.get('height', '-1')) == r ** 3 * 860
                            for n in root.findall('.//s:rect', ns)), 'Height rectangle lost the cubic scale')
                paths = root.findall('.//s:path', ns)
                require(len(paths) == (1 if region == 'remote' else 0), 'Wrong region shading geometry')
                if region == 'remote':
                    require(paths[0].get('fill-rule') == 'evenodd' and paths[0].get('d') ==
                            'M24 40H380V302H24Z M202 55a117 117 0 1 0 0 234a117 117 0 1 0 0 -234Z',
                            'Fixed remote shading changed with r')
                else:
                    rings = sorted(float(n.get('r')) for n in root.findall('.//s:circle', ns)
                                   if n.get('cx') == '202' and n.get('cy') == '172' and n.get('r') != '117')
                    require(rings == [r * 78, r * 117, r * 156], 'Annulus radii do not shrink with r')
        else:
            read_svg(kind, {'objects': [1, 2, 3]}, fragment, 'triple')
            page.locator('#object-two').focus(); page.keyboard.press('Space')
            expect(page.locator('#palette-kind')).to_have_text('Everything has a place')
            read_svg(kind, {'objects': [1, 3]}, fragment, 'one-three')
            for selector in ['#object-one', '#object-three']:
                page.locator(selector).focus(); page.keyboard.press('Space')
            expect(page.locator('#palette-kind')).to_have_text('Empty groups satisfy the rule too')
            read_svg(kind, {'objects': []}, fragment, 'empty')

        # History and reload must restore current projection, independently of other figures.
        last_citation = citation.input_value(); page.go_back()
        require(citation.input_value() and citation.input_value() != last_citation, 'Back retained stale teaching citation')
        page.go_forward(); expect(citation).to_have_value(last_citation)
        page.reload(); expect(citation).to_have_value(last_citation)
        native_button(copy); expect(status).to_contain_text('Copied figure citation')
        require(page.evaluate('navigator.clipboard.readText()') == last_citation, 'Reloaded teaching clipboard differs')

        # An old async copy cannot announce success after native change and Back.
        page.evaluate("sessionStorage.setItem('teaching-clipboard-test','defer')"); page.goto(start)
        initial = citation.input_value(); native_button(copy); expect(copy).to_be_disabled()
        if kind == 'pin': native_button(page.locator('#pin-half'))
        else: page.locator('#object-two').focus(); page.keyboard.press('Space')
        require(citation.input_value() != initial, 'Native edit retained pending citation')
        expect(copy).to_be_disabled(); page.go_back(); expect(citation).to_have_value(initial)
        page.evaluate('window.teachingPending.resolve()'); expect(copy).to_be_enabled()
        require('Copied figure citation' not in status.text_content(), 'Pending copy claimed success after history restoration')

        # Back restored this same URL within the deferred fixture's document.
        # Reload installs the denial fixture; goto(start) may only change a hash.
        page.evaluate("sessionStorage.setItem('teaching-clipboard-test','deny')"); page.reload()
        native_button(copy); expect(status).to_contain_text('manually')
        expect(details).to_have_attribute('open', ''); expect(citation).to_be_visible()
        require(bool(citation.input_value()), 'Clipboard refusal lost manual citation')
        page.evaluate("sessionStorage.setItem('teaching-clipboard-test','absent')"); page.reload()
        require(page.evaluate('typeof navigator.clipboard') == 'undefined', 'Clipboard absence simulation did not run before module')
        expect(copy).to_be_disabled(); expect(export).to_be_enabled()
        require(bool(citation.input_value()), 'Absent API lost manual citation')
        page.evaluate("sessionStorage.removeItem('teaching-clipboard-test')"); page.reload()

        # CSS geometry must not silently override the attribute-based diagram.
        css_point = page.locator('#pin-diagram circle.diagram-point').first if kind == 'pin' else page.locator('#palette-diagram circle[cx="125"][cy="202"]')
        original_x = css_point.get_attribute('cx')
        css_point.evaluate("n=>n.style.cx='190px'")
        require(css_point.get_attribute('cx') == original_x and css_point.evaluate("n=>getComputedStyle(n).getPropertyValue('cx')") == '190px',
                'CSS geometry negative fixture did not override the displayed primitive')
        native_button(export)
        expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
        if kind == 'pin': native_button(page.locator('#pin-half'))
        else: page.locator('#object-two').focus(); page.keyboard.press('Space')
        expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        # Stylesheet-only SVG2 auto geometry must not be mistaken for the
        # attribute geometry. No inline style attribute triggers these refusals.
        css_box = page.locator(f'#{kind}-diagram rect').first
        for axis in ['width', 'height']:
            original_dimension = css_box.get_attribute(axis)
            fixture = css_fixture(f'#{kind}-diagram rect{{{axis}:auto}}')
            observation = css_box.evaluate("""(n,axis)=>{
                const box=n.getBBox();
                return {axis,inlineStyle:n.getAttribute('style'),attribute:n.getAttribute(axis),
                    resolved:getComputedStyle(n).getPropertyValue(axis),
                    bbox:{width:box.width,height:box.height}};
            }""", axis)
            result['teaching_css_geometry_observations'].append({'kind':kind,**observation})
            require(observation['inlineStyle'] is None and observation['attribute'] == original_dimension
                    and observation['bbox'][axis] == 0,
                    f'CSS auto fixture did not suppress effective geometry with attributes unchanged: {observation}')
            native_button(export)
            expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
            remove_css_fixture(fixture)
            if kind == 'pin': native_button(page.locator('#pin-half'))
            else: page.locator('#object-two').focus(); page.keyboard.press('Space')
            expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        if kind == 'palette':
            require(css_box.get_attribute('rx') == '12', 'Palette fixture requires the native rounded rectangle')
            fixture = css_fixture('#palette-diagram rect{rx:auto;ry:auto}')
            require(css_box.get_attribute('style') is None
                    and css_box.evaluate("n=>['rx','ry'].map(k=>getComputedStyle(n).getPropertyValue(k))") == ['auto', 'auto'],
                    'CSS auto radii fixture did not remove the displayed native rounding')
            native_button(export)
            expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
            remove_css_fixture(fixture)
            page.locator('#object-two').focus(); page.keyboard.press('Space')
            expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        fixture = css_fixture(f'#{kind}-diagram{{opacity:.5}}')
        require(page.locator(f'#{kind}-diagram').evaluate("n=>getComputedStyle(n).opacity") == '0.5',
                'Root opacity fixture did not alter displayed group compositing')
        native_button(export)
        expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
        remove_css_fixture(fixture)
        if kind == 'pin': native_button(page.locator('#pin-half'))
        else: page.locator('#object-two').focus(); page.keyboard.press('Space')
        expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        if kind == 'pin':
            page.locator('input[name="region"][value="remote"]').check()
            path = page.locator('#pin-diagram path')
            path.evaluate("n=>n.style.d='none'")
            require(path.evaluate("n=>getComputedStyle(n).getPropertyValue('d')") == 'none',
                    'CSS path negative fixture did not hide the displayed path')
            native_button(export)
            expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
            native_button(page.locator('#pin-half')); expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        page.goto(start)

        # Semantic stale geometry is rejected; a native redraw recovers both actions.
        # These intended fixtures need the child serializers' semantic validators.
        if kind == 'pin':
            page.locator('#pin-diagram circle.diagram-point').first.evaluate("n=>n.setAttribute('cx','197.125')")
        else:
            page.locator('#palette-diagram text[x="333"][y="209"]').evaluate("n=>n.textContent='3'")
        native_button(export)
        expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
        expect(status).to_contain_text('unavailable')
        if kind == 'pin': native_button(page.locator('#pin-half'))
        else: page.locator('#object-two').focus(); page.keyboard.press('Space')
        expect(export).to_be_enabled(); expect(copy).to_be_enabled()
        require(bool(citation.input_value()), 'Native redraw did not recover teaching citation')
        if details.get_attribute('open') is None:
            native_button(details.locator('summary'))
        expect(citation).to_be_visible()
        require(page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Teaching controls overflow document')
        shot = output / f'{result["case"]}-{kind}-export-controls.png'
        page.locator('#' + fragment).screenshot(path=str(shot))
        result['other_teaching_screenshots'].append({'path': shot.name, 'sha256': sha256(shot.read_bytes()).hexdigest()})

    # BQ-26: real layout evidence, separate from mocked native-witness unit tests.
    result['pin_height_geometry'] = []
    for region in ['annulus', 'remote']:
        page.goto(origin + f'explore.html?r=0.25&region={region}#distance')
        distance = page.locator('#pin-distance')
        export = page.locator('#pin-export-svg'); copy = page.locator('#pin-copy-citation')
        citation = page.locator('#pin-citation-text')
        rect = page.locator('#pin-diagram rect')

        def height_observation(label):
            observation = rect.evaluate("""async n=>{
                const entry=new URL(document.querySelector('script[src]').src);
                const module=new URL('teaching-export.mjs'+entry.search,entry);
                const {captureTeachingDiagram}=await import(module.href);
                let captureError=null;
                try{captureTeachingDiagram(n.ownerSVGElement);}catch(error){captureError=String(error);}
                const reference=document.createElementNS(n.namespaceURI,'rect');
                reference.setAttribute('height',n.getAttribute('height'));
                return {attribute:n.getAttribute('height'),
                    computed:getComputedStyle(n).getPropertyValue('height'),
                    reflected:n.height.baseVal.value,used:n.getBBox().height,
                    parsed:reference.height.baseVal.value,
                    float32:Math.fround(Number(n.getAttribute('height'))),
                    typed:n.computedStyleMap?.().get('height')?.value,
                    captureError};
            }""")
            result['pin_height_geometry'].append({'region':region,'case':label,**observation})
            return observation

        for label, key in [('0.25-before', None), ('0.24-rounded', 'ArrowLeft'), ('0.25-after', 'ArrowRight')]:
            if key:
                distance.focus(); page.keyboard.press(key)
            expect(distance).to_have_value('0.24' if key == 'ArrowLeft' else '0.25')
            observation = height_observation(label)
            expect(export).to_be_enabled(); expect(copy).to_be_enabled()
            require(0 < observation['used'] <= 600
                    and observation['used'] == observation['reflected'] == observation['parsed'],
                    'Native used-height witness does not match the independently parsed attribute')
            if key == 'ArrowLeft':
                require(observation['attribute'] == '11.888639999999995'
                        and observation['computed'] == '11.8886px'
                        and abs(float(observation['computed'][:-2]) - float(observation['attribute'])) > 1e-6,
                        'Native rounding regression did not exercise the original refusal')
                with page.expect_download() as pending:
                    native_button(export)
                path = output / f'{result["case"]}-pin-0.24-{region}.svg'
                pending.value.save_as(str(path)); data = path.read_bytes()
                root = ElementTree.fromstring(data)
                metadata = json.loads(root.find('s:metadata', ns).text)
                require(metadata['params'] == {'r':0.24,'region':region}
                        and metadata['height_gap'] == 0.013824
                        and metadata['height_window'] == [0.986176,1]
                        and metadata['verification'] == 'not_performed',
                        'Rounded-height export changed canonical teaching metadata')
                require(any(n.get('height') == observation['attribute'] and n.get('width') == '113'
                            for n in root.findall('s:rect', ns)),
                        'Rounded-height export rewrote source geometry')
                require('non-certifying' in data.decode() and 'No field, count, probability, lifetime or theorem computation.' in data.decode(),
                        'Rounded-height export lost teaching/non-certification labels')
                result['other_teaching_svgs'].append({'path':path.name,'sha256':sha256(data).hexdigest(),'metadata':metadata})

        distance.focus(); page.keyboard.press('ArrowLeft')
        expect(distance).to_have_value('0.24')
        native = height_observation('native-before-overrides')
        for css_height in ['11.88861px', '11.88864px', '11.889px', '0px', 'auto']:
            fixture = css_fixture(f'#pin-diagram rect{{height:{css_height}}}')
            try:
                changed = height_observation('CSS-height-' + css_height)
                require(changed['attribute'] == native['attribute'] and changed['reflected'] == native['reflected']
                        and changed['used'] != native['used'], 'CSS fixture did not change only the used height')
                if css_height in ['11.88861px', '11.88864px']:
                    require(changed['computed'] == native['computed'],
                            'Nearby CSS override did not share the native rounded string')
                native_button(copy)
                expect(export).to_be_disabled(); expect(copy).to_be_disabled(); expect(citation).to_have_value('')
            finally:
                remove_css_fixture(fixture)
            distance.focus(); page.keyboard.press('ArrowRight'); page.keyboard.press('ArrowLeft')
            expect(export).to_be_enabled(); expect(copy).to_be_enabled()

        # Full supported hundredth-tick range, using actual native input/layout.
        # This is headless Chromium, not emulated computed styles or native zoom.
        distance.focus(); page.keyboard.press('Home')
        for tick in range(5, 61):
            if tick > 5:
                page.keyboard.press('ArrowRight')
            expect(distance).to_have_value(str(tick / 100))
            expect(export).to_be_enabled(); expect(copy).to_be_enabled()
            observed = height_observation(f'tick-{tick}')
            require(0 < observed['used'] <= 600 and observed['used'] == observed['reflected'] == observed['parsed'],
                    'Supported slider tick lost native geometry identity')
    result['steps'].append('BQ-26 native .25/.24/.25 recovery and SVG metadata, same-rounded-string real CSS override refusal, zero/auto refusal, and all 56 ticks in both highlights checked')

    # Fresh Explore page, never the previous Formal page. Static output describes
    # its true starting values even when the URL asks for a different example.
    static = page.context.browser.new_context(viewport=result['viewport'], color_scheme=result['color_scheme'], java_script_enabled=False)
    try:
        fallback = static.new_page()
        fallback.goto(origin + 'explore.html?r=0.25&region=remote&objects=#groups')
        expect(fallback.locator('#pin-distance')).to_have_value('0.5')
        expect(fallback.locator('#pin-distance')).to_be_disabled()
        expect(fallback.locator('input[name="region"][value="annulus"]')).to_be_checked()
        expect(fallback.locator('#palette-kind')).to_have_text('One object has no place')
        for selector in ['#object-one', '#object-two', '#object-three']:
            expect(fallback.locator(selector)).to_be_checked(); expect(fallback.locator(selector)).to_be_disabled()
        expect(fallback.locator('#explore-state-status')).to_contain_text('starting examples')
        for kind in ['pin', 'palette']:
            expect(fallback.locator(f'#{kind}-export-svg')).to_be_disabled()
            expect(fallback.locator(f'#{kind}-copy-citation')).to_be_disabled()
            fallback.locator(f'#{kind}-figure-citation summary').press('Enter')
            expect(fallback.locator(f'#{kind}-citation-text')).to_be_visible()
            require(bool(fallback.locator(f'#{kind}-citation-text').input_value()), 'No-JS starting citation is missing')
        require('r = 0.5;' in fallback.locator('#pin-citation-text').input_value()
                and '?r=0.5&region=annulus#distance' in fallback.locator('#pin-citation-text').input_value(),
                'No-JS pin citation falsely claims restored URL parameters')
        require('Starting selection: objects 1, 2, 3' in fallback.locator('#palette-citation-text').input_value()
                and '?objects=1,2,3#groups' in fallback.locator('#palette-citation-text').input_value(),
                'No-JS palette citation falsely claims the URL empty selection')
    finally:
        static.close()
    result['steps'].append('Pin (.5/.25, annulus/remote) and finite obstruction (triple/pair/empty) keyboard SVGs parsed/reopened; exact displayed geometry, current metadata, projected citations, real clipboard, refusal/absence, pending copy after native change/Back, history/reload, stale capture recovery and fresh Explore no-JS fallback checked')


def check_curvature_preset_flow(page, origin, expect, result, output):
    page.goto(origin+'explore.html?s=0.5&R=2&r=0.25&region=remote&objects=1,3&from=reader#peaks')
    center=page.locator('#curvature-center');spread=page.locator('#curvature-spread')
    expect(spread).to_have_value('2')
    spread.focus();page.keyboard.press('Tab')
    ns={'s':'http://www.w3.org/2000/svg'}
    rows=[
        ('maximum',-2,[-3,-1],'peak','Downward in both directions',245),
        ('saddle',0,[-1,1],'saddle','Down one way. Up the other.',165),
        ('singular-boundary',-1,[-2,0],'flat','At least one direction is flat',205),
        ('minimum',2,[1,3],'bowl','Upward in both directions',85),
    ]
    display=lambda value:f'{value:.1f}'.replace('-','−')
    for index,(name,s,eigenvalues,kind,label,cy) in enumerate(rows):
        button=page.locator(f'[data-curvature-preset="{name}"]')
        expect(button).to_be_focused();page.keyboard.press('Enter')
        expect(button).to_be_focused()
        expect(center).to_have_value(str(s));expect(spread).to_have_value('1')
        expect(page.locator('#curvature-center-value')).to_have_text(display(s))
        expect(page.locator('#curvature-spread-value')).to_have_text('1.0')
        for selector,value in zip(('#eigenvalue-first','#eigenvalue-second'),eigenvalues):
            expect(page.locator(selector)).to_have_text(display(value))
        expect(page.locator('#curvature-kind')).to_have_text(label)
        expect(page.locator('#curvature-diagram-description')).to_contain_text(label)
        point=page.locator('#curvature-diagram circle.diagram-point')
        expect(point).to_have_attribute('cx','241');expect(point).to_have_attribute('cy',str(cy))
        if kind=='flat':
            expect(page.locator('#curvature-summary')).to_contain_text('second-order test alone cannot classify')
        expected={'s':[str(s)],'R':['1'],'r':['0.25'],'region':['remote'],'objects':['1,3'],'from':['reader']}
        shared=page.locator('#explore-state-link').get_attribute('href')
        require(parse_qs(urlsplit(page.url).query)==expected,'Preset URL differs from controls')
        require(parse_qs(urlsplit(shared).query)==expected and urlsplit(shared).fragment=='peaks','Preset share link lost state or section')
        require(urlsplit(page.url).fragment=='peaks','Preset lost current section')
        citation=page.locator('#curvature-citation-text').input_value()
        require(f's = {s}, R = 1;' in citation and f'classification: {kind}.' in citation,'Preset citation is stale')
        with page.expect_download() as pending:
            page.locator('#curvature-export-svg').press('Enter')
        path=output/f'{result["case"]}-preset-{name}.svg';pending.value.save_as(str(path))
        root=ElementTree.fromstring(path.read_bytes());metadata=json.loads(root.find('s:metadata',ns).text)
        require(metadata['params']=={'s':s,'R':1} and metadata['eigenvalues']==eigenvalues and metadata['classification']==kind,'Preset export is stale')
        require(parse_qs(urlsplit(metadata['permalink']).query)=={'s':[str(s)],'R':['1']} and urlsplit(metadata['permalink']).fragment=='peaks','Preset figure permalink lost identity')
        exported=[node for node in root.findall('.//s:circle',ns) if float(node.get('r','0'))==9]
        require(len(exported)==1 and float(exported[0].get('cx'))==241 and float(exported[0].get('cy'))==cy,'Preset export marker differs from displayed marker')
        result.setdefault('curvature_preset_svgs',[]).append({'path':path.name,'sha256':sha256(path.read_bytes()).hexdigest(),'metadata':metadata})
        button.focus()
        if index<3: page.keyboard.press('Tab')
    page.go_back();expect(center).to_have_value('-1')
    expect(page.locator('#curvature-kind')).to_have_text('At least one direction is flat')
    page.go_forward();expect(center).to_have_value('2')
    expect(page.locator('#curvature-kind')).to_have_text('Upward in both directions')
    page.goto(shared);page.reload();expect(center).to_have_value('2')
    expect(page.locator('#curvature-diagram circle.diagram-point')).to_have_attribute('cy','85')
    require(parse_qs(urlsplit(page.url).query)==expected,'Reload lost preset state')
    require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Preset controls overflow document')
    result['steps'].append('All four preset buttons activated by Tab/Enter; visible readouts, diagram, share/history/reload and downloaded preset identity agree')
    return shared

def check_recorded_context_flow(page, origin, expect, result, output):
    parent='math.uniform-matrix-cap-lifetime'
    component='math.d1-component.marked-cylinder-cap'
    region='math.rn-region.fixed-remote';candidate='math.rn-fixed-remote-window'
    reading_rule=[
        'math.d1-component.congruence-erratum',
        'math.d1-component.section9-replacement-v1_1',
        component,'math.d1-component.reconciliation-record',
    ]
    labels={'reading_rule':'Reading rule','component_of':'Component of',
            'component_role':'Component role','coverage_source':'Coverage source'}
    def row(field):
        return page.locator('#recorded-context > div').filter(
            has=page.get_by_text(f'{labels[field]} ({field})',exact=True))
    def absent(field):
        expect(row(field).locator('dd')).to_have_text('Not recorded')
        expect(row(field).locator('button')).to_have_count(0)
    def context_anchor():
        link=page.get_by_role('link',name='Recorded reading context',exact=True)
        expect(link).to_have_attribute('href','#object-reading-context')
        link.focus();page.keyboard.press('Enter')
        expect(page.locator('#object-reading-context')).to_be_focused()
        require(urlsplit(page.url).fragment=='object-reading-context','Native context anchor lost its section')
        require(page.locator('#object-reading-context').evaluate(
            'el=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),
            'Keyboard context target lacks authored focus outline')
    page.goto(origin+f'dependencies.html?q={parent}&node={parent}&from=reader#node-detail')
    expect(page.locator('#detail-heading')).to_have_text(parent,timeout=45000)
    expect(page.locator('#node-detail')).to_be_focused()
    expect(page.locator('#recorded-context > div > dt')).to_have_text(
        [f'{label} ({field})' for field,label in labels.items()])
    context_anchor()
    expect(row('reading_rule').locator('dd > ol > li > button')).to_have_text(reading_rule)
    for field in ('component_of','component_role','coverage_source'): absent(field)
    target=row('reading_rule').get_by_role('button',name=component,exact=True)
    expect(target).to_have_attribute('type','button')
    target.focus();page.keyboard.press('Enter')
    expect(page.locator('#detail-heading')).to_have_text(component)
    expect(page.locator('#node-detail')).to_be_focused()
    expected={'q':[parent],'node':[component],'from':['reader']}
    require(parse_qs(urlsplit(page.url).query)==expected and urlsplit(page.url).fragment=='node-detail','Context reference did not preserve query/navigation state')
    expect(row('component_of').get_by_role('button',name=parent,exact=True)).to_be_visible()
    expect(row('component_role').locator('dd > p')).to_have_text('deterministic_support')
    expect(row('component_role').locator('button,a')).to_have_count(0)
    absent('reading_rule');absent('coverage_source')
    page.go_back();expect(page.locator('#detail-heading')).to_have_text(parent)
    expect(row('reading_rule').locator('ol > li > button')).to_have_text(reading_rule)
    require(parse_qs(urlsplit(page.url).query)=={'q':[parent],'node':[parent],'from':['reader']} and urlsplit(page.url).fragment=='node-detail','Back did not restore the prior native history entry')
    page.go_forward();expect(page.locator('#detail-heading')).to_have_text(component)
    expect(row('component_role').locator('dd > p')).to_have_text('deterministic_support')
    require(parse_qs(urlsplit(page.url).query)==expected,'Forward did not restore the component entry')
    context_anchor();bookmark=page.url
    page.reload()
    expect(page.locator('#detail-heading')).to_have_text(component,timeout=45000)
    expect(page.locator('#object-reading-context')).to_be_focused()
    require(page.url==bookmark,'Reload changed the context-section bookmark')
    expect(row('component_role').locator('dd > p')).to_have_text('deterministic_support')
    require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Recorded-context panel overflow')
    shot=output/f'{result["case"]}-recorded-context.png'
    page.locator('#object-reading-context').screenshot(path=str(shot))
    result['screenshots'].append({'page':'dependency-reading-context','path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()})
    backlink=row('component_of').get_by_role('button',name=parent,exact=True)
    backlink.focus();page.keyboard.press('Enter')
    expect(page.locator('#detail-heading')).to_have_text(parent)
    expect(page.locator('#node-detail')).to_be_focused()
    expect(row('reading_rule').locator('ol > li > button')).to_have_text(reading_rule)
    page.locator('#dependency-search').fill(region)
    result_button=page.locator(f'#search-results button:has(strong:text-is("{region}"))')
    expect(result_button).to_have_count(1)
    result_button.focus();page.keyboard.press('Enter')
    expect(page.locator('#detail-heading')).to_have_text(region)
    context_anchor()
    for field in ('reading_rule','component_of','component_role'): absent(field)
    coverage=row('coverage_source').get_by_role('button',name=candidate,exact=True)
    coverage.focus();page.keyboard.press('Enter')
    expect(page.locator('#detail-heading')).to_have_text(candidate)
    expect(page.locator('#node-detail')).to_be_focused()
    require(parse_qs(urlsplit(page.url).query)=={'q':[region],'node':[candidate],'from':['reader']} and urlsplit(page.url).fragment=='node-detail','Coverage reference lost navigation state')
    for field in labels: absent(field)
    expect(page.locator('#recorded-context button, #recorded-context a, #recorded-context ol')).to_have_count(0)
    # This candidate has actual dependency edges, but none of the four context fields.
    expect(page.locator('#dependency-list').get_by_role('button',name='math.rn-count-interface',exact=True)).to_be_visible()
    require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Absent-context view overflow')
    result['steps'].append('Native keyboard context navigation, recorded reading order/component/coverage references, real Back/Forward and context-section reload restore source data; absent fields remain explicit with no stale buttons')


def check_source_card_flow(page, origin, expect, result, output):
    check_latest_work_flow(page,origin,expect,result,output)
    check_reading_addendum_flow(page,origin,expect,result,output)
    result["reader_entry_screenshots"]=[]
    for entry in ["index","explore","cite","reproduce","formal"]:
        page.goto(origin+entry+".html")
        expect(page.locator("h1")).to_have_count(1)
        if entry == "formal":
            jump=page.get_by_role('link',name='Inspect the exact coverage and its limits ↓',exact=True)
            jump.focus();page.keyboard.press('Enter')
            expect(page.locator('#formal-coverage')).to_be_focused()
            require(page.locator('#formal-coverage').evaluate('(el)=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),'Formal coverage jump lacks authored focus outline')
            summary=page.locator('#formal-coverage-details summary')
            expect(page.locator('#formal-coverage-details')).to_have_attribute('open','')
            summary.focus();page.keyboard.press('Enter')
            require(page.locator('#formal-coverage-details').get_attribute('open') is None,'Formal scope did not close from its open default')
            page.keyboard.press('Enter')
            expect(page.locator('#formal-coverage-details')).to_have_attribute('open','')
            expect(page.locator('#formal-coverage-table tbody tr')).to_have_count(9)
            expect(page.locator('#formal-coverage-table tbody code')).to_have_count(13)
            expect(page.locator('#formal-coverage')).to_contain_text('not a current formalization inventory')
            expect(page.locator('.boundary')).to_contain_text('PENDING')
            region=page.get_by_role('region',name='Exact scalar companion scope table',exact=True)
            region.focus();page.keyboard.press('ArrowRight')
            require(region.evaluate('(el)=>parseFloat(getComputedStyle(el).outlineWidth)>=3 && getComputedStyle(el).outlineStyle!=="none"'),'Formal table region lacks authored focus outline')
            page.wait_for_function("()=>{const el=document.querySelector('#formal-coverage .table-wrap');return el.scrollWidth<=el.clientWidth||el.scrollLeft>0}")
            require(region.evaluate('(el)=>el.scrollWidth <= el.clientWidth || el.scrollLeft > 0'),'Overflowing scope table did not keyboard-scroll')
            require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Formal scope overflowed document')
            shot=output/f'{result["case"]}-formal-coverage.png'
            page.locator('#formal-coverage').screenshot(path=str(shot))
            result['formal_coverage_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
            summary.focus();page.keyboard.press('Enter')
            require(page.locator('#formal-coverage-details').get_attribute('open') is None,'Formal scope did not close')
            expect(summary).to_be_focused()
            static=page.context.browser.new_context(java_script_enabled=False)
            try:
                fallback=static.new_page();fallback.goto(origin+'formal.html#formal-coverage')
                fallback.locator('#formal-coverage-details summary').press('Enter')
                require(fallback.locator('#formal-coverage-details').get_attribute('open') is None,'Formal scope did not close without JavaScript')
                fallback.locator('#formal-coverage-details summary').press('Enter')
                expect(fallback.locator('#formal-coverage-details')).to_have_attribute('open','')
                expect(fallback.locator('#formal-coverage-table tbody tr')).to_have_count(9)
                expect(fallback.locator('#formal-coverage-table')).to_be_visible()
            finally:
                static.close()
            result['steps'].append('Formal snapshot exposes nine verbatim scope rows/13 declarations; keyboard disclosure, table scroll, close focus, no document overflow and JavaScript-off access checked')
        if entry == "explore":
            page.goto(origin+"explore.html?s=-1&R=1&r=0.25&region=remote&objects=#peaks")
            center=page.locator("#curvature-center")
            expect(center).to_be_enabled();expect(center).to_have_value("-1")
            expect(page.locator("#eigenvalue-second")).to_have_text("0.0")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            expect(page.locator("#pin-gap")).to_have_text("0.015625")
            expect(page.locator('input[name="region"][value="remote"]')).to_be_checked()
            expect(page.locator("#palette-kind")).to_contain_text("Empty groups")
            center.focus();page.keyboard.press("Home")
            expect(center).to_have_value("-3")
            require("s=-3" in page.url,"Keyboard change missing from URL")
            page.go_back();expect(center).to_have_value("-1")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            page.go_forward();expect(center).to_have_value("-3")
            page.get_by_role("button",name="Reset curvatures",exact=True).click()
            expect(center).to_have_value("-2")
            page.get_by_role("button",name="Compare half distance",exact=True).click()
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            page.locator('#object-three').check()
            shared=page.locator("#explore-state-link").get_attribute("href")
            require("objects=3" in shared and "r=0.5" in shared,"Share link missed controls")
            require(shared.endswith("#peaks"),"Share link lost section")
            page.goto(shared)
            expect(center).to_have_value("-2")
            expect(page.locator('#object-three')).to_be_checked()
            expect(page.locator('#object-one')).not_to_be_checked()
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            page.get_by_role("button",name="Restore all three",exact=True).click()
            expect(page.locator("#palette-kind")).to_have_text("One object has no place")
            page.goto(origin+"explore.html?s=NaN&R=2&r=0&region=other&objects=1,1")
            expect(page.locator("#explore-state-status")).to_contain_text("invalid")
            expect(center).to_have_value("-2")
            expect(page.locator("#curvature-spread")).to_have_value("2")
            expect(page.locator("#pin-distance")).to_have_value("0.5")
            expect(page.locator("#palette-kind")).to_have_text("One object has no place")
            preset_shared=check_curvature_preset_flow(page,origin,expect,result,output)
            # Static readers get an explicit boundary instead of a false restored state.
            static=page.context.browser.new_context(viewport=result['viewport'],color_scheme=result['color_scheme'],java_script_enabled=False)
            try:
                fallback=static.new_page()
                fallback.goto(preset_shared)
                expect(fallback.locator("#curvature-center")).to_be_disabled()
                expect(fallback.locator("#explore-state-link")).to_be_hidden()
                expect(fallback.locator("#explore-state-status")).to_contain_text("starting examples")
                expect(fallback.locator('#curvature-export-svg')).to_be_disabled()
                expect(fallback.locator('#curvature-copy-citation')).to_be_disabled()
                for name in ('maximum','saddle','singular-boundary','minimum'):
                    expect(fallback.locator(f'[data-curvature-preset="{name}"]')).to_be_disabled()
                expect(fallback.locator('#curvature-center-value')).to_have_text('−2.0')
                expect(fallback.locator('#curvature-kind')).to_have_text('Downward in both directions')
            finally:
                static.close()
            page.goto(origin+"explore.html?s=-1&R=1&r=0.25&region=remote&objects=1,3#peaks")
            expect(page.locator("#curvature-kind")).to_contain_text("flat")
            page.locator("#explore-state-link").scroll_into_view_if_needed()
            result["steps"].append("Explore URLs restore all controls and derived explanations; keyboard edits, reset, reload, Back/Forward, malformed fields and JavaScript-off fallback checked")
            check_curvature_export_flow(page,origin,expect,result,output)
            check_other_teaching_exports(page,origin,expect,result,output)
        if entry == "cite":
            commit="a414e77d6278a5d9ce6aa6e2bdca6146b048f27a"
            digest=sha256((ROOT/"CITATION.cff").read_bytes()).hexdigest()
            page.locator("#reference-commit").fill(commit)
            page.locator("#reference-path").fill("CITATION.cff")
            page.locator("#reference-sha256").fill(digest)
            page.get_by_role("button",name="Build reference",exact=True).click()
            expect(page.locator("#reference-link")).to_have_attribute("href",f"https://github.com/d6g8k5htny-coder/main/blob/{commit}/CITATION.cff")
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            copy=page.get_by_role("button",name="Copy source reference",exact=True)
            copy.focus();page.keyboard.press("Enter")
            expect(page.locator("#reference-status")).to_contain_text("Copied source reference")
            require(page.evaluate("navigator.clipboard.readText()") == page.locator("#reference-output").text_content(),"Source clipboard differs from visible reference")
            page.get_by_role("button",name="Copy JSON",exact=True).click()
            expect(page.locator("#reference-status")).to_contain_text("Copied JSON")
            record=json.loads(page.evaluate("navigator.clipboard.readText()"))
            require(record["commit"]==commit and record["sha256"]==digest and record["verification"]=="not_performed","JSON reference lost source or verification boundary")
            # Edits and invalid source URLs hide the whole action group. Keep
            # this locator addressable when asserting its disabled DOM state.
            bib=page.locator('#reference-copy-bibtex')
            bib.focus();page.keyboard.press('Enter')
            expect(page.locator('#reference-status')).to_contain_text('Copied BibTeX')
            summary=page.get_by_text('Inspect source-reference template (BibTeX)',exact=True)
            summary.focus();page.keyboard.press('Enter')
            expect(page.locator('#reference-bibtex')).to_be_visible()
            bibtex_with_digest=page.locator('#reference-bibtex').text_content()
            require(page.evaluate('navigator.clipboard.readText()')==bibtex_with_digest,'BibTeX clipboard differs from visible template')
            require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'BibTeX disclosure overflow')
            shared=page.locator("#reference-share").get_attribute("href")
            page.locator("#reference-path").fill("README.md")
            expect(page.locator("#reference-actions")).to_be_hidden()
            expect(page.locator("#reference-output")).to_be_empty()
            expect(page.locator("#reference-bibtex")).to_be_empty()
            expect(bib).to_be_disabled()
            page.locator("#reference-sha256").fill("")
            page.get_by_role("button",name="Build reference",exact=True).click()
            page.go_back();expect(page.locator("#reference-path")).to_have_value("CITATION.cff")
            expect(page.locator("#reference-sha256")).to_have_value(digest)
            expect(page.locator("#reference-bibtex")).to_have_text(bibtex_with_digest)
            page.go_forward();expect(page.locator("#reference-path")).to_have_value("README.md")
            expect(page.locator("#reference-sha256")).to_have_value("")
            expect(page.locator("#reference-bibtex")).to_contain_text("File path: README.md")
            expect(page.locator("#reference-bibtex")).not_to_contain_text(digest)
            page.goto(shared);expect(page.locator("#reference-path")).to_have_value("CITATION.cff")
            expect(page.locator("#reference-bibtex")).to_have_text(bibtex_with_digest)
            expect(page.locator("#reference-output")).to_contain_text(digest)
            page.goto(origin+"cite.html?repo=main&commit=main#reference-builder")
            expect(page.locator("#reference-status")).to_contain_text("Shared reference unavailable")
            expect(page.locator("#reference-output")).to_be_empty()
            expect(page.locator("#reference-bibtex")).to_be_empty()
            expect(bib).to_be_disabled()
            expect(page.locator("#reference-actions")).to_be_hidden()
            page.goto(origin+f"cite.html?repo=main&commit={commit}&path=%FF#reference-builder")
            expect(page.locator("#reference-status")).to_contain_text("malformed URL encoding")
            expect(page.locator("#reference-output")).to_be_empty()
            expect(page.locator("#reference-bibtex")).to_be_empty()
            expect(bib).to_be_disabled()
            page.add_init_script("const mode=sessionStorage.getItem('reference-clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:mode==='defer'?{writeText:text=>new Promise(resolve=>{window.referencePending={text,resolve}})}:undefined})")
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','defer')")
            page.goto(shared);bib.press('Enter')
            for selector in ('#reference-copy','#reference-copy-json','#reference-copy-bibtex'):
                expect(page.locator(selector)).to_be_disabled()
            require(page.evaluate('window.referencePending.text')==bibtex_with_digest,'Pending BibTeX copy differs from visible template')
            page.locator("#reference-path").fill("README.md")
            page.locator("#reference-sha256").fill("")
            page.get_by_role("button",name="Build reference",exact=True).click()
            for selector in ('#reference-copy','#reference-copy-json','#reference-copy-bibtex'):
                expect(page.locator(selector)).to_be_disabled()
            expect(page.locator('#reference-bibtex')).to_contain_text('File path: README.md')
            require(page.evaluate('window.referencePending.text')==bibtex_with_digest,'Pending BibTeX copy changed after edit')
            page.evaluate("window.referencePending.resolve()")
            expect(copy).to_be_enabled()
            expect(page.locator("#reference-status")).to_contain_text("Previous copy finished")
            expect(page.locator("#reference-status")).not_to_contain_text("Copied BibTeX")
            for selector in ("#reference-copy","#reference-copy-json","#reference-copy-bibtex"):
                expect(page.locator(selector)).to_be_enabled()
            copy.press('Enter')  # Existing Copy source reference button; current README.md reference.
            for selector in ('#reference-copy','#reference-copy-json','#reference-copy-bibtex'):
                expect(page.locator(selector)).to_be_disabled()
            require(page.evaluate('window.referencePending.text')==page.locator('#reference-output').text_content(),'Pending source copy differs from current text')
            page.evaluate('window.referencePending.resolve()')
            expect(bib).to_be_enabled()
            expect(page.locator('#reference-status')).to_contain_text('Copied source reference')
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','deny')")
            page.goto(shared);copy.click()
            expect(page.locator("#reference-status")).to_contain_text("copy it manually")
            expect(copy).to_be_enabled()
            bib.press('Enter')
            expect(page.locator('#reference-status')).to_contain_text('BibTeX')
            expect(page.locator('#reference-status')).to_contain_text('copy it manually')
            expect(page.locator('#reference-status')).not_to_contain_text('Copied')
            expect(bib).to_be_enabled()
            page.get_by_text('Inspect source-reference template (BibTeX)',exact=True).press('Enter')
            expect(page.locator('#reference-bibtex')).to_be_visible()
            expect(page.locator('#reference-bibtex')).to_have_text(bibtex_with_digest)
            page.evaluate("sessionStorage.setItem('reference-clipboard-test','unsupported')")
            page.reload();expect(copy).to_be_disabled()
            expect(bib).to_be_disabled()
            expect(page.locator('#reference-bibtex')).to_have_text(bibtex_with_digest)
            page.get_by_text('Inspect source-reference template (BibTeX)',exact=True).press('Enter')
            expect(page.locator('#reference-bibtex')).to_be_visible()
            expect(page.locator("#reference-output")).to_contain_text(digest)
            page.evaluate("sessionStorage.removeItem('reference-clipboard-test')")
            page.reload();expect(copy).to_be_enabled()
            expect(bib).to_be_enabled()
            expect(page.locator('#reference-bibtex')).to_have_text(bibtex_with_digest)
            page.get_by_text('Inspect source-reference template (BibTeX)',exact=True).press('Enter')
            expect(page.locator('#reference-bibtex')).to_be_visible()
            shot=output/f'{result["case"]}-bibtex.png'
            page.locator('#reference-builder').screenshot(path=str(shot))
            result['bibtex_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
            page.get_by_text("Inspect structured source identity (JSON)",exact=True).click()
            expect(page.locator("#reference-json")).to_be_visible()
            page.locator("#reference-actions").scroll_into_view_if_needed()
            static=page.context.browser.new_context(java_script_enabled=False)
            try:
                fallback=static.new_page();fallback.goto(shared)
                expect(fallback.locator("#reference-builder")).to_be_hidden()
                expect(fallback.get_by_role("heading",name="Build a source link without JavaScript")).to_be_visible()
            finally:
                static.close()
            result["steps"].append("Cite keyboard copies exact text/JSON/BibTeX with supplied digest; edit invalidation, share/reload, Back/Forward, invalid links, clipboard refusal and JavaScript-off fallback checked")
        if entry == "reproduce":
            recorded=[]
            for row in page.locator('#coefficient-enclosures tbody tr').all():
                recorded.append({'dimension':row.locator('th').inner_text(),'lower':row.locator('code').nth(0).inner_text(),'upper':row.locator('code').nth(1).inner_text()})
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            for format in ['text','json','latex']:
                button=page.locator('#copy-coefficients-'+format)
                button.focus();page.keyboard.press('Enter')
                expect(page.locator('#copy-coefficients-status')).to_contain_text('Copied')
                copied=page.evaluate('navigator.clipboard.readText()')
                require(copied==page.locator('#coefficient-export-'+format).text_content(),'Coefficient clipboard differs from recorded export')
                for row in recorded:
                    require(row['lower'] in copied and row['upper'] in copied,'Decimal enclosure lost precision')
                    if format=='text':
                        require(f"{row['lower']} < c_{row['dimension']},24 < {row['upper']}" in copied,'Text export weakened the strict SIDE24 statement')
                    if format=='latex':
                        require('\\['+row['lower']+' < c_{'+row['dimension']+',24} < '+row['upper']+'\\]' in copied,'LaTeX export lacks strict display math')
                if format=='json':
                    record=json.loads(copied)
                    require(record['intervals']==recorded and record['execution']=='not_performed' and record['source_state']=='historical_pinned','Coefficient export lost identity or scope')
                    require(record['bounds']=='strict' and record['coefficient']=='c_{d,24}','JSON export lost strict SIDE24 identity')
            page.get_by_text('Inspect or manually copy interval exports',exact=True).click()
            expect(page.locator('#coefficient-export-json')).to_be_visible()
            require(page.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Expanded interval export overflow')
            shot=output/f'{result["case"]}-coefficient-exports.png'
            page.locator('#coefficient-exports').screenshot(path=str(shot))
            result['coefficient_export_screenshot']={'path':shot.name,'sha256':sha256(shot.read_bytes()).hexdigest()}
            result['steps'].append('Exact historical coefficient endpoints copied as text/JSON/LaTeX with pinned source and no execution claim')
            commands=page.locator("#reproduction-commands").inner_text()
            page.context.grant_permissions(["clipboard-read","clipboard-write"],origin=origin.rstrip("/"))
            copy=page.get_by_role("button",name="Copy all commands",exact=True)
            expect(copy).to_be_enabled();copy.focus();page.keyboard.press("Enter")
            expect(page.locator("#copy-commands-status")).to_contain_text("Copied all commands")
            require(page.evaluate("navigator.clipboard.readText()") == commands,"Clipboard differs from visible pinned commands")
            expect(page.locator("#copy-commands-status")).to_contain_text("not run")
            page.add_init_script("const mode=sessionStorage.getItem('clipboard-test');if(mode)Object.defineProperty(navigator,'clipboard',{configurable:true,value:mode==='deny'?{writeText:async()=>{throw new Error('denied')}}:undefined})")
            page.evaluate("sessionStorage.setItem('clipboard-test','deny')")
            page.reload();copy.click()
            expect(page.locator("#copy-commands-status")).to_contain_text("Select the command block")
            expect(copy).to_be_enabled()
            page.locator('#copy-coefficients-json').click()
            expect(page.locator('#copy-coefficients-status')).to_contain_text('manually')
            expect(page.locator('#copy-coefficients-json')).to_be_enabled()
            expect(page.locator('#coefficient-export-json')).to_have_text(json.dumps(record,indent=2))
            page.evaluate("sessionStorage.setItem('clipboard-test','unsupported')")
            page.reload();expect(copy).to_be_disabled()
            expect(page.locator("#reproduction-commands")).to_have_text(commands)
            expect(page.locator("#copy-commands-status")).to_contain_text("Select the command block")
            expect(page.locator('#copy-coefficients-json')).to_be_disabled()
            expect(page.locator('#copy-coefficients-status')).to_contain_text('manually')
            page.evaluate("sessionStorage.removeItem('clipboard-test')")
            result["steps"].append("Reproduction keyboard copy matches exact visible commands; clipboard refusal and unsupported API preserve manual fallback without execution")
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),f"Reader entry overflow: {entry}")
        shot=output/f'{result["case"]}-{entry}.png'
        page.screenshot(path=str(shot))
        result["reader_entry_screenshots"].append({"page":entry,"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    result["steps"].append("Home, Explore, Cite, Reproduce and Formal entries have one main heading and no document overflow")
    # Enter through the public reading path, not a fabricated application state.
    page.goto(origin+"research.html")
    page.get_by_role("link",name="Inspect its review and exact scope",exact=True).click()
    card=page.locator("#d2-lifetime-remainder")
    expect(card).to_be_visible(timeout=45000)
    expect(page.locator("#museum-state")).to_contain_text("displayed source bytes verified",timeout=45000)
    expect(card).to_be_focused()
    readable=card.locator(".source-quote-readable")
    expect(readable.get_by_role("link",name="R1–R4 review",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841270276")
    expect(readable.get_by_role("link",name="R5/R6 delta review",exact=True)).to_have_attribute("href","https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206")
    expect(readable.locator("code")).to_have_text("O(1)")
    expect(readable).to_contain_text("Numerical constants/radii, a second coefficient, and RN/24-jet closure are outside this verdict.")
    original=card.locator(".source-quote-original")
    expect(original).to_be_hidden()
    # Every quoted character remains available through a native keyboard control.
    summary=card.locator("summary").filter(has_text="Exact source quote")
    summary.focus();page.keyboard.press("Enter")
    expect(summary).to_be_focused();expect(original).to_be_visible()
    manifest=json.loads((ROOT/"docs/site/museum.json").read_text())
    require(original.text_content()==manifest["claims"][0]["scope_quote"],"Original quoted source changed")
    page.keyboard.press("Enter");expect(original).to_be_hidden()
    page.keyboard.press("Enter");expect(original).to_be_visible()
    page.keyboard.press("Enter");expect(original).to_be_hidden()
    expect(card.locator(".object-class")).to_have_text("ACCEPT-scoped")
    expect(card).to_contain_text("Hashes, browser controls and replay output do not change status.")
    card.scroll_into_view_if_needed()
    result["layout"]=page.evaluate("""() => ({viewport: innerWidth, documentWidth: document.documentElement.scrollWidth})""")
    require(result["layout"]["documentWidth"]<=result["layout"]["viewport"],f"Source card overflow: {result['layout']}")
    result["steps"].extend(["Research route reaches verified D2 card with focus", "both exact review comments are interactive and qualifiers preserved", "keyboard disclosure repeated open/close preserves exact original quote", "source class and acceptance limits retained; no document overflow"])


def check_reader_tools_flow(page, origin, expect, result, output):
    result["screenshots"]=[]
    page.goto(origin+"research.html")
    measure=page.get_by_role("link",name="synthetic counts-to-density guide",exact=True)
    expect(measure).to_have_attribute("href","measure.html")
    measure.click()
    expect(page.locator("h1")).to_have_text("From counts to density.")
    expect(page.locator("#measure-controls")).to_be_enabled()
    expect(page.locator("#measure-mass")).to_have_text("0.5")
    expect(page.locator("#measure-density")).to_have_text("0.153846")
    source=page.locator("#measure-source a")
    source_href=source.get_attribute("href")
    expected_source="https://github.com/d6g8k5htny-coder/main/blob/e60629edde151c27541847b61672e38965752b77/docs/research-translation/20260930/EXPERIMENT.md#1-freeze-the-observable-before-generating-data"
    require(source_href==expected_source,"Measurement source identity drifted")
    page.locator("#measure-count").fill("1152")
    expect(page.locator("#measure-mass")).to_have_text("1")
    expect(page.locator("#measure-density")).to_have_text("0.307692")
    page.locator("#measure-lower").fill("0")
    expect(page.locator("#measure-error")).to_contain_text("greater than zero")
    expect(page.locator("#measure-lower")).to_have_attribute("aria-invalid","true")
    page.locator("#measure-reset").click()
    expect(page.locator("#measure-mass")).to_have_text("0.5")
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Measurement guide overflow")
    shot=output/f'{result["case"]}-measure.png';page.screenshot(path=str(shot),full_page=True)
    result["screenshots"].append({"page":"measure","path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    result["steps"].append("Research route opens the synthetic measurement guide; normalization, invalid edge and exact source cut checked")

    page.goto(origin+"research.html")
    dependencies=page.get_by_role("link",name="dated claim-dependency snapshot",exact=True)
    expect(dependencies).to_have_attribute("href","dependencies.html")
    dependencies.click()
    expect(page.locator("#load-status")).to_contain_text("Pinned Math commit 7858329974e2",timeout=45000)
    expect(page.locator("#node-count")).to_have_text("49")
    expect(page.locator("#edge-count")).to_have_text("55")
    expect(page.locator("#unresolved-count")).to_have_text("15")
    expect(page.locator(".boundary").first).to_contain_text("Dated read-only snapshot")
    expect(page.get_by_role("link",name="Provenance record",exact=True)).to_have_attribute("href","dependency-source/PROVENANCE.json")
    page.locator("#dependency-search").fill("math")
    expect(page.locator("#search-results > li")).to_have_count(39)
    expect(page.locator("#search-results")).to_contain_text("math.side24-coefficient")
    expect(page.locator("#search-results")).to_contain_text("math.uniform-matrix-cap-lifetime")
    result["steps"].append("broad dependency search exposes all 39 matching nodes")
    page.locator("#classification-filter").select_option("PROVED_REVIEWED")
    require(0 < page.locator("#search-results > li").count() < 39,"Classification must intersect search")
    require(parse_qs(urlsplit(page.url).query)=={"q":["math"],"classification":["PROVED_REVIEWED"]},"Filters missing from saved URL")
    page.reload()
    expect(page.locator("#classification-filter")).to_have_value("PROVED_REVIEWED",timeout=45000)
    expect(page.locator("#dependency-search")).to_have_value("math")
    page.locator("#clear-filters").click()
    expect(page.locator("#search-results > li")).to_have_count(49)
    page.locator("#dependency-search").fill("reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md")
    expect(page.locator("#search-results > li")).to_have_count(1)
    result_button=page.locator("#search-results button")
    page.keyboard.press("Tab")
    expect(page.locator("#classification-filter")).to_be_focused()
    page.keyboard.press("Tab")
    expect(page.locator("#clear-filters")).to_be_focused()
    page.keyboard.press("Tab")
    expect(result_button).to_be_focused()
    page.keyboard.press("Enter")
    expect(page.locator("#detail-heading")).to_have_text("math.rn-fixed-annulus-window")
    expect(page.locator("#node-detail")).to_be_focused()
    expect(page.locator("#object-scope")).to_contain_text("compact positive gaps")
    expect(page.locator("#object-notes")).to_contain_text("No global D5/RN/JETMOD")
    expect(page.locator("#evidence-body")).to_contain_text("Record linked")
    expect(page.locator("#evidence-body")).to_contain_text("Not recorded")
    page.locator("#object-audit summary").click()
    expect(page.locator("#node-metadata")).to_contain_text("Review source")
    expect(page.locator("#node-metadata")).to_contain_text("reviews/pr22_fixed_annulus_nonauthor_20260925/REVIEW.md")
    expect(page.locator("#dependency-paths > li")).to_have_count(1)
    expect(page.locator("#dependency-paths > li")).to_contain_text("math.rn-count-interface")
    expect(page.locator("#dependency-paths")).not_to_contain_text("hist.CH-LIFT")
    page.locator("#dependency-search").fill("math.lifetime-remainder")
    remainder_result=page.locator(
        '#search-results button:has(strong:text-is("math.lifetime-remainder"))'
    )
    expect(remainder_result).to_have_count(1)
    remainder_result.click()
    expected_remainder_source="https://github.com/d6g8k5htny-coder/Math-/blob/7858329974e28be79f29b22644370084ff43da4f/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md"
    expected_remainder_review="https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206"
    expect(page.locator(f'#node-metadata a[href="{expected_remainder_source}"]')).to_have_count(1)
    expect(page.locator(f'#node-metadata a[href="{expected_remainder_review}"]')).to_have_count(1)
    result["steps"].append("required dependency path and actionable source/review metadata checked")
    check_recorded_context_flow(page,origin,expect,result,output)
    page.goto(origin+"dependencies.html?node=math.rn-fixed-annulus-window#node-detail")
    require(parse_qs(urlsplit(page.url).query)=={"node":["math.rn-fixed-annulus-window"]},"Saved dependency selection missing")
    page.reload()
    expect(page.locator("#detail-heading")).to_have_text("math.rn-fixed-annulus-window",timeout=45000)
    expect(page.locator("#node-detail")).to_be_focused()
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Dependency viewer overflow")
    shot=output/f'{result["case"]}-dependencies.png';page.screenshot(path=str(shot),full_page=True)
    result["screenshots"].append({"page":"dependencies","path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()})
    for fragment in ("object-evidence", "object-audit"):
        page.goto(origin+"dependencies.html?node=math.rn-fixed-annulus-window#"+fragment)
        expect(page.locator("#"+fragment)).to_be_focused(timeout=45000)
        page.reload()
        expect(page.locator("#"+fragment)).to_be_focused(timeout=45000)
        require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Object deep-link overflow")
    page.locator("#object-audit summary").click()
    expect(page.locator("#node-record")).to_contain_text('"scientific_status_unchanged": true')
    page.goto(origin+"dependencies.html?node=missing.node#node-detail")
    expect(page.locator("#detail-heading")).to_have_text("Saved selection unavailable",timeout=45000)
    expect(page.locator("#selection-error")).to_contain_text("does not match this source snapshot")
    expect(page.locator("#recorded-context")).to_be_empty()
    expect(page.locator("#object-reading-context")).to_be_hidden()
    require(page.evaluate("document.documentElement.scrollWidth <= innerWidth"),"Dependency refusal overflow")
    result["steps"].append("Pinned 49/55/15 graph, review-source search, post-layout saved-link focus, invalid-ID refusal and narrow overflow checked")
    page.goto(origin+"dependencies.html?classification=ACCEPT_ALL")
    expect(page.locator("#filter-error")).to_contain_text("Unknown classification",timeout=45000)
    expect(page.locator("#search-results > li")).to_have_count(0)
    page.locator("#clear-filters").click()
    expect(page.locator("#filter-error")).to_be_hidden()
    expect(page.locator("#search-results > li")).to_have_count(49)
    result["steps"].append("Saved exact-classification filters, all-records recovery, scope/evidence boundaries and native audit disclosure checked")


# Rendered outcomes, not stylesheet parsing: whatever selector, sheet or !important produces the layout, these read what Chromium drew.
# A focused element's ring box is its border box grown by outline offset + width on every side; an ancestor whose overflow is not
# visible clips at its padding box on that axis, so any part of the ring past that edge is not drawn. The ring colour is composited
# over the nearest ancestor backgrounds (white when none is opaque) and its contrast taken by WCAG relative luminance. With a text
# scope, every rendered text line box inside it (not inside a closed disclosure) that is not the element's own text must stay outside the ring box.
FOCUS_RING_JS=r"""(el, scope) => {
  const name=e=>e.tagName.toLowerCase()+(e.id?'#'+e.id:'')+[...e.classList].map(c=>'.'+c).join('');
  const s=getComputedStyle(el), grow=s.outlineStyle==='none'?0:parseFloat(s.outlineWidth)+parseFloat(s.outlineOffset), b=el.getBoundingClientRect();
  const ring={left:b.left-grow,top:b.top-grow,right:b.right+grow,bottom:b.bottom+grow}, cuts=[];let room=Infinity;
  for(let a=el.parentElement;a;a=a.parentElement){
    const c=getComputedStyle(a), r=a.getBoundingClientRect();
    const clip={left:r.left+parseFloat(c.borderLeftWidth),top:r.top+parseFloat(c.borderTopWidth),right:r.right-parseFloat(c.borderRightWidth),bottom:r.bottom-parseFloat(c.borderBottomWidth)};
    const sides=[];
    if(c.overflowX!=='visible')sides.push(['left',clip.left-ring.left],['right',ring.right-clip.right]);
    if(c.overflowY!=='visible')sides.push(['top',clip.top-ring.top],['bottom',ring.bottom-clip.bottom]);
    for(const [side,cut] of sides){room=Math.min(room,-cut);if(cut>0.5)cuts.push(`${side} side cut ${cut.toFixed(1)}px by ${name(a)}`);}
  }
  const paint=document.createElement('canvas').getContext('2d',{willReadFrequently:true});
  const rgba=css=>{paint.clearRect(0,0,1,1);paint.fillStyle='rgba(0,0,0,0)';paint.fillStyle=css;paint.fillRect(0,0,1,1);const d=paint.getImageData(0,0,1,1).data;return [d[0],d[1],d[2],d[3]/255];};
  const layers=[];
  for(let a=el.parentElement;a;a=a.parentElement){const c=rgba(getComputedStyle(a).backgroundColor);if(c[3]>0)layers.unshift(c);if(c[3]>=1)break;}
  let surface=[255,255,255];for(const c of layers)surface=surface.map((v,i)=>c[i]*c[3]+v*(1-c[3]));
  const outline=rgba(s.outlineColor), drawn=surface.map((v,i)=>outline[i]*outline[3]+v*(1-outline[3]));
  const lum=c=>c.map(v=>{v/=255;return v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4;}).reduce((t,v,i)=>t+v*[0.2126,0.7152,0.0722][i],0);
  const [hi,lo]=[lum(drawn),lum(surface)].sort((x,y)=>y-x), contrast=(hi+0.05)/(lo+0.05);
  const hits=[];let gap=Infinity;
  if(scope)for(const host of document.querySelectorAll(scope)){
    const walker=document.createTreeWalker(host,NodeFilter.SHOW_TEXT);
    for(let t=walker.nextNode();t;t=walker.nextNode()){
      if(!t.data.trim()||el.contains(t)||!t.parentElement.checkVisibility({visibilityProperty:true}))continue;const range=document.createRange();range.selectNodeContents(t);
      for(const q of range.getClientRects()){
        if(q.width<=0||q.height<=0||Math.min(q.right,ring.right)-Math.max(q.left,ring.left)<=0.5)continue;
        const apart=Math.max(q.top-ring.bottom,ring.top-q.bottom);gap=Math.min(gap,apart);
        if(apart<-0.5)hits.push(`"${t.data.trim().replace(/\s+/g,' ').slice(0,40)}" (${q.top.toFixed(1)}-${q.bottom.toFixed(1)}px) inside the ring box (${ring.top.toFixed(1)}-${ring.bottom.toFixed(1)}px)`);
      }
    }
  }
  return {element:name(el)+' "'+(el.textContent||'').trim().replace(/\s+/g,' ').slice(0,48)+'"',focused:document.activeElement===el,visible:el.matches(':focus-visible'),
    style:s.outlineStyle,width:s.outlineWidth,offset:s.outlineOffset,color:s.outlineColor,alpha:outline[3],surface:surface.map(Math.round),contrast:Math.round(contrast*100)/100,
    rects:[...el.getClientRects()].filter(q=>q.width>0&&q.height>0).map(q=>({left:q.left,top:q.top,right:q.right,bottom:q.bottom})),
    room:Number.isFinite(room)?Math.round(room*10)/10+0:null,cuts,hits,gap:Number.isFinite(gap)?Math.round(gap*10)/10+0:null};
}"""
# Painted ring: two screenshots of the same clip, focused and blurred, decoded in the page itself (a second page would put this one in
# the background and slow every screenshot to a throttled frame). A pixel changed when a colour channel moved by more than 30. For each
# fragment rect (a link that wraps has one per line) and each side, every position along the side whose ring band (offset..offset+width
# outside the rect, one pixel of slack each way) belongs to the outline of the fragments' union is tested for a changed pixel across the
# band; this returns each side's share of such positions, and check_rendered_focus_rings requires at least 97%. Each outer corner
# square (width x width) of that outline must have changed throughout.
RING_PIXELS_JS=r"""async ([focused, blurred, at, rects, offset, width]) => {
  const decode=async data=>{const bytes=Uint8Array.from(atob(data),c=>c.charCodeAt(0));
    const bitmap=await createImageBitmap(new Blob([bytes],{type:'image/png'}),{colorSpaceConversion:'none',premultiplyAlpha:'none'});
    const canvas=new OffscreenCanvas(bitmap.width,bitmap.height),context=canvas.getContext('2d',{willReadFrequently:true});context.drawImage(bitmap,0,0);
    return context.getImageData(0,0,bitmap.width,bitmap.height);};
  const [a,b]=await Promise.all([decode(focused),decode(blurred)]);
  const changed=(x,y)=>{x-=at[0];y-=at[1];if(x<0||y<0||x>=Math.min(a.width,b.width)||y>=Math.min(a.height,b.height))return false;
    const i=(y*a.width+x)*4,j=(y*b.width+x)*4;return Math.max(Math.abs(a.data[i]-b.data[j]),Math.abs(a.data[i+1]-b.data[j+1]),Math.abs(a.data[i+2]-b.data[j+2]))>30;};
  const grown=d=>rects.map(q=>({left:q.left-d,top:q.top-d,right:q.right+d,bottom:q.bottom+d}));
  const outer=grown(offset+width), inner=grown(offset), within=(q,x,y)=>x>q.left&&x<q.right&&y>q.top&&y<q.bottom;
  const band=(x,y)=>outer.some(q=>within(q,x,y))&&!inner.some(q=>within(q,x,y));
  const g=offset+width, mid=offset+width/2, sides=[], corners=[];
  rects.forEach((q,n)=>{
    const runs={top:[q.left,q.right,p=>[p+0.5,q.top-mid],Math.floor(q.top-g-1),Math.ceil(q.top-offset+1),true],
                bottom:[q.left,q.right,p=>[p+0.5,q.bottom+mid],Math.floor(q.bottom+offset-1),Math.ceil(q.bottom+g+1),true],
                left:[q.top,q.bottom,p=>[q.left-mid,p+0.5],Math.floor(q.left-g-1),Math.ceil(q.left-offset+1),false],
                right:[q.top,q.bottom,p=>[q.right+mid,p+0.5],Math.floor(q.right+offset-1),Math.ceil(q.right+g+1),false]};
    for(const [side,[from,to,point,lo,hi,across]] of Object.entries(runs)){
      let total=0,hit=0;
      for(let p=Math.ceil(from);p<Math.floor(to);p++){
        if(!band(...point(p)))continue;total++;
        for(let c=lo;c<hi;c++)if(across?changed(p,c):changed(c,p)){hit++;break;}
      }
      if(total)sides.push({fragment:n,side,positions:total,share:Math.round(hit/total*1000)/1000});
    }
    for(const [corner,x0,y0] of [['top-left',q.left-g,q.top-g],['top-right',q.right+offset,q.top-g],['bottom-left',q.left-g,q.bottom+offset],['bottom-right',q.right+offset,q.bottom+offset]]){
      const cx=Math.round(x0),cy=Math.round(y0);if(!band(cx+width/2,cy+width/2))continue;
      let count=0;for(let y=cy;y<cy+width;y++)for(let x=cx;x<cx+width;x++)if(changed(x,y))count++;
      corners.push({fragment:n,corner,changed:count,of:width*width});
    }
  });
  return {sides,corners};
}"""
# A whitespace-separated token is 'json' if it holds a brace, or a straight double quote next to a colon or comma. Otherwise, with leading
# ( " ' “ ‘ [ and trailing ) " ' ” ’ ] , . ; : … stripped, it is a 'path' for a URL scheme, a leading /, ./ or ../, two or more slashes, a
# trailing slash, or a slash followed by a name with an extension; a 'hash' for seven or more hex digits mixing digits and letters; an
# 'id' for any digit, _ # @ = or backslash, a . or : between two letters or digits, or a lower-case letter followed by a capital; and
# otherwise a 'word' (so informal/formal, one slash and no extension, is a word).
TOKEN_KIND_JS=r"""token => {
  if(/[{}]|"[:,]|[:,]"/.test(token))return 'json';
  const core=token.replace(/^[("'“‘\[]+/u,'').replace(/[)"'”’\],.;:…]+$/u,'');
  if(/^[a-z][a-z0-9+.-]*:\/\//i.test(core)||/^\.{0,2}\//.test(core)||(core.match(/\//g)||[]).length>1||/\/$/.test(core)||/\/[^/]*\.[A-Za-z0-9]+$/.test(core))return 'path';
  if(/^[0-9a-f]{7,}$/i.test(core)&&/[0-9]/.test(core)&&/[a-f]/i.test(core))return 'hash';
  if(/[0-9_#@=\\]|[A-Za-z0-9][.:][A-Za-z0-9]|[a-z][A-Z]/.test(core))return 'id';
  return 'word';
}"""
# Every whitespace-separated word in the matched cells whose characters fall on more than one line box, and every word with a line box
# outside its cell's border box (by more than 0.5px). With ordinary=true, text inside a link or code element is skipped (those may break
# anywhere) and so is every token TOKEN_KIND_JS does not call a word; `kinds` counts the kinds of the tokens read. A split is `inside`
# unless each of its line changes comes right after a hyphen (U+002D or U+2010) or a slash in the word.
WORD_SPLITS_JS=r"""([selector, ordinary]) => {
  const kind=(""" + TOKEN_KIND_JS + r"""), kinds={}, splits=[], spills=[], details=[];let words=0;
  for(const cell of document.querySelectorAll(selector)){
    const box=cell.getBoundingClientRect(), row=cell.parentElement.firstElementChild.textContent.trim(), where=`${cell.tagName.toLowerCase()} of row "${row}"`;
    const walker=document.createTreeWalker(cell,NodeFilter.SHOW_TEXT);
    for(let text=walker.nextNode();text;text=walker.nextNode()){
      const linked=!!text.parentElement.closest('a, code');
      if(ordinary&&linked)continue;
      for(const match of text.data.matchAll(/\S+/g)){
        const word=match[0], k=kind(word);kinds[k]=(kinds[k]||0)+1;
        if(ordinary&&k!=='word')continue;
        words++;const range=document.createRange();range.setStart(text,match.index);range.setEnd(text,match.index+word.length);
        const lines=[];let out=null;
        for(const q of range.getClientRects()){
          if(!q.width&&!q.height)continue;const mid=(q.top+q.bottom)/2;if(!lines.some(y=>Math.abs(y-mid)<2))lines.push(mid);
          const past=Math.max(box.left-q.left,q.right-box.right,box.top-q.top,q.bottom-box.bottom);if(past>0.5)out=Math.max(out||0,past);
        }
        if(out!==null)spills.push(`"${word.slice(0,60)}" reaches ${out.toFixed(1)}px outside its ${where}`);
        if(lines.length>1){
          let inside=false,prev=null;
          for(let i=0;i<word.length&&!inside;i++){
            const one=document.createRange();one.setStart(text,match.index+i);one.setEnd(text,match.index+i+1);
            const qs=[...one.getClientRects()].filter(q=>q.width||q.height);if(!qs.length)continue;
            const y=(qs[qs.length-1].top+qs[qs.length-1].bottom)/2;
            if(prev&&Math.abs(y-prev.y)>2&&!/[-\/\u2010]/.test(word[prev.i]))inside=true;
            prev={i,y};
          }
          splits.push(`"${word.slice(0,60)}" on ${lines.length} lines in ${where}`);
          details.push({word:word.slice(0,120),kind:k,linked,inside,row,lines:lines.length});
        }
      }
    }
  }
  return {words,kinds,splits,spills,details};
}"""
# Document width, and every rendered text line box (checkVisibility: not hidden, not inside a closed disclosure) that is cut. Walking
# outward from the line, the subject that must stay visible starts as the line box. An ancestor that clips content (overflow hidden or
# clip on an axis; paint containment or content-visibility:auto on both) must hold the subject inside its padding box. An ancestor that
# scrolls on an axis (overflow auto or scroll) makes the text reachable there, so on that axis the subject becomes that ancestor's
# scrollport, which every ancestor further out, and the viewport, must still hold. An ancestor's clip-path inset() must hold the subject
# inside the inset rectangle of its border box (rounded corners are not measured); any other clip-path is reported as unmeasured.
DOCUMENT_WIDTH_JS=r"""() => {
  const root=document.documentElement,width=root.clientWidth,wide=[],cut=[],clips=new Map();
  const name=e=>e.tagName.toLowerCase()+(e.id?'#'+e.id:'')+[...e.classList].map(c=>'.'+c).join('');
  const scrolls=e=>{for(let a=e;a&&a!==root;a=a.parentElement)if(getComputedStyle(a).overflowX!=='visible')return true;return false;};
  const inset=(c,r)=>{const m=/^inset\((.*?)(?:\s+round\s.*)?\)(?:\s+border-box)?$/.exec(c.clipPath.trim());if(!m)return null;
    const v=m[1].trim().split(/\s+/),[t,ri,b,l]=[v[0],v[1]??v[0],v[2]??v[0],v[3]??v[1]??v[0]];
    const len=(x,ref)=>/^-?[\d.]+px$/.test(x)?parseFloat(x):/^-?[\d.]+%$/.test(x)?parseFloat(x)/100*ref:/^0$/.test(x)?0:NaN;
    const box={left:r.left+len(l,r.width),right:r.right-len(ri,r.width),top:r.top+len(t,r.height),bottom:r.bottom-len(b,r.height)};
    return Object.values(box).some(Number.isNaN)?null:box;};
  const clipping=a=>{if(!clips.has(a)){const c=getComputedStyle(a),contained=/paint|strict|content/.test(c.contain)||c.contentVisibility==='auto';
    const axis=o=>o==='auto'||o==='scroll'?'scroll':o==='hidden'||o==='clip'||contained?'clip':null;
    clips.set(a,{x:axis(c.overflowX),y:axis(c.overflowY),path:c.clipPath!=='none',c});}return clips.get(a);};
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let runs=0;
  for(let t=walker.nextNode();t;t=walker.nextNode()){
    if(!t.data.trim()||!t.parentElement||!t.parentElement.checkVisibility({visibilityProperty:true}))continue;
    const range=document.createRange();range.selectNodeContents(t);
    for(const q of range.getClientRects()){
      if(q.width<=0)continue;runs++;const s={left:q.left,right:q.right,top:q.top,bottom:q.bottom};let why=null;
      const outX=b=>s.left<b.left-0.5||s.right>b.right+0.5,outY=b=>s.top<b.top-0.5||s.bottom>b.bottom+0.5;
      for(let a=t.parentElement;a&&a!==root&&!why;a=a.parentElement){
        const k=clipping(a);if(!k.x&&!k.y&&!k.path)continue;
        const r=a.getBoundingClientRect(),c=k.c,pad={left:r.left+parseFloat(c.borderLeftWidth),right:r.right-parseFloat(c.borderRightWidth),top:r.top+parseFloat(c.borderTopWidth),bottom:r.bottom-parseFloat(c.borderBottomWidth)};
        if(k.x==='clip'&&outX(pad))why=`clipped by ${name(a)} [${pad.left.toFixed(1)}, ${pad.right.toFixed(1)}]`;
        else if(k.y==='clip'&&outY(pad))why=`clipped by ${name(a)} [${pad.top.toFixed(1)}, ${pad.bottom.toFixed(1)}] vertically`;
        if(why)break;
        if(k.x==='scroll'){s.left=pad.left;s.right=pad.right;}
        if(k.y==='scroll'){s.top=pad.top;s.bottom=pad.bottom;}
        if(k.path){const box=inset(c,r);
          if(!box)why=`under ${name(a)}, whose clip-path ${c.clipPath} is not measured`;
          else if(outX(box)||outY(box))why=`clipped by the clip-path of ${name(a)} [${box.left.toFixed(1)}, ${box.right.toFixed(1)}] x [${box.top.toFixed(1)}, ${box.bottom.toFixed(1)}]`;}
      }
      if(!why&&(s.left<-0.5||s.right>width+0.5))why=`past the ${width}px viewport`;
      if(why){cut.push(`${name(t.parentElement)} "${t.data.trim().slice(0,48)}" [${q.left.toFixed(1)}, ${q.right.toFixed(1)}] ${why}`);break;}
    }
  }
  if(root.scrollWidth>width)
    for(const e of document.body.querySelectorAll('*')){const r=e.getBoundingClientRect();if(r.width>0&&r.right>width+0.5&&!scrolls(e.parentElement))wide.push(name(e)+' to '+r.right.toFixed(1)+'px');}
  const p=getComputedStyle(document.querySelector('main p'));
  return {scrollWidth:root.scrollWidth,clientWidth:width,wide:wide.slice(-4),runs,cut:cut.slice(0,4),cuts:cut.length,spacing:[p.letterSpacing,p.wordSpacing]};
}"""
EVIDENCE_REGION_JS="""() => {const region=document.querySelector('#object-evidence .evidence-scroll'),table=region.querySelector('.evidence-table');
  return {table:Math.round(table.getBoundingClientRect().width*10)/10,client:region.clientWidth,scroll:region.scrollWidth,
          tabindex:region.tabIndex,overflowX:getComputedStyle(region).overflowX};}"""
# The evidence region's focus state and scroll range, its scrollLeft once unchanged over three animation frames, and how far the table's
# right edge passes the region's inner right edge.
REGION_FOCUS_JS="""e => {const s=getComputedStyle(e);return {focused:e===document.activeElement,focus_visible:e.matches(':focus-visible'),outline:s.outlineStyle,
  width:parseFloat(s.outlineWidth),max:e.scrollWidth-e.clientWidth,left:e.scrollLeft};}"""
SCROLL_SETTLED_JS="e => new Promise(done => {let last=NaN,same=0,n=0;const step=()=>{const x=e.scrollLeft;same=x===last?same+1:0;last=x;if(same>=3||++n>120)done(x);else requestAnimationFrame(step);};requestAnimationFrame(step);})"
TABLE_EDGE_JS="e => {const t=e.querySelector('.evidence-table').getBoundingClientRect(),r=e.getBoundingClientRect();return Math.round((t.right-(r.left+e.clientLeft+e.clientWidth))*10)/10;}"
# Resolves true once the element's box is the same over three animation frames (an exhibit redraws after a viewport change), false after 60.
SETTLED_JS="el => new Promise(done => {let last='',same=0,frames=0;const step=()=>{const r=el.getBoundingClientRect(),key=[r.left,r.top,r.width,r.height].join();same=key===last?same+1:0;last=key;if(same>=2||++frames>60)done(same>=2);else requestAnimationFrame(step);};requestAnimationFrame(step);})"
# The museum's two status lines: every height each takes from its first text (verification held at the manifest) to its last.
STATUS_HEIGHTS_JS="""() => {const seen=window.__statusHeights={};
  for(const id of ['museum-state','conditional-route-status']){const line=document.getElementById(id),list=seen[id]=[];
    const note=()=>{const text=line.textContent;if(!list.length||list[list.length-1].text!==text)list.push({text,height:line.getBoundingClientRect().height});};
    note();new MutationObserver(note).observe(line,{childList:true,characterData:true,subtree:true});}
  return Object.fromEntries(Object.entries(seen).map(([id,list])=>[id,list[0]]));}"""
# WCAG 1.4.12 text-spacing override, served same-origin so the pages' style-src 'self' policy admits it.
TEXT_SPACING_CSS="*,*::before,*::after{line-height:1.5!important;letter-spacing:.12em!important;word-spacing:.16em!important}p{margin-bottom:2em!important}"

# Focuses the element and checks its ring: computed (focused with :focus-visible; solid 3px at a 4px offset; non-zero alpha and 3:1
# against the surface behind it; no clipping ancestor cutting it; with a scope, no other text inside the ring box) and painted (focused
# against blurred screenshots: changed pixels along at least 97% of every side of the ring band and in every outer corner). Leaves the
# element blurred.
def check_ring(page, locator, where, scope=None):
    def settle():
        locator.evaluate("e => e.scrollIntoView({block:'center',inline:'nearest'})")
        require(locator.evaluate(SETTLED_JS),f"{where}: the focus target kept moving after scrolling into view")
        locator.focus()
        return locator.evaluate(FOCUS_RING_JS,scope)
    state=settle();label=f'{where}: {state["element"]}'
    require(state["focused"] and state["visible"],f"{label} is not keyboard-focused with :focus-visible: {state}")
    require((state["style"],state["width"],state["offset"])==("solid","3px","4px"),f'{label} ring is {state["style"]} {state["width"]} at offset {state["offset"]}, not solid 3px at 4px')
    offset,width=float(state["offset"][:-2]),float(state["width"][:-2]);grow=offset+width+2
    view=restore=page.viewport_size
    need=max(q["bottom"] for q in state["rects"])-min(q["top"] for q in state["rects"])+2*grow+16
    if need>view["height"]:
        # A target taller than the viewport is screenshotted whole: the viewport grows in height only, for these screenshots, and the
        # ring box is centred in it (scrollIntoView honours scroll margins, which can push a tall box's ring past the bottom edge).
        page.set_viewport_size({"width":view["width"],"height":int(need)+48});view=page.viewport_size;state=settle()
        top=min(q["top"] for q in state["rects"])-grow;bottom=max(q["bottom"] for q in state["rects"])+grow
        if top<0 or bottom>view["height"]:
            page.evaluate("d => window.scrollBy(0,d)",top-(view["height"]-(bottom-top))/2)
            require(locator.evaluate(SETTLED_JS),f"{where}: the focus target kept moving after centring its ring box");state=locator.evaluate(FOCUS_RING_JS,scope)
    try:
        top=min(q["top"] for q in state["rects"])-grow;bottom=max(q["bottom"] for q in state["rects"])+grow
        require(top>=0 and bottom<=view["height"] or need<=restore["height"],f'{label}: its ring box [{top:.1f}, {bottom:.1f}] does not fit the {view["height"]}px viewport')
        require(state["alpha"]>0 and state["contrast"]>=3,f'{label} ring colour {state["color"]} has {state["contrast"]}:1 against the surface rgb{tuple(state["surface"])} behind it, not 3:1')
        require(not state["cuts"],f'{label} ring is clipped: {"; ".join(state["cuts"])}')
        require(not state["hits"],f'{label} ring box crosses other text: {"; ".join(state["hits"][:3])}')
        left=max(0,int(min(q["left"] for q in state["rects"])-grow));top=max(0,int(min(q["top"] for q in state["rects"])-grow))
        right=min(view["width"],int(max(q["right"] for q in state["rects"])+grow)+1);bottom=min(view["height"],int(max(q["bottom"] for q in state["rects"])+grow)+1)
        clip={"x":left,"y":top,"width":right-left,"height":bottom-top}
        focused=page.screenshot(clip=clip,animations="disabled")
        locator.evaluate("e => e.blur()")
        blurred=page.screenshot(clip=clip,animations="disabled")
    finally:
        if view!=restore:page.set_viewport_size(restore)
    painted=page.evaluate(RING_PIXELS_JS,[base64.b64encode(focused).decode(),base64.b64encode(blurred).decode(),[left,top],state["rects"],offset,int(width)])
    short=[f'{x["side"]} side {x["share"]:.1%}'+(f' (line {x["fragment"]+1})' if len(state["rects"])>1 else '') for x in painted["sides"] if x["share"]<0.97]
    short+=[f'{x["corner"]} corner {x["changed"]}/{x["of"]} px' for x in painted["corners"] if x["changed"]<x["of"]]
    require(painted["sides"] and not short,f'{label} ring is not painted whole (changed pixels, focused against blurred): {"; ".join(short) or "no ring band in view"}')
    state["least_side"]=min(x["share"] for x in painted["sides"])
    return state

def check_rendered_focus_rings(page, origin, expect, result, output):
    result["rings"]=[]
    # Verification rewrites #museum-state and #conditional-route-status; at 1280 each keeps the height of its first text through its
    # last, so the page below it stays put.
    held=[]
    manifest=lambda url:urlsplit(url).path.endswith("/museum.json")
    page.route(manifest,lambda route:held.append(route))
    page.goto(origin+"museum.html")
    deadline=time.monotonic()+45
    while not held and time.monotonic()<deadline:
        page.wait_for_timeout(10)  # Pump Playwright events until the manifest request is held.
    require(len(held)==1,"The museum manifest request was not held before verification")
    first=page.evaluate(STATUS_HEIGHTS_JS)
    for line,start in first.items():
        require(start["text"].startswith("Verifying"),f'#{line} first text is not the verifying line: {start["text"]!r}')
    held[0].continue_();page.unroute(manifest)
    expect(page.locator("#museum-state")).to_contain_text("displayed source bytes verified",timeout=45000)
    expect(page.locator("#conditional-route-status")).to_contain_text("source bytes verified",timeout=45000)
    heights=page.evaluate("() => window.__statusHeights")
    result["status_lines"]=[]
    for line,start in first.items():
        moved=[f'{x["height"]:.1f}px for "{x["text"][:48]}"' for x in heights[line] if abs(x["height"]-start["height"])>0.5]
        require(not moved,f'museum.html at 1280: #{line} is {start["height"]:.1f}px for its first text and {"; ".join(moved)}')
        result["status_lines"].append({"line":line,"width":1280,"texts":len(heights[line]),"height":round(start["height"],1)})
    ring=lambda locator,where,scope=None:check_ring(page,locator,where,scope)
    route=page.locator("#conditional-route")
    expect(route.locator("summary")).to_have_count(6,timeout=45000)
    # Rings are checked in both colour schemes: the card surface, the ring colour and any scheme-scoped rule differ between them.
    for scheme in ["light","dark"]:
        page.emulate_media(color_scheme=scheme)
        page.keyboard.press("Shift")
        for size in [{"width":1280,"height":900},{"width":390,"height":844},{"width":320,"height":800}]:
            page.set_viewport_size(size);where=f'museum.html at {size["width"]} ({scheme})'
            stops=route.locator("summary, a[href]")
            require(stops.count()==8,f"{where}: expected 6 summaries and 2 links in #conditional-route, found {stops.count()}")
            states=[ring(stops.nth(i),where,"#conditional-route" if i<6 else None) for i in range(stops.count())]
            require(all(x["element"].startswith("summary") for x in states[:6]),f"{where}: the first six stops are not the six summaries")
            if size["width"]==390:
                first=route.locator("summary").first
                first.focus();page.keyboard.press("Enter")
                quote=route.locator("details").first.locator("blockquote")
                expect(quote).to_be_visible()
                margins=quote.evaluate("q => [getComputedStyle(q).marginLeft, getComputedStyle(q).marginRight]")
                require(margins==["0px","0px"],f"{where}: opened quote has inline margins {margins}, not 0")
                page.keyboard.press("Enter");expect(quote).to_be_hidden()
            result["rings"].append({"page":"museum.html","scheme":scheme,"width":size["width"],"stops":len(states),
                                    "least_clip_room":min((x["room"] for x in states if x["room"] is not None),default=None),
                                    "least_gap_to_other_text":min((x["gap"] for x in states[:6] if x["gap"] is not None),default=None),
                                    "least_painted_side":min(x["least_side"] for x in states),"least_contrast":min(x["contrast"] for x in states)})
    # When the optional Three.js library (geometry.mjs THREE_CDN) and WebGL both load, as they can on a hosted runner, EC-014's 3D view
    # replaces its 2D figure. Refusing the library makes every run check the same 2D figure, the one a reader without it sees. The route
    # is held only for EC-014's load (routing turns off the HTTP cache); geometry.mjs keeps the refused load and does not ask again.
    refused=result["refused_requests"]=[]
    def refuse_cdn(route):
        refused.append(route.request.url);route.abort()
    for view in ["ec014","remote","annulus","p15"]:
        page.emulate_media(color_scheme="light")
        if view=="ec014":page.route("https://cdn.jsdelivr.net/**",refuse_cdn)
        page.goto(origin+f"museum.html?view={view}#active-exhibit")
        figure=page.locator("#active-exhibit figure.museum-visual")
        expect(figure).to_be_visible(timeout=45000)
        if view=="ec014":
            expect(page.locator("#active-exhibit")).to_contain_text("2D fallback:",timeout=45000)
            require(any("/three@" in url for url in refused),f"museum.html?view=ec014: the Three.js request was not refused; requests refused: {refused}")
            page.unroute("https://cdn.jsdelivr.net/**",refuse_cdn)
            expect(figure).to_be_visible()
        for scheme in ["light","dark"]:
            page.emulate_media(color_scheme=scheme)
            page.keyboard.press("Shift")
            for size in [{"width":1280,"height":900},{"width":390,"height":844},{"width":320,"height":800}]:
                page.set_viewport_size(size)
                state=ring(figure,f'museum.html?view={view} at {size["width"]} ({scheme})')
                result["rings"].append({"page":f"museum.html?view={view}","scheme":scheme,"width":size["width"],"stops":1,"least_clip_room":state["room"],
                                        "least_painted_side":state["least_side"],"least_contrast":state["contrast"]})
    result["steps"].extend(["#museum-state and #conditional-route-status each keep one height from their first to their last text at 1280",
                            "each conditional-route summary and link, and the exhibit figure of each of the four views (EC-014's 2D figure, with the Three.js request refused), "
                            "at 1280, 390 and 320 in light and dark: a solid 3px ring at 4px "
                            "in a colour with 3:1 against the surface behind it, outside every clipping ancestor, and painted (focused against blurred) along every side and corner",
                            "no summary's ring box crosses any other text line in the card; an opened quote at 390 has no inline margin"])

def check_rendered_text_layout(page, origin, expect, result, output):
    result["layouts"]=[]
    override=origin+"text-spacing-override.css"
    page.route(override,lambda route:route.fulfill(status=200,content_type="text/css",body=TEXT_SPACING_CSS))
    def spaced():
        page.add_style_tag(url=override)
        spacing=page.evaluate(DOCUMENT_WIDTH_JS)["spacing"]
        require(spacing[1]!="0px","Text-spacing override was not applied: "+str(spacing))
    def no_overflow(where):
        width=page.evaluate(DOCUMENT_WIDTH_JS)
        require(width["scrollWidth"]<=width["clientWidth"],f'{where}: document is {width["scrollWidth"]}px wide in a {width["clientWidth"]}px viewport; past the viewport: boxes {width["wide"]}, text {width["cut"]}')
        require(width["runs"]>0 and not width["cut"],f'{where}: {width["cuts"]} text line(s) cut where the reader cannot scroll to them: {width["cut"]}')
        return width
    def keyboard_scroll(where):
        # A wider table is allowed only if a keyboard reader can use its region: Tab reaches it from the previous stop, its ring is drawn
        # (check_ring), arrow keys scroll it to the table's far edge and back without moving focus, and Tab moves on (WCAG 2.1.1, 2.1.2,
        # 2.4.7).
        region=page.locator("#object-evidence .evidence-scroll")
        region.focus();page.keyboard.press("Shift+Tab")
        require(not region.evaluate("e => e===document.activeElement"),f"{where}: Shift+Tab does not move focus off the .evidence-scroll region")
        page.keyboard.press("Tab")
        state=region.evaluate(REGION_FOCUS_JS)
        require(state["focused"] and state["focus_visible"],f"{where}: Tab from the previous stop does not focus the .evidence-scroll region with :focus-visible: {state}")
        # Its ring must be seen, not only matched: the same computed and painted checks as every other ring.
        drawn=check_ring(page,region,f"{where}, .evidence-scroll region")
        region.focus()
        def press_until(key,done):
            for presses in range(21):
                if done(region.evaluate(SCROLL_SETTLED_JS)):return presses
                page.keyboard.press(key)
            return None
        right=press_until("ArrowRight",lambda x:x>=state["max"]-0.5);edge=region.evaluate(TABLE_EDGE_JS)
        require(right is not None and edge<=0.5,f'{where}: arrow keys do not scroll the .evidence-scroll region to the table\'s far edge '
                f'({right} presses for a {state["max"]}px range; the table ends {edge}px past the region)')
        left=press_until("ArrowLeft",lambda x:x<=0)
        require(left is not None,f"{where}: arrow keys do not scroll the .evidence-scroll region back to its start")
        require(region.evaluate("e => e===document.activeElement"),f"{where}: arrow keys moved focus off the .evidence-scroll region")
        page.keyboard.press("Tab")
        require(not region.evaluate("e => e===document.activeElement"),f"{where}: Tab does not move focus on from the .evidence-scroll region")
        return {"range":state["max"],"right_presses":right,"left_presses":left,"ring_contrast":drawn["contrast"],"ring_least_painted_side":drawn["least_side"]}
    nodes=["hist.CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip","hist.allcell_fdz_enclosures.json","math.side24-coefficient",
           "math.d5-component.punctured-pin-proof","hist.CH-LIFT"]
    for width in [320,390]:
        page.set_viewport_size({"width":width,"height":844 if width==390 else 800})
        for node in nodes:
            page.goto(origin+f"dependencies.html?node={node}#node-detail")
            expect(page.locator("#evidence-body tr")).to_have_count(5,timeout=45000)
            for spacing in ["default spacing","text-spacing override"]:
                if spacing=="text-spacing override":spaced()
                where=f"{node} at {width} with {spacing}"
                runs=no_overflow(where)["runs"]
                words=page.evaluate(WORD_SPLITS_JS,[".evidence-table th, .evidence-table td:nth-child(2)",False])
                require(words["words"]>0 and not words["splits"],f'{where}: Lane/Record state words split across lines: {words["splits"]}')
                require(not words["spills"],f'{where}: Lane/Record state words outside their cells: {words["spills"]}')
                region=page.evaluate(EVIDENCE_REGION_JS)
                fits=region["table"]<=region["client"]+0.5 and region["scroll"]<=region["client"]
                # WCAG 1.4.10 lets a data table's two-dimensional layout scroll, but not the text of its cells or the content around it, and
                # 1.4.12 still requires no loss of content or function: the cut-text, whole-word and page-width checks above still hold. Under
                # the override the table may outgrow its region only if the region scrolls to the table's full width and a keyboard reader
                # can use it (keyboard_scroll); at default spacing it must fit.
                scrolls=region["tabindex"]==0 and region["overflowX"] in ("auto","scroll") and region["scroll"]>=region["table"]-0.5
                require(fits or (spacing=="text-spacing override" and scrolls),
                        f'{where}: the evidence table is {region["table"]}px wide (content {region["scroll"]}px) in its {region["client"]}px .evidence-scroll region '
                        f'(tabindex {region["tabindex"]}, overflow-x {region["overflowX"]})')
                entry={"page":node,"width":width,"spacing":spacing,"lane_and_state_words":words["words"],"text_runs":runs,"table_width":region["table"],"region_width":region["client"],
                       "table_scrolls_in_region":not fits}
                if not fits:entry["keyboard_scroll"]=keyboard_scroll(where)
                if width==390 and spacing=="default spacing":
                    detail=page.evaluate(WORD_SPLITS_JS,[".evidence-table td:nth-child(3)",True])
                    entry["source_detail_unlinked_json"]=detail["kinds"].get("json",0)
                    if not entry["source_detail_unlinked_json"]:
                        # A break right after a word's own hyphen or slash depends on the font's widths and is ordinary line breaking; any
                        # other line change inside a word is a split.
                        inside=[f'"{x["word"][:60]}" on {x["lines"]} lines in td of row "{x["row"]}"' for x in detail["details"] if x["inside"]]
                        require(detail["words"]>0 and not inside,f'{where}: ordinary Source detail words split other than right after a hyphen or slash: {inside}')
                        require(not detail["spills"],f'{where}: ordinary Source detail words outside their cells: {detail["spills"]}')
                        entry["source_detail_words"]=detail["words"];entry["source_detail_breaks_at_hyphen_or_slash"]=len(detail["details"])-len(inside)
                result["layouts"].append(entry)
    require(sum(1 for x in result["layouts"] if "source_detail_words" in x)>=1,"No case node without unlinked JSON metadata had its Source detail words checked at 390")
    page.goto(origin+"dependencies.html")
    expect(page.locator("#classification-filter")).to_be_enabled(timeout=45000)
    spaced();no_overflow("dependencies.html at 390 with text-spacing override")
    page.set_viewport_size({"width":320,"height":800});no_overflow("dependencies.html at 320 with text-spacing override")
    result["steps"].extend(["five long-token nodes at 320 and 390, with and without WCAG 1.4.12 text spacing: no document overflow, no text line cut where the reader "
                            "cannot scroll, every Lane and Record state word on one line inside its cell, and the evidence table within its scroll region (under the override, "
                            "wider only if that region scrolls to the table's full width and a keyboard reader can use it: Tab reaches it with a visible ring, arrow "
                            "keys scroll it to the table's far edge and back, and Tab moves on)",
                            "at 390 with default spacing, on each of those nodes without unlinked JSON metadata, no ordinary Source detail word split except right after "
                            "its own hyphen or slash, and each inside its cell",
                            "the Dependencies index at 390 and 320 with text spacing: no document overflow and no cut text line"])


def check_visitor_recovery(page, origin, expect, result):
    if result["case"] == "query-refusal":
        config=json.loads((ROOT/"docs/site/config.json").read_text())
        hits=[]
        def refuse_query(route):
            hits.append(route.request.url)
            route.fulfill(status=503,headers={"Access-Control-Allow-Origin":"*"},body="")
        page.route(config["query"]["url"],refuse_query)
        page.goto(origin+"workspace.html#board")
        expect(page.locator("#query-note")).to_contain_text("Source unavailable (503). No result inferred.",timeout=45000)
        expect(page.locator("#custody-note")).to_contain_text("9 byte-copy imports landed",timeout=45000)
        expect(page.locator("#query-identity")).to_be_empty()
        require(len(hits)==1,"Expected exactly one refused query request")
        result["steps"].append("query 503 ends loading without erasing verified import information")
    else:
        manifest=json.loads((ROOT/"docs/site/museum.json").read_text())
        held=[]
        page.route(manifest["claims"][0]["proof"]["url"],lambda route:held.append(route))
        page.goto(origin+"museum.html#d3-side24-coefficient",wait_until="domcontentloaded")
        expect(page.locator("#claim-cards")).to_contain_text("Verifying",timeout=45000)
        deadline=time.monotonic()+45
        while not held and time.monotonic()<deadline:
            page.wait_for_timeout(10)  # Pump Playwright events until the route callback ran.
        require(len(held)==1,"The delayed source was not intercepted before reader interaction")
        page.get_by_role("link",name="Reproduce",exact=True).first.focus()
        page.keyboard.press("Tab")
        focused=page.get_by_role("link",name="Cite",exact=True).first
        expect(focused).to_be_focused()
        held[0].fulfill(response=held[0].fetch())
        expect(page.locator("#museum-state")).to_contain_text("displayed source bytes verified",timeout=45000)
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        expect(focused).to_be_focused()
        require(page.evaluate("Math.abs(document.querySelector('#d3-side24-coefficient').getBoundingClientRect().top)>100"),"Delayed claim completion pulled the reader back")
        result["steps"].append("keyboard navigation during museum verification keeps focus and reading position")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=True)
    report={"scientific_effect":"NONE","browser_run_started":False,"cases":[],"limitations":["Headless Chromium only; mobile viewport emulation, not physical devices","Screenshots require inspection; not full accessibility certification","Local checkout preview, not deployed Pages evidence"]}
    try:
        report["checked_commit"]=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
        if os.environ.get("GITHUB_SHA") and os.environ["GITHUB_SHA"]!=report["checked_commit"]:
            raise ValueError("Runner commit differs from GITHUB_SHA")
        report["run_id"]=os.environ.get("GITHUB_RUN_ID")
        report["run_attempt"]=os.environ.get("GITHUB_RUN_ATTEMPT")
        report["pr_head"]=os.environ.get("BROWSER_PR_HEAD")
        report["tree"]=subprocess.check_output(["git","rev-parse","HEAD^{tree}"],cwd=ROOT,text=True).strip()
        report["playwright"]=importlib.metadata.version("playwright")
        from playwright.sync_api import sync_playwright, expect
        with local_docs() as origin, sync_playwright() as playwright:
            executable=Path("/opt/google/chrome/chrome")
            require(executable.is_file(),"The runner’s packaged Chrome binary is missing")
            report["browser_channel"]="chrome"
            report["browser_executable"]=str(executable)
            with executable.open("rb") as binary:
                report["browser_executable_sha256"]=file_digest(binary,"sha256").hexdigest()
            report["runner_image_os"]=os.environ.get("ImageOS")
            report["runner_image_version"]=os.environ.get("ImageVersion")
            browser=playwright.chromium.launch(channel="chrome",chromium_sandbox=True)
            try:
                report["browser_run_started"]=True;report["browser"]=browser.version
                warm={"case":"warm-asset-release","steps":[],"passed":False};report["cases"].append(warm)
                try:
                    check_warm_asset_release(browser,expect,warm,output);warm["passed"]=True
                except Exception: warm["error"]=traceback.format_exc()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_flow(page,origin,expect,result)
                            require(not errors,f"Page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors
                            shot=output/f"{label}.png"
                            try:
                                page.locator("#inventory").screenshot(path=str(shot),timeout=10000)
                                result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                            except Exception:
                                result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                            context.close()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'source-{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_source_card_flow(page,origin,expect,result,output)
                            require(not errors,f"Source page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors;shot=output/f"{label}.png"
                            try:
                                page.locator("#d2-lifetime-remainder").screenshot(path=str(shot),timeout=10000)
                                result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                            except Exception:
                                result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                            context.close()
                for size in [{"width":1200,"height":900},{"width":390,"height":844}]:
                    for scheme in ["light","dark"]:
                        label=f'reader-tools-{size["width"]}-{scheme}'
                        result={"case":label,"viewport":size,"color_scheme":scheme,"steps":[],"passed":False}
                        report["cases"].append(result)
                        context=browser.new_context(viewport=size,color_scheme=scheme,reduced_motion="reduce")
                        page=context.new_page();page.set_default_timeout(15000);errors=[]
                        page.on("pageerror",lambda error:errors.append(str(error)))
                        try:
                            check_reader_tools_flow(page,origin,expect,result,output)
                            require(not errors,f"Reader tool page errors: {errors}")
                            result["passed"]=True
                        except Exception:
                            result["error"]=traceback.format_exc()
                        finally:
                            result["page_errors"]=errors
                            context.close()
                for label in ["query-refusal","museum-reader-intent"]:
                    result={"case":label,"steps":[],"passed":False};report["cases"].append(result)
                    context=browser.new_context(viewport={"width":390,"height":844},color_scheme="light",reduced_motion="reduce")
                    page=context.new_page();page.set_default_timeout(15000);errors=[]
                    page.on("pageerror",lambda error:errors.append(str(error)))
                    try:
                        check_visitor_recovery(page,origin,expect,result)
                        require(not errors,f"Visitor recovery page errors: {errors}")
                        result["passed"]=True
                    except Exception:
                        result["error"]=traceback.format_exc()
                    finally:
                        result["page_errors"]=errors;shot=output/f"{label}.png"
                        try:
                            page.screenshot(path=str(shot))
                            result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                        except Exception:
                            result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                        context.close()
                result={"case":"inventory-refusal","steps":[],"passed":False};report["cases"].append(result)
                context=browser.new_context(viewport={"width":390,"height":844},color_scheme="dark")
                page=context.new_page();refusal_errors=[];route_hits=[]
                page.on("pageerror",lambda error:refusal_errors.append(str(error)))
                try:
                    config=json.loads((ROOT/"docs/site/config.json").read_text())
                    def refuse_inventory(route):
                        route_hits.append(route.request.url)
                        route.fulfill(status=503,headers={"Access-Control-Allow-Origin":"*"},body="")
                    page.route(config["inventory"]["url"],refuse_inventory)
                    page.goto(origin+"workspace.html?q=SIDE24#inventory")
                    expect(page.locator("#inventory-state")).to_contain_text("Source unavailable (503). No result inferred.",timeout=45000)
                    require(len(route_hits)==1,"Expected exactly one refused inventory request")
                    for selector in ["#search","#repository-filter","#path-filter","#catalog-clear"]:
                        expect(page.locator(selector)).to_be_disabled()
                    require(page.locator("#catalog-link").get_attribute("href") is None,"Unavailable catalog has an active search link")
                    expect(page.locator("#catalog tr")).to_have_count(0)
                    expect(page.locator("#more")).to_be_hidden()
                    require(not refusal_errors,f"Refusal page errors: {refusal_errors}")
                    result["steps"].append("pinned inventory 503 keeps filters/link unavailable and no rows")
                    result["passed"]=True
                except Exception:
                    result["error"]=traceback.format_exc()
                finally:
                    result["page_errors"]=refusal_errors;result["refused_requests"]=route_hits
                    shot=output/"inventory-refusal.png"
                    try:
                        page.locator("#inventory").screenshot(path=str(shot),timeout=10000)
                        result["screenshot"]={"path":shot.name,"sha256":sha256(shot.read_bytes()).hexdigest()}
                    except Exception:
                        result["screenshot_error"]=traceback.format_exc();result["passed"]=False
                    context.close()
                for label,flow in [("rendered-focus-rings",check_rendered_focus_rings),("rendered-text-layout",check_rendered_text_layout)]:
                    result={"case":label,"color_scheme":"light","steps":[],"passed":False};report["cases"].append(result)
                    context=browser.new_context(viewport={"width":1280,"height":900},color_scheme="light",reduced_motion="reduce")
                    page=context.new_page();page.set_default_timeout(15000);errors=[];started=time.monotonic()
                    page.on("pageerror",lambda error:errors.append(str(error)))
                    try:
                        flow(page,origin,expect,result,output)
                        require(not errors,f"Rendered layout page errors: {errors}")
                        result["passed"]=True
                    except Exception:
                        result["error"]=traceback.format_exc()
                    finally:
                        result["page_errors"]=errors;result["seconds"]=round(time.monotonic()-started,1)
                        context.close()
            finally:
                browser.close()
        report["passed"]=len(report["cases"])==18 and all(case["passed"] for case in report["cases"])
    except Exception:
        report["passed"]=False;report["error"]=traceback.format_exc()
    finally:
        (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["passed"] else 1

if __name__=="__main__":raise SystemExit(main())
