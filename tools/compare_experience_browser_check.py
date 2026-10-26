#!/usr/bin/env python3
"""Bounded Compare paging and demand-loaded source-view checks on real artifacts."""
import argparse
import functools
import hashlib
import http.server
import json
from pathlib import Path
import subprocess
import threading
from urllib.parse import urlsplit, parse_qs
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', default='/usr/bin/chromium')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path(__file__).resolve().parents[1] / 'docs/site')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args): pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(args.site.resolve())))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}/'
    report = {'checks': [], 'source_sha256': {name: hashlib.sha256((args.site/name).read_bytes()).hexdigest()
               for name in ['compare.html', 'compare.css', 'compare.mjs', 'compare-source-panel.mjs', 'current.mjs', 'current-model.mjs', 'compare-model.mjs']},
              'browser': subprocess.run([args.browser, '--version'], text=True, capture_output=True, check=True).stdout.strip(),
              'scientific_effect': 'NONE', 'scope': 'Local Chromium DOM, requests and keyboard focus; no spoken AT or Safari claims'}
    errors = []
    def watch(page):
        requests = []
        page.on('request', lambda r: requests.append(r.url))
        page.on('pageerror', lambda e: errors.append(str(e)))
        return requests
    def current_requests(requests):
        return [u for u in requests if urlsplit(u).path == '/current.json']
    def completed(page):
        page.locator('#compare-state[data-state="completed"]').wait_for()
        page.wait_for_load_state('networkidle')
    def synced(page):
        assert page.locator('#page-status').inner_text() == page.locator('#page-status-top').inner_text()
        for which in ['previous', 'next']:
            assert page.locator('#'+which).is_disabled() == page.locator('#'+which+'-top').is_disabled()
        assert page.locator('#pagination').is_visible() == page.locator('#pagination-top').is_visible()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser, headless=True, chromium_sandbox=True)
            for width, height, scheme in [(1440,1000,'light'),(1440,1000,'dark'),(390,844,'light'),(390,844,'dark')]:
                page = browser.new_page(viewport={'width':width,'height':height}, color_scheme=scheme)
                requests = watch(page)
                page.goto(origin+'compare.html'); completed(page)
                assert len(current_requests(requests)) == 1, current_requests(requests)
                assert page.locator('.current-search').is_disabled()
                assert 'pinned and unverified' in page.locator('#compare-source-note').inner_text()
                assert page.locator('.current-state').inner_text().startswith('Pinned source observation:')
                assert not page.locator('.current-source-details').get_attribute('open')
                assert page.locator('#pagination-top').is_visible()
                assert page.locator('#pagination-top').evaluate("e=>e.compareDocumentPosition(document.getElementById('comparison-results')) & Node.DOCUMENT_POSITION_FOLLOWING")
                assert page.locator('#pagination-top').get_attribute('aria-label') != page.locator('#pagination').get_attribute('aria-label')
                assert page.locator('#page-status-top').get_attribute('role') is None
                assert page.locator('#page-status-top').get_attribute('aria-live') is None
                synced(page)
                assert page.locator('#previous-top').is_disabled()
                first = page.locator('#comparison-results .change code').first.inner_text()
                page.locator('#next-top').click(); synced(page)
                assert page.locator('#comparison-results .change code').first.inner_text() != first
                assert page.locator('#results-heading').evaluate('e=>e===document.activeElement')
                page.locator('#previous').click(); synced(page)
                assert page.locator('#comparison-results .change code').first.inner_text() == first
                assert page.locator('#results-heading').evaluate('e=>e===document.activeElement')
                # Requesting the separate source view adds precisely its own unchanged acquisition.
                page.locator('.current-source-details > summary').click()
                page.locator('#compare').focus()
                page.locator('#current-research[data-compare-source="ready"]').wait_for()
                assert len(current_requests(requests)) == 2
                assert page.locator('.current-search').is_enabled()
                assert page.locator('.current-state').inner_text().startswith('Verified source observation:')
                assert page.locator('#compare').evaluate('e=>e===document.activeElement')
                page.locator('.current-source-details > summary').click()
                page.locator('.current-source-details > summary').click()
                page.wait_for_load_state('networkidle')
                assert len(current_requests(requests)) == 2
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                expected = json.loads((args.site/'current.json').read_bytes())['sources'][0]
                href = f'https://github.com/{expected["repository"]}/blob/{expected["commit"]}/{expected["path"]}'
                assert page.locator('.current-results > li a').first.get_attribute('href') == href
                page.emulate_media(media='print')
                assert page.locator('#pagination-top').is_hidden() and page.locator('#pagination').is_hidden()
                page.emulate_media(media='screen')
                if scheme == 'light' and width == 1440:
                    # Exercise the real final page without hundreds of browser roundtrips.
                    steps = page.evaluate("""()=>{let n=0;while(!document.getElementById('next-top').disabled&&n<500){document.getElementById('next-top').click();n++;}return n;}""")
                    assert steps < 500
                    assert page.locator('#next-top').is_disabled(); synced(page)
                    page.locator('#previous-top').click(); synced(page)
                    report['actual_last_page_steps'] = steps
                # A changed selection clears and disables both paging surfaces.
                same = page.locator('#from').input_value()
                page.locator('#to').select_option(same)
                for suffix in ['', '-top']:
                    assert page.locator('#pagination'+suffix).is_hidden()
                    assert page.locator('#previous'+suffix).is_disabled()
                    assert page.locator('#next'+suffix).is_disabled()
                page.locator('#compare').click(); completed(page)
                assert page.locator('#comparison-results .change').count() == 0
                synced(page)
                page.screenshot(path=str(args.output/f'compare-{width}-{scheme}.png'), full_page=True)
                report['checks'].append(f'{width}x{height} {scheme}: one initial current acquisition; dual paging/focus/clear/zero differences; independent requested source verification/exact URI; reopen reuse; outside focus; print/no overflow')
                page.close()
            # Keep the optional import pending while the reader closes the source
            # disclosure and moves focus elsewhere. Completion must respect both.
            page=browser.new_page();requests=watch(page);pending=[]
            page.route('**/current.mjs?*',lambda route:pending.append(route))
            page.goto(origin+'compare.html');completed(page)
            with page.expect_request('**/current.mjs?*'):
                page.locator('.current-source-details > summary').click()
            page.evaluate("document.querySelector('.current-source-details').open=false")
            page.locator('#compare').focus()
            assert pending
            for route in pending:route.continue_()
            page.unroute('**/current.mjs?*')
            page.locator('#current-research[data-compare-source="ready"]').wait_for()
            assert not page.locator('.current-source-details').evaluate('e=>e.open')
            assert page.locator('#compare').evaluate('e=>e===document.activeElement')
            assert len(current_requests(requests))==2
            page.close()
            report['checks'].append('Pending optional import completion preserves a newly closed source disclosure and newer outside focus')
            for suffix in ['?source-q=', '?source-q=SIDE24', '#current-research', '#current-date-explanation', '#current-inventory-legend']:
                page=browser.new_page(); requests=watch(page)
                page.goto(origin+'compare.html'+suffix);completed(page)
                page.locator('#current-research[data-compare-source="ready"]').wait_for()
                assert len(current_requests(requests)) == 2
                assert page.locator('.current-source-details').evaluate('e=>e.open')
                if 'source-q' in suffix:
                    assert page.locator('.current-search').is_visible()
                    assert page.locator('.current-search').input_value() == parse_qs(urlsplit(suffix).query, keep_blank_values=True)['source-q'][0]
                page.close()
            report['checks'].append('Empty/nonempty source-q and actual source-section fragments demand the unchanged verifier initially')
            page=browser.new_page();watch(page)
            page.goto(origin+'compare.html');completed(page)
            page.evaluate("location.hash='current-date-explanation'")
            page.locator('#current-research[data-compare-source="ready"]').wait_for()
            assert page.locator('#current-date-explanation').evaluate('e=>e.open')
            page.close()
            report['checks'].append('Later native source-fragment navigation opens its target disclosure and verifies source inventory')
            # Fail only the optional source module; initial comparison remains admitted.
            page=browser.new_page();watch(page)
            page.route('**/current.mjs?*', lambda route: route.abort('failed'))
            page.goto(origin+'compare.html');completed(page)
            share=page.locator('#share').get_attribute('href')
            page.goto(share);completed(page)
            selected_query=urlsplit(page.url).query
            page.locator('.current-source-details > summary').click()
            page.locator('#current-research[data-compare-source="failed"]').wait_for()
            assert page.locator('#compare-source-retry').is_visible()
            assert page.locator('#compare-state').get_attribute('data-state') == 'completed'
            assert page.locator('.current-search').is_disabled()
            page.unroute('**/current.mjs?*')
            page.locator('#compare-source-retry').click();completed(page)
            page.locator('#current-research[data-compare-source="ready"]').wait_for()
            assert urlsplit(page.url).query == selected_query
            assert urlsplit(page.url).fragment == 'current-research'
            page.close()
            report['checks'].append('Optional-module import refusal stays visible and comparison intact; explicit reload retries with exact comparison selection retained')
            # Payload corruption remains an unchanged source-verifier refusal.
            page=browser.new_page();requests=watch(page)
            page.goto(origin+'compare.html');completed(page)
            def tamper(route):
                response=route.fetch();body=bytearray(response.body());body[0]=ord('[')
                route.fulfill(response=response,body=bytes(body))
            page.route('**/current.json*', tamper)
            page.locator('.current-source-details > summary').click()
            page.locator('#current-research[data-compare-source="failed"]').wait_for()
            assert page.locator('.current-state').inner_text().startswith('Unavailable:')
            assert page.locator('.current-search').is_disabled()
            assert page.locator('#compare-state').get_attribute('data-state') == 'completed'
            page.unroute('**/current.json*',tamper)
            page.locator('#compare-source-retry').click();completed(page)
            page.locator('#current-research[data-compare-source="ready"]').wait_for()
            page.close()
            report['checks'].append('Optional source payload corruption refuses without enabling search; explicit source reload recovers unchanged byte admission')
            # Comparison Retry still reacquires its required metadata, registry and payload.
            page=browser.new_page();requests=watch(page)
            page.goto(origin+'compare.html');completed(page);requests.clear()
            page.locator('#retry').click();completed(page)
            assert any(urlsplit(u).path=='/current-meta.json' for u in requests)
            assert any(urlsplit(u).path=='/snapshots/index.json' for u in requests)
            assert len(current_requests(requests))==1
            assert len([u for u in requests if '/snapshots/' in u and u.endswith('/current.json')])==2
            assert page.locator('.current-search').is_disabled()
            page.route('**/snapshots/*/current.json',lambda route:route.abort('failed'))
            page.locator('#retry').click();page.locator('#compare-state[data-state="refused"]').wait_for()
            for suffix in ['', '-top']:
                assert page.locator('#pagination'+suffix).is_hidden()
                assert page.locator('#next'+suffix).is_disabled()
            page.close()
            report['checks'].append('Comparison Retry freshly verifies metadata/registry/current/two archives; comparison refusal clears both pagers')
            nojs=browser.new_page(java_script_enabled=False,viewport={'width':390,'height':844});nojs.goto(origin+'compare.html')
            assert nojs.locator('#compare').is_disabled()
            nojs.locator('.current-source-details > summary').click()
            assert nojs.locator('.current-source-details a[href="current.json"]').is_visible()
            assert nojs.locator('.current-state').inner_text().startswith('Pinned source observation:')
            assert nojs.locator('noscript').first.is_visible()
            nojs.close();report['checks'].append('No-JavaScript pinned source identities, exact payload and static inventory/history fallback retained')
            assert not errors,errors
            report['page_errors']=errors
            browser.close()
    except Exception as error:
        report['failure']=str(error)
        raise
    finally:
        server.shutdown();server.server_close()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
