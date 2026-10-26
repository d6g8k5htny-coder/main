#!/usr/bin/env python3
"""Compare the same admitted Atlas journey before/after presentation changes.

--observe retains baseline failures; it never turns them into acceptance.
No source responses, browser security or application assertions are replaced.
"""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import threading
import time
import traceback
from urllib.parse import urlencode

from playwright.sync_api import expect, sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', default='/usr/bin/chromium')
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = (args.site / 'current.json').read_bytes()
    data = json.loads(raw)
    meta = json.loads((args.site / 'current-meta.json').read_bytes())
    assert len(raw) == meta['bytes'] and hashlib.sha256(raw).hexdigest() == meta['sha256']

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *unused):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    report = {'payload_sha256': meta['sha256'], 'source_repositories': data['repositories'],
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'environment': {'platform': platform.platform(), 'sandbox': True,
                              'renderer': 'Chromium software SwiftShader', 'DPR': 1},
              'served_modules': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in args.site.iterdir() if p.suffix in ('.mjs', '.css')},
              'cases': []}
    def save():
        (args.output / 'approachable-browser.json').write_text(json.dumps(report, indent=2) + '\n')

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True,
                                         args=['--enable-unsafe-swiftshader', '--disable-background-networking'])
            report['environment']['browser'] = browser.version
            for width, height in [(1200, 900), (390, 844)]:
                for scheme in ['light', 'dark']:
                    row = {'case': f'{width}-{scheme}', 'failures': [], 'page_errors': []}
                    report['cases'].append(row)
                    context = browser.new_context(viewport={'width': width, 'height': height},
                                                  color_scheme=scheme, reduced_motion='reduce')
                    page = context.new_page()
                    page.on('pageerror', lambda error, r=row: r['page_errors'].append(str(error)))
                    page.set_default_timeout(12000)
                    def check(condition, label):
                        if not condition:
                            row['failures'].append(label)
                    try:
                        start = time.perf_counter()
                        page.goto(origin + '/dependencies.html?' + urlencode({'atlas-node': 'math.side24-coefficient', 'atlas-view': 'list'}))
                        expect(page.locator('#atlas-search')).to_be_enabled()
                        row['load_ms'] = (time.perf_counter() - start) * 1000
                        row['status_text'] = page.locator('#atlas-status').inner_text()
                        row['search_top'] = page.locator('#atlas-search').bounding_box()['y']
                        row['resources_before_3d'] = page.evaluate('() => performance.getEntriesByType("resource").reduce((n,r)=>n+r.decodedBodySize,0)')
                        row['selected'] = page.locator('#atlas-inspector h3').text_content()
                        check(page.locator('#atlas-status').is_visible(), 'persistent status visible')
                        check(data['observed_at'] not in row['status_text'], 'observation metadata collapsed by default')
                        details = page.locator('#atlas-snapshot-details')
                        check(details.count() == 1, 'native provenance toggle exists')
                        check(page.locator('#atlas-foundations').count() == 1, 'native foundational guide exists')
                        page.screenshot(path=str(args.output / f'initial-{width}-{scheme}.png'))
                        if details.count():
                            expect(details).to_be_visible()
                            check(not details.evaluate('(e)=>e.open'), 'provenance initially collapsed')
                            summary = details.locator(':scope > summary')
                            summary.focus()
                            before_url = page.url
                            start = time.perf_counter()
                            page.keyboard.press('Enter')
                            expect(details).to_have_attribute('open', '')
                            row['disclosure_ms'] = (time.perf_counter() - start) * 1000
                            check(summary.evaluate('(e)=>document.activeElement===e'), 'Enter preserves summary focus')
                            expect(page.locator('#atlas-snapshot-record')).to_contain_text(data['observed_at'])
                            for repo in data['repositories']:
                                link = details.get_by_role('link', name=repo['repository'] + ' · ' + repo['commit'], exact=True)
                                expect(link).to_have_attribute('href', f'https://github.com/{repo["repository"]}/tree/{repo["commit"]}')
                            check(page.url == before_url, 'toggle leaves share URL unchanged')
                            page.keyboard.press('Space')
                            expect(details).not_to_have_attribute('open', '')
                            check(summary.evaluate('(e)=>document.activeElement===e'), 'Space preserves summary focus')
                            # Existing renderer replaces inspector children, not provenance.
                            page.locator('#atlas-search').fill('side24')
                            check(page.locator('#atlas-inspector h3').text_content() == row['selected'], 'search preserves selected claim')
                            page.locator('#atlas-search').fill('')
                        if page.locator('#atlas-foundations').count():
                            guide = page.locator('#atlas-foundations')
                            guide.locator('summary').focus()
                            page.keyboard.press('Enter')
                            expect(guide).to_have_attribute('open', '')
                            for word in ['Random landscape', 'Threshold', 'Persistence', 'Hypotheses']:
                                expect(guide).to_contain_text(word)
                            page.keyboard.press('Enter')
                        check(not row['page_errors'], 'no page errors')
                    except Exception as error:
                        row['failures'].append(str(error))
                        row['traceback'] = traceback.format_exc()
                    finally:
                        row['passed'] = not row['failures']
                        context.close()
                        save()
            # Static guide remains useful before admission/without JS.
            row = {'case': 'refused-provenance-history', 'failures': []}
            report['cases'].append(row)
            context = browser.new_context(viewport={'width': 390, 'height': 844})
            page = context.new_page()
            try:
                bad_url = origin + '/dependencies.html?snapshot=' + '0' * 40
                page.goto(bad_url)
                expect(page.locator('#atlas-error')).to_be_visible()
                details = page.locator('#atlas-snapshot-details')
                expect(details).to_have_count(1)
                expect(details).to_be_hidden()
                page.locator('#atlas-search').fill('side24')
                expect(details).to_be_hidden()
                expect(page.locator('#atlas-share')).to_have_attribute('href', page.url)
                page.goto(origin + '/dependencies.html?atlas-node=math.side24-coefficient')
                expect(details).to_be_visible()
                page.go_back()
                expect(page.locator('#atlas-error')).to_be_visible()
                expect(details).to_be_hidden()
                page.go_forward()
                expect(details).to_be_visible()
                row['passed'] = True
            except Exception as error:
                row['failures'].append(str(error))
                row['traceback'] = traceback.format_exc()
                row['passed'] = False
            finally:
                context.close()
                save()
            for javascript in [False, True]:
                row = {'case': 'no-js' if not javascript else 'bad-source', 'failures': []}
                report['cases'].append(row)
                context = browser.new_context(java_script_enabled=javascript, viewport={'width': 390, 'height': 844})
                page = context.new_page()
                try:
                    if javascript:
                        page.route('**/current.json*', lambda route: route.fulfill(status=200, content_type='application/json', body='{}'))
                    page.goto(origin + '/dependencies.html')
                    if javascript:
                        expect(page.locator('#atlas-status')).to_contain_text('did not verify')
                        expect(page.locator('#atlas-status')).to_be_visible()
                        expect(page.locator('#atlas-search')).to_be_disabled()
                    else:
                        expect(page.locator('noscript a[href="nodes/index.html"]').first).to_be_visible()
                    if page.locator('#atlas-foundations').count() != 1:
                        row['failures'].append('foundational guide absent')
                    else:
                        guide = page.locator('#atlas-foundations')
                        guide.locator('summary').focus()
                        page.keyboard.press('Enter')
                        expect(guide).to_have_attribute('open', '')
                        expect(guide).to_contain_text('Persistence')
                    if page.locator('#atlas-snapshot-details').count():
                        expect(page.locator('#atlas-snapshot-details')).to_be_hidden()
                except Exception as error:
                    row['failures'].append(str(error))
                    row['traceback'] = traceback.format_exc()
                finally:
                    row['passed'] = not row['failures']
                    context.close()
                    save()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        report['passed'] = sum(c['passed'] for c in report['cases'])
        report['total'] = len(report['cases'])
        save()
    print(json.dumps({'passed': report['passed'], 'total': report['total'],
                      'failures': [c['failures'] for c in report['cases'] if not c['passed']]}))
    return 0 if args.observe or report['passed'] == report['total'] == 7 else 1


if __name__ == '__main__':
    raise SystemExit(main())
