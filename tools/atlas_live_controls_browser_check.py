#!/usr/bin/env python3
"""Source-pinned live Atlas control regressions; --observe keeps baseline failures."""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from playwright.sync_api import expect, sync_playwright


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--browser', default='/usr/bin/chromium')
    ap.add_argument('--observe', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = (args.site / 'current.json').read_bytes()
    data = json.loads(raw)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *unused): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}'
    report = {'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'payload_sha256': hashlib.sha256(raw).hexdigest(), 'repositories': data['repositories'],
              'dataset': {'nodes': len(data['graph']['data']['nodes']), 'edges': len(data['graph']['data']['edges']), 'excerpts': len(data['equations'])},
              'cases': []}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True,
                                         args=['--enable-unsafe-swiftshader'])
            report['browser'] = browser.version
            for width, scheme in [(1200, 'light'), (1200, 'dark'), (390, 'light'), (390, 'dark')]:
                row = {'case': f'{width}-{scheme}', 'failures': []}; report['cases'].append(row)
                context = browser.new_context(viewport={'width': width, 'height': 900}, color_scheme=scheme, reduced_motion='reduce')
                page = context.new_page()
                def check(condition, name):
                    if not condition: row['failures'].append(name)
                try:
                    t = time.perf_counter(); page.goto(origin + '/dependencies.html?atlas-view=list')
                    expect(page.locator('#atlas-view')).to_be_enabled()
                    row['ready_ms'] = (time.perf_counter() - t) * 1000
                    row['decoded_resource_bytes'] = page.evaluate('() => performance.getEntriesByType("resource").reduce((n,r)=>n+r.decodedBodySize,0)')
                    expect(page.locator('#atlas-view')).to_have_value('list')
                    page.locator('#atlas-view').focus(); page.keyboard.press('Home'); page.keyboard.press('Enter')
                    expect(page.locator('#atlas-view')).to_have_value('graph')
                    page.locator('#atlas-view').select_option('list')
                    page.locator('#equation-search').scroll_into_view_if_needed()
                    # Observation only: no replacement of app responses or state.
                    page.evaluate('''() => {
                      window.optionBefore = [...document.querySelector('#equation-select').options];
                      window.optionEvents = 0; window.countEvents = 0;
                      new MutationObserver(rs => { for (const r of rs) window.optionEvents += r.addedNodes.length + r.removedNodes.length; }).observe(document.querySelector('#equation-select'), {childList:true,subtree:true});
                      new MutationObserver(rs => window.countEvents += rs.length).observe(document.querySelector('#equation-count'), {childList:true,subtree:true,characterData:true});
                    }''')
                    t = time.perf_counter(); page.locator('#equation-search').press_sequentially('sigma', delay=45)
                    row['typing_ms'] = (time.perf_counter() - t) * 1000
                    matches = [e for e in data['equations'] if 'sigma' in f"{e['text']} {e['path']} {e['repository']}".lower()]
                    expect(page.locator('#equation-count')).to_contain_text(f'{len(matches)} matching excerpts')
                    expect(page.locator('#equation-select')).to_have_value('')
                    row['option_events'] = page.evaluate('window.optionEvents')
                    row['count_events'] = page.evaluate('window.countEvents')
                    check(page.evaluate('''() => [...document.querySelector('#equation-select').options].every(o=>window.optionBefore.includes(o))'''), 'remaining options retain their identity')
                    check(row['option_events'] <= len(data['equations']), 'option event volume bounded by removed excerpts')
                    announcement = page.locator('#equation-announcement')
                    check(announcement.count() == 1, 'separate settled announcement exists')
                    if announcement.count():
                        t = time.perf_counter()
                        expect(announcement).to_have_text(f'{len(matches)} matching excerpts of {len(data["equations"])} acquired excerpts.')
                        row['announcement_wait_ms'] = (time.perf_counter() - t) * 1000
                        page.locator('#equation-search').fill('not-a-source-xyz')
                        page.locator('#equation-search').fill('sigma')
                        expect(announcement).to_have_text(f'{len(matches)} matching excerpts of {len(data["equations"])} acquired excerpts.')
                        # Observe beyond the deadline: the abandoned zero result must never reappear.
                        page.wait_for_timeout(650)
                        check('0 matching' not in announcement.text_content(), 'abandoned zero-result announcement cancelled')
                    if matches:
                        chosen = matches[0]
                        page.locator('#equation-select').select_option(chosen['id'])
                        expect(page.locator('#equation-content pre')).to_have_text(chosen['text'])
                        href = page.locator('#equation-content a').get_attribute('href')
                        check(chosen['commit'] in href and f'#L{chosen["line_start"]}-L{chosen["line_end"]}' in href, 'exact source revision and lines')
                        page.locator('#equation-search').fill('sigm')
                        expect(page.locator('#equation-select')).to_have_value(chosen['id'])
                        share = page.locator('#atlas-share').get_attribute('href')
                        page.goto(share); expect(page.locator('#equation-select')).to_have_value(chosen['id'])
                        expect(page.locator('#equation-content pre')).to_have_text(chosen['text'])
                        page.locator('#equation-search').fill('not-a-source-xyz')
                        expect(page.locator('#equation-select')).to_have_value('')
                        expect(page.locator('#equation-content')).to_contain_text('Select an acquired excerpt')
                    page.locator('#equation-search').fill('sigma')
                    page.locator('.atlas-equation').screenshot(path=str(args.output / f'equations-{width}-{scheme}.png'))
                    check(page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'no horizontal page overflow')
                    page.goto(origin + '/dependencies.html?snapshot=' + '0'*40 + '&atlas-view=list')
                    expect(page.locator('#atlas-error')).to_be_visible()
                    page.locator('#atlas-view').select_option('graph')
                    check('snapshot='+'0'*40 in page.locator('#atlas-share').get_attribute('href'), 'View preserves refused snapshot')
                    expect(page.locator('#atlas-error')).to_be_visible()
                    if announcement.count():
                        expect(announcement).to_have_text('Source-bound bookmark refused; no current equation substituted.')
                    page.go_back(); expect(page.locator('#atlas-error')).to_be_hidden()
                    page.go_forward(); expect(page.locator('#atlas-error')).to_be_visible()
                except Exception as exc:
                    row['failures'].append(str(exc))
                context.close(); row['passed'] = not row['failures']
                (args.output / 'controls-browser.json').write_text(json.dumps(report, indent=2)+'\n')
            browser.close()
    finally:
        server.shutdown(); server.server_close()
    print(json.dumps({'passed':sum(r['passed'] for r in report['cases']), 'total':len(report['cases']), 'report':str(args.output/'controls-browser.json')}))
    if not args.observe and any(not r['passed'] for r in report['cases']): raise SystemExit(1)


if __name__ == '__main__': main()
