#!/usr/bin/env python3
"""Real reading-surface/navigation regressions; no scientific or full-WCAG claim."""
import argparse
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def serve(site):
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try: yield f'http://127.0.0.1:{server.server_port}/'
    finally: server.shutdown(); server.server_close(); worker.join()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--site', type=Path, default=ROOT / 'docs/site')
    ap.add_argument('--origin', help='Served origin; local checkout metadata never identifies remote code.')
    ap.add_argument('--browser', default=shutil.which('chromium'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--observe', action='store_true', help='Record baseline failures without acceptance.')
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright, expect
    report = {'cases': [], 'page_errors': [], 'mode': 'served-origin' if args.origin else 'local-site',
              'local_checkout_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'local_checkout_sha256': {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in ['docs/site/reader-jump.css', 'docs/site/reader-jump.mjs', 'docs/site/reader-shell.mjs']},
              'identity_scope': 'Checkout identities describe local files only; served identity needs a separate source/publication byte proof.',
              'limits': 'Sandboxed Linux Chromium emulation, not physical Safari/touch, spoken AT, or full WCAG conformance.'}

    def run(origin, browser):
        report['origin'] = origin

        def case(name, width, height, scheme, action, scripted=True):
            row = {'case': name, 'width': width, 'height': height, 'scheme': scheme, 'scripted': scripted}
            report['cases'].append(row)
            c = browser.new_context(viewport={'width': width, 'height': height}, color_scheme=scheme,
                                    java_script_enabled=scripted, reduced_motion='reduce')
            c.route('**/*', lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
            p = c.new_page(); p.set_default_timeout(7000)
            p.on('pageerror', lambda error: report['page_errors'].append(str(error)))
            try:
                action(p, row)
                assert p.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'Horizontal document overflow'
                row['passed'] = True
            except Exception:
                row['passed'] = False; row['error'] = traceback.format_exc()
            finally:
                p.screenshot(path=str(args.output / (name + '.png')))
                c.close()

        for width, scheme in [(320, 'light'), (320, 'dark'), (390, 'light'), (390, 'dark'), (1440, 'light'), (1440, 'dark')]:
            def surface(p, row):
                p.goto(origin + 'workspace.html')
                disclosure = p.locator('#current-research > .current-source-details')
                disclosure.evaluate('(e)=>e.open=true')
                capacity = p.locator('.current-capacity')
                capacity.evaluate('(e)=>e.scrollIntoView({block:"end",behavior:"instant"})')
                p.evaluate('document.activeElement.blur()')
                geometry = p.evaluate('''() => {
                    const launcher=document.querySelector('.reader-jump > summary').getBoundingClientRect();
                    const text=document.querySelector('.current-capacity'),range=document.createRange();range.selectNodeContents(text);
                    const rects=[...range.getClientRects()].filter(r=>r.width&&r.height&&r.bottom>0&&r.top<innerHeight);
                    const overlap=r=>r.left<launcher.right&&r.right>launcher.left&&r.top<launcher.bottom&&r.bottom>launcher.top;
                    return {launcher:launcher.toJSON(),textRects:rects.map(r=>r.toJSON()),overlap:rects.some(overlap)};
                }''')
                row['geometry'] = geometry
                assert not geometry['overlap'], 'Closed Go to covers source capacity text'
                assert p.locator('.reader-tools > .reader-jump').count() == 1, 'Use the existing tools strip'
            case(f'reading-surface-{width}-{scheme}', width, 844, scheme, surface)

            def repeat(p, row):
                p.goto(origin + 'workspace.html#reading-list')
                reading = p.locator('#reading-list'); expect(reading).to_have_attribute('open', '')
                reading.locator(':scope > summary').click(); assert not reading.evaluate('(e)=>e.open')
                p.locator('.reader-jump > summary').click()
                p.locator('.reader-jump').get_by_role('link', name='Reading list', exact=True).click()
                expect(reading).to_have_attribute('open', '')
                assert p.url.endswith('#reading-list')
                row['target_bounds'] = reading.locator(':scope > summary').bounding_box()
                assert not p.locator('.reader-jump').evaluate('(e)=>e.open')
            case(f'repeated-reading-list-{width}-{scheme}', width, 844, scheme, repeat)

            def categories(p, row):
                p.goto(origin + 'compare.html')
                expect(p.locator('#compare-state')).to_have_attribute('data-state', 'completed')
                choice = p.locator('#compare-category')
                expect(choice).to_have_value('all')
                initial = p.locator('#comparison-results').inner_text()
                pair = [p.locator('#' + key).input_value() for key in ['from', 'to']]
                share = p.locator('#share').get_attribute('href')
                choice.focus(); choice.select_option('content')
                expect(choice).to_be_focused()
                kinds = p.locator('#comparison-results article h3').all_text_contents()
                assert kinds and all(x == 'Source blob / byte identities' for x in kinds)
                row['direct_content_rows'] = len(kinds)
                assert p.locator('#page-status').inner_text().startswith('1–')
                choice.select_option('references'); expect(choice).to_be_focused()
                assert all(x == 'Reference identity records' for x in p.locator('#comparison-results article h3').all_text_contents())
                empty = next((o.get_attribute('value') for o in choice.locator('option').all() if o.inner_text().endswith('(0)')), None)
                if empty:
                    choice.select_option(empty)
                    expect(p.locator('#comparison-results article')).to_have_count(0)
                    expect(p.locator('#comparison-results')).to_contain_text('No record differences')
                    expect(p.locator('#pagination-top')).to_be_hidden()
                choice.select_option('all')
                assert p.locator('#comparison-results').inner_text() == initial, 'All-category order or records changed'
                if p.locator('#next-top').is_enabled():
                    p.locator('#next-top').click()
                    assert p.locator('#page-status').inner_text().startswith('21–')
                    choice.focus(); choice.select_option('content'); expect(choice).to_be_focused()
                    assert p.locator('#page-status').inner_text().startswith('1–')
                assert pair == [p.locator('#' + key).input_value() for key in ['from', 'to']]
                assert share == p.locator('#share').get_attribute('href'), 'Category browsing changed the source-pair link'
                # Changing either observation must not leave a stale category view.
                p.locator('#to').select_option(pair[0])
                expect(choice).to_have_count(0)
                expect(p.locator('#comparison-results article')).to_have_count(0)
                expect(p.locator('#compare-state')).to_have_attribute('data-state', 'awaiting')
            case(f'compare-categories-{width}-{scheme}', width, 844, scheme, categories)

        for width, height in [(320, 390), (844, 390), (390, 844)]:
            for scripted in [True, False]:
                def menu(p, row):
                    p.goto(origin + ('workspace.html' if scripted else 'formal.html'))
                    summary = p.locator('.reader-jump > summary'); summary.focus(); summary.press('Space')
                    nav = p.locator('.reader-jump > nav'); expect(nav).to_be_visible()
                    # Native-only pages retain ordinary page/inner-menu scrolling.
                    if not scripted:
                        p.locator('.reader-jump').evaluate('(e)=>e.scrollIntoView({block:"start",behavior:"instant"})')
                    last = nav.locator('a').last; last.focus()
                    box = last.bounding_box(); row['last_link_bounds'] = box
                    assert box and box['y'] >= 0 and box['y'] + box['height'] <= height, 'Last keyboard link is clipped'
                    assert nav.evaluate('(e)=>e.scrollHeight>=e.clientHeight')
                    if scripted:
                        last.press('Escape'); expect(summary).to_be_focused()
                        assert not p.locator('.reader-jump').evaluate('(e)=>e.open')
                    else:
                        assert '<script' not in p.content().lower(), 'Formal remains script-free'
                case(f'bounded-menu-{width}-{height}-js{int(scripted)}', width, height, 'light', menu, scripted)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True)
        report['browser'] = browser.version
        if args.origin: run(args.origin.rstrip('/') + '/', browser)
        else:
            with serve(args.site) as origin: run(origin, browser)
        browser.close()
    report['passed'] = sum(x['passed'] for x in report['cases'])
    report['failed'] = len(report['cases']) - report['passed']
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['browser', 'passed', 'failed', 'page_errors']}))
    return 1 if report['failed'] or report['page_errors'] else 0


if __name__ == '__main__': raise SystemExit(main())
