#!/usr/bin/env python3
"""Observe or verify the selected claim's source/scope/evidence reading path."""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import platform
import subprocess
import threading
import time
import traceback

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def require(value, message):
    if not value:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT / 'docs/site')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', default='/usr/bin/chromium')
    parser.add_argument('--observe', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site.parent)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f'http://127.0.0.1:{server.server_port}/{args.site.name}/dependencies.html'
    payload = (args.site / 'current.json').read_bytes()
    data = json.loads(payload)
    selected = 'math.uniform-matrix-cap-lifetime'
    node = data['graph']['data']['nodes'][selected]
    report = {
        'mode': 'baseline-observation' if args.observe else 'acceptance',
        'checked_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'source_repositories': data['repositories'],
        'platform': platform.platform(), 'chromium_sandbox': True,
        'dataset': {'nodes': len(data['graph']['data']['nodes']), 'edges': len(data['graph']['data']['edges'])},
        'cases': [],
    }
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser, chromium_sandbox=True,
                                       args=['--enable-unsafe-swiftshader', '--disable-background-networking'])
            report['browser'] = browser.version
            for width, height in [(1200, 900), (390, 844)]:
                for scheme in ['light', 'dark']:
                    row = {'case': 'source-first-reader', 'width': width, 'scheme': scheme, 'passed': False, 'errors': []}
                    report['cases'].append(row)
                    context = browser.new_context(viewport={'width': width, 'height': height}, color_scheme=scheme, reduced_motion='reduce')
                    page = context.new_page()
                    page.on('pageerror', lambda error: row['errors'].append(str(error)))
                    page.set_default_timeout(12000)
                    try:
                        start = time.perf_counter()
                        page.goto(origin + '?atlas-node=' + selected + '&atlas-view=graph')
                        expect(page.locator('#atlas-inspector h3')).to_have_text(selected)
                        expect(page.locator('#atlas-formal-evidence')).to_be_attached()
                        row['load_ms'] = (time.perf_counter() - start) * 1000
                        if width < 721:
                            page.locator('#atlas-open-reader').click()
                        inspector = page.locator('#atlas-inspector')
                        source = inspector.locator('a', has_text='Immutable source reference')
                        expect(source).to_be_visible()
                        row['source'] = source.get_attribute('href')
                        expect(inspector).to_contain_text(node['scope'])
                        row['geometry'] = page.evaluate('''() => {
                          const panel=document.querySelector('#atlas-inspector'), base=panel.getBoundingClientRect();
                          const source=[...panel.querySelectorAll('a')].find(a=>a.textContent.includes('Immutable source reference'));
                          const formal=document.querySelector('#atlas-formal-evidence');
                          return {source_offset:source.getBoundingClientRect().top-base.top,
                            formal_offset:formal.getBoundingClientRect().top-base.top,
                            inspector_height:base.height,formal_height:formal.getBoundingClientRect().height};
                        }''')
                        inspector.evaluate('e=>e.scrollIntoView({block:"start"})')
                        page.screenshot(path=str(args.output / f'reader-{width}-{scheme}.png'))
                        inspector.screenshot(path=str(args.output / f'inspector-{width}-{scheme}.png'))
                        text = inspector.text_content()
                        require(text.index('Immutable source reference') < text.index('Resume this node') if 'Resume this node' in text else True,
                                'Primary source follows operational metadata')
                        require(page.locator('#atlas-claim-summary #atlas-formal-evidence').count() == 1,
                                'Formal gap is outside the selected-claim summary')
                        for identity in ['atlas-status', 'equation-count', 'atlas-space-status']:
                            require(page.locator('#' + identity).evaluate('(e)=>e.tagName==="P"&&e.parentElement.getAttribute("role")==="status"'),
                                    identity + ' lacks a readable paragraph inside its live region')
                        formal = page.locator('#atlas-formal-details')
                        require(not formal.evaluate('(e)=>e.open'), 'Formal details unexpectedly open by default')
                        expect(page.locator('#atlas-formal-evidence > p').first).to_be_visible()
                        start = time.perf_counter()
                        formal.locator('summary').first.focus()
                        page.keyboard.press('Enter')
                        expect(formal).to_have_attribute('open', '')
                        row['keyboard_disclosure_ms'] = (time.perf_counter() - start) * 1000
                        expect(formal).to_contain_text('No mapped trusted kernel-execution receipt')
                        review = page.locator('#atlas-review-details')
                        review.locator('summary').first.focus()
                        page.keyboard.press('Enter')
                        expect(page.locator('#atlas-resume')).to_be_visible()
                        if width < 721:
                            page.locator('#atlas-browse-link').click()
                        page.locator('#atlas-view').select_option('list')
                        expect(formal).to_have_attribute('open', '')
                        expect(review).to_have_attribute('open', '')
                        require(source.get_attribute('href') == row['source'], 'Layout changed the exact source')
                        page.reload()
                        expect(page.locator('#atlas-inspector h3')).to_have_text(selected)
                        expect(page.locator('#atlas-formal-details')).not_to_have_attribute('open', '')
                        require(page.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Document overflow')
                        require(not row['errors'], str(row['errors']))
                        row['passed'] = True
                    except Exception as error:
                        row['failure'] = str(error)
                        row['traceback'] = traceback.format_exc()
                    finally:
                        context.close()
            browser.close()
    finally:
        server.shutdown()
        report['passed'] = sum(row['passed'] for row in report['cases'])
        report['total'] = len(report['cases'])
        (args.output / 'summary-browser.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mode': report['mode'], 'passed': report['passed'], 'total': report['total']}))
    return 0 if args.observe or report['passed'] == report['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
