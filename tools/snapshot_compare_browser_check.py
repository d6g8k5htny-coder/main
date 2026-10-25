#!/usr/bin/env python3
"""Bounded real-browser checks for the standalone retained snapshot comparison."""
import argparse
import functools
import http.server
import json
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--origin', help='Existing site root URL; otherwise serve local docs/site')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    server = None
    if args.origin:
        origin = args.origin.rstrip('/') + '/'
    else:
        site = Path(__file__).resolve().parents[1] / 'docs/site'
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Quiet, directory=str(site)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f'http://127.0.0.1:{server.server_port}/'
    report = {'origin': origin, 'checks': [], 'screenshots': []}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser, headless=True, chromium_sandbox=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, color_scheme='light')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(origin + 'compare.html')
            page.locator('#compare-state[data-state="completed"]').wait_for()
            summary = page.locator('#comparison-summary').inner_text()
            assert 'Source commit pins' in summary and 'Source blob / byte identities' in summary
            report['actual_summary'] = summary
            assert page.locator('#comparison-results .change').count() <= 20
            first = page.locator('#comparison-results .change code').first.inner_text()
            page.locator('#next').click()
            assert page.locator('#comparison-results .change code').first.inner_text() != first
            report['checks'].append('Actual retained archives verified; independently rendered pin/content categories; bounded pagination')
            page.locator('#compare').click()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            share = page.locator('#share').get_attribute('href')
            assert 'from=' in share and 'to=' in share
            page.goto(share)
            page.locator('#compare-state[data-state="completed"]').wait_for()
            old_to = page.locator('#to').input_value()
            current = page.locator('#to option').last.get_attribute('value')
            page.locator('#to').select_option(current)
            assert page.locator('#compare-state').get_attribute('data-state') == 'awaiting'
            assert page.locator('#comparison-results .change').count() == 0
            page.locator('#compare').click()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            page.reload()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            assert page.locator('#to').input_value() == current
            page.go_back()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            assert page.locator('#to').input_value() == old_to
            report['checks'].append('Validated share URL, current selection, reload and native back')
            pending = []
            page.route('**/snapshots/*/current.json', lambda route: pending.append(route))
            with page.expect_request('**/snapshots/*/current.json'):
                page.locator('#compare').click()
            page.locator('#to').select_option(current)
            for route in pending:
                route.continue_()
            page.unroute('**/snapshots/*/current.json')
            page.wait_for_load_state('networkidle')
            assert page.locator('#compare-state').get_attribute('data-state') == 'awaiting'
            assert page.locator('#comparison-results .change').count() == 0
            page.locator('#to').select_option(old_to)
            page.locator('#compare').click()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            report['checks'].append('Late archive completion cannot render after selector change')
            page.keyboard.press('Tab')
            assert page.evaluate('document.activeElement !== document.body')
            screenshot = args.output / 'compare-desktop-light.png'
            page.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            page.emulate_media(color_scheme='dark', reduced_motion='reduce')
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            screenshot = args.output / 'compare-mobile-dark.png'
            page.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            report['checks'].append('Narrow dark/reduced-motion layout and keyboard focus')
            # Tamper a selected archived payload while preserving its advertised descriptor.
            def tamper(route):
                response = route.fetch()
                body = bytearray(response.body())
                body[0] = ord('[')
                route.fulfill(response=response, body=bytes(body))
            page.route('**/snapshots/*/current.json', tamper)
            page.reload()
            page.locator('#compare-state[data-state="refused"]').wait_for()
            assert 'digest' in page.locator('#compare-state').inner_text().lower()
            assert page.locator('#comparison-results .change').count() == 0
            assert page.locator('#retry').is_visible()
            screenshot = args.output / 'compare-refused.png'
            page.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            page.unroute('**/snapshots/*/current.json', tamper)
            page.locator('#retry').click()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            report['checks'].append('Digest tamper refusal clears results; explicit retry recovers exact publication')
            page.goto(origin + 'compare.html?from=forged&to=forged')
            page.locator('#compare-state[data-state="refused"]').wait_for()
            assert page.locator('#comparison-results .change').count() == 0
            report['checks'].append('Forged selection IDs refuse without substitute comparison')
            nojs = browser.new_page(java_script_enabled=False, viewport={'width': 390, 'height': 844})
            nojs.goto(origin + 'compare.html')
            assert nojs.locator('noscript').first.is_visible()
            assert nojs.locator('#compare').is_disabled()
            screenshot = args.output / 'compare-nojs.png'
            nojs.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            report['checks'].append('No-JavaScript static inventory/registry route and disabled comparison')
            assert not errors, errors
            report['page_errors'] = errors
            browser.close()
    finally:
        if server:
            server.shutdown()
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
