#!/usr/bin/env python3
"""Bounded real-browser checks for the standalone retained snapshot comparison."""
import argparse
import functools
import hashlib
import subprocess
import http.server
import json
import re
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright
from current_site_history import (validate_manifest, mathematical_projection, mathematical_digest,
                                  manifest_bytes, render_static_v2, archive_record, history_size, capacity_record)


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
    try:
        source_head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_head = 'source ZIP or checkout without Git metadata'
    browser_path = Path(args.browser).resolve()
    browser_version = subprocess.run([str(browser_path), '--version'], capture_output=True, text=True, check=True).stdout.strip()
    report = {'origin': origin, 'checks': [], 'screenshots': [],
              'browser': {'executable': str(browser_path), 'sha256': hashlib.sha256(browser_path.read_bytes()).hexdigest(), 'version': browser_version, 'chromium_sandbox': True},
              'source': {'git_head': source_head, 'site': args.origin or str(site), 'comparison_files_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / 'docs/site' / name).read_bytes()).hexdigest() for name in ['compare.html', 'compare.mjs', 'compare-model.mjs', 'compare.css']}}}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.browser, headless=True, chromium_sandbox=True)
            page = browser.new_page(viewport={'width': 1440, 'height': 1000}, color_scheme='light')
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(origin + 'compare.html')
            page.locator('#compare-state[data-state="completed"]').wait_for()
            assert page.locator('body.reader-shell').count() == 1
            assert page.locator('meta[http-equiv="Content-Security-Policy"]').get_attribute('content') == "default-src 'self'; connect-src 'self' https://raw.githubusercontent.com; style-src 'self'; script-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; form-action 'none'"
            assert page.locator('body > header nav a').count() == 5
            assert page.locator('body > header nav a', has_text='Library').get_attribute('href') == 'workspace.html'
            assert page.locator('.reader-tools a').all_text_contents() == ['Reproduce', 'Cite']
            assert page.locator('footer .site-map a').count() == 14
            assert page.locator('#compare-state').evaluate("e => e.parentElement.getAttribute('role')") == 'status'
            for link in page.locator('body > header a').all():
                assert link.is_visible()
            for link in page.locator('body > header nav a').all():
                assert link.bounding_box()['height'] >= 44
            report['checks'].append('Shared shell, exact CSP, five primary links, 44px header targets, Research tools and 14-link footer map')
            assert page.locator('#current-research .current-search').is_disabled()
            assert 'pinned and unverified' in page.locator('#compare-source-note').inner_text()
            page.locator('#current-research .current-source-details > summary').click()
            page.wait_for_function("() => !document.querySelector('#current-research .current-search').disabled")
            report['checks'].append('Separate source inventory stays pinned until requested; existing current loader then verifies and initializes controls')
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
            for link in page.locator('body > header a').all():
                assert link.is_visible()
            for link in page.locator('body > header nav a').all():
                assert link.bounding_box()['height'] >= 44
            assert page.locator('footer .site-map a').count() == 14
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
            page.locator('#to').select_option(page.locator('#to option').last.get_attribute('value'))
            assert page.locator('#compare').is_enabled()
            page.locator('#compare').click()
            page.locator('#compare-state[data-state="completed"]').wait_for()
            assert 'forged' not in page.url
            report['checks'].append('Explicit valid selector choice recovers invalid share URL through Load')
            # A producer-admitted publication can expose the same full payload
            # as both a retained archive and the current observation. Preserve roles.
            alias_registry = page.request.get(origin + 'snapshots/index.json').json()
            alias_archive = next(a for a in alias_registry['archives'] if a['id'] == alias_registry['current'])
            archive_descriptor = next(d for d in alias_archive['files'] if d['path'] == 'current.json')
            alias_payload = page.request.get(origin + f"snapshots/{alias_archive['id']}/current.json").body()
            alias_registry['total_bytes'] += archive_descriptor['bytes'] - alias_registry['current_observation']['payload']['bytes']
            alias_registry['capacity']['bytes'] = alias_registry['total_bytes']
            alias_registry['current_observation']['payload'] = {**archive_descriptor, 'path': 'current.json'}
            validate_manifest(alias_registry)
            alias_registry_raw = json.dumps(alias_registry).encode()
            alias_meta = {'archive': alias_archive['id'], 'sha256': archive_descriptor['sha256'],
                          'bytes': archive_descriptor['bytes'], 'observed_at': alias_archive['observed_at'],
                          'history_manifest_sha256': hashlib.sha256(alias_registry_raw).hexdigest()}
            alias_html = page.request.get(origin + 'compare.html').text()
            alias_html = re.sub(r'data-current-digest="[0-9a-f]{64}"', f'data-current-digest="{archive_descriptor["sha256"]}"', alias_html)
            alias_html = re.sub(r'data-current-bytes="[0-9]+"', f'data-current-bytes="{archive_descriptor["bytes"]}"', alias_html)
            alias_page = browser.new_page(viewport={'width': 390, 'height': 844}, color_scheme='dark')
            alias_page.on('pageerror', lambda error: errors.append(str(error)))
            fixtures = {'compare.html': (alias_html, 'text/html'),
                        'current-meta.json': (json.dumps(alias_meta), 'application/json'),
                        'snapshots/index.json': (alias_registry_raw, 'application/json'),
                        'current.json': (alias_payload, 'application/json')}
            def alias_response(route):
                body, content_type = fixtures[route.request.url.removeprefix(origin)]
                route.fulfill(body=body, content_type=content_type)
            for path in fixtures:
                alias_page.route(origin + path, alias_response)
            alias_page.goto(origin + 'compare.html')
            alias_page.locator('#compare-state[data-state="completed"]').wait_for()
            alias_page.locator('#from').select_option('archive:' + alias_archive['id'])
            alias_page.locator('#to').select_option('current:' + alias_archive['id'])
            alias_page.locator('#compare').click()
            alias_page.locator('#compare-state[data-state="completed"]').wait_for()
            assert all(count == '0' for count in alias_page.locator('.comparison-counts dd').all_text_contents())
            exact = alias_page.locator('.exact-selection').inner_text()
            assert 'retained archive' in exact and 'current observation' in exact
            assert ('payload SHA-256 ' + archive_descriptor['sha256']) in exact
            assert (str(archive_descriptor['bytes']) + ' bytes') in exact
            screenshot = args.output / 'compare-identical-current-archive-fixture.png'
            alias_page.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            report['checks'].append('Producer-validated same-byte archive/current fixture preserves typed roles and yields zero deltas')
            # Produce and admit an actual schema2 archive in memory. Its material
            # identity deliberately differs from the exact projection payload hash.
            projection = mathematical_projection(json.loads(alias_payload))
            projection_raw = manifest_bytes(projection)
            material_id = mathematical_digest(projection)
            projection_meta = {'sha256': hashlib.sha256(projection_raw).hexdigest(),
                               'bytes': len(projection_raw), 'observed_at': projection['observed_at'],
                               'mathematical_sha256': material_id, 'archive_version': 2}
            projection_files = {**render_static_v2(projection, projection_meta),
                                'current.json': projection_raw, 'current-meta.json': manifest_bytes(projection_meta)}
            material_entry, _, _ = archive_record(material_id, projection_files)
            alias_registry['archives'].append(material_entry)
            alias_registry['total_bytes'] = history_size(alias_registry['archives'], alias_registry['aliases'], alias_registry['current_observation'])
            alias_registry['capacity'] = capacity_record(alias_registry['archives'], alias_registry['total_bytes'])
            validate_manifest(alias_registry)
            material_registry_raw = manifest_bytes(alias_registry)
            alias_meta['history_manifest_sha256'] = hashlib.sha256(material_registry_raw).hexdigest()
            fixtures['current-meta.json'] = (json.dumps(alias_meta), 'application/json')
            fixtures['snapshots/index.json'] = (material_registry_raw, 'application/json')
            material_path = f'snapshots/{material_id}/current.json'
            fixtures[material_path] = (projection_raw, 'application/json')
            alias_page.route(origin + material_path, alias_response)
            alias_page.goto(origin + 'compare.html')
            alias_page.locator('#compare-state[data-state="completed"]').wait_for()
            alias_page.locator('#from').select_option('archive:' + alias_archive['id'])
            alias_page.locator('#to').select_option('archive:' + material_id)
            alias_page.locator('#compare').click()
            alias_page.locator('#compare-state[data-state="completed"]').wait_for()
            material_text = alias_page.locator('.exact-selection').inner_text()
            assert ('material archive ID ' + material_id) in material_text
            assert ('payload SHA-256 ' + projection_meta['sha256']) in material_text
            assert (str(len(projection_raw)) + ' bytes') in material_text
            assert material_id != projection_meta['sha256']
            assert 'Inventory coverage: incomparable' in alias_page.locator('#comparison-summary').inner_text()
            assert all(count == '0' for count in alias_page.locator('.comparison-counts dd').all_text_contents())
            screenshot = args.output / 'compare-material-archive-fixture.png'
            alias_page.screenshot(path=str(screenshot), full_page=True)
            report['screenshots'].append(str(screenshot))
            report['checks'].append('Producer-generated/admitted schema2 fixture displays distinct material ID and exact payload SHA/bytes; omitted inventory incomparable')
            alias_page.close()
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
