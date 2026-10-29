#!/usr/bin/env python3
"""Observe real Atlas fallback geometry and pointer selection, including context loss."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import threading
import time
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SELECTED = 'math.uniform-matrix-cap-lifetime'
TARGET = 'math.d1-component.congruence-erratum'
GEOMETRY = """() => {
 const cards=[...document.querySelectorAll('#atlas-canvas .atlas-node')];
 const rows=cards.map(e=>{const r=e.getBoundingClientRect();return {
  id:e.firstChild.textContent,x:r.x,y:r.y,width:r.width,height:r.height};});
 const overlaps=[];
 for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length;j++){
  const a=rows[i],b=rows[j],w=Math.min(a.x+a.width,b.x+b.width)-Math.max(a.x,b.x),
   h=Math.min(a.y+a.height,b.y+b.height)-Math.max(a.y,b.y);
  if(w>1&&h>1)overlaps.push({a:a.id,b:b.id,width:w,height:h});
 }
 return {cards:rows,overlaps};
}"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT / 'docs/site')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--browser', default='/usr/bin/chromium')
    parser.add_argument('--base-url', help='Replay the same assertions on a verified public origin')
    parser.add_argument('--only', default='', help='Exact case name')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'environment': {'platform': platform.platform(), 'browser': args.browser,
                        'sandbox': True, 'renderer': 'SwiftShader software WebGL',
                        'reduced_motion': 'reduce', 'physical_device': False},
        'cases': [],
        'limits': ['No human audition, physical touch/GPU or full WCAG certification.',
                   'Dataset and exact product/input identities must accompany this report.'],
    }
    if (args.site / 'current.json').exists():
        raw = (args.site / 'current.json').read_bytes()
        report['input'] = {'payload_bytes': len(raw), 'payload_sha256': hashlib.sha256(raw).hexdigest()}

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = None
    if args.base_url:
        origin = args.base_url.rstrip('/') + '/dependencies.html'
    else:
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(args.site)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        origin = f'http://127.0.0.1:{server.server_port}/dependencies.html'
    cases = [{'name': f'no-webgl-{width}-{theme}', 'width': width, 'theme': theme,
              'failure': 'webgl', 'view': 'graph'}
             for width in [1200, 390, 320] for theme in ['dark', 'light']]
    cases += [{'name': f'no-webgl-320-forced-{theme}', 'width': 320, 'theme': theme,
               'failure': 'webgl', 'view': 'graph', 'forced': True} for theme in ['dark', 'light']]
    cases += [{'name': 'module-failure-390', 'width': 390, 'theme': 'light',
               'failure': 'module', 'view': 'graph'},
              {'name': 'context-loss-filtered-1200', 'width': 1200, 'theme': 'dark',
               'failure': 'context', 'view': 'graph'},
              {'name': 'list-fallback-390', 'width': 390, 'theme': 'dark',
               'failure': 'webgl', 'view': 'list'},
              {'name': 'refused-source-390', 'width': 390, 'theme': 'light',
               'failure': 'webgl', 'view': 'graph', 'refused': True}]

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def check(row, name, condition, detail=None):
        row['checks'].append({'name': name, 'passed': bool(condition), 'detail': detail})

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=args.browser, chromium_sandbox=True,
                                         args=['--enable-unsafe-swiftshader'])
            report['environment']['browser_version'] = browser.version
            for spec in cases:
                if args.only and spec['name'] != args.only:
                    continue
                row = {**spec, 'checks': [], 'page_errors': [], 'passed': False}
                report['cases'].append(row)
                context = browser.new_context(viewport={'width': spec['width'], 'height': 900},
                                              color_scheme=spec['theme'], reduced_motion='reduce')
                if spec['failure'] == 'webgl':
                    context.add_init_script("const original=HTMLCanvasElement.prototype.getContext;"
                        "HTMLCanvasElement.prototype.getContext=function(type,...args){return "
                        "String(type).includes('webgl')?null:original.call(this,type,...args)}")
                if spec['failure'] == 'module':
                    context.route('**/atlas-spatial.mjs*', lambda route: route.abort())
                page = context.new_page()
                page.set_default_timeout(15000)
                page.on('pageerror', lambda error: row['page_errors'].append(str(error)))
                if spec.get('forced'):
                    page.emulate_media(forced_colors='active')
                try:
                    query = f'?atlas-node={SELECTED}&atlas-view={spec["view"]}'
                    if spec.get('refused'):
                        query += '&snapshot=' + '0' * 40
                    page.goto(origin + query)
                    expect(page.locator('#atlas-space-toggle')).to_be_visible()
                    if spec.get('refused'):
                        expect(page.locator('#atlas-error')).to_be_visible()
                    else:
                        expect(page.locator('#atlas-inspector h3')).to_have_text(SELECTED)
                    before_url = page.url
                    before_history = page.evaluate('history.length')
                    before_source = page.locator('#atlas-snapshot-record').inner_text()
                    if spec['view'] == 'graph' and not spec.get('refused'):
                        row['before_geometry'] = page.evaluate(GEOMETRY)
                        check(row, 'original visible graph has no overlaps', not row['before_geometry']['overlaps'])
                    started = time.perf_counter()
                    page.locator('#atlas-space-toggle').click()
                    if spec['failure'] == 'context':
                        host = page.locator('#atlas-space-host')
                        host.scroll_into_view_if_needed()
                        expect(host).to_have_attribute('data-ready', 'true')
                        canvas = page.locator('#atlas-space-host canvas')
                        check(row, 'actual renderer produced a canvas and frames',
                              canvas.count() == 1 and int(host.get_attribute('data-frames') or 0) > 0)
                        page.locator('#atlas-search').fill('math.d1-component')
                        expect(page.locator('#atlas-count')).to_contain_text('matching source nodes')
                        row['filtered_count'] = page.locator('#atlas-count').inner_text()
                        canvas.evaluate("e=>{const gl=e.getContext('webgl2');"
                                        "gl.getExtension('WEBGL_lose_context').loseContext()}")
                        expect(page.locator('#atlas-space-status')).to_contain_text('context lost')
                    elif spec.get('refused'):
                        expect(page.locator('#atlas-space-status')).to_contain_text('refused')
                    else:
                        expect(page.locator('#atlas-space-status')).to_contain_text('unavailable')
                    row['restoration_observed_ms'] = (time.perf_counter() - started) * 1000
                    failed_url = page.url
                    row['failure_url'] = failed_url
                    check(row, 'one activation history entry without a failure entry',
                          page.evaluate('history.length') == before_history + 1)
                    check(row, 'source observation retained',
                          page.locator('#atlas-snapshot-record').inner_text() == before_source)
                    check(row, 'failed renderer released', page.locator('#atlas-space-host canvas').count() == 0)
                    if spec.get('refused'):
                        check(row, 'refused source retains its original pin', 'snapshot=' + '0' * 40 in page.url)
                        check(row, 'refusal substitutes no current nodes',
                              page.locator('#atlas-canvas .atlas-node').count() == 0
                              and page.locator('#atlas-list button').count() == 0)
                    elif spec['view'] == 'list':
                        expect(page.locator('#atlas-list')).to_be_visible()
                        target = page.locator('#atlas-list button').filter(has_text=TARGET).first
                        target.focus()
                        page.keyboard.press('Space')
                        expect(page.locator('#atlas-inspector h3')).to_have_text(TARGET)
                        check(row, 'list fallback supports keyboard selection', True)
                    else:
                        expect(page.locator('#atlas-stage')).to_be_visible()
                        row['after_geometry'] = page.evaluate(GEOMETRY)
                        check(row, 'restored graph has no overlapping cards', not row['after_geometry']['overlaps'],
                              row['after_geometry']['overlaps'])
                        stage = page.locator('#atlas-stage')
                        card = page.locator('#atlas-canvas .atlas-node').filter(has_text=TARGET).first
                        stage.scroll_into_view_if_needed()
                        card.evaluate("e=>{const s=e.closest('#atlas-stage');s.scrollTo(e.offsetLeft-16,e.offsetTop-16)}")
                        page.wait_for_timeout(50)
                        point = card.locator('small').last.evaluate("e=>{const r=e.getBoundingClientRect();"
                            "const x=r.x+r.width/2,y=r.y+r.height/2;return {x,y,hit:"
                            "document.elementFromPoint(x,y)?.closest('.atlas-node')?.firstChild.textContent}}")
                        row['metadata_pointer'] = point
                        check(row, 'metadata point hits its own claim', point['hit'] == TARGET, point)
                        stage.screenshot(path=str(args.output / (spec['name'] + '-restored.png')))
                        page.mouse.click(point['x'], point['y'])
                        actual = page.locator('#atlas-inspector h3').inner_text()
                        row['actual_selected_after_mouse'] = actual
                        check(row, 'real mouse click selects the intended exact claim', actual == TARGET, actual)
                    if not spec.get('refused') and spec['view'] == 'graph' and spec['failure'] != 'context':
                        # Start again to observe failure traversal without the later deliberate selection.
                        page.goto(before_url)
                        expect(page.locator('#atlas-inspector h3')).to_have_text(SELECTED)
                        page.locator('#atlas-space-toggle').click()
                        expect(page.locator('#atlas-space-status')).to_contain_text('unavailable')
                        failed_url = page.url
                        length = page.evaluate('history.length')
                        page.go_back()
                        check(row, 'Back restores the original address', page.url == before_url)
                        page.go_forward()
                        expect(page.locator('#atlas-space-status')).to_contain_text('unavailable')
                        check(row, 'Forward retains failure intent without growing history',
                              page.url == failed_url and page.evaluate('history.length') == length)
                        check(row, 'Forward failure also restores nonoverlapping geometry', not page.evaluate(GEOMETRY)['overlaps'])
                    check(row, 'no uncaught page errors', not row['page_errors'], row['page_errors'])
                    row['passed'] = all(item['passed'] for item in row['checks'])
                except Exception as exc:
                    row['execution_error'] = str(exc)
                    row['traceback'] = traceback.format_exc()
                finally:
                    context.close()
                    save()
                    print(json.dumps({'case': row['name'], 'passed': row['passed'],
                        'failed_checks': [item['name'] for item in row['checks'] if not item['passed']],
                        'execution_error': row.get('execution_error')}), flush=True)
            browser.close()
    finally:
        if server:
            server.shutdown()
            server.server_close()
        report['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['passed'] = bool(report['cases']) and all(row['passed'] for row in report['cases'])
        save()
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
