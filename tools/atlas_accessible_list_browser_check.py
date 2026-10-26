#!/usr/bin/env python3
"""Check actual 3D accessible-list metadata contrast with the inherited button ink."""
import argparse
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from playwright.sync_api import expect, sync_playwright


CONTRAST = r'''() => {
 const rgb=s=>s.match(/[\d.]+/g).slice(0,3).map(Number);
 const luminance=c=>c.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((n,v,i)=>n+v*[.2126,.7152,.0722][i],0);
 return [...document.querySelectorAll('.atlas-space ul button small')].map(e=>{
  const text=getComputedStyle(e).color,button=getComputedStyle(e.parentElement),background=button.backgroundColor;
  const a=luminance(rgb(text)),b=luminance(rgb(background));
  return {text,background,button_text:button.color,ratio:(Math.max(a,b)+.05)/(Math.min(a,b)+.05),font_size:getComputedStyle(e).fontSize};
 });
}'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--browser', default='/usr/bin/chromium')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *unused):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(a.site)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    report = {'payload_sha256': hashlib.sha256((a.site / 'current.json').read_bytes()).hexdigest(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'stylesheet_sha256': hashlib.sha256((a.site / 'atlas-spatial.css').read_bytes()).hexdigest(),
              'sandbox': True, 'renderer': 'Chromium SwiftShader software WebGL', 'cases': []}
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=a.browser, chromium_sandbox=True,
                                         args=['--enable-unsafe-swiftshader'])
            report['browser'] = browser.version
            for scheme in ['light', 'dark']:
                row = {'scheme': scheme, 'passed': False}
                report['cases'].append(row)
                context = browser.new_context(viewport={'width': 1200, 'height': 900}, color_scheme=scheme, reduced_motion='reduce')
                page = context.new_page()
                try:
                    page.goto(f'http://127.0.0.1:{server.server_port}/dependencies.html?atlas-view=graph&atlas-node=math.uniform-matrix-cap-lifetime')
                    page.locator('#atlas-space-toggle').click()
                    page.locator('#atlas-space-host').scroll_into_view_if_needed()
                    expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready', 'true')
                    expect(page.locator('#atlas-space-host canvas')).to_have_count(1)
                    control = page.locator('.atlas-space ul button').first
                    control.focus()
                    row['metadata'] = page.evaluate(CONTRAST)
                    assert row['metadata'], 'No metadata tested'
                    assert all(x['ratio'] >= 4.5 for x in row['metadata']), 'Metadata text contrast below 4.5:1'
                    row['passed'] = True
                except Exception as error:
                    row['failure'] = str(error)
                finally:
                    context.close()
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        (a.output / 'contrast-browser.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': sum(x['passed'] for x in report['cases']), 'total': len(report['cases'])}))
    return 0 if len(report['cases']) == 2 and all(x['passed'] for x in report['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
