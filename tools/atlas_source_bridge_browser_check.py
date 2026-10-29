"""Exercise the explicit current-to-historical bridge in the real Atlas.

Fresh output only. Application CSP, source admission and original receipts stay
unchanged. Successful software WebGL is tested separately from any fallback.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import importlib.metadata
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright,expect

from atlas_formal_evidence_browser_check import identity,digest

NODE='math.side24-coefficient'
PROFILES={
    '8228e896c728d95ab94391899320e9d44ab7cdfc':('8228',15,0,0),
    'a22f06a3a7b9ab451e9e344d7b2345d2e2ab0553':('a22f',13,2,2),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--browser',default='/usr/bin/chromium')
    parser.add_argument('--base-url',help='Verified public origin; no assertion changes')
    args=parser.parse_args();site=args.site.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    raw=(site/'current.json').read_bytes();data=json.loads(raw)
    meta=(site/'current-meta.json').read_bytes();m=json.loads(meta)
    assert m['sha256']==digest(raw) and m['bytes']==len(raw)
    mainpin=next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))
    mathpin=data['graph']['identity']['commit']
    manifest=next(s for s in data['sources'] if s['repository']=='d6g8k5htny-coder/main' and s['path']=='formal/manifest.json')
    profile,identical_count,documentation_count,annotation_count=PROFILES[manifest['blob']]
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*values):requests.append({'path':self.path,'message':fmt%values})
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(site)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    origin=args.base_url.rstrip('/') if args.base_url else f'http://127.0.0.1:{server.server_port}'
    query={'atlas-node':NODE,'atlas-view':'graph','snapshot':mathpin,'main-snapshot':mainpin}
    url=origin+'/dependencies.html?'+urlencode(query)
    report={'scope':'Source-bridge browser engineering; no AT, semantic alignment or scientific acceptance.',
            'payload':identity(site/'current.json'),'repositories':data['repositories'],
            'driver':identity(Path(__file__)),'playwright':importlib.metadata.version('playwright'),
            'sandbox':True,'webgl':'SwiftShader; real rendering, not physical GPU qualification',
            'served_files':{p.relative_to(site).as_posix():identity(p) for p in sorted(site.rglob('*')) if p.is_file()},
            'cases':[]}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=args.browser,chromium_sandbox=True,
                proxy={'server':'http://proxy:8080'} if args.base_url else None,
                args=['--enable-unsafe-swiftshader','--disable-background-networking'])
            report['browser']=browser.version
            for width,scheme in [(1200,'light'),(1200,'dark'),(390,'light'),(390,'dark')]:
                row={'case':f'{width}-{scheme}','failures':[]};report['cases'].append(row)
                context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce')
                page=context.new_page()
                try:
                    started=time.perf_counter();page.goto(url)
                    expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                    row['ready_ms']=(time.perf_counter()-started)*1000
                    assert not page.evaluate('performance.getEntriesByType("resource").some(e=>e.name.includes("three.module"))')
                    row['decoded_bytes_before_3d']=page.evaluate('performance.getEntriesByType("resource").reduce((n,r)=>n+r.decodedBodySize,0)')
                    if width<721:page.locator('#atlas-open-reader').click()
                    formal=page.locator('#atlas-formal-details');formal.locator(':scope > summary').focus();page.keyboard.press('Enter')
                    expect(formal).to_have_attribute('open','')
                    expect(page.locator('#atlas-formal-connection')).to_be_visible()
                    expect(page.locator('#atlas-formal-evidence > p')).to_contain_text('historical arithmetic companion')
                    bridge=page.locator('#atlas-formal-bridge')
                    expect(bridge).to_be_attached(timeout=1000)
                    bridge.locator(':scope > summary').focus();page.keyboard.press('Enter')
                    expect(bridge).to_have_attribute('open','')
                    expect(bridge).to_contain_text('historical')
                    expect(bridge).to_contain_text(str(identical_count))
                    bridge_link=bridge.get_by_role('link',name='Verified source bridge (JSON)',exact=True)
                    response=context.request.get(bridge_link.get_attribute('href'));assert response.ok
                    original=(site/('evidence/side24-main-3b39/source-bridge-'+profile+'.json')).read_bytes()
                    assert response.body()==original
                    record=json.loads(original);assert len(record['identical_formal_files'])==identical_count
                    assert len(record.get('changed_documentation_files',{}))==documentation_count
                    assert len(record.get('changed_target_annotations',{}))==annotation_count
                    assert set(record['changed_control_files'])=={'tools/formal_gate_check.py','tests/test_formal_gate.py'}
                    assert record['historical_artifact']['commit']=='3b39c42d6edcc98ae165270be346f0e31c57f81d'
                    row['bridge_record_sha256']=digest(original)
                    receipt=formal.get_by_role('link',name='Original execution receipt (JSON)',exact=True)
                    response=context.request.get(receipt.get_attribute('href'));assert response.ok
                    assert digest(response.body())=='1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0'
                    expect(formal).to_contain_text('REJECTED')
                    expect(formal).to_contain_text('PENDING_INDEPENDENT_REVIEW')
                    guide=page.locator('#atlas-formal-guide');guide.locator('summary').focus();page.keyboard.press('Enter')
                    expect(guide).to_have_attribute('open','')
                    expect(guide).to_contain_text('assumptions')
                    expect(formal).to_contain_text('no current execution')
                    page.locator('#atlas-inspector').screenshot(path=str(out/f'inspector-{width}-{scheme}.png'))
                    if width<721:page.locator('#atlas-browse-link').click()
                    page.locator('#atlas-view').select_option('list')
                    share=page.locator('#atlas-share').get_attribute('href')
                    assert mainpin in share and mathpin in share
                    started=time.perf_counter();page.locator('#atlas-space-toggle').click()
                    page.locator('#atlas-space-host').scroll_into_view_if_needed()
                    expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
                    assert page.locator('#atlas-space-host canvas').count()==1
                    row['real_3d_ms']=(time.perf_counter()-started)*1000
                    row['rendered_nodes']=int(page.locator('#atlas-space-host').get_attribute('data-nodes'))
                    assert row['rendered_nodes']>0
                    expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                    assert page.locator('#atlas-formal-bridge').count()==1
                    page.goto(share);page.reload();expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                    alternate={**query,'atlas-node':'math.uniform-matrix-cap-lifetime'}
                    page.goto(origin+'/dependencies.html?'+urlencode(alternate))
                    expect(page.locator('#atlas-inspector h3')).to_have_text(alternate['atlas-node'])
                    assert page.locator('#atlas-formal-connection').count()==0
                    expect(page.locator('#atlas-formal-evidence')).to_contain_text('No verified graph-to-formal-artifact mapping')
                    expect(page.locator('#atlas-formal-guide')).to_contain_text('No mapped declaration')
                    expect(page.locator('#atlas-formal-guide')).not_to_contain_text('Lean checked the named')
                    page.go_back();expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                    assert page.locator('#atlas-formal-bridge').count()==1
                    page.go_forward();expect(page.locator('#atlas-inspector h3')).to_have_text(alternate['atlas-node'])
                    page.goto(origin+'/dependencies.html?'+urlencode({**query,'main-snapshot':'0'*40}))
                    expect(page.locator('#atlas-error')).to_be_visible()
                    assert page.locator('#atlas-formal-connection').count()==0
                    if args.base_url:assert context.request.get(origin+'/current-meta.json').body()==meta
                except Exception as exc:
                    row['failures'].append(str(exc));page.screenshot(path=str(out/f'failure-{width}-{scheme}.png'))
                row['passed']=not row['failures'];context.close()
                (out/'source-bridge-browser.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.close()
    finally:
        server.shutdown();server.server_close()
        (out/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
    print(json.dumps({'passed':sum(r['passed'] for r in report['cases']),'total':len(report['cases'])}))
    if any(not r['passed'] for r in report['cases']):raise SystemExit(1)


if __name__=='__main__':main()
