#!/usr/bin/env python3
"""Check complete real3D labels on an explicitly generated source-bound site."""
import argparse,hashlib,json,platform,subprocess,threading,time,traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlencode
from playwright.sync_api import sync_playwright,expect

CLAIM='math.d5-component.punctured-pin-continuum-check'
GEOMETRY='''() => [...document.querySelectorAll('.atlas-space-labels button')].filter(e=>!e.hidden).map(e=>{const r=e.getBoundingClientRect(),h=e.closest('.atlas-space-host').getBoundingClientRect(),s=getComputedStyle(e);return {id:e.textContent,selected:e.classList.contains('selected'),left:r.left-h.left,top:r.top-h.top,width:r.width,height:r.height,clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,clientHeight:e.clientHeight,scrollHeight:e.scrollHeight,textOverflow:s.textOverflow,whiteSpace:s.whiteSpace};})'''
def check_geometry(page,row):
    host=page.locator('#atlas-space-host').bounding_box();boxes=page.evaluate(GEOMETRY)
    row['geometry'].append(boxes)
    assert any(b['selected'] for b in boxes),'Selected label is missing'
    for b in boxes:
        assert b['scrollWidth']<=b['clientWidth']+1 and b['scrollHeight']<=b['clientHeight']+1,'Identifier is visually truncated: '+b['id']
        assert b['textOverflow']!='ellipsis','Visible identifier uses ellipsis'
        assert b['left']>=2 and b['top']>=2 and b['left']+b['width']<=host['width']-2 and b['top']+b['height']<=host['height']-2,'Label leaves viewport'
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            assert not(a['left']<b['left']+b['width'] and a['left']+a['width']>b['left'] and a['top']<b['top']+b['height'] and a['top']+a['height']>b['top']),'Visible labels overlap'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--site',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--browser',default='/usr/bin/chromium');a=p.parse_args();a.site=a.site.resolve();a.output.mkdir(parents=True,exist_ok=True)
    raw=(a.site/'current.json').read_bytes();data=json.loads(raw);meta=json.loads((a.site/'current-meta.json').read_bytes());assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256'];assert CLAIM in data['graph']['data']['nodes']
    mainpin=next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'));mathpin=data['graph']['identity']['commit']
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(a.site)));threading.Thread(target=server.serve_forever,daemon=True).start();origin=f'http://127.0.0.1:{server.server_port}'
    report={'source_repositories':data['repositories'],'payload_sha256':hashlib.sha256(raw).hexdigest(),'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'tested_modules':{n:hashlib.sha256((a.site/n).read_bytes()).hexdigest() for n in ['atlas-spatial.mjs','atlas-spatial.css','atlas-spatial-labels.mjs'] if(a.site/n).exists()},'environment':{'platform':platform.platform(),'chromium_sandbox':True,'GPU':'Chromium SwiftShader software WebGL','reduced_motion':'reduce'},'cases':[]}
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=a.browser,chromium_sandbox=True,args=['--enable-unsafe-swiftshader','--disable-background-networking']);report['environment']['browser']=browser.version
            for width,height in [(1200,900),(390,844)]:
                for scheme in ['light','dark']:
                    row={'case':f'{width}x{height}-{scheme}','geometry':[],'errors':[]};report['cases'].append(row);context=browser.new_context(viewport={'width':width,'height':height},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);page.on('pageerror',lambda e,r=row:r['errors'].append(str(e)))
                    try:
                        query=urlencode({'atlas-view':'graph','atlas-node':CLAIM,'atlas-q':'.','snapshot':mathpin,'main-snapshot':mainpin});t=time.perf_counter();page.goto(origin+'/dependencies.html?'+query);expect(page.locator('#atlas-inspector h3')).to_have_text(CLAIM);row['load_ms']=(time.perf_counter()-t)*1000
                        expected=page.locator('#atlas-list button').evaluate_all('(es)=>es.map(e=>e.firstChild.textContent)');t=time.perf_counter();page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true');row['activate_ms']=(time.perf_counter()-t)*1000
                        expect(page.locator('#atlas-space-host canvas')).to_have_count(1);actual=page.locator('.atlas-space > details ul').first.locator('button').evaluate_all('(es)=>es.map(e=>e.firstChild.textContent)');assert actual==expected;row['nodes']=len(actual);row['edges']=page.locator('#atlas-space-host').get_attribute('data-edges');row['selected_text']=page.locator('.atlas-space-labels button.selected').inner_text();assert row['selected_text']==CLAIM
                        page.locator('#atlas-space-host').screenshot(path=str(a.output/f'labels-{row["case"]}.png'));check_geometry(page,row)
                        canvas=page.locator('#atlas-space-host canvas');canvas.focus();row['keyboard_to_frame_ms']=[];row['render_cpu_ms']=[]
                        for _ in range(12):
                            frame=page.locator('#atlas-space-host').get_attribute('data-frames');t=time.perf_counter();page.keyboard.press('ArrowRight');page.wait_for_function('(f)=>document.querySelector("#atlas-space-host").dataset.frames!==f',arg=frame);row['keyboard_to_frame_ms'].append((time.perf_counter()-t)*1000);row['render_cpu_ms'].append(float(page.locator('#atlas-space-host').get_attribute('data-render-ms')));check_geometry(page,row)
                        page.locator('#atlas-space-host').screenshot(path=str(a.output/f'labels-orbit-{row["case"]}.png'));page.get_by_role('button',name='Reset camera',exact=True).click();check_geometry(page,row);assert not row['errors'];row['passed']=True
                    except Exception as e:
                        row['passed']=False;row['failure']=str(e);row['traceback']=traceback.format_exc();page.screenshot(path=str(a.output/f'failure-{row["case"]}.png'))
                    finally:context.close();(a.output/'labels-browser.json').write_text(json.dumps(report,indent=2)+'\n')
            browser.close()
    finally:server.shutdown();server.server_close();report['passed']=sum(c.get('passed',False) for c in report['cases']);report['total']=len(report['cases']);(a.output/'labels-browser.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'total':report['total'],'failures':[c.get('failure') for c in report['cases'] if not c.get('passed')]}));return 0 if report['passed']==report['total']==4 else 1
if __name__=='__main__':raise SystemExit(main())
