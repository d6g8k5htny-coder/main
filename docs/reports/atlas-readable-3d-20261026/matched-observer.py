#!/usr/bin/env python3
"""Standalone matched 3D observation; no renderer-state injection or request routes.

Run unchanged against before/after copies with the identical --payload fixture.
All artifacts and Chromium temporary files stay under this driver's scratch root.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, os, platform, statistics
import tempfile, threading, time, traceback, subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlencode
from PIL import Image
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parent
NODE='math.d5-component.punctured-pin-continuum-check'
CODE='c447d1029aebd44ea7bf278f691ec713633551bf'
def identity(p):
    b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()}
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def require(v,message):
    if not v:raise AssertionError(message)

GEOMETRY=r'''() => {
 const host=document.querySelector('#atlas-space-host');
 const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom};};
 const hr=rect(host),vr={x:0,y:0,right:innerWidth,bottom:innerHeight};
 const outside=(a,b)=>a.x<b.x-.5||a.y<b.y-.5||a.right>b.right+.5||a.bottom>b.bottom+.5;
 const labels=[...host.querySelectorAll('.atlas-space-labels button')].map(e=>{
   const r=rect(e),cs=getComputedStyle(e),range=document.createRange();range.selectNodeContents(e);
   const visible=!e.hidden&&cs.display!=='none'&&cs.visibility!=='hidden'&&r.width>0&&r.height>0;
   const overflowX=e.scrollWidth>e.clientWidth+1,overflowY=e.scrollHeight>e.clientHeight+1;
   return {id:e.textContent,selected:e.classList.contains('selected'),visible,hidden:e.hidden,rect:r,text_rects:[...range.getClientRects()].map(t=>({x:t.x,y:t.y,width:t.width,height:t.height,right:t.right,bottom:t.bottom})),clientWidth:e.clientWidth,scrollWidth:e.scrollWidth,clientHeight:e.clientHeight,scrollHeight:e.scrollHeight,css:{font:cs.font,whiteSpace:cs.whiteSpace,textOverflow:cs.textOverflow,overflow:cs.overflow,maxWidth:cs.maxWidth},text_overflow_x:overflowX,text_overflow_y:overflowY,clipped_by_host:visible&&outside(r,hr),clipped_by_viewport:visible&&outside(r,vr),full_text_readable:visible&&!overflowX&&!overflowY&&!outside(r,hr)&&!outside(r,vr)};
 });
 const visible=labels.filter(l=>l.visible),intersections=[];
 for(let i=0;i<visible.length;i++)for(let j=i+1;j<visible.length;j++){
   const a=visible[i].rect,b=visible[j].rect,w=Math.min(a.right,b.right)-Math.max(a.x,b.x),h=Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y);
   if(w>.5&&h>.5)intersections.push({a:visible[i].id,b:visible[j].id,width:w,height:h,area:w*h});
 }
 return {host:hr,canvas:rect(host.querySelector('canvas')),viewport:{width:innerWidth,height:innerHeight,devicePixelRatio},scroll:{x:scrollX,y:scrollY},nodes:Number(host.dataset.nodes),edges:Number(host.dataset.edges),frames:Number(host.dataset.frames),render_cpu_ms:Number(host.dataset.renderMs),dom_label_count:labels.length,visible_label_count:visible.length,hidden_label_count:labels.length-visible.length,truncated_visible_count:visible.filter(l=>l.text_overflow_x||l.text_overflow_y).length,host_clipped_visible_count:visible.filter(l=>l.clipped_by_host).length,viewport_clipped_visible_count:visible.filter(l=>l.clipped_by_viewport).length,fully_readable_visible_count:visible.filter(l=>l.full_text_readable).length,intersection_count:intersections.length,intersections,selected_label:labels.find(l=>l.selected)||null,labels};
}'''
PANEL=r'''() => ({panel:document.querySelector('#current-atlas').getAttribute('data-atlas-panel'),reader_visible:!!document.querySelector('#atlas-reader').getClientRects().length,host_visible:!!document.querySelector('#atlas-space-host').getClientRects().length,selected:document.querySelector('#atlas-inspector h3')?.textContent,url:location.href,scroll:{x:scrollX,y:scrollY}})'''
SIGNATURE=r'''() => ({selected:document.querySelector('#atlas-inspector h3')?.textContent,scope:document.querySelector('#atlas-claim-summary > p')?.textContent,source_links:[...document.querySelectorAll('#atlas-claim-summary a')].map(a=>({text:a.textContent,href:a.href})),inspector_links:[...document.querySelectorAll('#atlas-inspector a')].map(a=>({text:a.textContent,href:a.href})),share:document.querySelector('#atlas-share')?.href,url:location.href})'''
INVENTORY=r'''() => {
 const button=e=>({id:e.firstChild.textContent,accessible_text:e.innerText,pressed:e.getAttribute('aria-pressed'),kind_classification:e.querySelector('small')?.textContent});
 return {controller:[...document.querySelectorAll('#atlas-list button')].map(button),spatial:[...document.querySelectorAll('.atlas-space > details > ul:first-of-type button')].map(button),relationships:[...document.querySelectorAll('.atlas-space > details > ul:nth-of-type(2) > li')].map(e=>e.textContent),disclosure_open:document.querySelector('.atlas-space > details').open};
}'''

def return_browse(page):
    if not page.locator('#atlas-space-host').is_visible():
        page.get_by_role('link',name='Return to atlas browse',exact=True).click()
    page.locator('#atlas-space-host').scroll_into_view_if_needed()
    expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
    page.wait_for_timeout(80)

def capture(page,out,name,row):
    p=out/(name+'.png');page.screenshot(path=str(p));row['captures'][name]={'file':p.name,**identity(p)}
    q=out/(name+'-graph.png');page.locator('#atlas-space-host').screenshot(path=str(q));row['captures'][name+'-graph']={'file':q.name,**identity(q)}

def pick_canvas(page,out,row):
    canvas=page.locator('#atlas-space-host canvas');canvas.scroll_into_view_if_needed();page.wait_for_timeout(70)
    shot=canvas.screenshot();(out/'pick-candidate-canvas.png').write_bytes(shot)
    pixels=Image.open(BytesIO(shot)).convert('RGB');points=set()
    for y in range(pixels.height):
        for x in range(pixels.width):
            r,g,b=pixels.getpixel((x,y))
            if abs(r-114)<3 and abs(g-189)<3 and abs(b-237)<3:points.add((x,y))
    clusters=[]
    while points:
        seed=points.pop();component=[seed]
        for x,y in component:
            for p in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if p in points:points.remove(p);component.append(p)
        xs=[p[0] for p in component];ys=[p[1] for p in component]
        if len(component)>=2 and 1<=max(xs)-min(xs)<20 and 1<=max(ys)-min(ys)<20:
            clusters.append({'x':sum(xs)/len(xs),'y':sum(ys)/len(ys),'pixels':len(component),'width':max(xs)-min(xs),'height':max(ys)-min(ys)})
    result={'method':'Observed blue-sphere pixel components; elementFromPoint must equal the actual canvas before any mouse click. Inspector must change. No renderer internals used.','candidate_count':len(clusters),'attempts':[],'verified_canvas_pick':False}
    box=canvas.bounding_box()
    for p in sorted(clusters,key=lambda p:(-p['y'],p['x']))[:24]:
        x=box['x']+(p['x']+.5)*box['width']/pixels.width;y=box['y']+(p['y']+.5)*box['height']/pixels.height
        hit=page.evaluate('''({x,y})=>{const e=document.elementFromPoint(x,y);return {canvas:e===document.querySelector('#atlas-space-host canvas'),tag:e?.tagName,id:e?.id,class:e?.className,text:e?.textContent?.slice(0,150)}}''',{'x':x,'y':y})
        attempt={'pixel_component':p,'client_point':{'x':x,'y':y},'hit_test':hit};result['attempts'].append(attempt)
        if not hit['canvas']:continue
        before=page.locator('#atlas-inspector h3').inner_text();page.mouse.click(x,y);page.wait_for_timeout(120)
        after=page.locator('#atlas-inspector h3').inner_text();attempt.update(before=before,after=after)
        if after!=before:
            result.update(verified_canvas_pick=True,selected=after,signature=page.evaluate(SIGNATURE));page.screenshot(path=str(out/'verified-pick-reader.png'));break
    if not result['verified_canvas_pick']:result['limit']='No verified sphere pick in the bounded observed candidates; no GPU-pick success claimed.'
    row['canvas_pick']=result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--site',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--payload',type=Path,required=True);ap.add_argument('--browser',default='/usr/bin/chromium');ap.add_argument('--code-label',default=CODE);args=ap.parse_args()
    args.site=args.site.resolve();args.output=args.output.resolve();args.payload=args.payload.resolve()
    require(args.output.is_relative_to(ROOT),'Output must remain in the authorized scratch subtree');args.output.mkdir(parents=True,exist_ok=True)
    tmp=ROOT/'runtime';tmp.mkdir(exist_ok=True);os.environ['TMPDIR']=str(tmp);tempfile.tempdir=str(tmp)
    raw=args.payload.read_bytes();data=json.loads(raw);meta=json.loads((args.site/'current-meta.json').read_bytes())
    require((args.site/'current.json').read_bytes()==raw,'Served payload differs from frozen fixture');require(meta['sha256']==hashlib.sha256(raw).hexdigest() and meta['bytes']==len(raw),'Served metadata differs from frozen fixture')
    require(NODE in data['graph']['data']['nodes'],'Long node absent from actual payload')
    math=data['graph']['identity']['commit'];mainpin=next(r['commit'] for r in data['repositories'] if r['repository'].endswith('/main'))
    modules={str(p.relative_to(args.site)):identity(p) for p in sorted(args.site.rglob('*')) if p.is_file() and (p.suffix in ['.mjs','.js','.css'] or p.name=='dependencies.html')}
    report={'role':'finite implementation-side baseline observer','provider':'OpenAI','agent':'Codex','family':'GPT-6 as declared in developer instructions; runtime identifier unavailable','parent_author':'OpenAI/Codex','organizational_independence_credit':0,'review_credit':False,'scientific_effect':'NONE','CODE_source_label':args.code_label,'observer_checkout_is_not_served_source':True,'served_site':str(args.site),'fixture':identity(args.payload),'payload_repositories':data['repositories'],'graph_identity':data['graph']['identity'],'dataset':{'nodes':len(data['graph']['data']['nodes']),'edges':len(data['graph']['data']['edges'])},'selected_node':NODE,'node_record':data['graph']['data']['nodes'][NODE],'driver':identity(Path(__file__)),'served_modules':modules,'environment':{'platform':platform.platform(),'chromium_path':args.browser,'chromium_executable':identity(Path(args.browser)),'chromium_command_version':subprocess.check_output([args.browser,'--version'],text=True).strip(),'playwright':importlib.metadata.version('playwright'),'chromium_sandbox':True,'args':['--enable-unsafe-swiftshader','--disable-background-networking'],'headless':True,'device_scale_factor':1,'reduced_motion':'reduce','gpu':'SwiftShader software renderer; no physical mobile/GPU qualification','CSP':'unmodified','security_bypass':False,'routes':False,'application_state_injection':False,'transport':'loopback static serving; normal inherited proxy/CA retained'},'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'cases':[]}
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self,fmt,*a):requests.append({'path':self.path,'message':fmt%a})
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(args.site)));threading.Thread(target=server.serve_forever,daemon=True).start();origin=f'http://127.0.0.1:{server.server_port}'
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(executable_path=args.browser,headless=True,chromium_sandbox=True,args=report['environment']['args']);report['environment']['browser_version']=browser.version
            for width,height in [(1200,900),(390,844)]:
                for scheme in ['light','dark']:
                    name=f'{width}x{height}-{scheme}';out=args.output/name;out.mkdir(exist_ok=True)
                    row={'case':name,'viewport':{'width':width,'height':height},'scheme':scheme,'captures':{},'errors':[],'pageerrors':[],'console_errors':[],'failed_requests':[],'http_errors':[],'geometry':[]};report['cases'].append(row)
                    context=browser.new_context(viewport=row['viewport'],color_scheme=scheme,reduced_motion='reduce',device_scale_factor=1);page=context.new_page();page.set_default_timeout(12000)
                    page.on('pageerror',lambda e,r=row:r['pageerrors'].append(str(e)))
                    page.on('console',lambda m,r=row:r['console_errors'].append(m.text) if m.type=='error' else None)
                    page.on('requestfailed',lambda q,r=row:r['failed_requests'].append({'url':q.url,'failure':q.failure}))
                    page.on('response',lambda q,r=row:r['http_errors'].append({'url':q.url,'status':q.status}) if q.status>=400 else None)
                    try:
                        t=time.perf_counter();page.goto(origin+'/dependencies.html?atlas-view=graph#current-atlas');page.wait_for_function("document.querySelector('#atlas-status').textContent.startsWith('Source snapshot')");row['load_to_admitted_graph_ms']=(time.perf_counter()-t)*1000
                        row['initial_panel']=page.evaluate(PANEL);row['initial_status']=page.locator('#atlas-status').inner_text()
                        row['resources_before_activation']=page.evaluate("performance.getEntriesByType('resource').map(e=>({name:e.name,bytes:e.decodedBodySize,ms:e.duration}))")
                        # '.' is an actual existing search control; it exposes all actual IDs.
                        page.locator('#atlas-search').fill('.')
                        page.wait_for_timeout(100)
                        row['controller_before_3d']=page.locator('#atlas-list button').evaluate_all('(es)=>es.map(e=>e.firstChild.textContent)')
                        t=time.perf_counter();page.get_by_role('button',name='Explore in 3D',exact=True).click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true');row['activation_to_ready_ms']=(time.perf_counter()-t)*1000
                        require(page.locator('#atlas-space-host canvas').count()==1,'Actual canvas missing')
                        row['inventory_initial']=page.evaluate(INVENTORY)
                        # Select the long ID through its real accessible-list button.
                        long=page.locator('.atlas-space > details > ul').first.locator('button').filter(has_text=NODE)
                        require(long.count()==1,'Long ID list button not unique');long.click();expect(page.locator('#atlas-inspector h3')).to_have_text(NODE)
                        row['selected_reader_panel']=page.evaluate(PANEL);row['selected_signature']=page.evaluate(SIGNATURE)
                        page.locator('#atlas-reader').scroll_into_view_if_needed();page.screenshot(path=str(out/'selected-reader.png'))
                        # Reload first, then inspect actual phone browse/reader mode explicitly.
                        page.reload();page.wait_for_function("document.querySelector('#atlas-status').textContent.startsWith('Source snapshot')");expect(page.locator('#atlas-inspector h3')).to_have_text(NODE);row['reload_panel_before_action']=page.evaluate(PANEL)
                        if page.locator('#current-atlas').get_attribute('data-atlas-panel')=='browse':page.get_by_role('button',name='Read selected node',exact=True).click()
                        row['reload_reader_checked']=page.evaluate(PANEL);require(row['reload_reader_checked']['reader_visible'],'Reader unavailable after reload and visible control')
                        row['reloaded_signature']=page.evaluate(SIGNATURE)
                        page.screenshot(path=str(out/'reload-reader.png'))
                        return_browse(page);row['reload_browse_checked']=page.evaluate(PANEL);require(row['reload_browse_checked']['host_visible'],'3D host hidden after return-to-browse control')
                        require(int(page.locator('#atlas-space-host').get_attribute('data-nodes'))==len(data['graph']['data']['nodes']),'3D is not full actual graph')
                        page.get_by_role('button',name='Reset camera',exact=True).click();page.locator('#atlas-space-host').scroll_into_view_if_needed();page.wait_for_timeout(100)
                        row['inventory_selected']=page.evaluate(INVENTORY)
                        require([x['id'] for x in row['inventory_selected']['spatial']]==row['controller_before_3d'],'Accessible list differs from controller inventory')
                        canvas=page.locator('#atlas-space-host canvas');canvas.focus();page.wait_for_timeout(60)
                        row['geometry'].append({'stage':'initial',**page.evaluate(GEOMETRY)});capture(page,out,'initial',row)
                        row['orbit_metrics']=[]
                        for i in range(12):
                            before=page.locator('#atlas-space-host').get_attribute('data-frames');t=time.perf_counter();page.keyboard.press('ArrowRight');page.wait_for_function('(f)=>document.querySelector("#atlas-space-host").dataset.frames!==f',arg=before)
                            elapsed=(time.perf_counter()-t)*1000;g=page.evaluate(GEOMETRY);row['geometry'].append({'stage':f'orbit-{i+1}',**g});row['orbit_metrics'].append({'key':'ArrowRight','ordinal':i+1,'before_frame':int(before),'observed_frame':g['frames'],'key_to_observed_frame_ms':elapsed,'render_cpu_ms':g['render_cpu_ms']})
                        capture(page,out,'after-12-orbits',row)
                        f=page.locator('#atlas-space-host').get_attribute('data-frames');page.wait_for_timeout(250);row['idle_frames']=[int(f),int(page.locator('#atlas-space-host').get_attribute('data-frames'))]
                        pick_canvas(page,out,row)
                        return_browse(page)
                        # Restore long selection through accessible list after any pick.
                        page.locator('.atlas-space > details > ul').first.locator('button').filter(has_text=NODE).click();expect(page.locator('#atlas-inspector h3')).to_have_text(NODE);return_browse(page)
                        before_focus=page.url;page.get_by_role('button',name='Focus dependencies',exact=True).click();page.wait_for_timeout(80);focused=page.url;row['focus_history']={'before':before_focus,'focused':focused,'focused_nodes':page.locator('#atlas-space-host').get_attribute('data-nodes')}
                        page.go_back();page.wait_for_timeout(100);row['focus_history']['back']=page.url;page.go_forward();page.wait_for_timeout(100);row['focus_history']['forward']=page.url
                        row['final_signature']=page.evaluate(SIGNATURE)
                        row['performance']=page.evaluate("({navigation:performance.getEntriesByType('navigation').map(e=>e.toJSON()),resources:performance.getEntriesByType('resource').map(e=>({name:e.name,bytes:e.decodedBodySize,ms:e.duration}))})")
                        row['document_no_horizontal_overflow']=page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                        row['journey_completed']=True
                    except Exception as e:
                        row['errors'].append(str(e));row['traceback']=traceback.format_exc();row['journey_completed']=False
                        try:page.screenshot(path=str(out/'failure.png'));row['failure_panel']=page.evaluate(PANEL)
                        except Exception:pass
                    finally:
                        write(out/'case.json',row);context.close();write(args.output/'baseline-report.json',report)
            browser.close()
    finally:
        server.shutdown();server.server_close();write(args.output/'server-requests.json',requests)
        report['finished_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());report['completed_journeys']=sum(c.get('journey_completed',False) for c in report['cases']);report['readability_is_not_a_pass_gate']=True
        report['limits']=['Implementation observation only, no review/scientific/human/AT qualification.','One frozen 49-node fixture if reported as 49; no larger-dataset extrapolation.','Single run per case: no statistical speed claim. Key-to-observed-frame includes Playwright/driver overhead; render CPU is page-reported renderer draw duration, not GPU completion.','DOM geometry/overflow and raw screenshots measure clipping and overlaps; full_text_readable does not certify human legibility or AT.','Local c447 code rendered offline with actual frozen public payload. Not a live deployed-source replay. Payload Math tip records reported process-breach merge without endorsement.']
        write(args.output/'baseline-report.json',report)
    print(json.dumps({'completed_journeys':report['completed_journeys'],'cases':[{'case':c['case'],'completed':c.get('journey_completed'),'errors':c['errors'],'verified_canvas_pick':c.get('canvas_pick',{}).get('verified_canvas_pick')} for c in report['cases']]}))
    return 0 if report['completed_journeys']==4 else 1
if __name__=='__main__':raise SystemExit(main())
