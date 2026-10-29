#!/usr/bin/env python3
"""Optional-status resilience and actual source-derived 3D camera regressions."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,threading,time,traceback
from pathlib import Path
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]
def require(value,message):
 if not value:raise AssertionError(message)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--site',type=Path,default=ROOT/'docs/site');ap.add_argument('--origin',help='Served site directory, independently source-qualified before replay.');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--browser',default='/usr/bin/chromium');ap.add_argument('--overlay-source',action='store_true',help='Use current Atlas modules over the supplied admitted observation.');ap.add_argument('--observe',action='store_true');ap.add_argument('--only',help='Run one exact named regression.');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 require(not (a.origin and a.overlay_source),'A remote source cannot be replaced by a local overlay')
 class Handler(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
  def translate_path(self,path):
   name=urlsplit(path).path.rsplit('/',1)[-1]
   if a.overlay_source and name in ('atlas.mjs','atlas-spatial.mjs'):return str(ROOT/'docs/site'/name)
   return super().translate_path(path)
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(a.site.parent)));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();origin=f'http://127.0.0.1:{server.server_port}';url=origin+'/'+a.site.name+'/dependencies.html'
 if a.origin:
  parsed=urlsplit(a.origin);require(parsed.scheme in ('http','https') and parsed.netloc and not parsed.username and not parsed.password,'Expected an ordinary HTTP(S) site directory');origin=parsed.scheme+'://'+parsed.netloc;url=a.origin.rstrip('/')+'/dependencies.html'
 report={'local_checkout_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'local_source_files':{n:hashlib.sha256((ROOT/'docs/site'/n).read_bytes()).hexdigest() for n in ('atlas.mjs','atlas-spatial.mjs')},'overlay_source':a.overlay_source,'local_payload_sha256':hashlib.sha256((a.site/'current.json').read_bytes()).hexdigest(),'mode':'served-origin' if a.origin else 'local-site','served_page':url,'source_qualification':'Local checkout and fixture metadata do not identify remote served bytes; separate source/public byte evidence is required.','chromium_sandbox':True,'cases':[],'page_errors':[],'scientific_effect':'NONE'}
 statuses=json.loads((a.site/'node-status.json').read_text())
 def snapshot(page):return page.evaluate(r'''async()=>{const u=performance.getEntriesByType('resource').find(x=>/\/current\.mjs\?/.test(x.name)).name;return (await import(u)).currentSnapshot.then(d=>d?{nodes:Object.keys(d.graph.data.nodes).length}:null)}''')
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path=a.browser,headless=True,chromium_sandbox=True,args=['--enable-unsafe-swiftshader','--disable-background-networking']);report['browser']=browser.version
   for kind in ['held-valid','held-bookmark-refusal','held-source-failure','late-current-selection','late-focused-resume','late-after-refusal','pagehide-late-status','pageshow-retry','late-stale-status','optional-feed-failure','optional-feed-malformed','actual-camera-scope']:
    if a.only and a.only!=kind:continue
    row={'case':kind};report['cases'].append(row);context=browser.new_context(viewport={'width':1440,'height':900},reduced_motion='reduce');page=context.new_page();page.set_default_timeout(5000);held=[]
    page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
    def route(request):
     path=urlsplit(request.request.url).path
     if path.endswith('/node-status.json'):
      if kind=='optional-feed-failure':request.fulfill(status=503,body='unavailable')
      elif kind=='optional-feed-malformed':request.fulfill(status=200,content_type='application/json',body='{broken-json')
      elif kind=='actual-camera-scope':request.continue_()
      else:held.append(request)
     elif kind=='held-source-failure' and path.endswith('/current.json'):request.abort()
     elif request.request.url.startswith(origin+'/'):request.continue_()
     else:request.abort()
    context.route('**/*',route)
    try:
     page.goto(url+('?snapshot='+'0'*40 if kind=='held-bookmark-refusal' else ''),wait_until='domcontentloaded');row['primary_snapshot']=snapshot(page)
     if kind=='held-source-failure':
      page.wait_for_timeout(100);require('Unavailable:' in page.locator('#atlas-status').inner_text(),'Source failure waited for optional status');require(page.locator('#atlas-search').is_disabled(),'Failed source enabled search');require(page.locator('#atlas-list button').count()==0,'Failed source exposed nodes')
     elif kind=='held-bookmark-refusal':
      page.wait_for_timeout(100);require(page.locator('#atlas-error').is_visible(),'Bookmark refusal waited for optional status');require(page.locator('#atlas-list button').count()==0,'Refused bookmark substituted nodes');require(page.locator('#atlas-formal-evidence').count()==0,'Refused bookmark exposed formal evidence')
     else:
      page.wait_for_timeout(100);require(not page.locator('#atlas-search').is_disabled(),'Primary Atlas initialization waited for optional node status');expect(page.locator('#atlas-status')).to_contain_text('Source snapshot')
      if kind=='held-valid':require(page.locator('#atlas-list button').count()>0,'No primary graph without optional status')
      elif kind in ['optional-feed-failure','optional-feed-malformed']:
       page.locator('#atlas-review-details').evaluate('(e)=>e.open=true')
       require('not recorded in this snapshot' in page.locator('#atlas-resume').inner_text().lower(),'Failed optional feed lost the snapshot-only projection')
      elif kind in ['late-current-selection','late-focused-resume','late-after-refusal','pagehide-late-status','pageshow-retry','late-stale-status']:
       require(len(held)==1,'Expected one held optional request')
       initial=page.locator('#atlas-inspector h3').inner_text();page.locator('#atlas-review-details').evaluate('(e)=>e.open=true');page.locator('#atlas-node-details').evaluate('(e)=>e.open=true')
       if kind=='late-current-selection':
        page.locator('#atlas-view').select_option('list');target=page.locator('#atlas-list button').nth(1);target_id=target.evaluate('(e)=>e.firstChild.textContent');target.click();expect(page.locator('#atlas-inspector h3')).to_have_text(target_id);page.locator('#atlas-review-details').evaluate('(e)=>e.open=true');page.locator('#atlas-node-details').evaluate('(e)=>e.open=true')
       else:target_id=initial
       fixture=json.loads(json.dumps(statuses));fixture[target_id]={**next(iter(fixture.values())),'node_id':target_id,'next_action':'LATE_OPERATIONAL_STATUS_TEST_MARKER'}
       if kind=='late-stale-status':fixture[target_id].update(status='fresh',observed_at='2020-01-01T00:00:00Z')
       page.locator('#atlas-resume').evaluate('(e)=>{const a=document.createElement("a");a.href="#";a.id="resilience-focus-probe";a.textContent="Focused operational evidence";e.append(a)}')
       focus=page.locator('#resilience-focus-probe') if kind=='late-focused-resume' else page.locator('#atlas-search');focus.focus();before_url=page.url;before_history=page.evaluate('history.length');page.evaluate('window.__resilienceInspector=document.querySelector("#atlas-inspector");window.__resilienceHeading=document.querySelector("#atlas-inspector h3");window.__resilienceResume=document.querySelector("#atlas-resume")')
       if kind=='late-after-refusal':
        page.evaluate('''()=>{const u=new URL(location);u.searchParams.set('snapshot','0'.repeat(40));history.pushState(null,'',u);dispatchEvent(new PopStateEvent('popstate'))}''');expect(page.locator('#atlas-error')).to_be_visible()
       elif kind in ['pagehide-late-status','pageshow-retry']:page.evaluate("dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}))")
       try:held.pop().fulfill(status=200,content_type='application/json',body=json.dumps(fixture))
       except Exception:
        require(kind in ['pagehide-late-status','pageshow-retry'],'Unexpected failed route fulfillment')
       page.wait_for_timeout(120)
       if kind=='late-after-refusal':require(page.locator('#atlas-resume').count()==0 and page.locator('#atlas-formal-evidence').count()==0,'Late status revived refused evidence')
       elif kind=='pagehide-late-status':require('LATE_OPERATIONAL_STATUS_TEST_MARKER' not in page.locator('#atlas-resume').inner_text(),'Late status changed a hidden/disposed page')
       elif kind=='pageshow-retry':
        require('LATE_OPERATIONAL_STATUS_TEST_MARKER' not in page.locator('#atlas-resume').inner_text(),'Late status changed a hidden page');page.evaluate("dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true}))");page.wait_for_timeout(100);require(len(held)==1,'Restored page did not make one fresh optional request');held.pop().fulfill(status=200,content_type='application/json',body=json.dumps(fixture));expect(page.locator('#atlas-resume')).to_contain_text('LATE_OPERATIONAL_STATUS_TEST_MARKER')
       else:
        require(page.evaluate('window.__resilienceInspector===document.querySelector("#atlas-inspector")&&window.__resilienceHeading===document.querySelector("#atlas-inspector h3")'),'Late status rebuilt the inspector');require(page.url==before_url and page.evaluate('history.length')==before_history,'Late status wrote navigation');require(page.locator('#atlas-node-details').evaluate('(e)=>e.open'),'Late status closed disclosure');expect(focus).to_be_focused();expect(page.locator('#atlas-inspector h3')).to_have_text(target_id)
        if kind=='late-focused-resume':
         require(page.evaluate('window.__resilienceResume===document.querySelector("#atlas-resume")'),'Late status replaced a focused resume link');require('LATE_OPERATIONAL_STATUS_TEST_MARKER' not in page.locator('#atlas-resume').inner_text(),'Focused operational block changed early');page.locator('#atlas-search').focus()
        expect(page.locator('#atlas-resume')).to_contain_text('LATE_OPERATIONAL_STATUS_TEST_MARKER')
        if kind=='late-stale-status':expect(page.locator('#atlas-resume')).to_contain_text('Node-status view stale')
      elif kind=='actual-camera-scope':
       # Observational original-calling camera hook; not a timing benchmark.
       page.evaluate("""async()=>{const T=await import('./vendor/three-0.160.0/three.module.js?site-release=f2988301e38ea3d44631f3220044accc019d63ae79b737e79b7b8ab7a2d5f778');window.__auditThree=T;const old=T.PerspectiveCamera.prototype.updateProjectionMatrix;T.PerspectiveCamera.prototype.updateProjectionMatrix=function(...args){window.__auditCamera=this;return old.apply(this,args)}}""")
       page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
       def camera():return page.evaluate(r'''async()=>{await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));const rs=performance.getEntriesByType('resource');const m=await import(rs.find(x=>/\/atlas-spatial-model\.mjs\?/.test(x.name)).name),d=await (await import(rs.find(x=>/\/current\.mjs\?/.test(x.name)).name)).currentSnapshot;const ids=[...document.querySelectorAll('#atlas-list button')].map(e=>e.firstChild.textContent),id=document.querySelector('#atlas-inspector h3').textContent,focus=document.querySelector('.atlas-space-toolbar button[aria-pressed]:not(#atlas-space-toggle)')?.getAttribute('aria-pressed')==='true';const v=m.spatialProjection(d.graph.data,ids,id,focus),T=window.__auditThree,c=window.__auditCamera;return{nodes:v.ids.length,inside:v.ids.filter(id=>{const p=new T.Vector3(...v.positions[id]).project(c);return p.x>=-1&&p.x<=1&&p.y>=-1&&p.y<=1&&p.z>=-1&&p.z<=1}).length,position:c.position.toArray(),quaternion:c.quaternion.toArray(),frames:document.querySelector('#atlas-space-host').dataset.frames}}''')
       row['initial']=camera();page.locator('#atlas-search').fill('.');page.locator('#atlas-space-host').scroll_into_view_if_needed();page.wait_for_function("document.querySelector('#atlas-space-host').dataset.nodes==='49'");row['expanded']=camera();require(row['expanded']['inside']==49,'Displayed-set expansion did not fit all49 source node centres')
       canvas=page.locator('#atlas-space-host canvas');canvas.focus();page.keyboard.press('ArrowRight');page.keyboard.press('+');page.wait_for_timeout(70);pose=camera();target=page.locator('.atlas-space > details ul').first.locator('button').nth(2);target.click();page.locator('#atlas-space-host').scroll_into_view_if_needed();page.wait_for_timeout(80);same=camera();require(same['position']==pose['position'] and same['quaternion']==pose['quaternion'],'Same-set selection reset user camera')
       page.locator('#atlas-list').evaluate('(e)=>e.replaceChildren(...[...e.children].reverse())');page.wait_for_timeout(80);reordered=camera();require(reordered['position']==same['position'] and reordered['quaternion']==same['quaternion'],'Reordered same set reset user camera')
       page.get_by_role('button',name='Focus dependencies',exact=True).click();page.locator('#atlas-space-host').scroll_into_view_if_needed();page.wait_for_timeout(80);focused=camera();require(focused['inside']==focused['nodes'],'Focused set did not fit')
       page.get_by_role('button',name='Focus dependencies',exact=True).click();page.locator('#atlas-search').fill('ZZZ_NO_SOURCE_MATCH');page.wait_for_function("document.querySelector('#atlas-space-host').dataset.nodes==='0'");page.locator('#atlas-search').fill('.');page.locator('#atlas-space-host').scroll_into_view_if_needed();page.wait_for_function("document.querySelector('#atlas-space-host').dataset.nodes==='49'");restored=camera();require(restored['inside']==49,'Restored nonempty set did not fit');row['focused']=focused;row['restored']=restored;frames=restored['frames'];page.wait_for_timeout(160);require(camera()['frames']==frames,'Camera changes introduced an idle render loop');page.locator('#atlas-space-host').screenshot(path=str(a.output/'camera49.png'))
     row['passed']=True
    except Exception as error:
     row['passed']=False;row['error']=str(error);row['traceback']=traceback.format_exc();page.screenshot(path=str(a.output/(kind+'-failure.png')))
    finally:
     for request in held:
      try:request.abort()
      except Exception:pass
     context.close()
   browser.close()
 finally:
  server.shutdown();server.server_close();thread.join(5);report['owned_server_closed']=not thread.is_alive();report['passed']=sum(c['passed'] for c in report['cases']);report['failed']=len(report['cases'])-report['passed'];(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['passed','failed','page_errors','owned_server_closed']}));return 1 if (report['failed'] or report['page_errors']) and not a.observe else 0
if __name__=='__main__':raise SystemExit(main())
