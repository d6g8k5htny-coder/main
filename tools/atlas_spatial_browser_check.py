#!/usr/bin/env python3
"""Real optional-3D browser acceptance, including successful GPU node picking."""
from __future__ import annotations
import argparse,json,time,threading,platform,traceback,hashlib,subprocess
from io import BytesIO
from PIL import Image
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
def require(value,message):
 if not value:raise AssertionError(message)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--browser',default='/usr/bin/chromium');parser.add_argument('--site',type=Path,default=ROOT/'docs/site');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
 class Handler(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(args.site.parent)));threading.Thread(target=server.serve_forever,daemon=True).start();origin=f'http://127.0.0.1:{server.server_port}/{args.site.name}/dependencies.html'
 report={'environment':{'platform':platform.platform(),'browser_executable':args.browser,'GPU':'Chromium SwiftShader software WebGL; not physical mobile GPU','dataset_bytes':(args.site/'current.json').stat().st_size},'cases':[]}
 payload=(args.site/'current.json').read_bytes();report['source_payload_sha256']=hashlib.sha256(payload).hexdigest();report['source_repositories']=json.loads(payload)['repositories'];report['checked_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 try:
  with sync_playwright() as p:
   browser=p.chromium.launch(executable_path=args.browser,headless=True,args=['--enable-unsafe-swiftshader','--disable-background-networking']);report['environment']['browser']=browser.version
   for width,height,scheme in [(1200,900,'dark'),(390,844,'light')]:
    row={'case':'real-render-pick-history-keyboard','viewport':[width,height],'scheme':scheme,'errors':[]};report['cases'].append(row)
    context=browser.new_context(viewport={'width':width,'height':height},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.on('pageerror',lambda e:row['errors'].append(str(e)));page.set_default_timeout(15000)
    t=time.perf_counter();page.goto(origin+'?atlas-view=graph');page.wait_for_function("document.querySelector('#atlas-status').textContent.startsWith('Source snapshot')");row['load_ms']=(time.perf_counter()-t)*1000
    require(not any('three' in x['name'] for x in page.evaluate('performance.getEntriesByType("resource").map(x=>({name:x.name}))')),'3D dependency eagerly loaded')
    t=time.perf_counter();page.locator('#atlas-search').fill('lifetime');page.wait_for_timeout(100);row['search_with_100ms_settle_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true');row['activate_ms']=(time.perf_counter()-t)*1000;row['resources']=page.evaluate('performance.getEntriesByType("resource").filter(x=>/atlas-spatial|three.module/.test(x.name)).map(x=>({name:x.name.split("/").pop(),bytes:x.decodedBodySize,ms:x.duration}))')
    expect(page.locator('#atlas-space-status')).to_contain_text('3D ·');require(page.locator('#atlas-space-host canvas').count()==1,'Renderer missing')
    row['nodes']=int(page.locator('#atlas-space-host').get_attribute('data-nodes'));row['edges']=int(page.locator('#atlas-space-host').get_attribute('data-edges'));require(0<row['nodes']<=150,'Unbounded node view')
    page.locator('#atlas-space-host').screenshot(path=str(out/f'after-3d-{width}.png'));page.locator('#current-atlas').screenshot(path=str(out/f'after-{width}.png'))
    # Pick a real sphere through the WebGL canvas, below its DOM label; this is
    # neither a synthetic onPick nor a Three.js-blocked fallback test.
    canvas=page.locator('#atlas-space-host canvas');before=page.locator('#atlas-inspector h3').inner_text();pixels=Image.open(BytesIO(canvas.screenshot())).convert('RGB');points=set()
    for y in range(pixels.height):
     for x in range(pixels.width):
      r,g,b=pixels.getpixel((x,y))
      if abs(r-114)<3 and abs(g-189)<3 and abs(b-237)<3:points.add((x,y))
    clusters=[]
    while points:
     seed=points.pop();component=[seed]
     for x,y in component:
      for neighbor in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
       if neighbor in points:points.remove(neighbor);component.append(neighbor)
     xs=[x for x,y in component];ys=[y for x,y in component]
     if len(component)>8 and 3<max(xs)-min(xs)<20 and 3<max(ys)-min(ys)<20:clusters.append((sum(xs)/len(xs),sum(ys)/len(ys)))
    require(clusters,'No visible unselected sphere pixels');box=canvas.bounding_box();x,y=sorted(clusters,key=lambda point:point[1])[-1];page.mouse.click(box['x']+x,box['y']+y);expect(page.locator('#atlas-inspector h3')).not_to_have_text(before);row['picked']=page.locator('#atlas-inspector h3').inner_text()
    expect(page.locator('#atlas-formal-evidence')).to_contain_text('No mapped trusted kernel-execution receipt');row['source']=page.locator('#atlas-inspector a',has_text='Immutable source reference').get_attribute('href') if page.locator('#atlas-inspector a',has_text='Immutable source reference').count() else None
    deps=page.locator('#atlas-inspector ul button');row['relationships']=deps.count()
    if deps.count():deps.first.click()
    share=page.locator('#atlas-share').get_attribute('href');require('atlas-space=3d' in share,'Share lost 3D');row['share']=share;selected=page.locator('#atlas-inspector h3').inner_text()
    page.reload();expect(page.locator('#atlas-inspector h3')).to_have_text(selected)
    if width<=720:page.locator('#atlas-browse-link').click()
    page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
    # Keyboard camera must draw, then stop: reduced motion has no idle frames.
    canvas=page.locator('#atlas-space-host canvas');canvas.focus();before=int(page.locator('#atlas-space-host').get_attribute('data-frames'));page.keyboard.press('ArrowLeft');page.wait_for_timeout(100);after=int(page.locator('#atlas-space-host').get_attribute('data-frames'));require(after>before,'Keyboard orbit did not render');page.wait_for_timeout(200);require(page.locator('#atlas-space-host').get_attribute('data-frames')==str(after),'Idle render loop');row['keyboard_frames']=[before,after]
    page.get_by_role('button',name='Focus dependencies',exact=True).click();require('atlas-space-focus=dependencies' in page.url,'Focus not shared');focus_url=page.url;page.go_back();require('atlas-space-focus' not in page.url,'Back did not restore focus');page.go_forward();require(page.url==focus_url,'Forward did not restore focus')
    page.get_by_role('button',name='Reset camera',exact=True).click();page.get_by_role('button',name='Zoom in',exact=True).click();page.get_by_role('button',name='Zoom out',exact=True).click()
    require(page.evaluate('document.documentElement.scrollWidth<=innerWidth'),'Narrow document overflow')
    # Real context loss must release the canvas and retain useful fallback.
    canvas.evaluate("e=>e.getContext('webgl2').getExtension('WEBGL_lose_context').loseContext()");expect(page.locator('#atlas-space-status')).to_contain_text('context lost');expect(page.locator('#atlas-stage')).to_be_visible();require(page.locator('#atlas-space-host canvas').count()==0,'Lost renderer retained');page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true');page.locator('#atlas-space-toggle').click();require(page.locator('#atlas-space-host canvas').count()==0,'View exit leaked canvas');row['context_loss']='PASS'
    require(not row['errors'],f"Page errors: {row['errors']}");row['passed']=True;context.close()
   for kind in ['rapid-exit','exact-filters','drag-and-hidden','keyboard-selection','full-dataset-performance']:
    row={'case':kind};report['cases'].append(row);context=browser.new_context(viewport={'width':1200,'height':900},reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000)
    query='?atlas-view=graph&atlas-classification=OPEN_ACTIVE' if kind=='exact-filters' else '?atlas-view=graph'
    page.goto(origin+query);page.wait_for_function("document.querySelector('#atlas-status').textContent.startsWith('Source snapshot')")
    if kind=='rapid-exit':
     page.route('**/atlas-spatial.mjs*',lambda route:(time.sleep(.3),route.continue_()))
     page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-toggle').click();page.wait_for_timeout(500);require(page.locator('#atlas-space-host canvas').count()==0,'Stale async load resurrected renderer')
    else:

     if kind=='full-dataset-performance':page.locator('#atlas-search').fill('.')
     expected=page.locator('#atlas-list button').evaluate_all('(nodes)=>nodes.map(n=>n.firstChild.textContent)');page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true')
     actual=page.locator('.atlas-space > details ul').first.locator('button').all_text_contents();require(actual==expected,'3D page differs from controller filters');row['nodes']=actual
     if kind=='full-dataset-performance':
      canvas=page.locator('#atlas-space-host canvas');canvas.focus();draws=[];interactions=[]
      for _ in range(12):
       frame=page.locator('#atlas-space-host').get_attribute('data-frames');t=time.perf_counter();page.keyboard.press('ArrowRight');page.wait_for_function('(frame)=>document.querySelector("#atlas-space-host").dataset.frames!==frame',arg=frame);interactions.append((time.perf_counter()-t)*1000);draws.append(float(page.locator('#atlas-space-host').get_attribute('data-render-ms')))
      row['draw_cpu_ms']=draws;row['keyboard_to_observed_frame_ms']=interactions;row['relationships']=int(page.locator('#atlas-space-host').get_attribute('data-edges'));canvas.screenshot(path=str(out/'full-dataset.png'))
     if kind=='drag-and-hidden':
      canvas=page.locator('#atlas-space-host canvas');box=canvas.bounding_box();before=page.locator('#atlas-inspector h3').inner_text();page.mouse.move(box['x']+100,box['y']+100);page.mouse.down();page.mouse.move(box['x']+160,box['y']+130,steps=5);page.mouse.up();require(page.locator('#atlas-inspector h3').inner_text()==before,'Drag picked a node')
      page.locator('#atlas-reader').scroll_into_view_if_needed();page.evaluate('scrollTo(0,document.body.scrollHeight)');page.wait_for_timeout(100);frames=page.locator('#atlas-space-host').get_attribute('data-frames');page.wait_for_timeout(200);require(page.locator('#atlas-space-host').get_attribute('data-frames')==frames,'Offscreen idle work');row['paused_frames']=frames
     if kind=='keyboard-selection':
      button=page.locator('.atlas-space > details ul').first.locator('button').last;target=button.inner_text();button.focus();page.keyboard.press('Enter');expect(page.locator('#atlas-inspector h3')).to_have_text(target);expect(page.locator('#atlas-inspector')).to_be_focused()
    row['passed']=True;context.close()
   for kind in ['no-webgl','blocked-module','refused-bookmark','no-javascript']:
    row={'case':kind};report['cases'].append(row);context=browser.new_context(java_script_enabled=kind!='no-javascript');page=context.new_page();page.set_default_timeout(15000)
    if kind=='no-webgl':page.add_init_script("const original=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return type.includes('webgl')?null:original.call(this,type,...args)}")
    if kind=='blocked-module':page.route('**/atlas-spatial.mjs*',lambda route:route.abort())
    if kind=='no-javascript':
     page.goto(origin);require(page.locator('#current-research a').count()>0,'Missing static source fallback');row['passed']=True;context.close();continue
    page.goto(origin+('?snapshot='+'0'*40+'&atlas-space=3d' if kind=='refused-bookmark' else '?atlas-view=graph'));page.wait_for_function("document.querySelector('#atlas-status').textContent.startsWith('Source snapshot')")
    if kind=='refused-bookmark':expect(page.locator('#atlas-error')).to_be_visible();expect(page.locator('#atlas-space-status')).to_contain_text('unavailable');require(page.locator('#atlas-space-host canvas').count()==0,'Refused bookmark rendered nodes')
    else:
     page.locator('#atlas-space-toggle').click();expect(page.locator('#atlas-space-status')).to_contain_text('unavailable');expect(page.locator('#atlas-stage')).to_be_visible();page.locator('#atlas-view').select_option('list');expect(page.locator('#atlas-list')).to_be_visible()
    row['passed']=True;context.close()
   browser.close()
 except Exception as e:report['failure']=str(e);report['traceback']=traceback.format_exc();raise
 finally:
  server.shutdown();(out/'spatial-browser.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
