from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
import json,hashlib,threading,time,statistics,platform,subprocess
from playwright.sync_api import sync_playwright,expect
ROOT=Path('/workspace/atlas-refinement-20261026');OUT=Path('/workspace/scratch/atlas-refinement-20261026/performance.json')
SITES={'baseline':Path('/workspace/scratch/atlas-refinement-20261026/public-before/site'),'candidate':Path('/workspace/scratch/atlas-refinement-20261026/coherent-fixture/docs/site')}
class Handler(SimpleHTTPRequestHandler):
 def log_message(self,*args):pass
servers={};origins={}
for name,site in SITES.items():
 s=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(site)));threading.Thread(target=s.serve_forever,daemon=True).start();servers[name]=s;origins[name]=f'http://127.0.0.1:{s.server_port}'
report={'environment':{'platform':platform.platform(),'playwright':'1.62.0','sandbox':True,'renderer':'Chromium SwiftShader software WebGL','DPR':1,'reduced_motion':'reduce','isolated_contexts':True,'network':'local HTTP; shared execution host; no throttling'},'dataset':{'payload_sha256':hashlib.sha256((SITES['baseline']/'current.json').read_bytes()).hexdigest(),'bytes':(SITES['baseline']/'current.json').stat().st_size,'nodes':49,'relationships':55,'equations':108},'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'journeys':[],'restore':[]}
def save():OUT.write_text(json.dumps(report,indent=2)+'\n')
try:
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='/usr/bin/chromium',chromium_sandbox=True,args=['--enable-unsafe-swiftshader']);report['environment']['browser']=browser.version
  for i,(width,scheme) in enumerate([(1200,'light'),(390,'light'),(1200,'dark'),(390,'dark')]):
   for side in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
    context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.set_default_timeout(15000);row={'side':side,'width':width,'scheme':scheme};report['journeys'].append(row)
    t=time.perf_counter();page.goto(origins[side]+'/dependencies.html?atlas-view=graph&atlas-node=math.uniform-matrix-cap-lifetime&atlas-q=.');expect(page.locator('#atlas-inspector h3')).to_have_text('math.uniform-matrix-cap-lifetime');expect(page.locator('#current-research .current-state')).to_contain_text('Verified source observation:');row['ready_ms']=(time.perf_counter()-t)*1000
    row['before_3d_resources']=page.evaluate('() => performance.getEntriesByType("resource").map(e=>({url:new URL(e.name).pathname,decoded:e.decodedBodySize,transfer:e.transferSize}))');assert not any('three' in r['url'] for r in row['before_3d_resources'])
    page.locator('#atlas-space-toggle').scroll_into_view_if_needed();t=time.perf_counter();page.locator('#atlas-space-toggle').click();page.locator('#atlas-space-host').scroll_into_view_if_needed();expect(page.locator('#atlas-space-host')).to_have_attribute('data-ready','true');expect(page.locator('#atlas-space-host canvas')).to_have_count(1);row['activate_3d_ms']=(time.perf_counter()-t)*1000
    page.locator('#atlas-space-host').scroll_into_view_if_needed();box=page.locator('#atlas-space-host canvas').bounding_box();row['rendered_nodes']=page.locator('#atlas-space-host').get_attribute('data-nodes');row['rendered_edges']=page.locator('#atlas-space-host').get_attribute('data-edges')
    page.evaluate('() => {window.frameSamples=[];window.frameStop=false;let last=performance.now();function tick(now){frameSamples.push(now-last);last=now;if(!frameStop)requestAnimationFrame(tick);}requestAnimationFrame(tick);}')
    t=time.perf_counter();page.mouse.move(box['x']+box['width']*.45,box['y']+box['height']*.4);page.mouse.down();page.mouse.move(box['x']+box['width']*.65,box['y']+box['height']*.5,steps=20);page.mouse.up();page.wait_for_timeout(250);row['orbit_wall_ms']=(time.perf_counter()-t)*1000
    frames=page.evaluate('() => {frameStop=true;return frameSamples;}');row['animation_frame_intervals_ms']=frames;row['frame_sample_note']='Browser animation-frame intervals during pointer orbit, not GPU timing or a sustained FPS certification';page_errors=[]
    context.close();save()
  browser.close()

finally:
 for s in servers.values():s.shutdown();s.server_close()
print(OUT)
