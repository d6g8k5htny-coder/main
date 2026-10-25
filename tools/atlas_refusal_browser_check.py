#!/usr/bin/env python3
"""Verify refused bookmark identity through real Atlas controls and traversal."""
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess,threading,time,traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit,parse_qs,urlencode
from playwright.sync_api import sync_playwright,expect
ROOT=Path(__file__).resolve().parents[1]
def require(value,message):
 if not value:raise AssertionError(message)
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--site',type=Path,default=ROOT/'docs/site');parser.add_argument('--output',type=Path,required=True);parser.add_argument('--browser',default='/usr/bin/chromium');parser.add_argument('--observe',action='store_true',help='Record baseline failures; not acceptance');args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
 class Handler(SimpleHTTPRequestHandler):
  def log_message(self,*args):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),partial(Handler,directory=str(args.site.parent)));threading.Thread(target=server.serve_forever,daemon=True).start();origin=f'http://127.0.0.1:{server.server_port}/{args.site.name}/dependencies.html'
 payload=(args.site/'current.json').read_bytes();data=json.loads(payload);math=data['graph']['identity']['commit'];main=next(r['commit']for r in data['repositories']if r['repository'].endswith('/main'));node=next(iter(data['graph']['data']['nodes']));bad='0'*40
 report={'mode':'baseline-observation'if args.observe else'acceptance','checked_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'platform':platform.platform(),'chromium_sandbox':True,'payload_sha256':hashlib.sha256(payload).hexdigest(),'source_repositories':data['repositories'],'cases':[]}
 fields=['snapshot','main-snapshot','atlas-node']
 def pins(url):
  q=parse_qs(urlsplit(url).query,keep_blank_values=True);return{k:q.get(k,[])for k in fields}
 try:
  with sync_playwright()as p:
   browser=p.chromium.launch(executable_path=args.browser,headless=True,chromium_sandbox=True,args=['--enable-unsafe-swiftshader','--disable-background-networking']);report['browser']=browser.version
   for width,scheme in [(1200,'light'),(390,'dark')]:
    for kind,items in [('math',[('snapshot',bad),('main-snapshot',main)]),('main',[('snapshot',math),('main-snapshot',bad)]),('duplicates',[('snapshot',bad),('snapshot',math),('main-snapshot',bad),('main-snapshot',main)]),('malformed',[('snapshot','malformed'),('main-snapshot',main)])]:
     row={'case':kind,'width':width,'scheme':scheme,'passed':False,'states':[],'errors':[]};report['cases'].append(row);context=browser.new_context(viewport={'width':width,'height':900},color_scheme=scheme,reduced_motion='reduce');page=context.new_page();page.on('pageerror',lambda e:row['errors'].append(str(e)));page.set_default_timeout(10000)
     try:
      query=urlencode(items+[('atlas-node',node),('source-q','preserve'),('unrelated','reader value')]);address=origin+'?'+query+'#current-atlas';original=pins(address);page.goto(address);expect(page.locator('#atlas-error')).to_be_visible();expect(page.locator('#atlas-count')).to_contain_text('refused')
      def state(label):
       entry={'label':label,'url':page.url,'share':page.locator('#atlas-share').get_attribute('href'),'view':page.locator('#atlas-view').input_value(),'error':page.locator('#atlas-error').inner_text(),'count':page.locator('#atlas-count').inner_text(),'nodes':page.locator('#atlas-canvas .atlas-node').count()+page.locator('#atlas-list button').count()};row['states'].append(entry);return entry
      initial=state('initial-refusal');row['initial_share_preserved']=pins(initial['share'])==original
      first='graph'if width<721 else'list';t=time.perf_counter();page.locator('#atlas-view').select_option(first);expect(page.locator('#atlas-view')).to_have_value(first);changed=state('after-view-change');row['view_change_ms']=(time.perf_counter()-t)*1000
      page.locator('#current-atlas').screenshot(path=str(args.output/f'{kind}-{width}.png'))
      require(pins(changed['url'])==original,'View change replaced refused source pins');require(pins(changed['share'])==original,'Share replaced refused source pins');require(row['initial_share_preserved'],'Initial Share lost refused source identity');require(changed['nodes']==0,'Refused view rendered current nodes');require(parse_qs(urlsplit(page.url).query).get('atlas-view')==[first],'Address did not retain view choice')
      require(parse_qs(urlsplit(page.url).query).get('unrelated')==['reader value'],'Unrelated address state lost');share=changed['share'];page.goto(share);expect(page.locator('#atlas-error')).to_be_visible();require(pins(page.url)==original,'Opening Share changed source identity');page.reload();expect(page.locator('#atlas-error')).to_be_visible();require(pins(page.url)==original,'Reload changed source identity');state('share-reload')
      # A real layout action creates history; neither traversal may adopt source.
      page.locator('#atlas-open-reader'if width<721 else'#atlas-tab-evidence').click();expect(page.locator('#atlas-inspector')).to_contain_text('snapshot'if kind!='malformed'else'snapshot');page.go_back();expect(page.locator('#atlas-error')).to_be_visible();require(pins(page.url)==original,'Back replaced source identity');page.go_forward();expect(page.locator('#atlas-error')).to_be_visible();require(pins(page.url)==original,'Forward replaced source identity');state('history-refusal')
      # Explicit current-node adoption is a separate canonical user action.
      page.locator('#atlas-tab-diagnostics').click();page.locator('#atlas-diagnostics').evaluate('(e)=>e.closest("details").open=true');page.locator('#atlas-diagnostics button').first.click();expect(page.locator('#atlas-error')).to_be_hidden();adopted=state('explicit-current-selection');require(pins(adopted['url'])['snapshot']==[math]and pins(adopted['url'])['main-snapshot']==[main],'Explicit current selection did not adopt actual pins');page.go_back();expect(page.locator('#atlas-error')).to_be_visible();require(pins(page.url)==original,'Back from explicit adoption did not restore refused identity');page.go_forward();expect(page.locator('#atlas-error')).to_be_hidden();require(not row['errors'],str(row['errors']));row['passed']=True
     except Exception as error:row['failure']=str(error);row['traceback']=traceback.format_exc()
     finally:context.close()
   # Whole-payload byte refusal remains ahead of controls, independently of URL.
   row={'case':'unverified-payload','passed':False};report['cases'].append(row);context=browser.new_context();page=context.new_page();page.route('**/current.json*',lambda route:route.fulfill(status=200,content_type='application/json',body=b'{}'))
   try:
    page.goto(origin);expect(page.locator('#atlas-status')).to_contain_text('did not verify');require(page.locator('#atlas-view').is_disabled(),'Unverified payload enabled controls');row['passed']=True
   except Exception as error:row['failure']=str(error)
   finally:context.close()
   row={'case':'no-javascript-static-reading','passed':False};report['cases'].append(row);context=browser.new_context(java_script_enabled=False);page=context.new_page()
   try:page.goto(origin);require(page.locator('#current-research a').count()>0,'Static source links missing');row['passed']=True
   except Exception as error:row['failure']=str(error)
   finally:context.close()
   browser.close()
 finally:
  server.shutdown();report['passed']=sum(c['passed']for c in report['cases']);report['total']=len(report['cases']);(args.output/'refusal-browser.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'mode':report['mode'],'passed':report['passed'],'total':report['total'],'report':str(args.output/'refusal-browser.json')}))
 return 0 if args.observe or report['passed']==report['total'] else 1
if __name__=='__main__':raise SystemExit(main())
