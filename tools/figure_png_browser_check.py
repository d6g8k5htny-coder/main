#!/usr/bin/env python3
"""Real-browser PNG download, provenance, stale-work and refusal controls.

Requires optional Playwright and Pillow. Serves local docs by default; no source
retrieval, mathematical checking, spoken-AT or cross-browser verdict is implied.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import struct
import subprocess
import threading
import traceback
import zlib

ROOT = Path(__file__).resolve().parents[1]
HOOKS = r'''(() => {
const src = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, 'src');
window.__pngImages = []; window.__pngCanvases = []; window.__pngHeldImages = [];
window.__pngHeldBlobs = []; window.__pngURLs = new Set(); window.__pngStarts = 0;
const create = Document.prototype.createElement;
Document.prototype.createElement = function(tag, ...args) {
  const node = create.call(this, tag, ...args);
  if (tag === 'canvas') window.__pngCanvases.push(node);
  return node;
};
Object.defineProperty(HTMLImageElement.prototype, 'src', {...src, set(value) {
  if (String(value).startsWith('data:image/svg+xml')) {
    window.__pngImages.push(this); window.__pngStarts++;
    if (window.__pngDecodeFail) {queueMicrotask(() => this.dispatchEvent(new Event('error'))); return;}
    if (window.__pngHoldDecode) {window.__pngHeldImages.push([this, value]); return;}
  }
  return src.set.call(this, value);
}});
window.__pngReleaseImages = () => {
  window.__pngHoldDecode = false;
  for(const [image, value] of window.__pngHeldImages.splice(0))if(image.onload)src.set.call(image, value);
};
const getContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(...args) {
  return window.__pngMissingContext ? null : getContext.apply(this, args);
};
const toBlob = HTMLCanvasElement.prototype.toBlob;
HTMLCanvasElement.prototype.toBlob = function(callback, ...args) {
  if (window.__pngEncodeFail)throw new Error('injected encoding failure');
  if (window.__pngNullBlob) {queueMicrotask(() => callback(null)); return;}
  if (window.__pngHoldEncode)return toBlob.call(this, blob => window.__pngHeldBlobs.push(() => callback(blob)), ...args);
  return toBlob.call(this, callback, ...args);
};
window.__pngReleaseBlobs = () => {
  window.__pngHoldEncode = false;
  for(const release of window.__pngHeldBlobs.splice(0))release();
};
const make = URL.createObjectURL, revoke = URL.revokeObjectURL;
URL.createObjectURL = blob => {const url = make.call(URL, blob); window.__pngURLs.add(url); return url;};
URL.revokeObjectURL = url => {window.__pngURLs.delete(url); return revoke.call(URL, url);};
})();'''


@contextmanager
def serve():
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Quiet, directory=str(ROOT / 'docs/site')))
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/'
    finally: server.shutdown(); server.server_close(); thread.join()


def inspect_png(path, expected, citation, background):
    from PIL import Image, ImageChops
    raw = path.read_bytes(); assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    p = 8; texts = {}
    while p < len(raw):
        size = struct.unpack('>I', raw[p:p+4])[0]; kind = raw[p+4:p+8]
        body = raw[p+8:p+8+size]
        assert len(body) == size
        assert zlib.crc32(kind + body) & 0xffffffff == struct.unpack('>I', raw[p+8+size:p+12+size])[0]
        if kind == b'iTXt':
            keyword, rest = body.split(b'\0', 1); assert rest[:4] == b'\0\0\0\0'
            texts[keyword.decode('latin1')] = rest[4:].decode('utf8')
        p += size + 12
    assert p == len(raw)
    metadata = json.loads(texts['Universal Law figure']); raster = json.loads(texts['Universal Law raster'])
    assert metadata['verification'] == 'not_performed' and metadata['mode'] == 'teaching_model'
    assert metadata['params'] == expected['params']
    assert texts['Description'] == citation and 'not mathematical result acceptance' in citation
    if 'decomposable' in expected: assert metadata['decomposable'] == expected['decomposable']
    datetime.fromisoformat(metadata['generated_at'].replace('Z', '+00:00'))
    identities = metadata.get('sources', [metadata.get('source')])
    for identity in identities:
        assert len(identity['commit']) == 40 and len(identity['blob']) == 40
        assert len(identity['sha256']) == 64 and identity['bytes'] > 0
    with Image.open(io.BytesIO(raw)) as image: image.verify()
    with Image.open(io.BytesIO(raw)) as original:
        image = original.convert('RGBA'); assert image.size == expected['size']
        assert image.getpixel((0, 0)) == (*background, 255)
        assert image.getextrema()[3] == (255, 255)
        plain = Image.new('RGBA', image.size, (*background, 255))
        difference = ImageChops.difference(image.convert('RGB'), plain.convert('RGB'))
        assert difference.getbbox() is not None
        # Caption occupies the lower portion of the full exported SVG.
        assert difference.crop((0, expected['diagram_height']*2, image.width, image.height)).getbbox() is not None
    assert raster['width'] == expected['size'][0] and raster['height'] == expected['size'][1]
    assert raster['scale'] == 2 and raster['mathematical_verification'] == 'not_performed'
    return {'file': path.name, 'bytes': len(raw), 'sha256': sha256(raw).hexdigest(),
            'params': metadata['params'], 'size': list(expected['size']), 'metadata': metadata, 'raster': raster}


def run(origin, browser, out, report, selected=None):
    from playwright.sync_api import expect
    serial = 0
    def context(width=1440, scheme='light', nojs=False):
        c=browser.new_context(viewport={'width':width,'height':900 if width>600 else 844},
                              color_scheme=scheme,device_scale_factor=3 if width<600 else 1,
                              reduced_motion='reduce',accept_downloads=True,java_script_enabled=not nojs)
        c.route('https://**', lambda route: (report['external_requests'].append(route.request.url), route.abort())[1])
        if not nojs: c.add_init_script(HOOKS)
        return c
    def page(c):
        p=c.new_page();p.set_default_timeout(15000);p.on('pageerror',lambda error:report['page_errors'].append(str(error)))
        p.goto(origin+'explore.html');return p
    def close(c): c.unroute_all(behavior='wait');c.close()
    def case(name, fn):
        if selected and not any(part in name for part in selected): return
        row={'case':name};report['cases'].append(row)
        try: fn();row['passed']=True
        except Exception: row['passed']=False;row['error']=traceback.format_exc();raise
    def download(p,prefix,expected):
        nonlocal serial
        button=p.locator(f'#{prefix}-export-png');expect(button).to_be_enabled();button.focus()
        background=p.locator(f'#{prefix}-diagram').evaluate("e => getComputedStyle(e).backgroundColor.match(/\\d+/g).map(Number)")
        with p.expect_download() as event: p.keyboard.press('Enter')
        actual=event.value;assert actual.failure() is None
        serial+=1;path=out/f'{serial:02d}-{prefix}.png';actual.save_as(path)
        assert actual.suggested_filename.endswith('.png')
        citation=p.locator(f'#{prefix}-citation-text').input_value()
        row=inspect_png(path,expected,citation,background);report['downloads'].append(row)
        expect(button).to_be_enabled()
        expect(p.locator(f'#{prefix}-png-message')).to_contain_text('PNG download started')
        p.wait_for_function('() => window.__pngURLs.size === 0')
        assert p.evaluate('window.__pngCanvases.every(c => c.width === 0 && c.height === 0)')
        assert p.evaluate("window.__pngImages.every(i => !i.hasAttribute('src'))")
    defaults={'curvature':{'params':{'s':-2,'R':1},'size':(1200,1070),'diagram_height':350},
              'pin':{'params':{'r':.5,'region':'annulus'},'size':(1200,1164),'diagram_height':350},
              'palette':{'params':{'objects':[1,2,3]},'size':(1200,992),'diagram_height':300,'decomposable':False}}
    for width, scheme in [(1440,'light'),(1440,'dark'),(390,'light'),(320,'dark')]:
        def matrix():
            c=context(width,scheme);p=page(c)
            for prefix,expected in defaults.items():
                download(p,prefix,expected)
                assert p.evaluate('document.documentElement.scrollWidth <= innerWidth')
                p.locator(f'#{prefix}-export-png').scroll_into_view_if_needed()
                file=out/f'controls-{width}-{scheme}-{prefix}.png';p.screenshot(path=str(file));report['screenshots'].append(file.name)
            close(c)
        case(f'Keyboard PNG downloads for three figures at{width}px {scheme} DPR{3 if width<600 else 1}, opaque captioned pixels, exact metadata, cleanup',matrix)
    def parameter_matrix():
        c=context();p=page(c)
        for s,R in [(-.3,.3),(.3,.1),(0,0)]:
            p.locator('#curvature-center').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',str(s))
            p.locator('#curvature-spread').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',str(R))
            download(p,'curvature',{'params':{'s':s,'R':R},'size':(1200,1070),'diagram_height':350})
        for r in [.5,.25]:
            p.locator('#pin-distance').evaluate('(e,v)=>{e.value=v;e.dispatchEvent(new Event("input",{bubbles:true}))}',str(r))
            for region in ['annulus','remote']:
                p.locator(f'input[name=region][value={region}]').check()
                download(p,'pin',{'params':{'r':r,'region':region},'size':(1200,1164),'diagram_height':350})
        for mask in range(8):
            objects=[]
            for index,id in enumerate(['object-one','object-two','object-three']):
                checked=bool(mask&(1<<index));p.locator('#'+id).set_checked(checked)
                if checked:objects.append(index+1)
            download(p,'palette',{'params':{'objects':objects},'size':(1200,992 if mask==7 else 956),
                                  'diagram_height':300,'decomposable':mask!=7})
        close(c)
    case('Curvature decimal and zero boundary; pin highlights at both scales; all8finite selections',parameter_matrix)
    def stale_controls():
        for stage in ['Decode','Encode']:
            for prefix in ['curvature','pin','palette']:
                c=context();p=page(c);downloads=[];p.on('download',lambda d:downloads.append(d.suggested_filename))
                p.evaluate(f'window.__pngHold{stage}=true');button=p.locator(f'#{prefix}-export-png');button.click()
                p.wait_for_function('() => window.__pngHeldImages.length > 0' if stage=='Decode' else '() => window.__pngHeldBlobs.length > 0')
                if prefix=='curvature':p.locator('#curvature-center').focus();p.keyboard.press('ArrowRight')
                elif prefix=='pin':p.locator('#pin-half').click()
                else:p.locator('#object-three').uncheck()
                expect(button).to_be_enabled();expect(p.locator(f'#{prefix}-png-message')).to_contain_text('canceled')
                if prefix=='curvature':p.keyboard.press('ArrowLeft')
                elif prefix=='pin':p.locator('#pin-half').click()
                else:p.locator('#object-three').check()
                expect(button).to_be_enabled()
                p.evaluate('window.__pngReleaseImages()' if stage=='Decode' else 'window.__pngReleaseBlobs()')
                p.wait_for_function('() => window.__pngImages.every(i=>!i.hasAttribute("src"))')
                assert not downloads,downloads
                download(p,prefix,defaults[prefix]);close(c)
        for stage in ['Decode','Encode']:
            c=context();p=page(c);downloads=[];p.on('download',lambda d:downloads.append(d.suggested_filename))
            p.evaluate(f'window.__pngHold{stage}=true');p.locator('#curvature-export-png').click()
            p.wait_for_function('() => window.__pngHeldImages.length > 0' if stage=='Decode' else '() => window.__pngHeldBlobs.length > 0')
            p.emulate_media(color_scheme='dark');expect(p.locator('#curvature-png-message')).to_contain_text('canceled')
            p.evaluate('window.__pngReleaseImages()' if stage=='Decode' else 'window.__pngReleaseBlobs()')
            p.wait_for_timeout(100);assert not downloads
            download(p,'curvature',defaults['curvature']);close(c)
    case('Control changes away/back and theme changes during decode or encode cancel old PNGs; retry works',stale_controls)
    def failures():
        for flag in ['DecodeFail','MissingContext','EncodeFail','NullBlob']:
            c=context();p=page(c);downloads=[];p.on('download',lambda d:downloads.append(d.suggested_filename))
            p.evaluate(f'window.__png{flag}=true');p.locator('#curvature-export-png').click()
            expect(p.locator('#curvature-png-message')).to_contain_text('PNG could not complete')
            assert not downloads;p.evaluate(f'window.__png{flag}=false');download(p,'curvature',defaults['curvature']);close(c)
        c=context();p=page(c);p.evaluate('window.__pngHoldDecode=true');p.locator('#curvature-export-png').click()
        expect(p.locator('#curvature-export-png')).to_be_disabled()
        p.evaluate("document.getElementById('curvature-export-png').dispatchEvent(new MouseEvent('click'))")
        assert p.evaluate('window.__pngStarts')==1
        with p.expect_download():p.evaluate('window.__pngReleaseImages()')
        expect(p.locator('#curvature-export-png')).to_be_enabled();close(c)
    case('Decode/context/encoding/null failures remain explicit and retry; pending repeats coalesce',failures)
    def tamper():
        for prefix,mutation in [('curvature',"e.querySelector('circle').setAttribute('cx','1')"),
                                ('pin',"e.querySelector('circle').setAttribute('r','1')"),
                                ('palette',"e.appendChild(document.createElementNS('http://www.w3.org/2000/svg','image'))")]:
            c=context();p=page(c);downloads=[];p.on('download',lambda d:downloads.append(d.suggested_filename))
            p.locator(f'#{prefix}-diagram').evaluate('e => {'+mutation+'}')
            p.locator(f'#{prefix}-export-png').click();expect(p.locator(f'#{prefix}-png-message')).to_contain_text('PNG could not complete')
            assert not downloads
            assert p.locator(f'#{prefix}-citation-text').input_value()
            close(c)
        c=context();p=page(c);downloads=[];p.on('download',lambda d:downloads.append(d.suggested_filename))
        p.evaluate('window.__pngHoldEncode=true');p.locator('#curvature-export-png').click()
        p.wait_for_function('() => window.__pngHeldBlobs.length > 0')
        p.locator('#curvature-diagram circle').evaluate("e=>e.setAttribute('cx','1')")
        p.evaluate('window.__pngReleaseBlobs()');expect(p.locator('#curvature-png-message')).to_contain_text('PNG could not complete')
        assert not downloads;assert p.locator('#curvature-citation-text').input_value();close(c)
    case('Existing geometry/resource validators and final recapture refuse tampered content without clearing citations',tamper)
    def fallbacks():
        c=context(nojs=True);p=page(c)
        for prefix in defaults:
            expect(p.locator(f'#{prefix}-export-png')).to_be_disabled()
            expect(p.locator(f'#{prefix}-export-svg')).to_be_disabled()
            citation=p.locator(f'#{prefix}-citation-text').input_value()
            assert 'teaching figure' in citation and 'Source identity is recorded provenance' in citation
            assert len(citation)>300 and 'not performed' in citation.lower()
        close(c)
        c=context();p=page(c)
        for prefix in defaults:
            with p.expect_download() as event:p.locator(f'#{prefix}-export-svg').click()
            assert event.value.suggested_filename.endswith('.svg')
            data=Path(event.value.path()).read_text();assert '<metadata' in data and 'not_performed' in data
        p.emulate_media(media='print')
        for prefix in defaults:expect(p.locator(f'#{prefix}-export-png')).not_to_be_visible()
        close(c)
    case('No-JavaScript PNG/SVG disabled and readable manual citations; existing SVG downloads unchanged',fallbacks)
    def unavailable_png():
        c=context();c.add_init_script("HTMLCanvasElement.prototype.toBlob = undefined")
        p=page(c)
        for prefix in defaults:
            expect(p.locator(f'#{prefix}-export-png')).to_be_disabled()
            expect(p.locator(f'#{prefix}-export-svg')).to_be_enabled()
            expect(p.locator(f'#{prefix}-png-message')).to_contain_text('unavailable')
            assert p.locator(f'#{prefix}-citation-text').input_value()
        close(c)
        c=context();c.add_init_script("""const original=window.matchMedia;
            window.matchMedia=(...args)=>{const q=original.apply(window,args);q.addEventListener=undefined;return q;}""")
        p=page(c)
        for prefix in defaults:expect(p.locator(f'#{prefix}-export-svg')).to_be_enabled()
        p.evaluate('window.__pngHoldDecode=true');p.locator('#curvature-export-png').click()
        p.wait_for_function('() => window.__pngHeldImages.length > 0')
        p.emulate_media(color_scheme='dark');expect(p.locator('#curvature-png-message')).to_contain_text('canceled')
        p.evaluate('window.__pngReleaseImages()')
        download(p,'curvature',defaults['curvature']);close(c)
    case('Unavailable PNG encoding and legacy media listeners preserve existing figure controls',unavailable_png)
    def late_completion():
        c=context();p=page(c)
        p.evaluate('window.__pngHoldEncode=true');p.locator('#curvature-export-png').click()
        p.wait_for_function('() => window.__pngHeldBlobs.length > 0')
        p.locator('#curvature-center').focus();p.keyboard.press('ArrowRight')
        expect(p.locator('#curvature-png-message')).to_contain_text('canceled')
        p.evaluate('window.__pngHoldEncode=false')
        download(p,'curvature',{'params':{'s':-1.9,'R':1},'size':(1200,1070),'diagram_height':350})
        status=p.locator('#curvature-png-message').inner_text();citation=p.locator('#curvature-citation-text').input_value()
        p.evaluate('window.__pngReleaseBlobs()');p.wait_for_timeout(100)
        assert p.locator('#curvature-png-message').inner_text()==status
        assert p.locator('#curvature-citation-text').input_value()==citation
        close(c)
    case('Late canceled encoding cannot replace newer successful PNG state or current citation',late_completion)




def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--browser',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--origin');parser.add_argument('--case',action='append',dest='selected',help='Run matching case-name substrings only')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    browser_path=Path(args.browser).resolve()
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    files=['docs/site/figure-png.mjs','docs/site/explore.mjs','docs/site/explore.html','tools/figure_png_browser_check.py']
    report={'started_at':datetime.now(timezone.utc).isoformat(),'cases':[],'downloads':[],'screenshots':[],
            'page_errors':[],'external_requests':[],'source':{'git_head':git.stdout.strip() if git.returncode==0 else None,
            'files_sha256':{f:sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).is_file()}},
            'browser':{'executable':str(browser_path),'sha256':sha256(browser_path.read_bytes()).hexdigest(),
            'version':subprocess.run([str(browser_path),'--version'],capture_output=True,text=True,check=True).stdout.strip(),
            'chromium_sandbox':True},'limitations':'Chromium/Linux only; author-side engineering tests; no mathematical or spoken-AT verdict.'}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=str(browser_path),headless=True,chromium_sandbox=True)
            try:
                if args.origin:run(args.origin.rstrip('/')+'/',browser,args.output,report,args.selected)
                else:
                    with serve() as origin:run(origin,browser,args.output,report,args.selected)
                assert report['cases'], 'No browser case matched the filter'
                assert not report['page_errors'],report['page_errors']
                assert not report['external_requests'],report['external_requests']
                report['passed']=True
            finally:browser.close()
    except Exception:
        report['passed']=False;report['error']=traceback.format_exc();raise
    finally:
        report['finished_at']=datetime.now(timezone.utc).isoformat()
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'downloads':len(report['downloads'])}))

if __name__=='__main__':main()
