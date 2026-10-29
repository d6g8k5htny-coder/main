#!/usr/bin/env python3
"""Bounded real Orca session; owned children are always terminated on exit."""
import argparse, contextlib, functools, hashlib, http.server, json, os, pathlib, re, shutil, signal
import socket, subprocess, threading, time, traceback
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--label', default='comparison')
ap.add_argument('--url')
ap.add_argument('--serve', default=str(ROOT))
ap.add_argument('--path', default='comparison.html')
ap.add_argument('--browse-steps', type=int, default=24)
ap.add_argument('--sayall-seconds', type=float, default=22)
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--scenario', type=pathlib.Path)
args = ap.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]+',args.label): ap.error('label must use letters, numbers, _ or -')
out = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+args.label)
out.mkdir(parents=True)
children=[]; handles=[]
env=os.environ.copy()
for key, name in [('XDG_CONFIG_HOME','config'),('XDG_CACHE_HOME','cache'),
                  ('XDG_DATA_HOME','data'),('XDG_STATE_HOME','state'),
                  ('XDG_RUNTIME_DIR','runtime'),('TMPDIR','tmp')]:
    d=out/name; d.mkdir(mode=0o700); env[key]=str(d)
# Unix socket filenames must fit sun_path (108 bytes). Keep Chromium temp short.
shorttmp=ROOT/'tmp'/str(os.getpid()); shorttmp.mkdir(parents=True,mode=0o700)
env['TMPDIR']=str(shorttmp)
env['DISPLAY']=':97'
env['PULSE_SERVER']='unix:'+str(out/'pulse.sock')
env['SPEECHD_ADDRESS']='unix_socket:'+str(out/'speechd.sock')
env['SPEECHD_SOCKET']=str(out/'speechd.sock')
actions=(out/'actions.jsonl').open('w'); handles.append(actions)
def mark(action, **data):
    r=dict(time=time.time(),action=action,**data)
    actions.write(json.dumps(r,ensure_ascii=False,default=str)+'\n'); actions.flush()
    print(json.dumps(r,default=str),flush=True)
def spawn(name, command):
    f=(out/(name+'.log')).open('w'); handles.append(f)
    p=subprocess.Popen([str(x) for x in command],env=env,stdout=f,stderr=subprocess.STDOUT,
                       start_new_session=True)
    children.append((name,p)); mark('spawn',name=name,pid=p.pid,command=[str(x) for x in command])
    return p
def command(argv, check=True):
    return subprocess.run(argv,env=env,text=True,capture_output=True,check=check,timeout=15)
def wait_for(pred, name, seconds=12):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        if pred(): return
        time.sleep(.15)
    raise RuntimeError('Readiness timeout: '+name)
def key(sequence, delay=1.4):
    mark('key',sequence=sequence)
    command(['xdotool','key','--clearmodifiers',sequence])
    time.sleep(delay)

server=None
try:
    mark('begin',directory=str(out),arguments=vars(args),audio='PulseAudio null sink; no human listening')
    # Preserve exact harness bytes with each execution, including failed runs.
    snapshot=out/'harness'; snapshot.mkdir()
    for name in ['run.sh','env.sh','session.py','atspi-events.py','comparison.html','analyze.py']:
        shutil.copy2(ROOT/name,snapshot/name)
    if args.scenario:
        shutil.copy2(args.scenario,snapshot/'scenario.json')
    versions=command(['/usr/bin/chromium','--version']).stdout
    versions+=command(['/usr/bin/python3',ROOT/'runtime/usr/bin/orca','--version']).stdout
    versions+=command(['speech-dispatcher','--version']).stdout
    versions+=command(['espeak-ng','--version']).stdout
    (out/'versions.txt').write_text(versions)
    spawn('xvfb',['Xvfb',env['DISPLAY'],'-screen','0','1440x1000x24','-nolisten','tcp'])
    wait_for(lambda: command(['xdpyinfo'],False).returncode==0,'Xvfb')
    pulsemods=ROOT/'runtime/usr/lib/pulse-17.0+dfsg1/modules'
    spawn('pulse',['pulseaudio','-n','--daemonize=no','--exit-idle-time=-1',
                  '--use-pid-file=no','--disable-shm=yes','--dl-search-path='+str(pulsemods),
                  '--load=module-native-protocol-unix socket='+str(out/'pulse.sock'),
                  '--load=module-null-sink sink_name=at_null'])
    wait_for(lambda: (out/'pulse.sock').exists(),'PulseAudio')
    pcm=(out/'audio-s16le-22050-mono.raw').open('wb'); handles.append(pcm)
    pcmerr=(out/'audio-recorder.log').open('w'); handles.append(pcmerr)
    recorder=subprocess.Popen(['parec','--raw','--format=s16le','--rate=22050',
        '--channels=1','--device=at_null.monitor'],env=env,stdout=pcm,stderr=pcmerr,
        start_new_session=True)
    children.append(('audio-recorder',recorder))
    conf=out/'speech-config'; (conf/'modules').mkdir(parents=True)
    (conf/'modules/espeak-ng.conf').write_text((ROOT/'runtime/etc/speech-dispatcher/modules/espeak-ng.conf').read_text())
    logs=out/'speechd'; logs.mkdir()
    (conf/'speechd.conf').write_text(f'''LogLevel 5
LogDir "{logs}"
CustomLogFile "protocol" "{logs}/protocol.log"
AudioOutputMethod "pulse"
AddModule "espeak-ng" "{ROOT}/runtime/usr/lib/speech-dispatcher-modules/sd_espeak-ng" "espeak-ng.conf"
DefaultModule espeak-ng
DefaultLanguage en
DefaultRate 75
''')
    spawn('speech-dispatcher',['speech-dispatcher','-s','-t','0','-C',conf,
          '-S',out/'speechd.sock','-P',out/'speechd.pid','-L',logs,
          '-m',ROOT/'runtime/usr/lib/speech-dispatcher-modules'])
    wait_for(lambda: (out/'speechd.sock').exists(),'speech-dispatcher')
    prefs=out/'orca-prefs'; prefs.mkdir()
    (prefs/'user-settings.conf').write_text(json.dumps({
       'general':{'startingProfile':['Default','default']},
       'profiles':{'default':{'profile':['Default','default'],'enableSpeech':True,
       'enableBraille':False,'enableTutorialMessages':False,'keyboardLayout':1,
       'speechServerFactory':'speechdispatcherfactory',
       'voices':{'default':{'rate':85.0}}}}, 'pronunciations':{},'keybindings':{}},indent=2))
    spawn('atspi-events',['/usr/bin/python3',ROOT/'atspi-events.py'])
    spawn('orca',['/usr/bin/python3',ROOT/'runtime/usr/bin/orca','--debug-file',out/'orca-debug.log',
                  '--user-prefs',prefs,'--disable','braille','--speech-system','speechdispatcherfactory'])
    time.sleep(3)
    handler=functools.partial(http.server.SimpleHTTPRequestHandler,directory=args.serve)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    url=args.url or f'http://127.0.0.1:{server.server_port}/'+args.path.lstrip('/')
    with sync_playwright() as pw:
        browser=pw.chromium.launch_persistent_context(str(out/'chromium-profile'),
            executable_path='/usr/bin/chromium',headless=False,chromium_sandbox=True,
            env=env,
            args=['--force-renderer-accessibility','--disable-dev-shm-usage','--no-first-run'],
            viewport={'width':1380,'height':900})
        page=browser.pages[0]
        page.set_default_timeout(12000)
        responses=[]
        def response_record(response):
            # Identity metadata only: no response rewriting or accessibility interception.
            try:
                body=response.body()
                responses.append(dict(url=response.url,status=response.status,bytes=len(body),
                                      sha256=hashlib.sha256(body).hexdigest()))
            except Exception as exc:
                responses.append(dict(url=response.url,error=str(exc)))
        page.on('response',response_record)
        page.on('console',lambda msg: mark('browser-console',type=msg.type,text=msg.text))
        page.goto(url,wait_until='networkidle'); page.bring_to_front(); time.sleep(3)
        mark('loaded',url=url,title=page.title())
        page.screenshot(path=str(out/'loaded.png'),full_page=True)
        (out/'loaded.html').write_text(page.content())
        # Chromium command lines are read only for this isolated profile, without environment data.
        cmds=[]
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try:
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if str(out/'chromium-profile') in cmd: cmds.append(cmd)
            except (OSError,PermissionError): pass
        (out/'chromium-command-lines.txt').write_text('\n'.join(cmds))
        if any('--no-sandbox' in c for c in cmds): raise RuntimeError('Unexpected disabled Chromium sandbox')
        if args.scenario:
            for step in json.loads(args.scenario.read_text()):
                action=step['action']; mark('scenario-step',step=step)
                if action=='phase': mark('phase',name=step['name'])
                elif action=='click': page.locator(step['selector']).click(); time.sleep(step.get('delay',1))
                elif action=='focus': page.locator(step['selector']).focus(); time.sleep(step.get('delay',1))
                elif action=='key':
                    for _ in range(step.get('repeat',1)): key(step['sequence'],step.get('delay',1))
                elif action=='type':
                    command(['xdotool','type','--clearmodifiers','--delay','45',step['text']]); time.sleep(step.get('delay',2))
                elif action=='wait': time.sleep(min(step['seconds'],45))
                elif action=='snapshot':
                    name=step['name']
                    if not re.fullmatch(r'[A-Za-z0-9_-]+',name): raise ValueError('Invalid snapshot name')
                    page.screenshot(path=str(out/(name+'.png')),full_page=True)
                    (out/(name+'.html')).write_text(page.content())
                    texts={s:page.locator(s).all_text_contents() for s in step.get('selectors',[])}
                    (out/(name+'-texts.json')).write_text(json.dumps(texts,indent=2))
                else: raise ValueError('Unknown scenario action '+action)
            mark('phase',name='finished')
        elif not args.smoke:
            # Escape leaves controls; Ctrl+Home is standard browser document top.
            mark('phase',name='browse')
            # A real pointer click establishes document focus before keyboard traversal.
            page.locator('h1').first.click(); time.sleep(1)
            key('Escape',.3); key('ctrl+Home',2)
            for _ in range(args.browse_steps): key('Down',.7)
            mark('phase',name='sayall')
            key('ctrl+Home',1.5)
            # Default desktop layout: keypad Add = Say All.
            key('KP_Add',args.sayall_seconds)
            key('Control_L',.5)
            if not args.url and args.path=='comparison.html':
                for case in ['direct','wrapped','polite']:
                    mark('phase',name='update-'+case)
                    page.locator('#'+case+'-update').focus(); time.sleep(1)
                    key('Return',4)
                (out/'dom-updates.json').write_text(json.dumps(page.evaluate('window.changes'),indent=2))
            mark('phase',name='finished')
        (out/'response-identities.json').write_text(json.dumps(responses,indent=2))
        mark('process-health',status={name:p.poll() for name,p in children})
        browser.close()
    mark('completed')
except BaseException as exc:
    mark('failure',error=str(exc)); (out/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    if server: server.shutdown(); server.server_close()
    for name,p in reversed(children):
        if p.poll() is None:
            with contextlib.suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGTERM)
        try: p.wait(timeout=4)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError): os.killpg(p.pid,signal.SIGKILL)
            p.wait(timeout=2)
        mark('child-exit',name=name,returncode=p.returncode)
    for f in handles: f.close()
    print('Evidence: '+str(out),flush=True)
