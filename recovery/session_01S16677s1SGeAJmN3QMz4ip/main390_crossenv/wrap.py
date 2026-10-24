"""Run the candidate's own tools/current_site_browser_check.py unchanged, with one environment shim:
Playwright's bundled Linux Chromium (no Google Chrome channel, no sandbox as root in a container)."""
import runpy, sys
from playwright.sync_api import BrowserType
_orig = BrowserType.launch
def _launch(self, *a, **kw):
    kw.pop('channel', None); kw['chromium_sandbox'] = False
    kw['executable_path'] = '/opt/pw-browsers/chromium'
    return _orig(self, *a, **kw)
BrowserType.launch = _launch
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name='__main__')
