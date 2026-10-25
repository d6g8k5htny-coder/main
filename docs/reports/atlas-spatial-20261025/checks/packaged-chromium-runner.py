import sys
sys.path.insert(0,'/workspace/atlas-spatial');sys.path.insert(0,'/workspace/atlas-spatial/tools')
from playwright.sync_api import BrowserType
original=BrowserType.launch
def launch(self,**kwargs):
 kwargs.pop('channel',None);kwargs['executable_path']='/usr/bin/chromium';kwargs['args']=['--enable-unsafe-swiftshader','--disable-background-networking']
 return original(self,**kwargs)
BrowserType.launch=launch
from tools.current_site_browser_check import main
sys.argv=['current_site_browser_check.py','--output','/workspace/scratch/current-chromium'];main()
