#!/usr/bin/python3.12 -I
# Generate one real X key press+release through AT-SPI's device event controller (XTest),
# so that Orca's keyboard listener sees the key (unlike CDP Input.dispatchKeyEvent).
# usage: xtest_key.py Return|space
import sys, gi
gi.require_version('Atspi', '2.0')
from gi.repository import Atspi
keysyms = {'Return': 0xff0d, 'space': 0x0020}
ok = Atspi.generate_keyboard_event(keysyms[sys.argv[1]], None, Atspi.KeySynthType.SYM)
print('generated', sys.argv[1], ok)
