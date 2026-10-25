#!/bin/bash
# usage: run_stack.sh SCENARIO [extra chrome args...]
# One Xvfb/AT-SPI/speech-dispatcher/Orca/Chromium stack per call.
SC="$1"; shift
R=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220/replicate
OUT=$R/evidence/$SC; rm -rf "$OUT"; mkdir -p "$OUT/sdlog"
THREE=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/three/package/build/three.js
cleanup_strays() {
  pkill -x Xvfb; pkill -f '/usr/bin/orca'; pkill -f chrome-linux/chrome; pkill -x speech-dispatch; pkill -x sd_espeak-ng; pkill -x sd_openjtalk
  pkill -f at-spi-bus-launcher; pkill -f at-spi2-registryd; pkill -f rt_sink.py
  sleep 1
  pkill -9 -x Xvfb; pkill -9 -f '/usr/bin/orca'; pkill -9 -f chrome-linux/chrome; pkill -9 -x speech-dispatch; pkill -9 -x sd_espeak-ng
  rm -f /root/.cache/speech-dispatcher/speechd.sock /tmp/.X99-lock
}
cleanup_strays
echo "stray check: $(ps aux | egrep 'Xvfb|/usr/bin/orca|chrome-linux|speech-dispatch|sd_espeak' | grep -v grep | wc -l)" > "$OUT/stack.log"
export DISPLAY=:99
Xvfb :99 -screen 0 1280x900x24 >"$OUT/xvfb.log" 2>&1 & XP=$!
sleep 1
eval "$(dbus-launch --sh-syntax)"; echo "dbus $DBUS_SESSION_BUS_PID" >> "$OUT/stack.log"
export GTK_MODULES=gail:atk-bridge ACCESSIBILITY_ENABLED=1 NO_AT_BRIDGE=0
/usr/libexec/at-spi-bus-launcher --launch-immediately >"$OUT/atspi.log" 2>&1 & AP=$!
sleep 1
mkdir -p /root/.cache/speech-dispatcher
RT_SINK_LOG="$OUT/sink.log" HOME=$R/tmp/sdhome speech-dispatcher -s -l 5 -L "$OUT/sdlog" -C $R/scripts/sdconf \
  -c unix_socket -S /root/.cache/speech-dispatcher/speechd.sock -t 0 >"$OUT/speechd.out" 2>&1 & SP=$!
for i in $(seq 50); do [ -S /root/.cache/speech-dispatcher/speechd.sock ] && break; sleep 0.1; done
export SPEECHD_ADDRESS=unix_socket:/root/.cache/speech-dispatcher/speechd.sock
mkdir -p "$OUT/orca-prefs"
/usr/bin/python3.12 /usr/bin/orca --replace -u "$OUT/orca-prefs" --debug-file="$OUT/orca-debug.txt" >"$OUT/orca.out" 2>&1 & OP=$!
sleep 5
START='data:text/html,<!doctype html><html lang="en"><head><title>c220 start</title></head><body><p>start page</p></body></html>'
/opt/pw-browsers/chromium-1194/chrome-linux/chrome --no-sandbox --force-renderer-accessibility --no-first-run --no-default-browser-check \
  --disable-background-networking --user-data-dir="$OUT/profile" --window-size=1200,850 --remote-debugging-port=9333 \
  --use-angle=swiftshader --enable-unsafe-swiftshader "$@" "$START" >"$OUT/chrome.log" 2>&1 & CP=$!
echo "pids X=$XP atspi=$AP speechd=$SP orca=$OP chrome=$CP extra_args=$*" >> "$OUT/stack.log"
/opt/node22/bin/node $R/scripts/driver.mjs "$SC" "$OUT" 9333 "$THREE" $R/evidence/raw_cache > "$OUT/driver.out" 2>&1
echo "driver rc=$?" >> "$OUT/stack.log"
kill $CP 2>/dev/null; sleep 2
kill -TERM $OP 2>/dev/null
for i in $(seq 100); do kill -0 $OP 2>/dev/null || break; sleep 0.1; done
kill -0 $OP 2>/dev/null && { echo "orca did not exit; SIGKILL" >> "$OUT/stack.log"; kill -9 $OP; } || echo "orca exited cleanly" >> "$OUT/stack.log"
sleep 1
kill $SP $AP 2>/dev/null; kill $DBUS_SESSION_BUS_PID 2>/dev/null; kill $XP 2>/dev/null
sleep 1; cleanup_strays; rm -rf "$OUT/profile"
wait 2>/dev/null
echo done >> "$OUT/stack.log"
