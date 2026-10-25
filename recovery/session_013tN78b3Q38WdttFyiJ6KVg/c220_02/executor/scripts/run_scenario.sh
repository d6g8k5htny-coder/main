#!/bin/bash
# C220-02: one Xvfb + AT-SPI + speech-dispatcher(espeak-ng, real-time paced sink) + Orca + Chromium stack per scenario.
# usage: run_scenario.sh SCENARIO_ID URL THREE_MODE(fulfill|fail) [extra chrome args...]
set -u
C=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220
ID="$1"; URL="$2"; MODE="$3"; shift 3
OUT="$C/evidence/$ID"; SOCK="$C/evidence/sd.sock"; PORT=9333
CHROME=/opt/pw-browsers/chromium-1194/chrome-linux/chrome

cleanup() {
  pkill -f "$CHROME" 2>/dev/null; pkill -f /usr/bin/orca 2>/dev/null; pkill -x Xvfb 2>/dev/null
  pkill -x speech-dispatch 2>/dev/null; pkill -x sd_espeak-ng 2>/dev/null; pkill -x sd_openjtalk 2>/dev/null
  pkill -f at-spi-bus-launcher 2>/dev/null; pkill -f at-spi2-registryd 2>/dev/null
  pkill -f "$C/scripts/audio_pacer.py" 2>/dev/null
  sleep 1
  pkill -9 -f "$CHROME" 2>/dev/null; pkill -9 -f /usr/bin/orca 2>/dev/null; pkill -9 -x Xvfb 2>/dev/null
  pkill -9 -x speech-dispatch 2>/dev/null; pkill -9 -x sd_espeak-ng 2>/dev/null
  rm -f /root/.cache/speech-dispatcher/speechd.sock "$SOCK" /tmp/.X99-lock
}
cleanup
rm -rf "$OUT"; mkdir -p "$OUT/sdlog"
echo "scenario=$ID url=$URL three=$MODE chrome_extra=$*" > "$OUT/run.txt"
date -u +"start %Y-%m-%dT%H:%M:%S.%NZ" >> "$OUT/run.txt"

export DISPLAY=:99
Xvfb :99 -screen 0 1280x900x24 -nolisten tcp >"$OUT/xvfb.log" 2>&1 &
sleep 1
eval "$(dbus-launch --sh-syntax)"
echo "dbus=$DBUS_SESSION_BUS_ADDRESS pid=$DBUS_SESSION_BUS_PID" >> "$OUT/run.txt"
export GTK_MODULES=gail:atk-bridge ACCESSIBILITY_ENABLED=1 NO_AT_BRIDGE=0
/usr/libexec/at-spi-bus-launcher --launch-immediately >"$OUT/atspi.log" 2>&1 &
sleep 1

# speech-dispatcher: scratch config, espeak-ng module, ALSA file-pipe into a real-time pacer.
export SPEECHD_ADDRESS="unix_socket:$SOCK"
C220_AUDIO_DIR="$OUT" LANG=C.UTF-8 speech-dispatcher -s -t 0 -l 5 -L "$OUT/sdlog" -C "$C/scripts/speechd-conf" -S "$SOCK" -P "$OUT/speechd.pid" >"$OUT/speechd.out" 2>&1 &
for i in $(seq 1 40); do [ -S "$SOCK" ] && break; sleep 0.25; done
[ -S "$SOCK" ] || echo "PROBLEM: speechd socket missing" >> "$OUT/run.txt"

/usr/bin/python3.12 /usr/bin/orca --replace --debug-file="$OUT/orca-debug.txt" >"$OUT/orca.out" 2>&1 &
sleep 5
grep -q "Speech Dispatcher service failed to connect" "$OUT/orca-debug.txt" && echo "PROBLEM: orca could not connect to speechd" >> "$OUT/run.txt"

"$CHROME" --no-sandbox --force-renderer-accessibility --no-first-run --no-default-browser-check \
  --user-data-dir="$OUT/profile" --window-size=1200,850 --remote-debugging-port=$PORT \
  --disable-background-networking --disable-component-update --disable-sync "$@" "data:text/html,<title>c220 start</title><p>start</p>" >"$OUT/chrome.log" 2>&1 &
sleep 3
/opt/node22/bin/node "$C/scripts/cdp_driver.mjs" "$ID" "$URL" "$MODE" "$OUT" $PORT >"$OUT/driver.log" 2>&1
echo "driver_exit=$?" >> "$OUT/run.txt"
sleep 1
date -u +"end %Y-%m-%dT%H:%M:%S.%NZ" >> "$OUT/run.txt"
cleanup
kill $DBUS_SESSION_BUS_PID 2>/dev/null
rm -rf "$OUT/profile"
sha256sum "$OUT/orca-debug.txt" > "$OUT/orca-debug.sha256"
cat "$OUT/run.txt"; cat "$OUT/driver.log"
