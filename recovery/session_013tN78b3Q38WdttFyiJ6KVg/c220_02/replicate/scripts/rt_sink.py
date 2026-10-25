#!/usr/bin/env python3
# Real-time audio sink for ALSA "file" plugin pipe: consumes raw S16 mono 22050 Hz
# at wall-clock rate so speech-dispatcher sees playback that takes real time.
import fcntl, os, sys, time
F_SETPIPE_SZ = 1031
BPS = 22050 * 2  # bytes per second (espeak-ng: 22050 Hz, mono, 16-bit)
fd = sys.stdin.fileno()
try:
    fcntl.fcntl(fd, F_SETPIPE_SZ, 4096)
except OSError:
    pass
log = open(os.environ.get('RT_SINK_LOG', '/dev/null'), 'a', buffering=1)
log.write(f'{time.time():.3f} sink start pid={os.getpid()}\n')
total = 0
t_free = time.monotonic()  # when the virtual DAC becomes free
while True:
    chunk = os.read(fd, 882)  # 20 ms of audio
    if not chunk:
        break
    now = time.monotonic()
    if t_free < now:
        t_free = now
    t_free += len(chunk) / BPS
    total += len(chunk)
    delay = t_free - time.monotonic()
    if delay > 0:
        time.sleep(delay)
log.write(f'{time.time():.3f} sink eof bytes={total}\n')
