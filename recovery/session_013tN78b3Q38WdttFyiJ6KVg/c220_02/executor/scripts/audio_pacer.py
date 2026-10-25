#!/usr/bin/python3 -I
# Real-time audio sink for speech-dispatcher's ALSA "file" pcm pipe.
# usage: audio_pacer.py  (writes $C220_AUDIO_DIR/audio_pacer.log and audio.raw)
# Consumes s16 mono 22050 Hz PCM at real-time rate so that speech-dispatcher
# playback (and therefore CANCEL/interrupt effects and SayAll pacing) is real time.
# Logs one line per burst start/end with wall-clock timestamps.
import fcntl, os, sys, time
d = os.environ.get('C220_AUDIO_DIR', '.')
log_path, raw_path = os.path.join(d, 'audio_pacer.log'), os.path.join(d, 'audio.raw')
rate = 44100.0
F_SETPIPE_SZ = 1031
try:
    fcntl.fcntl(0, F_SETPIPE_SZ, 4096)
except OSError:
    pass
log = open(log_path, 'a', buffering=1)
raw = open(raw_path, 'ab')
log.write(f"{time.time():.6f} OPEN pid={os.getpid()}\n")
clock = 0.0; burst_start = None; burst_bytes = 0; last = 0.0
fd = 0
while True:
    data = os.read(fd, 1024)
    if not data:
        break
    now = time.time()
    if burst_start is None or now - last > 0.25 and now > clock + 0.25:
        if burst_start is not None:
            log.write(f"{last:.6f} BURST_END bytes={burst_bytes} audio_s={burst_bytes/rate:.3f}\n")
        burst_start = now; burst_bytes = 0
        log.write(f"{now:.6f} BURST_START\n")
    clock = max(clock, now) + len(data) / rate
    burst_bytes += len(data)
    raw.write(data)
    delay = clock - time.time()
    if delay > 0:
        time.sleep(delay)
    last = time.time()
if burst_start is not None:
    log.write(f"{last:.6f} BURST_END bytes={burst_bytes} audio_s={burst_bytes/rate:.3f}\n")
log.write(f"{time.time():.6f} CLOSE\n")
