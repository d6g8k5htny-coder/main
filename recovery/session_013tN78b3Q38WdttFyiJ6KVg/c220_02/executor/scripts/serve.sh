#!/bin/bash
# Serve the git-archive extract of docs/site at f61419d8 on 127.0.0.1:8765 (cwd outside the extract).
C=/tmp/claude-0/-home-user-main/d5c985ca-97e5-58d8-a9d4-1e0a19065a22/scratchpad/c220
cd "$C/scripts" && exec python3 -I -m http.server 8765 --bind 127.0.0.1 --directory "$C/site/docs/site"
