#!/bin/sh
# V0 control: the unmodified cd32303 tree and manifest render fully verified under the strict raw-at-commit stub.
H=$(cd "$(dirname "$0")" && pwd); CD="$H/../../../cd32303"; export PATH=/opt/node22/bin:$PATH
echo "--- V0 strict render of the unmodified cd32303 museum.json"; node "$H/render.mjs" "$CD" "$CD/docs/site/museum.json"
