#!/usr/bin/env bash
# Runs inside `lake env bash`: one lean at a time, waiting for any other Lean process first.
set -u
PLAN="$1"; OUTDIR="$2"; MOD="$3"
export LEAN_PATH="$MOD:${LEAN_PATH:-}"
export LEAN_SRC_PATH="$MOD:${LEAN_SRC_PATH:-}"
wait_free() {
  local n=0
  while pgrep -f 'toolchains/[^ ]*/bin/lean ' >/dev/null 2>&1; do
    sleep 3; n=$((n+1))
    if [ "$n" -gt 600 ]; then echo "WAIT_TIMEOUT"; return 1; fi
  done
}
while IFS=$'\t' read -r label flags path; do
  [ -z "$label" ] && continue
  [ "$flags" = "-" ] && flags=""
  wait_free || exit 98
  echo "START $label $(date -u +%T)"
  code=0
  # shellcheck disable=SC2086
  lean $flags "$path" > "$OUTDIR/$label.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$OUTDIR/$label.status"
  echo "END $label exit=$code bytes=$(wc -c < "$OUTDIR/$label.log") $(date -u +%T)"
done < "$PLAN"
echo "PLAN_DONE"
