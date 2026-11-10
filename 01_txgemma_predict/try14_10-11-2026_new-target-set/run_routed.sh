#!/bin/bash
# Dispatcher used by both job B arrays: read job A's routing decision for $TARGET and run
# the harness it chose. Kept as a separate file so the two array scripts cannot drift
# apart, and so the routing logic lives in exactly one place.
set -euo pipefail

TRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUTING=/work/hdd/bisa/sshah21/results/txgemma_lora_orphan/try14_new_targets/routing.json

if [[ -z "${TARGET:-}" ]]; then
  echo "TARGET is unset -- refusing to guess. (This is the try05 scope-coupling rule.)" >&2
  exit 2
fi
if [[ ! -f "$ROUTING" ]]; then
  echo "no routing.json at $ROUTING -- job A did not finish. Nothing to run." >&2
  exit 3
fi

PATH_USED=$(python -c "
import json,sys
r=json.load(open('$ROUTING')).get('$TARGET')
print(r['path'] if r else 'absent')
")
echo "=== TARGET=$TARGET routed to: $PATH_USED ==="
python -c "
import json
r=json.load(open('$ROUTING')).get('$TARGET') or {}
print('  reason:', r.get('reason','(none)'))
print('  n_training:', r.get('n_training'), '| lines:', r.get('n_training_lines'),
      '| seq_aa:', r.get('sequence_length_aa'))
"

case "$PATH_USED" in
  stepstop)
    exec python "$TRY_DIR/step_stopping_eval.py"
    ;;
  crossval)
    exec python "$TRY_DIR/crossval_eval.py"
    ;;
  none)
    echo "$TARGET was reported UNTRAINABLE by job A -- exiting 0 without using the GPU."
    exit 0
    ;;
  *)
    echo "no routing entry for $TARGET (got '$PATH_USED')" >&2
    exit 4
    ;;
esac
