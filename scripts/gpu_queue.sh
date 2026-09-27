#!/usr/bin/env bash
# Runs a queue of GPU make targets back to back while holding data/ml/gpu.lock (the lock
# ml.extractor_b.gpu takes for training and export), so no trainer can start mid-queue and no
# idle gap waits on an orchestrator poll. Each item logs to logs/<item>.log, the first failure
# stops the queue, and a rerun skips items whose done-marker under logs/gpu_queue/ exists.
#
# Usage: scripts/gpu_queue.sh "extraction-a-adr0012" "eval-b" "cal-rows-b CAL_B_MODEL=extractor-b"
#        (make gpu-queue QUEUE="extraction-a-adr0012,eval-b"; detach with nohup ... &)
# Env:   GPU_QUEUE_LOCK (default data/ml/gpu.lock), GPU_QUEUE_LOGS (default logs),
#        MAKE (default make).
#
# The lock is written in gpu.py's format 1 (pid, job, started_at): a lock that module never
# reclaims, on any host, so a trainer refuses with a message naming this queue. Targets that take
# the lock themselves (training, export, guard fine-tune) would deadlock on it and are refused;
# run them outside the queue.
set -euo pipefail

LOCK="${GPU_QUEUE_LOCK:-data/ml/gpu.lock}"
LOGS="${GPU_QUEUE_LOGS:-logs}"
MARKERS="$LOGS/gpu_queue"
MAKE_BIN="${MAKE:-make}"
SELF_LOCKING=" train-b-probe train-b train-b-new export-b train-guard retrain "

die() { echo "gpu-queue: $*" >&2; exit 2; }

[ "$#" -gt 0 ] || die "no targets given (usage: $0 TARGET [TARGET VAR=value] ...)"

# Validate the whole queue before taking the lock: a bad item must not waste a GPU window.
names=()
for item in "$@"; do
  read -r target _ <<<"$item"
  [ -n "$target" ] || die "empty queue item"
  case "$SELF_LOCKING" in *" $target "*) die "'$target' takes $LOCK itself; run it outside the queue" ;; esac
  # Marker/log name: the item with every character outside [A-Za-z0-9._-] replaced by '_'.
  name="$(printf '%s' "$item" | tr -c 'A-Za-z0-9._-' '_')"
  names+=("$name")
done

mkdir -p "$(dirname "$LOCK")" "$MARKERS"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
job="gpu_queue: $(printf '%s, ' "$@" | sed 's/, $//; s/["\\]/_/g')"
# noclobber makes the redirect an O_EXCL create: a held lock (trainer or another queue) refuses.
if ! (set -o noclobber; printf '{"pid": %d, "job": "%s", "started_at": "%s", "format": 1}\n' \
      "$$" "$job" "$started" >"$LOCK") 2>/dev/null; then
  die "$LOCK is held: $(cat "$LOCK" 2>/dev/null || echo unreadable); wait for that job, or remove the lock once nothing runs"
fi
owner="$(cat "$LOCK")"
release() {
  # Remove the lock only while it is still ours (the same content we wrote).
  if [ -f "$LOCK" ] && [ "$(cat "$LOCK" 2>/dev/null)" = "$owner" ]; then
    rm -f "$LOCK"
  fi
}
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "gpu-queue: holding $LOCK for ${#names[@]} item(s)"
index=0
for item in "$@"; do
  name="${names[$index]}"
  index=$((index + 1))
  marker="$MARKERS/$name.done"
  if [ -f "$marker" ]; then
    echo "[$index/${#names[@]}] $item: done $(cat "$marker"), skipped (delete $marker to rerun)"
    continue
  fi
  log="$LOGS/$name.log"
  echo "[$index/${#names[@]}] $item: started $(date -u +%H:%M:%SZ), log $log"
  read -r -a words <<<"$item"
  status=0
  "$MAKE_BIN" "${words[@]}" >"$log" 2>&1 || status=$?
  if [ "$status" -ne 0 ]; then
    echo "gpu-queue: $item failed (exit $status); queue stopped. Tail of $log:" >&2
    tail -n 20 "$log" >&2 || true
    exit "$status"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ >"$marker"
  echo "[$index/${#names[@]}] $item: ok $(date -u +%H:%M:%SZ)"
done
echo "gpu-queue: all ${#names[@]} item(s) done; $LOCK released"
