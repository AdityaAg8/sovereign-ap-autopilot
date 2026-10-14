#!/usr/bin/env bash
# Runs a queue of GPU make targets back to back while holding data/ml/gpu.lock (the lock
# ml.extractor_b.gpu takes for training and export), so no trainer can start mid-queue and no
# idle gap waits on someone polling. Each item logs to logs/<item>.log, the first failure
# stops the queue, and a rerun skips items whose done-marker under logs/gpu_queue/ exists.
# An item prefixed with '+' is one whose non-zero exit is a verdict, not a broken stack (a gate
# that fails, e.g. "+gates-b"): its failure is recorded (<item>.failed marker, no done-marker, so
# a rerun retries it), the queue goes on, and the queue exits with that status at the end.
# Sub-items joined by '|' form a concurrent group (ADR-0018): they start together, each with its
# own log and done-marker, and the queue moves on once all of them have ended. Group only jobs
# whose outputs cannot depend on timing and whose VRAM fits together (e.g. B extraction on
# already-parsed test docs next to the OCR parse of the cal split); a plain sub-item's failure
# stops the queue after its siblings end.
#
# Usage: scripts/gpu_queue.sh "extraction-a-adr0012" "eval-b" "cal-rows-b CAL_B_MODEL=extractor-b"
#        scripts/gpu_queue.sh "parse-cache SPLITS=test" "eval-b|parse-cache SPLITS=cal" "cal-rows-a"
#        (make gpu-queue QUEUE="extraction-a-adr0012,eval-b,+gates-b"; detach with nohup ... &)
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

# Marker/log name of a sub-item: without a '+', every character outside [A-Za-z0-9._-] replaced
# by '_', so "+gates-b" and "gates-b" share one done-marker.
marker_name() { printf '%s' "${1#+}" | tr -c 'A-Za-z0-9._-' '_'; }

[ "$#" -gt 0 ] || die "no targets given (usage: $0 TARGET [TARGET VAR=value] ...)"

# Validate the whole queue before taking the lock: a bad item must not waste a GPU window.
total=0
for item in "$@"; do
  IFS='|' read -r -a subs <<<"$item"
  [ "${#subs[@]}" -gt 0 ] || die "empty queue item"
  for sub in "${subs[@]}"; do
    spec="${sub#+}"
    read -r target _ <<<"$spec" || true
    [ -n "${target:-}" ] || die "empty queue item in '$item'"
    case "$SELF_LOCKING" in *" $target "*) die "'$target' takes $LOCK itself; run it outside the queue" ;; esac
    total=$((total + 1))
  done
done

mkdir -p "$(dirname "$LOCK")" "$MARKERS"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
job="gpu_queue: $(printf '%s, ' "$@" | sed 's/, $//; s/["\]/_/g')"
# noclobber makes the redirect an O_EXCL create: a held lock (trainer or another queue) refuses.
if ! (set -o noclobber; printf '{"pid": %d, "job": "%s", "started_at": "%s", "format": 1}\n' \
      "$$" "$job" "$started" >"$LOCK") 2>/dev/null; then
  die "$LOCK is held: $(cat "$LOCK" 2>/dev/null || echo unreadable); wait for that job, or remove the lock once nothing runs"
fi
owner="$(cat "$LOCK")"
pids=()
release() {
  # A group interrupted mid-run: stop its make children before the lock goes.
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  # Remove the lock only while it is still ours (the same content we wrote).
  if [ -f "$LOCK" ] && [ "$(cat "$LOCK" 2>/dev/null)" = "$owner" ]; then
    rm -f "$LOCK"
  fi
}
trap release EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Runs one sub-item's make to its log and writes its done- or failed-marker; returns make's exit.
run_sub() {
  local sub="$1" name spec log status=0
  name="$(marker_name "$sub")"
  spec="${sub#+}"
  log="$LOGS/$name.log"
  read -r -a words <<<"$spec"
  "$MAKE_BIN" "${words[@]}" >"$log" 2>&1 || status=$?
  if [ "$status" -eq 0 ]; then
    rm -f "$MARKERS/$name.failed"
    date -u +%Y-%m-%dT%H:%M:%SZ >"$MARKERS/$name.done"
  elif [ "$spec" != "$sub" ]; then
    printf '%s exit %d\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >"$MARKERS/$name.failed"
  fi
  return "$status"
}

echo "gpu-queue: holding $LOCK for $total item(s)"
index=0
verdict_status=0
verdict_failed=()
stop_status=0
for item in "$@"; do
  IFS='|' read -r -a subs <<<"$item"
  pending=()
  for sub in "${subs[@]}"; do
    index=$((index + 1))
    name="$(marker_name "$sub")"
    if [ -f "$MARKERS/$name.done" ]; then
      echo "[$index/$total] ${sub#+}: done $(cat "$MARKERS/$name.done"), skipped (delete $MARKERS/$name.done to rerun)"
      continue
    fi
    echo "[$index/$total] ${sub#+}: started $(date -u +%H:%M:%SZ), log $LOGS/$name.log"
    pending+=("$sub")
  done
  [ "${#pending[@]}" -gt 0 ] || continue
  statuses=()
  if [ "${#pending[@]}" -eq 1 ]; then
    status=0
    run_sub "${pending[0]}" || status=$?
    statuses=("$status")
  else
    # Concurrent group: one background make per sub-item, then wait for every one of them.
    pids=()
    for sub in "${pending[@]}"; do
      run_sub "$sub" &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      status=0
      wait "$pid" || status=$?
      statuses+=("$status")
    done
    pids=()
  fi
  for i in "${!pending[@]}"; do
    sub="${pending[$i]}"
    spec="${sub#+}"
    status="${statuses[$i]}"
    log="$LOGS/$(marker_name "$sub").log"
    if [ "$status" -eq 0 ]; then
      echo "$spec: ok $(date -u +%H:%M:%SZ)"
    elif [ "$spec" != "$sub" ]; then
      echo "gpu-queue: $spec failed (exit $status); '+' item, the queue goes on. Tail of $log:" >&2
      tail -n 20 "$log" >&2 || true
      [ "$verdict_status" -ne 0 ] || verdict_status="$status"
      verdict_failed+=("$spec")
    else
      echo "gpu-queue: $spec failed (exit $status); queue stopped. Tail of $log:" >&2
      tail -n 20 "$log" >&2 || true
      [ "$stop_status" -ne 0 ] || stop_status="$status"
    fi
  done
  [ "$stop_status" -eq 0 ] || exit "$stop_status"
done
if [ "$verdict_status" -ne 0 ]; then
  echo "gpu-queue: every item ran; failed: ${verdict_failed[*]} (rerun retries them); $LOCK released" >&2
  exit "$verdict_status"
fi
echo "gpu-queue: all $total item(s) done; $LOCK released"
