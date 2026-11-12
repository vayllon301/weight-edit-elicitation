#!/bin/zsh
# All six runs with nobody calling the queue: keeps the Mac awake (caffeinate -i: idle sleep only, a closed lid still sleeps it)
# and calls queue.sh again after each exit 2.
# usage: code/harness/run_all.sh [segment_seconds, default 6600] [total_seconds, default no limit]
# total_seconds lets it fit one background command: at that limit it exits 2, and calling it again goes on. It is a soft limit: a
# driver finishes the step in hand, up to 960 s for a Python block. Leave 20 minutes of margin, so 6000 under a 2-hour limit.
# Exit: 0 all done; 2 total limit reached, call again; anything else is queue.sh's, for a human, with queue-status.txt left in place.
if [[ -z ${RUN_ALL_AWAKE:-} ]]; then
  RUN_ALL_AWAKE=1 exec caffeinate -i zsh ${0:A} "$@"
fi
zmodload zsh/system
runs_dir=${HARNESS_RUNS:-code/runs}
[[ $runs_dir == /* ]] || runs_dir=${0:A:h:h:h}/$runs_dir  # as queue.sh sees it, from the project folder: one queue, one lock
lock_file=$runs_dir/.run_all.lock
mkdir -p ${lock_file:h}
: >> $lock_file
if ! zsystem flock -t 0 -f run_all_fd $lock_file; then
  echo "another run_all is running"
  exit 4
fi
segment=${1:-6600}
total=${2:-0}
pause=${RUN_ALL_PAUSE:-300}  # an exit 2 in under a minute means it is waiting for the model: don't spin
start=$(date +%s)
while true; do
  seg=$segment
  if (( total > 0 )); then
    left=$(( total - ($(date +%s) - start) ))
    if (( left <= 0 )); then
      echo "total limit reached: call again${QUEUE_RUNS:+ with QUEUE_RUNS='$QUEUE_RUNS'}"
      exit 2
    fi
    (( left < seg )) && seg=$left
  fi
  began=$(date +%s)
  QUEUE_REVIEWED=1 zsh ${0:A:h}/queue.sh $seg
  rc=$?
  (( rc == 0 )) && exit 0
  (( rc == 2 )) || exit $rc
  # A quota window can outlast a segment. Wait for the recorded absolute reset without relaunching drivers.
  retry=$(awk '$1 == "retry_at:" && $2+0 > m {m=$2} END {printf "%.0f",m}' $runs_dir/queue-status.txt 2>/dev/null)
  now=$(date +%s)
  if [[ $retry == <-> ]] && (( retry > now )); then
    if (( total > 0 && retry >= start + total )); then
      echo "quota reset is after this supervisor's total limit; state saved"
      exit 2
    fi
    echo "waiting until quota reset epoch $retry; no model calls before then"
    while (( (remaining = retry - $(date +%s)) > 0 )); do
      sleep $remaining
    done
  elif (( $(date +%s) - began < 60 )); then
    sleep $pause
  fi
done
